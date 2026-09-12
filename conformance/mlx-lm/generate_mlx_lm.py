#!/usr/bin/env python3
import argparse
import contextlib
import hashlib
import importlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from manifest import ROOT, check_environment, write_json

SEED = 1729
DECODE_STEPS = 8
VARIANTS = ("llama-base", "llama-sliding", "llama-sharded", "llama-quant4", "qwen3-base", "qwen3-quant4")
QUANT_PARAMS = ("model.layers.0.self_attn.q_proj", "model.layers.0.mlp.down_proj")
SPECIAL_TOKENS = (
    "<unk>", "<pad>", "<|end_of_text|>", "<|eot_id|>", "<|begin_of_text|>",
    "<|start_header_id|>", "<|end_header_id|>", "<|im_start|>", "<|im_end|>",
    "<|endoftext|>", "<think>", "</think>",
)
LLAMA_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}"
    "{{ message['content'] + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
)
QWEN_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}"
    "{{- '<think>\n\n</think>\n\n' }}"
    "{%- endif %}{%- endif %}"
)
TOLERANCES = {
    "logits": {"atol": 2e-4, "rtol": 2e-4},
    "cache": {"atol": 2e-4, "rtol": 2e-4},
    "logprobs": {"atol": 1e-4, "rtol": 1e-4},
}


def numpy_array(value, packed=False):
    mx.eval(value)
    return np.array(value, dtype=np.uint32 if packed else np.float32, copy=True)


def save_arrays(path, arrays):
    from safetensors.numpy import save_file

    save_file({key: np.ascontiguousarray(arrays[key]) for key in sorted(arrays)}, str(path))


def model_config(variant):
    family = variant.split("-")[0]
    config = {
        "model_type": family,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "rms_norm_eps": 1e-5,
        "vocab_size": 64,
        "max_position_embeddings": 128,
        "rope_theta": 10000.0,
        "tie_word_embeddings": family == "qwen3" or variant == "llama-sliding",
        "eos_token_id": 9 if family == "llama" else 2,
    }
    if variant == "llama-sliding":
        config.update({
            "num_hidden_layers": 3,
            "layer_types": ["full_attention", "sliding_attention", "sliding_attention"],
            "sliding_window": 5,
            "rope_scaling": {
                "rope_type": "llama3", "factor": 8.0,
                "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                "original_max_position_embeddings": 64,
            },
        })
    if variant.endswith("quant4"):
        config["quantization"] = {"group_size": 32, "bits": 4, "mode": "affine"}
    return config


def resolved_config(model, config):
    args = model.args
    layer_types = getattr(args, "layer_types", ["full_attention"] * args.num_hidden_layers)
    return {
        "hidden_size": args.hidden_size,
        "layer_count": args.num_hidden_layers,
        "intermediate_size": args.intermediate_size,
        "attention_heads": args.num_attention_heads,
        "kv_heads": args.num_key_value_heads,
        "head_dim": args.head_dim,
        "vocabulary_size": args.vocab_size,
        "rms_norm_epsilon": args.rms_norm_eps,
        "rope": {
            "dimensions": args.head_dim, "theta": args.rope_theta,
            "traditional": getattr(args, "rope_traditional", False),
            "scaling": args.rope_scaling,
        },
        "attention_kinds": [
            {"sliding": args.sliding_window} if kind == "sliding_attention" else "full"
            for kind in layer_types
        ],
        "tie_word_embeddings": args.tie_word_embeddings,
        "quantization": (
            {key: config["quantization"][key] for key in ("group_size", "bits")}
            if "quantization" in config else None
        ),
    }


def write_tokenizer(directory, family):
    from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers

    words = list(SPECIAL_TOKENS) + [
        "hello", "world", "the", "small", "fox", "runs", "over", "green", "hill",
        "again", "system", "user", "assistant", "be", "brief", "answer", "yes",
        "café", "雪", ".", "!", "a", "b", "c",
    ]
    words += [f"word{i}" for i in range(64 - len(words))]
    vocab = {word: i for i, word in enumerate(words)}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.add_special_tokens([AddedToken(word, special=True) for word in SPECIAL_TOKENS])
    write_json(directory / "tokenizer.json", json.loads(tokenizer.to_str()))
    is_llama = family == "llama"
    write_json(directory / "tokenizer_config.json", {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>", "pad_token": "<pad>",
        "bos_token": "<|begin_of_text|>" if is_llama else None,
        "eos_token": "<|end_of_text|>" if is_llama else "<|endoftext|>",
        "additional_special_tokens": list(SPECIAL_TOKENS[2:]),
        "clean_up_tokenization_spaces": False,
        "model_max_length": 128,
        "chat_template": LLAMA_TEMPLATE if is_llama else QWEN_TEMPLATE,
    })
    write_json(directory / "generation_config.json", {"eos_token_id": [3, 2] if is_llama else [8, 9]})


def initial_weights(config):
    from mlx.utils import tree_flatten

    module = importlib.import_module(f"mlx_lm.models.{config['model_type']}")
    mx.random.seed(SEED)
    model = module.Model(module.ModelArgs.from_dict(config))
    rng = np.random.Generator(np.random.PCG64(SEED))
    weights = []
    for name, value in sorted(tree_flatten(model.parameters())):
        if value.ndim == 1:
            data = 1.0 + rng.normal(0.0, 0.03, value.shape)
        else:
            data = rng.normal(0.0, 0.14, value.shape)
        weights.append((name, mx.array(data.astype(np.float32))))
    model.load_weights(weights, strict=True)
    if "quantization" in config:
        nn.quantize(model, **config["quantization"])
    return {
        name: numpy_array(value, packed=value.dtype == mx.uint32)
        for name, value in sorted(tree_flatten(model.parameters()))
    }


def write_weights(directory, weights, sharded):
    if not sharded:
        save_arrays(directory / "model.safetensors", weights)
        return
    names = sorted(weights)
    midpoint = len(names) // 2
    weight_map = {}
    for index, keys in enumerate((names[:midpoint], names[midpoint:]), 1):
        filename = f"model-{index:05d}-of-00002.safetensors"
        save_arrays(directory / filename, {key: weights[key] for key in keys})
        weight_map.update({key: filename for key in keys})
    write_json(directory / "model.safetensors.index.json", {
        "metadata": {"total_size": sum(value.nbytes for value in weights.values())},
        "weight_map": weight_map,
    })


class RecordingModel:
    def __init__(self, model):
        self.model = model
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, tokens, **kwargs):
        logits = self.model(tokens, **kwargs)
        self.calls.append(numpy_array(logits))
        return logits


def snapshot_cache(arrays, label, cache, expected_offset, after_prefill=False):
    from mlx_lm.models.cache import RotatingKVCache

    metadata = []
    for index, entry in enumerate(cache):
        if entry.offset != expected_offset:
            raise RuntimeError(f"{label} layer {index}: offset {entry.offset} != {expected_offset}")
        retained = expected_offset
        if isinstance(entry, RotatingKVCache) and not after_prefill:
            retained = min(entry.max_size, expected_offset)
        for kind in ("keys", "values"):
            value = getattr(entry, kind)
            if isinstance(entry, RotatingKVCache):
                # Buffer order changes after a wrap; the contract exposes positions.
                value = entry._temporal_order(value)
            else:
                value = value[..., :entry.offset, :]
            if value.shape[2] != retained:
                raise RuntimeError(f"{label} layer {index}: retained {value.shape[2]} != {retained}")
            arrays[f"cache.{label}.layer{index}.{kind}"] = numpy_array(value)
        metadata.append({"offset": entry.offset, "retained_range": [expected_offset - retained, expected_offset]})
    return metadata


def forward_expectations(model, prompt, chunk_sizes, arrays):
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    tokens = mx.array(prompt, dtype=mx.int32)
    cache = make_prompt_cache(model)
    full = model(tokens[None], cache=cache)
    arrays["prefill.full.logits"] = numpy_array(full)
    cache_metadata = {"after_prefill": snapshot_cache(arrays, "after_prefill", cache, len(prompt), True)}
    logits = full[:, -1, :]
    greedy = []
    sampler = make_sampler(temp=0.0)
    for step in range(DECODE_STEPS):
        token = int(sampler(logits).item())
        greedy.append(token)
        logits = model(mx.array([[token]], dtype=mx.int32), cache=cache)[:, -1, :]
        arrays[f"decode.step{step}.logits"] = numpy_array(logits)
    cache_metadata["after_decode"] = snapshot_cache(arrays, "after_decode", cache, len(prompt) + DECODE_STEPS)
    for size in chunk_sizes:
        recording = RecordingModel(model)
        # Zero output tokens exhausts prefill without generate_step's decode lookahead.
        list(generate_step(tokens, recording, max_tokens=0, prefill_step_size=size, sampler=sampler))
        chunked = np.concatenate(recording.calls, axis=1)
        np.testing.assert_allclose(chunked, arrays["prefill.full.logits"], **TOLERANCES["logits"])
        arrays[f"prefill.chunk{size}.logits"] = chunked
    return greedy, cache_metadata


def sampling_cases():
    default = {
        "temperature": 0.7, "top_p": 0.0, "top_k": 0, "min_p": 0.0,
        "min_tokens_to_keep": 1, "repetition_penalty": None, "repetition_context_size": 20,
    }
    overrides = {
        "greedy": {"temperature": 0.0}, "temperature": {},
        "top_p": {"top_p": 0.9}, "top_k": {"top_k": 5},
        "min_p": {"min_p": 0.1}, "repetition_penalty": {"repetition_penalty": 1.3},
        "combined": {"top_p": 0.9, "min_p": 0.1, "top_k": 5},
    }
    return {name: {**default, **options} for name, options in overrides.items()}


def sampling_expectations(model, prompt, cases, arrays):
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p, make_logits_processors, make_sampler

    results = {}
    for name, options in cases.items():
        mx.random.seed(SEED)
        sampler = make_sampler(
            temp=options["temperature"], top_p=options["top_p"], top_k=options["top_k"],
            min_p=options["min_p"], min_tokens_to_keep=options["min_tokens_to_keep"],
        )
        processors = make_logits_processors(
            repetition_penalty=options["repetition_penalty"],
            repetition_context_size=options["repetition_context_size"],
        )
        history = []

        def record_history(tokens, logits):
            history.append(tokens.tolist())
            return logits

        if processors:
            processors = [record_history, *processors]
        with mx.stream(mx.cpu):
            steps = list(generate_step(
                mx.array(prompt, dtype=mx.int32), model, max_tokens=DECODE_STEPS,
                sampler=sampler, logits_processors=processors, prefill_step_size=len(prompt),
            ))
            filtered = steps[0][1][None]
            if options["temperature"] != 0:
                if 0 < options["top_p"] < 1:
                    filtered = apply_top_p(filtered, options["top_p"])
                if options["min_p"] != 0:
                    filtered = apply_min_p(filtered, options["min_p"], options["min_tokens_to_keep"])
                if options["top_k"] > 0:
                    filtered = apply_top_k(filtered, options["top_k"])
            arrays[f"sampling.{name}.filtered_logprobs"] = numpy_array(filtered[0])
        ids = [int(token) for token, _ in steps]
        if len(ids) != DECODE_STEPS or not np.isfinite(arrays[f"sampling.{name}.filtered_logprobs"][ids[0]]):
            raise RuntimeError(f"invalid sampler output for {name}")
        results[name] = {"options": options, "seed": SEED, "cpu_ids": ids}
        if history:
            results[name]["processor_token_histories"] = history[:DECODE_STEPS]
    return results


def stream_expectations(model, tokenizer, prompt):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    responses = list(stream_generate(
        model, tokenizer, prompt, max_tokens=DECODE_STEPS,
        sampler=make_sampler(temp=0.0), prefill_step_size=len(prompt),
    ))
    if not responses or sum(r.finish_reason is not None for r in responses) != 1:
        raise RuntimeError("stream must have exactly one final response")
    final = responses[-1]
    return {
        "text_deltas": [response.text for response in responses],
        "token_ids": [int(response.token) for response in responses],
        "finish_reason": final.finish_reason,
        "stop_token": int(final.token) if final.finish_reason == "stop" else None,
    }


def tokenizer_expectations(tokenizer, directory):
    prompts = {
        "canonical": "hello the small fox runs over green hill",
        "repeated": "hello hello world hello world",
        "unicode": "café 雪", "whitespace": "hello  world\nagain",
        "special": "<|im_start|>user\nhello<|im_end|>",
    }
    encodings = {name: tokenizer.encode(text, add_special_tokens=False) for name, text in prompts.items()}
    decodings = {name: tokenizer.decode(ids, skip_special_tokens=False) for name, ids in encodings.items()}
    encodings["special_with_defaults"] = tokenizer.encode(prompts["special"], add_special_tokens=True)
    decodings["special_skipped"] = tokenizer.decode(encodings["special"], skip_special_tokens=True)
    generation = json.loads((directory / "generation_config.json").read_text())
    eos = sorted(tokenizer.eos_token_ids)
    if eos != sorted(generation["eos_token_id"]):
        raise RuntimeError("generation_config EOS precedence changed")
    messages = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello café 雪"}]
    chat = {}
    for continuation in ("closed", "start_assistant", "continue_last"):
        case_messages = messages + ([{"role": "assistant", "content": "answer yes"}] if continuation != "start_assistant" else [])
        options = {
            "add_generation_prompt": continuation == "start_assistant",
            "continue_final_message": continuation == "continue_last",
            "enable_thinking": False,
        }
        rendered = tokenizer.apply_chat_template(case_messages, tokenize=False, **options)
        ids = tokenizer.apply_chat_template(case_messages, tokenize=True, **options)
        if ids != tokenizer.encode(rendered, add_special_tokens=False):
            raise RuntimeError("chat rendering and encoding disagree")
        chat[continuation] = {
            "messages": case_messages, "continuation": continuation,
            "rendered_utf8_hex": rendered.encode("utf-8").hex(), "token_ids": ids,
            "template_options": {"enable_thinking": False},
        }
    return prompts, {"encodings": encodings, "decodings": decodings, "eos_tokens": eos, "eos_source": "generation_config"}, chat


def error_expectations(directory, weights):
    cases = {
        "missing_shard": ("remove model-00002-of-00002.safetensors from a two-shard copy, retaining its index", "WeightError::MissingShard"),
        "duplicate_tensor": ("copy the first tensor from shard 1 into shard 2 without changing the index", "WeightError::DuplicateTensor"),
        "wrong_shape": ("remove the last row of model.layers.0.self_attn.q_proj.weight", "WeightError::ShapeMismatch"),
        "unsupported_rope": ("set rope_scaling.rope_type to unsupported_oracle_rope", "ConfigError::UnsupportedRope"),
        "unknown_model_type": ("set model_type to unsupported_oracle_model", "ConfigError::UnsupportedArchitecture"),
        "bad_quantization_bits": ("set quantization to affine, group_size 32, bits 5", "ConfigError::UnsupportedQuantization"),
    }
    results = {}
    for name, (mutation, error) in cases.items():
        with tempfile.TemporaryDirectory(prefix="mlx-lm-invalid-") as temporary:
            copied = Path(temporary) / "model"
            shutil.copytree(directory, copied)
            for path in copied.glob("model*.safetensors*"):
                path.unlink()
            mutated_weights = dict(weights)
            config = json.loads((copied / "config.json").read_text())
            if name == "wrong_shape":
                key = "model.layers.0.self_attn.q_proj.weight"
                mutated_weights[key] = mutated_weights[key][:-1].copy()
            elif name == "unsupported_rope":
                config["rope_scaling"] = {"rope_type": "unsupported_oracle_rope", "factor": 2.0}
            elif name == "unknown_model_type":
                config["model_type"] = "unsupported_oracle_model"
            elif name == "bad_quantization_bits":
                config["quantization"] = {"group_size": 32, "bits": 5, "mode": "affine"}
            write_json(copied / "config.json", config)
            write_weights(copied, mutated_weights, sharded=name in ("missing_shard", "duplicate_tensor"))
            if name == "missing_shard":
                (copied / "model-00002-of-00002.safetensors").unlink()
            elif name == "duplicate_tensor":
                from safetensors.numpy import load_file

                second = copied / "model-00002-of-00002.safetensors"
                tensors = load_file(str(second))
                key = sorted(weights)[0]
                tensors[key] = weights[key]
                save_arrays(second, tensors)
            try:
                mlx_lm.load(copied, tokenizer_config={"local_files_only": True, "trust_remote_code": False})
            except (ValueError, FileNotFoundError, KeyError, RuntimeError) as exc:
                python_result = type(exc).__name__
            else:
                python_result = "accepted"
        # Python's loader overwrites duplicate keys; Rust's strict boundary rejects them.
        results[name] = {"mutation": mutation, "expected_rust_error": error, "python_result": python_result}
    return results


def generate_fixture(directory, variant, manifest):
    directory.mkdir(parents=True)
    config = model_config(variant)
    write_json(directory / "config.json", config)
    write_tokenizer(directory, config["model_type"])
    weights = initial_weights(config)
    write_weights(directory, weights, sharded=variant == "llama-sharded")
    model, tokenizer = mlx_lm.load(directory, tokenizer_config={"local_files_only": True, "trust_remote_code": False})
    prompts, tokenizer_data, chat = tokenizer_expectations(tokenizer, directory)
    prompt = tokenizer_data["encodings"]["canonical"]
    if len(prompt) % 3 == 0 or len(prompt) <= config.get("sliding_window", 0):
        raise RuntimeError("canonical prompt must cross the window and have a non-divisor chunk size")
    chunks = [1, 3, len(prompt)]
    arrays = {}
    greedy, cache_metadata = forward_expectations(model, prompt, chunks, arrays)
    cases = sampling_cases()
    sampling = sampling_expectations(model, prompt, cases, arrays)
    if greedy != sampling["greedy"]["cpu_ids"]:
        raise RuntimeError("forward and generate_step greedy IDs disagree")
    stream = stream_expectations(model, tokenizer, prompt)
    if stream["token_ids"] != greedy[:len(stream["token_ids"])]:
        raise RuntimeError("stream_generate and forward greedy IDs disagree")
    eos = set(tokenizer.eos_token_ids)
    try:
        tokenizer.eos_token_ids = set()
        length_case = stream_expectations(model, tokenizer, prompt)
        tokenizer.eos_token_ids = {greedy[0]}
        stop_case = stream_expectations(model, tokenizer, prompt)
    finally:
        tokenizer.eos_token_ids = eos
    if length_case["finish_reason"] != "length" or stop_case["finish_reason"] != "stop":
        raise RuntimeError("forced stop/length cases did not reach their boundary")
    for parameter in QUANT_PARAMS if "quantization" in config else ():
        for suffix in ("weight", "scales", "biases"):
            key = f"{parameter}.{suffix}"
            arrays[f"quant.{key}"] = weights[key]
    errors = error_expectations(directory, weights)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    expectations = {
        "schema_version": 1,
        "provenance": {**manifest, "seed": SEED, "generator": f"generate_mlx_lm.py@{source_hash}"},
        "config": {"resolved": resolved_config(model, config)},
        "tokenizer": tokenizer_data, "chat": chat,
        "prefill": {"prompt": "canonical", "token_ids": prompt, "T": len(prompt)},
        "decode": {
            "greedy_ids": greedy, **stream,
            "length_case": {"eos_tokens": [], **length_case},
            "stop_case": {"eos_tokens": [greedy[0]], **stop_case},
        },
        "cache": cache_metadata,
        "sampling": sampling, "errors": errors, "tolerances": TOLERANCES,
    }
    write_json(directory / "inputs.json", {
        "seed": SEED, "prompts": prompts, "token_ids": tokenizer_data["encodings"],
        "canonical_prompt": "canonical", "decode_steps": DECODE_STEPS,
        "chunk_sizes": chunks, "sampling": cases, "sampling_seed": SEED,
        "quant_parameters": list(QUANT_PARAMS) if "quantization" in config else [],
    })
    write_json(directory / "expectations.json", expectations)
    save_arrays(directory / "expectations.safetensors", arrays)


def tree_hash(path):
    digest = hashlib.sha256()
    for item in sorted(entry for entry in path.rglob("*") if entry.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(item.read_bytes())
    return digest.hexdigest()


def generate_tree(path, manifest):
    for variant in VARIANTS:
        generate_fixture(path / variant, variant, manifest)
    from safetensors.numpy import load_file

    base = load_file(str(path / "llama-base" / "model.safetensors"))
    sharded = {}
    for shard in sorted((path / "llama-sharded").glob("model-*.safetensors")):
        sharded.update(load_file(str(shard)))
    if base.keys() != sharded.keys() or any(not np.array_equal(base[key], sharded[key]) for key in base):
        raise RuntimeError("sharded fixture must have exactly the base weights")


def main():
    parser = argparse.ArgumentParser(description="Regenerate the deterministic CPU mlx-lm oracle fixtures.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "fixtures")
    parser.add_argument("--lock", type=Path, default=ROOT / "requirements.lock")
    args = parser.parse_args()
    manifest = check_environment(args.lock)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    global mx, np, nn, mlx_lm
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    if mx.__version__ != manifest["mlx"]:
        raise SystemExit(f"requires MLX runtime {manifest['mlx']}, got {mx.__version__}")
    mx.set_default_device(mx.cpu)
    import mlx_lm

    generation = importlib.import_module("mlx_lm.generate")
    generation.generation_stream = mx.new_thread_local_stream(mx.cpu)
    # wired_limit reads Metal device_info, which the CPU default device does not carry.
    generation.wired_limit = lambda model, streams=None: contextlib.nullcontext()
    with tempfile.TemporaryDirectory(prefix="mlx-lm-a-") as first_dir, tempfile.TemporaryDirectory(prefix="mlx-lm-b-") as second_dir:
        first, second = Path(first_dir), Path(second_dir)
        with mx.stream(mx.cpu):
            generate_tree(first, manifest)
            generate_tree(second, manifest)
        first_hash, second_hash = tree_hash(first), tree_hash(second)
        if first_hash != second_hash:
            raise SystemExit(f"generation is not reproducible: {first_hash} != {second_hash}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for variant in VARIANTS:
            destination = args.output_dir / variant
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(first / variant, destination)
        print(first_hash)


if __name__ == "__main__":
    main()
