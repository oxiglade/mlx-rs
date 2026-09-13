"""Reviewed core-conversion bridge into pinned upstream architecture modules."""

import importlib
import re

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def checkpoint_key(key):
    names = {"token_embd": "model.embed_tokens", "output_norm": "model.norm", "output": "lm_head"}
    layer_names = {
        "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
        "attn_output": "self_attn.o_proj", "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
        "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
        "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
    }
    stem, slot = key.rsplit(".", 1)
    if stem in names:
        return names[stem] + "." + slot
    match = re.fullmatch(r"blk\.(0|[1-9][0-9]*)\.(\w+)", stem)
    if match is None or match[2] not in layer_names:
        raise ValueError(f"unmapped GGUF key {key}")
    return f"model.layers.{match[1]}.{layer_names[match[2]]}.{slot}"


def mapped_config(metadata, arrays):
    arch = metadata["general.architecture"]

    def scalar(key):
        value = metadata[f"{arch}.{key}"]
        return value.item() if isinstance(value, mx.array) else value

    config = {"model_type": arch, "hidden_size": scalar("embedding_length"),
              "num_hidden_layers": scalar("block_count"), "intermediate_size": scalar("feed_forward_length"),
              "num_attention_heads": scalar("attention.head_count"), "num_key_value_heads": scalar("attention.head_count_kv"),
              "head_dim": scalar("attention.key_length"), "rms_norm_eps": scalar("attention.layer_norm_rms_epsilon"),
              "rope_theta": scalar("rope.freq_base"), "max_position_embeddings": scalar("context_length"),
              "vocab_size": arrays["token_embd.weight"].shape[0], "tie_word_embeddings": "output.weight" not in arrays,
              "rope_scaling": None}
    if metadata.get(f"{arch}.rope.scaling.type", "none") == "linear":
        config["rope_scaling"] = {"rope_type": "linear", "factor": scalar("rope.scaling.factor")}
    return config


def build(config, converted, mutation=None):
    architecture = importlib.import_module("mlx_lm.models." + config["model_type"])
    model = architecture.Model(architecture.ModelArgs.from_dict(config))
    weights, settings = {}, {}
    for name, array in converted.items():
        key = checkpoint_key(name)
        q = ".self_attn.q_proj." in key
        k = ".self_attn.k_proj." in key
        if (q or k) and config["model_type"] == "llama" and mutation != "skip_llama_inverse":
            heads = config["num_key_value_heads"] if k else config["num_attention_heads"]
            array = array.reshape(heads, -1, 2, *array.shape[1:]).swapaxes(1, 2).reshape(array.shape)
        if (q or k) and config["model_type"] == "qwen3" and mutation == "permute_qwen3":
            heads = config["num_key_value_heads"] if k else config["num_attention_heads"]
            array = array.reshape(heads, 2, -1, *array.shape[1:]).swapaxes(1, 2).reshape(array.shape)
        weights[key] = array
    if mutation == "swap_gate_up":
        for suffix in ("weight", "scales", "biases"):
            a, b = (f"model.layers.0.mlp.{part}_proj.{suffix}" for part in ("gate", "up"))
            if a in weights:
                weights[a], weights[b] = weights[b], weights[a]
    if mutation == "swap_qk_norm":
        a, b = (f"model.layers.0.self_attn.{part}_norm.weight" for part in ("q", "k"))
        weights[a], weights[b] = weights[b], weights[a]
    for key, packed in list(weights.items()):
        if packed.dtype != mx.uint32:
            continue
        stem = key.removesuffix(".weight")
        scales = weights[stem + ".scales"]
        bits = packed.shape[-1] * 32 // (scales.shape[-1] * 32)
        settings[stem] = {"group_size": 32, "bits": bits, "mode": "affine"}
        parent = model
        parts = stem.split(".")
        for part in parts[:-1]:
            parent = parent[int(part)] if isinstance(parent, list) else getattr(parent, part)
        cls = nn.QuantizedEmbedding if stem == "model.embed_tokens" else nn.QuantizedLinear
        # Upstream constructors quantize random weights. Bypass initialization, inject all slots.
        module = cls.__new__(cls)
        nn.Module.__init__(module)
        module.group_size, module.bits, module.mode = 32, bits, "affine"
        for suffix in ("weight", "scales", "biases"):
            setattr(module, suffix, weights[stem + "." + suffix])
        if cls is nn.QuantizedEmbedding:
            module.num_embeddings, module.dims = packed.shape[0], scales.shape[-1] * 32
        if mutation == "wrong_group_bits" and stem == "lm_head":
            # Keep the logical width: reinterpret 8-bit words as 4-bit lanes, then truncate.
            words = np.array(packed)
            lanes = ((words[..., None] >> (np.arange(8, dtype=np.uint32) * 4)) & 15).reshape(words.shape[0], -1)
            lanes = lanes[:, :scales.shape[-1] * 32]
            repacked = np.zeros((lanes.shape[0], lanes.shape[1] // 8), dtype=np.uint32)
            for lane in range(8):
                repacked |= lanes[:, lane::8] << (4 * lane)
            weights[key] = mx.array(repacked)
            module.weight, module.bits = weights[key], 4
        setattr(parent, parts[-1], module)
    model.load_weights(list(weights.items()), strict=True)
    if settings:
        for key, array in weights.items():
            if key.endswith(".weight") and array.ndim == 2:
                settings.setdefault(key.removesuffix(".weight"), False)
    if mutation == "bypass_tied_affine":
        embedding = model.model.embed_tokens
        embedding.as_linear = lambda x: mx.quantized_matmul(
            x, embedding.weight, mx.ones_like(embedding.scales), mx.zeros_like(embedding.biases),
            transpose=True, group_size=32, bits=embedding.bits)
    mx.eval(model.parameters())
    return model, settings


def capture(model, prompt):
    from mlx_lm.models.cache import make_prompt_cache

    tensors, native, states = {}, {}, {}

    def record(name, value):
        mx.eval(value)
        native[name] = str(value.dtype).removeprefix("mlx.core.")
        tensors[name] = np.array(value.astype(mx.float32))

    def cache_state(label, cache):
        states[label] = []
        for i, layer in enumerate(cache):
            states[label].append({"offset": layer.offset, "retained_range": [0, layer.offset]})
            for kind in ("keys", "values"):
                record(f"cache.{label}.layer{i}.{kind}", getattr(layer, kind)[..., :layer.offset, :])

    cache = make_prompt_cache(model)
    logits = model(mx.array([prompt], dtype=mx.int32), cache=cache)
    record("prefill.full.logits", logits)
    cache_state("after_prefill", cache)
    greedy = []
    for step in range(8):
        token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        greedy.append(token)
        logits = model(mx.array([[token]], dtype=mx.int32), cache=cache)
        record(f"decode.step{step}.logits", logits[:, -1, :])
    cache_state("after_decode", cache)
    for chunk in (1, 3, 8):
        cache = make_prompt_cache(model)
        chunks = [model(mx.array([prompt[start:start + chunk]], dtype=mx.int32), cache=cache)
                  for start in range(0, len(prompt), chunk)]
        record(f"prefill.chunk{chunk}.logits", mx.concatenate(chunks, axis=1))
    return tensors, native, states, greedy
