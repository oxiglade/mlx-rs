#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

import gguf_writer as writer
from gguf_recipes import freeze_hashes
from numpy_reference.common import load_weights, read_json, write_safetensors
from numpy_reference import gguf as independent

ROOT = Path(__file__).resolve().parent
CAPABILITY = "mlx-lm 0.31.3 supplies the architecture and generation oracle and a Llama-family GGUF exporter; it does NOT supply a GGUF model loader. GGUF ingestion is qualified by Python MLX container conversion plus a reviewed reference bridge and independent NumPy decoding."
FORMATS = ("f32", "f16", "q4_0", "q4_1", "q8_0")
TOLERANCE_POLICIES = {
    "f32": ("f32-v1", 2e-4, 2e-4),
    "f16": ("gguf-f16-v1", 5e-3, 5e-3),
    **{storage: ("gguf-affine-v1", 2e-2, 2e-3) for storage in ("q4_0", "q4_1", "q8_0")},
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def basis(base, family, storage):
    config = read_json(base / "config.json")
    weights = load_weights(base)
    tied = storage in ("f16", "q4_0")
    config.update(tie_word_embeddings=tied, rope_scaling=None)
    if storage in ("f16", "q4_1"):
        config["rope_scaling"] = {"rope_type": "linear", "factor": 2.0}
    if tied:
        weights.pop("lm_head.weight", None)
    elif "lm_head.weight" not in weights:
        weights["lm_head.weight"] = np.roll(weights["model.embed_tokens.weight"], 7, axis=0).copy()
    # Preserve the base random vectors, magnifying their differences to witness norm swaps.
    for key in weights:
        if key.endswith(("q_norm.weight", "k_norm.weight")):
            weights[key] = (1 + 8 * (weights[key] - 1)).astype(np.float32)
    return config, weights


def metadata_for(config, tokenizer):
    a = config["model_type"]
    metadata = {"general.architecture": (8, a), "general.alignment": (4, 32), "general.file_type": (4, 1)}
    for target, source in {"embedding_length": "hidden_size", "feed_forward_length": "intermediate_size",
                           "block_count": "num_hidden_layers", "attention.head_count": "num_attention_heads",
                           "attention.head_count_kv": "num_key_value_heads", "attention.key_length": "head_dim",
                           "attention.value_length": "head_dim", "rope.dimension_count": "head_dim",
                           "context_length": "max_position_embeddings"}.items():
        metadata[a + "." + target] = (4, config[source])
    metadata[a + ".attention.layer_norm_rms_epsilon"] = (6, config["rms_norm_eps"])
    metadata[a + ".rope.freq_base"] = (6, config["rope_theta"])
    metadata[a + ".rope.scaling.type"] = (8, "linear" if config["rope_scaling"] else "none")
    if config["rope_scaling"]:
        metadata[a + ".rope.scaling.factor"] = (6, 2.0)
    if tokenizer.bos_token_id is not None:
        # GGUF leaves the key out entirely when the tokenizer has no BOS, as Qwen3 does.
        metadata["tokenizer.ggml.bos_token_id"] = (4, tokenizer.bos_token_id)
    metadata["tokenizer.ggml.eos_token_id"] = (4, min(tokenizer.eos_token_ids))
    return metadata


def encoded_weights(config, weights, storage):
    tensors = {}
    for key, array in sorted(weights.items()):
        rows = array.tolist()
        if config["model_type"] == "llama" and key.endswith(("q_proj.weight", "k_proj.weight")):
            heads = config["num_key_value_heads"] if key.endswith("k_proj.weight") else config["num_attention_heads"]
            rows = writer.export_rows(rows, heads)
        kind = storage if array.ndim == 2 else "f32"
        if storage == "q4_1" and key == "model.layers.0.self_attn.v_proj.weight":
            kind = "f16"
        if storage == "q4_1" and key == "lm_head.weight":
            kind = "q8_0"
        values = [x for row in rows for x in row] if array.ndim == 2 else rows
        tensors[writer.external_name(key)] = (array.shape, kind, writer.encode_tensor(values, array.shape, kind))
    return tensors


def qualify_exporter(config, weights, tensors):
    import mlx.core as mx
    from mlx_lm.gguf import permute_weights, translate_weight_names

    if writer.export_rows(list(range(16)), 1) != writer.LLAMA_D16:
        raise AssertionError("literal Llama orientation witness failed")
    np.testing.assert_array_equal(np.array(permute_weights(mx.arange(16).reshape(16, 1), 1)).ravel(), writer.LLAMA_D16)
    for key, array in weights.items():
        if config["model_type"] == "llama":
            assert writer.external_name(key) == translate_weight_names(key), key
            if key.endswith(("q_proj.weight", "k_proj.weight")):
                heads = config["num_key_value_heads"] if key.endswith("k_proj.weight") else config["num_attention_heads"]
                shape, _, raw = tensors[writer.external_name(key)]
                np.testing.assert_array_equal(np.frombuffer(raw, dtype="<u4").reshape(shape),
                                              np.array(permute_weights(mx.array(array), heads)).view(np.uint32))
        elif key.endswith(("q_proj.weight", "k_proj.weight")):
            shape, _, raw = tensors[writer.external_name(key)]
            np.testing.assert_array_equal(np.frombuffer(raw, dtype="<u4").reshape(shape), array.view(np.uint32))


def exact_conversion(path, converted):
    metadata, kinds, descriptors, parsed, _ = independent.read(path)
    if converted.keys() != parsed.keys():
        raise AssertionError("core/NumPy converted key mismatch")
    for key, expected in parsed.items():
        actual = np.array(converted[key])
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise AssertionError(f"core/NumPy converted dtype/shape: {key}")
        np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8), err_msg=key)
    return metadata, kinds, descriptors


def qualify_converted_mutations(converted):
    arrays = {key: np.array(value) for key, value in converted.items()}
    mutations = {}
    has_q8 = False
    for key, value in sorted(arrays.items()):
        candidates = {}
        if value.dtype == np.uint32:
            candidate = value.copy()
            candidate.flat[0] ^= np.uint32(1)
            candidates["packed_lane"] = candidate
            scales = arrays[key.removesuffix(".weight") + ".scales"]
            if value.shape[-1] // scales.shape[-1] == 8:
                has_q8 = True
                candidate = value.copy()
                candidate.flat[0] ^= np.uint32(0x80)
                candidates["signed_rebasing"] = candidate
        elif key.endswith(".biases"):
            candidates["bias_sign"] = -value
        elif key.endswith(".scales"):
            candidates["scale_row"] = np.roll(value, 1, axis=0)
        for name, candidate in candidates.items():
            if name not in mutations and not np.array_equal(candidate, value):
                mutations[name] = {"key": key, "class": "value"}
    if any(value.dtype == np.uint32 for value in arrays.values()):
        required = {"packed_lane", "bias_sign", "scale_row"}
        if has_q8:
            required.add("signed_rebasing")
        if set(mutations) != required:
            raise RuntimeError("degenerate converted affine witnesses")
        mutations["drop_companion"] = {"key": next(key for key in sorted(arrays) if key.endswith(".scales")), "class": "output_count"}
    return mutations


def compare(reference, actual, atol, rtol):
    if reference.keys() != actual.keys():
        raise AssertionError("reference tensor keys differ")
    result = {}
    for key, expected in reference.items():
        value = actual[key]
        if value.shape != expected.shape or not np.isfinite(value).all() or not np.isfinite(expected).all():
            raise AssertionError(f"reference shape/nonfinite: {key}")
        delta = np.abs(expected.astype(np.float64) - value.astype(np.float64))
        magnitude = np.abs(expected.astype(np.float64))
        bound = atol + rtol * magnitude
        max_abs = float(delta.max(initial=0))
        result[key] = {"passed": bool(np.all(delta <= bound)), "max_abs": max_abs,
                       "max_rel": float((delta / np.maximum(magnitude, np.finfo(np.float32).tiny)).max(initial=0)),
                       "rel_to_peak": max_abs / max(float(magnitude.max(initial=0)), np.finfo(np.float32).tiny),
                       "margin": float((delta / bound).max(initial=0))}
    return result


def require_reference_agreement(agreement, greedy, numpy_greedy, fixture):
    if greedy != numpy_greedy:
        raise RuntimeError(f"STOP: Python/NumPy greedy IDs differ for {fixture}: {greedy} != {numpy_greedy}")
    if not all(report["passed"] for report in agreement.values()):
        raise RuntimeError(f"STOP: Python/NumPy tensors disagree for {fixture}; tolerance must not be widened")


def qualify_mutation(agreement, mutation, fixture, affine):
    failures = [key for key, report in agreement.items() if not report["passed"]]
    if not failures and not affine:
        raise RuntimeError(f"inert value mutation {fixture}/{mutation}; redesign witnesses")
    candidates = list(agreement) if affine else failures
    candidates = [key for key in candidates if key.startswith("cache.") and mutation == "swap_qk_norm"] or candidates
    witness = max(candidates, key=lambda key: agreement[key]["margin"]) if affine else candidates[0]
    margin = agreement[witness]["margin"]
    print(f"mutation: {fixture}/{mutation} witness={witness} margin={margin:.9g}x", flush=True)
    if affine and margin < 10:
        raise RuntimeError(f"weak affine mutation {fixture}/{mutation}: margin={margin:.9g}x < 10x; redesign witnesses")
    return {"witness": witness, "class": "cache_value" if witness.startswith("cache.") else "value",
            "failed_tensors": failures, "margin": margin, "max_abs": agreement[witness]["max_abs"]}


def generate(base, destination, family, storage, manifest):
    import mlx.core as mx
    from mlx_lm.tokenizer_utils import load as load_tokenizer
    import gguf_bridge as bridge

    destination.mkdir(parents=True)
    config, weights = basis(base, family, storage)
    tokenizer = load_tokenizer(base, {"local_files_only": True, "trust_remote_code": False},
                               eos_token_ids=read_json(base / "generation_config.json")["eos_token_id"])
    metadata = metadata_for(config, tokenizer)
    encoded = encoded_weights(config, weights, storage)
    if storage == "f32":
        qualify_exporter(config, weights, encoded)
    path = destination / "model.gguf"
    writer.write(path, metadata, encoded)
    converted, core_metadata = mx.load(str(path), return_metadata=True)
    metadata_values, metadata_kinds, descriptors = exact_conversion(path, converted)
    converted_mutations = qualify_converted_mutations(converted)
    mapped = bridge.mapped_config(core_metadata, converted)
    assert mapped == independent.model_config(metadata_values, descriptors), "independent config mapping"
    model, quantization = bridge.build(mapped, converted)
    inputs = read_json(base / "inputs.json")
    inputs = {key: inputs[key] for key in ("seed", "prompts", "token_ids", "canonical_prompt", "decode_steps", "chunk_sizes")}
    prompt = inputs["token_ids"]["canonical"]
    for name, text in inputs["prompts"].items():
        if tokenizer.encode(text, add_special_tokens=False) != inputs["token_ids"][name]:
            raise AssertionError(f"base tokenizer prompt changed: {name}")
    arrays, native, cache, greedy = bridge.capture(model, prompt)
    reference, numpy_greedy = independent.reference(path, prompt)
    policy, atol, rtol = TOLERANCE_POLICIES[storage]
    agreement = compare(arrays, reference, atol, rtol)
    reference_report = {"passed": all(x["passed"] for x in agreement.values()) and greedy == numpy_greedy,
                        "greedy_ids_equal": greedy == numpy_greedy, "python_greedy_ids": greedy,
                        "numpy_greedy_ids": numpy_greedy, "keys": agreement,
                        **{metric: max(x[metric] for x in agreement.values()) for metric in ("max_abs", "max_rel", "margin")}}
    write_json(destination / "reference-agreement.json", reference_report)
    require_reference_agreement(agreement, greedy, numpy_greedy, destination.name)
    mutations = ["skip_llama_inverse" if family == "llama" else "permute_qwen3", "swap_gate_up"]
    if family == "qwen3":
        mutations.append("swap_qk_norm")
    if storage == "q4_0":
        mutations.append("bypass_tied_affine")
    if storage == "q4_1":
        mutations.append("wrong_group_bits")
    mutation_report, mutation_tensors = {}, {}
    for mutation in mutations:
        changed, _ = bridge.build(mapped, converted, mutation)
        observed, _, _, changed_ids = bridge.capture(changed, prompt)
        report = qualify_mutation(compare(arrays, observed, atol, rtol), mutation, destination.name,
                                  policy == "gguf-affine-v1")
        witness = report["witness"]
        mutation_tensors[mutation + "::" + witness] = observed[witness]
        mutation_report[mutation] = {**report, "greedy_ids": changed_ids}
    from generate_mlx_lm import resolved_config
    resolved = resolved_config(model, mapped)
    resolved["quantization"] = None
    sources = {name: digest(ROOT / name) for name in ("generate_gguf.py", "gguf_writer.py", "gguf_bridge.py", "gguf_recipes.py", "gguf_cases.json", "numpy_reference/gguf.py", "numpy_reference/common.py", "numpy_reference/llama.py", "numpy_reference/qwen3.py")}
    upstream = {}
    for name in ("mlx_lm.gguf", "mlx_lm.models." + family, "mlx.nn.layers.quantized", "mlx_lm.models.cache", "mlx_lm.models.rope_utils", "mlx_lm.models.base"):
        upstream[name] = digest(importlib.util.find_spec(name).origin)
    tokenizer_files = {name: {"path": f"../{base.name}/{name}", "sha256": digest(base / name)}
                       for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "config.json")}
    doc = {"schema_version": 1, "source": "gguf", "provenance": {**manifest, "capability": CAPABILITY,
           "sources": sources, "upstream_sources": upstream, "base": base.name, "reference_agreement": reference_report,
           "base_model_sha256": digest(base / "model.safetensors"), "model_sha256": digest(path)},
           "config": {"resolved": resolved, "bridge": mapped, "quantization": quantization},
           "gguf": {"architecture": family, "storage": storage, "metadata": metadata_values, "metadata_types": metadata_kinds,
                    "tensor_types": {key: value[1] for key, value in descriptors.items()}, "tokenizer_files": tokenizer_files,
                    "orientation_d16": writer.LLAMA_D16 if family == "llama" else writer.QWEN3_D16},
           "tokenizer": {"encodings": inputs["token_ids"], "eos_tokens": sorted(tokenizer.eos_token_ids), "bos_token_id": tokenizer.bos_token_id},
           "prefill": {"prompt": "canonical", "token_ids": prompt, "T": len(prompt)},
           "decode": {"greedy_ids": greedy}, "cache": cache, "native_dtypes": native,
           "tolerance_policy": policy,
           "tolerances": {key: {"atol": atol, "rtol": rtol} for key in ("cache", "logits")},
           "mutations": mutation_report, "converted_mutations": converted_mutations, "reference_qualified": True}
    write_json(destination / "inputs.json", inputs)
    write_json(destination / "expectations.json", doc)
    write_safetensors(destination / "expectations.safetensors", arrays)
    write_safetensors(destination / "converted.safetensors", {key: np.array(value) for key, value in converted.items()})
    write_safetensors(destination / "mutations.safetensors", mutation_tensors)
    return doc


def main():
    parser = argparse.ArgumentParser(description="Generate and qualify ten GGUF model fixtures on the Metal host")
    parser.add_argument("--base-fixtures", type=Path, default=ROOT / "fixtures")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "fixtures")
    parser.add_argument("--recipes-only", action="store_true",
                        help="freeze malformed-recipe hashes from existing fixtures without Python MLX")
    args = parser.parse_args()
    if args.recipes_only:
        freeze_hashes(args.base_fixtures, ROOT / "gguf_cases.json")
        return
    from manifest import check_environment
    manifest = check_environment()
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    # Keep diagnostics if ruling C fails; publish no partially qualified fixture family.
    staging = Path(tempfile.mkdtemp(prefix="gguf-oracle-"))
    print(f"staging: {staging}", flush=True)
    for family in ("llama", "qwen3"):
        for storage in FORMATS:
            name = f"gguf-{family}-{storage}"
            generate(args.base_fixtures / f"{family}-base", staging / name, family, storage, manifest)
            print(f"qualified: {name}", flush=True)
    freeze_hashes(staging, ROOT / "gguf_cases.json")
    for path in sorted(staging.iterdir()):
        doc = read_json(path / "expectations.json")
        doc["provenance"]["sources"]["gguf_cases.json"] = digest(ROOT / "gguf_cases.json")
        write_json(path / "expectations.json", doc)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(staging.iterdir()):
        shutil.copytree(path, args.output_dir / path.name, dirs_exist_ok=True)


if __name__ == "__main__":
    main()
