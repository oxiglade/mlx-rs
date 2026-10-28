#!/usr/bin/env python3
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import struct
import sys
import tempfile
from pathlib import Path

EXPECTED_PYTHON = (3, 12, 14)
EXPECTED_ARCH = "arm64"
EXPECTED_MLX = "0.30.6"
EXPECTED_NUMPY = "2.2.6"
SEED = 0x51A7E11A
ROOT = Path(__file__).resolve().parent
VENV = ROOT.parent / ".venv"
FIXTURE_NAME = "tiny-llama"
PROMPT = "red blue green yellow"
PROMPT_IDS = [1, 2, 3, 4]
DECODE_STEPS = 8
ATOL = 2.0e-4
RTOL = 2.0e-4
PERTURBED_TENSOR = "model.layers.0.self_attn.o_proj.weight"
PERTURBATION_SCALE = -1.5

CONFIG = {
    "model_type": "llama",
    "hidden_size": 12,
    "num_hidden_layers": 2,
    "intermediate_size": 24,
    "num_attention_heads": 3,
    "rms_norm_eps": 1.0e-5,
    "vocab_size": 24,
    "num_key_value_heads": 1,
    "max_position_embeddings": 64,
    "rope_theta": 10000.0,
    "head_dim": 4,
    "tie_word_embeddings": True,
    "attention_bias": False,
    "mlp_bias": False,
    "rope_scaling": None,
}

VOCAB = {
    "<unk>": 0,
    "red": 1,
    "blue": 2,
    "green": 3,
    "yellow": 4,
    "orange": 5,
    "purple": 6,
    "black": 7,
    "white": 8,
    "circle": 9,
    "square": 10,
    "triangle": 11,
    "small": 12,
    "large": 13,
    "near": 14,
    "far": 15,
    "one": 16,
    "two": 17,
    "three": 18,
    "four": 19,
    "alpha": 20,
    "beta": 21,
    "gamma": 22,
    "delta": 23,
}


def check_environment():
    if sys.version_info[:3] != EXPECTED_PYTHON:
        raise SystemExit(f"requires Python 3.12.14, got {platform.python_version()}")
    if platform.machine() != EXPECTED_ARCH:
        raise SystemExit(f"requires arm64, got {platform.machine()}")
    if Path(sys.prefix).resolve() != VENV.resolve():
        raise SystemExit(f"requires pinned venv {VENV}, got {sys.prefix}")
    if importlib.metadata.version("mlx") != EXPECTED_MLX:
        raise SystemExit(f"requires mlx {EXPECTED_MLX}")
    if importlib.metadata.version("mlx-metal") != EXPECTED_MLX:
        raise SystemExit(f"requires mlx-metal {EXPECTED_MLX}")

    import numpy as np

    if np.__version__ != EXPECTED_NUMPY:
        raise SystemExit(f"requires numpy {EXPECTED_NUMPY}, got {np.__version__}")
    return np


def tensor_bytes(np, array):
    value = np.asarray(array)
    dtype = {
        np.dtype(np.float32): "F32",
        np.dtype(np.uint32): "U32",
    }.get(value.dtype)
    if dtype is None:
        raise ValueError(f"unsupported dtype {value.dtype}")
    return dtype, value.tobytes(order="C")


def write_safetensors(path, tensors, np):
    header = {}
    payloads = []
    offset = 0
    for name, array in sorted(tensors.items()):
        value = np.ascontiguousarray(array)
        dtype, payload = tensor_bytes(np, value)
        end = offset + len(payload)
        header[name] = {
            "dtype": dtype,
            "shape": list(value.shape),
            "data_offsets": [offset, end],
        }
        payloads.append(payload)
        offset = end
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(payloads))


def tokenizer_json():
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordLevel",
            "vocab": VOCAB,
            "unk_token": "<unk>",
        },
    }


def make_weights(np):
    rng = np.random.default_rng(SEED)
    hidden = CONFIG["hidden_size"]
    head_dim = CONFIG["head_dim"]
    kv_hidden = CONFIG["num_key_value_heads"] * head_dim
    intermediate = CONFIG["intermediate_size"]

    def normal(shape, scale):
        return (rng.standard_normal(shape, dtype=np.float32) * np.float32(scale)).astype(
            np.float32
        )

    weights = {
        "model.embed_tokens.weight": normal((CONFIG["vocab_size"], hidden), 0.28),
        "model.norm.weight": np.float32(1.0) + normal((hidden,), 0.04),
    }
    for layer in range(CONFIG["num_hidden_layers"]):
        prefix = f"model.layers.{layer}"
        weights[f"{prefix}.self_attn.q_proj.weight"] = normal((hidden, hidden), 0.16)
        weights[f"{prefix}.self_attn.k_proj.weight"] = normal((kv_hidden, hidden), 0.16)
        weights[f"{prefix}.self_attn.v_proj.weight"] = normal((kv_hidden, hidden), 0.16)
        weights[f"{prefix}.self_attn.o_proj.weight"] = normal((hidden, hidden), 0.16)
        weights[f"{prefix}.mlp.gate_proj.weight"] = normal((intermediate, hidden), 0.13)
        weights[f"{prefix}.mlp.down_proj.weight"] = normal((hidden, intermediate), 0.13)
        weights[f"{prefix}.mlp.up_proj.weight"] = normal((intermediate, hidden), 0.13)
        weights[f"{prefix}.input_layernorm.weight"] = np.float32(1.0) + normal(
            (hidden,), 0.04
        )
        weights[f"{prefix}.post_attention_layernorm.weight"] = np.float32(
            1.0
        ) + normal((hidden,), 0.04)
    return weights


def linear(np, x, weight):
    return np.matmul(x, weight.T).astype(np.float32)


def rms_norm(np, x, weight):
    square = np.multiply(x, x, dtype=np.float32)
    mean = np.mean(square, axis=-1, keepdims=True, dtype=np.float32)
    inverse = np.reciprocal(
        np.sqrt(mean + np.float32(CONFIG["rms_norm_eps"]), dtype=np.float32),
        dtype=np.float32,
    )
    return np.multiply(np.multiply(x, inverse, dtype=np.float32), weight, dtype=np.float32)


def rope(np, x, offset):
    dims = CONFIG["head_dim"]
    half = dims // 2
    exponents = np.arange(half, dtype=np.float32) * np.float32(2.0 / dims)
    inverse_frequencies = np.reciprocal(
        np.power(np.float32(CONFIG["rope_theta"]), exponents, dtype=np.float32),
        dtype=np.float32,
    )
    positions = np.arange(offset, offset + x.shape[-2], dtype=np.float32)
    angles = np.multiply(positions[:, None], inverse_frequencies[None, :], dtype=np.float32)
    cosines = np.cos(angles, dtype=np.float32)[None, None, :, :]
    sines = np.sin(angles, dtype=np.float32)[None, None, :, :]
    first = x[..., :half]
    second = x[..., half:dims]
    rotated_first = np.subtract(
        np.multiply(first, cosines, dtype=np.float32),
        np.multiply(second, sines, dtype=np.float32),
        dtype=np.float32,
    )
    rotated_second = np.add(
        np.multiply(second, cosines, dtype=np.float32),
        np.multiply(first, sines, dtype=np.float32),
        dtype=np.float32,
    )
    return np.concatenate((rotated_first, rotated_second), axis=-1).astype(np.float32)


def softmax(np, x):
    shifted = np.subtract(x, np.max(x, axis=-1, keepdims=True), dtype=np.float32)
    exponentials = np.exp(shifted, dtype=np.float32)
    return np.divide(
        exponentials,
        np.sum(exponentials, axis=-1, keepdims=True, dtype=np.float32),
        dtype=np.float32,
    )


def attention(np, x, weights, prefix, cache):
    batch, length, _ = x.shape
    heads = CONFIG["num_attention_heads"]
    kv_heads = CONFIG["num_key_value_heads"]
    head_dim = CONFIG["head_dim"]
    offset = 0 if cache["keys"] is None else cache["keys"].shape[-2]

    queries = linear(np, x, weights[f"{prefix}.q_proj.weight"])
    keys = linear(np, x, weights[f"{prefix}.k_proj.weight"])
    values = linear(np, x, weights[f"{prefix}.v_proj.weight"])
    queries = queries.reshape(batch, length, heads, head_dim).transpose(0, 2, 1, 3)
    keys = keys.reshape(batch, length, kv_heads, head_dim).transpose(0, 2, 1, 3)
    values = values.reshape(batch, length, kv_heads, head_dim).transpose(0, 2, 1, 3)
    queries = rope(np, queries, offset)
    keys = rope(np, keys, offset)
    cache["keys"] = (
        keys if cache["keys"] is None else np.concatenate((cache["keys"], keys), axis=-2)
    )
    cache["values"] = (
        values
        if cache["values"] is None
        else np.concatenate((cache["values"], values), axis=-2)
    )

    repeats = heads // kv_heads
    repeated_keys = np.repeat(cache["keys"], repeats, axis=1)
    repeated_values = np.repeat(cache["values"], repeats, axis=1)
    scaled_queries = np.multiply(
        queries, np.float32(1.0 / np.sqrt(np.float32(head_dim))), dtype=np.float32
    )
    scores = np.matmul(scaled_queries, repeated_keys.swapaxes(-1, -2)).astype(
        np.float32
    )
    if length > 1:
        causal = np.triu(np.ones((length, length), dtype=np.bool_), k=1)
        scores = np.where(causal[None, None, :, :], np.float32(-np.inf), scores)
    probabilities = softmax(np, scores)
    output = np.matmul(probabilities, repeated_values).astype(np.float32)
    output = output.transpose(0, 2, 1, 3).reshape(batch, length, heads * head_dim)
    return linear(np, output, weights[f"{prefix}.o_proj.weight"])


def forward(np, token_ids, weights, cache):
    hidden = weights["model.embed_tokens.weight"][np.asarray(token_ids, dtype=np.int64)]
    hidden = hidden.astype(np.float32, copy=False)
    for layer in range(CONFIG["num_hidden_layers"]):
        prefix = f"model.layers.{layer}"
        normalized = rms_norm(np, hidden, weights[f"{prefix}.input_layernorm.weight"])
        hidden = np.add(
            hidden,
            attention(np, normalized, weights, f"{prefix}.self_attn", cache[layer]),
            dtype=np.float32,
        )
        normalized = rms_norm(
            np, hidden, weights[f"{prefix}.post_attention_layernorm.weight"]
        )
        gate = linear(np, normalized, weights[f"{prefix}.mlp.gate_proj.weight"])
        up = linear(np, normalized, weights[f"{prefix}.mlp.up_proj.weight"])
        silu = np.divide(gate, np.add(np.float32(1.0), np.exp(-gate, dtype=np.float32)))
        mlp = linear(
            np,
            np.multiply(silu, up, dtype=np.float32),
            weights[f"{prefix}.mlp.down_proj.weight"],
        )
        hidden = np.add(hidden, mlp, dtype=np.float32)
    hidden = rms_norm(np, hidden, weights["model.norm.weight"])
    return linear(np, hidden, weights["model.embed_tokens.weight"])


def reference_run(np, weights):
    cache = [{"keys": None, "values": None} for _ in range(CONFIG["num_hidden_layers"])]
    logits = forward(np, [PROMPT_IDS], weights, cache)
    prefill_logits = np.ascontiguousarray(logits[0, -1], dtype=np.float32)
    prefill_cache = [
        {
            "keys": {"shape": list(layer["keys"].shape), "dtype": "F32"},
            "values": {"shape": list(layer["values"].shape), "dtype": "F32"},
        }
        for layer in cache
    ]
    tokens = []
    next_token = int(np.argmax(prefill_logits))
    for step in range(DECODE_STEPS):
        tokens.append(next_token)
        if step + 1 < DECODE_STEPS:
            logits = forward(np, [[next_token]], weights, cache)
            next_token = int(np.argmax(logits[0, -1]))
    return prefill_logits, tokens, prefill_cache


def comparator_rejects(np, expected, actual):
    difference = np.abs(np.subtract(actual, expected, dtype=np.float32))
    limit = np.add(
        np.float32(ATOL),
        np.multiply(np.float32(RTOL), np.abs(expected), dtype=np.float32),
        dtype=np.float32,
    )
    return bool(np.any(difference > limit)), float(np.max(difference))


def write_tree(target, np):
    fixture = target / FIXTURE_NAME
    fixture.mkdir(parents=True)
    weights = make_weights(np)
    prefill_logits, tokens, cache = reference_run(np, weights)
    perturbed = {name: value.copy() for name, value in weights.items()}
    perturbed[PERTURBED_TENSOR] *= np.float32(PERTURBATION_SCALE)
    perturbed_logits, perturbed_tokens, _ = reference_run(np, perturbed)
    rejected, maximum_difference = comparator_rejects(np, prefill_logits, perturbed_logits)
    if not rejected and perturbed_tokens == tokens:
        raise SystemExit("perturbation did not change tokens or logits beyond policy")

    (fixture / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    (fixture / "tokenizer.json").write_text(json.dumps(tokenizer_json(), indent=2) + "\n")
    write_safetensors(fixture / "model.safetensors", weights, np)
    write_safetensors(fixture / "model.perturbed.safetensors", perturbed, np)
    write_safetensors(
        fixture / "expectations.safetensors",
        {"prefill.final_logits": prefill_logits},
        np,
    )

    expectations = {
        "schema_version": 1,
        "fixture": FIXTURE_NAME,
        "prompt": {"text": PROMPT, "token_ids": PROMPT_IDS},
        "prefill_logits": {
            "tensor": "prefill.final_logits",
            "shape": list(prefill_logits.shape),
            "dtype": "F32",
            "policy": {"atol": ATOL, "rtol": RTOL},
        },
        "decode": {"steps": DECODE_STEPS, "token_ids": tokens},
        "prefill_cache": {"layers": cache},
        "qualification": {
            "weights": "model.perturbed.safetensors",
            "tensor": PERTURBED_TENSOR,
            "scale": PERTURBATION_SCALE,
            "comparator_rejects": rejected,
            "maximum_logit_difference": maximum_difference,
            "token_ids": perturbed_tokens,
        },
        "provenance": {
            "generator": "conformance/sentinel/generate_sentinel.py",
            "seed": SEED,
            "python": platform.python_version(),
            "architecture": platform.machine(),
            "numpy": np.__version__,
            "mlx_package_pin_checked_without_import": EXPECTED_MLX,
            "reference": "independent NumPy f32 Llama forward",
            "semantics": [
                "tied token embedding and output projection",
                "RMSNorm over the final axis",
                "split-half non-traditional RoPE",
                "grouped-query scaled dot-product attention with an additive causal mask",
                "SwiGLU MLP",
                "concatenating KV cache on sequence axis -2",
            ],
            "mlx_lm_parameters": {
                "rms_norm_eps_from_config": CONFIG["rms_norm_eps"],
                "rope_theta_from_config": CONFIG["rope_theta"],
                "rope_traditional_hardcoded": False,
                "rope_scale_default": 1.0,
                "rope_scaling_type_default": "default",
                "attention_scale": "queries multiplied by 1/sqrt(head_dim) before matmul",
                "prefill_mask": "automatic additive causal mask when sequence length > 1",
            },
            "weight_names": sorted(weights),
            "artifact_sha256": {
                name: hashlib.sha256((fixture / name).read_bytes()).hexdigest()
                for name in (
                    "model.safetensors",
                    "model.perturbed.safetensors",
                    "expectations.safetensors",
                )
            },
        },
    }
    (fixture / "expectations.json").write_text(json.dumps(expectations, indent=2) + "\n")


def tree_hash(path):
    digest = hashlib.sha256()
    for item in sorted(entry for entry in path.rglob("*") if entry.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(item.read_bytes())
    return digest.hexdigest()


def main():
    np = check_environment()
    first = Path(tempfile.mkdtemp(prefix="mlx-lm-sentinel-a-"))
    second = Path(tempfile.mkdtemp(prefix="mlx-lm-sentinel-b-"))
    try:
        write_tree(first, np)
        write_tree(second, np)
        first_hash = tree_hash(first)
        second_hash = tree_hash(second)
        if first_hash != second_hash:
            raise SystemExit(f"generation is not reproducible: {first_hash} != {second_hash}")
        destination = ROOT / "fixtures"
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(first, destination)
        sizes = {
            path.relative_to(destination).as_posix(): path.stat().st_size
            for path in sorted(destination.rglob("*"))
            if path.is_file()
        }
        print(json.dumps({"tree_sha256": first_hash, "sizes": sizes}, sort_keys=True))
    finally:
        shutil.rmtree(first, ignore_errors=True)
        shutil.rmtree(second, ignore_errors=True)


if __name__ == "__main__":
    main()
