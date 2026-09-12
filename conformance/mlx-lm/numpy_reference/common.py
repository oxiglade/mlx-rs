import json
import math
import struct
from pathlib import Path

import numpy as np


DTYPES = {
    "F32": "<f4",
    "F16": "<f2",
    "BF16": "<u2",
    "U32": "<u4",
    "I32": "<i4",
    "I64": "<i8",
    "U8": "u1",
    "BOOL": "?",
}


def parse_json(data):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(data, object_pairs_hook=unique_pairs)


def read_json(path):
    return parse_json(Path(path).read_bytes())


def read_safetensors(path, *, allow_bfloat16=False):
    data = Path(path).read_bytes()
    if len(data) < 8:
        raise ValueError(f"truncated safetensors: {path}")
    size = struct.unpack("<Q", data[:8])[0]
    if size > len(data) - 8:
        raise ValueError(f"truncated safetensors header: {path}")
    header = parse_json(data[8 : 8 + size])
    payload = data[8 + size :]
    tensors = {}
    intervals = []
    for name, entry in sorted(header.items()):
        if name == "__metadata__":
            continue
        dtype = np.dtype(DTYPES[entry["dtype"]])
        shape = entry["shape"]
        start, end = entry["data_offsets"]
        if any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f"invalid shape: {name}")
        if not 0 <= start <= end <= len(payload):
            raise ValueError(f"invalid data offsets: {name}")
        if end - start != math.prod(shape) * dtype.itemsize:
            raise ValueError(f"invalid payload size: {name}")
        intervals.append((start, end))
        value = np.frombuffer(payload[start:end], dtype=dtype).reshape(shape)
        if entry["dtype"] == "BF16":
            if not allow_bfloat16:
                raise ValueError(f"BF16 is only supported for input weights: {name}")
            value = (value.astype(np.uint32) << 16).view(np.float32)
        tensors[name] = value.copy()
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError(f"overlapping or missing tensor data: {path}")
        cursor = end
    if cursor != len(payload):
        raise ValueError(f"unclaimed tensor data: {path}")
    return tensors


def write_safetensors(path, tensors):
    header = {}
    payloads = []
    offset = 0
    dtype_names = {np.dtype(v): k for k, v in DTYPES.items() if k != "BF16"}
    for name, array in sorted(tensors.items()):
        value = np.asarray(array)
        payload = value.tobytes(order="C")
        end = offset + len(payload)
        header[name] = {
            "dtype": dtype_names[value.dtype],
            "shape": list(value.shape),
            "data_offsets": [offset, end],
        }
        payloads.append(payload)
        offset = end
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(payloads))


def load_weights(fixture):
    fixture = Path(fixture)
    index_path = fixture / "model.safetensors.index.json"
    if not index_path.exists():
        return read_safetensors(fixture / "model.safetensors", allow_bfloat16=True)
    weight_map = read_json(index_path)["weight_map"]
    if not weight_map:
        raise ValueError("empty shard index")
    weights = {}
    for shard in sorted(set(weight_map.values())):
        path = (fixture / shard).resolve()
        if not path.is_relative_to(fixture.resolve()):
            raise ValueError(f"shard escapes fixture directory: {shard}")
        tensors = read_safetensors(path, allow_bfloat16=True)
        expected = {name for name, source in weight_map.items() if source == shard}
        if tensors.keys() != expected:
            raise ValueError(f"shard keys disagree with index: {shard}")
        if weights.keys() & tensors.keys():
            raise ValueError(f"duplicate tensors in shard: {shard}")
        weights.update(tensors)
    return weights


def dequantize(packed, scales, biases, group_size, bits):
    if bits not in (4, 8) or packed.dtype != np.uint32 or packed.ndim < 2:
        raise ValueError("affine weights require uint32 words and 4 or 8 bits")
    per_word = 32 // bits
    width = packed.shape[-1] * per_word
    if group_size <= 0 or width % group_size:
        raise ValueError("affine group size must divide the unpacked width")
    expected = packed.shape[:-1] + (width // group_size,)
    if scales.shape != expected or biases.shape != expected:
        raise ValueError("affine scales/biases shape mismatch")
    shifts = np.arange(per_word, dtype=np.uint32) * np.uint32(bits)
    q = (packed[..., None] >> shifts) & np.uint32((1 << bits) - 1)
    q = q.reshape(packed.shape[:-1] + (width,)).astype(np.float32)
    return q * np.repeat(scales.astype(np.float32), group_size, axis=-1) + np.repeat(
        biases.astype(np.float32), group_size, axis=-1
    )


def float_weights(weights, config):
    result = {}
    quantization = config.get("quantization") or {}
    for name, value in sorted(weights.items()):
        if name.endswith((".scales", ".biases")):
            continue
        prefix = name.removesuffix(".weight")
        has_scales = prefix + ".scales" in weights
        has_biases = prefix + ".biases" in weights
        if name.endswith(".weight") and (has_scales or has_biases):
            settings = quantization.get(prefix, quantization)
            if not isinstance(settings, dict) or settings.get("mode", "affine") != "affine":
                raise ValueError(f"unsupported quantization: {prefix}")
            result[name] = dequantize(
                value,
                weights[prefix + ".scales"],
                weights[prefix + ".biases"],
                settings["group_size"],
                settings["bits"],
            )
        else:
            if value.dtype.kind != "f":
                raise ValueError(f"non-floating weight without affine metadata: {name}")
            result[name] = value.astype(np.float32)
    return result


def rms_norm(x, weight, eps):
    variance = np.mean(x * x, axis=-1, keepdims=True)
    return x * np.reciprocal(np.sqrt(variance + np.float32(eps))) * weight


def rope_frequencies(dimensions, theta=10000.0, scaling=None):
    if dimensions <= 0 or dimensions % 2:
        raise ValueError("RoPE dimensions must be positive and even")
    freqs = np.float32(theta) ** (
        np.arange(0, dimensions, 2, dtype=np.float32) / np.float32(dimensions)
    )
    scaling = scaling or {}
    kind = scaling.get("type") or scaling.get("rope_type", "default")
    if kind == "default":
        return freqs
    factor = np.float32(scaling["factor"])
    if kind == "linear":
        return freqs * factor
    if kind != "llama3":
        raise ValueError(f"unsupported RoPE: {kind}")
    low = np.float32(scaling.get("low_freq_factor", 1.0))
    high = np.float32(scaling.get("high_freq_factor", 4.0))
    context = np.float32(scaling.get("original_max_position_embeddings", 8192))
    wavelengths = np.float32(2 * np.pi) * freqs
    freqs = np.where(wavelengths > context / low, freqs * factor, freqs)
    medium = (wavelengths > context / high) & (wavelengths < context / low)
    smooth = (context / wavelengths - low) / (high - low)
    return np.where(medium, freqs / ((1 - smooth) / factor + smooth), freqs)


def rope(x, freqs, offset=0, traditional=False):
    dimensions = 2 * len(freqs)
    positions = np.arange(offset, offset + x.shape[-2], dtype=np.float32)
    angles = positions[:, None] / freqs
    cosine, sine = np.cos(angles), np.sin(angles)
    result = x.copy()
    if traditional:
        first, second = x[..., :dimensions:2], x[..., 1:dimensions:2]
        result[..., :dimensions:2] = first * cosine - second * sine
        result[..., 1:dimensions:2] = first * sine + second * cosine
    else:
        half = dimensions // 2
        first, second = x[..., :half], x[..., half:dimensions]
        result[..., :half] = first * cosine - second * sine
        result[..., half:dimensions] = first * sine + second * cosine
    return result


def causal_mask(length, offset=0, window=None):
    queries = np.arange(offset, offset + length)[:, None]
    keys = np.arange(offset + length)[None, :]
    mask = queries >= keys
    if window is not None:
        if window <= 0:
            raise ValueError("sliding window must be positive")
        mask &= queries < keys + window
    return mask


def repeat_kv(x, heads):
    kv_heads = x.shape[1]
    if heads < kv_heads or heads % kv_heads:
        raise ValueError("query heads must be a multiple of KV heads")
    return np.repeat(x, heads // kv_heads, axis=1)


class Decoder:
    def __init__(self, config, weights, traditional=False):
        self.config = config
        self.weights = float_weights(weights, config)
        self.heads = config["num_attention_heads"]
        self.kv_heads = config.get("num_key_value_heads") or self.heads
        self.head_dim = config.get("head_dim") or config["hidden_size"] // self.heads
        if self.heads % self.kv_heads:
            raise ValueError("query heads must be a multiple of KV heads")
        self.eps = config["rms_norm_eps"]
        self.traditional = traditional
        self.freqs = rope_frequencies(
            self.head_dim, config.get("rope_theta", 10000), config.get("rope_scaling")
        )
        layers = config["num_hidden_layers"]
        kinds = config.get("layer_types") or ["full_attention"] * layers
        if len(kinds) != layers:
            raise ValueError("layer_types length mismatch")
        self.windows = []
        for kind in kinds:
            if kind == "full_attention":
                self.windows.append(None)
            elif kind == "sliding_attention":
                window = config["sliding_window"]
                if not isinstance(window, int) or window <= 0:
                    raise ValueError("sliding window must be positive")
                self.windows.append(window)
            else:
                raise ValueError(f"unsupported attention kind: {kind}")
        self.cache = [None] * layers
        self.offset = 0

    def linear(self, x, name):
        result = x @ self.weights[name + ".weight"].T
        if name + ".bias" in self.weights:
            result += self.weights[name + ".bias"]
        return result

    def normalize_qk(self, queries, keys, prefix):
        return queries, keys

    def forward(self, tokens):
        tokens = np.asarray(tokens)
        if tokens.ndim != 2 or tokens.shape[1] == 0 or tokens.dtype.kind not in "iu":
            raise ValueError("tokens must be a nonempty [batch, length] integer array")
        if np.any(tokens < 0) or np.any(tokens >= self.config["vocab_size"]):
            raise ValueError("token outside vocabulary")
        x = self.weights["model.embed_tokens.weight"][tokens]
        batch, length = tokens.shape
        for i, window in enumerate(self.windows):
            prefix = f"model.layers.{i}"
            attention = prefix + ".self_attn"
            h = rms_norm(x, self.weights[prefix + ".input_layernorm.weight"], self.eps)
            q = self.linear(h, attention + ".q_proj").reshape(
                batch, length, self.heads, self.head_dim
            )
            k = self.linear(h, attention + ".k_proj").reshape(
                batch, length, self.kv_heads, self.head_dim
            )
            v = self.linear(h, attention + ".v_proj").reshape(
                batch, length, self.kv_heads, self.head_dim
            )
            q, k = self.normalize_qk(q, k, attention)
            q = rope(q.transpose(0, 2, 1, 3), self.freqs, self.offset, self.traditional)
            k = rope(k.transpose(0, 2, 1, 3), self.freqs, self.offset, self.traditional)
            v = v.transpose(0, 2, 1, 3)
            if self.cache[i] is not None:
                old_k, old_v = self.cache[i]
                k = np.concatenate((old_k, k), axis=2)
                v = np.concatenate((old_v, v), axis=2)
            self.cache[i] = (k, v)
            scaled_q = q * np.float32(self.head_dim**-0.5)
            scores = scaled_q @ repeat_kv(k, self.heads).swapaxes(-1, -2)
            scores = np.where(causal_mask(length, self.offset, window), scores, -np.inf)
            probabilities = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
            output = (probabilities @ repeat_kv(v, self.heads)).transpose(0, 2, 1, 3)
            x = x + self.linear(output.reshape(batch, length, -1), attention + ".o_proj")
            h = rms_norm(x, self.weights[prefix + ".post_attention_layernorm.weight"], self.eps)
            gate = self.linear(h, prefix + ".mlp.gate_proj")
            silu = gate / (1 + np.exp(-gate))
            gated = silu * self.linear(h, prefix + ".mlp.up_proj")
            x = x + self.linear(gated, prefix + ".mlp.down_proj")
        self.offset += length
        x = rms_norm(x, self.weights["model.norm.weight"], self.eps)
        if self.config.get("tie_word_embeddings", True):
            return x @ self.weights["model.embed_tokens.weight"].T
        return self.linear(x, "lm_head")

    def cache_tensors(self, stage):
        tensors = {}
        for i, (keys, values) in enumerate(self.cache):
            window = self.windows[i]
            # RotatingKVCache preserves the first concat, even when T exceeds W.
            if stage == "after_decode" and window is not None:
                keys, values = keys[..., -window:, :], values[..., -window:, :]
            prefix = f"cache.{stage}.layer{i}"
            tensors[prefix + ".keys"] = keys.copy()
            tensors[prefix + ".values"] = values.copy()
        return tensors
