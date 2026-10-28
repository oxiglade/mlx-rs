"""Deterministic little-endian GGUF v3 fixture writer; standard library only."""

import math
import re
import struct
from pathlib import Path

TYPES = {"f32": 0, "f16": 1, "q4_0": 2, "q4_1": 3, "q8_0": 8}
LLAMA_D16 = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]
QWEN3_D16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


def external_name(name):
    roots = {"model.embed_tokens": "token_embd", "model.norm": "output_norm", "lm_head": "output"}
    prefix, suffix = name.rsplit(".", 1)
    if prefix in roots:
        return roots[prefix] + "." + suffix
    match = re.fullmatch(r"model.layers.(\d+).(.+)", prefix)
    if match is None:
        raise ValueError(name)
    parts = {
        "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
        "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
        "self_attn.q_norm": "attn_q_norm", "self_attn.k_norm": "attn_k_norm",
        "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
        "input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm",
    }
    return f"blk.{match[1]}.{parts[match[2]]}.{suffix}"


def export_rows(rows, heads):
    dim = len(rows) // heads
    if dim % 2 or len(rows) % heads:
        raise ValueError("invalid Q/K head dimensions")
    return [rows[h * dim + half * (dim // 2) + j]
            for h in range(heads) for j in range(dim // 2) for half in range(2)]


def half(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]


def encode_tensor(values, shape, storage):
    values = list(values)
    if len(values) != math.prod(shape) or not all(map(math.isfinite, values)):
        raise ValueError("invalid tensor values")
    if storage in ("f32", "f16"):
        return struct.pack("<" + ("f" if storage == "f32" else "e") * len(values), *values)
    if len(shape) != 2 or shape[-1] % 32:
        raise ValueError("block tensors need a matrix with width divisible by 32")
    result = bytearray()
    for start in range(0, len(values), 32):
        block = values[start:start + 32]
        if storage == "q4_1":
            bias = half(min(block))
            scale = half((max(block) - bias) / 15) or half(2**-14)
            q = [max(0, min(15, round((v - bias) / scale))) for v in block]
            result.extend(struct.pack("<ee", scale, bias))
        elif storage == "q4_0":
            scale = half(max(block, key=abs) / -8) or half(2**-14)
            q = [max(0, min(15, round(v / scale) + 8)) for v in block]
            result.extend(struct.pack("<e", scale))
        elif storage == "q8_0":
            scale = half(max(map(abs, block)) / 127) or half(2**-14)
            q = [max(-128, min(127, round(v / scale))) for v in block]
            result.extend(struct.pack("<e32b", scale, *q))
            continue
        else:
            raise ValueError(storage)
        result.extend(bytes(q[j] | q[j + 16] << 4 for j in range(16)))
    return bytes(result)


def string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def write(path, metadata, tensors):
    """metadata maps keys to (type, value); tensors to (shape, storage, bytes)."""
    if metadata.get("general.alignment", (4, 32)) != (4, 32):
        raise ValueError("fixture writer requires alignment 32")
    def value(kind, item):
        if kind == 8:
            return string(item)
        if kind == 9:
            element, items = item
            return struct.pack("<IQ", element, len(items)) + b"".join(value(element, x) for x in items)
        return struct.pack({4: "<I", 5: "<i", 6: "<f", 7: "<?"}[kind], item)

    header = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(metadata)))
    for key, (kind, item) in sorted(metadata.items()):
        header.extend(string(key) + struct.pack("<I", kind) + value(kind, item))
    payload = bytearray()
    for key, (shape, storage, data) in sorted(tensors.items()):
        payload.extend(b"\0" * (-len(payload) % 32))
        header.extend(string(key) + struct.pack("<I", len(shape)))
        header.extend(struct.pack("<" + "Q" * len(shape), *reversed(shape)))
        header.extend(struct.pack("<IQ", TYPES[storage], len(payload)))
        payload.extend(data)
    header.extend(b"\0" * (-len(header) % 32))
    Path(path).write_bytes(header + payload)
