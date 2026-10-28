"""Independent GGUF reader: no writer, MLX, bridge, or Rust imports."""

import math
import re
import struct
from pathlib import Path

import numpy as np


def read(path):
    data = memoryview(Path(path).read_bytes())
    cursor = 0

    def take(n):
        nonlocal cursor
        if n < 0 or cursor + n > len(data):
            raise ValueError("truncated GGUF")
        result = data[cursor:cursor + n]
        cursor += n
        return result

    def unpack(fmt):
        return struct.unpack(fmt, take(struct.calcsize(fmt)))[0]

    def text():
        return bytes(take(unpack("<Q"))).decode("utf-8")

    def metadata_value(kind):
        if kind == 8:
            return text()
        if kind == 9:
            element, count = unpack("<I"), unpack("<Q")
            if element == 9 or count > len(data):
                raise ValueError("invalid GGUF metadata array")
            return [metadata_value(element) for _ in range(count)]
        formats = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
        if kind not in formats:
            raise ValueError("unsupported metadata type")
        return unpack(formats[kind])

    if bytes(take(4)) != b"GGUF" or unpack("<I") != 3:
        raise ValueError("expected little-endian GGUF v3")
    count, metadata_count = unpack("<Q"), unpack("<Q")
    if count + metadata_count > len(data):
        raise ValueError("invalid GGUF counts")
    metadata, kinds = {}, {}
    for _ in range(metadata_count):
        name, kind = text(), unpack("<I")
        if name in metadata:
            raise ValueError("duplicate metadata")
        kinds[name] = kind
        metadata[name] = metadata_value(kind)
    descriptors = {}
    for _ in range(count):
        name, rank = text(), unpack("<I")
        if name in descriptors or not 1 <= rank <= 4:
            raise ValueError("duplicate tensor or invalid rank")
        shape = tuple(reversed([unpack("<Q") for _ in range(rank)]))
        kind, offset = unpack("<I"), unpack("<Q")
        descriptors[name] = (shape, kind, offset)
    alignment = metadata.get("general.alignment", 32)
    if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("invalid alignment")
    base = (cursor + alignment - 1) // alignment * alignment
    arrays, decoded, spans = {}, {}, []

    def insert(name, value):
        if name in arrays:
            raise ValueError(f"duplicate converted slot: {name}")
        arrays[name] = value

    for name, (shape, kind, offset) in descriptors.items():
        size = math.prod(shape)
        if not size or offset % alignment:
            raise ValueError("invalid tensor dimensions or offset")
        if kind in (0, 1):
            length = size * (4 if kind == 0 else 2)
        elif kind in (2, 3, 8) and len(shape) == 2 and shape[-1] % 32 == 0:
            length = size // 32 * {2: 18, 3: 20, 8: 34}[kind]
        else:
            raise ValueError("outside admitted GGUF tensor subset")
        start, end = base + offset, base + offset + length
        if end > len(data):
            raise ValueError("truncated tensor")
        spans.append((start, end))
        raw = data[start:end]
        if kind in (0, 1):
            insert(name, np.frombuffer(raw, dtype="<f4" if kind == 0 else "<f2").reshape(shape).copy())
            decoded[name] = arrays[name].astype(np.float32)
            continue
        bits = 8 if kind == 8 else 4
        block_size = {2: 18, 3: 20, 8: 34}[kind]
        integers, scales, biases = [], [], []
        for block in range(size // 32):
            block_data = raw[block * block_size:(block + 1) * block_size]
            scale = struct.unpack_from("<e", block_data)[0]
            if kind == 8:
                q = np.frombuffer(block_data[2:], dtype=np.int8).astype(np.int16) + 128
                bias = np.float16(-128 * scale)
            else:
                nibble = np.frombuffer(block_data[4 if kind == 3 else 2:], dtype=np.uint8)
                q = np.concatenate((nibble & 15, nibble >> 4))
                bias = np.float16(struct.unpack_from("<e", block_data, 2)[0] if kind == 3 else -8 * scale)
            integers.extend(q.tolist())
            scales.append(scale)
            biases.append(bias)
        q = np.array(integers, dtype=np.uint32).reshape(shape)
        words = np.zeros((shape[0], shape[1] // (32 // bits)), dtype=np.uint32)
        for lane in range(32 // bits):
            words |= q[:, lane::32 // bits] << (lane * bits)
        prefix = name.removesuffix(".weight")
        scales = np.array(scales, dtype=np.float16).reshape(shape[0], shape[1] // 32)
        biases = np.array(biases, dtype=np.float16).reshape(scales.shape)
        insert(prefix + ".weight", words)
        insert(prefix + ".scales", scales)
        insert(prefix + ".biases", biases)
        # MLX rounds both the affine multiply and the addition in F16 (ruling J).
        product = np.multiply(q.astype(np.float16), np.repeat(scales, 32, axis=1), dtype=np.float16)
        decoded[name] = np.add(product, np.repeat(biases, 32, axis=1), dtype=np.float16).astype(np.float32)
    if any(left[1] > right[0] for left, right in zip(sorted(spans), sorted(spans)[1:])):
        raise ValueError("overlapping tensors")
    return metadata, kinds, descriptors, arrays, decoded


def canonical_weights(metadata, decoded):
    arch = metadata["general.architecture"]
    stems = {"token_embd": "model.embed_tokens", "output_norm": "model.norm", "output": "lm_head"}
    blocks = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
              "attn_output": "self_attn.o_proj", "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
              "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
              "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm"}
    result = {}
    for name, array in decoded.items():
        stem, slot = name.rsplit(".", 1)
        if stem in stems:
            target = stems[stem]
        else:
            match = re.fullmatch(r"blk\.(0|[1-9][0-9]*)\.(\w+)", stem)
            if match is None:
                raise ValueError(name)
            target = f"model.layers.{match[1]}.{blocks[match[2]]}"
            if arch == "llama" and match[2] in ("attn_q", "attn_k"):
                heads = metadata[f"{arch}.attention.head_count" + ("_kv" if match[2] == "attn_k" else "")]
                dim = array.shape[0] // heads
                indices = [h * dim + 2 * j + half for h in range(heads) for half in range(2) for j in range(dim // 2)]
                array = array[indices]
        result[target + "." + slot] = array
    return result


def model_config(metadata, descriptors):
    arch = metadata["general.architecture"]
    if arch not in ("llama", "qwen3"):
        raise ValueError(f"unsupported GGUF architecture: {arch}")
    fields = {"hidden_size": "embedding_length", "intermediate_size": "feed_forward_length",
              "num_hidden_layers": "block_count", "num_attention_heads": "attention.head_count",
              "num_key_value_heads": "attention.head_count_kv", "head_dim": "attention.key_length",
              "rms_norm_eps": "attention.layer_norm_rms_epsilon", "rope_theta": "rope.freq_base",
              "max_position_embeddings": "context_length"}
    result = {key: metadata[arch + "." + source] for key, source in fields.items()}
    result.update(model_type=arch, vocab_size=descriptors["token_embd.weight"][0][0],
                  tie_word_embeddings="output.weight" not in descriptors, rope_scaling=None)
    if metadata.get(arch + ".rope.scaling.type", "none") == "linear":
        result["rope_scaling"] = {"rope_type": "linear", "factor": metadata[arch + ".rope.scaling.factor"]}
    return result


def reference(path, prompt, decode_ids=None):
    from .llama import Llama
    from .qwen3 import Qwen3

    metadata, _, descriptors, _, decoded = read(path)
    config = model_config(metadata, descriptors)
    architecture = Llama if config["model_type"] == "llama" else Qwen3
    model = architecture(config, canonical_weights(metadata, decoded))
    logits = model.forward(np.array([prompt], dtype=np.int32))
    result = {"prefill.full.logits": logits}
    result.update(model.cache_tensors("after_prefill"))
    greedy = []
    for step in range(8):
        token = int(np.argmax(logits[:, -1, :]))
        greedy.append(token)
        feed = token if decode_ids is None else decode_ids[step]
        logits = model.forward(np.array([[feed]], dtype=np.int32))
        result[f"decode.step{step}.logits"] = logits[:, -1, :]
    result.update(model.cache_tensors("after_decode"))
    for chunk in (1, 3, 8):
        model = architecture(config, canonical_weights(metadata, decoded))
        result[f"prefill.chunk{chunk}.logits"] = np.concatenate([
            model.forward(np.array([prompt[i:i + chunk]], dtype=np.int32))
            for i in range(0, len(prompt), chunk)], axis=1)
    return result, greedy
