"""Writes the tiny seeded EmbeddingGemma 2 text fixture and its NumPy expectations.

The fixture exercises sliding and full layers, a `per_layer_config` override,
GQA, per-layer inputs, non-unit layer scalars and right padding. Each named
mutation of the reference must move some expectation beyond the tolerance.
"""

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from numpy_reference.common import read_json, read_safetensors, write_safetensors
from numpy_reference.embedding_gemma2 import EmbeddingGemma2Text

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "embedding-fixtures" / "embedding-gemma2-text"
NUMPY_VERSION = "2.2.6"
SEED = 1729
TOLERANCE = {"atol": 2e-4, "rtol": 2e-4}
MUTATIONS = ["no_v_norm", "window_exclusive", "no_layer_scalar", "pool_padding"]

SPECIAL = ["<pad>", "<eos>", "<bos>", "<unk>"]
WORDS = (
    "the a small large red green blue fox dog cat bird runs jumps sleeps over under "
    "near hill river tree stone house road quick slow bright dark old new one two three "
    "four five planet mars star moon sun task query text title none search result"
).split()
MEDIA = ["<|image|>", "<|audio|>", "<|video|>"]
VOCAB = SPECIAL + WORDS + [f"w{i}" for i in range(64 - len(SPECIAL) - len(WORDS) - len(MEDIA))] + MEDIA

CONFIG = {
    "model_type": "embedding_gemma2",
    "image_token_id": VOCAB.index("<|image|>"),
    "audio_token_id": VOCAB.index("<|audio|>"),
    "video_token_id": VOCAB.index("<|video|>"),
    "text_config": {
        "model_type": "embedding_gemma2_text",
        "attention_bias": False,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "pad_token_id": 0,
        "embedding_dim": 24,
        "head_dim": 8,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": 32,
        "hidden_size_per_layer_input": 8,
        "intermediate_size": 48,
        "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
        "max_position_embeddings": 64,
        "num_attention_heads": 4,
        "num_hidden_layers": 3,
        "num_key_value_heads": 2,
        "per_layer_config": {"02": {"head_dim": 16, "num_key_value_heads": 1}},
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "full_attention": {"rope_theta": 1000000.0, "rope_type": "default"},
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        },
        "sliding_window": 2,
        "vocab_size": len(VOCAB),
    },
}

# Lengths 3..12 with the batch padded to the longest; window 2 leaves most pairs masked.
TEXTS = [
    "task search result query planet mars",
    "the quick red fox jumps over the slow dog near the old stone house",
    "one",
    "title none text a bright star over the dark river",
]


def tokenizer_json():
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": VOCAB.index(t), "content": t, "single_word": False, "lstrip": False,
             "rstrip": False, "normalized": False, "special": True}
            for t in SPECIAL + MEDIA
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "<bos>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "<eos>", "type_id": 0}},
            ],
            "pair": [
                {"SpecialToken": {"id": "<bos>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 1}},
                {"SpecialToken": {"id": "<eos>", "type_id": 0}},
            ],
            "special_tokens": {
                t: {"id": t, "ids": [VOCAB.index(t)], "tokens": [t]} for t in ("<bos>", "<eos>")
            },
        },
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": {t: i for i, t in enumerate(VOCAB)}, "unk_token": "<unk>"},
    }


def shapes():
    text = CONFIG["text_config"]
    hidden, inter = text["hidden_size"], text["intermediate_size"]
    width, count = text["hidden_size_per_layer_input"], text["num_hidden_layers"]
    p = "language_model."
    out = {
        p + "embed_tokens.weight": (text["vocab_size"], hidden),
        p + "ple.per_layer_model_projection.weight": (count * width, hidden),
        p + "ple.per_layer_projection_norm.weight": (width,),
        p + "norm.weight": (hidden,),
        p + "embedding_projection.weight": (text["embedding_dim"], hidden),
    }
    for i in range(count):
        o = text["per_layer_config"].get(f"{i:02d}", {})
        d = o.get("head_dim", text["head_dim"])
        q = text["num_attention_heads"] * d
        kv = o.get("num_key_value_heads", text["num_key_value_heads"]) * d
        l = f"{p}layers.{i}."
        out.update({
            l + "self_attn.q_proj.weight": (q, hidden),
            l + "self_attn.k_proj.weight": (kv, hidden),
            l + "self_attn.v_proj.weight": (kv, hidden),
            l + "self_attn.o_proj.weight": (hidden, q),
            l + "self_attn.q_norm.weight": (d,),
            l + "self_attn.k_norm.weight": (d,),
            l + "mlp.gate_proj.weight": (inter, hidden),
            l + "mlp.up_proj.weight": (inter, hidden),
            l + "mlp.down_proj.weight": (hidden, inter),
            l + "input_layernorm.weight": (hidden,),
            l + "post_attention_layernorm.weight": (hidden,),
            l + "pre_feedforward_layernorm.weight": (hidden,),
            l + "post_feedforward_layernorm.weight": (hidden,),
            l + "ple_block.per_layer_input_gate.weight": (width, hidden),
            l + "ple_block.per_layer_projection.weight": (hidden, width),
            l + "ple_block.post_per_layer_input_norm.weight": (hidden,),
            l + "layer_scalar": (1,),
        })
    return out


def weights():
    rng = np.random.Generator(np.random.PCG64(SEED))
    tensors = {}
    for name, shape in sorted(shapes().items()):
        if len(shape) == 2:
            value = rng.normal(0, shape[1] ** -0.5, shape)
        elif name.endswith("layer_scalar"):
            value = rng.uniform(0.5, 1.5, shape)
        else:
            value = rng.uniform(0.5, 1.5, shape)
        tensors[name] = value.astype(np.float32)
    # Media towers are present in real checkpoints and must be ignored by the text embedder.
    tensors["vision_tower.patch_embedder.input_proj.weight"] = np.zeros((4, 4), np.float32)
    tensors["embed_audio.embedding_projection.weight"] = np.zeros((4, 4), np.float32)
    return tensors


def encode(text):
    return [2] + [VOCAB.index(word) for word in text.split()] + [1]


def write(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dump = lambda value: json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    (directory / "config.json").write_text(dump(CONFIG))
    (directory / "tokenizer.json").write_text(dump(tokenizer_json()))
    (directory / "tokenizer_config.json").write_text(dump({"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>"}))
    write_safetensors(directory / "model.safetensors", weights())
    tensors = read_safetensors(directory / "model.safetensors")
    ids = [encode(text) for text in TEXTS]
    batch = EmbeddingGemma2Text(CONFIG, tensors).embed(ids)
    singles = np.concatenate([EmbeddingGemma2Text(CONFIG, tensors).embed([s]) for s in ids])
    if not np.isfinite(batch).all():
        raise ValueError("reference embeddings are not finite")
    # Padding must not leak into valid rows; the reference itself proves it before blessing.
    if not np.allclose(batch, singles, atol=1e-6, rtol=0):
        raise ValueError("reference rows depend on batch padding")
    for mutation in MUTATIONS:
        mutated = EmbeddingGemma2Text(CONFIG, tensors, mutation).embed(ids)
        excess = np.abs(mutated - batch) - (TOLERANCE["atol"] + TOLERANCE["rtol"] * np.abs(batch))
        if excess.max() <= 0:
            raise ValueError(f"fixture does not detect mutation {mutation}")
    (directory / "inputs.json").write_text(dump({"texts": TEXTS, "token_ids": ids, "seed": SEED}))
    (directory / "expectations.json").write_text(dump({
        "provenance": f"numpy {NUMPY_VERSION} f32 reference, numpy_reference/embedding_gemma2.py",
        "tolerances": {"embeddings": TOLERANCE},
        "mutations_detected": MUTATIONS,
    }))
    write_safetensors(directory / "expectations.safetensors", {"embeddings": batch})


def digest(directory):
    h = hashlib.sha256()
    for path in sorted(Path(directory).iterdir()):
        h.update(path.name.encode() + b"\0" + path.read_bytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="regenerate and compare without writing")
    args = parser.parse_args()
    if np.__version__ != NUMPY_VERSION:
        parser.error(f"requires numpy {NUMPY_VERSION}, got {np.__version__}")
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        write(a)
        write(b)
        if digest(a) != digest(b):
            raise SystemExit("generation is not deterministic")
        if args.check:
            if not OUTPUT.is_dir() or digest(a) != digest(OUTPUT):
                raise SystemExit(f"{OUTPUT} differs from regeneration")
            print("fixture matches regeneration")
            return
    write(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
