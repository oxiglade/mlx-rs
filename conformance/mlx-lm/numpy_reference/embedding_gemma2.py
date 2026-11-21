"""Independent f32 NumPy EmbeddingGemma 2 text encoder.

Bidirectional Gemma 4 layers with per-layer inputs, mean pooling over valid tokens,
then L2 normalization, as in the EmbeddingGemma 2 text path of transformers and mlx-vlm.
"""

import numpy as np

from .common import rms_norm, rope_frequencies


def gelu_tanh(x):
    inner = np.float32(np.sqrt(2 / np.pi)) * (x + np.float32(0.044715) * x**3)
    return np.float32(0.5) * x * (np.float32(1) + np.tanh(inner))


def rotate_half(x, freqs):
    positions = np.arange(x.shape[-2], dtype=np.float32)
    angles = positions[:, None] / freqs
    angles = np.concatenate([angles, angles], axis=-1)
    half = x.shape[-1] // 2
    rotated = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * np.cos(angles) + rotated * np.sin(angles)


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


class LayerShape:
    def __init__(self, text, index):
        kind = text["layer_types"][index]
        overrides = text.get("per_layer_config", {}).get(f"{index:02d}", {})
        self.sliding = kind == "sliding_attention"
        self.heads = overrides.get("num_attention_heads", text["num_attention_heads"])
        self.kv_heads = overrides.get("num_key_value_heads", text["num_key_value_heads"])
        self.head_dim = overrides.get("head_dim", text["head_dim"])
        self.theta = text["rope_parameters"][kind]["rope_theta"]


class EmbeddingGemma2Text:
    """`mutation` names one deliberate deviation, used only to prove fixture sensitivity."""

    PREFIX = "language_model."

    def __init__(self, config, weights, mutation=None):
        self.text = config["text_config"]
        self.eps = np.float32(self.text["rms_norm_eps"])
        self.weights = {
            name[len(self.PREFIX):]: value.astype(np.float32)
            for name, value in weights.items()
            if name.startswith(self.PREFIX)
        }
        self.layers = [LayerShape(self.text, i) for i in range(self.text["num_hidden_layers"])]
        self.mutation = mutation

    def w(self, name):
        return self.weights[name]

    def linear(self, x, name):
        return x @ self.w(name + ".weight").T

    def norm(self, x, name):
        return rms_norm(x, self.w(name + ".weight"), self.eps)

    def attention(self, x, prefix, shape, valid):
        batch, length, _ = x.shape

        def heads(y, count):
            return y.reshape(batch, length, count, shape.head_dim)

        q = heads(self.linear(x, prefix + ".q_proj"), shape.heads)
        k = heads(self.linear(x, prefix + ".k_proj"), shape.kv_heads)
        v = heads(self.linear(x, prefix + ".v_proj"), shape.kv_heads)
        q = self.norm(q, prefix + ".q_norm").transpose(0, 2, 1, 3)
        k = self.norm(k, prefix + ".k_norm").transpose(0, 2, 1, 3)
        if self.mutation != "no_v_norm":
            v = rms_norm(v, np.float32(1), self.eps)
        v = v.transpose(0, 2, 1, 3)
        freqs = rope_frequencies(shape.head_dim, shape.theta)
        q, k = rotate_half(q, freqs), rotate_half(k, freqs)
        repeat = shape.heads // shape.kv_heads
        k, v = np.repeat(k, repeat, axis=1), np.repeat(v, repeat, axis=1)

        allowed = np.broadcast_to(valid[:, None, None, :], (batch, 1, length, length))
        if shape.sliding:
            positions = np.arange(length)
            window = self.text["sliding_window"]
            if self.mutation == "window_exclusive":
                window -= 1
            near = np.abs(positions[:, None] - positions[None, :]) <= window
            allowed = allowed & near
        scores = q @ k.transpose(0, 1, 3, 2)
        # Padded queries far from every valid key see no allowed key; a finite floor
        # keeps those rows finite so they cannot poison pooling through NaN * 0.
        scores = np.where(allowed, scores, np.finfo(np.float32).min)
        out = softmax(scores) @ v
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.linear(out, prefix + ".o_proj")

    def hidden(self, ids, valid):
        text = self.text
        hidden = text["hidden_size"]
        count, width = len(self.layers), text["hidden_size_per_layer_input"]
        x = self.w("embed_tokens.weight")[ids] * np.float32(np.sqrt(hidden))
        per_layer = self.linear(x, "ple.per_layer_model_projection") * np.float32(hidden**-0.5)
        per_layer = per_layer.reshape(*ids.shape, count, width)
        per_layer = self.norm(per_layer, "ple.per_layer_projection_norm")
        for i, shape in enumerate(self.layers):
            p = f"layers.{i}"
            h = self.attention(self.norm(x, p + ".input_layernorm"), p + ".self_attn", shape, valid)
            x = x + self.norm(h, p + ".post_attention_layernorm")
            h = self.norm(x, p + ".pre_feedforward_layernorm")
            h = gelu_tanh(self.linear(h, p + ".mlp.gate_proj")) * self.linear(h, p + ".mlp.up_proj")
            x = x + self.norm(self.linear(h, p + ".mlp.down_proj"), p + ".post_feedforward_layernorm")
            gate = gelu_tanh(self.linear(x, p + ".ple_block.per_layer_input_gate")) * per_layer[:, :, i]
            x = x + self.norm(
                self.linear(gate, p + ".ple_block.per_layer_projection"),
                p + ".ple_block.post_per_layer_input_norm",
            )
            if self.mutation != "no_layer_scalar":
                x = x * self.w(p + ".layer_scalar")
        return self.linear(self.norm(x, "norm"), "embedding_projection")

    def embed(self, sequences):
        """Embeds a list of token-id lists; returns `[batch, embedding_dim]` f32 rows."""
        length = max(map(len, sequences))
        pad = self.text["pad_token_id"]
        ids = np.array([s + [pad] * (length - len(s)) for s in sequences], np.int64)
        valid = np.array([[True] * len(s) + [False] * (length - len(s)) for s in sequences])
        states = self.hidden(ids, valid)
        mask = valid[..., None].astype(np.float32)
        if self.mutation == "pool_padding":
            mask = np.ones_like(mask)
        pooled = (states * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), np.float32(1e-9))
        norm = np.sqrt((pooled * pooled).sum(axis=-1, keepdims=True))
        return (pooled / np.maximum(norm, np.float32(1e-9))).astype(np.float32)
