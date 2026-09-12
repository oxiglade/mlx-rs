import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from coordinate import compare_tensors, coordinate, expected_shapes
from numpy_reference.common import (
    causal_mask,
    dequantize,
    load_weights,
    read_safetensors,
    repeat_kv,
    rope,
    rope_frequencies,
    write_safetensors,
)
from numpy_reference.llama import Llama
from numpy_reference.qwen3 import Qwen3
from numpy_reference.run import reference_tensors


class MathTests(unittest.TestCase):
    def test_llama3_smoothing(self):
        scaling = {
            "rope_type": "llama3",
            "factor": 8,
            "low_freq_factor": 1,
            "high_freq_factor": 4,
            "original_max_position_embeddings": 32,
        }
        # theta=64 gives denominators 1, 4, 16: one in each frequency band.
        smooth = (32 / (8 * np.pi) - 1) / 3
        expected = np.array([1, 4 / ((1 - smooth) / 8 + smooth), 128], np.float32)
        actual = rope_frequencies(6, 64, scaling)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
        scaling["original_max_position_embeddings"] = float(np.float32(8 * np.pi))
        np.testing.assert_allclose(
            rope_frequencies(6, 64, scaling), [1, 4, 128], rtol=1e-6
        )

    def test_rope_offset_and_layout(self):
        x = np.array([[[[1, 2, 3, 4]]]], np.float32)
        freqs = np.array([1, 2], np.float32)
        c, s = np.cos([1, 0.5]), np.sin([1, 0.5])
        expected = [
            c[0] - 3 * s[0], 2 * c[1] - 4 * s[1],
            s[0] + 3 * c[0], 2 * s[1] + 4 * c[1],
        ]
        np.testing.assert_allclose(rope(x, freqs, offset=1).ravel(), expected, atol=1e-6)
        traditional = [
            c[0] - 2 * s[0], s[0] + 2 * c[0],
            3 * c[1] - 4 * s[1], 3 * s[1] + 4 * c[1],
        ]
        np.testing.assert_allclose(rope(x, freqs, 1, True).ravel(), traditional, atol=1e-6)
        np.testing.assert_array_equal(
            rope_frequencies(4, 100, {"type": "linear", "factor": 3}), [3, 30]
        )

    def test_sliding_mask(self):
        expected = [
            [False, False, True, True, True, False, False],
            [False, False, False, True, True, True, False],
            [False, False, False, False, True, True, True],
        ]
        mask = causal_mask(3, offset=4, window=3)
        self.assertEqual(mask.shape, (3, 7))
        np.testing.assert_array_equal(mask, expected)
        np.testing.assert_array_equal(
            causal_mask(1, 5, 2), [[False, False, False, False, True, True]]
        )
        np.testing.assert_array_equal(causal_mask(3), np.tri(3, dtype=bool))

    def test_dequantize_four_bit(self):
        packed = np.array([[0x76543210]], dtype=np.uint32)
        scales = np.array([[2, 3]], np.float32)
        biases = np.array([[-1, 10]], np.float32)
        actual = dequantize(packed, scales, biases, group_size=4, bits=4)
        np.testing.assert_array_equal(actual, [[-1, 1, 3, 5, 22, 25, 28, 31]])
        self.assertEqual(actual.dtype, np.float32)

    def test_dequantize_eight_bit(self):
        packed = np.array([[0xFF800100]], dtype=np.uint32)
        actual = dequantize(
            packed, np.array([[0.5]], np.float32), np.array([[-2]], np.float32), 4, 8
        )
        np.testing.assert_array_equal(actual, [[-2, -1.5, 62, 125.5]])

    def test_gqa_repeat(self):
        keys = np.arange(12, dtype=np.float32).reshape(1, 2, 3, 2)
        actual = repeat_kv(keys, 6)
        self.assertEqual(actual.shape, (1, 6, 3, 2))
        np.testing.assert_array_equal(actual, keys[:, [0, 0, 0, 1, 1, 1]])
        with self.assertRaises(ValueError):
            repeat_kv(keys, 3)


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(
            hidden_size=8, intermediate_size=12, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=4,
            vocab_size=7, rms_norm_eps=1e-5, tie_word_embeddings=False,
            layer_types=["full_attention", "sliding_attention"], sliding_window=3,
        )
        rng = np.random.default_rng(17)
        self.weights = {
            "model.embed_tokens.weight": rng.normal(0, 0.2, (7, 8)).astype(np.float32),
            "model.norm.weight": np.ones(8, np.float32),
            "lm_head.weight": rng.normal(0, 0.2, (7, 8)).astype(np.float32),
        }
        for i in range(2):
            prefix = f"model.layers.{i}"
            for norm in ("input_layernorm", "post_attention_layernorm"):
                self.weights[f"{prefix}.{norm}.weight"] = np.ones(8, np.float32)
            for norm in ("q_norm", "k_norm"):
                self.weights[f"{prefix}.self_attn.{norm}.weight"] = np.array(
                    [0.5, 1, 1.5, 2], np.float32
                )
            for name, shape in {
                "self_attn.q_proj": (8, 8), "self_attn.k_proj": (4, 8),
                "self_attn.v_proj": (4, 8), "self_attn.o_proj": (8, 8),
                "mlp.gate_proj": (12, 8), "mlp.up_proj": (12, 8), "mlp.down_proj": (8, 12),
            }.items():
                self.weights[f"{prefix}.{name}.weight"] = rng.normal(
                    0, 0.2, shape
                ).astype(np.float32)

    def test_cached_forward_and_temporal_cache(self):
        tokens = np.array([[1, 3, 2, 0, 4, 6, 2, 1, 3, 0, 2, 5, 1]])
        for architecture in (Llama, Qwen3):
            with self.subTest(architecture=architecture.__name__):
                full = architecture(self.config, self.weights)
                all_logits = full.forward(tokens)
                cached = architecture(self.config, self.weights)
                logits = [cached.forward(tokens[:, :5])]
                prefill = cached.cache_tensors("after_prefill")
                self.assertEqual(
                    prefill["cache.after_prefill.layer1.keys"].shape, (1, 1, 5, 4)
                )
                for step in range(5, 13):
                    logits.append(cached.forward(tokens[:, step:step + 1]))
                np.testing.assert_allclose(
                    np.concatenate(logits, axis=1), all_logits, atol=2e-6, rtol=2e-6
                )
                cache = cached.cache_tensors("after_decode")
                self.assertEqual(
                    cache["cache.after_decode.layer0.keys"].shape, (1, 1, 13, 4)
                )
                for part, values in zip(("keys", "values"), full.cache[1]):
                    np.testing.assert_allclose(
                        cache[f"cache.after_decode.layer1.{part}"],
                        values[..., -3:, :], atol=2e-6,
                    )
                self.assertEqual(all_logits.dtype, np.float32)

    def test_qwen_norm_is_per_head(self):
        model = Qwen3(self.config, self.weights)
        q = np.array([[[[1, 2, 3, 4], [10, 20, 30, 40]]]], np.float32)
        k = q[:, :, :1]
        queries, keys = model.normalize_qk(q, k, "model.layers.0.self_attn")
        weight = np.array([0.5, 1, 1.5, 2])
        variance = np.mean(q.astype(np.float64)**2, axis=-1, keepdims=True)
        expected = q.astype(np.float64) / np.sqrt(variance + 1e-5) * weight
        np.testing.assert_allclose(queries, expected, rtol=2e-7)
        np.testing.assert_allclose(keys, expected[:, :, :1], rtol=2e-7)

    def test_tied_head(self):
        tied = Llama(dict(self.config, tie_word_embeddings=True), self.weights)
        weights = dict(self.weights)
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
        untied = Llama(self.config, weights)
        tokens = np.array([[1, 2, 3]])
        np.testing.assert_array_equal(tied.forward(tokens), untied.forward(tokens))

    def test_affine_embedding_and_projection(self):
        packed = dict(self.weights)
        expanded = dict(self.weights)
        for prefix, rows, words in (
            ("model.embed_tokens", 7, [0x76543210]),
            ("model.layers.0.self_attn.q_proj", 8, [0x03020100, 0x07060504]),
        ):
            packed[prefix + ".weight"] = np.tile(np.array([words], np.uint32), (rows, 1))
            packed[prefix + ".scales"] = np.full((rows, 2), 0.125, np.float32)
            packed[prefix + ".biases"] = np.full((rows, 2), -0.5, np.float32)
            expanded[prefix + ".weight"] = np.tile(
                np.array([[-0.5, -0.375, -0.25, -0.125, 0, 0.125, 0.25, 0.375]], np.float32),
                (rows, 1),
            )
        quantization = {
            "group_size": 4, "bits": 8,
            "model.embed_tokens": {"group_size": 4, "bits": 4},
        }
        for tied in (False, True):
            config = dict(self.config, tie_word_embeddings=tied)
            model = Llama(dict(config, quantization=quantization), packed)
            plain = Llama(config, expanded)
            np.testing.assert_array_equal(
                model.forward(np.array([[1, 2, 3]])), plain.forward(np.array([[1, 2, 3]]))
            )

    def test_reference_outputs_eight_greedy_forwards(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            config = dict(self.config, model_type="qwen3")
            prompt = [1, 3, 2, 0, 4]
            (fixture / "config.json").write_text(json.dumps(config))
            (fixture / "expectations.json").write_text(json.dumps({
                "prefill": {"token_ids": prompt, "T": len(prompt)}
            }))
            write_safetensors(fixture / "model.safetensors", self.weights)
            actual = reference_tensors(fixture)
            self.assertEqual(actual.keys(), expected_shapes(config, len(prompt)).keys())
            for key, shape in expected_shapes(config, len(prompt)).items():
                self.assertEqual(actual[key].shape, shape)
                self.assertEqual(actual[key].dtype, np.float32)
            model = Qwen3(config, self.weights)
            logits = model.forward(np.array([prompt]))
            for step in range(8):
                token = int(np.argmax(logits[0, -1]))
                logits = model.forward(np.array([[token]]))
                np.testing.assert_array_equal(actual[f"decode.step{step}.logits"], logits[:, -1])


class CoordinatorTests(unittest.TestCase):
    def test_comparison_rejects_mutations(self):
        key = "prefill.full.logits"
        oracle = {key: np.array([[[0, 1, 2]]], np.float32)}
        shapes = {key: (1, 1, 3)}
        tolerances = {name: dict(atol=2e-4, rtol=2e-4) for name in ("logits", "cache")}
        self.assertTrue(compare_tensors(oracle, oracle, shapes, tolerances)[key]["passed"])
        for mutated in (
            {}, {key: oracle[key].astype(np.float64)}, {key: oracle[key].reshape(3)},
            {key: oracle[key] + 0.01}, {key: np.full((1, 1, 3), np.nan, np.float32)},
        ):
            with self.subTest(mutated=mutated):
                report = compare_tensors(oracle, mutated, shapes, tolerances)[key]
                self.assertFalse(report["passed"])
        nearby = {key: oracle[key] + np.float32(1e-4)}
        report = compare_tensors(oracle, nearby, shapes, tolerances)[key]
        self.assertTrue(report["passed"])
        self.assertGreater(report["max_abs"], 0)
        json.dumps(report, allow_nan=False)

    def test_safetensors_shards_and_determinism(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays = {
                "z": np.arange(6, dtype=np.float32).reshape(2, 3),
                "a": np.array([[0xFEDCBA98]], np.uint32),
            }
            write_safetensors(root / "one", arrays)
            write_safetensors(root / "two", dict(reversed(list(arrays.items()))))
            self.assertEqual((root / "one").read_bytes(), (root / "two").read_bytes())
            for name, array in arrays.items():
                np.testing.assert_array_equal(read_safetensors(root / "one")[name], array)
                write_safetensors(root / f"{name}.safetensors", {name: array})
            index = {"weight_map": {name: f"{name}.safetensors" for name in arrays}}
            (root / "model.safetensors.index.json").write_text(json.dumps(index))
            for name, array in load_weights(root).items():
                np.testing.assert_array_equal(array, arrays[name])
            index["weight_map"]["a"] = "z.safetensors"
            (root / "model.safetensors.index.json").write_text(json.dumps(index))
            with self.assertRaises(ValueError):
                load_weights(root)

    def test_bfloat16_cannot_hide_output_dtype_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bf16.safetensors"
            header = json.dumps({
                "x": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}
            }).encode()
            header += b" " * (-len(header) % 8)
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x80\x3f")
            with self.assertRaises(ValueError):
                read_safetensors(path)
            np.testing.assert_array_equal(read_safetensors(path, allow_bfloat16=True)["x"], [1])

    def test_corpus_requires_agreement_and_hashes_all_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixtures" / "llama-test"
            fixture.mkdir(parents=True)
            out = root / "out"
            config = dict(
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                hidden_size=8, vocab_size=3, layer_types=["sliding_attention"],
                sliding_window=2,
            )
            expectations = {
                "prefill": {"T": 4},
                "tolerances": {
                    name: dict(atol=2e-4, rtol=2e-4) for name in ("logits", "cache")
                },
            }
            (fixture / "config.json").write_text(json.dumps(config))
            (fixture / "expectations.json").write_text(json.dumps(expectations))
            arrays = {
                key: np.zeros(shape, np.float32)
                for key, shape in expected_shapes(config, 4).items()
            }
            write_safetensors(fixture / "expectations.safetensors", arrays)
            write_safetensors(out / "llama-test.safetensors", arrays)
            report, manifest = coordinate([fixture], out, fixture.parent)
            self.assertTrue(report["passed"])
            self.assertEqual(len(manifest["files"]), 4)
            self.assertEqual(coordinate([fixture], out, fixture.parent), (report, manifest))
            command = [
                sys.executable, str(Path(__file__).with_name("coordinate.py")),
                "--fixtures-root", str(fixture.parent), "--numpy-dir", str(out),
                "--corpus", str(root / "corpus.json"), "-o", str(root / "result.json"),
            ]
            process = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(json.loads(process.stdout), report)
            self.assertEqual(json.loads((root / "result.json").read_text()), report)
            self.assertEqual(json.loads((root / "corpus.json").read_text()), manifest)
            corpus_bytes = (root / "corpus.json").read_bytes()
            arrays["decode.step7.logits"][0, 0] = 1
            write_safetensors(out / "llama-test.safetensors", arrays)
            report, manifest = coordinate([fixture], out, fixture.parent)
            self.assertFalse(report["passed"])
            self.assertIsNone(manifest)
            process = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(process.returncode, 1, process.stderr)
            self.assertFalse(json.loads((root / "result.json").read_text())["passed"])
            self.assertEqual((root / "corpus.json").read_bytes(), corpus_bytes)


if __name__ == "__main__":
    unittest.main()
