import builtins
import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import gguf_writer as writer
import gguf_recipes
import generate_gguf
from numpy_reference import gguf


class GgufTests(unittest.TestCase):
    def test_literal_orientation_and_inverse_do_not_cancel(self):
        expected = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]
        self.assertEqual(writer.export_rows(list(range(16)), 1), expected)
        values = np.array(expected, dtype=np.float32).reshape(16, 1)
        metadata = {"general.architecture": "llama", "llama.attention.head_count": 1}
        restored = gguf.canonical_weights(metadata, {"blk.0.attn_q.weight": values})
        np.testing.assert_array_equal(restored["model.layers.0.self_attn.q_proj.weight"].ravel(), np.arange(16))
        metadata = {"general.architecture": "qwen3"}
        np.testing.assert_array_equal(gguf.canonical_weights(metadata, {"blk.0.attn_q.weight": values})[
            "model.layers.0.self_attn.q_proj.weight"], values)
        self.assertNotEqual(writer.export_rows(expected, 1), list(range(16)))

    def parse_raw_block(self, storage, raw):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "w.gguf"
            writer.write(path, {}, {"x.weight": ((1, 32), storage, raw)})
            return gguf.read(path)

    def test_hand_authored_blocks_pin_nibbles_rebasing_bias_and_rounding(self):
        raw = bytes(range(16))
        for storage, header, bias in (("q4_0", struct.pack("<e", .25), -2),
                                     ("q4_1", struct.pack("<ee", .25, -.75), -.75)):
            _, _, _, arrays, decoded = self.parse_raw_block(storage, header + raw)
            self.assertEqual(int(arrays["x.weight"][0, 0]), 0x76543210)
            self.assertEqual(int(arrays["x.weight"][0, 1]), 0xFEDCBA98)
            self.assertEqual(int(arrays["x.weight"][0, 2]), 0)
            self.assertEqual(float(arrays["x.biases"][0, 0]), bias)
            expected = (np.array(list(range(16)) + [0] * 16, dtype=np.float32) * .25 + bias).astype(np.float16)
            np.testing.assert_array_equal(decoded["x.weight"].ravel(), expected)
        signed = [-128, -1, 0, 127] * 8
        _, _, _, arrays, decoded = self.parse_raw_block("q8_0", struct.pack("<e32b", .125, *signed))
        self.assertEqual(int(arrays["x.weight"][0, 0]), 0xFF807F00)
        self.assertEqual(float(arrays["x.biases"][0, 0]), -16)
        np.testing.assert_array_equal(decoded["x.weight"].ravel(), np.array(signed) * .125)

    def test_non_square_shapes_and_deterministic_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            a, b = (Path(temporary) / name for name in ("a.gguf", "b.gguf"))
            tensors = {"z.weight": ((3, 64), "f16", writer.encode_tensor(range(192), (3, 64), "f16")),
                       "a.weight": ((2, 3), "f32", writer.encode_tensor(range(6), (2, 3), "f32"))}
            writer.write(a, {"test": (8, "雪")}, tensors)
            writer.write(b, {"test": (8, "雪")}, dict(reversed(list(tensors.items()))))
            self.assertEqual(a.read_bytes(), b.read_bytes())
            metadata, _, descriptors, arrays, _ = gguf.read(a)
            self.assertEqual(metadata["test"], "雪")
            self.assertEqual(descriptors["z.weight"][0], (3, 64))
            np.testing.assert_array_equal(arrays["a.weight"], np.arange(6).reshape(2, 3))
            for length in (0, 4, 23, len(a.read_bytes()) - 1):
                b.write_bytes(a.read_bytes()[:length])
                with self.assertRaises((ValueError, struct.error)):
                    gguf.read(b)

    def test_affine_half_rounding_and_exact_raw_block_triples(self):
        q4 = bytes.fromhex("66aa997788ee8888") * 2
        cases = (
            ("q4_0", bytes.fromhex("dea8") + q4, [0x88E879A6] * 4, 0x34DE,
             [0x2CDE, 0xACE0, 0xA8E0, 0x28E0, 0, 0xB34C, 0, 0]),
            ("q4_1", bytes.fromhex("dea8e62f") + q4, [0x88E879A6] * 4, 0x2FE6,
             [0xAEB4, 0xB41C, 0xB301, 0xB091, 0xB1C9, 0xB68A, 0xB1C9, 0xB1C9]),
            ("q8_0", bytes.fromhex("dea8") + bytes.fromhex("8081ff0001027e7f") * 4,
             [0x807F0100, 0xFFFE8281] * 4, 0x44DE,
             [0x44DE, 0x44D4, 0x2900, 0, 0xA900, 0xACC0, 0xC4CA, 0xC4D4]),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "block.gguf"
            for storage, raw, words, bias, rounded in cases:
                with self.subTest(storage=storage):
                    expected = {
                        "x.weight": np.array([words], dtype=np.uint32),
                        "x.scales": np.array([[0xA8DE]], dtype=np.uint16).view(np.float16),
                        "x.biases": np.array([[bias]], dtype=np.uint16).view(np.float16),
                    }
                    writer.write(path, {}, {"x.weight": ((1, 32), storage, raw)})
                    generate_gguf.exact_conversion(path, expected)
                    decoded = gguf.read(path)[4]["x.weight"]
                    self.assertEqual(decoded.dtype, np.float32)
                    np.testing.assert_array_equal(decoded.astype(np.float16).view(np.uint16),
                                                  np.array([rounded * 4], dtype=np.uint16))
                    bits = 8 if storage == "q8_0" else 4
                    for lane in range(32):
                        changed = {key: value.copy() for key, value in expected.items()}
                        changed["x.weight"].flat[lane // (32 // bits)] ^= np.uint32(1 << (bits * (lane % (32 // bits))))
                        with self.assertRaises(AssertionError):
                            generate_gguf.exact_conversion(path, changed)
                    for slot in ("x.scales", "x.biases"):
                        changed = {key: value.copy() for key, value in expected.items()}
                        changed[slot].view(np.uint16).flat[0] ^= np.uint16(1)
                        with self.assertRaises(AssertionError):
                            generate_gguf.exact_conversion(path, changed)
                    for replacement in (None, expected["x.scales"].astype(np.float32),
                                        expected["x.scales"].reshape(1)):
                        changed = expected.copy()
                        if replacement is None:
                            del changed["x.scales"]
                        else:
                            changed["x.scales"] = replacement
                        with self.assertRaises(AssertionError):
                            generate_gguf.exact_conversion(path, changed)

    def test_exact_conversion_preserves_zero_sign_bits(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "zero.gguf"
            writer.write(path, {}, {"x.weight": ((1, 32), "q4_0", bytes.fromhex("0080") + bytes(16))})
            expected = {"x.weight": np.zeros((1, 4), dtype=np.uint32),
                        "x.scales": np.full((1, 1), -0.0, dtype=np.float16),
                        "x.biases": np.zeros((1, 1), dtype=np.float16)}
            generate_gguf.exact_conversion(path, expected)
            expected["x.scales"][0, 0] = 0.0
            with self.assertRaises(AssertionError):
                generate_gguf.exact_conversion(path, expected)

    def test_writer_rejects_invalid_blocks(self):
        for shape in ((32,), (2, 17)):
            with self.assertRaises(ValueError):
                writer.encode_tensor([1] * np.prod(shape), shape, "q4_0")

    def test_malformed_recipes_preserve_unrelated_payloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, destination = (Path(temporary) / name for name in ("source.gguf", "mutant.gguf"))
            tensors = {"x.weight": ((2, 32), "q8_0", writer.encode_tensor(range(64), (2, 32), "q8_0")),
                       "norm.weight": ((32,), "f32", writer.encode_tensor(range(32), (32,), "f32"))}
            writer.write(source, {"a": (4, 2), "b": (8, "test")}, tensors)
            gguf_recipes.materialize(source, destination, {"operation": "set_metadata", "key": "a", "type": 9, "value": [4, [2]]})
            before, after = gguf.read(source), gguf.read(destination)
            self.assertEqual(after[0]["a"], [2])
            for key in before[3]:
                np.testing.assert_array_equal(before[3][key], after[3][key])
            gguf_recipes.materialize(source, destination, {"operation": "remove_tensor", "key": "norm.weight"})
            self.assertNotIn("norm.weight", gguf.read(destination)[3])
            gguf_recipes.materialize(source, destination, {"operation": "truncate_rows", "key": "x.weight", "rows": 1})
            self.assertEqual(gguf.read(destination)[2]["x.weight"][0], (1, 32))

    def test_numpy_reader_has_no_oracle_or_runtime_imports(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.startswith(("mlx", "gguf_writer", "gguf_bridge")):
                raise AssertionError(name)
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", guarded):
            spec = importlib.util.spec_from_file_location("isolated_gguf", gguf.__file__)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

    def test_save_checker_mutation_kills_are_pure(self):
        path = Path(__file__).parents[1] / "qualify_gguf_save.py"
        spec = importlib.util.spec_from_file_location("save_checker", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        expected = module.expectations()
        module.compare(expected, expected)
        self.assertEqual(module.qualify_mutations(expected), {key: "killed" for key in ("value", "dtype", "metadata_kind")})
        for dtype in ("f16", "f32"):
            changed = module.expectations()
            changed[0]["tensor." + dtype][0, 2] = -0.0
            with self.assertRaisesRegex(AssertionError, "array bits"):
                module.compare(expected, changed)

    def test_fixed_tolerance_rejects_mapping_values(self):
        reference = {"prefill.full.logits": np.array([0, 1], dtype=np.float32)}
        within = {"prefill.full.logits": np.array([.004, 1.009], dtype=np.float32)}
        outside = {"prefill.full.logits": np.array([.006, 1.02], dtype=np.float32)}
        self.assertTrue(generate_gguf.compare(reference, within, .005, .005)["prefill.full.logits"]["passed"])
        self.assertFalse(generate_gguf.compare(reference, outside, .005, .005)["prefill.full.logits"]["passed"])

    def test_affine_calibration_and_error_provenance(self):
        self.assertEqual(generate_gguf.TOLERANCE_POLICIES["f32"], ("f32-v1", 2e-4, 2e-4))
        self.assertEqual(generate_gguf.TOLERANCE_POLICIES["f16"], ("gguf-f16-v1", 5e-3, 5e-3))
        for storage in ("q4_0", "q4_1", "q8_0"):
            self.assertEqual(generate_gguf.TOLERANCE_POLICIES[storage], ("gguf-affine-v1", 2e-2, 2e-3))
        reference = {"logits": np.array([0., 1., 100.])}
        actual = {"logits": np.array([.0093, 1.01, 100.21])}
        report = generate_gguf.compare(reference, actual, .02, .002)["logits"]
        self.assertTrue(report["passed"])
        self.assertAlmostEqual(report["max_abs"], .21)
        self.assertAlmostEqual(report["rel_to_peak"], .0021)
        self.assertEqual(report["max_rel"], .0093 / np.finfo(np.float32).tiny)
        self.assertAlmostEqual(report["margin"], .21 / .22)
        actual["logits"][-1] = 100.23
        self.assertFalse(generate_gguf.compare(reference, actual, .02, .002)["logits"]["passed"])

    def test_reference_ids_are_exact_even_when_tensors_agree(self):
        arrays = {"logits": np.array([1.])}
        agreement = generate_gguf.compare(arrays, arrays, .02, .002)
        generate_gguf.require_reference_agreement(agreement, [0, 36], [0, 36], "affine")
        with self.assertRaisesRegex(RuntimeError, "greedy IDs differ"):
            generate_gguf.require_reference_agreement(agreement, [0, 36], [0, 35], "affine")
        outside = generate_gguf.compare(arrays, {"logits": np.array([2.])}, .02, .002)
        with self.assertRaisesRegex(RuntimeError, "tensors disagree"):
            generate_gguf.require_reference_agreement(outside, [0, 36], [0, 36], "affine")

    def test_affine_mutation_requires_ten_times_the_combined_bound(self):
        arrays = {"cache.keys": np.array([100.]), "logits": np.array([0.])}
        changed = {"cache.keys": np.array([102.19]), "logits": np.array([2.])}
        agreement = generate_gguf.compare(arrays, changed, .02, .002)
        with self.assertRaisesRegex(RuntimeError, "weak affine mutation.*< 10x"):
            generate_gguf.qualify_mutation(agreement, "swap_qk_norm", "affine", True)
        report = generate_gguf.qualify_mutation(agreement, "swap_gate_up", "affine", True)
        self.assertEqual((report["witness"], report["margin"]), ("logits", 100.))
        changed["cache.keys"][0] = 102.2
        agreement = generate_gguf.compare(arrays, changed, .02, .002)
        report = generate_gguf.qualify_mutation(agreement, "swap_qk_norm", "affine", True)
        self.assertEqual(report["class"], "cache_value")
        self.assertAlmostEqual(report["margin"], 10.)
        changed["cache.keys"][0] = 101.
        agreement = generate_gguf.compare(arrays, changed, .02, .002)
        generate_gguf.qualify_mutation(agreement, "swap_qk_norm", "f16", False)

    def test_degenerate_affine_witnesses_are_refused(self):
        arrays = {"x.weight": np.ones((2, 8), dtype=np.uint32),
                  "x.scales": np.ones((2, 1), dtype=np.float16),
                  "x.biases": np.zeros((2, 1), dtype=np.float16)}
        with self.assertRaisesRegex(RuntimeError, "degenerate"):
            generate_gguf.qualify_converted_mutations(arrays)
        arrays["x.scales"][1] = 2
        arrays["x.biases"][1] = -1
        self.assertEqual(len(generate_gguf.qualify_converted_mutations(arrays)), 5)


if __name__ == "__main__":
    unittest.main()
