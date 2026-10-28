import builtins
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import text_reference


class TextReferenceTests(unittest.TestCase):
    def test_eager_overlapping_stops_depend_on_chunking(self):
        whole = text_reference.filter_case(["abc", "b"], ["abc"])
        split = text_reference.filter_case(["abc", "b"], ["ab", "c"])
        self.assertEqual(whole["events"], [{"text": "", "held": "", "stopped": True}])
        self.assertEqual(split["events"], [{"text": "a", "held": "", "stopped": True}])
        self.assertIsNone(split["finish_flush"])

    def test_final_delta_matches_before_unmatched_suffix_release(self):
        matched = text_reference.filter_case(["END"], ["hi E"], "ND hidden")
        mismatch = text_reference.filter_case(["END"], ["hi E"], "x")
        held = text_reference.filter_case(["END"], ["hi EN"])
        self.assertEqual((matched["finish_reason"], matched["finish_flush"]), ("stop", ""))
        self.assertEqual((mismatch["finish_reason"], mismatch["finish_flush"]), ("length", "Ex"))
        self.assertEqual(held["finish_flush"], "EN")
        with self.assertRaisesRegex(ValueError, "EmptyStopString"):
            text_reference.filter_case([""], [])

    def test_no_normalization(self):
        self.assertFalse(text_reference.filter_case(["é"], ["e", "\u0301"])["stopped"])
        self.assertTrue(text_reference.filter_case(["e\u0301"], ["e", "\u0301"])["stopped"])

    def test_reference_runs_with_mlx_imports_forbidden(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name == "mlx" or name.startswith("mlx.") or name == "mlx_lm" or name.startswith("mlx_lm."):
                raise AssertionError(f"reference imported {name}")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", guarded):
            spec = importlib.util.spec_from_file_location("isolated_text_reference", Path(text_reference.__file__))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            first = module.reference()
            second = module.reference()
        self.assertEqual(first, second)
        self.assertEqual(first["decoders"]["byte_fallback_rewrite"]["expected_error"], "DecodeStreamError::InvalidPrefix")
        self.assertEqual(first["decoders"]["byte_level_incomplete_tail"]["finish_flush"], "�")


if __name__ == "__main__":
    unittest.main()
