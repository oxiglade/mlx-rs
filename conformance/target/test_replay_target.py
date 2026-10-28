import copy
import json
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import replay_target


class ReplayComparisonTests(unittest.TestCase):
    def test_float_policy_classifies_within_tolerance_as_identical(self):
        expected = replay_target.TensorValue("F32", [2], np.array([1.0, 2.0], dtype=np.float32))
        actual = replay_target.TensorValue(
            "F32", [2], np.array([1.0 + 5.0e-7, 2.0], dtype=np.float32)
        )

        comparison = replay_target.compare_outputs(
            [expected],
            [actual],
            {"kind": "float", "atol": 1.0e-6, "rtol": 0.0, "nan_equal": True},
        )

        self.assertEqual(comparison["verdict"], "identical")

    def test_float_policy_reports_value_change_and_max_error(self):
        expected = replay_target.TensorValue("F32", [2], np.array([1.0, 2.0], dtype=np.float32))
        actual = replay_target.TensorValue("F32", [2], np.array([1.0, 2.25], dtype=np.float32))

        comparison = replay_target.compare_outputs(
            [expected],
            [actual],
            {"kind": "float", "atol": 1.0e-6, "rtol": 1.0e-6, "nan_equal": True},
        )

        self.assertEqual(comparison, {"verdict": "value_changed", "max_error": 0.25})

    def test_value_change_reports_max_error_across_all_outputs(self):
        expected = [
            replay_target.TensorValue("F32", [1], np.array([0.0], dtype=np.float32)),
            replay_target.TensorValue("F32", [1], np.array([0.0], dtype=np.float32)),
        ]
        actual = [
            replay_target.TensorValue("F32", [1], np.array([0.25], dtype=np.float32)),
            replay_target.TensorValue("F32", [1], np.array([4.0], dtype=np.float32)),
        ]

        comparison = replay_target.compare_outputs(
            expected,
            actual,
            {"kind": "float", "atol": 0.0, "rtol": 0.0, "nan_equal": True},
        )

        self.assertEqual(comparison, {"verdict": "value_changed", "max_error": 4.0})

    def test_dtype_change_wins_over_equal_numeric_values(self):
        expected = replay_target.TensorValue("F32", [2], np.array([1.0, 2.0], dtype=np.float32))
        actual = replay_target.TensorValue("F16", [2], np.array([1.0, 2.0], dtype=np.float16))

        comparison = replay_target.compare_outputs(
            [expected], [actual], {"kind": "exact_numeric"}
        )

        self.assertEqual(comparison["verdict"], "dtype_or_shape_changed")
        self.assertEqual(comparison["output"], 0)

    def test_bfloat16_policy_compares_decoded_values(self):
        expected = replay_target.TensorValue(
            "BF16", [1], np.array([0x3F80], dtype=np.uint16)
        )
        actual = replay_target.TensorValue(
            "BF16", [1], np.array([0x3F81], dtype=np.uint16)
        )

        comparison = replay_target.compare_outputs(
            [expected],
            [actual],
            {"kind": "float", "atol": 0.008, "rtol": 0.0, "nan_equal": True},
        )

        self.assertEqual(comparison["verdict"], "identical")

    def test_non_finite_change_has_json_safe_max_error(self):
        expected = replay_target.TensorValue(
            "F32", [1], np.array([np.nan], dtype=np.float32)
        )
        actual = replay_target.TensorValue(
            "F32", [1], np.array([1.0], dtype=np.float32)
        )

        comparison = replay_target.compare_outputs(
            [expected],
            [actual],
            {"kind": "float", "atol": 0.0, "rtol": 0.0, "nan_equal": True},
        )

        self.assertEqual(comparison["verdict"], "value_changed")
        self.assertIsNone(comparison["max_error"])
        json.dumps(comparison, allow_nan=False)

    def test_error_inversion_is_error_behavior_change(self):
        comparison = replay_target.compare_behavior(
            {"status": "error", "exception": {"module": "builtins", "type": "ValueError"}},
            {"status": "success", "outputs": []},
            None,
        )

        self.assertEqual(comparison["verdict"], "error_behavior_changed")


class ReplayIntegrityTests(unittest.TestCase):
    def test_payload_hash_detects_tampering(self):
        payload = {"handshake": {"mlx": "0.32.2"}, "suites": [{"id": "arithmetic"}]}
        digest = replay_target.payload_sha256(payload)
        tampered = copy.deepcopy(payload)
        tampered["suites"][0]["id"] = "shapes"

        self.assertNotEqual(replay_target.payload_sha256(tampered), digest)

    def test_source_artifacts_include_gguf_inputs(self):
        corpus = json.loads((replay_target.CONFORMANCE / "corpus.json").read_text())
        sources = replay_target.source_artifacts(corpus)
        for path, fixture in corpus["gguf_fixtures"].items():
            self.assertEqual(sources.get(f"conformance/{path}"), fixture["sha256"])

    def test_aborted_suite_preserves_completed_cases_and_fills_remaining(self):
        completed = [{"id": "errors.001", "verdict": "identical"}]

        cases = replay_target.complete_aborted_suite(
            ["errors.001", "errors.002", "errors.003"], completed, -signal.SIGABRT
        )

        self.assertEqual(cases[0], completed[0])
        self.assertEqual(cases[1]["id"], "errors.002")
        self.assertEqual(cases[1]["verdict"], "recipe_failed")
        self.assertEqual(cases[1]["process"]["signal"], signal.SIGABRT)
        self.assertEqual(cases[2]["verdict"], "recipe_failed")

    def test_expected_abort_probe_classifies_abort_as_identical(self):
        result = replay_target.classify_expected_abort(-signal.SIGABRT)

        self.assertEqual(result["verdict"], "identical")
        self.assertEqual(result["target_behavior"], "process_abort")

    def test_expected_abort_probe_rejects_other_signals(self):
        result = replay_target.classify_expected_abort(-signal.SIGSEGV)

        self.assertEqual(result["verdict"], "recipe_failed")
        self.assertEqual(result["process"]["signal"], signal.SIGSEGV)

    def test_existing_baseline_requires_explicit_update(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            original = (
                replay_target.TARGET,
                replay_target.REPORT,
                replay_target.EXPECTATIONS,
            )
            replay_target.TARGET = target
            replay_target.REPORT = target / "replay-report.json"
            replay_target.EXPECTATIONS = target / "target-expectations"
            payload = {"handshake": {}, "suites": []}
            digest = replay_target.payload_sha256({"payload": payload, "shards": {}})
            try:
                self.assertEqual(
                    replay_target.publish_report(payload, {}, digest, digest, False),
                    "created",
                )
                replay_target.REPORT.write_text("tampered")

                with self.assertRaises(SystemExit):
                    replay_target.publish_report(payload, {}, digest, digest, False)

                self.assertEqual(replay_target.REPORT.read_text(), "tampered")
                self.assertEqual(
                    replay_target.publish_report(payload, {}, digest, digest, True),
                    "updated",
                )
            finally:
                (
                    replay_target.TARGET,
                    replay_target.REPORT,
                    replay_target.EXPECTATIONS,
                ) = original


class ReplayWorkerTests(unittest.TestCase):
    def replay(self, kind, suite=None):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "suite.json"
            with patch.object(replay_target, "VENV", Path(sys.prefix)):
                replay_target.run_worker(kind, suite, output)
            return json.loads(output.read_text())["cases"]

    def test_math_replays_empty_output_cases(self):
        cases = self.replay("ops", "math")
        suite = json.loads((replay_target.CONFORMANCE / "suites/math.json").read_text())
        self.assertEqual([case["id"] for case in cases], [case["id"] for case in suite["cases"]])
        self.assertTrue(all(case["verdict"] == "identical" for case in cases))
        self.assertTrue(any(case["target"].get("outputs") == [] for case in cases))

    def test_gguf_replays_every_recipe(self):
        cases = self.replay("gguf", "gguf")
        suite = json.loads((replay_target.CONFORMANCE / "suites/gguf.json").read_text())
        self.assertEqual([case["id"] for case in cases], [case["id"] for case in suite["cases"]])
        self.assertEqual({case["verdict"] for case in cases}, {"identical"})

    def test_gguf_load_failure_reports_error_behavior_change(self):
        import mlx.core as mx

        load = mx.load

        def changed_load(path, **kwargs):
            if Path(path).name == "basic.gguf":
                raise RuntimeError("load regression")
            return load(path, **kwargs)

        with patch.object(mx, "load", side_effect=changed_load):
            cases = self.replay("gguf", "gguf")
        changed = [case for case in cases if case["id"] in {"gguf.001", "gguf.018", "gguf.019"}]
        self.assertEqual({case["verdict"] for case in changed}, {"error_behavior_changed"})

    def test_singular_inverse_reaches_cpu_evaluation(self):
        case, = self.replay("probe_singular_inv")
        self.assertEqual(case["target"], {
            "status": "error",
            "stage": "eval",
            "exception": {"module": "builtins", "type": "RuntimeError"},
        })
        self.assertEqual(case["old_behavior"], "process_abort")
        self.assertEqual(case["verdict"], "error_behavior_changed")


if __name__ == "__main__":
    unittest.main()
