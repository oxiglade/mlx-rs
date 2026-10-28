import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "conformance/target"))
import replay_target


class ReplayCoverageTests(unittest.TestCase):
    def test_suite_coverage_matches_corpus(self):
        corpus = json.loads((replay_target.CONFORMANCE / "corpus.json").read_text())
        expected = {Path(path).stem for path in corpus["suites"]}
        expected.update({"state", "transforms", "probe_oob_take", "probe_singular_inv"})
        specs = replay_target.suite_specs(corpus)
        self.assertEqual({suite for suite, _, _ in specs}, expected)
        self.assertEqual(len(specs), len(expected))
        for _, kind, _ in specs:
            self.assertIn(kind, replay_target.WORKERS)
        self.assertIn(("gguf", "gguf", "gguf"), specs)


if __name__ == "__main__":
    unittest.main()
