import json
import shutil
import tempfile
import unittest
from pathlib import Path

import generate_mlx_lm as generator
from manifest import ROOT, write_json


class GeneratorContractTests(unittest.TestCase):
    def test_preservation_checks_every_existing_fixture(self):
        for name in generator.VARIANTS:
            fixture = ROOT / "fixtures" / name
            generator.check_preserved_fixture(fixture, fixture)

    def test_preservation_accepts_only_the_reviewed_bos_encoding_change(self):
        original = ROOT / "fixtures/llama-base"
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "llama-base"
            shutil.copytree(original, target)
            path = target / "expectations.json"
            data = json.loads(path.read_text())
            ids = data["tokenizer"]["encodings"]["special_with_defaults"]
            data["tokenizer"]["encodings"]["special_with_defaults"] = ids if ids[:1] == [4] else [4, *ids]
            write_json(path, data)
            generator.check_preserved_fixture(original, target)
            data["decode"]["greedy_ids"][0] += 1
            write_json(path, data)
            with self.assertRaisesRegex(RuntimeError, "decode.greedy_ids"):
                generator.check_preserved_fixture(original, target)

    def test_tree_hash_covers_text_artifact_bytes_and_path(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a, b = Path(first), Path(second)
            for root in (a, b):
                (root / "text_cases.json").write_text('{"text":"é"}')
            self.assertEqual(generator.tree_hash(a), generator.tree_hash(b))
            (b / "text_cases.json").write_text('{"text":"é"}')
            self.assertNotEqual(generator.tree_hash(a), generator.tree_hash(b))
            (b / "text_cases.json").write_bytes((a / "text_cases.json").read_bytes())
            (b / "text_cases.json").rename(b / "other.json")
            self.assertNotEqual(generator.tree_hash(a), generator.tree_hash(b))


if __name__ == "__main__":
    unittest.main()
