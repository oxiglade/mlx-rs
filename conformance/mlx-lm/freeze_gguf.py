#!/usr/bin/env python3
"""Freeze only repeatable GGUF artifacts after independent reference qualification."""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from coordinate import coordinate, write_json
from numpy_reference.common import read_json, write_safetensors
from numpy_reference.run import reference_tensors

ROOT = Path(__file__).resolve().parent
NAMES = [f"gguf-{family}-{kind}" for family in ("llama", "qwen3") for kind in ("f32", "f16", "q4_0", "q4_1", "q8_0")]


def hashes(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", required=True, type=Path)
    parser.add_argument("--repeat", required=True, type=Path)
    args = parser.parse_args()
    for name in NAMES:
        first, second = args.generated / name, args.repeat / name
        if not first.is_dir() or not second.is_dir() or hashes(first) != hashes(second):
            raise ValueError(f"missing or nondeterministic fixture: {name}")
        doc = read_json(first / "expectations.json")
        if not doc["reference_qualified"] or not doc["mutations"]:
            raise ValueError(f"unqualified fixture: {name}")
        for source, digest in doc["provenance"]["sources"].items():
            if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != digest:
                raise ValueError(f"source changed after generation: {source}")
        for asset in doc["gguf"]["tokenizer_files"].values():
            canonical = ROOT / "fixtures" / doc["provenance"]["base"] / Path(asset["path"]).name
            if hashlib.sha256(canonical.read_bytes()).hexdigest() != asset["sha256"]:
                raise ValueError(f"tokenizer sidecar changed: {canonical}")
    with tempfile.TemporaryDirectory(prefix="gguf-freeze-") as temporary:
        stage = Path(temporary)
        fixtures, numpy_dir = stage / "fixtures", stage / "numpy"
        shutil.copytree(ROOT / "fixtures", fixtures)
        shutil.copytree(ROOT / "numpy_reference/out", numpy_dir)
        for name in NAMES:
            shutil.copytree(args.generated / name, fixtures / name, dirs_exist_ok=True)
            write_safetensors(numpy_dir / f"{name}.safetensors", reference_tensors(fixtures / name))
        report, corpus = coordinate(list(fixtures.iterdir()), numpy_dir, fixtures)
        write_json(stage / "coordination.json", report)
        if corpus is None:
            raise ValueError(json.dumps(report, indent=2))
        for name in NAMES:
            shutil.copytree(fixtures / name, ROOT / "fixtures" / name, dirs_exist_ok=True)
            shutil.copy2(numpy_dir / f"{name}.safetensors", ROOT / "numpy_reference/out" / f"{name}.safetensors")
        write_json(ROOT / "gguf-coordination.json", {name: report["fixtures"][name] for name in NAMES})
        corpus["gguf_qualification"] = {"status": "python_numpy_qualified", "fixtures": NAMES,
                                      "rust_cpu": "not_run", "rust_metal": "not_run"}
        write_json(ROOT / "corpus.json", corpus)
    manifest = read_json(ROOT / "manifest.json")
    manifest["gguf_oracle"] = {"status": "python_numpy_qualified", "fixtures": NAMES,
                               "provenance": read_json(ROOT / "fixtures" / NAMES[0] / "expectations.json")["provenance"]}
    write_json(ROOT / "manifest.json", manifest)
    ledger_path = ROOT.parents[1] / "ledger/mlx-lm-parity.json"
    ledger = read_json(ledger_path)
    ledger["gguf_qualification"]["tiny_fixtures"] = "python_numpy_qualified"
    ledger["gguf_qualification"]["python_numpy"] = "qualified; fixed f32-v1, gguf-f16-v1 and gguf-affine-v1 before Rust"
    ledger["gguf_qualification"]["core_container_conversion"] = "Converted slots match independent raw-block NumPy derivation bit for bit; NumPy dequantization rounds affine multiply and addition in F16"
    write_json(ledger_path, ledger)
    print("Python/NumPy GGUF corpus frozen; Rust CPU/Metal parity still requires the runtime implementation")


if __name__ == "__main__":
    main()
