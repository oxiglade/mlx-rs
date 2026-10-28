#!/usr/bin/env python3
"""Download the pinned bump fixtures and generate the independent runner Python oracle."""

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "conformance/mlx-lm"))
from real_checkpoints import read_manifest, sha256, verify_entry


def provision(cache, output, python):
    started = time.monotonic()
    oracle = ROOT / "conformance/mlx-lm/bump-oracle.json"
    report = json.loads(oracle.read_text())
    downloaded = 0
    for name, entry in report["contract"]["entries"].items():
        directory = cache / name / entry["revision"]
        directory.mkdir(parents=True, exist_ok=True)
        entry["local_directory"] = str(directory)
        for filename, digest in entry["files"].items():
            path = directory / filename
            if path.exists() and sha256(path) == digest:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".partial")
            url = f'https://huggingface.co/{entry["repository"]}/resolve/{entry["revision"]}/{filename}'
            subprocess.run(["curl", "--fail", "--location", "--retry", "5", "--silent",
                            "--show-error", "--output", str(temporary), url], check=True)
            if sha256(temporary) != digest:
                raise ValueError(f"download hash mismatch: {name}/{filename}")
            downloaded += temporary.stat().st_size
            temporary.replace(path)
        verify_entry(entry)
        print(f'{name}: {entry["bytes"]} bytes, revision {entry["revision"]}', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "python-real-report.json"

    manifest = copy.deepcopy(report["contract"])
    manifest["expected_report"] = {"path": str(report_path), "sha256": None}
    manifest_path = output / "real-checkpoints.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    read_manifest(manifest_path)
    subprocess.run([str(python), "-B", str(ROOT / "conformance/mlx-lm/live_oracle.py"),
                    "--manifest", str(manifest_path), "--output", str(report_path)], check=True)
    manifest["expected_report"]["sha256"] = sha256(report_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    receipt = {"downloaded_bytes": downloaded, "elapsed_seconds": time.monotonic() - started,
               "oracle_sha256": sha256(oracle), "entries": manifest["entries"]}
    (output / "checkpoint-download.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "entries"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()
    provision(args.cache.resolve(), args.output.resolve(), args.python.absolute())
