#!/usr/bin/env python3
"""Generate runner expectations from the pinned Python runtime and frozen contract."""

import argparse
import copy
import json
import os
import platform
from pathlib import Path

from real_checkpoints import contract, generate_report, read_manifest, sha256, write_new

ROOT = Path(__file__).resolve().parent


def normalized_contract(contract):
    contract = copy.deepcopy(contract)
    for name, entry in contract["entries"].items():
        entry["local_directory"] = name
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.python_version() != "3.12.14":
        raise ValueError("requires uv-managed Python 3.12.14")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    committed = json.loads((ROOT / "bump-oracle.json").read_text())
    manifest = read_manifest(args.manifest)
    if normalized_contract(contract(manifest)) != committed["contract"]:
        raise ValueError("live oracle contract differs from committed contract")
    live = generate_report(manifest)
    write_new(args.output, live)
    receipt = {
        "python_version": platform.python_version(),
        "generator_sha256": sha256(__file__),
        "oracle_generator_sha256": sha256(ROOT / "real_checkpoints.py"),
        "requirements_sha256": sha256(ROOT / "requirements.lock"),
        "committed_sha256": sha256(ROOT / "bump-oracle.json"),
        "live_sha256": sha256(args.output),
    }
    write_new(args.output.with_name("live-oracle-generation.json"), receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    if any(not result["completed"] for devices in live["results"].values() for result in devices.values()):
        raise SystemExit("incomplete live Python oracle")


if __name__ == "__main__":
    main()
