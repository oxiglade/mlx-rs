#!/usr/bin/env python3
import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UPSTREAM_COMMIT = "ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd"
EXPECTED = {"mlx-lm": "0.31.3", "mlx": "0.32.2", "mlx-metal": "0.32.2", "numpy": "2.2.6"}


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def read_lock(path):
    entries = {}
    for line in path.read_text().replace("\\\n", " ").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"([\w.-]+)==([^\s]+)\s+--hash=sha256:([0-9a-f]{64})\s*", line)
        if match is None:
            raise SystemExit(f"expected a single hashed wheel pin: {line!r}")
        name, version, digest = match.groups()
        name = normalize(name)
        if name in entries:
            raise SystemExit(f"duplicate lock entry: {name}")
        entries[name] = (version, digest)
    return entries


def check_environment(lock_path=ROOT / "requirements.lock"):
    if sys.version_info[:3] != (3, 12, 14):
        raise SystemExit(f"requires Python 3.12.14, got {platform.python_version()}")
    if platform.machine() != "arm64" or sys.platform != "darwin":
        raise SystemExit(f"requires macOS arm64, got {sys.platform} {platform.machine()}")
    entries = read_lock(lock_path)
    pins = {}
    for line in (ROOT / "requirements.in").read_text().splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==")
            pins[normalize(name)] = version
    if {name: entry[0] for name, entry in entries.items()} != pins:
        raise SystemExit("requirements.lock does not match requirements.in")
    for name, version in EXPECTED.items():
        if pins.get(name) != version:
            raise SystemExit(f"requires {name}=={version}")
    for name, version in pins.items():
        installed = importlib.metadata.version(name)
        if installed != version:
            raise SystemExit(f"requires {name}=={version}, got {installed}")
    if pins["transformers"].split(".")[0] != "5":
        raise SystemExit("requires Transformers major version 5")
    return {
        "arch": platform.machine(),
        "python": platform.python_version(),
        "mlx_lm": pins["mlx-lm"],
        "mlx": pins["mlx"],
        "mlx-metal": pins["mlx-metal"],
        "numpy": pins["numpy"],
        "transformers": pins["transformers"],
        "transformers_major": 5,
        "upstream_commit": UPSTREAM_COMMIT,
        "mlx_lm_wheel_sha256": entries["mlx-lm"][1],
        "requirements_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "packages": pins,
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Record the pinned mlx-lm oracle environment.")
    parser.add_argument("--lock", type=Path, default=ROOT / "requirements.lock")
    parser.add_argument("-o", "--output", type=Path, default=ROOT / "manifest.json")
    args = parser.parse_args()
    write_json(args.output, check_environment(args.lock))


if __name__ == "__main__":
    main()
