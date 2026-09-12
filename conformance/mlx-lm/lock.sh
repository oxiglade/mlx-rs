#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
oracle_python=${PYTHON:-python3.12}
"$oracle_python" -c 'import platform, sys; assert sys.version_info[:3] == (3, 12, 14), "requires Python 3.12.14"; assert platform.machine() == "arm64", "requires arm64"; assert sys.platform == "darwin", "requires macOS"'
wheel_dir=$(mktemp -d "${TMPDIR:-/tmp}/mlx-lm-lock.XXXXXX")
trap 'rm -rf "$wheel_dir"' EXIT HUP INT TERM

"$oracle_python" -m pip download --only-binary=:all: --no-deps \
    --dest "$wheel_dir" -r "$root/requirements.in"
"$oracle_python" -m pip install --dry-run --ignore-installed --no-index \
    --find-links "$wheel_dir" -r "$root/requirements.in"
"$oracle_python" - "$root" "$wheel_dir" <<'PY'
import email
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

root, wheel_dir = map(Path, sys.argv[1:])
normalize = lambda name: re.sub(r"[-_.]+", "-", name).lower()
pins = {}
for line in (root / "requirements.in").read_text().splitlines():
    if line and not line.startswith("#"):
        name, version = line.split("==")
        pins[normalize(name)] = version
entries = {}
for wheel in sorted(wheel_dir.glob("*.whl")):
    with zipfile.ZipFile(wheel) as archive:
        metadata_files = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        if len(metadata_files) != 1:
            raise SystemExit(f"ambiguous wheel metadata: {wheel.name}")
        metadata = email.message_from_bytes(archive.read(metadata_files[0]))
    name, version = normalize(metadata["Name"]), metadata["Version"]
    if name in entries or pins.get(name) != version:
        raise SystemExit(f"unexpected or duplicate wheel: {wheel.name}")
    output = subprocess.check_output(
        [sys.executable, "-m", "pip", "hash", "--algorithm", "sha256", str(wheel)],
        text=True,
    )
    hashes = re.findall(r"--hash=sha256:[0-9a-f]{64}", output)
    if len(hashes) != 1:
        raise SystemExit(f"expected one wheel hash: {wheel.name}")
    entries[name] = f"{name}=={version} \\\n    {hashes[0]}\n"
if entries.keys() != pins.keys():
    raise SystemExit("download did not produce every pinned wheel")
temporary = wheel_dir / "requirements.lock"
temporary.write_text("".join(entries[name] for name in sorted(entries)))
# The temporary directory may be on a different filesystem from the checkout.
destination = root / "requirements.lock"
staged = destination.with_suffix(".lock.tmp")
try:
    staged.write_bytes(temporary.read_bytes())
    os.replace(staged, destination)
finally:
    staged.unlink(missing_ok=True)
PY
