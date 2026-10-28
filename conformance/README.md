# Committed-golden CPU ops conformance

This directory contains the CPU-only committed corpus. Optimizer qualification and the live
Python worker are separate later milestones.

## Regeneration

Use the repository's devenv environment on an arm64 Mac, then run:

```sh
python3.12 -m venv conformance/.venv
conformance/.venv/bin/pip install --require-hashes -r conformance/requirements.lock
conformance/.venv/bin/python conformance/generate.py
cargo test -p mlx-tests --test conformance -- --test-threads=1
```

The generator requires Python 3.12.14 on arm64, `mlx==0.32.2`, `mlx-metal==0.32.2`,
`numpy==2.2.6`, and an MLX 0.32.2 runtime. It exits before generation when the reference
environment does not match. The lock file pins every wheel with a SHA-256 hash and must be
installed with `--require-hashes`.

On 2026-08-30 the committed corpus became 0.32.2-canonical; `target/` remains the historical
0.30.6-to-0.32.2 replay baseline.

Generation builds two independent temporary trees and compares their hashes. It replaces the
committed catalog, suite files, qualification file, and fixture shards only when both trees are
identical. Generated data has no timestamps or local paths. Rust tests only read committed
fixtures; there is no bless or update-goldens path.

The generator encodes fixture shards with its in-repo safetensors writer, so MLX I/O is not part
of either fixture path. The Rust runner cross-validates that encoding through its safetensors crate.

The historical singular-inverse probe records a process abort on MLX 0.30.6. On CPU,
MLX 0.32.2 raises a catchable `RuntimeError` during evaluation. The replay preserves that
`error_behavior_changed` result against the historical reference. Running this probe on the
default GPU instead would only observe an unsupported-operation `ValueError` at invocation.

## Exact-target replay

Replay discovers every suite in `corpus.json`, then runs the state trajectories, transforms,
and two historical probes in separate subprocesses. It calls the generator's ops recipes with
their recorded execution targets, including recipes with no outputs. The GGUF worker loads the
committed GGUF inputs, shares the generator's result encoding, and exercises metadata, dequantization,
save rejection and save/load recipes through Python MLX. Duplicate-key insertion errors in
`gguf.020` are generator literal contracts, recorded separately from Python execution evidence:
Python dictionaries cannot exercise the Rust insertion API.

To keep the pinned environment and both replay runs outside the checkout:

```sh
export REPLAY_ROOT="$HOME/hub/scratch/replay-runner"
mkdir -p "$REPLAY_ROOT"
python3.12 -m venv "$REPLAY_ROOT/venv"
"$REPLAY_ROOT/venv/bin/python" -m pip install --require-hashes -r conformance/requirements.lock
TMPDIR="$REPLAY_ROOT" "$REPLAY_ROOT/venv/bin/python" -B conformance/target/replay_target.py \
  --venv "$REPLAY_ROOT/venv" --output-dir "$REPLAY_ROOT/reports/replay"
TMPDIR="$REPLAY_ROOT" "$REPLAY_ROOT/venv/bin/python" -B -m unittest discover \
  -s conformance/target -p 'test_*.py'
python3 -B -m unittest discover -s xtask -p 'test_*.py'
```

The report hashes both complete runs and rejects nondeterminism. Its case verdicts compare against
the committed corpus using the existing tolerance policies; the historical probes retain their
0.30.6 reference environment. `--output-dir` leaves the historical committed replay reports intact.
Replacing an existing report requires `--update`. The replay tests include pinned-environment
worker regressions. The cheap suite-coverage check runs in the existing xtask Python test
discovery, including the bump workflow, without MLX or NumPy; it adds no CI gate.

## Oracle separation

`generate.py` imports Python MLX and NumPy only. It does not import, build, invoke, or inspect
mlx-rs or Cargo artifacts. Python MLX on CPU is the oracle. NumPy provenance is recorded only
when it agrees with MLX on output count, dtype, shape, and values.

After the bootstrap commit, an mlx-rs implementation change may not share a commit with recipes,
fixture bytes, tolerance policies, comparator logic, or qualification mutations. Those oracle
changes require a separate commit and review.

The enforced boundary is listed in `protected-paths.json`: all of `conformance/` and the Rust
oracle, entry-point, and checker files are protected, while `mlx-rs/src/` and
`mlx-tests/tests/conformance/adapters.rs` are implementation-owned. Run
`cargo xtask verify-oracle-boundary` for the working tree or add `--base <ref>` to check
each non-merge commit in a range. Skipped merge hashes appear in `skipped_merge_commits`; a
deliberately staged commit that must cross both sides uses an `oracle-change:` commit-subject
prefix, which is reported as a loud override and requires focused oracle review rather than
silently weakening the boundary.

Tolerance policies are named in `corpus.json`; there is no default policy. Changing or widening a
policy is an oracle-only change and must not accompany an implementation change.
