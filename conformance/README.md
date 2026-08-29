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

The generator requires Python 3.12.14 on arm64, `mlx==0.30.6`, `mlx-metal==0.30.6`,
`numpy==2.2.6`, and an MLX 0.30.6 runtime. It exits before generation when the reference
environment does not match. The lock file pins every wheel with a SHA-256 hash and must be
installed with `--require-hashes`.

Generation builds two independent temporary trees and compares their hashes. It replaces the
committed catalog, suite files, qualification file, and fixture shards only when both trees are
identical. Generated data has no timestamps or local paths. Rust tests only read committed
fixtures; there is no bless or update-goldens path.

The generator encodes fixture shards with its in-repo safetensors writer, so MLX I/O is not part
of either fixture path. The Rust runner cross-validates that encoding through its safetensors crate.

MLX 0.30.6 validates shape, broadcast, reshape, and axis errors eagerly, so no catchable
`eval_only` operation error is available at this pin. Inverting a singular matrix does fail during
evaluation, but its C++ exception aborts the Python process; an `eval_only` case returns with the
next mlx-c version bump.

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
`cargo run -p xtask -- verify-oracle-boundary` for the working tree or add `--base <ref>` to check
each non-merge commit in a range. Skipped merge hashes appear in `skipped_merge_commits`; a
deliberately staged commit that must cross both sides uses an `oracle-change:` commit-subject
prefix, which is reported as a loud override and requires focused oracle review rather than
silently weakening the boundary.

Tolerance policies are named in `corpus.json`; there is no default policy. Changing or widening a
policy is an oracle-only change and must not accompany an implementation change.
