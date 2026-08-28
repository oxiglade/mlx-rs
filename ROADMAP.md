# Verification roadmap

Catching up to upstream MLX is gated less by missing API surface than by the fact that a green
test run is not currently evidence. The work below builds the substrate that makes machine-
generated catch-up work trustworthy, then does the catch-up.

## Why not simply port MLX's tests

MLX has 892 Python test methods (~29k lines). Translating them into Rust produces a large amount
of code while leaving the two hard problems untouched: FFI ownership, and oracle independence.
Upstream's tests also lean on NumPy as the oracle (`assertCmpNumpy` in `python/tests/mlx_tests.py`)
with randomly generated inputs, so a literal port has nothing to compare against. Keep NumPy on the
oracle side instead — see "Conformance runner" below.

## Status

- [x] **Reproducible toolchain.** `devenv.nix` provides rust, cmake, ninja and libclang. Metal
      needs the system Xcode; see the `xcrun` shim comment in `devenv.nix` for why.
- [x] **Tranche 1 — six FFI defects fixed**, each with a regression test demonstrated to fail
      without its fix. Baseline moved 945 → 952 passing, 60 ignored.
- [x] **Tranche 2 — the leak/UAF gate.** `cargo run -p xtask -- verify-ffi`, plus the three
      carried-forward FFI items. Baseline moved 952 → 965 passing, 61 ignored.
- [ ] **Conformance runner.** Below.

## Known-weak coverage

Recorded because none of it shows up in a test count, and all of it survives adding more tests.

- `assert_array_eq!` (`mlx-rs/src/macros/assert.rs`) passes its single tolerance argument as *both*
  `rtol` and `atol`, and never compares dtype. `test_ada_delta` asserts a mean of `-0.348442` with
  tolerance `0.348442`, which admits roughly `[-0.818, +0.121]` — an optimizer that did nothing
  would pass. Fixing the macro is expected to turn several currently-green tests red; that is the
  point, and it must land as its own change so the failures are attributable.
- Several `compile`/optimizer tests run at learning rate `0.0`, removing the state transition they
  exist to exercise.
- 47 doctests use `rust,ignore` and never execute: 35 in `nn/activation.rs`, 6 on the `lib.rs`
  front page. Note that grepping for ```` ```ignore ```` misses these. Most appear to be a fence
  copied forward rather than a deliberate choice.
- `mlx-lm` has **zero executing tests**; all 6 are `#[ignore]` on missing model files. This is the
  newest code in the workspace.
- CI runs `cargo clippy -- -D warnings` without `--all-targets`, so lints in test code have never
  been gated. There are ~40 such warnings today.

## Tranche 2 — leak and use-after-free gate (done)

    cargo run -p xtask -- verify-ffi                 # JSON to stdout, progress to stderr
    cargo run -p xtask -- verify-ffi --guard-malloc  # slower use-after-free lane

**The gate is qualified — it has been made to fail on a real defect.** Deleting the body of
`StringMapIterator::drop` in `mlx-rs/src/utils/io.rs` reintroduces the tranche-1 leak, and the gate
reports for the `ffi_safety` binary:

    count 201, bytes 3232, baseline_subtracted true,
    regression_count 200, regression_bytes 3200,
    named_sites [{site: "<malloc in mlx_map_string_to_string_iterator_new>", count: 200, bytes: 3200}]

Restoring the guard returns it to zero regressions and `verdict: pass`. Re-run that procedure after
any change to the parser or baseline logic; a gate that has never failed is not known to work.

Clean-tree expectation: `verdict: pass`, exit 0, leak statuses `{pass: 18, not_applicable: 2}`,
every binary `test: passed`.

`leaks --atExit` is the mechanism: it sees FFI-level leaks, names the responsible C function, and
scales linearly with call count.

- Every test binary reports a constant `1 leak for 32 total leaked bytes`
  (`ROOT LEAK: <NSArray>`, from Objective-C init). Gate against that baseline.
- **`leaks` exits non-zero whenever any leak exists, including the baseline.** The gate must
  compare counts, not trust the exit code.
- Pair with `libgmalloc` (`DYLD_INSERT_LIBRARIES`) for use-after-free and double-free. This covers
  what ASan would give us: LeakSanitizer is unsupported on macOS, and `-Zsanitizer` needs nightly
  while CI pins 1.85/stable.
- Emit structured JSON so the verdict is machine-readable rather than a diff to read.
- Proc-macro crates are excluded by target kind and reported `not_applicable`. Their test binaries
  link the compiler host's dynamic libstd and cannot be spawned standalone (`no LC_RPATH's found`),
  so `leaks` never gets a process; they also contain zero `mlx_sys` references, so there is nothing
  for this gate to check. Filter on kind, never on crate name.
- Discover binaries via `cargo test --no-run --message-format=json`, never by globbing
  `target/debug/deps`. That directory accumulates stale binaries from earlier builds, including
  ones built without the Metal backend, and running the wrong one produces confident nonsense —
  `Cannot set gpu device without gpu backend` instead of the real Metal assertion.

## Conformance runner

- Committed goldens generated by NumPy/Python MLX for required PR checks; a pinned, out-of-process
  Python MLX worker over a pipe for broader differential testing. Do not embed Python in the test
  process — `mlx-sys` statically links MLX and a second runtime brings its own global caches.
- CPU is the canonical golden device. Metal compares separately; reductions, FFT, linalg and
  quantized paths are not expected to be bitwise equal across the two.
- Qualify the harness with deliberate faults — wrong axis, swapped operands, wrong dtype, a no-op
  optimizer — before trusting a single green result from it.

### Anti-self-oracle rules

These exist to stop generated tests from asserting whatever the implementation currently does.

1. The corpus generator must never link or execute `mlx-rs`.
2. Every expected value carries provenance: a NumPy expression, a pinned Python MLX version, an
   upstream literal, or a named mathematical invariant.
3. An implementation change may not modify goldens, tolerances, or comparator logic in the same
   commit. Oracle changes are reviewed on their own.
4. Tolerances come from a central registry. Widening one is an oracle change.

## Follow-ups

- **ASan for the use-after-free lane.** devenv can supply nightly trivially, which unlocks
  `-Zsanitizer=address`; that would beat `libgmalloc` on speed and diagnostics. Verified limits on
  this machine: `clang -fsanitize=leak` is rejected outright for `arm64-apple-darwin`, and ASan
  reports `detect_leaks is not supported on this platform` — so `leaks` stays the only leak
  detector regardless of toolchain, and only the UAF lane is upgradeable. Full value needs MLX
  itself rebuilt with `-fsanitize=address` through the cmake crate, which is the real work.
  It is a backend swap behind the existing gate architecture, so nothing already built is wasted.
- **`parse_root_leak` also matches `STACK OF … 'ROOT LEAK: …'` header lines**, which carry no
  `[bytes]` suffix. In `named_sites` this inflates the real site's count by one and leaves a
  phantom entry for the baseline site with `bytes: 0`. The verdict is unaffected —
  `regression_count`/`regression_bytes` come from the summary line — but the diagnostics misreport.
  Skip stack-header lines in the parser.
- **The gate runs each binary twice** (once for test status, once under `leaks`; three times with
  `--guard-malloc`). `leaks` already passes the child's stdout through, so the test result line
  could be parsed from the run already being done.

## Open items carried from tranche 1

- **The threading contract is undocumented, and `--test-threads=1` is load-bearing.** Running
  `mlx-rs`'s lib tests with default parallelism aborts with SIGABRT:

      -[AGXG13GFamilyCommandBuffer tryCoalescingPreviousComputeCommandEncoderWithConfig:...]:1090:
      failed assertion `A command encoder is already encoding to this command buffer'

  That is the Metal driver catching two threads encoding to one command buffer, because MLX
  v0.30.6 has no per-thread stream concept at all — it exposes only `default_stream`,
  `set_default_stream` and `new_stream`. Upstream main added the entire model afterwards
  (`new_thread_unsafe_stream`, `new_thread_local_stream`, `clear_streams`, and a `default_stream`
  documented as per-thread), so this is not fixable at our pin.

  Note the abort comes from concurrent *operations* on a shared stream, not from moving an `Array`
  across threads. So `unsafe impl Send for Array` is defensible under the contract *arrays may move
  between threads; MLX operations must be externally serialised*.

  **Resolved in tranche 2:** that contract is now stated at crate level in `lib.rs` and beside the
  `unsafe impl` in `array/mod.rs`, and
  `ffi_safety.rs::concurrent_gpu_operations_abort_without_thread_local_streams` reproduces the
  abort under `#[ignore]`. Verified to produce the Metal assertion above, not some other failure.
  The mlx-c bump can relax the contract once thread-local streams are available.

  A second, independent race exists by inspection and is *not* what aborts here: `mlx-rs`
  registers the error handler from a thread-local `Once` (`error.rs:229`) into mlx-c's file-static
  globals (`mlx-c/mlx/c/error.cpp:17`), so each thread races on an unsynchronised `shared_ptr`
  assignment.
- `mlx-sys` is versioned `0.2.0` while its own manifest says it follows mlx-c, which is pinned at
  `v0.5.0`. Either enforce that policy or replace it with an explicit version-tuple policy.

## Then: the mlx-c bump

Pinned at mlx-c `v0.5.0` (MLX `v0.30.6`); upstream main builds MLX `v0.32.2` and adds 86 exported
symbols — gguf I/O, graph export, a compile cache, `slice_update_*`, `searchsorted`, `unstack`,
`vecdot`, `diff`, `flip`, `median`, window functions, `fftfreq`, `linalg_det`/`slogdet`.

Do not bump before the conformance runner exists. The corpus's purpose is to separate "MLX
legitimately changed this result" from "we broke it", and that needs a *before* picture; bumping
first makes every later difference unattributable. The concurrency win is not a reason to reorder:
196s of the ~220s suite is doctests, which run in a separate rustdoc process, so serialisation
costs about 25s.

Gate on zero *unclassified* changes rather than a raw symbol-count ratio: a ratio target rewards
adding shallow wrappers. Each new API lands vertically — parity mapping, independently sourced
conformance cases, ownership classification, docs — rather than in horizontal batches of untested
wrappers.
