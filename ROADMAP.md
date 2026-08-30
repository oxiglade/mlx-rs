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
- [x] **Conformance runner — committed-golden CPU ops lane.** 81 oracle-verified cases across six
      suites; qualified mutation matrix; suite moved 965 → 967 passing. The live Python worker and
      optimizer qualification remain (phase 2). Below.

## Known-weak coverage

Recorded because none of it shows up in a test count, and all of it survives adding more tests.

- `assert_array_eq!` (`mlx-rs/src/macros/assert.rs`) passes its single tolerance argument as *both*
  `rtol` and `atol`, and never compares dtype. `test_ada_delta` asserts a mean of `-0.348442` with
  tolerance `0.348442`, which admits roughly `[-0.818, +0.121]` — an optimizer that did nothing
  would pass. Fixing the macro is expected to turn several currently-green tests red; that is the
  point, and it must land as its own change so the failures are attributable.
- Several `compile`/optimizer tests run at learning rate `0.0`, removing the state transition they
  exist to exercise.
- At the mlx-c `v0.5.0` pin, `compile_with_state` re-traces its Rust closure on every outer call:
  a four-call, no-error hardware measurement observed invocation counts `1,2,3,4`, while Python
  `mx.compile` traces once. The wrapper creates and drops an inner `Compiled` per call, and its drop
  erases the `fun_id` cache entry, so the per-call closure overhead is real and any cached-graph
  benefit is unverified. Evaluate the bump target's `mlx_compile_cache` object family as the fix
  during the bump.
- 47 doctests use `rust,ignore` and never execute: 35 in `nn/activation.rs`, 6 on the `lib.rs`
  front page. Note that grepping for ```` ```ignore ```` misses these. Most appear to be a fence
  copied forward rather than a deliberate choice.
- `mlx-lm` has **zero executing tests**; all 6 are `#[ignore]` on missing model files. This is the
  newest code in the workspace.
- CI runs `cargo clippy -- -D warnings` without `--all-targets`, so lints in test code have never
  been gated. There are ~40 such warnings today.
- Measured on MLX 0.30.6 CPU: out-of-bounds gather indices return unchecked values rather than an
  error, while eval-stage C++ exceptions such as singular-matrix inversion escape the bindings as
  process aborts. The committed 0.32.2 replay reports singular inversion as a catchable invoke-stage
  error; the bump runtime pass must confirm the Rust assertion.
- `concatenate` claimed first-axis semantics in its documentation, but its binding flattens all
  arrays. The conformance corpus found the mismatch and the documentation is fixed; the API naming
  trap remains until the bump review.
- Measured at 0.30.6 CPU after the error-handler fix: eight threads concurrently constructing
  graphs and receiving invoke-stage errors is safe with per-thread delivery. The unsafe surface is
  specifically concurrent *eval* on the shared default stream, not graph construction.
- Measured on MLX 0.32.2 Metal: `conv3d` output is no longer row-major contiguous. Its logical
  values remain correct when read through the stride-aware strict comparator; tests must not use
  `as_slice` as an implicit layout assertion.

## Fixed findings

- The strict comparator surfaced six wrong test expectations (dtype-blind literals), initially
  misattributed first to operation bugs and then to an observer defect. Sub-assertion-level
  analysis located them; the operations and observer were correct.
- Fixed: `compile_with_state` restores caller state after tracing or compiled-apply errors and before retrying.
- Fixed: `try_as_slice` rejects non-row-contiguous views; safetensors conversion propagates that error.
- Fixed: `AdaDelta` now defaults `rho` to Python MLX's `0.9`; `0.99` remains available explicitly.
- The state-pack oracle's first fixtures were corrupted by Python in-place mutation aliasing (Adafactor weight decay); caught because Rust matched causal math and refused the corrupted expectation; snapshots now copy at capture.

### Coverage note from the strict-comparator migration

The FFT and `full` tests' literals are order-invariant (constant inputs), so they could not and
still cannot distinguish a transposed read; asymmetric FFT cases belong in the conformance corpus
when the FFT suite is added.

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

## Conformance runner — committed-golden CPU ops lane (done); live worker (later)

    conformance/                                # oracle side: generator, corpus, qualification
    cargo test -p mlx-tests --test conformance -- --test-threads=1

**The lane is qualified.** `harness_qualification` runs a 16-class mutation matrix (dtype, shape,
output count/order, absolute and relative thresholds, NaN, infinity sign, swapped noncommutative
operands, wrong axis, error inversion, F16/BF16 decoder calibration, endianness, empty tensors),
each required to fail as its declared mismatch class. On its first live run the corpus caught the
`concatenate` doc/semantics mismatch and the `try_as_slice` findings above, and generation-time
checks caught three wrong upstream-behavior assumptions (silent OOB gather; eager error timing —
no catchable eval-stage error exists at this pin; singular-`inv` aborting the process).

Regeneration: `conformance/README.md` documents the pinned-venv procedure; the generator refuses a
mismatched environment and double-runs to identical tree hashes. Python MLX 0.30.6 is the binding
oracle; NumPy corroborates only where generation-time agreement held (recorded per case).

## Conformance design notes

- Committed goldens generated by NumPy/Python MLX for required PR checks; a pinned, out-of-process
  Python MLX worker over a pipe for broader differential testing. Do not embed Python in the test
  process — `mlx-sys` statically links MLX and a second runtime brings its own global caches.
- CPU is the canonical golden device. Metal compares separately; reductions, FFT, linalg and
  quantized paths are not expected to be bitwise equal across the two.
- Qualify the harness with deliberate faults — wrong axis, swapped operands, wrong dtype, a no-op
  optimizer — before trusting a single green result from it.
- Sandboxed agent builds must never reuse the shared `target/` with modified feature flags. Three
  corruption incidents on 2026-08-28 produced the Metal JIT symptom
  `unknown type name 'bfloat16_t'`; use a scratch `CARGO_TARGET_DIR`.
- Multiple full MLX debug builds fill the disk; prune stale `target/` directories.

### Anti-self-oracle rules

These exist to stop generated tests from asserting whatever the implementation currently does.

1. The corpus generator must never link or execute `mlx-rs`.
2. Every expected value carries provenance: a NumPy expression, a pinned Python MLX version, an
   upstream literal, or a named mathematical invariant.
3. An implementation change may not modify goldens, tolerances, or comparator logic in the same
   commit. Oracle changes are reviewed on their own.
4. Tolerances come from a central registry. Widening one is an oracle change.

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

## Bump admission contract

The admitted dependency change is one immutable tuple to another:

- **Old tuple:** mlx-c commit `a1290d221f92bd020af805b7d14207eee4ec973b`, tagged `v0.5.0`,
  building MLX `v0.30.6`.
- **Target tuple:** mlx-c commit `c74db5307cc8ce122f48d97ef951b30578674e7f`, observed at
  `origin/main` on 2026-08-28, building MLX `v0.32.2` through its CMake pin. mlx-c tag `v0.6.0`
  builds MLX `0.31.1` and is not the target. Resolve the nested MLX pin again when the bump starts
  and reject a mismatch; never substitute `main`, `latest`, or a newer tag for the recorded commit.
- **Old-version oracle:** arm64 Python `3.12.14`, `mlx==0.30.6`, `mlx-metal==0.30.6`, and
  `numpy==2.2.6`, pinned by hashes in `conformance/requirements.lock`.

The bump must not begin until all eight admission items exist. Public API admitted after the bump
also follows the idiom-wave rules in [CHARTER.md](CHARTER.md):

1. Exact target commit/version tuple and checked-in playbook.
2. Enforced oracle boundary, digest integrity, and staged-case process.
3. Known error-registration race eliminated and concurrent behavior tested.
4. Exact-SHA qualified FFI leak verdict, either in CI or through an explicitly lower-trust
   recorded admission procedure.
   **Status: done at CI strength: run 33277976712's ffi-gate job passed all four calibration phases on a blaze runner (deliberate leak detected on the runner itself); calibration JSON retained as a run artifact. Per-child leaks/guard-malloc timeouts convert runner incapability into a named verdict rather than a hang.**
5. Strict tensor comparator and removal or reclassification of `lr=0` false claims.
6. Minimal target ABI-delta ledger, Rust public-API baseline, and supported feature matrix.
   **Status: complete** — the classified target delta, stable syn-based Rust API baseline, supported build matrix, and qualified `verify-ledger` gate are present.
   The baseline also gates the post-bump idiom wave defined by [CHARTER.md](CHARTER.md).
7. Qualified stateful optimizer, compile, and transform packs.
8. Deterministic replay against the exact target Python MLX version.

Ledger evidence is typed. `conf:` identifies committed-golden or differential semantic cases;
`ffi:` identifies ABI, ownership, lifetime, callback, or leak evidence; `state:` identifies full
state trajectories, identity, pruning, retry, or error-atomicity evidence; `thread:` identifies
concurrency, stream, serialization, or error-handler evidence; and `api:` identifies a Rust public
path, public-surface baseline, feature/build result, or API contract check. A ledger entry can cite
more than one class. A conformance case alone is not sufficient evidence for an ownership or
threading claim.

A waiver applies to one named check or ledger entry. It records the exact scope, failure or missing
capability, risk, compensating evidence, owner, reviewer approval, and an expiry date or milestone
no later than the next dependency bump. Waivers do not change the immutable target and do not turn
an unclassified delta into a classified one. Missing, blanket, unreviewed, or expired waivers fail
admission. The aggregate bump verdict lists every active waiver.

Not required before the bump are a full Python parity ledger, broad randomized differential
testing, expanded mlx-lm coverage, all doctests, all-target clippy cleanup, or ASan.

## Execution waves

The eight admission items above are the bump gate. Work in the same wave can proceed independently
except for the listed shared-file constraints; a non-admission item does not become a bump gate
merely because it shares a wave with one. These waves establish bump evidence; the later idiom wave
is governed by [CHARTER.md](CHARTER.md).

| Wave | Work | Done criteria | Parallelism and serialization |
|---:|---|---|---|
| 0 | Freeze target and admission contract | Exact old and target SHAs, nested MLX versions, oracle environment, waiver policy, evidence classes, minimum gate, and this 13-step playbook are checked in. | Completes before target-ledger generation or target replay. |
| 1 | Enforce the oracle boundary; remove the error-handler race and qualify FFI reporting on an exact SHA; begin canonical ABI fingerprinting; add one mlx-lm sentinel. | Protected oracle/schema/comparator/qualification code is separated from implementation adapters; staged cases work; mixed oracle/implementation changes and digest tampering fail; process-global error registration and a concurrent regression replace the race; clean and deliberate-leak calibration qualifies an environment-rich `verify-ffi` JSON report in CI or records the lower-trust fallback; old and target canonical fingerprints can be generated; one tiny offline prefill/decode/cache sentinel has independent expectations. Parser diagnostic correction and duplicate-run reduction are folded into this FFI work only where needed, and any such change requalifies the gate. | One integration owner edits `.github/workflows/validate.yml`. Do not assign concurrent edits to the conformance monolith or generator. The mlx-lm sentinel is independent and is not itself an admission item. |
| 2 | Migrate to the strict tensor comparator; complete target-delta classifications and the Rust public-API baseline. | Comparisons qualify separate `rtol`/`atol`, dtype, shape, NaN, infinity, and boundary behavior; tensor `PartialEq` assertions are audited; optimizer/compile correctness no longer rests on `lr=0` or scalar summaries; canonical function and ABI-type fingerprints produce a zero-unclassified target delta with typed evidence; wrapped entries resolve to real Rust paths; public-API and supported-feature baselines exist; synthetic add/remove/signature/type/evidence mutations fail the ledger. The baseline is the gate for the [charter's idiom wave](CHARTER.md#charter-rules). | The comparator contract settles before Wave 3 packs. Serialize changes to `xtask/src/main.rs`; Wave 2 consumes Wave 1 fingerprints. |
| 3 | Add stateful optimizer, compile-state, and transform packs. | Full parameter and optimizer-slot tensors match independent expectations for two or three nonzero updates; compile-state cases cover frozen, nested, changed/unchanged/pruned state, repeated and fallible calls, duplicate-retry prevention, and error atomicity; nonlinear multi-input/output grad, VJP, and JVP values are checked; the named no-op/stuck-counter/reordered-state/frozen-mutation/output-split/duplicate-retry fault matrix fails as expected. | Put packs in new focused test files. Do not have pack owners append concurrently to `mlx-tests/tests/conformance.rs`. |
| 4 | Add deterministic target-version replay and aggregate bump admission. | The worker rejects any handshake other than Python MLX `0.32.2`; named corpus and state cases produce structured, reproducible old-versus-target verdicts with reset/isolation checks; legitimate semantic changes retain separate reviewed baselines; the aggregate verdict consumes the Wave 1-4 reports, verifies the recorded tuple and fingerprints, lists waivers, and fails on any unmet admission item. | Wait for the case schema and Wave 3 state recipes to stabilize. One owner integrates the aggregate command in `xtask/src/main.rs`. |
| 5 | Add scheduled seeded differential breadth and deferred hygiene. | Scheduled cases record reproducible seeds, timeouts and crashes, minimize failures, and promote accepted cases into the committed corpus; genuine Rust doctests compile while formulas remain text; all-target clippy is clean or no-new-warning gated; optional ASan work proceeds only after a spike demonstrates findings distinct from the qualified leak/Guard Malloc gate. | Broad worker changes and wide documentation or lint churn come last. One owner coordinates any workflow changes. |

Wave 2 done criteria:

- [x] Comparisons qualify separate tolerances, dtype, shape, non-finite values, and boundary behavior; tensor equality and optimizer/compile false claims are audited.
- [x] Canonical fingerprints produce a zero-unclassified target delta with typed, resolvable evidence and real Rust paths for wrapped entries.
- [x] Rust public-API and supported-feature baselines exist, and synthetic add, remove, signature, evidence, and unclassified mutations fail the ledger gate.

Wave 3 progress:

- [x] Wave 3a: all nine optimizers have independent three-step parameter/slot oracles, exact flattened-state key-set checks, frozen-parameter coverage, eager/compiled consistency sentinels, and the no-op/stuck-counter/reordered-state/frozen-mutation/wrong-step qualification matrix.
- [ ] Wave 3b: complete compile-state behavior (nested, changed/unchanged/pruned, repeated/fallible calls, duplicate-retry prevention, and error atomicity) and nonlinear grad/VJP/JVP transform values plus the remaining output-split and duplicate-retry mutations.

The shared-file conflict list is `mlx-tests/tests/conformance.rs`, `conformance/generate.py`,
`.github/workflows/validate.yml`, and `xtask/src/main.rs`. Give each shared file one integration
owner at a time.

### Deferred and cut work

- **Defer the full Python API ledger until module catch-up begins.** The bump needs the exact
  target delta, not a universal taxonomy.
- **Cut a full historical C-to-Rust mapping.** Classifying only the old-to-target delta preserves
  the pre-bump evidence at lower review cost.
- **Cut the requirement to execute all 47 ignored doctests.** Wave 5 compiles genuine examples and
  marks mathematical pseudocode as text.
- **Defer broad all-target clippy cleanup to Wave 5.** It is hygiene rather than semantic
  admission evidence.
- **Defer broad randomized differential work to Wave 5.** Deterministic state packs and exact
  target replay provide lower-triage admission evidence first.
- **Cut mlx-lm expansion beyond one sentinel.** A tiny offline model is the upper-layer canary;
  architectures, large tokenizers, remote models, and generation modes are outside bump admission.
- **Cut full ASan/MLX C++ instrumentation from admission.** LeakSanitizer is unavailable on this
  arm64 macOS host and the qualified `leaks` plus Guard Malloc lanes already cover the current
  gate; Wave 5 requires a distinct-findings spike before further investment.
- **Cut standalone parser and single-run optimization projects.** Fold the known
  `parse_root_leak` diagnostic error and redundant binary executions into Wave 1 FFI CI work only
  where diagnostics or runtime require them, then requalify `verify-ffi`.

## Repeatable `mlx-c` bump playbook

### Charter deltas

- [x] Retain each `Compiled` value's originating compile-cache handle and use that exact handle for
  erase; resolve the caller's current cache separately for each `clear_cache()` call
  ([CHARTER.md](CHARTER.md#concrete-0322-rebind-decisions)).
- [x] Make `Compiled` structurally `!Send + !Sync` for this tuple
  ([CHARTER.md](CHARTER.md#charter-rules)).
- [x] Preserve monotonic `fun_id` allocation, including fresh IDs for clones, and consume erase status
  without panicking in `Drop` ([CHARTER.md](CHARTER.md#charter-rules)).
- [x] Delete the unconditional compile-with-state retry; one call performs one attempted state
  transition, whether it succeeds or fails ([CHARTER.md](CHARTER.md#charter-rules)).
- [ ] Exercise the nested cold-cache deadlock shape in a subprocess with a hard deadline and proof that
  the cold trace ran; do not add a process-global lock without target evidence
  ([CHARTER.md](CHARTER.md#charter-rules)).
- [x] At bump time, fail closed on compiled-state count, key-layout, and optional-presence mismatches;
  do not truncate positional updates ([CHARTER.md](CHARTER.md#charter-rules)).
- [ ] Expand stream admission to explicit identity/pass-through, nested stream and device restoration
  after success and panic, cross-thread isolation, CPU and Metal per-thread defaults, moved and
  cloned arrays, and stream create/free churn ([CHARTER.md](CHARTER.md#charter-rules)).

1. [ ] **Resolve the immutable target tuple.** Start from the old and target commits recorded in the
   admission contract. Re-read the target mlx-c CMake pin and require MLX `v0.32.2`; record runtime
   `mlx_version`, Xcode, arm64 architecture, Rust toolchain, and supported feature set. Never target
   “latest”.
2. [x] **Generate the bump plan in isolated worktrees.** Produce normalized public-header AST,
   bindgen signature, ABI-relevant type, and exported-symbol fingerprints for both commits without
   changing the working submodule. Wave 1 delivers canonical fingerprint generation; Wave 2
   qualifies the classified delta.
3. [x] **Classify every target API and ABI change.** Mark every added, removed, or changed entry as
   wrapped, deferred, intentionally unexposed, removed, or blocked. Record the Rust path, risk or
   ownership class, and typed evidence IDs for affected entries. Every new `new/free` handle family
   needs an ownership entry. The Wave 2 ledger must report zero unclassified entries.
4. [ ] **Update provenance coherently.** Update the submodule commit and every recorded version tuple
   together. Either enforce the statement that `mlx-sys` follows mlx-c versioning or replace it
   with an explicit tuple policy; do not leave the existing `mlx-sys 0.2.0` versus mlx-c `v0.5.0`
   ambiguity in place.
5. [ ] **Verify generated and linked surfaces.** Require header, bindgen, ABI-type, and exported-symbol
   fingerprints to match the plan; no removed symbol may remain referenced. Runtime `mlx_version`
   must report `0.32.2`. This consumes the Wave 2 ledger and Wave 4 target handshake.
6. [ ] **Build dependency canaries.** Build CPU debug, CPU release, and the supported default
   Metal/Accelerate configuration. Run mlx-c examples and an appropriate upstream MLX C++ test
   subset to distinguish dependency breakage from Rust breakage. Add a sanitized build only if its
   lane has been qualified.
7. [ ] **Run the qualified FFI safety gate.** Run `cargo run -p xtask -- verify-ffi` for leak and test
   status, plus `--guard-malloc` for the required use-after-free/double-free check. Consume the
   exact-SHA, environment-bound Wave 1 report; callback, error, clone, drop, and concurrent-error
   cases must be green, with no unexpected leak above the qualified baseline.
8. [ ] **Run old semantic baselines first.** Replay the committed-golden conformance lane before
   changing oracle authority. Classify failures as candidate upstream changes; do not regenerate
   them away in the bump change.
9. [ ] **Run target differential and target baselines.** Use the Wave 4 out-of-process worker with an
   exact Python MLX `0.32.2` handshake. CPU is canonical; bounded Metal comparison is separate.
   Preserve reviewed old and target baselines where MLX legitimately changed semantics.
10. [ ] **Run the high-risk packs.** Run dtype comparison/classification, shape and index properties,
    retained fuzz regressions, gradients, eager/compiled multi-step state, frozen and pruned state,
    all optimizer state, expanded stream admission, and supported quantized/model smoke cases.
    Wave 3 delivers the required optimizer, compile-state, and transform packs; any wider pack is
    required only when its ledger surface is affected. Compiled-state and stream cases enforce the
    bump-time subset of the [charter](CHARTER.md#charter-rules).
11. [x] **Run parity and public-surface gates.** Require zero unclassified target C/ABI delta and zero
    unexplained Rust public-API drift. Every newly exposed Rust API needs an ownership disposition
    and appropriate typed evidence. Full Python parity remains deferred, but any Python-qualified
    surface changed by the bump must be classified. Post-bump admission must also satisfy the
    [charter's idiom-wave rules](CHARTER.md#charter-rules).
12. [ ] **Run the supported workspace matrix.** Run declared MSRV and stable configurations,
    debug/release where relevant, the single-thread legacy suite, explicit supported-thread tests,
    genuine doctests, and the one tiny deterministic local mlx-lm decode. Do not use Hub, network,
    or user-cache fixtures.
13. [ ] **Aggregate one verdict.** The Wave 4 aggregate command consumes all structured reports and
    exits nonzero unless every admission check passes. It verifies the submodule commit, nested MLX
    version, generated fingerprints, evidence links, report environment and expiry of every
    waiver against the recorded tuple.

A dependency-only bump is complete when supported behavior has no unclassified regression and
every new upstream entity is classified. Full catch-up is complete only when the designated
deferred semantic set reaches zero. Each new Rust API lands vertically under the
[charter](CHARTER.md), with its parity mapping, independently sourced cases, ownership
classification, required golden or differential result, applicable property or gradient evidence,
and public documentation.

The first post-bump vertical cohort is delivered: the four window functions and the two FFT
frequency constructors have canonical Rust paths, checked integer conversion, public examples,
NumPy-corroborated MLX 0.32.2 signal cases, and wrapped ledger dispositions.
