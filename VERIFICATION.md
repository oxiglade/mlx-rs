# Verification

`cargo xtask <command>` is the root alias for
`cargo run --locked --manifest-path xtask/Cargo.toml -- <command>`. Tooling has its own
workspace and lockfile, so its dependency features cannot unify with product
crates. Its subprocesses run in the product workspace. CI checks xtask with
explicit `cargo fmt`, `cargo clippy`, and `cargo test` commands using
`--manifest-path xtask/Cargo.toml`; root `--workspace` commands cover only product
crates. The FFI inventory likewise contains only product test binaries.

## FFI leak and use-after-free gate

    cargo xtask verify-ffi
    cargo xtask verify-ffi --guard-malloc

For admission, run the qualified combined gate from a clean checkout and put its artifacts outside
that checkout:

    mkdir -p "$RUNNER_TEMP/ffi-gate"
    cargo xtask verify-ffi --calibrate --guard-malloc \
      --budget-seconds 3600 --leaks-timeout-seconds 600 \
      --guard-malloc-timeout-seconds 1200 \
      --report "$RUNNER_TEMP/ffi-gate/verify-ffi.json" \
      > "$RUNNER_TEMP/ffi-gate/calibration.json"

Locally, use a scratch directory in place of `$RUNNER_TEMP/ffi-gate`. The calibration JSON
on stdout contains the detailed controls. The command fails if execution or calibration fails.
Requalify after parser or baseline changes.

`leaks --atExit` reports a constant Objective-C initialization leak (32 bytes), so compare against
the qualified baseline rather than trusting its exit code. Discover binaries through Cargo JSON;
exclude proc-macro binaries by target kind. Guard Malloc supplies the separate use-after-free and
double-free lane. Calibration must detect a deliberate iterator leak, use-after-free, and double-free,
and pass clean controls before its combined inventory report can qualify admission. Each Guard Malloc
process must show that libgmalloc loaded; unavailable instrumentation and timeouts fail the gate.

CI runs the calibrated leak gate on every PR, proving that leaks can attach and detect the deliberate
leak. Pushes to main and bump admission also run Guard Malloc. `verify-ffi --calibrate` binds its
clean inventory execution to the calibration result and exits nonzero if either fails. Inventory
completeness is checked against Cargo metadata inside the command. Leak-only validation cannot
substitute for the bump workflow's combined gate.

Reports record the checked-out `source_commit`, `github_sha` and `github_pr_head_sha` when available,
plus compiler and platform details for diagnosis. PR jobs run on the merge SHA. The clean gate has
a 60-minute budget, each leak subprocess a 10-minute timeout, and each Guard Malloc subprocess a
20-minute timeout. The calibration step has a 75-minute timeout including builds; the job has a
90-minute limit. Calibration reports and logs upload even on failure.

## Conformance runner

    cargo test -p mlx-tests --test conformance -- --test-threads=1

The qualified strict comparator checks dtype, shape, output count/order, separate tolerances,
non-finite values, and layout-sensitive observations. Regeneration uses Python 3.12.14,
`mlx==0.32.2`, `mlx-metal==0.32.2`, and `numpy==2.2.6`, pinned in
[requirements.lock](conformance/requirements.lock); it requires identical hashes from two runs.

[replay_target.py](conformance/target/replay_target.py) implements deterministic subprocess replay
with an exact-version handshake. It discovers all corpus suites and also runs state, transforms,
and the historical probes. Its committed reports compare MLX 0.30.6 with 0.32.2;
[scratch replay](conformance/README.md#exact-target-replay) compares the current corpus without
replacing those reports.

### Anti-self-oracle rules

1. The corpus generator must never link or execute `mlx-rs`.
2. Every expected value carries provenance: a NumPy expression, a pinned Python MLX version, an
   upstream literal, or a named mathematical invariant.
3. An implementation change may not modify goldens, tolerances, or comparator logic in the same
   commit. Oracle changes are reviewed on their own.
4. Tolerances come from a central registry. Widening one is an oracle change.

Python stays out of the Rust process because both runtimes carry global caches. CPU is canonical;
Metal comparisons are separate. Use a scratch `CARGO_TARGET_DIR` for different feature sets.

## Executed configuration reports

The required entries in `ledger/supported-feature-matrix.json` identify the default stable/MSRV,
CPU-only stable debug/release and Metal JIT stable debug lanes, including package selection,
Rust profile, toolchain, resolved backend features and native CMake settings. Run each configuration with its own
`CARGO_TARGET_DIR` and put reports outside the checkout:

    cargo xtask config-report macos-arm64-default \
      --report "$RUNNER_TEMP/config-reports/macos-arm64-default.json"
    cargo xtask config-report macos-arm64-cpu-only \
      --report "$RUNNER_TEMP/config-reports/macos-arm64-cpu-only.json"
    cargo xtask config-report macos-arm64-cpu-only-release \
      --report "$RUNNER_TEMP/config-reports/macos-arm64-cpu-only-release.json"
    cargo xtask config-report macos-arm64-metal-jit \
      --report "$RUNNER_TEMP/config-reports/macos-arm64-metal-jit.json"

Use the selected toolchain for the entire invocation; `macos-arm64-default-msrv` requires Rust
1.88.0. Give each configuration a separate `MLX_RS_METAL_PATH` as well; JIT and non-JIT
metallibs must not share a directory. Locally, substitute a scratch directory for `$RUNNER_TEMP`.
The wrapper executes the ledger's Cargo test command once, including doctests, with serial tests
and captured test output.
It writes a JSON report plus a sibling `.log` file and exits nonzero if Cargo fails,
qualification fails, or the report cannot be written. A zero exit status means qualified execution.

Reports bind the clean checkout SHA, mlx-c commit, nested MLX version, Cargo.lock SHA-256,
`rustc --version --verbose`, macOS/SDK versions, resolved features, Rust profile and compiled
optimization/debug-assertion settings, effective CMake cache settings (including build type and `MLX_METAL_JIT`), exact argument vector,
build-plus-test wall time, and each binary/doctest's counts and outcome.
`github_sha` and `github_pr_head_sha` provide run context. Missing runs,
incomplete summaries, filtered tests, `NOT RUN`, and unexpected zero-test runs fail qualification.
Each build declares `checkpoints_required` for bump admission. Only the default configuration
requires checkpoint inference, and only when `MLX_RS_BUMP_RUN=1`. Ordinary runs record
`checkpoints_required: false`. The bump workflow enables the ledger's checkpoint requirement in
the default configuration command. Where checkpoints are not required, the exact manifest-absent
`real_checkpoints` message remains visible with `not_run: true` and `excluded_by_configuration: true`; every other `NOT RUN`
still fails. The wrapper removes the checkpoint manifest from those configurations' environments.
The ledger's `allow_empty` lists intentionally empty and compile-only harnesses, the ignored leak
calibration probe, and documentation targets without runnable tests. Ignored individual tests remain
visible in counts. Nested harness summaries are recorded separately from the enclosing binary
so a filtered child regression probe does not inflate its parent's counts.

CI uploads reports and logs even when tests fail. The workflow derives its configuration matrix
from every required ledger entry, so adding a required configuration adds a job automatically.
Ordinary PR runs exclude the manifest-absent checkpoint test. Bump runs supply the pinned fixtures
and independent Python expectations described below. Reports are diagnostic output; admission
uses job success, without rereading reports in a later job.

The stable default configuration enables `mlx-lm/oracle-hooks`, running the committed model
parity and sentinel fixtures in the existing workspace test job. This reuses its native build;
MSRV and JIT do not repeat these suites. `config-report` requires both harnesses, plus conformance and the optimizer, compile-state and
transform packs, to pass with nonzero tests and no ignored, filtered, or `NOT RUN` tests.
Fixture presence and digests alone do not establish sentinel execution.

## Compiled public API inventory

Install the extraction toolchain without changing the active toolchain:

    rustup toolchain install "$(cat xtask/api-nightly.txt)" --profile minimal
    cargo xtask verify-ledger

`xtask/api-nightly.txt` pins nightly for rustdoc JSON extraction only. `xtask` builds on stable;
`public-api = 0.50.0` normalizes JSON schema 55. Extraction requires an `aarch64-apple-darwin`
host and uses the pinned compiler and rustdoc explicitly. Use `MLX_RS_API_TARGET_DIR` for scratch
builds and per-configuration JSON files, and `MLX_RS_METAL_PATH` for scratch metallibs. The default
build directory is `target/api-inventory`. Nightly targets are separate from stable targets.
Feature sets with identical `mlx-sys` features share a Cargo target directory: CPU-only,
Metal, and Metal JIT each have one directory, shared with their `safetensors` variant.

The scope is `mlx-rs`. The four API feature sets are `default`, `cpu-only`
(`--no-default-features --features accelerate`), `safetensors` (default plus `safetensors`),
and `cpu-only-safetensors`. Extraction also compares `metal-jit` with and without `safetensors`
against the corresponding non-JIT inventory and fails if their public surfaces differ, requiring
a distinct feature set before regeneration. JIT metallibs use a separate `metal-jit` subdirectory.

The committed baseline merges identical entries and lists their feature-set availability.
It sorts entries by path and declaration and writes one JSON entry per line. Each entry records
the public path, kind, normalized signature, generated and hidden flags, and the enclosing impl
header, deprecation metadata, and procedural-macro helper attributes where applicable. Simple
local type aliases without generic parameters or target arguments record their compiled target; member entries retain canonical paths, and ledger
lookup resolves those alias prefixes with the intersection of alias and member availability.
Specialized aliases remain unexpanded because their method bounds require substitution. Impl
declarations and generic receiver members use the impl header as
their identity because they have no single crate-rooted Rust path. Public re-exports are expanded
by rustdoc and `public-api`. The adapter fills external procedural-macro globs from separately
compiled dependency JSON because rustdoc leaves those targets unresolved. Other unresolved
re-exports fail extraction rather than leaving wildcard entries. Source parsing only classifies
provenance against compiled spans.
It does not discover or admit inventory entries. A generated flag means the declaration was
emitted by a macro, derive, or compiler, rather than a matching declaration at its source span.

Blanket implementations and their members are omitted. Auto-trait implementations, including
thread traits, and implementations derived for mlx-rs types remain tracked.

Public `#[doc(hidden)]` items remain included. Their flags come from their attributes and a
comparison of public item identities with a second rustdoc run without hidden items, which also
covers hidden re-exports without confusing a hidden field with its visible containing type.
Every extraction qualifies the adapter with compiled builder, derive, re-export, hidden-public,
and cfg fixtures. Five separately compiled omission controls (including an external derive
re-export), two compiled Send/Sync change controls, and 24 incorrect-field controls must fail.
Stable consumer probes compile the fixture in both configurations. Run just these controls with
`api-baseline --qualify`.

`verify-ledger` regenerates the compiled inventory and rejects baseline drift before checking
wrapped paths, behavioral renames, admission paths, symbol mappings, and `api:path:` evidence.
An unqualified reference claims every API feature set. An object's `api_feature_sets` list
narrows that claim for its references and descendants; unknown, empty, or duplicate sets fail.
Metal-only references explicitly claim `default` and `safetensors`. A `wrapped` entry with no
supported configuration can cite `unsupported_cfg` as
`api:cfg:<source-file>#<function>#not(target_os = "macos")`, repeat that citation in `evidence`,
and supply a one-line `rationale`. This classification currently accepts only the canonical
`mlx_rs::ops::qqmm` path and its `mlx-rs/src/ops/quantization.rs` citation. It requires a macOS
inventory target, checks that the named public function carries the exact cfg, and rejects a
path present in any supported feature set. It does not admit that wrapper to the supported surface.

The mlx-lm [support decisions](mlx-lm/SUPPORT.md) describe its supported subset.
Its executed model parity tests and external feature consumers remain separate
from compiled API evidence for `mlx-rs`.

Regenerate with `api-baseline --out ledger/rust-api-baseline.json` and review the diff. CI runs
extraction, its qualification controls, baseline comparison, and ledger verification once through
`verify-ledger` in the existing `checks` job. Running `api-baseline --check` separately repeats
extraction. A baseline match establishes compiled surface coverage, not semantic correctness
or linked-library exports.

## Bump admission contract

A bump is admitted when the `admission` job in `.github/workflows/bump-admission.yml` succeeds
with `ordinary: false`. PR runs use the checked-out merge SHA; `workflow_dispatch` uses the
selected ref's SHA. The final job runs with `if: always()`, needs every required job, and fails
unless every result is `success`. A missing, skipped, cancelled or failed job cannot admit a bump.
The configuration matrix disables fail-fast, so one failed configuration does not cancel the others.

The required graph is:

| Job | Executable requirements |
| --- | --- |
| `dependencies` | Generate one product `Cargo.lock` with Rust 1.88.0, upload it, and select every required configuration from `ledger/supported-feature-matrix.json`. |
| `checks` | Product and xtask fmt/clippy, xtask tests, oracle boundary, `verify-bump`, compiled API and ABI/ledger checks, and the Cargo-archive link/version canary. |
| `ffi-gate` | `verify-ffi --calibrate --guard-malloc`, including clean controls and deliberate leak, use-after-free and double-free detection. |
| `configurations (macos-arm64-default, stable)` | Qualified workspace tests and doctests with `oracle-hooks`, including parity, sentinel, semantic/state packs and all twelve real-checkpoint case/device executions. |
| `configurations (macos-arm64-default-msrv, 1.88.0)` | Qualified default workspace tests and doctests on MSRV. |
| `configurations (macos-arm64-cpu-only, stable)` | Qualified CPU/Accelerate debug tests and doctests. |
| `configurations (macos-arm64-cpu-only-release, stable)` | Qualified CPU/Accelerate release tests and doctests. |
| `configurations (macos-arm64-metal-jit, stable)` | Qualified Metal JIT workspace tests and doctests. |
| `admission` | Require successful `dependencies`, `checks`, `ffi-gate` and every `configurations` matrix leg. |

`validate.yml` calls this same graph with `ordinary: true` on ordinary PRs and main pushes.
That mode omits checkpoint provisioning and uses leak-only calibration on PRs; main pushes still
run Guard Malloc. It is ordinary validation, not bump admission. A gitlink-changing PR also starts
the full bump workflow. No result from a separate workflow run is imported for admission, and a
manual bump dispatch runs the whole graph without needing a concurrent `validate.yml` run.

Every build job downloads the one product lockfile. Product Cargo commands use `--locked`,
including config tests, FFI discovery and nightly API extraction. xtask uses its own committed
lockfile. Each configuration has separate build and metallib directories. The jobs can run in
parallel after dependency resolution; they do not rebuild MLX merely to reconcile artifacts.

`config-report` enforces the declared toolchain, features, profile, native settings and complete
binary/doctest inventory before returning success. It also validates the default configuration's
checkpoint contract and live oracle against actual case/device output. `verify-ledger` extracts
and qualifies the compiled Rust API, compares its baseline, regenerates the ABI delta and rejects
unclassified, extra or mismatched entries. `linked-exports` checks planned symbols and executes
the link/version canary against Cargo's archives. `verify-ffi` owns instrumentation calibration
and inventory completeness. Each command's exit status controls its job.

`verify-bump` is a small failing command for checks that remain outside those execution commands:
the current immutable mlx-c/MLX tuple and its classification endpoints, the named error-registration
concurrency regression, sentinel fixture digests,
and strict exact-target replay coverage, provenance and deterministic double-run hashes. It does
not aggregate an admission verdict or rerun the semantic suites. Replay generation and its
committed numerical artifacts retain their existing contract.

Ledger evidence stays typed: `conf:` for semantic cases, `ffi:` for ABI/ownership/lifetime evidence,
`state:` for trajectories and atomicity, `thread:` for concurrency, and `api:` for public paths and
build contracts. Semantic cases alone cannot prove ownership or threading safety.

### Repeatable `mlx-c` bump playbook

Freeze the old and target commits and nested MLX version, update the tuple checked by
`verify-bump` and the ABI classification endpoints, then classify every generated delta.
The active comparison is mlx-c `c74db5307cc8ce122f48d97ef951b30578674e7f` to
`a341b4925024b88b2c593468f16e12f5e17315da`; both pin MLX `v0.32.2`.
The prior classification is retained in
`ledger/mlx-c-c74db5307cc8ce122f48d97ef951b30578674e7f-classification.json`.

Run `cargo xtask verify-bump` against the target checkout. Prepare exact-target replay using
[the replay procedure](conformance/README.md#exact-target-replay) when the target or replay inputs
change. Keep oracle changes separate from implementation changes, and never regenerate away a
semantic regression. The `checks` job runs the oracle boundary against the PR base, or `origin/main` for a manual
run. Pushes to main check working-tree integrity.

`ledger/abi-toolchain.json` pins the full Rust compiler identity and bindgen version.
Install that compiler with rustfmt before running the ledger gate:

    rustup toolchain install "$(python3 -c 'import json; print(json.load(open("ledger/abi-toolchain.json"))["rustc"].split()[1])')" --profile minimal --component rustfmt
    export MLX_RS_ABI_OUTPUT_DIR="$BUMP_ROOT/reports/abi"
    cargo xtask verify-ledger

`verify-ledger` regenerates both immutable commits from the active classification with that
same toolchain, checks the current pin belongs to the tuple, and validates every delta entry
against its human classification. Missing, extra or mismatched classifications fail. Unchanged
entities inherited from the old pin do not require new classifications. The generator rejects
compiler or bindgen version drift. Rust selection is confined to extraction subprocesses;
product and xtask checks continue to use the selected stable toolchain.

Inventories and `delta.json` are generated artifacts, retained under `MLX_RS_ABI_OUTPUT_DIR`
(default `target/abi-inventory`). Ordinary CI uploads them as `current-abi-inventory`; bump
runs upload the same artifact. The generation time is logged separately from API
extraction. `fingerprint --ref <commit> --out <path>` and `fingerprint-delta --old <file>
--new <file>` remain available for inspection; run the generator under the recorded Rust
compiler. For a new bump, freeze the old/target commits in the classification and bump
tuple, generate both inventories, and classify the resulting delta. Keep the toolchain pin
and classifications in Git; review a toolchain change with regenerated evidence for both sides.

Every ABI extraction checks `xtask/tests/fixtures/abi-extractor`, a small C header set with
independent expected entries for functions, typedefs, structs, unions, enums, callbacks,
function-pointer fields, constants and owned handles. The same fixture runs in xtask tests
using the recorded rustfmt, which must be installed even when testing on a newer stable Rust.
An extractor that drops an entity on both sides cannot silently erase that class of delta.
The linked-symbol check remains separate: header inventory cannot prove linked exports.

`python3 -B xtask/provision_checkpoints.py --cache "$BUMP_ROOT/checkpoints" --output
"$BUMP_ROOT/reports/checkpoints"` downloads 6,199,630,244 bytes across six checkpoints and two
tokenizer directories. `conformance/mlx-lm/bump-oracle.json` pins every repository revision and
file SHA-256. Files already in the cache directory are rehashed before they are reused.
Downloads use immutable revision URLs and temporary files; mismatches fail before installation.
The provisioner reuses `real_checkpoints.py`'s manifest validation and relocates only local paths in
the committed contract. Pass `--python "$BUMP_PYTHON"`, pointing at a uv-managed Python 3.12.14
virtualenv installed with `uv pip sync --require-hashes conformance/mlx-lm/requirements.lock`.
It runs `conformance/mlx-lm/live_oracle.py` in a separate Python process using MLX 0.32.2 and
mlx-lm 0.31.3. The generator never imports or runs mlx-rs. The live report supplies expectations;
`live-oracle-generation.json` records the Python version and generator, lock, contract and
live-report hashes. The committed oracle contains only the input contract and environment pins:
case/device inventory, prompts and prompt IDs, generation parameters, checkpoint revisions and
file hashes, and tokenizer entries. Numerical expectations are generated on the runner.
Contract and oracle-generator changes must remain separate from implementation changes.

Set `MLX_LM_REAL_CHECKPOINTS="$BUMP_ROOT/reports/checkpoints/real-checkpoints.json"` and
`MLX_RS_BUMP_RUN=1`, then run the default `config-report`. The workflow does this only for the
default stable matrix leg. The command requires the manifest, checks the pinned input contract,
live report and generator hashes, and requires exactly one passing output for every checkpoint
case on CPU and Metal. The Rust harness hashes the files and compares inference with the independent
Python results. Missing input, incomplete execution or a failed comparison fails the job.
Other configurations remove the checkpoint manifest from their test environment.

The Cargo-archive canary runs in `checks` on ordinary PRs and bumps, reusing clippy's native build:

    cargo clippy --locked --message-format=json-render-diagnostics -- -D warnings \
      > "$BUMP_ROOT/reports/cargo-artifacts.json"
    cargo xtask linked-exports --cargo-output "$BUMP_ROOT/reports/cargo-artifacts.json" \
      --report "$BUMP_ROOT/reports/cargo-exports.json"

It resolves the archives named by Cargo's mlx-sys build script and `mlx-sys/native-libraries.txt`:
libmlx, libmlxc, libgguflib, and libjaccl when built. The Xcode compiler-runtime archive comes from
the same lookup as the product build. Archive paths and SHA-256 hashes appear in job output and
the diagnostic report. The checker requires every planned C function to be present. Other exports
are recorded without a deny policy. Header/type fingerprints remain necessary because a matching
symbol name does not prove its type.

The generated C++ canary includes the same headers, retains a volatile reference to every planned
function, checks its object's undefined symbols, and links the exact Cargo archives. It calls
`mlx_version` through mlx-c and `mlx::core::version`; both must return the pinned MLX version.
mlx-c has no independent runtime version string, so its identity is the source commit.
The canary does not create another native build.

For local verification, use scratch paths for `CARGO_TARGET_DIR`, `MLX_RS_METAL_PATH`,
`MLX_RS_API_TARGET_DIR` and `MLX_RS_ABI_OUTPUT_DIR`. Generate the product lockfile once before
locked commands. Run the commands above, all required configurations, the calibrated FFI gate,
`verify-ledger`, `verify-oracle-boundary --base main` and `verify-bump`. These checks diagnose a
branch; admission is the successful CI graph. Keep checkpoint caches and reports outside the
checkout and remove scratch build directories afterwards.

Push the reviewed branch and run `bump-admission.yml` with `workflow_dispatch`, or let a
PR gitlink change trigger it. Inspect the final job and failed jobs' logs. Reports and logs upload
on failure too. A real run confirms runner instrumentation, artifact transfer, matrix scheduling
and end-to-end checkpoint execution. Configuration jobs allow 180 minutes each; the FFI job allows
90 minutes. On free runners a full run takes about 100 minutes, nearly all of it the default
configuration's checkpoint download and execution; the FFI job with Guard Malloc takes about 40.

New public APIs land vertically, as their own change under the [charter](CHARTER.md#charter-rules):
parity mapping, independently sourced cases with a golden or differential result, ownership
classification, applicable thread/state and property or gradient evidence, and public documentation.
