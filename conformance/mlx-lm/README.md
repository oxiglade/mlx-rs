# Python mlx-lm oracle

This oracle pins `mlx-lm==0.31.3` on MLX 0.32.2, Python 3.12.14, and macOS arm64.
The interface is [SCHEMA.md](SCHEMA.md). Generation uses Python `mlx_lm` on CPU and
does not read or execute Rust code or Cargo artifacts. MLX still requires an
available Metal device during initialization on this macOS wheel.

## Environment and regeneration

From the repository root, with Python 3.12.14 available:

```sh
python3.12 -m venv conformance/.venv-mlx-lm
conformance/.venv-mlx-lm/bin/python -m pip install --require-hashes -r conformance/mlx-lm/requirements.lock
conformance/.venv-mlx-lm/bin/python conformance/mlx-lm/manifest.py
conformance/.venv-mlx-lm/bin/python conformance/mlx-lm/generate_mlx_lm.py
```

For the initial lock, or an approved dependency change, run this before the
installation step:

```sh
PYTHON=conformance/.venv-mlx-lm/bin/python sh conformance/mlx-lm/lock.sh
```

`requirements.in` contains every version from the approved
`oracle/requirements-lm.freeze`, with its NumPy entry replaced by the schema's
2.2.6 pin. `lock.sh` checks Python and architecture, downloads one compatible wheel
per pin into a temporary directory, checks dependency closure offline, and uses
`pip hash` to write `requirements.lock`. It needs package-index access. It does
not resolve newer transitive versions. Install the result with `--require-hashes`.
The separate core oracle environment at `conformance/.venv` is never modified.

`manifest.py` checks every installed dependency against the lock and input pins.
It records the environment, Transformers major version, lock digest, mlx-lm wheel
digest, and upstream commit `ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd`.
The wheel digest comes from the lock; installation with pip's hash checking is
the link between that artifact and the installed environment. Use `--lock PATH`
to select a lock and `-o PATH` to select the manifest destination.

The generator checks the same handshake before importing MLX. It creates two
complete temporary fixture trees, compares their path-and-content hashes, and
copies the six variant directories into `fixtures/` only if both trees match.
`--output-dir PATH` selects another fixture root. It does not write a manifest
or corpus catalog; run the manifest command and the separate NumPy/coordinator
workflow for those artifacts. Outputs contain no timestamps or temporary paths.

## Fixture conventions

The six variants are `llama-base`, `llama-sliding`, `llama-sharded`,
`llama-quant4`, `qwen3-base`, and `qwen3-quant4`. All use hidden size 64,
four query heads, two KV heads, head dimension 16, and vocabulary size 64.
Llama base has an untied output head; sliding Llama and Qwen3 have tied embeddings.
Sliding Llama has three layers, a five-position window, and Llama3 RoPE whose
frequencies cover low, middle, and high bands. The other variants have two layers.
The sharded fixture has exactly the base weights split across two numbered
safetensors files, with `model.safetensors.index.json` mapping their keys.

Weights use NumPy PCG64 seed 1729 and are loaded into the upstream model before
serialization. Affine variants call `mlx.nn.quantize` with group size 32 and four
bits. Every expectation is then produced from `mlx_lm.load` of the saved fixture.
Packed `quant.*.weight` expectations retain uint32; scales, biases, and all other
array expectations are float32. Both model and expectation files use the
`safetensors` library. JSON keys are sorted and non-finite JSON numbers are rejected;
filtered logprob tensors retain negative infinity in safetensors.

The canonical prompt has eight tokens. Full prefill is one model call. Chunked
prefill records all output positions from upstream `generate_step` with
`prefill_step_size` 1, 3, and 8. That function always processes the final prompt
token separately, including when the requested chunk size is 8. Generation with
zero output tokens collects prefill without its extra decode lookahead.

`decode.greedy_ids[j]` is selected from the preceding logits (the last prefill
position for j=0). `decode.step<j>.logits` is the forward result after feeding that
ID back to the model. Thus the final cache offset is T+8. Sliding caches preserve
all T positions after one-shot prefill and emit the last five positions after
decode, reordered with the upstream cache's temporal-order method. The additional
`cache` JSON object records processed offsets and half-open retained ranges.

Sampling resets `mx.random.seed(1729)` before every case and uses the CPU generation
stream. Each case records eight IDs from upstream `generate_step`. Filtered
logprobs come from its first returned logprobs, passed through upstream top-p,
min-p, and top-k functions before temperature scaling. A combined case pins filter
ordering. Upstream 0.31.3's repetition processor sees only the final prompt token
on its first sampling call, because prefill bypasses the processor. The fixture
records `processor_token_histories` to make that behavior explicit.

The local WordLevel tokenizer includes Unicode, special tokens, and textual chat
templates using Llama-3 or Qwen3 delimiters. Qwen3 renders with thinking disabled.
All three chat continuations record rendered UTF-8 bytes and tokenizer IDs.
Generation EOS values differ from config and tokenizer defaults to exercise
precedence. `decode.token_ids` and `text_deltas` record every upstream streaming
response, including the final detokenizer flush. Additional `length_case` and
`stop_case` records use an empty EOS set and the first greedy ID as EOS,
respectively, to guarantee both finish paths are covered.

## Loading errors and protection

Each error recipe is applied to a temporary fixture copy and attempted with the
Python loader. The recorded `python_result` is an exception class or `accepted`;
it is separate from the expected Rust class. Python accepts duplicate shard keys
by overwriting them, while the approved Rust boundary is strict.

| Mutation | Expected Rust class |
| --- | --- |
| Missing indexed shard | `WeightError::MissingShard` |
| Duplicate tensor across shards | `WeightError::DuplicateTensor` |
| Wrong query-projection shape | `WeightError::ShapeMismatch` |
| Unsupported RoPE type | `ConfigError::UnsupportedRope` |
| Unknown model type | `ConfigError::UnsupportedArchitecture` |
| Unsupported affine bit count | `ConfigError::UnsupportedQuantization` |

The design explicitly names `UnsupportedRope`; the other variant names above
spell out its required stable failure categories for the later Rust loader.

The [core oracle boundary](../README.md#oracle-separation) protects conformance
inputs, generators, dependency pins, manifests, fixtures, tolerance policies, and
comparators. Updates belong in a separate oracle change, reviewed independently
from inference implementation changes. Rust conformance tests consume committed
fixtures offline and do not regenerate or bless them.

## Tranche 3 regeneration and gates

The five cohorts and their keys are documented in [SCHEMA.md](SCHEMA.md#tranche-3-cohorts).
They follow position-astra.md's “Oracle-change list” and DECISIONS A/I/J.
The additive table and seeded cases use pinned Python processors; progress
comes from the pinned callback, and wrap/trim comes from the guarded cache
helper. The BOS cases record actual model-call inputs from string prompts.
Only llama-base receives the BOS post-processor; token-ID-driven forward,
sampling, decode, and chat goldens must remain byte-identical.

`text_reference.py` imports no MLX. It independently computes stop filtering
and uses pinned Python tokenizers for final decode bytes. The generator writes
its output inside both temporary trees, so the path-and-content determinism
check covers `text_cases.json` along with every model-dependent artifact.
Its `--check` mode recomputes the document without modifying it.

```sh
conformance/.venv-mlx-lm/bin/python -B conformance/mlx-lm/manifest.py
conformance/.venv-mlx-lm/bin/python -B conformance/mlx-lm/generate_mlx_lm.py
conformance/.venv-mlx-lm/bin/python -B conformance/mlx-lm/text_reference.py --check conformance/mlx-lm/fixtures/llama-base/text_cases.json
conformance/.venv-mlx-lm/bin/python -B conformance/mlx-lm/numpy_reference/run.py
conformance/.venv-mlx-lm/bin/python -B conformance/mlx-lm/coordinate.py
cargo test -p mlx-lm --features oracle-hooks --test parity -- --test-threads=1
cargo run -p xtask -- verify-lm-parity
cargo run -p xtask -- api-baseline --crate mlx-lm --check ledger/mlx-lm-api-baseline.json
cargo run -p xtask -- verify-oracle-boundary --base 919622fd56fddf0268b5db22c88a2618a0bc2a1a
```

`api-baseline` defaults to mlx-rs. Select mlx-lm with `--crate mlx-lm`, write
with `--out PATH`, or compare bytes with `--check PATH`. Check mode never
rewrites the baseline and rejects drift. The core baseline is independent of
the mlx-lm one.

The Python-side tests for the text reference and the generator contract run
under the pinned interpreter without extra packages:

```sh
conformance/.venv-mlx-lm/bin/python -B -m unittest discover -s conformance/mlx-lm -p 'test_*.py'
```
## GGUF ingestion oracle (tranche 4a)

mlx-lm 0.31.3 supplies the architecture and generation oracle and a Llama-family GGUF exporter; it does NOT supply a GGUF model loader. GGUF ingestion is qualified by Python MLX container conversion plus a reviewed reference bridge and independent NumPy decoding.

The ten tiny `gguf-{llama,qwen3}-{f32,f16,q4_0,q4_1,q8_0}` cases establish
self-consistency with our writer, mapping, orientation and quantization arithmetic.
Llama naming also matches the pinned upstream exporter. Third-party compatibility
requires the local real-checkpoint entries: TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
(`tinyllama-1.1b-chat-v1.0.Q4_0.gguf`) and ggml-org/Qwen3-0.6B-GGUF
(`Qwen3-0.6B-Q8_0.gguf`). Tiny Qwen3 fixtures do not establish compatibility with
real-world Qwen3 GGUF files. Real entries are release gates in 4b, not 4a gates.

The qualified source-format set is F32, F16, Q4_0, Q4_1 and Q8_0. The runtime sees
converted arrays, so fallback successes from BF16 or K-quants are unqualified,
not an enforceable original-format allowlist. The initial GGUF profile is dense,
biasless full attention with full-head RoPE and none/linear scaling. Embedded
tokenizer construction is excluded. Pair the file with `Tokenizer::from_dir` for
the original model's pinned sidecars; size and BOS/EOS agreement cannot establish
complete tokenizer identity. `from_file`/`from_bytes` omit directory-sidecar facts.

Generation requires the pinned Python environment on a host where MLX can import.
The sandbox can run pure tests and compile the Rust adapter. Host commands:

```sh
conformance/.venv-mlx-lm/bin/python -B conformance/mlx-lm/generate_gguf.py --base-fixtures conformance/mlx-lm/fixtures --output-dir /private/tmp/t4-gguf-a
conformance/.venv-mlx-lm/bin/python -B conformance/mlx-lm/generate_gguf.py --base-fixtures conformance/mlx-lm/fixtures --output-dir /private/tmp/t4-gguf-b
diff -qr /private/tmp/t4-gguf-a /private/tmp/t4-gguf-b
conformance/.venv-mlx-lm/bin/python -B conformance/mlx-lm/freeze_gguf.py --generated /private/tmp/t4-gguf-a --repeat /private/tmp/t4-gguf-b
cargo test --offline -p mlx-lm --features oracle-hooks --test parity gguf_ -- --test-threads=1
```

The generator checks exact Python-core/NumPy conversion before model values,
then Python–NumPy tolerance agreement before Rust comparisons. Failure retains a
diagnostic staging tree and stops; changing the tolerance is forbidden. Goldens
retain native dtypes separately from F32 comparison values. The forward cache
after eight decode inputs has `T+8` positions; public generation of eight tokens
has `T+7`. See [GGUF_BRIDGE.md](GGUF_BRIDGE.md) for mapping review, witness
transformations, mutation requirements and outstanding source qualification.

The pending core save record closes only after the existing ignored Rust writer
and independent Python checker run on the host:

```sh
GGUF_QUALIFY_OUT=/private/tmp/t4-gguf-save.gguf cargo test --offline -p mlx-tests --test gguf write_save_qualification_artifact -- --ignored --exact --test-threads=1
conformance/.venv-mlx-lm/bin/python -B conformance/qualify_gguf_save.py --input /private/tmp/t4-gguf-save.gguf --output /private/tmp/t4-gguf-save.json --producer-revision <full-revision-that-built-the-writer>
```

Record the actual producer revision and artifact hash, then copy the passing
report to `conformance/qualification/gguf-save.json`. This qualifies save
semantics, not serialized byte identity or model ingestion. Do not run the core
generator afterward: it resets that record to pending.

## Hub and CLI contracts (tranche 4b additions)

`hub_cases.json` adds 23 reviewed Rust snapshot plans. `cli_cases.json` adds two
scripted writer cases, one info JSON case, argument failures and a separately
tracked public Text capture. These cohorts are not folded into safetensors or
GGUF model counts. Hub contracts are not Python parity claims. The canonical
text capture is currently **NOT RUN**: its expected output stays null until
Python MLX runs on the host. The committed eight-ID generation expectations are
not a substitute for this nine-ID, BOS-prefixed prompt.

Pure generation, comparator qualification and source/corpus checking:

```sh
conformance/.venv-mlx-lm/bin/python -B -m unittest discover -s conformance/mlx-lm -p 'test_*.py'
python3 -B conformance/mlx-lm/generate_contracts.py --check --qualify
cargo test --offline -p xtask verify_lm_features
cargo run --offline -p xtask -- verify-lm-features
```

Capture and freeze only these additions on the pinned MLX host:

```sh
conformance/.venv-mlx-lm/bin/python -B conformance/mlx-lm/generate_contracts.py --capture --output-dir /private/tmp/t4b-contract-a --require-capture
conformance/.venv-mlx-lm/bin/python -B conformance/mlx-lm/generate_contracts.py --capture --output-dir /private/tmp/t4b-contract-b --require-capture
python3 -B conformance/mlx-lm/generate_contracts.py --freeze-from /private/tmp/t4b-contract-a --repeat /private/tmp/t4b-contract-b
python3 -B conformance/mlx-lm/generate_contracts.py --check --qualify --require-capture
```

Freezing compares both outputs, verifies generator/source identities, qualifies
comparators, preserves every old corpus digest and updates the two cohort counts.
No Rust output is accepted as an expectation source. `--check` verifies the
committed state, including an explicit pending capture; release verification must
also pass `--require-capture`. The existing manifest is source-pinned in the new
provenance rather than regenerated with unrelated fixture changes.

The compiled gate creates a temporary external Cargo workspace with only the
requested mlx-lm features. It checks all reviewed root exports and the field,
variant and call inventory, plus forbidden Hub/hook imports. It prints a JSON
report and returns nonzero for any missing required declaration, feature leak or
build failure. It complements `api-baseline`, whose source inventory does not
evaluate cfg. See SCHEMA.md for exact coverage and failure classes.

At the 2ef76383 base, the final gate still requires the Hub owner's additions:
`#[cfg(feature = "hf-hub")] pub struct HubProvenance` with public String fields
`repo`, `requested_revision`, `resolved_revision`, Debug/Clone/PartialEq/Eq and
`#[non_exhaustive]`; its gated root re-export; and
`Model::hub_provenance(&self) -> Option<&HubProvenance>`. HubError also needs
`InvalidRepository(String)`, `MissingFile { repo: String, revision: String,
filename: String }`, `UnsafePath { path: PathBuf }`, `RevisionMismatch {
expected: String, actual: String }`, `SnapshotIntegrity { path: PathBuf,
reason: String }`, and `CacheDirectoryUnavailable`. This oracle item does not
edit those runtime declarations.

The Hub/CLI owners must still run their production adapters against these cases,
including temporary-cache link/integrity tests, anonymous-client construction,
public offline success/error preservation, real generate/info processes and all
four writer fault recipes. No runtime or CLI adapter is added by this ownership
slice. The parity ledger and global source manifest need the integration owner's
corresponding cohort entries; this slice updates only the owned corpus/provenance.

## Local real checkpoints

`real_checkpoints.py` writes independent Python expectations. The Rust
`real_checkpoints` test verifies the same local files and compares six cases on
CPU and Metal: Llama-3.2-1B BF16/4-bit, Qwen3-0.6B BF16/4-bit, TinyLlama Q4_0
GGUF, and Qwen3 Q8_0 GGUF. It checks the canonical tokenizer round trip, exact
original-string prompt IDs, every resolved config field, 32-token generation,
and length completion. CPU requires exact equality for all 32 greedy IDs.

On Metal, greedy equality is claimed only while the choice is not a tie at the
model's precision (DECISIONS Q). Python records 32 `greedy_steps` alongside
`greedy_ids` for every case/device. Each step contains `top1_id`, `top2_id`,
`top1_logit`, `gap` (top-1 minus top-2), and `dtype` (`bfloat16`, `float16`, or
`float32`). These are raw model logits before log-softmax; `dtype` is the actual
logits dtype, including mixed-precision promotion in GGUF models, not the packed
weight bit count. The recorder observes upstream's logits without changing them
or its greedy sampler and excludes the final unused lookahead from the report.

Rust computes one ULP at `abs(top1_logit)` in that dtype: spacing is
`2^(max(floor(log2(abs(top1_logit))), minimum_normal_exponent) - fraction_bits)`,
with minimum subnormal spacing at zero. BF16, F16, and F32 have 7, 10, and 23
fraction bits respectively. On Metal, every ID before the first gap at or below
one ULP must match exactly. At that first tie, Rust must still choose one of the
recorded top-1 or top-2 IDs, even if it agrees with Python. The result says
`tie-limited at step N, gap g`, with zero-based steps. Comparison stops there;
both generators still must complete 32 tokens, but no equality is claimed for
the remaining suffix. A mismatch above the threshold or a choice outside those
two candidates fails. CPU has no tie allowance.

The measured Llama-1B BF16 Metal divergence was at step 18: Python chose 832 at
20.875 and Rust chose 264, whose Python logit was 20.750. The 0.125 gap is one
BF16 ULP at that magnitude. Adjacent representable scores let reduction order
determine the winner; requiring the same winner would qualify an arbitrary
tie-break. This exception does not change numeric tolerances or the benchmark
contract.

After shared manifest/report admission, both harnesses attempt all six cases on
both devices and print one PASS or FAIL per case/device. Rust catches case
errors and unwinding panics, reports the remaining cases, then fails if any did.
Python retains exceptions as `completed: false` with an `error` message, writes
the entire report, and exits nonzero after all 12 attempts if any failed.

Neither real-checkpoint execution nor benchmarking runs in CI. Both reject the
presence of common CI markers, even when their value is `false`. With
`MLX_LM_REAL_CHECKPOINTS` absent, the Rust test emits exactly one
`NOT RUN: local real-checkpoint manifest not supplied` line. Supplying a broken
manifest, missing entry/file/hash/report, or an unavailable device fails the run.
The pure harness tests require neither checkpoints nor Python MLX.

The existing asset manifest remains schema version 1. `--prepare` writes a new
manifest, retaining `entries`, each exact repository, full 40-character revision,
absolute `local_directory`, `files` SHA-256 map, and `bytes`. Additions are:

| Field | Meaning |
| --- | --- |
| `devices` | Exactly `["cpu", "metal"]`, required for release qualification. |
| `python_environment` | Full identity returned by the existing pinned `manifest.check_environment`, including package versions, upstream commit, and lock/wheel hashes. |
| `cases.<case>.canonical_text` | Original text, without a chat template. Preparation uses `The capital of France is`. |
| `cases.<case>.prompt_ids` | Explicit IDs from the pinned tokenizer and the original-string BOS rule. Encoding without specials and decoded text are also recorded in the report. |
| `cases.<case>.tokenizer_entry` | Same entry for safetensors; `gguf-tinyllama-tokenizer` or `gguf-qwen3-tokenizer` for GGUF. |
| `cases.<case>.max_tokens`, `prefill_chunk_size`, `stop_token_ids` | Exactly 32 tokens, a positive shared chunk size (prepared as 128), and an empty exact stop set. All caches are full. |
| `expected_report.path`, `expected_report.sha256` | Absolute Python-report path and SHA-256. Preparation leaves the hash null; binding fills it after independent Python generation. Rust requires both. |
| `benchmark` | Fixed case, source text, explicit 128-ID prefix, `prompt_rule`, tokenizer hash, 256-token request, chunk size, and empty stop set. |

The eight repository identities and two GGUF filenames are fixed by section 5.8.
All listed files are hashed before loading. Loader-discoverable sidecars and
shards must also be listed; an extra unlisted JSON, safetensors, GGUF, tokenizer
model, text, Jinja, or Python file fails admission. Hugging Face blob symlinks are
allowed for files. Relative paths cannot contain traversal. Rust uses the macOS
`shasum -a 256` tool, without activating the library's optional Hub dependency.
No harness downloads, resolves a remote ref, saves converted weights, or reads
Rust output to construct a Python expectation. Keep the assets unchanged during
verification; this is a reproducibility check, not a concurrent-write sandbox.

Safetensors expectations use pinned upstream loading. GGUF expectations use
`mx.load` on the actual hashed container followed by the reviewed
`gguf_bridge.build` architecture adapter. The harness checks actual GGUF v3 tensor
types, records tensor descriptors and architecture metadata, and checks the
representative full-attention/full-head-RoPE profile. Missing explicit head length
is derived from agreeing Q/K/V row counts before calling the bridge. Each GGUF
result separately records the container hash, bridge hash, tokenizer entry,
resolved metadata config, head derivation, and per-group quantization. Original
unquantized model weights never supply those expectations. The real files provide
third-party compatibility evidence; tiny fixtures alone establish mapping and
arithmetic self-consistency. Original types outside F32/F16/Q4_0/Q4_1/Q8_0 are
recorded as `unqualified_original_type_fallbacks`, not rejected by a source-type
allowlist: admission follows the core-converted layouts and strict bridge load.
The downloaded TinyLlama Q4_0 file includes a type-14 output tensor; its fallback
does not extend the qualified original-format table.

Run from the repository root on an idle MLX host. These commands create new files;
choose a fresh directory for every attempt. They use tools already on PATH and
the pinned Python environment, with no environment installation step:

```sh
export T4_RUN="$(mktemp -d /private/tmp/mlx-lm-t4b.XXXXXX)"
export T4_ASSETS=/Users/ci/hub/assets/mlx-lm-checkpoints/real-checkpoints.json
export T4_PYTHON="$PWD/conformance/.venv-mlx-lm/bin/python"

"$T4_PYTHON" -B conformance/mlx-lm/real_checkpoints.py \
  --manifest "$T4_ASSETS" --prepare \
  --expected-report "$T4_RUN/python-real-report.json" \
  --output "$T4_RUN/prepared.json"

"$T4_PYTHON" -B conformance/mlx-lm/real_checkpoints.py \
  --manifest "$T4_RUN/prepared.json" --devices cpu metal \
  --output "$T4_RUN/python-real-report.json"

"$T4_PYTHON" -B conformance/mlx-lm/real_checkpoints.py \
  --manifest "$T4_RUN/prepared.json" \
  --bind-report "$T4_RUN/python-real-report.json" \
  --output "$T4_RUN/real-checkpoints.json"

MLX_LM_REAL_CHECKPOINTS="$T4_RUN/real-checkpoints.json" \
  cargo test --offline -p mlx-lm --release --test real_checkpoints \
  -- --test-threads=1 --nocapture
```

Binding checks the report's manifest contract and hashes the completed report.
That contract excludes `expected_report` to avoid a circular hash. Binding does
not claim Rust parity; only the subsequent test does. Retain all three manifests
and the report with release evidence. Review the prepared IDs before timing.

## Ruling-C benchmark

`benchmark.py` coordinates sequential Rust/Python processes, alternating AB/BA
with recorded order seed 41005. Each process loads and evaluates the same
Qwen3-0.6B-4bit checkpoint, performs three complete warm-ups with fresh caches,
then one measured request. Build, hashing, model loading, and output writes are
outside the measured request. CPU and Metal are explicit, including Python's
`generation_stream`. Controlled workers leave allocator and wired-memory policy
at their defaults. Do not run model processes, Metal work, or leak probes alongside
the experiment.

Both workers feed the manifest's 128 IDs directly and request 256 greedy IDs with
full caches, equal prefill chunks, no reuse, penalties, filters, EOS stops, or
text stops. Rust uses `StopTokenPolicy::Exact(Vec::new())`. The source text is
encoded without special tokens and its first 128 IDs must equal the frozen
prompt in each worker. All warm-up and measured greedy IDs must agree within and
across languages before that pair contributes any throughput statistic.

The primary Python path drives the unmodified pinned `generate_step` and one
stateful streaming detokenizer through finalization. Rust drives public
`Generation` into an in-memory sink. Each sample retains start, every token, and
drained completion timestamps from a monotonic clock, IDs, text, and forward
count. Rates are computed by the coordinator from those boundaries:

- Steady decode: `(N - 1) / (t_N - t_1)`, the primary figure.
- Prefill/first-token latency: `t_1 - start`.
- Whole-request rate: `N / (t_N - start)`.
- Teardown/drain: `completion - t_N`.

Python's final unused forward is counted and its generation stream is drained
before another process starts. With chunk size 128, Rust performs 257 forwards
and Python performs 258. Forward counts include prefill. A separate unmodified
`stream_generate` measurement on Metal disables EOS and retains upstream
`wired_limit`; its policy is recorded. Its wrapper-reported `generation_tps` is
not used. No wrapper measurement is required on CPU.

The coordinator preserves every warm-up and measured sample, stdout/stderr, exact
command, process order, and worker-report hash in `<output>.samples/`. It never
overwrites reports, including failed reports. Per-device summaries contain all
paired Rust/Python rate ratios, their median, a 10,000-resample paired percentile
bootstrap 95% interval (seed 41006), MAD, range, and per-language rate dispersion.
The gap is `1 - rust_tps/python_tps`; its confidence bounds reverse the ratio
bounds. An interval straddling 5% automatically extends that device to 30 pairs.
If it still straddles, the result is inconclusive. Slow samples are retained.

Before running, write `$T4_RUN/context.json` with the actual build identities:

```json
{
  "native_mlx": {
    "version": "0.32.2",
    "rust_source_commit": "FULL_NATIVE_MLX_COMMIT",
    "python_source_commit": "SAME_FULL_NATIVE_MLX_COMMIT",
    "rust_build": "path and identity of the native Rust-side build",
    "python_build": "identity of the pinned native MLX wheel/build"
  },
  "rust_release_flags": "actual release profile, RUSTFLAGS and native build flags",
  "idle_machine_confirmed": true
}
```

These are launcher attestations: version strings alone cannot prove matching
native builds. Replace the example text with the actual source commits and build
records; validation requires identical full native source commits. The report
also captures the Rust executable hash/path, source commit/diff and harness-file
hashes, Cargo.lock hash, rustc identity, pinned Python environment, OS, hardware,
memory, and `pmset` power mode. Compare the two native build configurations before
running; retain their records alongside `context.json`.

```sh
cargo build --offline -p mlx-lm --release --example benchmark

"$T4_PYTHON" -B conformance/mlx-lm/benchmark.py \
  --manifest "$T4_RUN/real-checkpoints.json" --case qwen3-06b-4bit \
  --rust-runner "${CARGO_TARGET_DIR:-target}/release/examples/benchmark" \
  --run-context "$T4_RUN/context.json" --devices cpu metal \
  --prompt-tokens 128 --max-tokens 256 --warmups 3 --pairs 10 \
  --output "$T4_RUN/c-baseline.json"

"$T4_PYTHON" -B conformance/mlx-lm/benchmark.py \
  --manifest "$T4_RUN/real-checkpoints.json" --case qwen3-06b-4bit \
  --rust-runner "${CARGO_TARGET_DIR:-target}/release/examples/benchmark" \
  --run-context "$T4_RUN/context.json" --devices cpu metal \
  --pairs 10 --python-no-lookahead --baseline "$T4_RUN/c-baseline.json" \
  --output "$T4_RUN/c-no-lookahead.json"
```

The ablation makes an in-memory copy of the pinned `generate_step`, relocates the
exact next-step construction/submission block below the current yield, and
records both source hashes. It preserves sampling, logits, stopping, cache work,
and the final extra forward; it does not merely replace `async_eval` with `eval`.
It leaves the installation and the unmodified Metal wrapper untouched. Ablated
IDs must equal the independent baseline IDs. It is never an oracle golden.
Baseline and ablation must have equal per-device pair counts for the paired
counterfactual; if an extension differs, collect fresh 30-pair reports for both.

The ablation report gives the paired added Python decode time, its bootstrap
interval, and the fraction of the baseline Rust-minus-Python excess explained.
At least half the excess plus a positive interval is necessary. The launcher
must additionally capture an untimed scheduling profile subset covering host
synchronization, device idle gaps, graph construction, cache copying, sampling
validation, and detokenization, and attach those evidence paths. Aggregate timing
alone cannot establish that mechanism. These harnesses do not automate platform
profiling or implement a Rust pipeline.

The harness can close C as `C closed: no qualifying gap` only when both devices
meet the bound. A qualifying gap, ablation result, or inconclusive interval keeps
C open. The other allowed final rulings are `C closed: gap attributed elsewhere;
no pipeline` and `C closed: qualified pipeline admitted`; both require reviewed
external evidence. A pipeline candidate must recover at least half the
attributable excess with a positive paired interval, avoid a greater-than-5%
regression on the other device, and pass the paper's behavior qualification.

Pure checks and compilation, without model execution:

```sh
python3 -B conformance/mlx-lm/real_checkpoints.py --self-test
python3 -B conformance/mlx-lm/benchmark.py --self-test
cargo test --offline -p mlx-lm --test real_checkpoints --no-run
cargo build --offline -p mlx-lm --example benchmark
cargo test --offline -p mlx-lm --test real_checkpoints pure_ -- --test-threads=1
# With MLX_LM_REAL_CHECKPOINTS unset, exercise the single explicit NOT RUN result:
env -u MLX_LM_REAL_CHECKPOINTS cargo test --offline -p mlx-lm \
  --test real_checkpoints real_checkpoints -- --exact --nocapture
```

The local release runs remain unexecuted until the launcher records the Python
matrix, Rust matrix, baseline, unmodified Metal user path, and any required
ablation/profile evidence. Compilation and pure checks do not qualify real model
parity or performance.
