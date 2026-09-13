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
