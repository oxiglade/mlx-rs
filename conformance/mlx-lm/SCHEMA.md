# mlx-lm oracle fixtures: layout and expectation schema (tranche 1 contract)

This file is the interface between the three tranche-1 work items (Python mlx_lm generator,
NumPy reference generator, Rust comparator). Changing it is an `oracle-change:` commit.

## Environment
- Oracle venv: `conformance/.venv-mlx-lm` built from `conformance/mlx-lm/requirements.lock`
  (pip hashes; python 3.12.14; mlx-lm==0.31.3; mlx==0.32.2; mlx-metal==0.32.2; numpy==2.2.6;
  transformers 5.x). `manifest.json` records the handshake (arch=arm64, versions, upstream
  commit ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd, sha256 of the mlx-lm wheel).
- The core oracle venv `conformance/.venv` is never touched.

## Fixture directories
`conformance/mlx-lm/fixtures/<model>-<variant>/` where `<model>` in {llama, qwen3} and
`<variant>` in {base, quant4} (quant4 = affine, group_size 32, bits 4; Llama base also carries
a sliding-window layer pattern and llama3 rope in the `llama-sliding` variant). Each directory:
- `config.json` — HF-style config as mlx_lm expects (tiny: hidden<=64, layers 2..3, vocab<=64).
- `model.safetensors` (+ `model.safetensors.index.json` for the one `llama-sharded` variant).
- `tokenizer.json`, `tokenizer_config.json` (chat_template included), `generation_config.json`
  (eos_token_id present to exercise precedence).
- `expectations.json` — scalars, ids, strings, error cases, provenance (see below).
- `expectations.safetensors` — every array expectation, f32 unless stated, keys below.
- `inputs.json` — the prompt strings, token id lists, seeds, sampler cases, chunk sizes used.

## Shared math pins
- RoPE llama3 smoothing follows mlx_lm 0.31.3 `rope_utils.py` exactly (angle = pos / f_i with
  f_i = theta^(2i/d); low band f_i *= factor; mid band f_i /= ((1-s)/factor + s)); both
  generators derive from that file's math, the NumPy one without importing it.
- Affine dequant per /Users/ci/hub/scratch/mlx-lm/t1/affine-packing.md (recorded in-repo as
  conformance/mlx-lm/numpy_reference/PACKING.md by the NumPy item).

## Array keys in expectations.safetensors
- `prefill.full.logits` — `[1, T, V]` for the canonical prompt, one-shot prefill.
- `prefill.chunk<N>.logits` — same tensor produced with prefill chunk size N, for N in
  {1, 3, T}; N=3 must be a non-divisor of T.
- `cache.after_prefill.layer<i>.keys` / `.values` — logical K/V `[1, kv_heads, T, head_dim]`
  after a ONE-SHOT prefill, in temporal (position) order. Sliding-window layers emit all T
  positions here too: mlx's RotatingKVCache keeps the whole first concat and only trims on
  later updates, and both oracles pin that exact state rather than an idealized window.
- `decode.step<j>.logits` — `[1, V]` for greedy decode steps j in 0..8.
- `cache.after_decode.layer<i>.keys` / `.values` — logical K/V after the 8 decode steps, in
  temporal order: full layers all T+8 positions; sliding layers (window W) the last
  min(W, T+8) positions. mlx stores the rotating buffer in rotated memory order; the Python
  generator must emit temporal order (reorder by the cache's rotation index), and the NumPy
  reference keeps a logical temporal cache and emits its tail.
- `sampling.<case>.filtered_logprobs` — `[V]` logprobs after the case's filters (top_p, min_p,
  top_k) at the first decode position, before temperature division; -inf for filtered entries.
- `quant.<param>` — for quant4 variants: the packed weight/scales/biases arrays of two named
  parameters, to pin layout.

## expectations.json (JSON, sorted keys, no NaN)
{
  "schema_version": 1,
  "provenance": { "python": "3.12.14", "mlx_lm": "0.31.3", "mlx": "0.32.2", "numpy": "2.2.6",
                  "seed": <int>, "upstream_commit": "ed1fca4c…", "generator": "<file>@<sha256>" },
  "config": { "resolved": { hidden_size, layer_count, intermediate_size, attention_heads,
               kv_heads, head_dim, vocabulary_size, rms_norm_epsilon, rope: {dimensions, theta,
               traditional, scaling}, attention_kinds: ["full"|{"sliding": W}, …],
               tie_word_embeddings, quantization: null|{group_size, bits} } },
  "tokenizer": { "encodings": { "<prompt name>": [ids…] }, "decodings": { "<name>": "text" },
                 "eos_tokens": [ids…], "eos_source": "generation_config"|"config"|"tokenizer" },
  "chat": { "<case>": { "messages": [...], "continuation": "closed"|"start_assistant"|"continue_last",
             "rendered_utf8_hex": "…", "token_ids": [ … ] } },
  "prefill": { "prompt": "<name>", "token_ids": [ … ], "T": <int> },
  "decode": { "greedy_ids": [8 ids], "text_deltas": ["…", …], "finish_reason": "length"|"stop",
              "stop_token": <id|null> },
  "sampling": { "<case>": { "options": { temperature, top_p, top_k, min_p, min_tokens_to_keep,
                 repetition_penalty, repetition_context_size }, "seed": <int>,
                 "cpu_ids": [ … 8 ids … ] } },
  "errors": { "<case>": { "mutation": "<what was changed in a copy of the fixture>",
              "expected_rust_error": "ConfigError::…|WeightError::…|LoadError::…" } },
  "tolerances": { "logits": {"atol": 2e-4, "rtol": 2e-4}, "cache": {"atol": 2e-4, "rtol": 2e-4},
                  "logprobs": {"atol": 1e-4, "rtol": 1e-4} }
}

## NumPy reference output
`conformance/mlx-lm/numpy_reference/out/<model>-<variant>.safetensors` with the SAME array keys
for prefill.full.logits, cache.after_prefill.*, decode.step<j>.logits, produced from the committed
fixture inputs by an independent f32 NumPy implementation (no mlx import). The coordinator
`conformance/mlx-lm/coordinate.py` asserts agreement within `tolerances` and writes
`conformance/mlx-lm/corpus.json` (sha256 of every committed file + policies). Two full runs must
produce identical trees.

## Comparator failure classes (Rust side, one class per failure)
config, tokenizer, chat, shape, dtype, value, cache_offset, cache_range, cache_value,
sampling_support, sampled_id, text_delta, finish_reason, error_class, output_count,
processor, text_stop, progress, cache_trim.

The protected parity adapter (`mlx-lm/tests/parity/**`) produces observations for every class
except `processor`, `text_stop` and `cache_trim`; those three are consumed from the same
goldens by the crate's own tests (`sampling/tests.rs`, `tokenizer/text_tests.rs`,
`cache/tests.rs`), while their mutation qualification stays in `tests/parity/mutations.rs`.

## Tranche 3 cohorts

The normative contracts are position-astra.md, “Processor and sampler contract”,
“EOS, length, stop strings and stable text”, and “Oracle-change list / Minimal new
golden cohorts”, with DECISIONS A/I/J overriding the prompt rule. No tolerance
changes accompany these additions.

Only llama-base adds the following expectation keys:

- `sampling.presence_penalty`, `sampling.frequency_penalty`, and
  `sampling.penalties_combined`, with matching `inputs.sampling` recipes and
  `sampling.<case>.filtered_logprobs` tensors. They use temperature 0.7, disabled
  support filters, seed 1729, eight CPU IDs, and context 3. Enabled coefficients
  are repetition 1.3, presence 0.6, and frequency 0.4. Options add
  `presence_penalty`, `presence_context_size`, `frequency_penalty`, and
  `frequency_context_size` as applicable. `processor_token_histories` records
  each of the eight calls before per-processor cropping; history starts with the
  final prompt token and appends previously accepted generated IDs.
- `processing.<case>` contains `options`, `input_logits`, and `histories`.
  Cases are `presence_negative`, `presence_zero`, `presence_positive`,
  `frequency_negative`, `frequency_zero`, `frequency_positive`, `repetition`,
  and `combined`. The additive coefficients are -0.5, 0, 0.5; repetition is
  1.3; combined uses 1.3/0.6/0.4. All windows are 3. Each corresponding
  `processing.<case>.logits` f32 tensor has shape `[7, 5]`, with one row per
  nonempty prefix of `[1,2,1,2,4,1,3]`, processed from `[0,2,-3,1,-0.5]` by
  pinned `make_logits_processors`. The existing logits tolerance applies.
- `prefill.progress.<case>` contains `token_ids`, `ceiling`, and ordered
  `[processed,total]` `pairs`. Cases are `eight_chunk1`, `eight_chunk3`,
  `eight_chunk8`, and `one_chunk1`. Callback capture uses `max_tokens=0` and
  is cross-checked with an independent integer schedule. Eight-token counts
  are `[0,1,2,3,4,5,6,7,8]`, `[0,3,6,7,8]`, and `[0,7,8]`; one token gives
  `[0,1]`. Initial and final pairs occur exactly once.
- `tokenizer.prompt_encoding` contains `canonical`, `canonical_with_bos`,
  and `whitespace_before_bos` ID lists captured at the model call from
  `stream_generate`. The base tokenizer has a TemplateProcessing post-processor
  prepending configured BOS ID 4. The first two cases forward `[4,12,...]`;
  the third forwards `[4,4,12,...]`. The decision uses the original string's
  prefix, without trimming, encoding first, or decoding the first ID.
  `tokenizer.encodings.special_with_defaults` and the corresponding input
  encoding gain ID 4. All token-ID-driven arrays, sampling, decode, and chat
  expectations remain unchanged.

Only llama-sliding adds `cache.trim_after_wrap`. Its `capacity`, `keep`,
`trim_requested`, and `trim_return` are 5, 2, 2, and 0. Stages `before_trim`,
`after_trim`, and `after_append` record `offset`, `rotation_index`, `can_trim`,
`temporal_positions`, `retained_prefix`, and `retained_tail`. Prefix and tail
are position lists, not a contiguous retained range. Temporal positions are
`[0,1,7,8,9]` before/after trim and `[0,1,8,9,10]` after appending position 10.
For each stage, `cache.trim_after_wrap.<stage>.<kind>` stores f32 `[1,1,5,1]`
for kinds `raw_keys`, `raw_values`, `temporal_keys`, and `temporal_values`.
Keys encode position p; values encode `100 + 3*p`. These synthetic tensors
compare with exact bits. The guarded helper `trim_prompt_cache` is called;
calling the unguarded cache object's `trim` would change wrapped state.

### text_cases.json

This independent llama-base artifact has `schema_version: 1` and `provenance`
with the pinned `tokenizers` version and `text_reference.py@sha256`.
`stop_strings.<case>` contains input `stops`, `deltas`, `final_delta`, the
consumed `events` (`text`, `held`, `stopped`), `finish_flush`, `stopped`, and
`finish_reason`. A null flush means a match already terminated the stream;
an empty string means finishing ran and emitted nothing. Events stop at the
first complete match; later input deltas remain recorded but are not consumed.
Matching is exact Unicode content/UTF-8, without normalization. An unmatched
held suffix is released only after processing the final delta. With stops
`abc` and `b`, `abc` emits nothing, while `ab`, `c` emits `a` and stops in
its first event.

`decoders.<case>` contains a standalone `tokenizer` definition, `token_ids`,
`decoded_utf8_hex` from pinned Python tokenizers, `emitted_prefix`,
`preserves_prefix`, `finish_flush`, and `expected_error`. Named cases reproduce
ByteFallback complete/incomplete/empty/rewrite and ByteLevel incomplete-tail
inputs from tokenizer/tests.rs. Emitted prefixes are the explicitly admitted
Rust traces; final decode bytes are independently computed. The ByteFallback
rewrite has `DecodeStreamError::InvalidPrefix` and a null flush.
`errors.empty_stop` records `stops: [""]` and `expected_error: EmptyStopString`.

### Comparison and qualification

New classes are `processor` (raw table and processor histories), `text_stop`
(the text artifact excluding provenance), `progress` (including counts), and
`cache_trim` (synthetic wrapped-cache metadata and tensors). BOS encoding stays
in `tokenizer`. Existing classes and tolerance policies are unchanged.
Mutations qualify presence-as-frequency, frequency deduplication, expired
context retention, reversed processor order, partial-stop leakage, match
emission, missing held-suffix flush, omitted/duplicated final progress, changed
wrapped trim state, and first-ID/decoded-first-ID BOS shortcuts.
## GGUF cohort, schema 1 with `source: "gguf"`

GGUF fixtures use a separate document reader. The six safetensors fixtures and
their policies retain the schema below unchanged. Each GGUF directory contains:

| File | Contract |
| --- | --- |
| `model.gguf` | Standard-library writer output, never Rust save output. |
| `inputs.json` | Existing base prompts, IDs, seed, eight steps, chunks 1/3/8. |
| `expectations.json` | Source identity, complete resolved config, keyed matrix quantization, metadata values/types, tokenizer references, cache states, IDs, native dtypes, fixed tolerance, effective mutation records. |
| `expectations.safetensors` | F32 comparison payloads: full and chunked prefill, eight decode logits, K/V after prefill and eight fed decode IDs. |
| `converted.safetensors` | Exact external key set and native F32/F16/U32 arrays from Python core. |
| `mutations.safetensors` | Actual wrong-bridge numerical outputs, keyed `<mutation>::<original_tensor_key>`. |
| `reference-agreement.json` | Python–NumPy per-key and per-fixture measured errors, bound ratios, and separate exact greedy-ID agreement, measured before Rust. Also frozen under `provenance.reference_agreement` in `expectations.json`. |

`gguf.tokenizer_files` references the existing base's four sidecars by relative
path and SHA-256. No tokenizer assets are duplicated. `config.bridge` is the
explicit upstream architecture config derived from GGUF. `config.resolved`
uses the existing dimension/RoPE schema; `config.quantization` separately maps
every matrix to `{group_size:32,bits:4|8,mode:"affine"}` or `false` when any
matrix is quantized. The Rust default is the first quantized canonical group;
all matrix overrides are explicit. Model type, max positions and both bias flags
are also checked through public Config. `gguf.metadata_types` stores original
GGUF metadata type IDs; Rust checks the corresponding converted kind, rank,
dtype and value. `gguf.tensor_types` records original types as provenance, not
a Rust-observable allowlist.

Each architecture has five storage fixtures. F32 is untied/unscaled; F16 is
tied/linear; Q4_0 is tied/unscaled; Q4_1 is untied/linear with layer-0 V in F16
and output in Q8_0; Q8_0 is untied/unscaled. Norms are F32. The deterministic
witness transformations and explicit row mapping are reviewed in GGUF_BRIDGE.md.

The three fixed logits/cache policies are:

| Policy | Fixture storage | atol | rtol |
| --- | --- | --- | --- |
| `f32-v1` | F32 | 2e-4 | 2e-4 |
| `gguf-f16-v1` | F16 | 5e-3 | 5e-3 |
| `gguf-affine-v1` | Q4_0, Q4_1, Q8_0 only | 2e-2 | 2e-3 |

F32 and F16 qualified at their existing numbers. Ruling K calibrates the affine
policy from the launcher's Python-versus-NumPy probe of `gguf-llama-q4_0`:

| Key | max \|x\| | max absolute error | error relative to peak |
| --- | --- | --- | --- |
| `cache.after_prefill.layer0.keys` | 3.51 | 0.00254 | 7.2e-04 |
| `cache.after_prefill.layer1.keys` | 3.42 | 0.00703 | 2.1e-03 |
| `prefill.full.logits` | 3.03 | 0.00893 | 3.0e-03 |
| `decode.step7.logits` | 3.40 | 0.00930 | 2.7e-03 |

Both references chose exactly `[0, 36, 12, 12, 40, 28, 52, 52]`. The smoothly
growing discrepancy reflects accumulated F16 noise from MLX's fused quantized
matmul versus NumPy's dequantize-then-F32 matmul. The affine absolute bound is
about twice the measured worst error, calibrated before Rust observations.

`provenance.reference_agreement.keys` records `max_abs`, elementwise `max_rel`
(`max(abs(Python - NumPy) / max(abs(Python), float32.tiny))`), `rel_to_peak`
(`max_abs / max(max(abs(Python)), float32.tiny)`), and `margin`
(`max(abs(Python - NumPy) / (atol + rtol * abs(Python)))`) for every key.
The fixture-level `max_abs`, `max_rel`, and `margin` are maxima across keys.
These measurements are provenance, never expected outputs or tolerance inputs.
The report also records both greedy-ID lists and `greedy_ids_equal`; a mismatch
stops generation independently of numerical agreement.

`native_dtypes` maps every numerical output key to
its Python native dtype; Rust compares this exactly before casting observations
to F32. Only numerical comparisons use tolerance. Names, integers, shapes,
metadata kinds, converted affine slots, tokenizer IDs and greedy IDs are exact.
Before forward comparison, independently derived packed U32 words and F16 affine
slots must equal core observations bit for bit. NumPy dequantizes these slots as
`F16(F16(scale*q)+bias)`, then promotes the result to F32 for its model forward.
Reference disagreement stops the oracle item and cannot widen a tolerance.

Required effective mutations are Llama inverse omission, Qwen3 wrong
permutation, gate/up swap, Qwen3 Q/K norm swap, tied affine-output bypass, and
mixed output wrong bits. Each applicable mutation must fail numerical comparison
under the fixture's policy. For every affine value mutation, qualification requires
a margin of at least 10x: `max(abs(original - mutated) / (atol + rtol * abs(original)))`
on the saved witness tensor. Q/K norm swaps retain a cache witness. The generator
prints and freezes the actual `margin` and witness `max_abs` per mutation; the
Rust qualification recomputes and reports the margin from the saved tensors.
A margin below 10x stops qualification and requires redesigned witness values
before freezing. Exact converted-slot and structural mutations retain their
zero-tolerance requirements. Every fixture kills at least one value mutation.
The Rust comparator consumes the actual mutated output and checks the recorded
`value` or `cache_value` class. Offset/range, dtype, packed-word, tokenizer,
metadata and error-expectation corruption qualify exact comparison separately.
Malformed data recipes and typed fields live in `gguf_cases.json`; they do not
multiply the model files. Q5_0 is a pre-Model opaque core exception.
Each case's `materialized_sha256` is the lowercase, 64-digit SHA-256 of the
complete file produced by `gguf_recipes.materialize`. For `load_core` it hashes
the existing core fixture unchanged. `generate_gguf.py` freezes these hashes;
`--recipes-only` refreshes them from `--base-fixtures` without Python MLX.
Rust materializes the recipes from committed base bytes, checks the frozen hash
before loading, and then checks the exact typed error. Its tests run offline
without Python. Metadata types and tensor operations remain in the JSON contract.

Tiny fixtures prove reader/writer self-consistency plus mapping, orientation and
quantization arithmetic. They do not prove third-party GGUF compatibility.
Llama naming is additionally qualified against its pinned upstream exporter.
Real compatibility evidence is local-only: TinyLlama Q4_0 and
ggml-org/Qwen3-0.6B-GGUF Q8_0, paired with pinned original tokenizer assets.
Those release entries remain distinct from the tiny-fixture evidence.

## Tranche 4b Hub and CLI additions

`hub_contract_v1` and `cli_contract_v1` are separate, version-1 cohorts in
`corpus.json`. They add no safetensors model cases or numerical tolerances.
`generate_contracts.py` records SHA-256 identities of its source, companion tests,
the existing pinned manifest/lock and fixture inputs in each cohort's provenance.
The old manifest and old fixture bytes are preserved. The corpus also records
named comparator mutations and their observed failure classes.

### Hub logical snapshot plans

`hub_cases.json` contains reviewed Rust product contracts. These are **not Python
parity claims**. Commit strings identify synthetic transport responses, not remote
repositories. Each case provides the request, returned full commit, sorted sibling
set, selected files, recorded absences, fixture basis, cache setup and mutations,
and the exact expected transport log, receipt, provenance or typed error.
`setup.refs` is the logical ref map; an immutable SHA needs no entry. Online
requests call `info` once, then download by the returned SHA. The reviewed order
fetches the index first when present, then the remaining selected names in sorted
order. An invalid index stops before downloading its referenced shards; its
`selected_files` records only the index selected so far.

Materializers copy only selected assets from `fixtures/<fixture>/`, create the
same-repository blob links, and apply `setup.mutations` in listed order at their
named boundary. `copy_fixture_file` copies the named asset from its fixture;
`set_json` uses sorted, two-space-indented UTF-8 JSON with a terminal newline,
as `generate_contracts.serialized` does. `flip_byte` XORs the byte at `offset`
with 1, without changing size. `replace_bytes` uses exactly the supplied UTF-8.
`before_receipt` mutations precede receipt creation; `before_validation` mutations
follow it. Download mutations alter the result of the indicated call. A transport
error is an injected `HubError::Api`; no dependency message bytes are frozen.

`same_repository_blob_links` means each selected basename points to a blob in
this repository's canonical `blobs` directory. The negative link operations
instead target another repository's blob, another snapshot's regular file, or a
missing blob. `link_snapshot_directory_outside_cache` replaces the snapshot
directory with a symlink to a directory outside the cache. `remove_file` removes
the snapshot link. Tests use fresh temporary roots; none of these operations
address the user's actual cache. The `snapshot/` path symbol expands to the
case's canonical snapshot. Error observations normalize only this temporary-root
prefix. Other paths, including the relative shard names in `ConflictingIndex`,
remain exact. String tuple payloads use `fields.value`; structured errors use
their public field names. Duplicate and traversing index entries preserve the
shared parser's `HubError::Load(LoadError::Weights(ConflictingIndex))` chain.

Receipts contain sorted selected relative paths, byte sizes, SHA-256 hashes,
weight mode and explicit absences. Single-file snapshots also record index
absence because a new index changes local discovery. Receipt serialization is a
logical contract; the private on-disk serializer need not use this JSON layout.
`SnapshotIntegrity.reason` is a nonempty local diagnostic, excluded from the
normalized typed observation; it is not a dependency-string golden. Unsafe paths
normalize the offending snapshot entry, including when its target escapes.
On failure, `expected.receipt` describes the existing receipt, or null when no
online completion was committed; it never authorizes returning a model.
`provenance` is null for every error. The local-load-error case rehashes its
modified config before validation, so it tests error preservation after successful
snapshot validation rather than failing early on a stale hash.

`compare_hub(expected, actual)` consumes the normalized observation and returns
one named failure class or null. Precedence is `hub_transport` (calls and client
construction), `error_class`, `hub_provenance`, `hub_selection`, `hub_integrity`
(absences and receipt). Every offline case requires zero client constructions
and an empty transport log, including offline failures. Anonymous-client testing
must separately inspect the configured Authorization header; the contract makes
no claim that hf-hub never discovers a token file. These goldens and comparator
mutations do not themselves execute the production resolver or its filesystem
checks. That adapter belongs to the Hub implementation owner.

### CLI wire records and capture status

`cli_cases.json` separates the public text-generation capture, a typed info
contract, scripted private writer records and argument failures. `{fixture}` in
argv is replaced with the case's local fixture directory by the process adapter.
`feature` selects any build, an hf-hub build, or a build without hf-hub.
`before_load` requires rejection even if the supplied model directory is absent.
No byte golden contains timings, absolute paths, dependency messages or hardware
strings. `stdout_utf8` means exactly its UTF-8 bytes, including any explicit
newline. JSONL records order fields as `version`, `event`, then `processed,total`
or `token_id,text,finish_reason`; each ends in one newline. Empty Token deltas
remain records. Info JSON is compact, recursively sorts map keys, and ends with
one newline. Root keys are exactly `version,config,tokenizer,hub_provenance` as a
set; dimensions and variant encodings mirror the public types in section 2.8.

The canonical prompt is `hello the small fox runs over green hill`. Ruling A's
original-string BOS rule yields `[4,12,14,15,16,17,18,19,20]`. Its default 2048-token
prefill ceiling yields `(0,9),(8,9),(9,9)`. The generator first checks its integer
schedule against **all four** committed `prefill.progress` goldens, then applies
that schedule to the nine-ID text prompt. Token/text expectations must come from
`mlx_lm.stream_generate` with the original Python string, max_tokens 8 and a
greedy sampler. They must never be copied from the eight-ID stream goldens.

Until that pinned host capture runs, `generation.capture_status` is `not_run`
and both `capture` and `expected` are null. This is an incomplete cohort, not an
empty-output expectation. `--require-capture` exits nonzero in this state, and
`compare_cli(None, actual)` raises an error. `--capture` records public Python
responses and source/environment identities; two identical generations are
required by `--freeze-from ... --repeat ...`. Scripted writer records use explicit
private `OutputEvent` values and make no model-output claim. They do not construct
non-exhaustive library events across the consumer boundary.

`compare_cli(expected, actual)` takes `exit_code`, `stdout_utf8`, `stderr_utf8`;
it parses actual stdout, never trusts an adapter's decoded record list, and
finally compares exact bytes. Its failure precedence is `exit_code`, JSONL
`output_count`, `finish_reason`, `sampled_id`, `text_delta`, `progress`, info
`config`, exact `stdout`, then `stderr`. Malformed JSON diagnostics in stdout fail
as `stdout`. Stderr rules are empty, nonempty, or optional diagnostics; diagnostic
wording is not frozen. Runtime error stderr must preserve the source chain in
implementation tests. BrokenPipe exits 0 and drops generation; other output,
load, generation and unknown-event errors exit 1; syntax exits 2.

Qualification deletes an empty Token, changes a terminal reason and token ID,
moves diagnostics to stdout, appends a text newline, changes hidden_size and
accepts a zero positive flag. Their classes are respectively `output_count`,
`finish_reason`, `sampled_id`, `stdout`, `stdout`, `config`, `exit_code`.
The writer fault recipes name short write, Interrupted, ordinary error and
BrokenPipe checks; executing them and real generate/info/offline process tests
belongs to the CLI owner. Comparator qualification alone is not those tests.

### Compiled feature inventory

`xtask verify-lm-features` compiles an isolated external consumer against the
checkout for default, no-default-features, hf-hub, oracle-hooks and both features.
The reviewed inventory is the literal Rust in `xtask/src/verify_lm_features.rs`:
all approved root exports, public option/config/event/cache/chat fields and
variants, GGUF error additions, final Hub errors, gated provenance and hooks.
It includes explicit field types and public call results. HubError itself and
its semantic variants remain unconditional; only HubError::Api is Hub-gated.

Each combination first compiles a dependency control. Separate negative consumers
must reject HubOptions, HubProvenance, Model::from_hub, Model::hub_provenance and
HubError::Api without hf-hub, and oracle_hooks without oracle-hooks. A failed
build qualifies absence only for the expected rustc error code, named symbol,
consumer target and primary source span. Dependency failures and unrelated
compiler errors cannot pass an absence check. The report records every compiled
probe; skipped probes after a failed dependency control have no compiled evidence.
The gate checks this static reviewed inventory, not arbitrary future additions.
The source `api-baseline` detects source inventory drift and explicitly does not
evaluate cfg. Source enumeration is not compiled evidence; both gates are needed.
