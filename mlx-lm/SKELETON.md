# Local loading and generation

The crate loads Llama and Qwen3 safetensors and GGUF checkpoints, resolves Hub
snapshots behind `hf-hub`, and exposes one synchronous generation engine.
`publish = false` remains until release qualification is complete.

## Remaining work

| Owner | File | Remaining work | Current result |
| --- | --- | --- | --- |
| foundation, tranche 4b item 3 | `src/model/hub.rs`, `src/model/hub/snapshot.rs`, `src/model/hub/tests/contract.rs` | Integration revalidation against the table amended under ruling P | Receipts record hashes at completion; reuse checks metadata and receipt identity. The client-contract test verifies anonymous requests and zero offline client constructions. |
| harnesses, tranche 4b item 5a | `tests/real_checkpoints.rs`, `examples/benchmark.rs`, `../conformance/mlx-lm/{real_checkpoints,benchmark}.py` | Launcher runs the six-case CPU/Metal matrix and paired ruling-C protocol on the idle host | Harnesses implemented; real inference and timing remain unexecuted. No pipeline change is admitted. |
| text/FFI, tranche 3 item 5 | `tests/generation_ffi.rs` | Run the reduced workload with `xtask verify-ffi --guard-malloc` on the host under DECISIONS O | Ordinary runs retain the paper's "Memory, snapshots and the FFI gate" workload and proposed F envelope. `DYLD_INSERT_LIBRARIES` containing `libgmalloc` selects reduced coverage: Qwen3 full generation (40 samples), Llama reuse (40 calls, capacities 16/32/64), kept-prefix rotation (40 updates), sliding oversized prefill (25 tokens, chunks of 8) and one snapshot retained across a wrap (24 samples), tokenizer EOS, and one affine load/generate/drop cycle with length, exact-token/text stops and all three cancellation boundaries. Both modes retain behavioral assertions; reduced mode skips all allocator observations and byte guards and prints the completed scenario list. Default library fault-injection tests remain in `src/model/tests.rs`. |
| worker, tranche 3 item 5 | `examples/worker.rs` | Run on the Metal host with one positional checkpoint directory | Implements the paper's "Device, RNG and threading": model, GPU stream scope and cache stay on one OS thread; bounded owned request/event channels, worker-local error conversion, receiver-drop cancellation and receiver-before-join shutdown. Positive Send checks cover options/events/errors/requests. |

## Local release harnesses

The [conformance README](../conformance/mlx-lm/README.md#local-real-checkpoints)
defines manifest additions, contributor commands, independent GGUF provenance,
and the [ruling-C protocol](../conformance/mlx-lm/README.md#ruling-c-benchmark).
Both harnesses refuse CI execution. An absent real-checkpoint manifest emits one
explicit NOT RUN result; supplied incomplete material fails. The pinned Python
report is generated without reading Rust output, then hashed and bound before
Rust checks all six cases on both devices.

The coordinator owns sequential AB/BA processes and immutable sample reports.
Prefix identity precedes throughput interpretation. The Python no-lookahead
ablation is confined to an in-memory benchmark function and retains the extra
terminal forward. C remains open pending local evidence; any pipeline still
requires the paper's scheduling, counterfactual, performance, and behavior
qualification. This item neither publishes the crate nor changes generation.

Item 5a sandbox verification compiled both new targets in debug and release,
compiled all crate tests with all features, and checked all targets/all features.
The 66 pure tests passed: 9 Python harness tests, 3 Rust harness tests, and 54
existing config/tokenizer/stop/pure library tests. The absent-manifest result,
supplied missing-manifest failure, and CI refusals were checked separately.
Compilation used cached MLX 0.32.2 native libraries through a temporary CMake
configuration; it did not establish Metal runtime readiness.

The following local qualifications remain unexecuted on both CPU and Metal,
for both the independent Python report and the Rust `real_checkpoints` test:

- `llama-1b-bf16`
- `llama-1b-4bit`
- `qwen3-06b-bf16`
- `qwen3-06b-4bit`
- `gguf-tinyllama-q4_0`
- `gguf-qwen3-q8_0`

Also unexecuted: ruling-C baseline on CPU/Metal, no-lookahead ablation on
CPU/Metal, unmodified `stream_generate` on Metal, and the untimed attribution
profile subset. Existing MLX runtime library tests and the `generation_ffi`,
`hub_offline`, `parity`, and `sentinel` suites were compiled but not executed in
this item. No real parity or throughput result is claimed.

## Generation and completed boundaries

`Model::generate` and `Model::generate_with_cache` call the same constructor and state
machine in `src/model.rs`, using an internal owned-or-borrowed cache. `Model::new_cache`
uses the model-bound cache constructor. Loading creates one private `Rc<()>` identity;
generation does not reload weights. The negative `Send`/`Sync` assertions still apply to
`Model`, `Generation`, `Cache`, and `CacheSnapshot`; options and events are sendable values.

The implementation follows `t3/position-astra.md` sections "State transitions and completed
boundaries", "Complete the transaction through text preparation", "EOS, length, stop strings
and stable text", "Reuse is an exact-prefix operation", "Validation and error ordering", and
"Device, RNG and threading", with `t3/DECISIONS.md` rulings A–L overriding the paper.

Construction validates sampling and repetition/presence/frequency penalties, stop strings,
resolved sorted stop IDs, encoded/copied prompt IDs, borrowed cache compatibility/policy/
ledger/prefix/uncached suffix, then checked `P + M` and i32 representability. Text encoding
uses the tokenizer post-processor unless the original string starts with the configured BOS
content, without trimming or encoding first (A/J). Training `max_positions` is not a new cap.

Start emits `{ processed: 0, total: U }` without a forward or RNG draw. Prefill reserves the
last prompt token and commits each evaluated chunk before reporting progress. Final prefill
is a separate completed boundary from FirstSample. Decode forwards only the preceding sampled
token: after token event `j`, the cache represents `P + j - 1` tokens. Drop does no inference.
Every error yields once and fuses the stream.

Sampling uses the caller's stream scope and a private candidate RNG. Under ruling B it can
perform separate value-check and sampling evaluations. Decode holds the cache transaction
through these evaluations, checked token conversion, decoder step, stop filtering and final
flush. Only successful text preparation publishes the evaluated cache, RNG and accepted
history. Runtime evaluation exceptions are preserved as `GenerationError::Exception` (K);
forward construction, sampling construction/conversion, structural cache and tokenizer errors
retain their respective typed wrappers.

Penalty history starts with the last prompt token and adds accepted generated tokens; earlier
prompt tokens and reused prefixes are excluded. Each request starts fresh history and RNG.
EOS/explicit stop IDs are returned but never decoded. Text stops return the current raw ID,
exclude the matching text, and discard the decoder without flushing past the match. EOS and
length otherwise flush through the filter; a match in that flush takes precedence over Length.
Exactly one successful Token event carries a terminal reason.

## Internal seams and oracle adapter

- `Cache::new_for_model(architecture, layout, options, capacity, model_identity)` constructs
  a tracked cache. `Model::new_cache` supplies no capacity (ordinary full capacity 16); fresh
  generation supplies `cache::checked_capacity(P, max_tokens)`. Rotating policies retain their
  configured capacity and prefix.
- `Cache::validate_reuse(&identity, &architecture, layout, &options, &full_prompt)` checks
  identity/layout, resolved policy, aligned layer positions and ledger, exact prefix, then
  a nonempty suffix. Its result is the first uncached index. `ModelDefault` is not a wildcard.
- `Cache::step_with_tokens(ids)` stages input IDs and reserves ledger space. Reused generation
  calls `CacheStep::reserve(checked_total)` in its first staged forward. Failure does not
  publish growth. `evaluate(outputs)` jointly evaluates staged K/V and outputs and returns an
  `EvaluatedCacheStep`; its infallible `commit()` publishes layers and the reserved ledger append.
- `Cache::tokens()` includes evicted IDs. Snapshots share the ledger and array handles and copy
  layer metadata. The first append while a snapshot is retained uses a prepared ledger copy.
  Restore atomically restores layers, ledger and identity. A snapshot excludes generation RNG,
  decoder and iterator state. No public trim or iterator restore is exposed.
- `SamplingEngine::sample(history, logits, rng, capture_logprobs)` accepts `[1,V]` floating
  logits and returns `PendingSample { token, filtered_logprobs, rng }`. It does not publish
  request state. Greedy constructs no RNG. Stochastic sampling clones the last committed RNG
  and splits once. First-position diagnostic logprobs retain masks without renormalization.
- `Tokenizer::decode_stream()` returns `StreamingDecoder`; `step(TokenId)` returns an optional
  delta and consuming `finish()` returns final text or the typed InvalidPrefix source. The
  unused `from_dir_with_eos` helper was removed under L; loading resolves EOS through `from_dir`.
- `StopStringFilter::new(Vec<String>)` rejects the first empty string. `process(&str)` returns
  visible text and a match flag, holding back the longest proper-prefix suffix. Consuming
  `finish(final_delta)` processes the final flush and releases unmatched held text.

With the off-by-default `oracle-hooks` feature, the hidden module now exposes:

```rust
pub fn generate_with_logprobs<'m>(
    model: &'m mut Model,
    prompt: Prompt<'_>,
    options: GenerationOptions,
) -> Result<Generation<'m>, GenerationError>;
pub fn first_filtered_logprobs<'a>(
    generation: &'a Generation<'_>,
) -> Option<&'a mlx_rs::Array>;
```

The constructor enables capture in the public engine. The reader returns `None` until the
first successful Token event, then the evaluated `[1,V]` first-position distribution for the
remaining lifetime of the generator. Greedy capture returns normalized processed logprobs
without support filtering. It performs no extra sample or forward. Existing `prefill_logits`,
`OracleSession::decode_step` and `cache_view` remain forward/cache evidence; their cache after
feeding all eight oracle decode IDs intentionally differs from generation's final boundary.
`Cache::logical_layers()` returns contiguous temporal `[batch, kv_heads, positions, head_dim]`
views with separate prefix/tail ranges when needed.

## Loading and tokenizer contracts

The tranche 4a serial foundation extracts `ShardIndex::from_bytes(&[u8])` and
`shard_paths()` as crate-private seams. Byte parsing returns relative paths. Local
loading uses the same parser with the index directory so duplicate-entry errors
retain joined paths and invalid-path errors retain the original relative paths.
Local path admission, JSON error precedence, shard ordering, duplicate detection
and index reconciliation are unchanged; the stricter Hub path policy is separate.

`Model::from_gguf(file, tokenizer)` normalizes the exhaustive Llama/Qwen3 tensor
map into canonical checkpoint names, validates the explicit metadata profile and
uses the existing architecture factory and strict weight planner. In-memory GGUF
array handles survive the container. Llama Q/K row preparation applies to weights,
scales and affine biases on CPU; Qwen3 retains its native rows and per-head norms.
Safetensors reads remain pinned to CPU under ruling P. Staged parameters are
evaluated before atomic restore for both sources.

The GGUF profile admits full causal attention, no tensor biases, full-head RoPE,
and absent/none or linear scaling. Frequency-factor tensors are rejected. Metadata
is fetched by exact standard keys; unknown vendor metadata cannot be enumerated.
Converted affine groups require group size 32, 4 or 8 bits and F16 companions.
Every matrix receives an explicit quantized or floating override in mixed models.
F32/F16/Q4_0/Q4_1/Q8_0 are qualified source forms; core fallback conversion may
admit other original encodings that cannot be identified from converted arrays.

The caller supplies the tokenizer. Load its sidecars with `Tokenizer::from_dir`
from the exact original model revision. Pairing checks vocabulary and added-token
ID bounds, configured BOS/EOS bounds, embedded BOS/EOS agreement and special-token
spelling when embedded tokens are present. They cannot establish tokenizer identity:
two tokenizers can share sizes and special IDs but differ on ordinary tokens.
Retain hashes for tokenizer.json, tokenizer_config.json, config.json and
generation_config.json when present, the GGUF producer/source revision and a
canonical prompt's original IDs. `encode` adds no special tokens; text generation
uses the existing original-string BOS rule.

The module tests consume the ten frozen GGUF fixtures with each fixture's fixed
policy: f32-v1 (2e-4/2e-4), gguf-f16-v1 (5e-3/5e-3), or gguf-affine-v1
(2e-2/2e-3), expressed as atol/rtol. They compare every prefill position, chunks
1/3/8, per-layer K/V caches, eight fed decode steps and the exact greedy prefix.
Malformed-recipe tests materialize committed recipes in Rust and check their bytes
against frozen Python SHA-256 hashes before asserting typed errors. The byte checks
run without MLX initialization; forward and array-backed tests require it.
Tiny fixtures establish mapping and arithmetic self-consistency, not real-world
Qwen3 compatibility; the local real-checkpoint release qualification is separate.

`Model::from_dir` discovers one `WeightManifest` and calls the architecture factory once.
`WeightManifest::load_with(&mut StateProjection, disposition)` performs strict planning,
shape validation, materialization, evaluation and atomic assignment. Both architecture
factories use it with tied-aware mappings. Architecture modules own config parsing, model
math, quantized construction and per-layer overrides. No randomly initialized weights are
quantized by the loader.

Standalone tokenizer loading resolves nonempty/nonzero generation EOS before model EOS,
then tokenizer spelling. Invalid EOS and JSON are typed errors. Named chat templates select
`default`; invalid selected Jinja fails during loading and absent templates fail on rendering.
`encode` retains its no-specials contract; `encode_with_special_tokens` exposes the explicit
post-processor switch. Streaming preserves special tokens other than sampled stop IDs.
The decoder retains generated IDs and emitted text, so auxiliary host storage grows with
output. Exact per-event text parity is claimed only for the named WordLevel fixtures.

## Verification scope

`src/model/tests.rs` exercises public fixture streams, 45 seeded sampling trajectories,
progress goldens and cancellation, strict cache reuse, original-string BOS handling,
validation precedence, interleaved RNG isolation, terminal text and rollback boundaries.
Private scripted decoders inject partial-layer forward failure, empty support and lazy
singular-inverse evaluation failure; the ByteFallback fixture supplies a real final-prefix
error after evaluation. Tests remain enabled for execution on a host with MLX initialization.

Memory claims follow the proposed F guard. Rotation can
retain capacity plus a prefill chunk until subsequent updates; snapshots retain old storage.
No zero-allocation or constant-total-host-memory claim is made.

Under Guard Malloc the sampling distribution test draws a reduced sample (`src/sampling/tests.rs`), so the library binary completes the Guard Malloc phase; no other library test changes mode.
