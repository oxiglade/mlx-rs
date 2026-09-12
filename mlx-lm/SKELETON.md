# Typed skeleton and tranche 3 declarations

This crate declares the approved synchronous API and loads local Llama and Qwen3 models.
`publish = false` prevents publishing the skeleton through Cargo. Remove that setting only
when this table is empty and the model parity gates pass. No skeleton feature or public
placeholder error type is introduced. The private `NotYetImplemented` formatter supplies
“not yet implemented in this tranche” diagnostics to typed unsupported-operation errors.
It must be deleted before publication.

## Ownership and placeholders

Each row is an implementation claim point. An error row always fails; it does not return
partial state. Key-map hooks have the design's infallible signature, so their placeholder
rejects every key. Metadata accessors, defaults, newtype conversions, raw JSON deserialization,
and registry selection are implemented. Some private seams temporarily allow dead code;
remove those allowances as the seams gain callers.

| Owner | File | Placeholder | Current result |
| --- | --- | --- | --- |
| loader, tranche 4 | `src/model.rs` | `Model::from_gguf` | `LoadError::Weights(UnsupportedFormat)` |
| foundation, tranche 4 | `src/model.rs` | `Model::from_hub` | `HubError::Load(Config(UnsupportedArchitecture))` |
| foundation, tranche 3 | `src/model.rs` | `Model::generate` | `GenerationError::Inference(UnsupportedArchitecture)` |
| cache reuse, tranche 3 item 4 | `src/model.rs` | `Model::new_cache` | `CacheError::UnsupportedPolicy`; model identity binding awaits item 4 |
| foundation, tranche 3 engine | `src/model.rs` | `Model::generate_with_cache` | Same not-yet-implemented inference error as `generate` |
| foundation, tranche 3 engine | `src/model.rs` | `Generation::cache` | Borrows a private cache field; no generation constructor succeeds yet |
| foundation, tranche 3 engine | `src/model.rs` | `Generation::snapshot` | Delegates to the private cache field; no generation constructor succeeds yet |
| foundation, tranche 3 | `src/model.rs` | `Generation::next` | One typed inference error, then fused exhaustion |
| cache reuse, tranche 3 item 4 | `src/cache/mod.rs` | `Cache::tokens` | Empty slice until the represented-token ledger lands |
| cache reuse, tranche 3 item 4 | `src/cache/mod.rs` | `CacheStep::evaluate` | Not-yet-implemented `CacheError::InvalidState`; existing `evaluate_and_commit` is unchanged |
| cache reuse, tranche 3 item 4 | `src/cache/mod.rs` | `EvaluatedCacheStep::commit` | Uninhabited guard; no successful evaluation can construct it yet |
| sampling, tranche 3 item 3 | `src/sampling.rs` | `SamplingEngine`, `PendingSample` | Implemented; dead-code allowances stay until the generation engine calls `SamplingEngine::sample` |
| loader, tranche 4 | `src/weights/mod.rs` | `WeightManifest::from_gguf` | `WeightError::UnsupportedFormat` |
| llama, tranche 4 | `src/arch/llama/mod.rs` | `Factory::map_gguf_key` | `WeightDisposition::Reject` |
| qwen3, tranche 4 | `src/arch/qwen3/mod.rs` | `Factory::map_gguf_key` | `WeightDisposition::Reject` |

The tranche 3 serial foundation follows `t3/position-astra.md` sections "Public surface",
"Processor and sampler contract", "State transitions and completed boundaries",
"Reuse is an exact-prefix operation", and "Validation and error ordering", with
`t3/DECISIONS.md` rulings A/J (original-string BOS check) and H/K (error variants and
runtime-evaluation mapping). These are declarations; generation validation, sampling,
stop filtering, model-bound cache identity, token ledgers, and split transaction execution
remain with their implementation owners. Defaults and additive-penalty validation are
implemented. The existing negative `Send`/`Sync` assertions cover `Model`, `Generation`,
`Cache`, and `CacheSnapshot`.

The cache item implements the sealed `LayerCache` trait, full and rotating storage,
transaction rollback, and snapshots. Full storage accepts a known total capacity and grows
geometrically when that capacity is exceeded. Rotating storage uses functional index
replacement for single-token updates; one-shot and chunked prefill preserve the oracle's
oversized logical arrays until a subsequent update trims them. No bounded-memory claim is
made until Metal long-loop measurements qualify the lazy graph and allocator behavior.

Cache integration notes and deviations:

- `Cache::new` takes `architecture: ModelType` ahead of layout, options, and optional total
  capacity; the generation caller supplies `prompt_len + max_tokens` as that capacity.
  Architecture identity is part of the snapshot fingerprint.
- `CacheInfo::retained_prefix: Range<usize>` is an added public field. A single range cannot
  describe a retained prefix and a disjoint recent tail. `retained_positions` describes the
  recent range, excluding the prefix. With `keep_prefix = 0`, it keeps the original meaning.
- `Cache::logical_layers()` is crate-visible for decision 11's oracle hooks. Each
  `LogicalLayerView` has `layer`, contiguous `keys` and `values` shaped
  `[batch, kv_heads, positions.len(), head_dim]`, and an absolute `positions` range.
  Views are ordered by layer, then position. Full layers yield one view; rotating layers
  with a retained prefix yield separate prefix and tail views, omitting empty segments.
  An empty layer yields one empty `0..0` view. Reading views leaves cache state intact.
  A layer's views cover exactly the positions `CacheInfo` reports, so after the oracle's
  oversized prefill they include every stored position (`llama-sliding` after prefill: 8
  positions at capacity 5, `0..8`); after decode they match the fixture's `after_decode`
  arrays (sliding layer 1: `11..16`).
- `CacheStep::evaluate_and_commit(&[&Array])` evaluates logits/token outputs together with
  staged K/V. `commit()` uses the same path without extra outputs. Any failed layer update
  poisons the step; failed or incomplete steps cannot commit. Dropping staged handles leaves
  the committed cache intact. Every layer must be updated once to the same absolute position.
  `CacheStep::info(layer)` exposes staged offsets and retained ranges to model math.
- Snapshots clone handles keyed by layer and validate architecture, cardinality, layout,
  policy, dtype, backing shape, absolute position, logical length, and rotation index.
  Storage is always allocated, so there are no optional array slots. Full allocation sizes
  may differ when restoring an earlier snapshot; unused capacity is not semantic state.
- The oracle's oversized-prefill behavior is an exception to fixed-capacity storage:
  the first chunk retains all its tokens; later chunks retain up to capacity + chunk - 1.
  Single-token decode returns to the configured capacity and emits temporal order.
  Internal trimming follows the oracle's `is_trimmable` restriction: after reaching capacity,
  trimming returns `InvalidState` because evicted tokens cannot be recovered.

The loader implements safetensors discovery, indexed-shard reconciliation, metadata validation,
and strict application through `StateProjection`. Discovery reads headers without loading tensor
payloads. Assignment validates every mapped slot before materialization, evaluates all staged
arrays, then restores one keyed snapshot, retaining absent optional slots. GGUF remains tranche 4.

Architecture constructors use `affine_groups()` to detect packed groups and
`validate_affine_group(prefix, &AffineQuantization)` to check their configured layout before
allocating slots. They own quantized module construction and per-layer overrides. Core quantized
modules project `inner.weight`, `scales`, and `biases`; architecture key maps hide those names
and explicitly ignore redundant tied heads. No randomly initialized weight is quantized by the
loader. Each architecture owns `ParsedConfig`, its private JSON `WireConfig`, all model math,
and its key dispositions.

Loader signature choices: the skeleton's `WeightDisposition::Parameter`, `WeightError::MissingKey`,
and `WeightError::UnexpectedKey` retain their names (the brief calls these `Assign`, `MissingTensor`,
and `UnexpectedTensor`). Fixture spellings `MissingShard`, `DuplicateTensor`, and `ShapeMismatch`
are unchanged. No public API was added. The private loader tests live in `src/weights/tests.rs`;
its metadata-only tests can be selected with `weights::tests::pure`. Discovery parses headers with the
`safetensors` crate types and materialization loads each shard once through
`Array::load_safetensors`, validating every loaded array against the manifest before assignment.

The registry contains `llama::Factory` and `qwen3::Factory`, which construct the
corresponding `DecoderModel` implementations from resolved wire configs.

`Model::from_dir` delegates its single weight load to `ArchitectureFactory::build`.
The factory owns strict keyed assignment, tied-aware ignores such as a redundant
`lm_head.weight`, and weight evaluation before returning the decoder. `from_dir` only
discovers the manifest and passes it to the factory; it does not reload the projection.

The tokenizer item implements local JSON loading, encode/decode, EOS resolution, template
selection, pycompat, textual roles, continuation, and the private streaming decoder. The
incoming skeleton already had no tokenizer placeholder rows; none remain to remove. Missing
templates fail during rendering; invalid selected Jinja fails during loading. String templates,
named-template dictionaries, and the serialized list of `{name, template}` entries select
`default`. Other named templates are not compiled. No tokenizer HTTP feature is enabled.

The foundation loader can call crate-private
`Tokenizer::from_dir_with_eos(path, Vec<TokenId>) -> Result<Tokenizer, TokenizerError>` after
resolving model EOS metadata. It loads tokenizer assets without rereading model/generation
config, sorts and deduplicates the supplied IDs, and preserves an explicitly empty set.
Standalone `from_dir` follows mlx_lm 0.31.3 precedence: a nonempty generation list or nonzero
scalar overrides model config; null, an empty list, or scalar zero in generation config falls
back to model config. Null/absent model EOS falls back to the tokenizer's EOS spelling.
Invalid EOS types and out-of-range IDs fail with `InvalidEos`; invalid JSON is rejected rather
than silently ignored as in Python's generation-config reader.

Generation uses crate-private `Tokenizer::decode_stream() -> StreamingDecoder<'_>`,
`step(TokenId) -> Result<Option<String>, TokenizerError>`, and consuming
`finish() -> Result<String, TokenizerError>`. Feed only generated non-stop tokens; skip the
sampled EOS and call `finish` on either EOS or length completion. Discard the decoder after
an error. Special tokens are preserved, matching `decode` and Python’s streaming detokenizer.
UTF-8 fragments are buffered by `tokenizers::DecodeStream`,
and `finish` decodes the accumulated IDs once to emit any remaining text, including incomplete
UTF-8 replacement characters. If final decoding rewrites an already emitted prefix (for example,
ByteFallback can replace an entire byte run when it ends incomplete), `finish` returns the
wrapped `DecodeStreamError::InvalidPrefix`; it cannot retract prior deltas. The wrapper retains
IDs and emitted text, so auxiliary storage
is linear in the generated sequence; it does not decode the full history per step. The
filter and decoder seams carry narrow dead-code allowances until the generation engine, their
only caller, lands.

Tranche 3 item 5 implements private `stop::StopStringFilter`, following
`t3/position-astra.md` section "EOS, length, stop strings and stable text" and
`t3/DECISIONS.md` item 3. `new(Vec<String>) -> Result<Self, GenerationError>` rejects
the first empty stop string with its original index; call it during generation validation.
`process(&str) -> (String, bool)` returns visible text and whether a match stopped the
stream. It excludes the earliest complete match and everything after it, holding back
the longest proper-prefix suffix otherwise. Matching uses exact UTF-8 bytes.
`finish(self, final_delta: &str) -> (String, bool)` processes the decoder's final flush
before releasing an unmatched suffix. A true result means Stop, including at the length
limit. After a process call returns true, discard the decoder without flushing it.

The text tests read cohort 2's `llama-base/text_cases.json`: all stop-filter events,
held suffixes, terminal states and finish flushes, plus five ByteLevel/ByteFallback
cases with final decode bytes and typed `InvalidPrefix` payload checks. Pure integration
tests pass decoder flushes through the filter for both matches and mismatches. Existing
WordLevel, special-token and chat goldens remain covered by the tokenizer suite.
Generation-event ordering and cache rollback on decoder errors await the engine owner.

Tokenizer design deviations and signature choices:

- `ChatTemplateOptions` adds public `enable_thinking: Option<bool>` because the protected Qwen3
  goldens explicitly pass false. `None` leaves the variable undefined and preserves template
  defaults. No arbitrary template-kwargs map is exposed. The separately packaged
  `examples/lm/src/main.rs` literal needs `..Default::default()` when its owner next migrates it;
  it is outside this item's ownership and outside workspace checks.
- `decode` preserves special tokens, matching Python's default and the oracle's ordinary
  decodings; the incoming skeleton omitted them. Private helpers qualify the oracle's explicit
  add/skip-special-token cases without adding public option types.
- ContinueLast uses Transformers 5.17.0's marker algorithm, including spacing preservation,
  trimmed output, and empty final text. An empty conversation returns `IncompatibleContinuation`.
- `src/tokenizer/tests.rs` contains the tokenizer unit tests. It reads every protected fixture
  directory and compares all tokenizer expectations, EOS source, exact chat UTF-8 hex, chat
  token IDs, and per-step streaming deltas. Protected fixtures are never edited.

Tokenizer verification passed: `cargo fmt --check`,
`cargo clippy -p mlx-lm --all-targets -- -D warnings`, `cargo check --tests`,
`cargo doc -p mlx-lm --no-deps`, and
`cargo test -p mlx-lm --lib tokenizer::` (11 tests). The six fixture
directories cover 36 encodings, 36 decodings, EOS metadata/source, and 18 exact chat cases.
Five continuation edge cases were checked directly against the pinned Transformers renderer.
No MLX evaluation, Metal/model inference, or generation-event delta parity was executed.
The excluded LM example was not built. No commit was created.

## Test features

Normal Cargo test discovery is enabled. The explicit `parity` and `sentinel` targets use
`tests/parity.rs` and `tests/sentinel.rs`, both with `required-features = ["oracle-hooks"]`.
Run them with `cargo test -p mlx-lm --features oracle-hooks --test parity --test sentinel`.
The default feature set is empty, so plain `cargo test -p mlx-lm` does not link these targets.
`hf-hub` independently enables the optional Hub API; it is not needed for either test.
Per-module unit tests run with `cargo test -p mlx-lm --lib <module>::`.

Both adapters load through `Model::from_dir` and use `oracle_hooks` for prefill, cache views,
and greedy decode. The sentinel reads the tokenizer from the loaded model. Logical K/V
views have shape `[batch, kv_heads, positions.len(), head_dim]`, matching the sentinel's
pinned `[1, 1, 4, 4]` arrays directly; no layout mapping or fixture edit is needed.

## Signature choices where the design is silent

Decision 1 requires no trait signature change. The existing factory boundary is
`fn build(&self, parsed: ParsedArchitecture, weights: &WeightManifest) -> Result<Box<dyn DecoderModel>, LoadError>`.
Architecture implementations must complete their strict weight load inside this call.

Decision 11 uses the off-by-default `oracle-hooks` feature and hidden public
`src/oracle_hooks.rs` module. `prefill_logits` returns all-position logits and an
`OracleSession<'m> { model: &'m mut Model, cache: Cache }`; `decode_step` and `cache_view`
are session methods. This session-object shape is the decision 11 signature deviation.
`cache_view` maps `Cache::logical_layers()` into contiguous temporal `LayerView` arrays,
preserving separate position ranges when a layer retains a disjoint prefix and tail.
`InferenceError::EmptyPrompt` adds the `prompt is empty` diagnostic because the prefill
entry point returns `InferenceError`, not `GenerationError`.

`SamplingError::EmptyVocabulary` and `SamplingError::MinTokensToKeepExceedsVocabulary`
are accepted deviations: validation receives the runtime vocabulary, and neither a zero
vocabulary nor min-p support larger than the vocabulary fits the design's listed variants.

The design fixes all public option fields and both architecture trait signatures, but does
not spell out error payloads, `HubOptions` fields, `ParameterPath` access, validation method
names, or private cache/loader signatures. This step declares these seams in their owning
files. UnsupportedFormat, UnsupportedPolicy, UnsupportedMode, and inference's
UnsupportedArchitecture are ordinary typed boundary variants used for placeholder failures;
there is no catch-all boxed error. The six fixture error-class spellings are preserved:
`WeightError::MissingShard`, `WeightError::DuplicateTensor`, `WeightError::ShapeMismatch`,
`ConfigError::UnsupportedArchitecture`, `ConfigError::UnsupportedQuantization`, and
`ConfigError::UnsupportedRope`.

`Cache`, `CacheKind`, `LayerQuantization`, and `ParameterPath` are re-exported at the root in
addition to the design's root list because its public signatures name them. Generation
lives in `src/model.rs` as required by this step, with its types re-exported at the root.
Hub options use optional revision/cache-directory overrides and an explicit offline flag.
Sampling defaults to greedy with every probability filter disabled. TokenId converts to
and from u32; ModelType and ParameterPath expose string construction and borrowing.

The obsolete `examples/lm` consumer now uses only the new public model/tokenizer/generation
API and no longer depends on the deleted utilities crate. Local checkpoint loading is
implemented; generation remains a later tranche.

## Foundation verification (before the final round)

| Check | Result |
| --- | --- |
| `cargo fmt --check` | Passed |
| `cargo clippy -p mlx-lm --all-targets -- -D warnings` | Passed |
| `cargo check --tests` | Passed; existing warnings remain in core/test crates |
| `cargo doc -p mlx-lm --no-deps` | Passed |
| `cargo clippy -p mlx-lm --all-targets --all-features -- -D warnings` | Passed, including both unchanged regression adapters and Hub signatures |
| `cargo check -p mlx-lm --no-default-features --tests` | Passed |
| `cargo test -p mlx-lm --lib tokenizer::tests -- --test-threads=1` | Both Qwen3 template and Unicode continuation tests passed |
| Default normal dependency tree | No HTTP client, anyhow, clap, or utilities crate |
| Protected-path diff and fixture copy comparison | No protected changes; moved Qwen3 fixture is byte-identical |
| `git diff --check` and example `rustfmt --check` | Passed |

The preset Cargo target directory was retained. Initial checks could not resolve the registry
or link against the default SDK, and the native build tried to fetch MLX. Successful checks
used offline Cargo resolution, `/usr/bin/cc`, the installed Xcode SDK, and a temporary CMake
toolchain file pointing to clean cached MLX `v0.32.2` sources and their cached dependencies.
A temporary xcrun wrapper directs Metal compiler module-cache writes into `/private/tmp`.
These settings live in `/private/tmp/mlx-lm-t2-env.sh`; no core build files were changed and
neither devenv nor nix was run. Temporary verification logs use the
`/private/tmp/mlx-lm-t2-` prefix.

Model/Metal execution, parity and sentinel inference, real checkpoint loading, cache behavior,
and Hub downloads were not run. The tokenizer tests do not execute MLX operations. The LM
example was formatted and migrated but was not executed against a checkpoint.

`model::tests::local_loading_matches_fixture_expectations` is enabled now that both
architectures are integrated. Its assertions are unchanged and require Metal verification.

Decision 11 is implemented behind `oracle-hooks` in `src/oracle_hooks.rs` using the
session-object signature recorded above. Each prefill chunk and decode token uses its own
`Cache::step()` and `CacheStep::evaluate_and_commit(&[&logits])` transaction. Prefill
concatenates every position's logits on axis 1; `cache_view` uses `Cache::logical_layers()`.

The oracle-hook tests report `NOT RUN` and fail if fixture loading still encounters an
architecture placeholder; they cannot pass without exercising their assertions. Once
architectures land, the fixture test checks full and chunked prefill logits, retained K/V
against the corresponding positions in the full-prefill goldens, each decode step's logits,
and final decode K/V. The empty-prompt check performs no MLX operation after model loading;
its fixture setup still requires a real constructor. Metal inference remains unverified.

The foundation step changed implementation-owned files: workspace/package manifests,
`mlx-lm` sources and handoff documentation, adapter test entry points, the moved Qwen3 fixture,
the removed `mlx-lm-utils` crate, and the migrated `examples/lm` consumer. That step changed
no protected-oracle or ledger files and created no commit.
The final round also repoints the protected `tests/parity/prototype.rs` adapter; the launcher
will commit that edit separately through the oracle-change process.

## Tranche 3 serial foundation verification

The declaration step passed formatting, default and all-features Clippy with warnings
denied, workspace/all-target compilation, `cargo doc -p mlx-lm --no-deps`, and 60 pure
library tests (seven new tests).
Workspace compilation reported existing warnings in untouched `mlx-rs` and `mlx-tests`
tests. The excluded LM example passed `rustc --emit=metadata` against the updated library
and anyhow metadata in the preset target directory.

The initial Clippy attempt failed because the PATH-selected linker could not find
`libiconv`. Verification then used the existing `/private/tmp/mlx-lm-t2-env.sh` settings
described above, preserving the preset `CARGO_TARGET_DIR`. No devenv/nix command ran,
no repository build configuration changed, and no commit was created.

All default library tests compiled. The full unfiltered test command was not executed:
the sandbox has no Metal device. The test run used `--test-threads=1` with explicit
`--skip` arguments for these 18 MLX runtime tests; none were marked ignored in source.

| Module | Tests compiled but not executed |
| --- | --- |
| `arch::llama::tests` | `fixture_wrong_shape_errors`, `fixture_prefill_all_positions_cache_and_chunks`, `fixture_half_precision_forward` |
| `arch::qwen3::tests` | `qwen3_base_prefill_and_cache`, `qwen3_quant4_prefill_and_cache`, `causal_and_sliding_mask_with_prefix_positions` |
| `cache::tests` | `logical_full_layers_after_appends`, `logical_rotating_layers_after_wrap_and_oversized_prefill`, `full_append_and_growth`, `rotating_wrap_prefix_and_chunk_after_wrap`, `rotating_prefill_over_capacity_and_capacity_one`, `rotating_trim_before_capacity_and_restore_after_wrap`, `transaction_rollback_and_snapshot_restore`, `invalid_steps_and_snapshot_fingerprints`, `committed_fixture_cache_arrays` |
| `weights::tests::runtime` | `projection_round_trip_and_atomic_failure`, `packed_embedding_and_linear_slots` |
| `model::tests` | `local_loading_matches_fixture_expectations` |

Generation execution, sampling distributions, evaluation-error mapping, cache reuse,
Metal inference, checkpoint execution of the example, parity/sentinel execution, memory,
leaks, and Guard Malloc remain unverified. Their implementations or runtime qualification
belong to later tranche 3 steps.
