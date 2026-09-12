# Tranche 2 typed skeleton

This crate declares the approved synchronous API. It does not yet load or run models.
`publish = false` prevents publishing the skeleton through Cargo. Remove that setting only
when this table is empty and the model parity gates pass. No skeleton feature or public
placeholder error type is introduced. The private `NotYetImplemented` formatter supplies
“not yet implemented in this tranche” diagnostics to typed unsupported-operation errors.
It must be deleted before publication.

## Ownership and placeholders

Each row is an implementation claim point. An error row always fails; it does not return
partial state. Key-map hooks have the design's infallible signature, so their placeholder
rejects every key. Metadata accessors, defaults, newtype conversions, raw JSON deserialization,
and registry selection are implemented. The private seams temporarily allow dead code because
no model constructor succeeds yet; remove those allowances as the seams gain callers.

| Owner | File | Placeholder | Current result |
| --- | --- | --- | --- |
| loader, tranche 4 | `src/model.rs` | `Model::from_gguf` | `LoadError::Weights(UnsupportedFormat)` |
| foundation, tranche 4 | `src/model.rs` | `Model::from_hub` | `HubError::Load(Config(UnsupportedArchitecture))` |
| foundation, tranche 3 | `src/model.rs` | `Model::generate` | `GenerationError::Inference(UnsupportedArchitecture)` |
| foundation, tranche 3 | `src/model.rs` | `Generation::next` | One typed inference error, then fused exhaustion |
| loader, tranche 4 | `src/weights/mod.rs` | `WeightManifest::from_gguf` | `WeightError::UnsupportedFormat` |
| llama | `src/arch/llama/mod.rs` | `Factory::parse_config` | `ConfigError::UnsupportedArchitecture` |
| llama | `src/arch/llama/mod.rs` | `Factory::build` | `LoadError::Config(UnsupportedArchitecture)` |
| llama | `src/arch/llama/mod.rs` | `Factory::map_safetensors_key` | `WeightDisposition::Reject` |
| llama, tranche 4 | `src/arch/llama/mod.rs` | `Factory::map_gguf_key` | `WeightDisposition::Reject` |
| qwen3, tranche 4 | `src/arch/qwen3/mod.rs` | `Factory::map_gguf_key` | `WeightDisposition::Reject` |

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
are unchanged. No public API was added. Because test autodiscovery is disabled and the loader
is private, `tests/weights.rs` is included as a unit-test module from `src/weights/mod.rs`;
its metadata-only tests can be selected with `weights::tests::pure`. Discovery parses headers with the
`safetensors` crate types and materialization loads each shard once through
`Array::load_safetensors`, validating every loaded array against the manifest before assignment.

Wire configs currently retain raw fields without resolving defaults.
`DecoderModel` is declared exactly as designed; neither factory constructs an implementation.
The registry contains only `llama::Factory` and `qwen3::Factory`.

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
is linear in the generated sequence; it does not decode the full history per step. Remove its
narrow dead-code allowances when foundation/generation gain callers.

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

## Prototype removal

The `prototype-adapter` feature is off by default. Explicit `parity` and `sentinel` test
entry points alias their own crate as `mlx_lm` and re-export only `legacy::{cache, models}`.
This lets the existing test sources retain their imports byte-for-byte, without exposing
old modules at the library root. Both targets now require
`--features prototype-adapter,oracle-hooks` until the llama stage-B round deletes the
prototype adapter. Decision 11's hooks live in `src/oracle_hooks.rs`.
Auto-discovery is disabled so Cargo cannot also compile the old entry points directly.
The comparator's fixture/mutation tests share the parity binary and therefore also require
this feature until the adapter is ported. New integration tests must be registered explicitly.

There are no `tests/weights.rs`, `tests/cache.rs`, `tests/tokenizer.rs`,
`tests/arch_llama.rs`, or `tests/arch_qwen3.rs` integration entry points in this worktree.
Keep `autotests = false`; per-module tests run with
`cargo test -p mlx-lm --lib <module>::`, using `weights`, `cache`, `tokenizer`,
`arch::llama`, or `arch::qwen3` as the module name. Do not register `#[path]` shims.

The llama item must repoint the prototype adapter and sentinel at the new `Model`, preserve
their assertions, delete `src/legacy/` and both wrapper entry points, remove the feature,
and restore normal test discovery. Changes to the protected parity adapter still require
the repository's oracle-change process. No protected source is modified in this step.

Only the legacy Llama forward, parameter loader, RoPE, caller-provided cache trait, model
input, and greedy adapter remain. Legacy stochastic sampling, both generation iterators,
Qwen3 prototype math, concatenating cache implementations, FloatOrStr/FloatOrString, and
quantized cache wrappers are deleted. The remaining `sample` function exists only because
both unchanged regression adapters call it; it rejects nonzero temperature. The legacy
loader is deliberately not the new strict loader and must never back `Model::from_dir`.

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
API and no longer depends on the deleted utilities crate. It will return the typed loader
placeholder until the foundation and model items implement loading.

## Verification of this step

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

`model::tests::local_loading_matches_fixture_expectations` is explicitly ignored with
`NOT RUN` until llama/qwen3 construction and forwards land. Running it with `--ignored
--nocapture` reports each unavailable fixture and fails if any remain unavailable, so an
explicit run cannot count placeholder errors as a passing happy path. Remove the ignore
when those implementations are integrated.

Decision 11 is implemented behind `oracle-hooks` in `src/oracle_hooks.rs` using the
session-object signature recorded above. Each prefill chunk and decode token uses its own
`Cache::step()` and `CacheStep::evaluate_and_commit(&[&logits])` transaction. Prefill
concatenates every position's logits on axis 1; `cache_view` uses `Cache::logical_layers()`.
Parity/sentinel require `prototype-adapter,oracle-hooks` until the llama stage-B round
deletes the prototype adapter. No protected test sources changed.

The oracle-hook tests report `NOT RUN` and fail if fixture loading still encounters an
architecture placeholder; they cannot pass without exercising their assertions. Once
architectures land, the fixture test checks full and chunked prefill logits, retained K/V
against the corresponding positions in the full-prefill goldens, each decode step's logits,
and final decode K/V. The empty-prompt check performs no MLX operation after model loading;
its fixture setup still requires a real constructor. Metal inference remains unverified.

Changed files are implementation-owned: workspace/package manifests, `mlx-lm` sources and
handoff documentation, adapter test entry points, the moved Qwen3 fixture, the removed
`mlx-lm-utils` crate, and the migrated `examples/lm` consumer. Protected-oracle files: none.
Ledger files: none. No commit was created.
