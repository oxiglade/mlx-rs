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
| cache | `src/cache/mod.rs` | `Cache::new` | `CacheError::UnsupportedPolicy` |
| cache, tranche 3 | `src/cache/mod.rs` | `Cache::snapshot` | `CacheError::UnsupportedPolicy` |
| cache, tranche 3 | `src/cache/mod.rs` | `Cache::restore` | `CacheError::UnsupportedPolicy` |
| cache | `src/cache/mod.rs` | `Cache::step` | `CacheError::UnsupportedPolicy` |
| cache | `src/cache/mod.rs` | `CacheStep::update_and_fetch` | `CacheError::UnsupportedPolicy` |
| cache | `src/cache/mod.rs` | `CacheStep::commit` | `CacheError::UnsupportedPolicy` |
| cache | `src/cache/full.rs` | `FullCache::new` | `CacheError::UnsupportedPolicy` |
| cache | `src/cache/full.rs` | `FullCache::update_and_fetch` | `CacheError::UnsupportedPolicy` |
| cache | `src/cache/rotating.rs` | `RotatingCache::new` | `CacheError::UnsupportedPolicy` |
| cache | `src/cache/rotating.rs` | `RotatingCache::update_and_fetch` | `CacheError::UnsupportedPolicy` |
| loader | `src/weights/mod.rs` | `WeightManifest::discover` | `WeightError::UnsupportedFormat` |
| loader | `src/weights/mod.rs` | `WeightManifest::from_safetensors` | `WeightError::UnsupportedFormat` |
| loader | `src/weights/mod.rs` | `WeightManifest::from_sharded_index` | `WeightError::UnsupportedFormat` |
| loader, tranche 4 | `src/weights/mod.rs` | `WeightManifest::from_gguf` | `WeightError::UnsupportedFormat` |
| loader | `src/weights/mod.rs` | `WeightManifest::load_strict` | `WeightError::UnsupportedFormat` |
| loader | `src/weights/mod.rs` | `WeightManifest::read_tensor` | `WeightError::UnsupportedFormat` |
| llama | `src/arch/llama/mod.rs` | `Factory::parse_config` | `ConfigError::UnsupportedArchitecture` |
| llama | `src/arch/llama/mod.rs` | `Factory::build` | `LoadError::Config(UnsupportedArchitecture)` |
| llama | `src/arch/llama/mod.rs` | `Factory::map_safetensors_key` | `WeightDisposition::Reject` |
| llama, tranche 4 | `src/arch/llama/mod.rs` | `Factory::map_gguf_key` | `WeightDisposition::Reject` |
| qwen3 | `src/arch/qwen3/mod.rs` | `Factory::parse_config` | `ConfigError::UnsupportedArchitecture` |
| qwen3 | `src/arch/qwen3/mod.rs` | `Factory::build` | `LoadError::Config(UnsupportedArchitecture)` |
| qwen3 | `src/arch/qwen3/mod.rs` | `Factory::map_safetensors_key` | `WeightDisposition::Reject` |
| qwen3, tranche 4 | `src/arch/qwen3/mod.rs` | `Factory::map_gguf_key` | `WeightDisposition::Reject` |

The cache item owns the sealed `LayerCache` trait, `LayerCacheSpec`, `CacheStep`, full and
rotating storage, snapshot fingerprint contents, and transaction rollback. Snapshot and
transaction structs currently hold no staged arrays; no constructor can expose them.
The storage metadata implementations describe only their declared counters; the rotating
item must add prefix-aware retained metadata with real storage. No bounded-memory claim is made.

The loader owns manifest entries and strict application through `StateProjection`. Each
architecture owns `ParsedConfig`, its private JSON `WireConfig`, all model math, and its
key dispositions. Wire configs currently retain raw fields without resolving defaults.
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
`--features prototype-adapter,oracle-hooks` for the decision 11 migration.
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
`NOT RUN` until the architecture and loader items land. Running it with `--ignored
--nocapture` reports each unavailable fixture and fails if any remain unavailable, so an
explicit run cannot count placeholder errors as a passing happy path. Remove the ignore
when those implementations are integrated.

Decision 11 is blocked in this worktree: `Cache::new` and `Cache::step` still return
placeholder errors, and `CacheStep` has neither `info` nor `evaluate_and_commit`.
The wave worktree has the same placeholders. The cache item's real implementation has
those transaction APIs but exposes no logical K/V read accessor; its `LayerCache::arrays`
returns private storage buffers, which cannot substitute for temporal views of rotating
caches. The `oracle-hooks` feature is declared and required by parity/sentinel, but its
module and entry points remain unimplemented pending cache integration and a logical
per-layer K/V accessor. The feature is off by default; no protected test sources changed.

Changed files are implementation-owned: workspace/package manifests, `mlx-lm` sources and
handoff documentation, adapter test entry points, the moved Qwen3 fixture, the removed
`mlx-lm-utils` crate, and the migrated `examples/lm` consumer. Protected-oracle files: none.
Ledger files: none. No commit was created.
