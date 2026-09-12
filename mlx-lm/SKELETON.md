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
| foundation | `src/config/mod.rs` | `Config::validate` | `ConfigError::UnsupportedArchitecture` |
| foundation | `src/model.rs` | `Model::from_dir` | `LoadError::Config(UnsupportedArchitecture)` |
| loader, tranche 4 | `src/model.rs` | `Model::from_gguf` | `LoadError::Weights(UnsupportedFormat)` |
| foundation, tranche 4 | `src/model.rs` | `Model::from_hub` | `HubError::Load(Config(UnsupportedArchitecture))` |
| foundation, tranche 3 | `src/model.rs` | `Model::generate` | `GenerationError::Inference(UnsupportedArchitecture)` |
| foundation, tranche 3 | `src/model.rs` | `Generation::next` | One typed inference error, then fused exhaustion |
| foundation, tranche 3 | `src/model.rs` | `RepetitionPenaltyOptions::validate` | `SamplingError::UnsupportedMode` |
| foundation, tranche 3 | `src/sampling.rs` | `SamplerOptions::validate` | `SamplingError::UnsupportedMode` |
| foundation, tranche 3 | `src/sampling.rs` | `MinPOptions::validate` | `SamplingError::UnsupportedMode` |
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

The loader owns manifest entries and strict application through `StateProjection`. Each
architecture owns `ParsedConfig`, its private JSON `WireConfig`, all model math, and its
key dispositions. Wire configs currently retain raw fields without resolving defaults.
`DecoderModel` is declared exactly as designed; neither factory constructs an implementation.
The registry contains only `llama::Factory` and `qwen3::Factory`.

The tokenizer item owns `src/tokenizer/` and the moved Qwen3 template fixture. Local JSON
loading, encode/decode, scalar/list EOS precedence, compiled string templates, pycompat,
textual roles, and safe continuation are implemented. Named-template selection, additional
special-token cases, invalid EOS coverage, and exact oracle chat/tokenizer parity still need
qualification. Missing templates fail during rendering, while invalid selected Jinja fails
during loading. No remote code or tokenizer HTTP feature is enabled.

## Prototype removal

The `prototype-adapter` feature is off by default. Explicit `parity` and `sentinel` test
entry points alias their own crate as `mlx_lm` and re-export only `legacy::{cache, models}`.
This lets the existing test sources retain their imports byte-for-byte, without exposing
old modules at the library root. Run them with `--features prototype-adapter`.
Auto-discovery is disabled so Cargo cannot also compile the old entry points directly.
The comparator's fixture/mutation tests share the parity binary and therefore also require
this feature until the adapter is ported. New integration tests must be registered explicitly.

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

Changed files are implementation-owned: workspace/package manifests, `mlx-lm` sources and
handoff documentation, adapter test entry points, the moved Qwen3 fixture, the removed
`mlx-lm-utils` crate, and the migrated `examples/lm` consumer. Protected-oracle files: none.
Ledger files: none. No commit was created.
