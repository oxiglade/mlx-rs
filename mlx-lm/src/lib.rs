//! Synchronous, single-request text generation for Llama and Qwen3 with MLX.
//!
//! The crate provides strict loading, tokenization and textual chat templates,
//! streaming generation, sampling and model-bound KV caches. It targets macOS on
//! Apple silicon with native MLX; Rust 1.88 is the workspace minimum. Select CPU
//! or Metal through `mlx-rs` on the owning thread. Features do not select devices,
//! and CPU execution can still require Metal during native initialization.
//! Python MLX is an oracle dependency, not a runtime dependency of this crate.
//! Version 0.32.0 remains unpublished pending David's explicit go.
//!
//! # Supported sources
//!
//! - [`Model::from_dir`]: Llama and Qwen3 safetensors, single-file or indexed shards,
//!   floating weights including BF16 and affine quantization. Local configurations
//!   admit full/sliding attention and none, linear or Llama3 RoPE scaling within
//!   the architecture's validated profile.
//! - [`Model::from_gguf`]: Llama and Qwen3 in F32, F16, Q4_0, Q4_1 and Q8_0.
//!   GGUF models must be dense and biasless, with full causal attention, full-head
//!   RoPE and absent/none or linear scaling; frequency-factor tensors are rejected.
//!   The three converted affine forms use group size 32, 4 or 8 bits and F16
//!   companions. BF16 GGUF is unqualified. Admission examines converted arrays;
//!   other original encodings can succeed through core fallback conversion without
//!   being qualified. There is no original-format allowlist guarantee.
//! - With `hf-hub`, `Model::from_hub` resolves a commit and selected safetensors and
//!   JSON/tokenizer assets before using the local loader. It does not execute
//!   remote Python code or load Hub GGUF files.
//!
//! Tiny GGUF fixtures establish writer/reader self-consistency, mapping,
//! orientation and arithmetic; Llama naming also matches the pinned exporter.
//! Third-party compatibility comes from the separate local real-checkpoint report
//! for TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF Q4_0 and
//! ggml-org/Qwen3-0.6B-GGUF Q8_0, not the tiny fixtures.
//!
//! mlx-lm 0.31.3 supplies the architecture and generation oracle and a Llama-family GGUF exporter; it does NOT supply a GGUF model loader. GGUF ingestion is qualified by Python MLX container conversion plus a reviewed reference bridge and independent NumPy decoding.
//!
//! # GGUF tokenizer pairing
//!
//! Supply a [`Tokenizer`] loaded by [`Tokenizer::from_dir`] from the exact original
//! model revision. Embedded token strings do not reconstruct a tokenizer. Pairing
//! checks vocabulary/added-token bounds, configured BOS/EOS bounds and embedded
//! special-token IDs and vocabulary spellings. BOS agreement is checked only when
//! the supplied tokenizer declares a BOS. These checks cannot prove identity:
//! ordinary tokens can differ despite matching sizes and special IDs.
//! Retain the GGUF producer/source revision, hashes of `tokenizer.json`,
//! `tokenizer_config.json`, `config.json` and `generation_config.json` when present,
//! and a canonical prompt's original IDs. [`Tokenizer::encode`] adds no special
//! tokens; text generation uses the post-processor unless the original string
//! starts with the configured BOS string.
//!
//! # Features and Hub receipts
//!
//! Default features are empty. `hf-hub` enables the Hub API and optional `hf-hub`
//! and `sha2` dependencies. `oracle-hooks` exposes hidden conformance observations
//! independently. Neither is activated in the base library.
//!
//! With `hf-hub`, `HubOptions` selects revision, cache and offline mode;
//! `Model::hub_provenance` reports repository, requested revision and resolved
//! commit. Online requests are anonymous and downloads are pinned to that commit.
//! An online error does not silently fall back to offline loading. Offline mode
//! constructs no network client. An unreceipted third-party cache is an offline miss.
//!
//! Receipts prove completeness and recorded commit identity, not tamper resistance.
//! Content hashes are recorded once at download completion as provenance. Offline
//! and reuse validation check presence, paths, sizes and recorded commit/receipt
//! identity without rereading file contents. Same-size content changes are
//! undetected, and a cache writer can also change the receipt. Cache contents must
//! remain unchanged while loading.
//!
//! # Ownership
//!
//! [`Model`], [`Generation`], [`Cache`] and [`CacheSnapshot`] are `!Send` and `!Sync`.
//! Create, use and drop them on one OS thread inside the chosen device/stream scope.
//! [`GenerationOptions`] and [`GenerationEvent`] are owned, sendable values.
//! The packaged `examples/worker.rs` shows a Metal worker with bounded request/event
//! channels, worker-local error conversion, receiver-drop cancellation and orderly
//! shutdown. Run it with `cargo run -p mlx-lm --release --example worker -- /path/to/checkpoint`.
//!
//! ```no_run
//! use mlx_lm::{GenerationEvent, GenerationOptions, Model, Prompt};
//! use mlx_rs::{with_device, Device};
//!
//! with_device(Device::gpu(), || -> Result<(), Box<dyn std::error::Error>> {
//!     let mut model = Model::from_dir("/path/to/checkpoint")?;
//!     for event in model.generate(Prompt::Text("The capital of France is"), GenerationOptions::default())? {
//!         if let GenerationEvent::Token { text, .. } = event? {
//!             print!("{text}");
//!         }
//!     }
//!     Ok(())
//! })?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Drop performs no inference; a generation error yields once and fuses the iterator.
//! Each successful step publishes evaluated cache, RNG and history only after text
//! preparation succeeds. After token `j` (counted from 1), a prompt of length `P`
//! has `P + j - 1` forwarded tokens in the cache. Reuse requires the same model,
//! compatible policy, an exact cached prefix and a nonempty uncached suffix.
//! Snapshots exclude RNG, decoder and iterator state. Rotation can retain capacity
//! plus a prefill chunk, snapshots retain old storage, and decoding retains IDs and
//! emitted text. Constant total host memory and zero allocation are not promised.
//! Arbitrary chat templates have an unresolved resource-exhaustion limitation and
//! are not safely sandboxed; arbitrary cache mutation is not exposed.
//!
//! # Errors
//!
//! Errors preserve typed boundaries and source chains. Public error enums are
//! non-exhaustive; include fallback match arms.
//!
//! | Error | Boundary |
//! | --- | --- |
//! | [`ConfigError`] | Invalid or unsupported configuration and GGUF metadata. |
//! | [`WeightError`] | Discovery, container parsing, keys, shapes, dtypes and strict assignment. |
//! | [`LoadError`] | Local/GGUF loading and tokenizer pairing; wraps config, weight, tokenizer, template and native errors. |
//! | [`TokenizerError`], [`ChatTemplateError`] | Assets, encoding/decoding, template selection/rendering. |
//! | [`CacheError`] | Model identity, layout, policy, transaction or restoration. |
//! | [`SamplingError`] | Options, logits and candidate support. |
//! | [`InferenceError`] | Architecture execution, cache and native failures. |
//! | [`GenerationError`] | Prompt/stop/reuse/length validation and typed runtime wrappers; evaluation exceptions stay `GenerationError::Exception`. |
//! | [`HubError`] | Repository/revision, missing files, unsafe paths, receipt integrity, offline miss, transport and wrapped local load errors. The type exists without `hf-hub`; the API-error variant requires it. |
//!
//! # Throughput and open work
//!
//! On the same machine and weights (Metal, Qwen3-0.6B-4bit, 128-token prompt,
//! 256 greedy tokens), Rust measured **38.2 tokens/s** against upstream Python's
//! **125.5 tokens/s**, a **3.3× gap** under investigation in its own throughput
//! tranche. Reports: `/Users/ci/hub/scratch/mlx-lm/t4/bench/c-baseline-metal.json`
//! and `/Users/ci/hub/scratch/mlx-lm/t4/bench/c-ablation-metal.json`.
//! Ruling C is closed as “gap attributed elsewhere; no pipeline”; the no-lookahead
//! ablation measured 103.0 tokens/s and explained 9.6% of the excess time per token.
//!
//! Demand-gated capabilities include vision/images/audio and other multimodal
//! inputs; training, adapters, LoRA and distributed execution; batching, continuous
//! generation and speculative decoding; a server and async generation; custom
//! architecture registration; checkpoint conversion/upload; persisted prompt and
//! quantized KV caches; MXFP4, BitNet, AWQ/GPTQ conversion and activation
//! quantization; structured tool/document messages; XTC, logit bias and custom
//! sampling callables.
//!
//! The repository's `mlx-lm/README.md` contains the capability table and credits;
//! `mlx-lm/SKELETON.md` records completed tranches. Evidence and report hashes are
//! in `ledger/mlx-lm-release-identity.json`, separate from the protected parity
//! ledger's historical run-status fields.
#![deny(missing_docs)]

mod arch;
mod cache;
mod config;
mod error;
mod model;
mod sampling;
mod stop;
mod tokenizer;
mod weights;

pub use cache::{Cache, CacheInfo, CacheKind, CacheOptions, CachePolicy, CacheSnapshot};
pub use config::{
    AffineQuantization, AttentionKind, Config, LayerQuantization, ModelType, ParameterPath,
    QuantizationConfig, RopeConfig, RopeScaling, TransformerDimensions,
};
pub use error::{
    CacheError, ChatTemplateError, ConfigError, GenerationError, HubError, InferenceError,
    LoadError, SamplingError, TokenizerError, WeightError,
};
pub use model::{
    FinishReason, Generation, GenerationEvent, GenerationOptions, Model, Prompt,
    RepetitionPenaltyOptions, StopPolicy, StopTokenPolicy,
};
#[cfg(feature = "hf-hub")]
pub use model::{HubOptions, HubProvenance};
pub use sampling::{AdditivePenaltyOptions, MinPOptions, SamplerOptions};
pub use tokenizer::{ChatContinuation, ChatTemplateOptions, Message, Role, TokenId, Tokenizer};

#[cfg(feature = "oracle-hooks")]
#[doc(hidden)]
#[allow(missing_docs)]
pub mod oracle_hooks;

macro_rules! assert_not_impl_any {
    ($type:ty: $($trait:path),+ $(,)?) => {
        const _: fn() = || {
            trait AmbiguousIfImpl<T: ?Sized> { fn check() {} }
            impl<T: ?Sized> AmbiguousIfImpl<()> for T {}
            $({
                struct Invalid;
                impl<T: ?Sized + $trait> AmbiguousIfImpl<Invalid> for T {}
            })+
            let _ = <$type as AmbiguousIfImpl<_>>::check;
        };
    };
}
assert_not_impl_any!(Model: Send, Sync);
assert_not_impl_any!(Generation<'static>: Send, Sync);
assert_not_impl_any!(Cache: Send, Sync);
assert_not_impl_any!(CacheSnapshot: Send, Sync);
