//! Synchronous, thread-bound language model inference with typed configuration.
#![deny(missing_docs)]

#[allow(dead_code)] // Interfaces are claimed by the implementation items listed in SKELETON.md.
mod arch;
#[allow(dead_code)]
mod cache;
#[allow(dead_code)]
mod config;
mod error;
mod model;
mod sampling;
mod tokenizer;
#[allow(dead_code)]
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
#[cfg(feature = "hf-hub")]
pub use model::HubOptions;
pub use model::{
    FinishReason, Generation, GenerationEvent, GenerationOptions, Model, Prompt,
    RepetitionPenaltyOptions, StopTokenPolicy,
};
pub use sampling::{MinPOptions, SamplerOptions};
pub use tokenizer::{ChatContinuation, ChatTemplateOptions, Message, Role, TokenId, Tokenizer};

/// Temporary implementation access for the unchanged prototype regression adapters.
#[cfg(feature = "prototype-adapter")]
#[doc(hidden)]
#[allow(missing_docs)]
pub mod legacy;

struct NotYetImplemented(&'static str);
impl std::fmt::Display for NotYetImplemented {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: not yet implemented in this tranche", self.0)
    }
}

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
