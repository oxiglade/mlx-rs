//! Synchronous, thread-bound language model inference with typed configuration.
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
#[cfg(feature = "hf-hub")]
pub use model::HubOptions;
pub use model::{
    FinishReason, Generation, GenerationEvent, GenerationOptions, Model, Prompt,
    RepetitionPenaltyOptions, StopPolicy, StopTokenPolicy,
};
pub use sampling::{AdditivePenaltyOptions, MinPOptions, SamplerOptions};
pub use tokenizer::{ChatContinuation, ChatTemplateOptions, Message, Role, TokenId, Tokenizer};

#[cfg(feature = "oracle-hooks")]
#[doc(hidden)]
#[allow(missing_docs)]
pub mod oracle_hooks;

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
