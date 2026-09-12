pub use crate::error::GenerationError;
use crate::{
    arch::DecoderModel, CacheOptions, Config, LoadError, SamplerOptions, TokenId, Tokenizer,
};
use std::{marker::PhantomData, num::NonZeroUsize, path::Path, rc::Rc};

/// A loaded decoder and its resolved tokenizer and configuration.
pub struct Model {
    decoder: Box<dyn DecoderModel>,
    tokenizer: Tokenizer,
    config: Config,
}
impl Model {
    /// Loads a local model with strict safetensors weight matching.
    pub fn from_dir(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        let _ = path;
        Err(crate::ConfigError::UnsupportedArchitecture(
            crate::NotYetImplemented("local model loading").to_string(),
        )
        .into())
    }
    /// Consumes a typed GGUF container with a separately constructed tokenizer.
    pub fn from_gguf(file: mlx_rs::io::GgufFile, tokenizer: Tokenizer) -> Result<Self, LoadError> {
        let _ = (file, tokenizer);
        Err(crate::WeightError::UnsupportedFormat(
            crate::NotYetImplemented("GGUF model loading").to_string(),
        )
        .into())
    }
    /// Resolves an allowlisted Hub snapshot and loads it through the local loader.
    #[cfg(feature = "hf-hub")]
    pub fn from_hub(repo: &str, options: HubOptions) -> Result<Self, crate::HubError> {
        let _ = (repo, options);
        Err(
            LoadError::Config(crate::ConfigError::UnsupportedArchitecture(
                crate::NotYetImplemented("Hub loading").to_string(),
            ))
            .into(),
        )
    }
    /// Borrows the resolved model configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }
    /// Borrows the model-owned tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    /// Creates a synchronous single-request streaming generator.
    pub fn generate<'model>(
        &'model mut self,
        prompt: Prompt<'_>,
        options: GenerationOptions,
    ) -> Result<Generation<'model>, GenerationError> {
        let _ = (&self.decoder, prompt, options);
        Err(crate::InferenceError::UnsupportedArchitecture(
            crate::NotYetImplemented("generation").to_string(),
        )
        .into())
    }
}
/// Text or already-tokenized input for one request.
pub enum Prompt<'a> {
    /// Text to encode with the model tokenizer.
    Text(&'a str),
    /// Token IDs supplied by the caller.
    Tokens(&'a [TokenId]),
}
/// Reusable limits, sampling, stopping, and cache options.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GenerationOptions {
    /// Maximum number of sampled tokens.
    pub max_tokens: NonZeroUsize,
    /// Maximum number of prompt tokens per prefill step.
    pub prefill_chunk_size: NonZeroUsize,
    /// Temperature and support-filter settings.
    pub sampling: SamplerOptions,
    /// Optional history-dependent repetition penalty.
    pub repetition_penalty: Option<RepetitionPenaltyOptions>,
    /// Source and precedence of stop tokens.
    pub stop_tokens: StopTokenPolicy,
    /// Per-request cache policy.
    pub cache: CacheOptions,
}
impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_tokens: NonZeroUsize::new(256).unwrap_or(NonZeroUsize::MIN),
            prefill_chunk_size: NonZeroUsize::new(2048).unwrap_or(NonZeroUsize::MIN),
            sampling: SamplerOptions::default(),
            repetition_penalty: None,
            stop_tokens: StopTokenPolicy::Tokenizer,
            cache: CacheOptions::default(),
        }
    }
}
/// Selects the source of the effective stop-token set.
#[derive(Debug, Clone)]
pub enum StopTokenPolicy {
    /// Uses the tokenizer's resolved EOS set.
    Tokenizer,
    /// Extends the tokenizer's EOS set with explicit IDs.
    TokenizerPlus(Vec<TokenId>),
    /// Replaces the tokenizer's EOS set with explicit IDs.
    Exact(Vec<TokenId>),
}
/// History-dependent logits penalty settings.
#[derive(Debug, Clone)]
pub struct RepetitionPenaltyOptions {
    /// Positive finite multiplier applied to repeated tokens.
    pub penalty: f32,
    /// Maximum number of previous tokens considered.
    pub context_size: NonZeroUsize,
}
impl RepetitionPenaltyOptions {
    /// Validates the repetition penalty before generation.
    pub fn validate(&self) -> Result<(), crate::SamplingError> {
        Err(crate::SamplingError::UnsupportedMode(
            crate::NotYetImplemented("repetition validation").to_string(),
        ))
    }
}
/// Why the final successful event ended generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// A stop token was sampled.
    Stop,
    /// The token limit was reached.
    Length,
}
/// One sampled token and its stable decoded text delta.
pub struct GenerationEvent {
    /// Sampled token, including a final stop token.
    pub token_id: TokenId,
    /// Stable decoded text since the preceding event; it may be empty.
    pub text: String,
    /// Present only on the final successful event.
    pub finish_reason: Option<FinishReason>,
}
/// A synchronous stream fused after its final event or first error.
pub struct Generation<'model> {
    _decoder: PhantomData<&'model mut dyn DecoderModel>,
    _thread_bound: PhantomData<Rc<()>>,
    finished: bool,
}
impl Iterator for Generation<'_> {
    type Item = Result<GenerationEvent, GenerationError>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        self.finished = true;
        Some(Err(crate::InferenceError::UnsupportedArchitecture(
            crate::NotYetImplemented("generation step").to_string(),
        )
        .into()))
    }
}
impl std::iter::FusedIterator for Generation<'_> {}

/// Revision and local-cache controls for optional Hub loading.
#[cfg(feature = "hf-hub")]
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct HubOptions {
    /// Optional requested revision; immutable commit SHAs are recommended.
    pub revision: Option<String>,
    /// Requires all assets to be available without downloading.
    pub offline: bool,
    /// Optional cache directory override.
    pub cache_dir: Option<std::path::PathBuf>,
}
