//! Typed failures at language-model boundaries.
use mlx_rs::{error::Exception, io::GgufError, Dtype};
use std::path::PathBuf;

/// Configuration parsing and validation failed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ConfigError {
    /// A required JSON field is absent.
    #[error("missing configuration field {field}")]
    MissingField {
        /// Configuration or option field name.
        field: String,
    },
    /// A numeric field violates its domain.
    #[error("invalid {field}: {reason}")]
    InvalidNumericField {
        /// Configuration or option field name.
        field: String,
        /// Stable local explanation of the rejection.
        reason: String,
    },
    /// The attention pattern has the wrong length.
    #[error("expected {expected} attention layers, got {actual}")]
    InvalidLayerPattern {
        /// Required value or layout.
        expected: usize,
        /// Observed value or layout.
        actual: usize,
    },
    /// The architecture is not admitted.
    #[error("unsupported architecture: {0}")]
    UnsupportedArchitecture(String),
    /// The quantization scheme is not admitted.
    #[error("unsupported quantization: {0}")]
    UnsupportedQuantization(String),
    /// The rotary embedding scheme is not admitted.
    #[error("unsupported RoPE: {0}")]
    UnsupportedRope(String),
    /// Preserves the source diagnostic.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Preserves the source diagnostic.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

/// Weight discovery or strict keyed assignment failed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum WeightError {
    /// A required weights file is absent.
    #[error("missing weights file: {0:?}")]
    MissingFile(PathBuf),
    /// An indexed shard is absent.
    #[error("missing shard: {0:?}")]
    MissingShard(PathBuf),
    /// An expected tensor is absent.
    #[error("missing tensor: {0}")]
    MissingKey(String),
    /// An external tensor has no approved disposition.
    #[error("unexpected tensor: {0}")]
    UnexpectedKey(String),
    /// A tensor name appears more than once.
    #[error("duplicate tensor: {0}")]
    DuplicateTensor(String),
    /// A shard disagrees with its index.
    #[error("conflicting index entry for {key} in {shard:?}")]
    ConflictingIndex {
        /// External tensor or parameter key.
        key: String,
        /// Shard associated with the conflicting entry.
        shard: PathBuf,
    },
    /// A tensor shape differs from its parameter slot.
    #[error("shape mismatch for {key}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// External tensor or parameter key.
        key: String,
        /// Required value or layout.
        expected: Vec<usize>,
        /// Observed value or layout.
        actual: Vec<usize>,
    },
    /// A tensor dtype differs from its parameter slot.
    #[error("dtype mismatch for {key}: expected {expected:?}, got {actual:?}")]
    DtypeMismatch {
        /// External tensor or parameter key.
        key: String,
        /// Required value or layout.
        expected: Dtype,
        /// Observed value or layout.
        actual: Dtype,
    },
    /// A checkpoint dtype is not admitted.
    #[error("unsupported dtype {dtype} for {key}")]
    UnsupportedDtype {
        /// External tensor or parameter key.
        key: String,
        /// Unsupported checkpoint dtype.
        dtype: String,
    },
    /// The weight representation is not admitted.
    #[error("unsupported weights format: {0}")]
    UnsupportedFormat(String),
    /// Preserves the source diagnostic.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Preserves the source diagnostic.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Preserves the source diagnostic.
    #[error(transparent)]
    Safetensors(#[from] safetensors::SafeTensorError),
    /// Preserves the source diagnostic.
    #[error(transparent)]
    Gguf(#[from] GgufError),
    /// Preserves the source diagnostic.
    #[error(transparent)]
    Exception(#[from] Exception),
    /// Preserves the source diagnostic.
    #[error(transparent)]
    Projection(#[from] mlx_rs::error::StateProjectionError),
}

/// Local model loading failed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LoadError {
    /// A required model asset is absent.
    #[error("missing model file: {0:?}")]
    MissingFile(PathBuf),
    /// Preserves the source diagnostic.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Preserves the source diagnostic.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// Preserves the source diagnostic.
    #[error(transparent)]
    Weights(#[from] WeightError),
    /// Preserves the source diagnostic.
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
    /// Preserves the source diagnostic.
    #[error(transparent)]
    ChatTemplate(#[from] ChatTemplateError),
    /// Preserves the source diagnostic.
    #[error(transparent)]
    Gguf(#[from] GgufError),
    /// Preserves the source diagnostic.
    #[error(transparent)]
    Exception(#[from] Exception),
}

/// Tokenizer parsing, encoding, or decoding failed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TokenizerError {
    /// A required tokenizer asset is absent.
    #[error("missing tokenizer file: {0:?}")]
    MissingFile(PathBuf),
    /// EOS metadata is not a token ID or token ID list.
    #[error("invalid EOS tokens: {0}")]
    InvalidEos(String),
    /// Preserves the source diagnostic.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Preserves the source diagnostic.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Preserves the source diagnostic.
    #[error(transparent)]
    Tokenizer(#[from] tokenizers::Error),
    /// Preserves the source diagnostic.
    #[error(transparent)]
    ChatTemplate(#[from] ChatTemplateError),
}

/// Chat template selection or rendering failed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ChatTemplateError {
    /// No selected template is available.
    #[error("missing chat template")]
    MissingTemplate,
    /// The conversation cannot be continued.
    #[error("chat continuation requires a nonempty final message")]
    IncompatibleContinuation,
    /// The template removed the final message content.
    #[error("final message is absent from rendered chat")]
    FinalMessageNotFound,
    /// Preserves the source diagnostic.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Preserves the source diagnostic.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Preserves the source diagnostic.
    #[error(transparent)]
    Minijinja(#[from] minijinja::Error),
}

/// Cache shape, transaction, or restoration failed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CacheError {
    /// The snapshot belongs to an incompatible cache.
    #[error("cache fingerprint mismatch")]
    FingerprintMismatch,
    /// The resolved request policy differs from the supplied cache policy.
    #[error("cache policy mismatch")]
    PolicyMismatch,
    /// A layer update has an incompatible shape.
    #[error("cache shape mismatch at layer {layer}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Zero-based cache layer index.
        layer: usize,
        /// Required value or layout.
        expected: Vec<usize>,
        /// Observed value or layout.
        actual: Vec<usize>,
    },
    /// The operation conflicts with the cache transaction state.
    #[error("invalid cache state: {0}")]
    InvalidState(String),
    /// The requested cache policy is not implemented.
    #[error("unsupported cache policy: {0}")]
    UnsupportedPolicy(String),
    /// Preserves the source diagnostic.
    #[error(transparent)]
    Exception(#[from] Exception),
}

/// Sampling options or token selection failed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SamplingError {
    /// No token can be selected from an empty vocabulary.
    #[error("sampling vocabulary is empty")]
    EmptyVocabulary,
    /// Min-p requests more retained candidates than the vocabulary contains.
    #[error("min_tokens_to_keep {min_tokens_to_keep} exceeds vocabulary {vocabulary_size}")]
    MinTokensToKeepExceedsVocabulary {
        /// Requested minimum support size.
        min_tokens_to_keep: usize,
        /// Runtime vocabulary size.
        vocabulary_size: usize,
    },
    /// Temperature is negative or non-finite.
    #[error("invalid temperature: {0}")]
    InvalidTemperature(f32),
    /// A filtering probability is outside its range.
    #[error("invalid {field} probability: {value}")]
    InvalidProbability {
        /// Configuration or option field name.
        field: String,
        /// Rejected numeric option value.
        value: f32,
    },
    /// Top-k exceeds the runtime vocabulary.
    #[error("top-k {top_k} exceeds vocabulary {vocabulary_size}")]
    TopKExceedsVocabulary {
        /// Requested support size.
        top_k: usize,
        /// Runtime vocabulary size.
        vocabulary_size: usize,
    },
    /// The repetition penalty is non-positive or non-finite.
    #[error("invalid repetition penalty: {0}")]
    InvalidRepetitionPenalty(f32),
    /// A presence or frequency coefficient is non-finite.
    #[error("invalid additive penalty: {0}")]
    InvalidAdditivePenalty(f32),
    /// Logits do not contain one row of the runtime vocabulary.
    #[error("invalid logits shape: expected [1, {vocabulary_size}], got {shape:?}")]
    InvalidLogitsShape {
        /// Observed logits shape.
        shape: Vec<i32>,
        /// Runtime vocabulary size.
        vocabulary_size: usize,
    },
    /// Logits are not f16, bf16, or f32.
    #[error("invalid logits dtype: {0:?}")]
    InvalidLogitsDtype(Dtype),
    /// Logits contain NaN or positive infinity; negative infinity is a valid mask.
    #[error("logits contain NaN or positive infinity")]
    InvalidLogitsValue,
    /// No finite candidate remains available for sampling.
    #[error("sampling support is empty")]
    EmptySupport,
    /// Preserves the scalar-conversion diagnostic.
    #[error(transparent)]
    Conversion(#[from] mlx_rs::error::ConversionError),
    /// The requested sampling mode is not implemented.
    #[error("unsupported sampling mode: {0}")]
    UnsupportedMode(String),
    /// Preserves the source diagnostic.
    #[error(transparent)]
    Exception(#[from] Exception),
}

/// Architecture execution or evaluation failed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum InferenceError {
    /// The prompt contains no tokens.
    #[error("prompt is empty")]
    EmptyPrompt,
    /// The architecture forward path is not implemented.
    #[error("unsupported inference architecture: {0}")]
    UnsupportedArchitecture(String),
    /// Preserves the source diagnostic.
    #[error(transparent)]
    Exception(#[from] Exception),
    /// Preserves the source diagnostic.
    #[error(transparent)]
    Cache(#[from] CacheError),
}

/// Prompt preparation or streaming generation failed.
///
/// Construction validates sampler options, repetition/presence/frequency penalties,
/// stop strings and tokens, the encoded prompt, borrowed-cache compatibility and
/// prefix, then checked sequence lengths, before allocating or mutating a cache.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum GenerationError {
    /// The encoded prompt contains no tokens.
    #[error("prompt is empty")]
    EmptyPrompt,
    /// A configured stop string is empty.
    #[error("stop string at index {index} is empty")]
    EmptyStopString {
        /// Zero-based index in the supplied stop-string list.
        index: usize,
    },
    /// A prompt token is outside the runtime vocabulary.
    #[error("prompt token {token_id:?} at index {index} is outside vocabulary {vocabulary_size}")]
    PromptTokenOutOfRange {
        /// Zero-based index in the full encoded prompt.
        index: usize,
        /// Rejected token identifier.
        token_id: crate::TokenId,
        /// Runtime vocabulary size.
        vocabulary_size: usize,
    },
    /// An effective stop token is outside the runtime vocabulary.
    #[error("stop token {token_id:?} is outside vocabulary {vocabulary_size}")]
    StopTokenOutOfRange {
        /// Rejected token identifier.
        token_id: crate::TokenId,
        /// Runtime vocabulary size.
        vocabulary_size: usize,
    },
    /// The full prompt does not extend the exact represented cache prefix.
    #[error("cache prefix mismatch: matched {matched} tokens, cached {cached}, prompt {prompt}")]
    CachePrefixMismatch {
        /// Number of equal leading token IDs.
        matched: usize,
        /// Number of tokens represented by the cache.
        cached: usize,
        /// Number of tokens in the full prompt.
        prompt: usize,
    },
    /// The full prompt is already represented and has no suffix to prefill.
    #[error("prompt has no uncached tokens")]
    NoUncachedTokens,
    /// Adding the prompt length and sample limit overflows `usize`.
    #[error("generation length overflow")]
    LengthOverflow,
    /// The requested sequence exceeds MLX dimension or absolute-position limits.
    #[error("sequence length {length} exceeds limit {limit}")]
    SequenceTooLong {
        /// Requested sequence length.
        length: usize,
        /// Maximum representable length.
        limit: usize,
    },
    /// Preserves the source diagnostic.
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
    /// Sampling graph-construction or scalar-conversion failure.
    /// Runtime MLX evaluation failures instead use [`Self::Exception`].
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    /// Preserves the source diagnostic.
    #[error(transparent)]
    Cache(#[from] CacheError),
    /// Preserves the source diagnostic.
    #[error(transparent)]
    Inference(#[from] InferenceError),
    /// Preserves the original exception from any runtime MLX evaluation failure,
    /// including joint evaluation, sampling-only evaluation, and the first sample.
    #[error(transparent)]
    Exception(#[from] Exception),
}

/// Optional Hub resolution, download, or local loading failed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HubError {
    /// The requested revision is unavailable offline.
    #[error("offline cache miss for {repo}@{revision}")]
    OfflineCacheMiss {
        /// Requested repository identifier.
        repo: String,
        /// Requested revision.
        revision: String,
    },
    /// The requested revision is invalid.
    #[error("invalid revision: {0}")]
    InvalidRevision(String),
    /// Preserves the source diagnostic.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Preserves the source diagnostic.
    #[error(transparent)]
    Load(#[from] LoadError),
    /// Preserves the Hub client diagnostic.
    #[cfg(feature = "hf-hub")]
    #[error(transparent)]
    Api(#[from] hf_hub::api::sync::ApiError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_error_payloads_and_display() {
        let cases = [
            (
                GenerationError::EmptyStopString { index: 2 },
                "stop string at index 2 is empty",
            ),
            (
                GenerationError::PromptTokenOutOfRange {
                    index: 3,
                    token_id: 64.into(),
                    vocabulary_size: 64,
                },
                "prompt token TokenId(64) at index 3 is outside vocabulary 64",
            ),
            (
                GenerationError::StopTokenOutOfRange {
                    token_id: 65.into(),
                    vocabulary_size: 64,
                },
                "stop token TokenId(65) is outside vocabulary 64",
            ),
            (
                GenerationError::CachePrefixMismatch {
                    matched: 2,
                    cached: 8,
                    prompt: 4,
                },
                "cache prefix mismatch: matched 2 tokens, cached 8, prompt 4",
            ),
            (
                GenerationError::NoUncachedTokens,
                "prompt has no uncached tokens",
            ),
            (
                GenerationError::LengthOverflow,
                "generation length overflow",
            ),
            (
                GenerationError::SequenceTooLong {
                    length: 9,
                    limit: 8,
                },
                "sequence length 9 exceeds limit 8",
            ),
        ];
        for (error, expected) in cases {
            assert_eq!(error.to_string(), expected);
        }
    }

    #[test]
    fn sampling_and_cache_error_payloads_and_wrapping() {
        let cases = [
            (
                SamplingError::InvalidAdditivePenalty(f32::INFINITY),
                "invalid additive penalty: inf",
            ),
            (
                SamplingError::InvalidLogitsShape {
                    shape: vec![2, 64],
                    vocabulary_size: 64,
                },
                "invalid logits shape: expected [1, 64], got [2, 64]",
            ),
            (
                SamplingError::InvalidLogitsDtype(Dtype::Int32),
                "invalid logits dtype: Int32",
            ),
            (
                SamplingError::InvalidLogitsValue,
                "logits contain NaN or positive infinity",
            ),
            (SamplingError::EmptySupport, "sampling support is empty"),
        ];
        for (error, expected) in cases {
            let wrapped = GenerationError::from(error);
            assert!(matches!(wrapped, GenerationError::Sampling(_)));
            assert_eq!(wrapped.to_string(), expected);
        }
        let wrapped = GenerationError::from(CacheError::PolicyMismatch);
        assert!(matches!(
            wrapped,
            GenerationError::Cache(CacheError::PolicyMismatch)
        ));
        assert_eq!(wrapped.to_string(), "cache policy mismatch");
    }

    #[test]
    fn scalar_conversion_preserves_the_original_error() {
        let source = mlx_rs::error::ConversionError::NotScalar { actual: 2 };
        let expected = source.to_string();
        let error = GenerationError::from(SamplingError::from(source));
        assert_eq!(error.to_string(), expected);
        assert!(matches!(
            error,
            GenerationError::Sampling(SamplingError::Conversion(
                mlx_rs::error::ConversionError::NotScalar { actual: 2 }
            ))
        ));
    }
}
