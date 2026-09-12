pub use crate::error::SamplingError;
use std::num::NonZeroUsize;

/// Reusable temperature and probability-filter settings.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SamplerOptions {
    /// Zero selects greedy argmax.
    pub temperature: f32,
    /// None disables nucleus filtering.
    pub top_p: Option<f32>,
    /// None disables top-k filtering.
    pub top_k: Option<NonZeroUsize>,
    /// None disables min-p filtering.
    pub min_p: Option<MinPOptions>,
}
/// Minimum relative probability and retained support size.
#[derive(Debug, Clone)]
pub struct MinPOptions {
    /// Minimum probability relative to the most likely token.
    pub probability: f32,
    /// Minimum number of candidates retained by min-p.
    pub min_tokens_to_keep: NonZeroUsize,
}
impl Default for SamplerOptions {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: None,
            top_k: None,
            min_p: None,
        }
    }
}
impl SamplerOptions {
    /// Validates sampler ranges against the runtime vocabulary.
    pub fn validate(&self, _vocabulary_size: usize) -> Result<(), SamplingError> {
        Err(SamplingError::UnsupportedMode(
            crate::NotYetImplemented("sampling validation").to_string(),
        ))
    }
}
impl MinPOptions {
    /// Validates the relative probability and retained support against the vocabulary.
    pub fn validate(&self, _vocabulary_size: usize) -> Result<(), SamplingError> {
        Err(SamplingError::UnsupportedMode(
            crate::NotYetImplemented("min-p validation").to_string(),
        ))
    }
}
