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
    pub fn validate(&self, vocabulary_size: usize) -> Result<(), SamplingError> {
        validate_vocabulary(vocabulary_size)?;
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(SamplingError::InvalidTemperature(self.temperature));
        }
        if let Some(top_p) = self.top_p {
            if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
                return Err(SamplingError::InvalidProbability {
                    field: "top_p".into(),
                    value: top_p,
                });
            }
        }
        if let Some(top_k) = self.top_k {
            if top_k.get() > vocabulary_size {
                return Err(SamplingError::TopKExceedsVocabulary {
                    top_k: top_k.get(),
                    vocabulary_size,
                });
            }
        }
        if let Some(min_p) = &self.min_p {
            min_p.validate(vocabulary_size)?;
        }
        Ok(())
    }
}
impl MinPOptions {
    /// Validates the relative probability and retained support against the vocabulary.
    pub fn validate(&self, vocabulary_size: usize) -> Result<(), SamplingError> {
        validate_vocabulary(vocabulary_size)?;
        if !self.probability.is_finite() || !(0.0..=1.0).contains(&self.probability) {
            return Err(SamplingError::InvalidProbability {
                field: "min_p".into(),
                value: self.probability,
            });
        }
        if self.min_tokens_to_keep.get() > vocabulary_size {
            return Err(SamplingError::MinTokensToKeepExceedsVocabulary {
                min_tokens_to_keep: self.min_tokens_to_keep.get(),
                vocabulary_size,
            });
        }
        Ok(())
    }
}

fn validate_vocabulary(vocabulary_size: usize) -> Result<(), SamplingError> {
    if vocabulary_size == 0 {
        return Err(SamplingError::EmptyVocabulary);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_ranges_apply_even_to_greedy() -> Result<(), SamplingError> {
        SamplerOptions::default().validate(64)?;
        assert!(matches!(
            SamplerOptions::default().validate(0),
            Err(SamplingError::EmptyVocabulary)
        ));
        for temperature in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1] {
            let options = SamplerOptions {
                temperature,
                ..Default::default()
            };
            assert!(matches!(
                options.validate(64),
                Err(SamplingError::InvalidTemperature(_))
            ));
        }
        for top_p in [f32::NAN, f32::INFINITY, -0.1, 0.0, 1.1] {
            let options = SamplerOptions {
                top_p: Some(top_p),
                ..Default::default()
            };
            assert!(
                matches!(options.validate(64), Err(SamplingError::InvalidProbability { field, .. }) if field == "top_p")
            );
        }
        for top_p in [f32::MIN_POSITIVE, 0.5, 1.0] {
            SamplerOptions {
                top_p: Some(top_p),
                ..Default::default()
            }
            .validate(64)?;
        }
        let options = SamplerOptions {
            top_k: NonZeroUsize::new(64),
            ..Default::default()
        };
        options.validate(64)?;
        assert!(matches!(
            options.validate(63),
            Err(SamplingError::TopKExceedsVocabulary {
                top_k: 64,
                vocabulary_size: 63
            })
        ));
        let min_p = MinPOptions {
            probability: f32::NAN,
            min_tokens_to_keep: NonZeroUsize::MIN,
        };
        assert!(matches!(
            SamplerOptions {
                min_p: Some(min_p),
                ..Default::default()
            }
            .validate(64),
            Err(SamplingError::InvalidProbability { .. })
        ));
        Ok(())
    }

    #[test]
    fn min_p_range_and_retained_support() -> Result<(), SamplingError> {
        let mut options = MinPOptions {
            probability: 0.0,
            min_tokens_to_keep: NonZeroUsize::MIN,
        };
        assert!(matches!(
            options.validate(0),
            Err(SamplingError::EmptyVocabulary)
        ));
        for probability in [0.0, 0.5, 1.0] {
            options.probability = probability;
            options.validate(1)?;
        }
        for probability in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1, 1.1] {
            options.probability = probability;
            assert!(
                matches!(options.validate(64), Err(SamplingError::InvalidProbability { field, .. }) if field == "min_p")
            );
        }
        options.probability = 0.1;
        options.min_tokens_to_keep = NonZeroUsize::new(64).ok_or(SamplingError::EmptyVocabulary)?;
        options.validate(64)?;
        assert!(matches!(
            options.validate(63),
            Err(SamplingError::MinTokensToKeepExceedsVocabulary {
                min_tokens_to_keep: 64,
                vocabulary_size: 63
            })
        ));
        Ok(())
    }
}
