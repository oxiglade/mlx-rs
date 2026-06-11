use mlx_rs::{
    nn::log_softmax,
    ops::{
        argsort_axis, cumsum,
        indexing::{argmax_axis, take_along_axis, Ellipsis, IndexOp, NewAxis},
        multiply, r#where, softmax_axis,
    },
    random::categorical,
    Array, Dtype,
};

use crate::error::Error;

/// Sampling strategy. Variants are mutually exclusive; nucleus
/// (`top_p`) can never silently override greedy because `Greedy` has no
/// `p` field. Default is `Greedy` (argmax).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Sampler {
    /// Argmax. No temperature, no top-p.
    #[default]
    Greedy,
    /// Categorical sampling at the given temperature (`> 0.0`).
    Temperature(f32),
    /// Categorical with a nucleus (top-p) mask after temperature scaling.
    TopP { temperature: f32, p: f32 },
}

impl Sampler {
    /// `None` for [`Self::Greedy`], else the temperature.
    pub fn temperature(self) -> Option<f32> {
        match self {
            Self::Greedy => None,
            Self::Temperature(t) | Self::TopP { temperature: t, .. } => Some(t),
        }
    }

    /// `Some(p)` for [`Self::TopP`]; `None` otherwise.
    pub fn top_p(self) -> Option<f32> {
        match self {
            Self::TopP { p, .. } => Some(p),
            _ => None,
        }
    }
}

/// Per-decode-loop sampler with cached scalar constants. Avoids
/// per-token host→device allocations for inverse-temperature, top-p
/// threshold, and the −∞ mask. Constants bind to the logits dtype on
/// first sample and are reused for every subsequent call.
pub struct SamplerState {
    sampler: Sampler,
    /// `1.0 / temperature` at logits dtype. `None` for greedy.
    inv_temp: Option<Array>,
    /// `top_p` threshold (f32). `None` when top-p is disabled.
    top_p_threshold: Option<Array>,
    /// `-inf` at logits dtype, the nucleus mask sentinel.
    neg_inf: Option<Array>,
    /// Dtype `inv_temp`/`neg_inf` were built against.
    bound_dtype: Option<Dtype>,
}

impl SamplerState {
    pub fn new(sampler: Sampler) -> Self {
        let top_p_threshold = sampler.top_p().map(Array::from_f32);
        Self {
            sampler,
            inv_temp: None,
            top_p_threshold,
            neg_inf: None,
            bound_dtype: None,
        }
    }

    /// The configured strategy. Lets speculative-decode callers branch
    /// on greedy vs top-p without re-deriving it.
    pub fn sampler(&self) -> Sampler {
        self.sampler
    }

    /// Temperature-scaled, optionally top-p-masked log-probabilities.
    /// Used by MTP rejection sampling to compare draft vs verify
    /// distributions. Errors on `Greedy` (no temperature).
    pub fn masked_log_probs(
        &mut self,
        logits: &Array,
        keep_mask: Option<&Array>,
    ) -> Result<Array, Error> {
        if matches!(self.sampler, Sampler::Greedy) {
            return Err(Error::config(
                "masked_log_probs: Sampler::Greedy has no temperature; greedy callers use argmax",
            ));
        }
        let dtype = logits.dtype();
        self.bind(dtype)?;
        let inv_temp = self
            .inv_temp
            .as_ref()
            .expect("inv_temp populated by bind()");
        masked_temp_log_probs(logits, keep_mask, inv_temp)
    }

    /// Sample one token from `logits`, reusing cached scalars.
    pub fn sample(&mut self, logits: &Array) -> Result<Array, Error> {
        if matches!(self.sampler, Sampler::Greedy) {
            return Ok(argmax_axis(logits, -1, None)?);
        }
        let dtype = logits.dtype();
        self.bind(dtype)?;
        let inv_temp = self
            .inv_temp
            .as_ref()
            .expect("inv_temp populated by bind()");
        let scaled = multiply(logits, inv_temp)?;
        match self.sampler {
            Sampler::Greedy => unreachable!("greedy handled above"),
            Sampler::Temperature(_) => Ok(categorical(&scaled, None, None, None)?),
            Sampler::TopP { .. } => self.top_p_sample(&scaled),
        }
    }

    fn bind(&mut self, dtype: Dtype) -> Result<(), Error> {
        if self.bound_dtype == Some(dtype) {
            return Ok(());
        }
        let t = self
            .sampler
            .temperature()
            .expect("bind() called on Sampler::Greedy");
        self.inv_temp = Some(Array::from_f32(1.0_f32 / t).as_dtype(dtype)?);
        self.neg_inf = Some(Array::from_f32(f32::NEG_INFINITY).as_dtype(dtype)?);
        self.bound_dtype = Some(dtype);
        Ok(())
    }

    fn top_p_sample(&self, logits: &Array) -> Result<Array, Error> {
        let p = self
            .top_p_threshold
            .as_ref()
            .expect("top_p_sample called without top_p set");
        let neg_inf = self.neg_inf.as_ref().expect("neg_inf populated by bind()");
        let probs = softmax_axis(logits, -1, true)?;
        let order = argsort_axis(&probs.negative()?, -1)?;
        let sorted_probs = take_along_axis(&probs, &order, -1)?;
        let csum = cumsum(&sorted_probs, -1, false, false)?;
        let keep = csum.subtract(&sorted_probs)?.lt(p)?;
        let sorted_logits = take_along_axis(logits, &order, -1)?;
        let masked = r#where(&keep, &sorted_logits, neg_inf)?;
        let sorted_pick = categorical(&masked, None, None, None)?;
        let pick = sorted_pick.index((Ellipsis, NewAxis));
        let token = take_along_axis(&order, &pick, -1)?;
        Ok(token.squeeze_axes(&[-1])?)
    }
}

/// Temperature-scaled, optionally top-p-masked log-probabilities.
/// `keep_mask` (when present) sets excluded ids to `-inf` before the
/// log-softmax, so they get `-inf` log-prob.
pub(crate) fn masked_temp_log_probs(
    logits: &Array,
    keep_mask: Option<&Array>,
    inv_temp: &Array,
) -> Result<Array, Error> {
    let scaled = multiply(logits, inv_temp)?;
    let masked = if let Some(mask) = keep_mask {
        let neg_inf = Array::from_f32(f32::NEG_INFINITY).as_dtype(scaled.dtype())?;
        r#where(mask, &scaled, &neg_inf)?
    } else {
        scaled
    };
    Ok(log_softmax(&masked, -1)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_p_returns_top_token() {
        let logits = Array::from_slice(&[-10.0_f32, -10.0, -10.0, 5.0, -10.0], &[1, 5]);
        let mut state = SamplerState::new(Sampler::TopP {
            temperature: 1.0,
            p: 0.5,
        });
        for _ in 0..16 {
            let tok = state.sample(&logits).unwrap();
            assert_eq!(tok.item::<u32>(), 3);
        }
    }

    #[test]
    fn greedy_returns_argmax() {
        let logits = Array::from_slice(&[0.1_f32, 0.9, 0.2], &[1, 3]);
        let mut state = SamplerState::new(Sampler::Greedy);
        assert_eq!(state.sample(&logits).unwrap().item::<u32>(), 1);
    }

    #[test]
    fn temperature_caches_across_calls() {
        let logits = Array::from_slice(&[0.1_f32, 0.9, 0.2], &[1, 3]);
        let mut state = SamplerState::new(Sampler::Temperature(0.7));
        for _ in 0..32 {
            assert!(state.sample(&logits).unwrap().item::<u32>() < 3);
        }
    }
}
