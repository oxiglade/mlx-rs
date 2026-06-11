//! `ClippableLinear`: a `Linear` wrapped in optional input/output clamps.
//!
//! Gemma 4's audio tower sets `use_clipped_linears: true`, so each projection
//! carries scalar `input_min`/`input_max`/`output_min`/`output_max` buffers and
//! applies `clip(x, in) → linear → clip(y, out)`. The clamps are LIVE (dropping
//! them silently corrupts the encoder). Weights are bf16 (unquantized).

use mlx_rs::{
    builder::Builder,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::clip,
    Array,
};

use crate::error::Error;

/// `Linear` (key `linear`) plus optional scalar clamp buffers.
#[derive(Debug, Clone, ModuleParameters)]
pub struct ClippableLinear {
    #[param]
    pub linear: nn::Linear,
    #[param]
    pub input_min: Param<Option<Array>>,
    #[param]
    pub input_max: Param<Option<Array>>,
    #[param]
    pub output_min: Param<Option<Array>>,
    #[param]
    pub output_max: Param<Option<Array>>,
}

impl ClippableLinear {
    pub fn new(inp: i32, out: i32) -> Result<Self, Error> {
        // Clip buffers init to ±inf (clip is a no-op) so the parameter walk
        // exposes them for binding; the checkpoint overwrites with real bounds.
        // A `None`-init `Param<Option<Array>>` is skipped by the walk, so the
        // buffer would never load → no clamping → wrong output.
        Ok(Self {
            linear: nn::LinearBuilder::new(inp, out).bias(false).build()?,
            input_min: Param::new(Some(Array::from_f32(f32::NEG_INFINITY))),
            input_max: Param::new(Some(Array::from_f32(f32::INFINITY))),
            output_min: Param::new(Some(Array::from_f32(f32::NEG_INFINITY))),
            output_max: Param::new(Some(Array::from_f32(f32::INFINITY))),
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Error> {
        let x = Self::clamp(x, &self.input_min, &self.input_max)?;
        let y = self.linear.forward(&x)?;
        Self::clamp(&y, &self.output_min, &self.output_max)
    }

    /// Clamp by whichever bounds are present (both, one, or neither).
    fn clamp(
        x: &Array,
        min: &Param<Option<Array>>,
        max: &Param<Option<Array>>,
    ) -> Result<Array, Error> {
        let dt = x.dtype();
        let lo = min.value.as_ref().map(|a| a.as_dtype(dt)).transpose()?;
        let hi = max.value.as_ref().map(|a| a.as_dtype(dt)).transpose()?;
        match (lo.as_ref(), hi.as_ref()) {
            (None, None) => Ok(x.clone()),
            (Some(lo), Some(hi)) => Ok(clip(x, (lo, hi))?),
            (Some(lo), None) => Ok(clip(x, (lo, ()))?),
            (None, Some(hi)) => Ok(clip(x, ((), hi))?),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    use super::*;

    #[test]
    fn clamps_input_and_output() {
        let mut l = ClippableLinear::new(2, 2).unwrap();
        // Identity-ish weight so the output reflects the (clamped) input.
        l.linear.weight = Param::new(Array::from_slice(&[1.0_f32, 0.0, 0.0, 1.0], &[2, 2]));
        l.input_min = Param::new(Some(Array::from_f32(-1.0)));
        l.input_max = Param::new(Some(Array::from_f32(1.0)));
        let x = Array::from_slice(&[5.0_f32, -5.0], &[1, 2]);
        let y = l.forward(&x).unwrap();
        // Input clamped to [-1, 1] before the matmul.
        assert_eq!(y.as_slice::<f32>(), &[1.0, -1.0]);
    }

    #[test]
    fn no_clamp_when_unset() {
        let mut l = ClippableLinear::new(2, 2).unwrap();
        l.linear.weight = Param::new(Array::from_slice(&[1.0_f32, 0.0, 0.0, 1.0], &[2, 2]));
        let x = Array::from_slice(&[5.0_f32, -5.0], &[1, 2]);
        let y = l.forward(&x).unwrap();
        assert_eq!(y.as_slice::<f32>(), &[5.0, -5.0]);
    }
}
