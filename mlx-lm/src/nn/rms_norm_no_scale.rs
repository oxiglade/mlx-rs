//! RMSNorm without a learnable scale.
//!
//! Mirrors [`mlx_rs::nn::RmsNorm`] but passes `None` for the weight, so it
//! normalises without a per-channel gain. Used where a checkpoint carries a
//! norm with no `weight` tensor (e.g. Gemma 4's value norm).

use mlx_rs::{error::Exception, macros::ModuleParameters, module::Module, Array};

/// RMSNorm with no learnable scale: `x / sqrt(mean(x²) + eps)`.
///
/// Holds only `eps` — no parameters — so it is inert in the module
/// parameter walk.
#[derive(Debug, Clone, ModuleParameters)]
pub struct RmsNormNoScale {
    /// Numerical-stability epsilon.
    pub eps: f32,
}

impl RmsNormNoScale {
    pub fn new(eps: f32) -> Self {
        Self { eps }
    }
}

impl Module<&Array> for RmsNormNoScale {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        mlx_rs::fast::rms_norm(x, None, self.eps)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    use super::*;
    use mlx_rs::ops::{mean_axes, rsqrt};

    #[test]
    fn matches_manual_rms_no_scale() {
        let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 4]);
        let eps = 1e-6;
        let mut norm = RmsNormNoScale::new(eps);
        let got = norm.forward(&x).unwrap();

        // manual: x * rsqrt(mean(x², axis=-1) + eps)
        let ms = mean_axes(x.multiply(&x).unwrap(), &[-1][..], true).unwrap();
        let denom = rsqrt(ms.add(Array::from_f32(eps)).unwrap()).unwrap();
        let want = x.multiply(&denom).unwrap();

        let max = got
            .subtract(&want)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(max < 1e-5, "RmsNormNoScale diverges from manual: {max}");
    }
}
