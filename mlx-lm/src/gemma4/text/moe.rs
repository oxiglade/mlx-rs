//! Gemma 4 MoE block: a top-k expert router + a packed expert FFN.
//!
//! Only the 26b-a4b variant (`enable_moe_block`) uses these. The router
//! is RMS-norm (with `scale·hidden^-0.5`) → linear → fused top-k softmax
//! (the shared [`crate::nn::router_topk`] kernel) → ×`per_expert_scale`.
//! Experts are a split-layout [`SplitSwitchFfn`] with the GeGLU
//! activation, matching the checkpoint's
//! `experts.switch_glu.{gate,up,down}_proj` keys.

use std::sync::OnceLock;

use mlx_rs::builder::Builder;
use mlx_rs::fast::{self, MetalKernel};
use mlx_rs::macros::{ModuleParameters, Quantizable};
use mlx_rs::module::{Module, Param};
use mlx_rs::nn;
use mlx_rs::ops::indexing::take_axis;
use mlx_rs::quantization::MaybeQuantized;
use mlx_rs::Array;

use crate::error::Error;
use crate::nn::router_topk::{make_router_topk_kernel, router_topk};
use crate::nn::switch::{GegluActivation, SplitSwitchFfn};

/// Process-wide cached router top-k kernel handle, shared across every MoE
/// layer (mirrors the qwen3.5 accessor).
fn router_topk_kernel() -> Result<&'static MetalKernel, Error> {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    if let Some(k) = KERNEL.get() {
        return Ok(k);
    }
    let built = make_router_topk_kernel()?;
    let _ = KERNEL.set(built);
    Ok(KERNEL.get().expect("just set"))
}

/// Expert router: `rms_norm(x, scale·root_size) → proj → top-k softmax ×
/// per_expert_scale`. `root_size = hidden^-0.5`.
#[derive(Debug, ModuleParameters, Quantizable)]
pub struct Router {
    eps: f32,
    top_k: i32,
    root_size: f32,

    #[quantizable]
    #[param]
    pub proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub scale: Param<Array>,
    #[param]
    pub per_expert_scale: Param<Array>,

    /// `scale · root_size`, staged in `scale`'s dtype once (weights are
    /// load-once). Keeps `rms_norm(x_bf16, w)` from promoting to f32.
    scaled_weight: OnceLock<Array>,
    /// `per_expert_scale` cast to the scores dtype once.
    per_expert_scale_cast: OnceLock<Array>,
}

impl Router {
    pub fn new(hidden_size: i32, num_experts: i32, top_k: i32, eps: f32) -> Result<Self, Error> {
        Ok(Self {
            eps,
            top_k,
            root_size: (hidden_size as f32).powf(-0.5),
            proj: MaybeQuantized::Original(
                nn::LinearBuilder::new(hidden_size, num_experts)
                    .bias(false)
                    .build()?,
            ),
            scale: Param::new(Array::ones::<f32>(&[hidden_size])?),
            per_expert_scale: Param::new(Array::ones::<f32>(&[num_experts])?),
            scaled_weight: OnceLock::new(),
            per_expert_scale_cast: OnceLock::new(),
        })
    }

    /// `x [B, L, D]` → `(top_k_indices [B, L, K] u32, top_k_weights [B, L, K])`.
    pub fn forward(&mut self, x: &Array) -> Result<(Array, Array), Error> {
        let weight = self.scaled_weight.get_or_init(|| {
            let scale = self.scale.as_ref();
            let root = Array::from_f32(self.root_size)
                .as_dtype(scale.dtype())
                .expect("root_size cast cannot fail");
            scale
                .multiply(&root)
                .expect("scale × root_size cannot fail")
        });
        let normed = fast::rms_norm(x, Some(weight), self.eps)?;
        let scores = self.proj.forward(&normed)?;

        let num_experts = *scores.shape().last().expect("scores has trailing dim");
        let (indices, weights) =
            router_topk(router_topk_kernel()?, &scores, num_experts, self.top_k)?;

        let per_expert_scale_cast = self.per_expert_scale_cast.get_or_init(|| {
            self.per_expert_scale
                .as_ref()
                .as_dtype(weights.dtype())
                .expect("per_expert_scale cast cannot fail")
        });
        let gathered = take_axis(per_expert_scale_cast, &indices, 0)?;
        let weights = weights.multiply(&gathered)?;
        Ok((indices, weights))
    }
}

/// Routed experts: a GeGLU [`SplitSwitchFfn`] over `num_experts`.
#[derive(Debug, ModuleParameters, Quantizable)]
pub struct Experts {
    #[quantizable]
    #[param]
    pub switch_glu: SplitSwitchFfn<GegluActivation>,
}

impl Experts {
    pub fn new(
        hidden_size: i32,
        moe_intermediate_size: i32,
        num_experts: i32,
        top_k: i32,
    ) -> Result<Self, Error> {
        Ok(Self {
            switch_glu: SplitSwitchFfn::<GegluActivation>::new(
                hidden_size,
                moe_intermediate_size,
                num_experts,
                top_k,
                false,
            )?,
        })
    }

    /// `x [B, L, D]`, `indices/weights [B, L, K]` → `[B, L, D]`.
    pub fn forward(&mut self, x: &Array, indices: &Array, weights: &Array) -> Result<Array, Error> {
        self.switch_glu.forward_with_combine(x, indices, weights)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    use super::*;
    use mlx_rs::transforms::eval;

    #[test]
    fn router_selects_top_k() {
        let (hidden, experts, k) = (16, 8, 2);
        let mut router = Router::new(hidden, experts, k, 1e-6).unwrap();
        let x = mlx_rs::random::uniform::<_, f32>(-1.0, 1.0, &[1, 3, hidden], None).unwrap();
        let (idx, w) = router.forward(&x).unwrap();
        eval([&idx, &w]).unwrap();
        assert_eq!(idx.shape(), &[1, 3, k]);
        assert_eq!(w.shape(), &[1, 3, k]);
        let min_w = w.min(None).unwrap().item::<f32>();
        assert!(min_w >= 0.0, "weights must be non-negative: {min_w}");
    }

    #[test]
    fn experts_round_trip_shape() {
        let (hidden, moe_int, experts, k) = (16, 8, 8, 2);
        let mut router = Router::new(hidden, experts, k, 1e-6).unwrap();
        let mut moe = Experts::new(hidden, moe_int, experts, k).unwrap();
        let x = mlx_rs::random::uniform::<_, f32>(-1.0, 1.0, &[1, 3, hidden], None).unwrap();
        let (idx, w) = router.forward(&x).unwrap();
        let out = moe.forward(&x, &idx, &w).unwrap();
        eval([&out]).unwrap();
        assert_eq!(out.shape(), &[1, 3, hidden]);
    }
}
