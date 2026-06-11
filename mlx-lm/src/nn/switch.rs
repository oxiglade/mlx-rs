//! Switched (expert-routed) FFN primitives for MoE models.
//!
//! [`SwitchLinear`] is the packed-expert linear; [`QuantizedSwitchLinear`]
//! its `gather_qmm` counterpart (auto-built via `try_into_quantized`).
//! [`SplitSwitchFfn`] is the qwen3.6-MoE layout (separate gate/up/down,
//! each `[E, H, D]`), parameterised over a [`SwitchActivation`].

use std::sync::OnceLock;

use mlx_rs::error::Exception;
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::{ModuleParameters, Param};
use mlx_rs::ops::indexing::take_axis;
use mlx_rs::ops::{
    argsort, expand_dims_axes, gather_mm, gather_qmm, quantize, sum_axes, swap_axes, unflatten,
};
use mlx_rs::quantization::{MaybeQuantized, Quantizable};
use mlx_rs::Array;

use crate::activations::{geglu, swiglu, GegluCache, SwigluCache};
use crate::error::Error;

/// Index count above which tokens are pre-sorted by expert id so
/// `gather_mm` hits contiguous expert rows. Below it, the argsort cost
/// outweighs the gather-locality win.
pub(crate) const SORT_THRESHOLD: usize = 2048;

// ─── Primitives ───────────────────────────────────────────────────

/// Dense per-expert linear. Weight `[num_experts, output_dims, input_dims]`.
#[derive(Debug, Clone, ModuleParameters)]
pub struct SwitchLinear {
    #[param]
    pub weight: Param<Array>,
    #[param]
    pub bias: Param<Option<Array>>,
}

impl SwitchLinear {
    pub fn new(
        input_dims: i32,
        output_dims: i32,
        num_experts: i32,
        bias: bool,
    ) -> Result<Self, Error> {
        let scale = (1.0 / input_dims as f32).sqrt();
        let weight = mlx_rs::random::uniform::<_, f32>(
            -scale,
            scale,
            &[num_experts, output_dims, input_dims],
            None,
        )?;
        let bias_arr = if bias {
            Some(Array::zeros::<f32>(&[num_experts, output_dims])?)
        } else {
            None
        };
        Ok(Self {
            weight: Param::new(weight),
            bias: Param::new(bias_arr),
        })
    }

    /// `gather_mm` apply. `x` is `[..., 1, 1, input_dims]`; `indices` is
    /// `[..., top_k]`. Unquantised path (bf16 checkpoints).
    pub fn apply(&self, x: &Array, indices: &Array, sorted: bool) -> Result<Array, Error> {
        let w = swap_axes(self.weight.as_ref(), -1, -2)?;
        let mut y = gather_mm(x, &w, None, Some(indices), Some(sorted))?;
        if let Some(b) = self.bias.as_ref() {
            let b_gather = take_axis(b, indices, 0)?;
            y = y.add(&expand_dims_axes(&b_gather, &[-2])?)?;
        }
        Ok(y)
    }
}

impl Quantizable for SwitchLinear {
    type Quantized = QuantizedSwitchLinear;
    type QuantizationError = Error;

    fn try_into_quantized(
        self,
        group_size: i32,
        bits: i32,
    ) -> Result<Self::Quantized, Self::QuantizationError> {
        QuantizedSwitchLinear::try_from_switch_linear(self, group_size, bits)
    }
}

/// Quantised per-expert linear: packed weight + per-group scales/biases.
#[derive(Debug, Clone, ModuleParameters)]
pub struct QuantizedSwitchLinear {
    pub group_size: i32,
    pub bits: i32,

    #[param]
    pub scales: Param<Array>,
    #[param]
    pub biases: Param<Array>,
    #[param]
    pub inner: SwitchLinear,
}

impl QuantizedSwitchLinear {
    pub fn try_from_switch_linear(
        linear: SwitchLinear,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, Error> {
        let (packed_w, scales, biases) = quantize(linear.weight.as_ref(), group_size, bits)?;
        Ok(Self {
            group_size,
            bits,
            scales: Param::new(scales),
            biases: Param::new(biases),
            inner: SwitchLinear {
                weight: Param::new(packed_w),
                bias: linear.bias,
            },
        })
    }

    /// `gather_qmm` apply. Hot path on every quantised MoE token.
    pub fn apply(&self, x: &Array, indices: &Array, sorted: bool) -> Result<Array, Error> {
        let mut y = gather_qmm(
            x,
            self.inner.weight.as_ref(),
            self.scales.as_ref(),
            Some(self.biases.as_ref()),
            None,
            Some(indices),
            Some(true),
            Some(self.group_size),
            Some(self.bits),
            Some(sorted),
        )?;
        if let Some(b) = self.inner.bias.as_ref() {
            let b_gather = take_axis(b, indices, 0)?;
            y = y.add(&expand_dims_axes(&b_gather, &[-2])?)?;
        }
        Ok(y)
    }
}

/// Inline dispatch for `MaybeQuantized<SwitchLinear>`. Hot path; the
/// `#[inline]` lets the optimiser fold the match away.
#[inline]
pub(crate) fn apply_proj(
    proj: &MaybeQuantized<SwitchLinear>,
    x: &Array,
    indices: &Array,
    sorted: bool,
) -> Result<Array, Error> {
    match proj {
        MaybeQuantized::Original(d) => d.apply(x, indices, sorted),
        MaybeQuantized::Quantized(q) => q.apply(x, indices, sorted),
    }
}

// ─── SwitchActivation ─────────────────────────────────────────────

/// Per-element `activation(gate) * up` for one MoE expert output.
pub trait SwitchActivation: ModuleParameters {
    fn activate(&mut self, gate: &Array, up: &Array) -> Result<Array, Error>;
}

/// `silu(gate) * up` — qwen3.6-MoE activation. Owns a compiled-graph
/// cache so the fused kernel is built once per layer and reused across
/// every decode step instead of recompiled per token.
#[derive(Debug, Default, ModuleParameters)]
pub struct SwigluActivation {
    cache: SwigluCache,
}

impl SwitchActivation for SwigluActivation {
    fn activate(&mut self, gate: &Array, up: &Array) -> Result<Array, Error> {
        Ok(swiglu(&mut self.cache, gate, up)?)
    }
}

/// `gelu_approx(gate) * up` — Gemma 4 MoE expert activation. Owns a
/// compiled-graph cache, built once per layer and reused across decode
/// steps.
#[derive(Debug, Default, ModuleParameters)]
pub struct GegluActivation {
    cache: GegluCache,
}

impl SwitchActivation for GegluActivation {
    fn activate(&mut self, gate: &Array, up: &Array) -> Result<Array, Error> {
        Ok(geglu(&mut self.cache, gate, up)?)
    }
}

/// down_proj + sum-combine (the 2-launch path), shared across the
/// sorted/unsorted forwards.
fn combine_with_weights(
    activated: &Array,
    down_proj: &MaybeQuantized<SwitchLinear>,
    indices: &Array,
    top_k_weights: &Array,
    sorted: bool,
) -> Result<Array, Error> {
    let y = apply_proj(down_proj, activated, indices, sorted)?;
    let y = y.squeeze_axes(&[-2])?;
    let w = expand_dims_axes(top_k_weights, &[-1])?;
    Ok(sum_axes(&w.multiply(&y)?, &[-2], false)?)
}

// ─── SplitSwitchFfn: qwen3.6-MoE layout ───────────────────────────

/// Qwen3.6-MoE FFN: separate `gate_proj` + `up_proj`, each `[E, H, D]`.
#[derive(Debug, ModuleParameters)]
pub struct SplitSwitchFfn<A: SwitchActivation> {
    #[param]
    pub gate_proj: MaybeQuantized<SwitchLinear>,
    #[param]
    pub up_proj: MaybeQuantized<SwitchLinear>,
    #[param]
    pub down_proj: MaybeQuantized<SwitchLinear>,
    pub activation: A,
    /// Experts routed per token (config `num_experts_per_tok`).
    top_k: i32,
    /// Cached 0-D `top_k` constant for `gather_sort`'s `floor_divide`.
    top_k_arr: OnceLock<Array>,
}

impl<A: SwitchActivation + Default> SplitSwitchFfn<A> {
    pub fn new(
        input_dims: i32,
        hidden_dims: i32,
        num_experts: i32,
        num_experts_per_tok: i32,
        bias: bool,
    ) -> Result<Self, Error> {
        Ok(Self {
            gate_proj: MaybeQuantized::Original(SwitchLinear::new(
                input_dims,
                hidden_dims,
                num_experts,
                bias,
            )?),
            up_proj: MaybeQuantized::Original(SwitchLinear::new(
                input_dims,
                hidden_dims,
                num_experts,
                bias,
            )?),
            down_proj: MaybeQuantized::Original(SwitchLinear::new(
                hidden_dims,
                input_dims,
                num_experts,
                bias,
            )?),
            activation: A::default(),
            top_k: num_experts_per_tok,
            top_k_arr: OnceLock::new(),
        })
    }
}

impl<A: SwitchActivation> SplitSwitchFfn<A> {
    /// Full MoE forward returning `[..., K, D]` per-expert outputs.
    pub fn forward(&mut self, x: &Array, indices: &Array) -> Result<Array, Error> {
        let x_exp = expand_dims_axes(x, &[-2, -3])?;
        let do_sort = indices.size() >= SORT_THRESHOLD;
        let k = self.top_k;
        let top_k_arr = self.top_k_arr.get_or_init(|| Array::from_int(k));
        let sorted = do_sort
            .then(|| gather_sort(&x_exp, indices, top_k_arr))
            .transpose()?;
        let (x_in, idx_in): (&Array, &Array) = match sorted.as_ref() {
            Some((xs, idxs, _)) => (xs, idxs),
            None => (&x_exp, indices),
        };

        let gate = apply_proj(&self.gate_proj, x_in, idx_in, do_sort)?;
        let up = apply_proj(&self.up_proj, x_in, idx_in, do_sort)?;
        let activated = self.activation.activate(&gate, &up)?;
        let mut y = apply_proj(&self.down_proj, &activated, idx_in, do_sort)?;

        if let Some((_, _, inv)) = sorted.as_ref() {
            y = scatter_unsort(&y, inv, indices.shape())?;
        }

        Ok(y.squeeze_axes(&[-2])?)
    }

    /// Forward + down-proj + weighted sum-combine in one call.
    pub fn forward_with_combine(
        &mut self,
        x: &Array,
        indices: &Array,
        top_k_weights: &Array,
    ) -> Result<Array, Error> {
        let do_sort = indices.size() >= SORT_THRESHOLD;
        if do_sort {
            let y = self.forward(x, indices)?;
            let w = expand_dims_axes(top_k_weights, &[-1])?;
            return Ok(sum_axes(&w.multiply(&y)?, &[-2], false)?);
        }

        let x_exp = expand_dims_axes(x, &[-2, -3])?;
        let gate = apply_proj(&self.gate_proj, &x_exp, indices, false)?;
        let up = apply_proj(&self.up_proj, &x_exp, indices, false)?;
        let activated = self.activation.activate(&gate, &up)?;
        combine_with_weights(&activated, &self.down_proj, indices, top_k_weights, false)
    }
}

impl<A: SwitchActivation> Quantizable for SplitSwitchFfn<A> {
    type Quantized = Self;
    type QuantizationError = Error;

    fn try_into_quantized(self, group_size: i32, bits: i32) -> Result<Self, Error> {
        Ok(Self {
            gate_proj: self.gate_proj.try_into_quantized(group_size, bits)?,
            up_proj: self.up_proj.try_into_quantized(group_size, bits)?,
            down_proj: self.down_proj.try_into_quantized(group_size, bits)?,
            activation: self.activation,
            top_k: self.top_k,
            top_k_arr: self.top_k_arr,
        })
    }
}

// ─── Sort helpers ─────────────────────────────────────────────────

/// Sort tokens by expert id for contiguous `gather_mm` access. Returns
/// `(sorted_x, sorted_indices, inv_order_to_unsort)`.
pub(crate) fn gather_sort(
    x: &Array,
    indices: &Array,
    top_k_arr: &Array,
) -> Result<(Array, Array, Array), Exception> {
    let flat_idx = indices.flatten(0, -1)?;
    let order = argsort(&flat_idx)?;
    let inv_order = argsort(&order)?;
    let x_flat = x.flatten(0, -3)?;
    let row_idx = order.floor_divide(top_k_arr)?;
    let x_sorted = take_axis(&x_flat, &row_idx, 0)?;
    let idx_sorted = take_axis(&flat_idx, &order, 0)?;
    Ok((x_sorted, idx_sorted, inv_order))
}

pub(crate) fn scatter_unsort(x: &Array, inv_order: &Array, shape: &[i32]) -> Result<Array, Error> {
    let unsorted = take_axis(x, inv_order, 0)?;
    Ok(unflatten(&unsorted, 0, shape)?)
}
