//! Fast implementations of commonly used multi-op functions.

use std::ffi::CStr;

use crate::error::Result;
use crate::utils::guard::Guarded;
use crate::utils::IntoOption;
use crate::{Array, Stream};
use mlx_internal_macros::{default_device, generate_macro};

/// Optimized implementation of `NN.RoPE`.
#[allow(clippy::too_many_arguments)]
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn rope_device<'a>(
    #[named] array: impl AsRef<Array>,
    #[named] dimensions: i32,
    #[named] traditional: bool,
    #[optional] base: impl Into<Option<f32>>,
    #[named] scale: f32,
    #[named] offset: i32,
    #[optional] freqs: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let base = base.into();
    let base = mlx_sys::mlx_optional_float {
        value: base.unwrap_or(0.0),
        has_value: base.is_some(),
    };
    let freqs = freqs.into();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_rope(
            res,
            array.as_ref().as_ptr(),
            dimensions,
            traditional,
            base,
            scale,
            offset,
            freqs
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Optimized implementation of `NN.RoPE` with dynamic (array) offset.
///
/// This variant allows specifying the offset as an array, enabling different
/// offsets for different positions in the input.
///
/// # Params
///
/// - `array`: Input array
/// - `dimensions`: The feature dimensions to apply rope to
/// - `traditional`: If true, uses the traditional rope implementation
/// - `base`: The base used to compute angular frequency for each dimension
/// - `scale`: The scale to apply to the positions
/// - `offset`: An array of position offsets
/// - `freqs`: Optional precomputed frequencies
/// - `stream`: Stream to evaluate on
#[allow(clippy::too_many_arguments)]
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn rope_dynamic_device<'a>(
    #[named] array: impl AsRef<Array>,
    #[named] dimensions: i32,
    #[named] traditional: bool,
    #[optional] base: impl Into<Option<f32>>,
    #[named] scale: f32,
    #[named] offset: impl AsRef<Array>,
    #[optional] freqs: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let base = base.into();
    let base = mlx_sys::mlx_optional_float {
        value: base.unwrap_or(0.0),
        has_value: base.is_some(),
    };
    let freqs = freqs.into();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_rope_dynamic(
            res,
            array.as_ref().as_ptr(),
            dimensions,
            traditional,
            base,
            scale,
            offset.as_ref().as_ptr(),
            freqs
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            stream.as_ref().as_ptr(),
        )
    })
}

const DEFAULT_MASK_MODE: &CStr = c"";
const CAUSAL_MASK_MODE: &CStr = c"causal";

/// Mask modes for scaled dot product attention.
#[derive(Debug)]
pub enum ScaledDotProductAttentionMask<'a> {
    /// A single mask array
    Array(&'a Array),

    /// Causal masking (no explicit mask array needed)
    Causal,
}

impl<'a> From<&'a Array> for ScaledDotProductAttentionMask<'a> {
    fn from(mask: &'a Array) -> Self {
        ScaledDotProductAttentionMask::Array(mask)
    }
}

impl<'a> IntoOption<ScaledDotProductAttentionMask<'a>> for &'a Array {
    fn into_option(self) -> Option<ScaledDotProductAttentionMask<'a>> {
        Some(ScaledDotProductAttentionMask::Array(self))
    }
}

impl ScaledDotProductAttentionMask<'_> {
    fn as_mode_and_mask(&self) -> (&'static CStr, mlx_sys::mlx_array) {
        match self {
            ScaledDotProductAttentionMask::Array(mask) => (DEFAULT_MASK_MODE, mask.as_ptr()),
            ScaledDotProductAttentionMask::Causal => {
                (CAUSAL_MASK_MODE, unsafe { mlx_sys::mlx_array_new() })
            }
        }
    }
}

/// A fast implementation of multi-head attention: `O = softmax(Q @ K.T, dim=-1) @ V`
///
/// Supports [Multi-Head Attention](https://arxiv.org/abs/1706.03762), [Grouped Query Attention](https://arxiv.org/abs/2305.13245), and [Multi-Query Attention](https://arxiv.org/abs/1911.02150).
///
/// This function will dispatch to an optimized Metal kernel when the query sequence length is 1. It handles other cases with regular MLX operations.
///
/// > Note: The softmax operation is performed in float32 precision regardless of input precision (float16 or float32).
///
/// > Note: For Grouped Query Attention and Multi-Query Attention, the input arrays for `key` and `value` should not be pre-tiled to match the `query` array.
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn scaled_dot_product_attention_device<'a>(
    queries: impl AsRef<Array>,
    keys: impl AsRef<Array>,
    values: impl AsRef<Array>,
    scale: f32,
    #[optional] mask: impl IntoOption<ScaledDotProductAttentionMask<'a>>,
    #[optional] sinks: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let (mask_mode, mask_arr) = mask.into_option().map_or_else(
        || (DEFAULT_MASK_MODE, unsafe { mlx_sys::mlx_array_new() }),
        |m| m.as_mode_and_mask(),
    );

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_scaled_dot_product_attention(
            res,
            queries.as_ref().as_ptr(),
            keys.as_ref().as_ptr(),
            values.as_ref().as_ptr(),
            scale,
            mask_mode.as_ptr(),
            mask_arr,
            sinks
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            stream.as_ref().as_ptr(),
        )
    })
}

// Head-dim sets supported by MLX's fused SDPA Metal kernels. Source:
// mlx/backend/metal/scaled_dot_product_attention.cpp lines 618-624 in
// upstream mlx (commented "sdpa_full_supported_head_dim" and
// "sdpa_vector_supported_head_dim"). Anything outside these sets falls
// back to a general-purpose attention path that materializes the
// (Q, K) score matrix — significantly slower at long sequence
// lengths. See [`scaled_dot_product_attention_pad_to_fused`] which
// pads/slices to bring the call into one of these sets.
//
// Vector kernel fires when query sequence length == 1 (decode step).
const SDPA_VECTOR_HEAD_DIMS: [i32; 4] = [64, 96, 128, 256];
// Full kernel fires for Q > 1 (prefill / encoder).
const SDPA_FULL_HEAD_DIMS: [i32; 3] = [64, 80, 128];

/// Smallest fused head_dim ≥ `d` for the given query length, or `None`
/// if no fused kernel can fit `d` (i.e. d > 256 for decode, d > 128
/// for prefill).
fn next_fused_head_dim(d: i32, q_seq_len: i32) -> Option<i32> {
    let table: &[i32] = if q_seq_len == 1 {
        &SDPA_VECTOR_HEAD_DIMS
    } else {
        &SDPA_FULL_HEAD_DIMS
    };
    table.iter().copied().find(|&t| t >= d)
}

/// Like [`scaled_dot_product_attention`] but transparently pads the
/// `head_dim` of `queries`/`keys`/`values` up to the smallest
/// supported fused-kernel size when the input `head_dim` falls
/// outside MLX's fused SDPA tables.
///
/// This mirrors the `ensure_fused_sdpa` pattern used by MLX's Python
/// VLM examples (see `mlx_vlm/models/base.py::ensure_fused_sdpa`):
/// pad q/k/v with zeros on the last axis to the next supported
/// head_dim, run the fused kernel, slice the output's last axis back
/// to the original size. The padded tail contributes zero to the
/// attention scores (q·k^T = 0 in the padded slots) and zero to the
/// post-softmax weighted-V sum, so the result is mathematically
/// equivalent to running attention at the original head_dim.
///
/// # When this matters
///
/// MLX dispatches fused Metal kernels for two head-dim sets:
/// - **Prefill (Q > 1):** {64, 80, 128}
/// - **Decode (Q = 1):** {64, 96, 128, 256}
///
/// Outside those sets, MLX falls back to materializing the full
/// score matrix and running general softmax/matmul — measurably
/// slower at long sequence lengths. Common cases hit by the
/// fallback:
///
/// | Model | head_dim | Pad to | Path |
/// |-------|---------:|-------:|------|
/// | Qwen3-VL vision tower | 72 | 80 | prefill |
/// | LLaMA-style 90-dim | 90 | 128 | prefill |
///
/// # Cost of the helper
///
/// One `mlx::ops::pad` per input (zero-fill, last axis only) and one
/// last-axis slice on the output. The pad and slice run on the GPU
/// stream and overlap with the SDPA kernel; on long-sequence prefill
/// the saved fused-kernel work dominates the pad/slice overhead.
///
/// **Pad-ratio caveat:** the helper grows the head_dim by `pad_to /
/// head_dim`. For small ratios (e.g. 72→80 = 1.11×) the fused kernel
/// is reliably faster than the fallback. For larger ratios (e.g.
/// 90→128 = 1.42×) the extra kernel work can outweigh the gain.
/// Real measurements on Apple M4 Max:
///
/// | Shape (B, H, L, D) | Plain SDPA | Padded SDPA | Δ |
/// |--------------------|-----------:|------------:|--:|
/// | (1, 16, 1024, 72)  | 1.31 ms    | 0.87 ms     | **1.51× faster** |
/// | (1, 16, 4096, 72)  | 13.9 ms    | 9.87 ms     | **1.41× faster** |
/// | (1, 32, 2048, 90)  | 7.62 ms    | 7.98 ms     | 0.96× slower (90→128 pad too wide) |
///
/// If your `head_dim` lands in the gap between supported sizes by
/// more than ~25 %, bench your specific shape before relying on this
/// helper.
///
/// Pass-through behavior: if the input head_dim is already in the
/// fused set OR if no supported size fits (head_dim > 128 in prefill,
/// head_dim > 256 in decode), this calls
/// [`scaled_dot_product_attention`] without any padding — the result
/// is exactly what the un-padded call would have produced.
///
/// # Shape contract
///
/// Identical to [`scaled_dot_product_attention`]: `queries` shaped
/// `[B, H_q, L_q, D]`, `keys`/`values` shaped `[B, H_kv, L_kv, D]`.
/// `H_kv` may be smaller than `H_q` for grouped-query / multi-query
/// attention. The returned tensor is shaped `[B, H_q, L_q, D]`.
///
/// # Example
///
/// ```rust, ignore
/// // head_dim = 72 (e.g. Qwen3-VL vision tower) → padded to 80,
/// // fused prefill kernel fires, output sliced back to 72.
/// let out = scaled_dot_product_attention_pad_to_fused(
///     &queries, &keys, &values, scale, None, None,
/// )?;
/// ```
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn scaled_dot_product_attention_pad_to_fused_device<'a>(
    queries: impl AsRef<Array>,
    keys: impl AsRef<Array>,
    values: impl AsRef<Array>,
    scale: f32,
    #[optional] mask: impl IntoOption<ScaledDotProductAttentionMask<'a>>,
    #[optional] sinks: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let q = queries.as_ref();
    let k = keys.as_ref();
    let v = values.as_ref();

    let q_shape = q.shape();
    let ndim = q_shape.len();
    if ndim < 2 {
        return Err(crate::error::Exception::custom(
            "scaled_dot_product_attention_pad_to_fused: queries must be at least 2-D",
        ));
    }
    let q_seq_len = q_shape[ndim - 2];
    let head_dim = q_shape[ndim - 1];

    // Same head_dim across q/k/v? Pure assertion — MLX would error
    // anyway; we just want to keep the dispatch decision clean.
    if k.shape()[k.shape().len() - 1] != head_dim || v.shape()[v.shape().len() - 1] != head_dim {
        return Err(crate::error::Exception::custom(
            "scaled_dot_product_attention_pad_to_fused: q/k/v head_dim mismatch",
        ));
    }

    let stream_ref = stream.as_ref();
    let mask_opt = mask.into_option();
    let sinks_opt: Option<&Array> = sinks.into();

    let target = next_fused_head_dim(head_dim, q_seq_len);
    let pad_to = match target {
        Some(t) if t > head_dim => t,
        _ => {
            // Already in a fused set OR head_dim too big to fit one —
            // either way the un-padded call is the right thing.
            return scaled_dot_product_attention_device(
                q,
                k,
                v,
                scale,
                mask_opt,
                sinks_opt,
                stream_ref,
            );
        }
    };

    let pad_amount = pad_to - head_dim;
    // Pad widths: zero on every axis except the last; on the last,
    // pad `pad_amount` on the high side. PadWidth::Widths takes a
    // slice of (low, high) pairs; build it on the stack.
    let mut widths: smallvec::SmallVec<[(i32, i32); crate::constants::DEFAULT_STACK_VEC_LEN]> =
        smallvec::SmallVec::with_capacity(ndim);
    for _ in 0..(ndim - 1) {
        widths.push((0, 0));
    }
    widths.push((0, pad_amount));
    let widths_slice: &[(i32, i32)] = &widths;

    let zero_q = Array::from_int(0).as_dtype(q.dtype())?;
    let zero_kv_q = Array::from_int(0).as_dtype(k.dtype())?;
    let zero_kv_v = Array::from_int(0).as_dtype(v.dtype())?;

    let q_padded = crate::ops::pad_device(q, widths_slice, zero_q, None, stream_ref)?;
    let k_padded = crate::ops::pad_device(k, widths_slice, zero_kv_q, None, stream_ref)?;
    let v_padded = crate::ops::pad_device(v, widths_slice, zero_kv_v, None, stream_ref)?;

    let attn = scaled_dot_product_attention_device(
        &q_padded,
        &k_padded,
        &v_padded,
        scale,
        mask_opt,
        sinks_opt,
        stream_ref,
    )?;

    // Slice the last axis back to the original head_dim. attn shape:
    // [..., L_q, pad_to] → [..., L_q, head_dim].
    use crate::ops::indexing::IndexOp;
    Ok(attn.index((.., .., .., 0..head_dim)))
}

/// Root Mean Square normalization (RMS norm).
///
/// The normalization is with respect to the last axis of the input `x`.
///
/// # Params
///
/// - x: input array
/// - weight: A multiplicative weight to scale the result by. The `weight` should be one-dimensional with the same size as the last axis of `x`.
/// - eps: A small additive constant for numerical stability
/// - stream: stream or device to evaluate on
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn rms_norm_device(
    x: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    eps: f32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_rms_norm(
            res,
            x.as_ref().as_ptr(),
            weight.as_ref().as_ptr(),
            eps,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Layer normalization.
///
/// The normalization is with respect to the last axis of the input `x`.
///
/// # Params
///
/// - x: input array
/// - weight: A multiplicative weight to scale the result by. The `weight` should be one-dimensional
///   with the same size as the last axis of `x`.  If not given no scaling will occur.
/// - bias: An additive offset to be added to the result. The `bias` should be one-dimensional
///   with the same size as the last axis of `x`.  It not given no offset will occur.
/// - eps: A small additive constant for numerical stability
/// - stream: stream or device to evaluate on
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn layer_norm_device<'a>(
    #[named] x: impl AsRef<Array>,
    #[optional] weight: impl Into<Option<&'a Array>>,
    #[optional] bias: impl Into<Option<&'a Array>>,
    #[named] eps: f32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_layer_norm(
            res,
            x.as_ref().as_ptr(),
            weight
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            bias.into()
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            eps,
            stream.as_ref().as_ptr(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ops::indexing::{ArrayIndexOp, IndexOp},
        random::normal,
    };
    use float_eq::assert_float_eq;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_rope() {
        crate::random::seed(71).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        let result = rope(a, 8, false, 10000., 1.0, 0, None).unwrap();
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item::<f32>(),
            0.456_253_77,
            abs <= 0.009_125_075
        );
        assert_float_eq!(
            result.sum(None).unwrap().item::<f32>(),
            116.800_964,
            abs <= 2.336_019_3
        );
    }

    // Test adapted from Python test_fast.py/test_rope - the Python test accepts both
    // int offset and array offset, which in C/Rust are separate functions
    #[test]
    fn test_rope_dynamic() {
        crate::random::seed(71).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        // Test with array offset - should produce similar results to int offset of 3
        let offset = crate::Array::from_int(3);
        let result = rope_dynamic(&a, 8, false, 10000., 1.0, &offset, None).unwrap();
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);

        // Compare with regular rope using int offset=3
        let result_int_offset = rope(&a, 8, false, 10000., 1.0, 3, None).unwrap();
        assert_eq!(result_int_offset.shape(), [2, 8, 16]);

        // The results should be close
        let diff = &result - &result_int_offset;
        let max_diff = diff.abs().unwrap().max(None).unwrap().item::<f32>();
        assert!(max_diff < 1e-5, "Max difference was {}", max_diff);
    }

    #[test]
    fn test_rms_norm() {
        crate::random::seed(103).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        let weight = Array::ones::<f32>(&[16]).unwrap();
        let result = rms_norm(a, weight, 1e-5).unwrap();
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item::<f32>(),
            0.872_938_75,
            abs <= 0.017_458_774
        );
        assert_float_eq!(
            result.sum(None).unwrap().item::<f32>(),
            223.472_32,
            abs <= 4.469_446
        );
    }

    #[test]
    pub fn test_layer_norm_affine() {
        crate::random::seed(635).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        let weight = Array::ones::<f32>(&[16]).unwrap();
        let bias = Array::zeros::<f32>(&[16]).unwrap();
        let result = layer_norm(a, &weight, &bias, 1e-5).unwrap();
        let result = result.index((ArrayIndexOp::Ellipsis, 0));
        assert_eq!(result.shape(), [2, 8]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item::<f32>(),
            0.290_990_38,
            abs <= 0.005_819_807_8
        );
        assert_float_eq!(
            result.sum(None).unwrap().item::<f32>(),
            4.655_846,
            abs <= 0.093_116_924
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_fast_sdpa() {
        // This test just makes sure that `scaled_dot_product_attention` is callable
        // in the various cases, based on the Python test `test_fast_sdpa`.

        let Dk = 64;
        let scale = 1.0 / (Dk as f32).sqrt();
        for seq_len in [63, 129, 400] {
            for dtype in [crate::Dtype::Float32, crate::Dtype::Float16] {
                let B = 2;
                let H = 24;
                let q = normal::<f32>(&[B, H, seq_len, Dk], None, None, None)
                    .unwrap()
                    .as_dtype(dtype)
                    .unwrap();
                let k = normal::<f32>(&[B, H, seq_len, Dk], None, None, None)
                    .unwrap()
                    .as_dtype(dtype)
                    .unwrap();
                let v = normal::<f32>(&[B, H, seq_len, Dk], None, None, None)
                    .unwrap()
                    .as_dtype(dtype)
                    .unwrap();

                let result = scaled_dot_product_attention(q, k, v, scale, None, None).unwrap();
                assert_eq!(result.shape(), [B, H, seq_len, Dk]);
                assert_eq!(result.dtype(), dtype);
            }
        }
    }

    #[test]
    fn test_next_fused_head_dim_decode() {
        // Decode (Q=1) supports {64, 96, 128, 256}
        assert_eq!(next_fused_head_dim(64, 1), Some(64));
        assert_eq!(next_fused_head_dim(72, 1), Some(96));
        assert_eq!(next_fused_head_dim(80, 1), Some(96));
        assert_eq!(next_fused_head_dim(96, 1), Some(96));
        assert_eq!(next_fused_head_dim(100, 1), Some(128));
        assert_eq!(next_fused_head_dim(128, 1), Some(128));
        assert_eq!(next_fused_head_dim(192, 1), Some(256));
        assert_eq!(next_fused_head_dim(256, 1), Some(256));
        // Out of range — caller falls through to unpadded path.
        assert_eq!(next_fused_head_dim(300, 1), None);
    }

    #[test]
    fn test_next_fused_head_dim_prefill() {
        // Prefill (Q>1) supports {64, 80, 128}
        assert_eq!(next_fused_head_dim(64, 1024), Some(64));
        assert_eq!(next_fused_head_dim(72, 1024), Some(80));
        assert_eq!(next_fused_head_dim(80, 1024), Some(80));
        assert_eq!(next_fused_head_dim(90, 1024), Some(128));
        assert_eq!(next_fused_head_dim(128, 1024), Some(128));
        // Out of range.
        assert_eq!(next_fused_head_dim(192, 1024), None);
        assert_eq!(next_fused_head_dim(256, 1024), None);
    }

    /// Mathematical equivalence: padding zeros on the head_dim axis is
    /// a no-op for the attention output. q·k^T contributions from the
    /// padded slots are zero (zero × anything = zero), softmax is
    /// shift-invariant, and the post-softmax weighted-V sum reads
    /// zero from the padded V slots. Slicing the output back to the
    /// original head_dim must therefore produce a result numerically
    /// identical (up to fp16/fp32 noise) to the unpadded call.
    #[test]
    fn test_pad_to_fused_matches_unpadded_at_head_dim_72() {
        crate::random::seed(7272).unwrap();
        let b = 2;
        let n_q = 16;
        let t_q = 128;
        let t_kv = 128;
        let d = 72; // Qwen3-VL vision tower head_dim — falls back without padding.
        let q = normal::<f32>(&[b, n_q, t_q, d], None, None, None).unwrap();
        let k = normal::<f32>(&[b, n_q, t_kv, d], None, None, None).unwrap();
        let v = normal::<f32>(&[b, n_q, t_kv, d], None, None, None).unwrap();
        let scale = (d as f32).powf(-0.5);

        let unpadded = scaled_dot_product_attention(&q, &k, &v, scale, None, None).unwrap();
        let padded = scaled_dot_product_attention_pad_to_fused(&q, &k, &v, scale, None, None)
            .unwrap();

        assert_eq!(padded.shape(), unpadded.shape());
        assert_eq!(padded.dtype(), unpadded.dtype());
        // Cosine similarity check — the two paths use different
        // kernels so we expect bit-equivalent up to rounding noise.
        let diff = (&padded - &unpadded).abs().unwrap();
        let max_abs = diff.max(None).unwrap().item::<f32>();
        assert!(
            max_abs < 1e-4,
            "padded SDPA diverged from unpadded by {max_abs} > 1e-4",
        );
    }

    #[test]
    fn test_pad_to_fused_passthrough_when_already_fused() {
        // head_dim=64 is already in both vector and full sets — the
        // helper must not allocate a pad/slice round-trip.
        crate::random::seed(64).unwrap();
        let q = normal::<f32>(&[1, 8, 128, 64], None, None, None).unwrap();
        let k = normal::<f32>(&[1, 8, 128, 64], None, None, None).unwrap();
        let v = normal::<f32>(&[1, 8, 128, 64], None, None, None).unwrap();
        let scale = 0.125;

        let unpadded = scaled_dot_product_attention(&q, &k, &v, scale, None, None).unwrap();
        let padded = scaled_dot_product_attention_pad_to_fused(&q, &k, &v, scale, None, None)
            .unwrap();

        let diff = (&padded - &unpadded).abs().unwrap();
        let max_abs = diff.max(None).unwrap().item::<f32>();
        // Bit-identical — same kernel, same inputs.
        assert!(max_abs == 0.0, "passthrough diverged: max_abs={max_abs}");
    }

    // Test adapted from Python test `test_fast_sdpa.py/test_sdpa_attention_sinks`
    #[test]
    fn test_fast_sdpa_with_sinks() {
        let b = 2;
        let n_q = 8;
        let t_q = 128;
        let t_kv = 128;
        let d = 64;

        let q = normal::<f32>(&[b, n_q, t_q, d], None, None, None).unwrap();
        let k = normal::<f32>(&[b, n_q, t_kv, d], None, None, None).unwrap();
        let v = normal::<f32>(&[b, n_q, t_kv, d], None, None, None).unwrap();
        let scale = (d as f32).powf(-0.5);

        // Test with sinks parameter
        let sinks = normal::<f32>(&[n_q], None, None, None).unwrap() * 10.0;

        let result = scaled_dot_product_attention(&q, &k, &v, scale, None, &sinks).unwrap();
        assert_eq!(result.shape(), &[b, n_q, t_q, d]);
    }
}
