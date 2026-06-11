//! Qwen3.5 multimodal rotary embedding (`mrope_interleaved`) with partial
//! rotation. Port of [`Qwen3_5RotaryEmbedding`] from `mlx_vlm`.
//!
//! For each token position the 3D position id `(t, h, w)` produces a fused
//! per-feature frequency where the three axes are interleaved across the first
//! `head_dim * partial_rotary_factor` features:
//!
//! - feature `i` with `i % 3 == 0` uses the **t** axis
//! - feature `i` with `i % 3 == 1`, while `i < mrope_section[1] * 3`, uses **h**
//! - feature `i` with `i % 3 == 2`, while `i < mrope_section[2] * 3`, uses **w**
//! - any feature outside those windows falls back to the **t** axis
//!
//! `cos`/`sin` returned here have shape `(B, S, rotary_dim)` where
//! `rotary_dim = 2 * (sum of mrope_section)`. The remaining `head_dim -
//! rotary_dim` features are passed through unchanged by
//! [`apply_multimodal_rotary_pos_emb`].

use mlx_rs::{
    ops::{
        arange, broadcast_to, concatenate_axis, cos as cos_op, expand_dims, matmul, r#where,
        reshape, sin as sin_op, split, split_sections, swap_axes,
    },
    Array, Dtype,
};

use crate::error::Error;

/// Stateless multimodal RoPE generator.
///
/// Construct once per attention block with the rotary dim, base, and
/// `mrope_section`; call [`Self::cos_sin`] each forward pass to materialise
/// `cos`/`sin` tables for the current `position_ids`.
#[derive(Debug, Clone)]
pub struct MultimodalRope {
    /// Number of features that get rotated (the remainder pass through).
    /// Equals `2 * sum(mrope_section)` and `head_dim * partial_rotary_factor`.
    rotary_dim: i32,
    /// Cached inverse frequencies, shape `[rotary_dim/2]`, dtype float32.
    inv_freq: Array,
    /// Pre-built `axis_index == 1` mask. Bool `[rotary_dim/2]`.
    /// Cached so `select_per_axis` doesn't allocate a fresh
    /// `Array::from_int(1)` + comparison per layer per token.
    mask_h: Array,
    /// Pre-built `axis_index == 2` mask. Bool `[rotary_dim/2]`.
    mask_w: Array,
}

impl MultimodalRope {
    /// Build a new multimodal RoPE.
    ///
    /// - `rotary_dim`: number of rotated features (must be even).
    /// - `base`: rotary base (`rope_theta`).
    /// - `mrope_section`: the three axis lengths from `config.json`. Their sum
    ///   must equal `rotary_dim / 2`.
    pub fn new(rotary_dim: i32, base: f32, mrope_section: &[i32]) -> Result<Self, Error> {
        if rotary_dim <= 0 || rotary_dim % 2 != 0 {
            return Err(Error::config(
                "MultimodalRope: rotary_dim must be positive and even",
            ));
        }
        if mrope_section.len() != 3 {
            return Err(Error::config(
                "MultimodalRope: mrope_section must have exactly 3 entries",
            ));
        }
        let half = rotary_dim / 2;
        let section_sum: i32 = mrope_section.iter().sum();
        if section_sum != half {
            return Err(Error::config(format!(
                "MultimodalRope: mrope_section sums to {section_sum}, expected {half} (rotary_dim/2)"
            )));
        }

        let indices = arange::<_, f32>(0.0, rotary_dim as f32, 2.0)?;
        let scale = (rotary_dim as f32).recip();
        let scaled = indices.multiply(Array::from_f32(scale))?;
        let inv_freq = Array::from_f32(base).power(&scaled)?.reciprocal()?;

        // Pre-build the bool axis-selector masks once. Avoids per-call
        // `Array::from_int(1/2)` + comparison inside the layer loop.
        let axis_index = build_axis_index(mrope_section);
        let mask_h = axis_index.eq(Array::from_int(1))?;
        let mask_w = axis_index.eq(Array::from_int(2))?;

        Ok(Self {
            rotary_dim,
            inv_freq,
            mask_h,
            mask_w,
        })
    }

    /// Number of rotated features.
    pub fn rotary_dim(&self) -> i32 {
        self.rotary_dim
    }

    /// Compute `(cos, sin)` tables for the given `position_ids`.
    ///
    /// - `position_ids` may be 2-D `[B, S]` (broadcast to all three axes — the
    ///   text-only path) or 3-D `[3, B, S]` (the multimodal path).
    ///
    /// Both returned arrays have shape `[B, S, rotary_dim]` and dtype f32.
    pub fn cos_sin(&self, position_ids: &Array) -> Result<(Array, Array), Error> {
        let pos = match position_ids.ndim() {
            2 => {
                let expanded = expand_dims(position_ids, 0)?;
                let shape = position_ids.shape();
                broadcast_to(&expanded, &[3, shape[0], shape[1]])?
            }
            3 => {
                let shape = position_ids.shape();
                if shape[0] != 3 {
                    return Err(Error::shape(
                        "MultimodalRope: 3-D position_ids must have shape [3, B, S]",
                    ));
                }
                position_ids.clone()
            }
            n => {
                return Err(Error::shape(format!(
                    "MultimodalRope: position_ids ndim {n} not in {{2, 3}}",
                )))
            }
        };
        let shape = pos.shape();
        let batch = shape[1];
        let half = self.rotary_dim / 2;

        // inv_freq broadcast: [3, B, half, 1]
        let inv_freq = reshape(&self.inv_freq, &[1, 1, half, 1])?;
        let inv_freq = broadcast_to(&inv_freq, &[3, batch, half, 1])?;

        // position_ids broadcast: [3, B, 1, S], float32
        let pos = pos.as_dtype(Dtype::Float32)?;
        let pos = expand_dims(&pos, 2)?;

        // matmul: [3, B, half, 1] @ [3, B, 1, S] -> [3, B, half, S]
        let freqs = matmul(&inv_freq, &pos)?;
        // -> [3, B, S, half]
        let freqs = swap_axes(&freqs, 2, 3)?;

        // Interleave the three axes per feature using the cached axis_index.
        let freqs_t = self.select_per_axis(&freqs)?;

        // [B, S, half] -> [B, S, rotary_dim]
        let emb = concatenate_axis(&[&freqs_t, &freqs_t], -1)?;
        let cos = cos_op(&emb)?;
        let sin = sin_op(&emb)?;
        Ok((cos, sin))
    }

    /// For each feature `i ∈ [0, half)`, pick `freqs[axis_index[i]]`.
    fn select_per_axis(&self, freqs: &Array) -> Result<Array, Error> {
        // freqs: [3, B, S, half]
        let parts = split(freqs, 3, 0)?;
        let t = parts[0].squeeze_axes(&[0])?;
        let h = parts[1].squeeze_axes(&[0])?;
        let w = parts[2].squeeze_axes(&[0])?;

        // axis_index masks are pre-built in `Self::new`; broadcast
        // `[half]` against `[B, S, half]` inside the `where` op.
        let pick_w = r#where(&self.mask_w, &w, &t)?;
        Ok(r#where(&self.mask_h, &h, &pick_w)?)
    }
}

/// Build the int32 axis selector for `apply_interleaved_mrope`.
///
/// Returns a `[rotary_dim/2]` array where each entry is one of `{0, 1, 2}`
/// indicating which of the three axes the matching feature reads from.
fn build_axis_index(mrope_section: &[i32]) -> Array {
    let h_len = mrope_section[1];
    let w_len = mrope_section[2];
    let half: i32 = mrope_section.iter().sum();
    let mut idx = vec![0i32; half as usize];
    for k in 0..h_len {
        let pos = (1 + 3 * k) as usize;
        if pos < idx.len() {
            idx[pos] = 1;
        }
    }
    for k in 0..w_len {
        let pos = (2 + 3 * k) as usize;
        if pos < idx.len() {
            idx[pos] = 2;
        }
    }
    Array::from_slice(&idx, &[half])
}

/// Apply `cos`/`sin` to `q`/`k` matching `apply_multimodal_rotary_pos_emb`.
///
/// Both `q` and `k` have shape `[B, n_heads, S, head_dim]`. The first
/// `rotary_dim` features of each head are rotated; the remainder are returned
/// as-is.
///
/// `cos`/`sin` MUST already be shaped `[B, 1, S, rotary_dim]` and cast to
/// `q`'s dtype — both are caller responsibilities. The decoder builds them
/// once per forward (see `Qwen35Decoder::cos_sin_for_forward` /
/// `MtpHead::cos_sin_for_run`) and threads the prepared pair into every
/// layer's `Attention::forward`, so per-layer expand_dims + cast launches
/// don't recur in the forward.
///
/// Returns `(q_embed, k_embed)` with `q`'s dtype preserved.
pub fn apply_multimodal_rotary_pos_emb(
    q: &Array,
    k: &Array,
    cos: &Array,
    sin: &Array,
) -> Result<(Array, Array), Error> {
    let rotary_dim = cos.shape()[3];
    let q_embed = apply_one(q, cos, sin, rotary_dim)?;
    let k_embed = apply_one(k, cos, sin, rotary_dim)?;
    Ok((q_embed, k_embed))
}

fn apply_one(x: &Array, cos: &Array, sin: &Array, rotary_dim: i32) -> Result<Array, Error> {
    let last = *x
        .shape()
        .last()
        .ok_or_else(|| Error::shape("apply_rope: x has no axes"))?;
    if rotary_dim == last {
        return rotate(x, cos, sin);
    }
    let parts = split_sections(x, &[rotary_dim], -1)?;
    let x_rot = &parts[0];
    let x_pass = &parts[1];
    let rotated = rotate(x_rot, cos, sin)?;
    Ok(concatenate_axis(&[&rotated, x_pass], -1)?)
}

/// `x * cos + rotate_half(x) * sin`. cos/sin must already be cast to
/// `x`'s dtype by the caller — see [`apply_multimodal_rotary_pos_emb`].
fn rotate(x: &Array, cos: &Array, sin: &Array) -> Result<Array, Error> {
    let lhs = x.multiply(cos)?;
    let rh = rotate_half(x)?;
    let rhs = rh.multiply(sin)?;
    Ok(lhs.add(&rhs)?)
}

fn rotate_half(x: &Array) -> Result<Array, Error> {
    let halves = split(x, 2, -1)?;
    let neg_x2 = halves[1].negative()?;
    Ok(concatenate_axis(&[&neg_x2, &halves[0]], -1)?)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    #![allow(clippy::print_stdout, reason = "test code")]
    #![allow(clippy::print_stderr, reason = "test code")]
    use super::*;
    use mlx_rs::transforms::eval;

    /// Force `arr` into row-major contiguous storage and return its flat values.
    /// `as_slice` reads the underlying buffer directly and ignores non-trivial
    /// strides, so any view-producing op (split / transpose / slice) must be
    /// materialised first.
    fn flatten_f32(arr: &Array) -> Vec<f32> {
        let f = arr.as_dtype(Dtype::Float32).unwrap();
        let total: i32 = f.shape().iter().product();
        let flat = reshape(&f, &[total]).unwrap();
        // Add 0 to force a fresh contiguous evaluation.
        let zero = Array::from_f32(0.0);
        let evald = flat.add(&zero).unwrap();
        eval([&evald]).unwrap();
        evald.as_slice::<f32>().to_vec()
    }

    fn approx_eq(got: &Array, want: &[f32], tol: f32) {
        let flat = flatten_f32(got);
        assert_eq!(
            flat.len(),
            want.len(),
            "length mismatch: {flat:?} vs {want:?}"
        );
        for (i, (g, w)) in flat.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() < tol,
                "mismatch at {i}: got {g} want {w} (tol {tol})\n full got: {flat:?}\n full want: {want:?}"
            );
        }
    }

    #[test]
    fn axis_index_for_chandra() {
        let idx = build_axis_index(&[11, 11, 10]);
        assert_eq!(idx.shape(), &[32]);
        let flat: Vec<i32> = idx.as_slice::<i32>().to_vec();
        // 11 t (i%3==0), 11 h (i%3==1, i<33), 10 w (i%3==2, i<30)
        for i in 0..32 {
            let expected = match (i % 3, i) {
                (1, i) if i < 33 => 1,
                (2, i) if i < 30 => 2,
                _ => 0,
            };
            assert_eq!(flat[i as usize], expected, "feature {i}");
        }
        let t_count = flat.iter().filter(|&&v| v == 0).count();
        let h_count = flat.iter().filter(|&&v| v == 1).count();
        let w_count = flat.iter().filter(|&&v| v == 2).count();
        assert_eq!((t_count, h_count, w_count), (11, 11, 10));
    }

    #[test]
    fn cos_sin_text_only_matches_standard_rope() {
        // With identical position ids on all three axes the multimodal mrope must
        // collapse to vanilla RoPE: cos(p * inv_freq), sin(p * inv_freq).
        let rotary_dim = 8_i32;
        let base = 10_000.0_f32;
        let section = [2_i32, 1, 1];
        let rope = MultimodalRope::new(rotary_dim, base, &section).unwrap();

        let position_ids = Array::from_slice(&[0_i32, 1, 2, 3], &[1, 4]);
        let (cos, sin) = rope.cos_sin(&position_ids).unwrap();
        assert_eq!(cos.shape(), &[1, 4, 8]);
        assert_eq!(sin.shape(), &[1, 4, 8]);

        // Reference: per-feature angle = pos * base^(-2k/rotary_dim) for k in 0..half,
        // tiled twice to length rotary_dim.
        let half = (rotary_dim / 2) as usize;
        let mut want_cos = Vec::new();
        let mut want_sin = Vec::new();
        for p in 0..4 {
            let mut row: Vec<f32> = (0..half)
                .map(|k| {
                    let inv = base.powf(-2.0 * k as f32 / rotary_dim as f32);
                    (p as f32) * inv
                })
                .collect();
            row.extend_from_slice(&row.clone());
            for theta in &row {
                want_cos.push(theta.cos());
                want_sin.push(theta.sin());
            }
        }
        approx_eq(&cos, &want_cos, 1e-5);
        approx_eq(&sin, &want_sin, 1e-5);
    }

    #[test]
    fn cos_sin_multimodal_uses_per_axis_positions() {
        // Distinct positions per axis: t=0..S, h=10+0..S, w=100+0..S.
        // Each feature must pick the matching axis according to the chandra mrope_section.
        let rotary_dim = 8_i32;
        let base = 10_000.0_f32;
        let section = [2_i32, 1, 1]; // half=4
        let rope = MultimodalRope::new(rotary_dim, base, &section).unwrap();

        let s = 3_usize;
        let mut data = Vec::with_capacity(3 * s);
        for axis in 0..3 {
            for p in 0..s {
                data.push((axis * 100 + p) as i32);
            }
        }
        let pos = Array::from_slice(&data, &[3, 1, s as i32]);
        let (cos, _sin) = rope.cos_sin(&pos).unwrap();
        let half = (rotary_dim / 2) as usize;
        // For section=[2,1,1] -> axis_index = [0, 1, 2, 0]
        let axis_per_feature = [0_i32, 1, 2, 0];

        // Reference: for each position step p, build per-feature angle = position[axis_index[k]] * inv_freq[k]
        let mut want = Vec::new();
        for p in 0..s {
            let mut row = Vec::new();
            for (k, &axis) in axis_per_feature.iter().enumerate() {
                let inv = base.powf(-2.0 * k as f32 / rotary_dim as f32);
                let position = data[axis as usize * s + p] as f32;
                row.push(position * inv);
            }
            row.extend_from_slice(&row.clone());
            for theta in &row {
                want.push(theta.cos());
            }
        }
        approx_eq(&cos, &want, 1e-5);
        assert_eq!(cos.shape(), &[1, s as i32, half as i32 * 2]);
    }

    #[test]
    fn apply_rope_preserves_pass_through_features() {
        // head_dim=10, rotary_dim=8 -> last 2 features must be untouched.
        let rotary_dim = 8_i32;
        let head_dim = 10_i32;
        let rope = MultimodalRope::new(rotary_dim, 10_000.0, &[2, 1, 1]).unwrap();
        let pos = Array::from_slice(&[0_i32, 1], &[1, 2]);
        let (cos, sin) = rope.cos_sin(&pos).unwrap();
        // `apply_multimodal_rotary_pos_emb` now expects cos/sin pre-shaped
        // to `[B, 1, S, rotary_dim]` (production does this once in the
        // decoder). q is already f32 so no `.as_dtype` cast.
        let cos = expand_dims(&cos, 1).unwrap();
        let sin = expand_dims(&sin, 1).unwrap();

        // q/k shape [B=1, H=1, S=2, head_dim=10]
        let q_data: Vec<f32> = (0..(2 * head_dim)).map(|v| v as f32 * 0.1).collect();
        let q = Array::from_slice(&q_data, &[1, 1, 2, head_dim]);
        let k = q.clone();
        let (q_out, k_out) = apply_multimodal_rotary_pos_emb(&q, &k, &cos, &sin).unwrap();
        assert_eq!(q_out.shape(), q.shape());
        assert_eq!(k_out.shape(), k.shape());

        // The tail is the last `head_dim - rotary_dim` features. For every
        // [B, H, S, *] those entries must be byte-identical to the input.
        let q_flat = flatten_f32(&q);
        let q_out_flat = flatten_f32(&q_out);
        for b in 0..1 {
            for h in 0..1 {
                for s in 0..2 {
                    for f in rotary_dim..head_dim {
                        let idx = ((b + h) * 2 + s) * head_dim + f;
                        let idx = idx as usize;
                        assert_eq!(
                            q_flat[idx], q_out_flat[idx],
                            "tail feature {f} at pos {s} changed: {} -> {}",
                            q_flat[idx], q_out_flat[idx]
                        );
                    }
                }
            }
        }

        // At position 0 the rotation is the identity (cos=1, sin=0), so the
        // entire row must be preserved.
        for f in 0..head_dim {
            let idx = f as usize;
            assert!(
                (q_flat[idx] - q_out_flat[idx]).abs() < 1e-5,
                "position 0 feature {f} changed: {} -> {}",
                q_flat[idx],
                q_out_flat[idx]
            );
        }
    }
}
