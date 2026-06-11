//! [`QuantizedKVCache`]: affine-quantised KV cache with independent
//! `k_bits` / `v_bits`.
//!
//! Stores K/V as packed `uint32` weights plus per-group `scales`/`biases`
//! (six buffers total), grown geometrically like [`super::kvcache::KVCache`].
//! K feeds softmax (error-sensitive, kept high); V feeds a weighted sum
//! (error-tolerant, compressed harder). On read:
//!
//! - `k_bits == v_bits`: packed path — score/attend straight off the packed
//!   buffers via [`quantized_scaled_dot_product_attention`] (one shared
//!   `bits`, no full dequant).
//! - `k_bits != v_bits`: dequantise K at `k_bits` and V at `v_bits`, then the
//!   dense `fast::scaled_dot_product_attention` (a single `quantized_matmul`
//!   cannot mix two bit-widths).

use log::warn;
use mlx_rs::{
    error::Exception,
    fast::scaled_dot_product_attention,
    ops::{
        arange, broadcast_to, dequantize, expand_dims,
        indexing::{Ellipsis, IndexOp, TryIndexMutOp},
        quantize, zeros_dtype,
    },
    Array, Dtype,
};

use super::kvcache::DEFAULT_KV_CACHE_INIT_CAPACITY;
use super::options::{DEFAULT_KV_GROUP_SIZE, MIN_K_BITS};
use super::trait_def::{assert_mask_matches_keys, resolve_sdpa_mask, KeyValueCache};
use crate::error::Error;
use crate::utils::{quantized_scaled_dot_product_attention, QuantizedKeys, QuantizedValues};

/// Affine-quant bit-widths mlx's `quantize` accepts.
const SUPPORTED_BITS: [i32; 5] = [2, 3, 4, 6, 8];

/// Packed `(weights, scales, biases)` triple returned by [`quantize`].
type PackedTriple = (Array, Array, Array);

/// Affine-quantised KV cache with separate K/V bit-widths.
///
/// Six packed buffers (`*_wq` packed `uint32`, `*_scales`, `*_biases`) over
/// the populated `[..offset]` range; geometric growth mirrors
/// [`super::kvcache::KVCache`]. No TurboQuant Π rotation (added later).
#[derive(Debug, Clone)]
pub struct QuantizedKVCache {
    keys_wq: Option<Array>,
    keys_scales: Option<Array>,
    keys_biases: Option<Array>,
    values_wq: Option<Array>,
    values_scales: Option<Array>,
    values_biases: Option<Array>,
    offset: i32,
    init_capacity: i32,
    group_size: i32,
    k_bits: i32,
    v_bits: i32,
    /// Original K/V dtype, captured on first append for the dequant output.
    dtype: Option<Dtype>,
}

impl Default for QuantizedKVCache {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantizedKVCache {
    /// 8-bit symmetric cache with default capacity and group size.
    pub fn new() -> Self {
        Self::with_config(DEFAULT_KV_CACHE_INIT_CAPACITY, DEFAULT_KV_GROUP_SIZE, 8, 8)
            .expect("default 8/8 config is valid")
    }

    /// Build a cache with explicit capacity / group size / bit-widths.
    ///
    /// Bit-widths outside [`SUPPORTED_BITS`] are a hard error. Otherwise the
    /// config is clamped to a safe shape (warning on stderr): `k_bits` rises
    /// to [`MIN_K_BITS`], then `v_bits` is capped at `k_bits` so the
    /// softmax-sensitive K is never coarser than V.
    pub fn with_config(
        init_capacity: i32,
        group_size: i32,
        k_bits: i32,
        v_bits: i32,
    ) -> Result<Self, Error> {
        if init_capacity <= 0 {
            return Err(Error::config(
                "QuantizedKVCache init_capacity must be positive",
            ));
        }
        if group_size <= 0 {
            return Err(Error::config(
                "QuantizedKVCache group_size must be positive",
            ));
        }
        for (name, bits) in [("k_bits", k_bits), ("v_bits", v_bits)] {
            if !SUPPORTED_BITS.contains(&bits) {
                return Err(Error::config(format!(
                    "QuantizedKVCache {name}={bits} unsupported; expected one of {SUPPORTED_BITS:?}"
                )));
            }
        }

        let k_bits = if k_bits < MIN_K_BITS {
            warn!(
                "QuantizedKVCache k_bits={k_bits} below the {MIN_K_BITS}-bit floor \
                 (softmax-sensitive); clamping to {MIN_K_BITS}"
            );
            MIN_K_BITS
        } else {
            k_bits
        };
        let v_bits = if v_bits > k_bits {
            warn!(
                "QuantizedKVCache v_bits={v_bits} exceeds k_bits={k_bits} \
                 (backwards asymmetry); clamping v_bits to {k_bits}"
            );
            k_bits
        } else {
            v_bits
        };

        Ok(Self {
            keys_wq: None,
            keys_scales: None,
            keys_biases: None,
            values_wq: None,
            values_scales: None,
            values_biases: None,
            offset: 0,
            init_capacity,
            group_size,
            k_bits,
            v_bits,
            dtype: None,
        })
    }

    pub fn group_size(&self) -> i32 {
        self.group_size
    }

    pub fn k_bits(&self) -> i32 {
        self.k_bits
    }

    pub fn v_bits(&self) -> i32 {
        self.v_bits
    }

    /// Populated capacity along the token axis (`0` before the first append).
    fn capacity(&self) -> i32 {
        self.keys_wq
            .as_ref()
            .map(|k| k.shape()[k.shape().len() - 2])
            .unwrap_or(0)
    }

    /// Zero buffer matching `template`'s shape/dtype but `capacity` tokens
    /// on the second-to-last (token) axis.
    fn alloc_like(template: &Array, capacity: i32) -> Result<Array, Exception> {
        let mut buf_shape = template.shape().to_vec();
        let t_axis = buf_shape.len() - 2;
        buf_shape[t_axis] = capacity;
        zeros_dtype(&buf_shape, template.dtype())
    }

    /// Copy the populated `[..offset]` slice of `old` into a freshly grown
    /// `target` buffer (token axis = -2).
    fn copy_into(target: &mut Array, old: &Array, offset: i32) -> Result<(), Exception> {
        if offset > 0 {
            target.try_index_mut(
                (Ellipsis, 0..offset, ..),
                old.index((Ellipsis, 0..offset, ..)),
            )?;
        }
        Ok(())
    }

    /// Ensure all six buffers hold at least `offset + s` tokens, growing
    /// geometrically and preserving existing entries. `templates` are the
    /// freshly-quantised tensors whose shapes/dtypes seed each buffer.
    fn grow_to_fit(&mut self, s: i32, k: &PackedTriple, v: &PackedTriple) -> Result<(), Exception> {
        let required = self.offset + s;
        let current = self.capacity();
        if required <= current {
            return Ok(());
        }
        let mut target = if current == 0 {
            self.init_capacity.max(required)
        } else {
            current
        };
        while target < required {
            target *= 2;
        }

        let slots: [(&mut Option<Array>, &Array); 6] = [
            (&mut self.keys_wq, &k.0),
            (&mut self.keys_scales, &k.1),
            (&mut self.keys_biases, &k.2),
            (&mut self.values_wq, &v.0),
            (&mut self.values_scales, &v.1),
            (&mut self.values_biases, &v.2),
        ];
        for (buf, template) in slots {
            let mut grown = Self::alloc_like(template, target)?;
            if let Some(old) = buf.take() {
                Self::copy_into(&mut grown, &old, self.offset)?;
            }
            *buf = Some(grown);
        }
        Ok(())
    }

    /// Quantise the incoming K/V (each at its own bit-width), append into the
    /// six buffers, and return graph-view slices over `[..offset]` of each.
    #[allow(
        clippy::type_complexity,
        reason = "six packed views mirror the buffer layout"
    )]
    fn append_quantised(
        &mut self,
        keys: &Array,
        values: &Array,
    ) -> Result<(PackedTriple, PackedTriple), Error> {
        if self.dtype.is_none() {
            self.dtype = Some(keys.dtype());
        }
        let key_shape = keys.shape();
        let s = key_shape[key_shape.len() - 2];

        let k_packed = quantize(keys, self.group_size, self.k_bits)?;
        let v_packed = quantize(values, self.group_size, self.v_bits)?;
        self.grow_to_fit(s, &k_packed, &v_packed)?;

        let lo = self.offset;
        let hi = self.offset + s;
        let writes: [(&mut Option<Array>, Array); 6] = [
            (&mut self.keys_wq, k_packed.0),
            (&mut self.keys_scales, k_packed.1),
            (&mut self.keys_biases, k_packed.2),
            (&mut self.values_wq, v_packed.0),
            (&mut self.values_scales, v_packed.1),
            (&mut self.values_biases, v_packed.2),
        ];
        for (buf, new) in writes {
            let dst = buf.as_mut().expect("allocated by grow_to_fit");
            dst.try_index_mut((Ellipsis, lo..hi, ..), new)?;
        }
        self.offset = hi;

        Ok((self.view_triple(true)?, self.view_triple(false)?))
    }

    /// `[..offset]` views of the K (`is_key`) or V packed triple.
    fn view_triple(&self, is_key: bool) -> Result<PackedTriple, Error> {
        let end = self.offset;
        let (wq, scales, biases) = if is_key {
            (&self.keys_wq, &self.keys_scales, &self.keys_biases)
        } else {
            (&self.values_wq, &self.values_scales, &self.values_biases)
        };
        let slice = |a: &Option<Array>| -> Array {
            a.as_ref()
                .expect("populated by append_quantised")
                .index((Ellipsis, 0..end, ..))
        };
        Ok((slice(wq), slice(scales), slice(biases)))
    }

    /// Bool causal mask `[1, 1, n_q, offset + n_q]` for a prefill against the
    /// current offset (call before the append). Mirrors the decoder's
    /// `build_full_attn_mask`; the packed SDPA path needs an explicit mask
    /// because it does not derive causality from `None`.
    fn causal_mask(&self, n_q: i32) -> Result<Array, Exception> {
        let total = self.offset + n_q;
        let rinds = arange::<_, i32>(0, total, None)?;
        let linds = arange::<_, i32>(self.offset, total, None)?;
        let linds_b = expand_dims(&linds, 1)?;
        let rinds_b = expand_dims(&rinds, 0)?;
        let mask = linds_b.ge(&rinds_b)?;
        let mask = expand_dims(&expand_dims(&mask, 0)?, 0)?;
        broadcast_to(&mask, &[1, 1, n_q, total])
    }
}

impl KeyValueCache for QuantizedKVCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    /// Dequantise path: append, then unpack K/V to dense at their own bits.
    /// The default `attention` impl drives this; `Self::attention` overrides
    /// it with the packed fast path when `k_bits == v_bits`.
    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        let (k, v) = self.append_quantised(&keys, &values)?;
        let k_dense = dequantize(&k.0, &k.1, &k.2, self.group_size, self.k_bits)?;
        let v_dense = dequantize(&v.0, &v.1, &v.2, self.group_size, self.v_bits)?;
        Ok((k_dense, v_dense))
    }

    /// Dequantise the currently-stored K/V to dense without appending.
    fn current_kv(&self) -> Result<Option<(Array, Array)>, Exception> {
        if self.offset == 0 || self.keys_wq.is_none() {
            return Ok(None);
        }
        let k = self.view_triple(true)?;
        let v = self.view_triple(false)?;
        let k_dense = dequantize(&k.0, &k.1, &k.2, self.group_size, self.k_bits)?;
        let v_dense = dequantize(&v.0, &v.1, &v.2, self.group_size, self.v_bits)?;
        Ok(Some((k_dense, v_dense)))
    }

    fn attention(
        &mut self,
        queries: &Array,
        keys: Array,
        values: Array,
        scale: f32,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let q_shape = queries.shape();
        let n_q = q_shape[q_shape.len() - 2];

        // Symmetric: score/attend straight off the packed buffers (one
        // shared `bits`). Unlike the dense `fast::scaled_dot_product_attention`,
        // the packed path does not synthesize a causal mask for `None`, so a
        // multi-query prefill with no explicit mask must get one built here
        // (offset read *before* the append). The decoder already passes a
        // bool causal mask on prefill; this covers a standalone caller.
        if self.k_bits == self.v_bits {
            let causal = if mask.is_none() && n_q > 1 {
                Some(self.causal_mask(n_q)?)
            } else {
                None
            };
            let effective_mask = mask.or(causal.as_ref());

            let (k, v) = self.append_quantised(&keys, &values)?;
            let q_keys = QuantizedKeys {
                keys: k.0,
                scales: k.1,
                biases: k.2,
            };
            let q_values = QuantizedValues {
                values: v.0,
                scales: v.1,
                biases: v.2,
            };
            return quantized_scaled_dot_product_attention(
                queries,
                q_keys,
                q_values,
                scale,
                effective_mask,
                self.group_size,
                self.k_bits,
            );
        }

        // Asymmetric: dequant each tensor at its own bits, then dense SDPA
        // (which synthesizes the causal mask for `None` itself).
        let (k_full, v_full) = self.update_and_fetch(keys, values)?;
        assert_mask_matches_keys(mask, &k_full);
        scaled_dot_product_attention(
            queries,
            k_full,
            v_full,
            scale,
            resolve_sdpa_mask(mask, n_q),
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    use super::*;
    use mlx_rs::ops::{abs, max, subtract};
    use mlx_rs::random::uniform;
    use mlx_rs::transforms::eval;

    use crate::cache::KVCache;

    const HEAD_DIM: i32 = 64;
    const N_KV_HEADS: i32 = 2;
    const N_Q_HEADS: i32 = 4;
    const GROUP: i32 = 64;
    const SCALE: f32 = 0.125; // 1/sqrt(64)

    fn kv(s: i32) -> (Array, Array) {
        let shape = [1, N_KV_HEADS, s, HEAD_DIM];
        let k = uniform::<_, f32>(-1.0, 1.0, &shape, None).unwrap();
        let v = uniform::<_, f32>(-1.0, 1.0, &shape, None).unwrap();
        (k, v)
    }

    fn queries(s: i32) -> Array {
        uniform::<_, f32>(-1.0, 1.0, &[1, N_Q_HEADS, s, HEAD_DIM], None).unwrap()
    }

    fn max_abs_diff(a: &Array, b: &Array) -> f32 {
        let d = abs(subtract(a, b).unwrap()).unwrap();
        let m = max(&d, None).unwrap();
        m.item::<f32>()
    }

    #[test]
    fn roundtrip_buffer_shapes() {
        let mut c = QuantizedKVCache::with_config(64, GROUP, 8, 8).unwrap();
        let s = 5;
        let (k, v) = kv(s);
        let (kt, vt) = c.append_quantised(&k, &v).unwrap();
        eval([&kt.0, &kt.1, &kt.2, &vt.0, &vt.1, &vt.2]).unwrap();
        assert_eq!(c.offset(), s);
        // scales/biases carry one entry per group along head_dim.
        assert_eq!(kt.1.shape(), &[1, N_KV_HEADS, s, HEAD_DIM / GROUP]);
        assert_eq!(vt.1.shape(), &[1, N_KV_HEADS, s, HEAD_DIM / GROUP]);
    }

    #[test]
    fn offset_extends_prefill_then_decode() {
        let mut c = QuantizedKVCache::with_config(2, GROUP, 8, 8).unwrap();
        let (k, v) = kv(6);
        c.update_and_fetch(k, v).unwrap();
        assert_eq!(c.offset(), 6);
        for _ in 0..3 {
            let (k1, v1) = kv(1);
            let (kf, _) = c.update_and_fetch(k1, v1).unwrap();
            eval([&kf]).unwrap();
        }
        assert_eq!(c.offset(), 9);
    }

    #[test]
    fn config_clamps_k_bits_up() {
        // k4 v4 -> k clamps to 8, v stays 4.
        let c = QuantizedKVCache::with_config(64, GROUP, 4, 4).unwrap();
        assert_eq!(c.k_bits(), 8);
        assert_eq!(c.v_bits(), 4);
    }

    #[test]
    fn config_clamps_v_down_to_k() {
        // v8 with k... k stays 8, v capped at 8 (already equal). Use k4,v8:
        // k clamps to 8, then v capped at 8.
        let c = QuantizedKVCache::with_config(64, GROUP, 4, 8).unwrap();
        assert_eq!(c.k_bits(), 8);
        assert_eq!(c.v_bits(), 8);
    }

    #[test]
    fn config_rejects_unsupported_bits() {
        assert!(QuantizedKVCache::with_config(64, GROUP, 5, 4).is_err());
        assert!(QuantizedKVCache::with_config(64, GROUP, 8, 5).is_err());
    }

    /// Both dispatch branches must track the dense `KVCache` + fp16 SDPA
    /// reference: k8/v8 exercises the packed path (tight tol), k8/v4 the
    /// dequant path (looser).
    fn dense_reference(q: &Array, k: &Array, v: &Array) -> Array {
        let mut dense = KVCache::new();
        dense
            .attention(q, k.clone(), v.clone(), SCALE, None)
            .unwrap()
    }

    #[test]
    fn packed_branch_matches_dense() {
        let q = queries(4);
        let (k, v) = kv(4);
        let want = dense_reference(&q, &k, &v);
        let mut c = QuantizedKVCache::with_config(64, GROUP, 8, 8).unwrap();
        let got = c.attention(&q, k, v, SCALE, None).unwrap();
        eval([&got, &want]).unwrap();
        let diff = max_abs_diff(&got, &want);
        assert!(diff < 0.05, "packed k8v8 diverged from dense: {diff}");
    }

    #[test]
    fn dequant_branch_matches_dense() {
        let q = queries(4);
        let (k, v) = kv(4);
        let want = dense_reference(&q, &k, &v);
        let mut c = QuantizedKVCache::with_config(64, GROUP, 8, 4).unwrap();
        let got = c.attention(&q, k, v, SCALE, None).unwrap();
        eval([&got, &want]).unwrap();
        assert!(max_abs_diff(&got, &want) < 0.2);
    }
}
