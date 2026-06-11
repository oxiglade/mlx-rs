//! [`KeyValueCache`] trait + blanket `&mut T` forwarding impl.

use mlx_rs::{
    error::Exception,
    fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask},
    Array,
};

/// `None` + `n_q > 1` is causal by definition; `None` + decode needs no mask.
#[inline]
pub(crate) fn resolve_sdpa_mask(
    mask: Option<&Array>,
    n_q: i32,
) -> Option<ScaledDotProductAttentionMask<'_>> {
    match mask {
        Some(m) => Some(ScaledDotProductAttentionMask::Array(m)),
        None if n_q > 1 => Some(ScaledDotProductAttentionMask::Causal),
        None => None,
    }
}

/// Catches a `[L, L]` causal mask built without `cache.offset()`: turn 1
/// passes silently, turn 2 fails inside SDPA with a cryptic broadcast
/// error.
#[inline]
pub(crate) fn assert_mask_matches_keys(mask: Option<&Array>, k_full: &Array) {
    if !cfg!(debug_assertions) {
        return;
    }
    let Some(mask) = mask else { return };
    let m_shape = mask.shape();
    let k_shape = k_full.shape();
    let m_last = m_shape.last().copied().unwrap_or(0);
    let k_last = k_shape[k_shape.len() - 2];
    debug_assert!(
        m_last == k_last,
        "mask key axis ({m_last}) does not match K seq len ({k_last}); \
         mask {m_shape:?}, k_full {k_shape:?}",
    );
}

/// Key-value cache for decoder-only attention.
pub trait KeyValueCache {
    fn is_quantized(&self) -> bool {
        false
    }

    fn group_size(&self) -> Option<i32> {
        None
    }

    fn bits(&self) -> Option<i32> {
        None
    }

    fn offset(&self) -> i32;

    fn max_size(&self) -> Option<i32>;

    fn update_and_fetch(&mut self, keys: Array, values: Array)
        -> Result<(Array, Array), Exception>;

    /// Dense `(keys, values)` over the currently-cached history WITHOUT
    /// appending or advancing the offset — the same tensors a zero-length
    /// [`Self::update_and_fetch`] would return. `None` when the cache is
    /// empty. Quantised caches dequantise to dense. Used by a cross-model
    /// draft head that borrows this cache's K/V read-only.
    fn current_kv(&self) -> Result<Option<(Array, Array)>, Exception> {
        Ok(None)
    }

    /// `softmax(scaled_q @ K.T) @ V` over the full cached history.
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

impl<T> KeyValueCache for &'_ mut T
where
    T: KeyValueCache,
{
    fn is_quantized(&self) -> bool {
        T::is_quantized(self)
    }

    fn group_size(&self) -> Option<i32> {
        T::group_size(self)
    }

    fn bits(&self) -> Option<i32> {
        T::bits(self)
    }

    fn offset(&self) -> i32 {
        T::offset(self)
    }

    fn max_size(&self) -> Option<i32> {
        T::max_size(self)
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        T::update_and_fetch(self, keys, values)
    }

    fn current_kv(&self) -> Result<Option<(Array, Array)>, Exception> {
        T::current_kv(self)
    }

    fn attention(
        &mut self,
        queries: &Array,
        keys: Array,
        values: Array,
        scale: f32,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        T::attention(self, queries, keys, values, scale, mask)
    }
}
