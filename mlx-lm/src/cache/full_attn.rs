//! `FullAttnCache` — full-attention KV slot. Either a dense [`KVCache`] or
//! an affine-quantised [`QuantizedKVCache`], selected by [`CacheKind`].

use mlx_rs::{error::Exception, Array};

use super::kvcache::{KVCache, DEFAULT_KV_CACHE_INIT_CAPACITY};
use super::options::{CacheKind, CacheOptions};
use super::quantized_kvcache::QuantizedKVCache;
use super::trait_def::KeyValueCache;

#[derive(Debug, Clone)]
pub enum FullAttnCache {
    Standard(KVCache),
    Quantized(QuantizedKVCache),
}

impl FullAttnCache {
    /// Build the slot for the configured [`CacheKind`]. Quantised configs
    /// come from validated [`CacheKind`] presets, so `with_config` cannot
    /// hit its bit-width error here — the `expect` guards a caller that
    /// hand-built an out-of-range `CacheKind::Quantized`.
    pub fn from_options(opts: CacheOptions) -> Self {
        match opts.kind {
            CacheKind::Dense => Self::Standard(KVCache::new()),
            CacheKind::Quantized {
                group_size,
                k_bits,
                v_bits,
            } => Self::Quantized(
                QuantizedKVCache::with_config(
                    DEFAULT_KV_CACHE_INIT_CAPACITY,
                    group_size,
                    k_bits,
                    v_bits,
                )
                .expect("CacheKind::Quantized carries a supported bit-width"),
            ),
        }
    }
}

impl Default for FullAttnCache {
    fn default() -> Self {
        Self::Standard(KVCache::new())
    }
}

impl KeyValueCache for FullAttnCache {
    fn is_quantized(&self) -> bool {
        match self {
            Self::Standard(c) => c.is_quantized(),
            Self::Quantized(c) => c.is_quantized(),
        }
    }

    fn group_size(&self) -> Option<i32> {
        match self {
            Self::Standard(c) => c.group_size(),
            Self::Quantized(c) => Some(c.group_size()),
        }
    }

    fn bits(&self) -> Option<i32> {
        match self {
            Self::Standard(c) => c.bits(),
            Self::Quantized(c) => Some(c.k_bits()),
        }
    }

    fn offset(&self) -> i32 {
        match self {
            Self::Standard(c) => c.offset(),
            Self::Quantized(c) => c.offset(),
        }
    }

    fn max_size(&self) -> Option<i32> {
        match self {
            Self::Standard(c) => c.max_size(),
            Self::Quantized(c) => c.max_size(),
        }
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        match self {
            Self::Standard(c) => c.update_and_fetch(keys, values),
            Self::Quantized(c) => c.update_and_fetch(keys, values),
        }
    }

    fn current_kv(&self) -> Result<Option<(Array, Array)>, Exception> {
        match self {
            Self::Standard(c) => c.current_kv(),
            Self::Quantized(c) => c.current_kv(),
        }
    }

    fn attention(
        &mut self,
        queries: &Array,
        keys: Array,
        values: Array,
        scale: f32,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        match self {
            Self::Standard(c) => c.attention(queries, keys, values, scale, mask),
            Self::Quantized(c) => c.attention(queries, keys, values, scale, mask),
        }
    }
}
