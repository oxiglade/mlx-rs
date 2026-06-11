//! KV-cache implementations for decoder-only models.
//!
//! - [`trait_def`] — the [`KeyValueCache`] trait + blanket `&mut T` impl
//! - [`kvcache`] — [`KVCache`], the default pre-allocated step-grown cache
//! - [`quantized_kvcache`] — [`QuantizedKVCache`], affine-quant K/V with
//!   independent `k_bits`/`v_bits`
//! - [`rotating_kvcache`] — [`RotatingKVCache`], sliding-window ring buffer
//!   (Gemma 3/4 sliding layers)
//! - [`full_attn`] — [`FullAttnCache`], the shared full-attention slot
//! - [`options`] — [`CacheOptions`] / [`CacheKind`] + prefill-chunk helpers

pub mod full_attn;
pub mod kvcache;
pub mod options;
pub mod quantized_kvcache;
pub mod rotating_kvcache;
pub mod trait_def;

pub use full_attn::FullAttnCache;
pub use kvcache::{KVCache, DEFAULT_KV_CACHE_INIT_CAPACITY};
pub use options::{
    CacheKind, CacheOptions, DEFAULT_KV_GROUP_SIZE, DEFAULT_PREFILL_CHUNK, MIN_K_BITS,
};
pub use quantized_kvcache::QuantizedKVCache;
pub use rotating_kvcache::RotatingKVCache;
pub use trait_def::KeyValueCache;

/// Effective per-chunk prefill cap for a cache stack: the smallest
/// bounded slot's `max_size` (e.g. a sliding window), combined with the
/// user's `max_prefill_chunk`. A windowed layer can never usefully attend
/// to more than its window in one pass, so chunks above that gain nothing.
pub fn effective_prefill_chunk_opt<C: KeyValueCache>(
    cache: &[Option<C>],
    user_cap: Option<i32>,
) -> Option<i32> {
    let window = cache.iter().filter_map(|c| c.as_ref()?.max_size()).min();
    match (window, user_cap) {
        (Some(w), Some(u)) => Some(w.min(u)),
        (Some(w), None) => Some(w),
        (None, u) => u,
    }
}
