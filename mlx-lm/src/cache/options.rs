//! `CacheOptions` — KV-cache backing + per-cache toggles.

/// Default affine-quantisation group size (elements quantised together).
/// Divides every supported `head_dim` (64/128/256).
pub const DEFAULT_KV_GROUP_SIZE: i32 = 64;

/// Minimum bit-width for the K cache. K feeds softmax, which amplifies
/// quantisation error; below 8-bit greedy decode degrades badly, so the
/// cache clamps any lower request up to this floor.
pub const MIN_K_BITS: i32 = 8;

/// Backing kind for full-attention layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CacheKind {
    #[default]
    Dense,
    /// Affine-quantised K/V with independent bit-widths. K is the
    /// softmax-sensitive tensor (kept high, ≥ [`MIN_K_BITS`]); V tolerates
    /// aggressive quantisation (weighted-sum averaging smooths error). The
    /// cache clamps `k_bits` up to [`MIN_K_BITS`] and `v_bits` down to
    /// `k_bits` if a backwards-asymmetric config is requested.
    Quantized {
        group_size: i32,
        k_bits: i32,
        v_bits: i32,
    },
}

impl CacheKind {
    /// Symmetric 8-bit K/V — near-lossless, ~2× KV memory reduction.
    pub fn quantized_q8() -> Self {
        Self::Quantized {
            group_size: DEFAULT_KV_GROUP_SIZE,
            k_bits: 8,
            v_bits: 8,
        }
    }

    /// Alias for [`Self::quantized_q8`] (symmetric 8-bit).
    pub fn quantized_k8_v8() -> Self {
        Self::quantized_q8()
    }

    /// Recommended mix: 8-bit K (protects attention) + 4-bit V (saves
    /// memory). Greedy decode stays effectively identical to fp16.
    pub fn quantized_k8_v4() -> Self {
        Self::Quantized {
            group_size: DEFAULT_KV_GROUP_SIZE,
            k_bits: 8,
            v_bits: 4,
        }
    }

    /// Symmetric 4-bit request. **K clamps up to 8-bit** at cache build
    /// (4-bit K breaks decode), so this is effectively k8/v4.
    pub fn quantized_q4() -> Self {
        Self::Quantized {
            group_size: DEFAULT_KV_GROUP_SIZE,
            k_bits: 4,
            v_bits: 4,
        }
    }
}

/// Default prefill chunk cap when neither user nor cache imposes one.
pub const DEFAULT_PREFILL_CHUNK: i32 = 2048;

#[derive(Debug, Clone, Copy)]
pub struct CacheOptions {
    pub kind: CacheKind,
    /// Max tokens per prefill forward. `None` = single-pass.
    pub max_prefill_chunk: Option<i32>,
}

impl Default for CacheOptions {
    fn default() -> Self {
        Self {
            kind: CacheKind::Dense,
            max_prefill_chunk: Some(DEFAULT_PREFILL_CHUNK),
        }
    }
}
