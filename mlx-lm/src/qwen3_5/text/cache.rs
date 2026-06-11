//! Qwen3.5 per-layer cache slot. Full-attn → shared `FullAttnCache`;
//! linear-attn → qwen-specific GDN recurrent state.

use mlx_rs::Array;

use crate::cache::{CacheOptions, FullAttnCache, KeyValueCache};
use crate::qwen3_5::text::config::ModelConfig;

/// Gated DeltaNet recurrent state. Not a KV cache.
/// `None` fields are treated as zero-initialised on first pass.
#[derive(Debug, Clone, Default)]
pub struct LinearAttnCache {
    /// `[B, conv_kernel_size - 1, conv_dim]`.
    pub(crate) conv_state: Option<Array>,
    /// `[B, Hv, Dv, Dk]`.
    pub(crate) recurrent_state: Option<Array>,
    pub(crate) offset: i32,
}

impl LinearAttnCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[derive(Debug, Clone)]
pub enum LayerCache {
    FullAttention(FullAttnCache),
    LinearAttention(LinearAttnCache),
}

impl LayerCache {
    pub fn for_layer(is_linear: bool, opts: CacheOptions) -> Self {
        if is_linear {
            Self::LinearAttention(LinearAttnCache::new())
        } else {
            Self::FullAttention(FullAttnCache::from_options(opts))
        }
    }

    pub fn as_full_attention_mut(&mut self) -> &mut FullAttnCache {
        match self {
            Self::FullAttention(c) => c,
            Self::LinearAttention(_) => panic!("expected FullAttention slot"),
        }
    }

    pub fn as_linear_attention_mut(&mut self) -> &mut LinearAttnCache {
        match self {
            Self::LinearAttention(c) => c,
            Self::FullAttention(_) => panic!("expected LinearAttention slot"),
        }
    }

    pub fn is_linear(&self) -> bool {
        matches!(self, Self::LinearAttention(_))
    }

    /// KV-slot count for full-attn layers. `None` for linear-attn —
    /// the GDN recurrent state isn't a positional KV slot count
    /// (see [`Self::token_count`] for the GDN absorb counter).
    pub fn kv_offset(&self) -> Option<i32> {
        match self {
            Self::FullAttention(c) => Some(c.offset()),
            Self::LinearAttention(_) => None,
        }
    }

    /// Tokens absorbed so far. For full-attn this matches `kv_offset`;
    /// for linear-attn this is the GDN recurrent counter (incremented
    /// per absorbed token, never trimmed).
    pub fn token_count(&self) -> i32 {
        match self {
            Self::FullAttention(c) => c.offset(),
            Self::LinearAttention(c) => c.offset,
        }
    }
}

pub fn make_caches(config: &ModelConfig, opts: CacheOptions) -> Vec<LayerCache> {
    let n = config.text_config.num_hidden_layers as usize;
    (0..n)
        .map(|i| LayerCache::for_layer(config.is_linear_layer(i), opts))
        .collect()
}

/// MTP layers are always self_attn — every slot is `FullAttention`.
pub fn make_mtp_caches(config: &ModelConfig, opts: CacheOptions) -> Vec<LayerCache> {
    let n = config.text_config.mtp_num_hidden_layers.max(0) as usize;
    (0..n).map(|_| LayerCache::for_layer(false, opts)).collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    #![allow(clippy::print_stdout, reason = "test code")]
    #![allow(clippy::print_stderr, reason = "test code")]
    use super::*;

    fn synthetic_config(layer_types: Vec<&str>) -> ModelConfig {
        let layers: Vec<String> = layer_types.into_iter().map(String::from).collect();
        let n = layers.len() as i32;
        let json = serde_json::json!({
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": n,
                "num_attention_heads": 4,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "rms_norm_eps": 1e-6,
                "vocab_size": 100,
                "max_position_embeddings": 256,
                "layer_types": layers,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 4,
                "linear_conv_kernel_dim": 4,
                "rope_parameters": {
                    "mrope_section": [2, 1, 1],
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 1.0,
                    "type": "default"
                }
            },
            "vision_config": {
                "depth": 2,
                "hidden_size": 16,
                "intermediate_size": 32,
                "out_hidden_size": 32,
                "num_heads": 2,
                "patch_size": 16,
                "in_channels": 3,
                "spatial_merge_size": 2
            }
        });
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn make_caches_matches_layer_types() {
        let cfg = synthetic_config(vec![
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
        ]);
        let caches = make_caches(&cfg, CacheOptions::default());
        assert_eq!(caches.len(), 4);
        assert!(caches[0].is_linear());
        assert!(caches[1].is_linear());
        assert!(!caches[2].is_linear());
        assert!(caches[3].is_linear());
    }

    #[test]
    fn for_layer_dispatches_to_correct_variant() {
        let a = LayerCache::for_layer(true, CacheOptions::default());
        assert!(matches!(a, LayerCache::LinearAttention(_)));
        let b = LayerCache::for_layer(false, CacheOptions::default());
        assert!(matches!(b, LayerCache::FullAttention(_)));
    }

    #[test]
    fn linear_attn_cache_starts_empty() {
        let c = LinearAttnCache::new();
        assert!(c.conv_state.is_none());
        assert!(c.recurrent_state.is_none());
        assert_eq!(c.offset, 0);
    }
}
