//! Gemma 4 per-layer cache slot. Full-attention layers use the shared
//! [`FullAttnCache`]; sliding-attention layers use [`RotatingKVCache`].

use mlx_rs::{error::Exception, Array};

use crate::cache::{CacheOptions, FullAttnCache, KeyValueCache, RotatingKVCache};
use crate::gemma4::text::config::{LayerKind, TextConfig};

/// One decoder layer's KV cache: a dense global slot or a sliding-window
/// ring buffer, chosen by the layer's [`LayerKind`].
#[derive(Debug, Clone)]
pub enum LayerCache {
    Global(FullAttnCache),
    Sliding(RotatingKVCache),
}

impl KeyValueCache for LayerCache {
    fn is_quantized(&self) -> bool {
        match self {
            Self::Global(c) => c.is_quantized(),
            Self::Sliding(c) => c.is_quantized(),
        }
    }

    fn group_size(&self) -> Option<i32> {
        match self {
            Self::Global(c) => c.group_size(),
            Self::Sliding(c) => c.group_size(),
        }
    }

    fn bits(&self) -> Option<i32> {
        match self {
            Self::Global(c) => c.bits(),
            Self::Sliding(c) => c.bits(),
        }
    }

    fn offset(&self) -> i32 {
        match self {
            Self::Global(c) => c.offset(),
            Self::Sliding(c) => c.offset(),
        }
    }

    fn max_size(&self) -> Option<i32> {
        match self {
            Self::Global(c) => c.max_size(),
            Self::Sliding(c) => c.max_size(),
        }
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        match self {
            Self::Global(c) => c.update_and_fetch(keys, values),
            Self::Sliding(c) => c.update_and_fetch(keys, values),
        }
    }

    fn current_kv(&self) -> Result<Option<(Array, Array)>, Exception> {
        match self {
            Self::Global(c) => c.current_kv(),
            Self::Sliding(c) => c.current_kv(),
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
            Self::Global(c) => c.attention(queries, keys, values, scale, mask),
            Self::Sliding(c) => c.attention(queries, keys, values, scale, mask),
        }
    }
}

/// Build one cache slot per layer, dispatching on the resolved layer
/// kinds: full-attention → dense [`FullAttnCache`], sliding-attention →
/// [`RotatingKVCache`] of `sliding_window` capacity (no keep prefix).
pub fn make_caches(args: &TextConfig, opts: CacheOptions) -> Vec<Option<LayerCache>> {
    // KV-shared layers own no cache (reuse a prior layer's K/V) → `None`.
    let first_kv_shared = args.num_hidden_layers - args.num_kv_shared_layers;
    args.layer_types_resolved()
        .into_iter()
        .enumerate()
        .map(|(i, kind)| {
            if args.num_kv_shared_layers > 0 && i as i32 >= first_kv_shared {
                return None;
            }
            Some(match kind {
                LayerKind::FullAttention => LayerCache::Global(FullAttnCache::from_options(opts)),
                LayerKind::SlidingAttention => {
                    LayerCache::Sliding(RotatingKVCache::new(args.sliding_window, 0))
                }
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    use super::*;
    use crate::gemma4::text::config::TextConfig;

    fn cfg(layers: Vec<&str>) -> TextConfig {
        let n = layers.len() as i32;
        let json = serde_json::json!({
            "num_hidden_layers": n,
            "sliding_window": 16,
            "layer_types": layers,
        });
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn make_caches_dispatches_per_layer_kind() {
        let c = cfg(vec![
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ]);
        let caches = make_caches(&c, CacheOptions::default());
        assert_eq!(caches.len(), 3);
        assert!(matches!(caches[0], Some(LayerCache::Sliding(_))));
        assert!(matches!(caches[1], Some(LayerCache::Sliding(_))));
        assert!(matches!(caches[2], Some(LayerCache::Global(_))));
    }

    #[test]
    fn sliding_slot_reports_window_as_max_size() {
        let c = cfg(vec!["sliding_attention", "full_attention"]);
        let caches = make_caches(&c, CacheOptions::default());
        // The sliding slot's max_size drives the windowed attention mask.
        assert_eq!(caches[0].as_ref().unwrap().max_size(), Some(16));
        // The global slot is unbounded.
        assert_eq!(caches[1].as_ref().unwrap().max_size(), None);
    }
}
