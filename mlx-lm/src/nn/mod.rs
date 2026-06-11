//! Building blocks shared across decoder models (llama, qwen3, …).

pub mod attention_input;
pub mod model_input;
pub mod rms_norm_no_scale;
pub mod router_topk;
pub mod swiglu_mlp;
pub mod switch;

pub use attention_input::AttentionInput;
pub use model_input::ModelInput;
pub use rms_norm_no_scale::RmsNormNoScale;
pub use swiglu_mlp::SwigluMlp;

use crate::cache::KeyValueCache;

/// Populate `cache` with `len` default-constructed slots if empty;
/// no-op once populated.
pub fn ensure_cache_populated<C>(cache: &mut Vec<Option<C>>, len: usize)
where
    C: KeyValueCache + Default,
{
    if cache.is_empty() {
        *cache = (0..len).map(|_| Some(C::default())).collect();
    }
}
