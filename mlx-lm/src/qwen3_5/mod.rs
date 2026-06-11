//! Qwen3.5 family: hybrid full-attention + gated-delta-net linear
//! attention. Dense + MoE text paths, and (behind the `image` feature)
//! the VLM path.

#[cfg(feature = "image")]
pub mod image;
pub mod text;

use std::path::Path;

use crate::config::{Family, ModelConfig};
use crate::error::Error;
use crate::family::LoadedContext;

pub(crate) fn load_context(cfg: &ModelConfig, dir: &Path) -> Result<LoadedContext, Error> {
    let env = cfg
        .family
        .as_qwen35()
        .ok_or_else(|| Error::config("qwen3_5::load_context: not a qwen3.5 config"))?;
    // MoE checkpoints are text-only (VL-MoE is out of scope).
    if matches!(cfg.family, Family::Qwen35Moe(_)) {
        return text::adapter_moe::load_context_moe(cfg, env, dir);
    }
    // Dense or VL: a checkpoint carrying `vision_config` + a
    // `preprocessor_config.json` is a VLM; route to the vision adapter
    // when the `image` feature is on, else fall back to text-only.
    let is_vlm = env.vision_config.is_some() && dir.join("preprocessor_config.json").exists();
    if is_vlm {
        #[cfg(feature = "image")]
        {
            return image::adapter::load_context_vlm(cfg, env, dir);
        }
        #[cfg(not(feature = "image"))]
        {
            log::warn!(
                "qwen3_5: VL checkpoint at {} loaded text-only (build without `image` feature)",
                dir.display()
            );
        }
    }
    text::load_context_dense(cfg, env, dir)
}
