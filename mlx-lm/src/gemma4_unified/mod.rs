//! Gemma 4 Unified family (`gemma4_unified`): encoder-free multimodal 12B.
//!
//! Text milestone: dense decoder (reusing [`crate::gemma4::text`]) + MTP
//! speculative decode (reusing [`crate::gemma4::mtp`]). Encoder-free vision
//! and audio front-ends land in follow-on milestones.

use std::path::Path;

use crate::config::ModelConfig;
use crate::error::Error;
use crate::family::LoadedContext;

pub mod adapter;
#[cfg(feature = "audio")]
pub mod audio;
pub mod config;
#[cfg(feature = "image")]
pub mod image;

pub(crate) fn load_context(
    cfg: &ModelConfig,
    dir: &Path,
    draft_dir: Option<&Path>,
) -> Result<LoadedContext, Error> {
    let env = cfg.family.as_gemma4_unified().ok_or_else(|| {
        Error::config("gemma4_unified::load_context: not a gemma4_unified config")
    })?;

    // VLM checkpoint (vision_config + processor_config.json) → multimodal
    // adapter. A drafter composes: image affects prefill only, MTP decodes off
    // the populated KV cache.
    #[cfg(feature = "image")]
    if env.vision_config.is_some() && dir.join("processor_config.json").exists() {
        return image::adapter::load_context_vlm(cfg, env, dir, draft_dir);
    }

    adapter::load_context(cfg, env, dir, draft_dir)
}
