//! Gemma 4 family (text). Dense base: sliding/global hybrid attention,
//! four norms per layer, GeGLU MLP, logit soft-capping, tied embeddings.
//! MoE / per-layer-input embeddings / KV-sharing / vision are deferred.

use std::path::Path;

use crate::config::ModelConfig;
use crate::error::Error;
use crate::family::LoadedContext;

#[cfg(feature = "audio")]
pub mod audio;
#[cfg(feature = "image")]
pub mod image;
pub mod mtp;
pub mod text;
#[cfg(feature = "image")]
pub mod vlm;

pub(crate) fn load_context(
    cfg: &ModelConfig,
    dir: &Path,
    draft_dir: Option<&Path>,
) -> Result<LoadedContext, Error> {
    let env = cfg
        .family
        .as_gemma4()
        .ok_or_else(|| Error::config("gemma4::load_context: not a gemma4 config"))?;

    // VLM checkpoint (`vision_config` + `processor_config.json`) → multimodal
    // adapter, unless a drafter is requested: MTP runs on the text path only,
    // so route to the text loader (tower keys dropped).
    #[cfg(feature = "image")]
    if draft_dir.is_none()
        && env.vision_config.is_some()
        && dir.join("processor_config.json").exists()
    {
        return vlm::adapter::load_context_vlm(cfg, env, dir);
    }

    text::load_context(cfg, env, dir, draft_dir)
}
