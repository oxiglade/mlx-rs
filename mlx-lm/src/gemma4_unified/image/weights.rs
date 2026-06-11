//! Gemma 4 Unified VLM loader: the text [`Model`] (via the shared text loader)
//! plus the encoder-free [`VisionEmbedder`].
//!
//! Vision keys (`vision_embedder.*`, `embed_vision.embedding_projection.*`) are
//! dropped by the text loader, so they bind here in a second pass. The
//! embedder's `patch_dense` / `embedding_projection` are quantised per
//! `cfg.quantization()` (body bits; they carry no per-tensor override).

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::module::ModuleParameters;
use mlx_rs::quantization::Quantizable;
use mlx_rs::transforms::eval_params;
use mlx_rs::Array;

use crate::config::ModelConfig as Config;
use crate::error::Error;
use crate::gemma4::text::text::Model;
use crate::gemma4::text::weights::load_model;
use crate::gemma4_unified::config::ModelConfig;
use crate::gemma4_unified::image::embedder::VisionEmbedder;
use crate::loader::{apply_post_load_memory_policy, list_shards, rewrite_quantised_keys};

/// The text model + the encoder-free vision embedder.
pub struct LoadedVlm {
    pub text: Model,
    pub embedder: VisionEmbedder,
}

/// Strip the `embed_vision.embedding_projection` prefix onto the embedder's
/// `embedding_projection` param, and keep `vision_embedder.*` verbatim (the
/// embedder field names match the checkpoint).
fn rewrite_vision_key(key: &str) -> Option<String> {
    if let Some(rest) = key.strip_prefix("vision_embedder.") {
        return Some(rest.to_owned());
    }
    if let Some(rest) = key.strip_prefix("embed_vision.") {
        return Some(rest.to_owned());
    }
    None
}

pub(crate) fn load_full_model(
    cfg: &Config,
    env: &ModelConfig,
    dir: &Path,
) -> Result<LoadedVlm, Error> {
    let text = load_model(cfg, &env.text_config, dir)?;

    let vcfg = env
        .vision_config
        .as_ref()
        .ok_or_else(|| Error::config("gemma4_unified vlm: vision_config missing"))?;
    let mut embedder = VisionEmbedder::new(vcfg)?;
    if let Some(q) = cfg.quantization() {
        embedder = embedder.try_into_quantized(q.group_size, q.bits)?;
    }

    let shards = list_shards(dir)?;
    let mut raw: HashMap<String, Array> = HashMap::new();
    for path in shards {
        let loaded = Array::load_safetensors(&path).map_err(Error::LoadWeights)?;
        for (k, v) in loaded {
            if let Some(key) = rewrite_vision_key(&k) {
                raw.insert(key, v);
            }
        }
    }
    let weights = rewrite_quantised_keys(raw);

    let mut leftover: Vec<String> = Vec::new();
    {
        let mut params = embedder.parameters_mut().flatten();
        for (k, v) in weights {
            if let Some(slot) = params.get_mut(&*k) {
                **slot = v;
            } else {
                leftover.push(k);
            }
        }
    }
    if !leftover.is_empty() {
        leftover.sort();
        return Err(Error::Other(
            format!(
                "gemma4_unified vlm: {} unbound vision key(s); first 8: {:?}",
                leftover.len(),
                &leftover.iter().take(8).collect::<Vec<_>>()
            )
            .into(),
        ));
    }
    eval_params(embedder.parameters()).map_err(Error::Exception)?;
    apply_post_load_memory_policy();
    Ok(LoadedVlm { text, embedder })
}
