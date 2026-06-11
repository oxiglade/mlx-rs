//! Gemma 4 Unified audio embedder loader: bind `embed_audio.embedding_projection`.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::module::ModuleParameters;
use mlx_rs::quantization::Quantizable;
use mlx_rs::transforms::eval_params;
use mlx_rs::Array;

use crate::config::ModelConfig as Config;
use crate::error::Error;
use crate::gemma4_unified::audio::config::AudioConfig;
use crate::gemma4_unified::audio::embedder::AudioEmbedder;
use crate::loader::{apply_post_load_memory_policy, list_shards, rewrite_quantised_keys};

pub(crate) fn load_audio_embedder(
    cfg: &Config,
    acfg: &AudioConfig,
    dir: &Path,
) -> Result<AudioEmbedder, Error> {
    let mut embedder = AudioEmbedder::new(acfg)?;
    if let Some(q) = cfg.quantization() {
        embedder = embedder.try_into_quantized(q.group_size, q.bits)?;
    }

    let shards = list_shards(dir)?;
    let mut raw: HashMap<String, Array> = HashMap::new();
    for path in shards {
        let loaded = Array::load_safetensors(&path).map_err(Error::LoadWeights)?;
        for (k, v) in loaded {
            if let Some(rest) = k.strip_prefix("embed_audio.") {
                raw.insert(rest.to_owned(), v);
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
                "gemma4_unified audio: {} unbound key(s); first 8: {:?}",
                leftover.len(),
                &leftover.iter().take(8).collect::<Vec<_>>()
            )
            .into(),
        ));
    }
    eval_params(embedder.parameters()).map_err(Error::Exception)?;
    apply_post_load_memory_policy();
    Ok(embedder)
}
