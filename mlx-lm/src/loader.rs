//! HuggingFace Hub loader for mlx-lm.
//!
//! Provides a sync `snapshot_download` that resolves the files needed to load
//! a model from the Hub (config, tokenizer, weights, weight index). The
//! returned `LoadedFiles` is intended to be consumed by per-model
//! `load_<arch>_model` helpers.
//!
//! API style mirrors `examples/mistral/src/main.rs` (sync, no tokio).

use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use hf_hub::{
    api::sync::{ApiBuilder, ApiRepo},
    Repo, RepoType,
};
use serde::Deserialize;
use serde_json::Value;

use crate::error::Error;

/// Mirror of the `model.safetensors.index.json` schema produced by HF
/// converters.
#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub metadata: HashMap<String, Value>,
    pub weight_map: HashMap<String, String>,
}

/// Optional knobs for `snapshot_download`. `Default` selects the `main`
/// revision, the user's default cache directory, and no auth token.
#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    /// Branch, tag, or commit SHA. Defaults to `main`.
    pub revision: Option<String>,
    /// Override the HF cache directory (otherwise `HF_HOME` / `~/.cache/huggingface`).
    pub cache_dir: Option<PathBuf>,
    /// Bearer token for gated or private repositories.
    pub token: Option<String>,
}

/// Paths to the files needed to instantiate a model. All paths point at
/// locations inside the HF cache (or wherever `LoadOptions::cache_dir`
/// pointed). `model_dir` is the parent of `config_path` and is the right
/// argument for the existing `load_<arch>_model(&Path)` helpers, provided
/// the model uses a single safetensors file in the same directory; for
/// sharded models prefer `weight_files` directly.
#[derive(Debug, Clone)]
pub struct LoadedFiles {
    pub model_dir: PathBuf,
    pub tokenizer_path: PathBuf,
    pub tokenizer_config_path: Option<PathBuf>,
    pub config_path: PathBuf,
    pub weight_files: Vec<PathBuf>,
}

fn fetch_weight_files(repo: &ApiRepo) -> Result<Vec<PathBuf>, Error> {
    match repo.get("model.safetensors.index.json") {
        Ok(index_path) => {
            let file = std::fs::File::open(&index_path)?;
            let weight_map: WeightMap = serde_json::from_reader(file)?;

            // Deduplicate by file name. Order is not significant for
            // `load_safetensors`, which is called once per file.
            let unique: HashSet<&String> = weight_map.weight_map.values().collect();
            let mut files = Vec::with_capacity(unique.len());
            for shard in unique {
                files.push(repo.get(shard)?);
            }
            Ok(files)
        }
        Err(_) => {
            // Single-file model. Surface any error here as-is.
            let single = repo.get("model.safetensors")?;
            Ok(vec![single])
        }
    }
}

/// Download (or look up in cache) the files needed to load `model_id` from
/// the HuggingFace Hub. Sync; safe to call from `main` or a test.
pub fn snapshot_download(model_id: &str, opts: &LoadOptions) -> Result<LoadedFiles, Error> {
    let mut builder = ApiBuilder::new();
    if let Some(cache_dir) = opts.cache_dir.as_ref() {
        builder = builder.with_cache_dir(cache_dir.clone());
    }
    if let Some(token) = opts.token.as_ref() {
        builder = builder.with_token(Some(token.clone()));
    }
    let api = builder.build()?;

    let revision = opts.revision.clone().unwrap_or_else(|| "main".to_string());
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision,
    ));

    let config_path = repo.get("config.json")?;
    let tokenizer_path = repo.get("tokenizer.json")?;
    let tokenizer_config_path = repo.get("tokenizer_config.json").ok();
    let weight_files = fetch_weight_files(&repo)?;

    let model_dir = config_path
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));

    Ok(LoadedFiles {
        model_dir,
        tokenizer_path,
        tokenizer_config_path,
        config_path,
        weight_files,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_weight_map_extracts_unique_files() {
        let json = r#"{"metadata":{}, "weight_map":{
            "model.layers.0.w": "shard-1.safetensors",
            "model.layers.1.w": "shard-2.safetensors",
            "model.layers.2.w": "shard-1.safetensors"
        }}"#;
        let map: WeightMap = serde_json::from_str(json).unwrap();
        let unique: HashSet<&String> = map.weight_map.values().collect();
        assert_eq!(unique.len(), 2);
    }

    #[cfg(feature = "hub-test")]
    #[test]
    #[ignore = "downloads from HF Hub"]
    fn snapshot_download_fetches_config() {
        let files =
            snapshot_download("mlx-community/Qwen3-0.6B-bf16", &LoadOptions::default()).unwrap();
        assert!(files.config_path.exists());
    }
}
