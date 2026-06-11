//! Shared loader helpers: weight-shard discovery + post-load memory policy.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use mlx_rs::Array;
use tokenizers::Tokenizer;

use crate::error::Error;

/// MLX cache-pool cap applied after every weight load (20 MB, matches
/// mlx-swift LLM guidance). Override via `set_cache_limit_override` or
/// the `MLX_LM_CACHE_LIMIT_BYTES` env var.
pub const DEFAULT_CACHE_LIMIT_BYTES: usize = 20 * 1024 * 1024;

static CACHE_LIMIT_OVERRIDE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// First call wins. `0` disables reuse entirely.
pub fn set_cache_limit_override(bytes: usize) {
    let _ = CACHE_LIMIT_OVERRIDE.set(bytes);
}

/// Precedence: `set_cache_limit_override` > `MLX_LM_CACHE_LIMIT_BYTES` env > [`DEFAULT_CACHE_LIMIT_BYTES`].
fn resolved_cache_limit() -> usize {
    if let Some(&n) = CACHE_LIMIT_OVERRIDE.get() {
        return n;
    }
    if let Some(n) = parse_env_bytes("MLX_LM_CACHE_LIMIT_BYTES") {
        return n;
    }
    DEFAULT_CACHE_LIMIT_BYTES
}

/// Drain the MLX cache pool then apply the resolved cap. Reclaims the
/// safetensors scratch buffers parked in the reuse pool after load.
pub fn apply_post_load_memory_policy() {
    mlx_rs::memory::clear_cache();
    mlx_rs::memory::set_cache_limit(resolved_cache_limit());
}

fn parse_env_bytes(name: &str) -> Option<usize> {
    let raw = std::env::var(name).ok()?;
    raw.trim().parse::<usize>().ok()
}

/// Load `<dir>/tokenizer.json`.
pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

/// BOS id to prepend to raw prompts, or `None` when the model declares no
/// BOS or opts out (`add_bos_token: false`).
///
/// The fast tokenizer's `post_processor` frequently omits BOS even with
/// `add_special_tokens` (e.g. mlx-community Gemma drops it), so we resolve
/// the policy from `tokenizer_config.json` and prepend in the processor.
/// `add_bos_token` is honoured when present; when absent we follow the
/// HuggingFace default of prepending whenever a `bos_token` exists
/// (Gemma, Llama 3 — both BOS-sensitive). Qwen sets `bos_token: null`, so
/// this returns `None` there.
pub fn resolve_bos_id(model_dir: impl AsRef<Path>, tokenizer: &Tokenizer) -> Option<u32> {
    let path = model_dir.as_ref().join("tokenizer_config.json");
    let raw = std::fs::read_to_string(path).ok()?;
    let cfg: serde_json::Value = serde_json::from_str(&raw).ok()?;
    if cfg.get("add_bos_token").and_then(|v| v.as_bool()) == Some(false) {
        return None;
    }
    let bos = cfg.get("bos_token").and_then(bos_token_str)?;
    tokenizer.token_to_id(bos)
}

/// `bos_token` is either a plain string or an `AddedToken` object with a
/// `content` field.
fn bos_token_str(v: &serde_json::Value) -> Option<&str> {
    v.as_str()
        .or_else(|| v.get("content").and_then(|c| c.as_str()))
}

/// Safetensors shard paths: single `model.safetensors`, else the unique
/// shards in `model.safetensors.index.json`, sorted.
pub fn list_shards(model_dir: &Path) -> Result<Vec<PathBuf>, Error> {
    let single = model_dir.join("model.safetensors");
    if single.is_file() {
        return Ok(vec![single]);
    }
    let index = model_dir.join("model.safetensors.index.json");
    let json = std::fs::read_to_string(&index)?;
    let parsed: serde_json::Value = serde_json::from_str(&json)?;
    let weight_map = parsed
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| Error::config("index.json missing weight_map"))?;
    let mut shards: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(String::from))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    shards.sort();
    Ok(shards.into_iter().map(|s| model_dir.join(s)).collect())
}

/// Redirect quantised `<prefix>.weight` → `<prefix>.inner.weight` (and a
/// `<prefix>.bias`, if any, → `<prefix>.inner.bias`) for keys whose
/// `<prefix>.scales` sibling exists (MaybeQuantized layout). A quantised
/// `Linear` keeps its real weight/bias on the wrapped `inner` linear; the
/// quant `scales`/`biases` are siblings.
pub fn rewrite_quantised_keys(raw: HashMap<String, Array>) -> HashMap<String, Array> {
    let quantised_prefixes: HashSet<String> = raw
        .keys()
        .filter_map(|k| k.strip_suffix(".scales").map(|p| p.to_string()))
        .collect();
    raw.into_iter()
        .map(|(k, v)| {
            if let Some(prefix) = k.strip_suffix(".weight") {
                if quantised_prefixes.contains(prefix) {
                    return (format!("{prefix}.inner.weight"), v);
                }
            }
            if let Some(prefix) = k.strip_suffix(".bias") {
                if quantised_prefixes.contains(prefix) {
                    return (format!("{prefix}.inner.bias"), v);
                }
            }
            (k, v)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    use super::*;

    /// A quantised linear with a bias: `weight`→`inner.weight`,
    /// `bias`→`inner.bias`; the quant `scales`/`biases` siblings stay put.
    #[test]
    fn quantised_keys_redirect_weight_and_bias_to_inner() {
        let mk = || Array::from_slice(&[0.0f32], &[1]);
        let raw: HashMap<String, Array> = [
            ("patch_dense.weight", mk()),
            ("patch_dense.bias", mk()),
            ("patch_dense.scales", mk()),
            ("patch_dense.biases", mk()),
            ("norm.weight", mk()), // un-quantised: untouched
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();

        let out = rewrite_quantised_keys(raw);
        assert!(out.contains_key("patch_dense.inner.weight"));
        assert!(out.contains_key("patch_dense.inner.bias"));
        // Quant siblings and un-quantised keys are unchanged.
        assert!(out.contains_key("patch_dense.scales"));
        assert!(out.contains_key("patch_dense.biases"));
        assert!(out.contains_key("norm.weight"));
        assert!(!out.contains_key("patch_dense.weight"));
        assert!(!out.contains_key("patch_dense.bias"));
    }
}
