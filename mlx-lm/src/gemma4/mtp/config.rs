//! Gemma 4 assistant (MTP drafter) config (`gemma4_assistant`).
//!
//! The drafter transformer reuses the gemma4 [`TextConfig`] (its `text_config`
//! is `model_type "gemma4_text"` with `num_kv_shared_layers ==
//! num_hidden_layers` → every layer is Q-only). The outer envelope adds the
//! backbone width (for the pre/post projections), the centroid lm-head flag,
//! and the draft depth.

use std::path::Path;

use serde::Deserialize;

use crate::error::Error;
use crate::gemma4::text::config::TextConfig;
use crate::quantization::QuantizationConfig;

/// Default γ. With the draft confidence gate (see
/// [`crate::speculative::draft_gate_for`]), depth-2 is the throughput optimum:
/// the second draft only extends when the head is near-certain (gate 0.999),
/// so it adds accepted tokens on easy positions without paying verify cost on
/// hard ones. Override with `--mtp-depth`.
pub const DEFAULT_DRAFT_DEPTH: u32 = 2;

/// Centroid-masked lm-head: number of top centroids whose tokens get exact
/// logits (config default).
pub const DEFAULT_CENTROID_TOP_K: i32 = 32;

#[derive(Debug, Clone, Deserialize)]
pub struct DrafterConfig {
    /// The drafter's own transformer config (draft hidden dim, 4 layers, all
    /// KV-shared).
    pub text_config: TextConfig,
    /// Target (backbone) hidden size — the pre/post projection widths.
    pub backbone_hidden_size: i32,
    /// E2B/E4B use the centroid-masked sparse lm head.
    #[serde(default)]
    pub use_ordered_embeddings: bool,
    #[serde(default = "default_num_centroids")]
    pub num_centroids: i32,
    #[serde(default = "default_centroid_top_k")]
    pub centroid_intermediate_top_k: i32,
    /// Quantisation of the drafter weights (present on quantised assistant
    /// checkpoints, e.g. `*-assistant-8bit`); `None` for bf16 drafters.
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

const fn default_num_centroids() -> i32 {
    2048
}
const fn default_centroid_top_k() -> i32 {
    DEFAULT_CENTROID_TOP_K
}

impl DrafterConfig {
    /// Parse `<dir>/config.json` into a drafter config.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, Error> {
        let path = dir.as_ref().join("config.json");
        let raw = std::fs::read_to_string(&path)?;
        Ok(serde_json::from_str(&raw)?)
    }

    /// Default γ ([`DEFAULT_DRAFT_DEPTH`]); override with `--mtp-depth`.
    pub fn default_depth(&self) -> u32 {
        DEFAULT_DRAFT_DEPTH
    }
}
