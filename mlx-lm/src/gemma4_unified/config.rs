//! Gemma 4 Unified (`gemma4_unified`) config envelope.
//!
//! The unified 12B is encoder-free multimodal: vision is a matmul +
//! positional embedding + norms, audio a raw projection — both bypass the
//! SigLIP/USM towers of the older [`crate::gemma4`] family. The text
//! backbone is a dense Gemma-4 decoder, so `text_config` reuses
//! [`crate::gemma4::text::config::TextConfig`] verbatim.
//!
//! Vision/audio sub-configs and their token ids are parsed but unused until
//! those milestones land.

use serde::Deserialize;

use crate::family::EosSpec;
use crate::gemma4::text::config::TextConfig;
#[cfg(feature = "audio")]
use crate::gemma4_unified::audio::config::AudioConfig;
#[cfg(feature = "image")]
use crate::gemma4_unified::image::config::VisionConfig;

/// Outer `config.json` body for `model_type: gemma4_unified`. Stored inside
/// [`crate::config::Family::Gemma4Unified`]; quantisation lives on the outer
/// [`crate::config::ModelConfig`].
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub text_config: TextConfig,
    /// Optional explicit EOS token(s): a single id or a list.
    #[serde(default)]
    pub eos_token_id: Option<EosSpec>,

    /// Encoder-free vision embedder config; present on VLM checkpoints.
    #[cfg(feature = "image")]
    pub vision_config: Option<VisionConfig>,

    /// Encoder-free audio embedder config; present on audio checkpoints.
    #[cfg(feature = "audio")]
    pub audio_config: Option<AudioConfig>,

    /// Placeholder token each `<image>` expands to (one per soft token).
    #[serde(default = "default_image_token_id")]
    pub image_token_id: u32,
    /// Begin/end-of-image marker tokens wrapping the placeholder block.
    #[serde(default = "default_boi_token_id")]
    pub boi_token_id: u32,
    #[serde(default = "default_eoi_token_id")]
    pub eoi_token_id: u32,

    /// Placeholder token each `<audio>` expands to (one per soft token).
    #[serde(default = "default_audio_token_id")]
    pub audio_token_id: u32,
    /// Begin/end-of-audio marker tokens wrapping the placeholder block.
    #[serde(default = "default_boa_token_id")]
    pub boa_token_id: u32,
    #[serde(default = "default_eoa_token_id")]
    pub eoa_token_id: u32,
}

const fn default_image_token_id() -> u32 {
    258880
}
const fn default_boi_token_id() -> u32 {
    255999
}
const fn default_eoi_token_id() -> u32 {
    258882
}
const fn default_audio_token_id() -> u32 {
    258881
}
const fn default_boa_token_id() -> u32 {
    256000
}
const fn default_eoa_token_id() -> u32 {
    258883
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    use crate::config::Family;
    use crate::gemma4::text::config::LayerKind;

    /// The unified `model_type` routes to [`Family::Gemma4Unified`] and the
    /// 12B text backbone parses (dense, hybrid sliding/global, proportional
    /// rope on full layers).
    #[test]
    fn unified_config_routes_to_gemma4_unified() {
        let json = serde_json::json!({
            "model_type": "gemma4_unified",
            "image_token_id": 258880,
            "audio_token_id": 258881,
            "text_config": {
                "model_type": "gemma4_unified_text",
                "hidden_size": 3840,
                "num_hidden_layers": 6,
                "intermediate_size": 15360,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "num_global_key_value_heads": 1,
                "head_dim": 256,
                "global_head_dim": 512,
                "attention_k_eq_v": true,
                "sliding_window": 1024,
                "final_logit_softcapping": 30.0,
                "tie_word_embeddings": true,
                "layer_types": [
                    "sliding_attention", "sliding_attention", "sliding_attention",
                    "sliding_attention", "sliding_attention", "full_attention"
                ],
                "rope_parameters": {
                    "full_attention": {
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1000000.0,
                        "rope_type": "proportional"
                    },
                    "sliding_attention": {
                        "rope_theta": 10000.0,
                        "rope_type": "default"
                    }
                }
            }
        });
        let family: Family = serde_json::from_value(json).unwrap();
        let env = family
            .as_gemma4_unified()
            .expect("routes to gemma4_unified");
        assert_eq!(env.image_token_id, 258880);
        let types = env.text_config.layer_types_resolved();
        assert_eq!(types[0], LayerKind::SlidingAttention);
        assert_eq!(types[5], LayerKind::FullAttention);
    }
}
