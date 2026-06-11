//! Configuration types for the Qwen3.5 family of models.
//!
//! Both [`TextConfig`] and [`VisionConfig`] are deserialized straight
//! from the `text_config` / `vision_config` sub-objects of the
//! model's `config.json`.

use serde::Deserialize;

use crate::error::Error;
use crate::family::EosSpec;

/// Per-layer architecture tag from `config.json::layer_types`. Unknown
/// strings hard-error rather than silently routing as one or the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QwenLayerKind {
    /// Gated DeltaNet (linear-attention SSM).
    LinearAttention,
    /// Regular GQA full attention.
    FullAttention,
}

/// Rotary embedding variant from `rope_parameters.type`. Qwen3.5
/// implements `default` and `mrope`; yarn / longrope are rejected at
/// model build time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QwenRopeType {
    /// Standard RoPE.
    #[default]
    Default,
    /// Multimodal RoPE used by the VL variants.
    Mrope,
}

/// Default EOS for Qwen chat templates. Used as a fallback when `eos_token_id`
/// is missing from `config.json`.
pub const QWEN_CHAT_EOS_TOKEN_ID: u32 = 248046;

/// Parameters for the Qwen3.5 multimodal RoPE.
///
/// `mrope_section` slices the rotary dimension into three independent axes
/// (t/h/w of the multimodal grid). `partial_rotary_factor` keeps only the
/// first `head_dim * partial_rotary_factor` features rotated and passes the
/// rest through unchanged.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    /// Lengths of the three mrope axes — must sum to `head_dim *
    /// partial_rotary_factor / 2`.
    pub mrope_section: Vec<i32>,
    /// Base used to compute angular frequency. Renamed to `rope_theta` in some
    /// checkpoints; serde reads either via the field name.
    pub rope_theta: f32,
    /// Fraction of `head_dim` that is rotated. 0.25 for Qwen3.5.
    pub partial_rotary_factor: f32,
    /// RoPE variant. Qwen3.5 implements `default` and `mrope`;
    /// yarn / longrope deserialize-fail before reaching `Attention::new`.
    #[serde(default, rename = "type", alias = "rope_type")]
    pub rope_type: QwenRopeType,
    /// Some configs emit a top-level `mrope_interleaved` flag; we accept it but
    /// the interleaved layout is the only one currently implemented.
    #[serde(default)]
    pub mrope_interleaved: bool,
}

/// Text-decoder hyperparameters for Qwen3.5.
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: i32,
    /// Dense MLP intermediate size. Absent on MoE variants
    /// (Qwen3.6-35B-A3B etc.) which use [`Self::moe_intermediate_size`]
    /// for the routed experts instead.
    #[serde(default)]
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub max_position_embeddings: i32,

    /// Per-layer architecture tag. Length must equal `num_hidden_layers`.
    pub layer_types: Vec<QwenLayerKind>,

    /// Every `full_attention_interval`-th layer is a full-attention layer.
    /// Used when [`Self::layer_types`] is empty.
    #[serde(default = "default_full_attention_interval")]
    pub full_attention_interval: i32,

    /// Number of key heads for the linear-attention (Gated DeltaNet) block.
    pub linear_num_key_heads: i32,
    /// Number of value heads for the linear-attention block.
    pub linear_num_value_heads: i32,
    /// Per-head key dim for the linear-attention block.
    pub linear_key_head_dim: i32,
    /// Per-head value dim for the linear-attention block.
    pub linear_value_head_dim: i32,
    /// Causal Conv1d kernel size used inside the GDN block.
    pub linear_conv_kernel_dim: i32,

    /// Whether the LM head is tied to `embed_tokens`.
    #[serde(default)]
    pub tie_word_embeddings: bool,

    /// Bias on `q_proj`/`k_proj`/`v_proj`/`o_proj`. Always false for Qwen3.5.
    #[serde(default)]
    pub attention_bias: bool,

    /// If true, full-attention layers add a sigmoid gate on the attention
    /// output (`output = o_proj(attn_out * sigmoid(gate))`). Always true for
    /// Qwen3.5.
    ///
    /// Qwen 3.6 configs also carry `output_gate_type: "swish"` — vestigial;
    /// upstream implementations unconditionally compute
    /// `output * sigmoid(gate)` regardless. Field is silently dropped by
    /// serde here; do not re-parse without confirming the reference path
    /// actually branches on it.
    #[serde(default = "default_attn_output_gate")]
    pub attn_output_gate: bool,

    /// Rotary embedding parameters.
    pub rope_parameters: RopeParameters,

    // ── MoE fields (Qwen3.6-35B-A3B; absent on dense checkpoints) ──
    /// Number of routed experts. `0` on dense checkpoints; `is_moe()`
    /// gates on this.
    #[serde(default)]
    pub num_experts: i32,
    /// Top-k routing fan-out per token.
    #[serde(default)]
    pub num_experts_per_tok: i32,
    /// Inner hidden width per routed expert.
    #[serde(default)]
    pub moe_intermediate_size: i32,
    /// Inner hidden width of the always-on dense shared expert.
    #[serde(default)]
    pub shared_expert_intermediate_size: i32,

    // ── MTP (Multi-Token Prediction) fields ──
    /// Number of MTP layers (0 disables).
    #[serde(default)]
    pub mtp_num_hidden_layers: i32,
    /// If false, the MTP head shares `embed_tokens` with the main decoder.
    #[serde(default)]
    pub mtp_use_dedicated_embeddings: bool,
}

impl TextConfig {
    /// True for MoE variants (any non-zero `num_experts`).
    pub fn is_moe(&self) -> bool {
        self.num_experts > 0
    }

    /// True if decoder layer `i` is linear-attention (Gated DeltaNet)
    /// rather than full-attention. Prefers explicit `layer_types`; falls
    /// back to the `full_attention_interval` heuristic for checkpoints
    /// that omit the per-layer list (every `interval`-th layer is full).
    pub fn is_linear_layer(&self, layer_idx: usize) -> bool {
        if !self.layer_types.is_empty() {
            return self
                .layer_types
                .get(layer_idx)
                .map(|k| *k == QwenLayerKind::LinearAttention)
                .unwrap_or(false);
        }
        let interval = self.full_attention_interval;
        interval > 0 && ((layer_idx as i32 + 1) % interval) != 0
    }

    /// Reject a config that can't drive per-layer dispatch. Either an
    /// explicit `layer_types` sized to `num_hidden_layers`, or a positive
    /// `full_attention_interval` to fall back on, is required.
    pub fn validate(&self) -> Result<(), Error> {
        if !self.layer_types.is_empty() && self.layer_types.len() != self.num_hidden_layers as usize
        {
            return Err(Error::config(format!(
                "qwen3.5: layer_types len {} != num_hidden_layers {}",
                self.layer_types.len(),
                self.num_hidden_layers
            )));
        }
        if self.layer_types.is_empty() && self.full_attention_interval <= 0 {
            return Err(Error::config(
                "qwen3.5: empty layer_types and full_attention_interval <= 0; \
                 cannot dispatch linear vs full attention",
            ));
        }
        Ok(())
    }
}

fn default_full_attention_interval() -> i32 {
    4
}

fn default_attn_output_gate() -> bool {
    true
}

/// Vision-tower model variants understood by [`VisionModel`]. Deserialise
/// rejects unknown discriminants at config-load time; that catches a typo
/// or a future Qwen vision-tower variant up-front rather than at the
/// `VisionModel::new` shape check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum VisionModelType {
    /// Qwen3-VL (legacy 3-series VL tower).
    #[serde(rename = "qwen3_vl")]
    Qwen3Vl,
    /// Qwen 3.5 dense / VLM tower.
    #[serde(rename = "qwen3_5")]
    Qwen35,
    /// Qwen 3.5 MoE tower (shares the same architecture as `Qwen35`).
    #[serde(rename = "qwen3_5_moe")]
    Qwen35Moe,
}

/// Vision-tower hyperparameters (shared with Qwen3-VL, with Qwen3.5 defaults).
#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    #[serde(default = "default_vision_model_type")]
    pub model_type: VisionModelType,
    pub depth: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub out_hidden_size: i32,
    pub num_heads: i32,
    pub patch_size: i32,
    pub in_channels: i32,
    pub spatial_merge_size: i32,
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: i32,
    #[serde(default = "default_num_position_embeddings")]
    pub num_position_embeddings: i32,
    /// Layer indices in the ViT where intermediate hidden states are injected
    /// back into the LM residual stream. Empty for Qwen3.5 (deepstack
    /// disabled).
    #[serde(default)]
    pub deepstack_visual_indexes: Vec<i32>,
}

fn default_vision_model_type() -> VisionModelType {
    VisionModelType::Qwen35
}

fn default_temporal_patch_size() -> i32 {
    2
}

fn default_num_position_embeddings() -> i32 {
    2304
}

/// Qwen-family envelope of `config.json`. Stored inside
/// [`crate::config::Family::Qwen35`] / [`crate::config::Family::Qwen35Moe`].
/// Quantisation lives on the outer [`crate::config::ModelConfig`].
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub text_config: TextConfig,
    /// Text-only checkpoints (Qwen 3.6 MoE) omit this entirely; the
    /// VLM adapter requires it and errors at its own load_context if
    /// absent.
    #[serde(default)]
    pub vision_config: Option<VisionConfig>,

    #[serde(default = "default_image_token_id")]
    pub image_token_id: u32,
    #[serde(default = "default_video_token_id")]
    pub video_token_id: u32,
    #[serde(default = "default_vision_start_token_id")]
    pub vision_start_token_id: u32,
    #[serde(default = "default_vision_end_token_id")]
    pub vision_end_token_id: u32,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub eos_token_id: Option<EosSpec>,
}

fn default_image_token_id() -> u32 {
    248056
}
fn default_video_token_id() -> u32 {
    248057
}
fn default_vision_start_token_id() -> u32 {
    248045
}
fn default_vision_end_token_id() -> u32 {
    248046
}

impl ModelConfig {
    /// Returns true for the `i`-th decoder layer if it is a linear-attention
    /// (Gated DeltaNet) layer rather than a full-attention layer.
    ///
    /// Delegates to [`TextConfig::is_linear_layer`] (explicit `layer_types`
    /// with `full_attention_interval` fallback).
    pub fn is_linear_layer(&self, layer_idx: usize) -> bool {
        self.text_config.is_linear_layer(layer_idx)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    use super::*;
    use crate::config::ModelConfig as Config;

    const DENSE_CONFIG_JSON: &str = r#"
    {
        "model_type": "qwen3_5",
        "text_config": {
            "attn_output_gate": true,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_size": 2560,
            "intermediate_size": 9216,
            "layer_types": [
                "linear_attention", "linear_attention", "linear_attention", "full_attention"
            ],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "max_position_embeddings": 262144,
            "model_type": "qwen3_5_text",
            "num_attention_heads": 16,
            "num_hidden_layers": 4,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default"
            },
            "tie_word_embeddings": true,
            "vocab_size": 248320
        }
    }
    "#;

    fn env(json: &str) -> ModelConfig {
        let cfg: Config = serde_json::from_str(json).unwrap();
        cfg.family.as_qwen35().expect("qwen3_5 family").clone()
    }

    #[test]
    fn parses_dense_config() {
        let e = env(DENSE_CONFIG_JSON);
        assert_eq!(e.text_config.hidden_size, 2560);
        assert_eq!(e.text_config.num_hidden_layers, 4);
        assert_eq!(e.text_config.layer_types.len(), 4);
        assert!(!e.text_config.is_moe());
    }

    #[test]
    fn layer_type_dispatch() {
        let e = env(DENSE_CONFIG_JSON);
        assert!(e.is_linear_layer(0));
        assert!(e.is_linear_layer(2));
        assert!(!e.is_linear_layer(3));
    }

    /// Checkpoints that omit `layer_types` must fall back to the
    /// `full_attention_interval` heuristic (every `interval`-th layer
    /// full), and `validate()` must accept them.
    #[test]
    fn interval_fallback_when_layer_types_empty() {
        let json = DENSE_CONFIG_JSON.replace(
            r#""layer_types": [
                "linear_attention", "linear_attention", "linear_attention", "full_attention"
            ],"#,
            r#""layer_types": [],"#,
        );
        let e = env(&json);
        assert!(e.text_config.layer_types.is_empty());
        e.text_config.validate().unwrap();
        // interval 4: layers 0,1,2 linear; layer 3 (idx+1 % 4 == 0) full.
        assert!(e.is_linear_layer(0));
        assert!(e.is_linear_layer(2));
        assert!(!e.is_linear_layer(3));
    }

    /// A short `layer_types` (present but mis-sized) is still rejected.
    #[test]
    fn validate_rejects_mismatched_layer_types() {
        let json = DENSE_CONFIG_JSON.replace(
            r#""layer_types": [
                "linear_attention", "linear_attention", "linear_attention", "full_attention"
            ],"#,
            r#""layer_types": ["linear_attention"],"#,
        );
        let e = env(&json);
        assert!(e.text_config.validate().is_err());
    }
}
