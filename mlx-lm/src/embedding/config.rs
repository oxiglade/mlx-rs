use crate::ConfigError;
use serde::Deserialize;
use std::collections::BTreeMap;

/// The only `config.json` model type the embedder admits.
pub(super) const MODEL_TYPE: &str = "embedding_gemma2";

/// Validated text-encoder facts; every dimension fits MLX's `i32` shapes.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct EncoderConfig {
    pub(super) vocabulary_size: usize,
    pub(super) hidden_size: usize,
    pub(super) intermediate_size: usize,
    pub(super) per_layer_width: usize,
    pub(super) embedding_dim: usize,
    pub(super) rms_norm_eps: f32,
    pub(super) sliding_window: usize,
    pub(super) max_positions: usize,
    pub(super) pad_token_id: u32,
    pub(super) media_tokens: Vec<u32>,
    pub(super) layers: Vec<LayerShape>,
}

/// One encoder layer's attention geometry after `per_layer_config` overrides.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct LayerShape {
    pub(super) heads: usize,
    pub(super) kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) sliding: bool,
    pub(super) rope_theta: f32,
}

#[derive(Deserialize)]
struct RawConfig {
    text_config: RawText,
    image_token_id: Option<u32>,
    audio_token_id: Option<u32>,
    video_token_id: Option<u32>,
}

#[derive(Deserialize)]
struct RawText {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    hidden_size_per_layer_input: usize,
    embedding_dim: usize,
    rms_norm_eps: f64,
    sliding_window: usize,
    max_position_embeddings: usize,
    attention_bias: bool,
    hidden_activation: String,
    pad_token_id: u32,
    layer_types: Vec<String>,
    per_layer_config: BTreeMap<String, Override>,
    rope_parameters: BTreeMap<String, RawRope>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Override {
    head_dim: Option<usize>,
    num_key_value_heads: Option<usize>,
    num_attention_heads: Option<usize>,
}

#[derive(Deserialize)]
struct RawRope {
    rope_theta: f64,
    rope_type: String,
}

fn invalid(field: impl Into<String>, reason: impl Into<String>) -> ConfigError {
    ConfigError::InvalidNumericField {
        field: field.into(),
        reason: reason.into(),
    }
}

fn positive(field: &str, value: usize) -> Result<usize, ConfigError> {
    if value == 0 || i32::try_from(value).is_err() {
        return Err(invalid(field, "must be positive and fit in i32"));
    }
    Ok(value)
}

pub(super) fn parse(bytes: &[u8]) -> Result<EncoderConfig, ConfigError> {
    let value: serde_json::Value = serde_json::from_slice(bytes)?;
    let model_type = value
        .get("model_type")
        .ok_or_else(|| ConfigError::MissingField {
            field: "model_type".into(),
        })?;
    if model_type != MODEL_TYPE {
        return Err(ConfigError::UnsupportedArchitecture(
            model_type
                .as_str()
                .unwrap_or("non-string model_type")
                .into(),
        ));
    }
    let raw: RawConfig = serde_json::from_value(value)?;
    let text = raw.text_config;
    if text.attention_bias {
        return Err(ConfigError::UnsupportedArchitecture(format!(
            "{MODEL_TYPE} with attention_bias"
        )));
    }
    if text.hidden_activation != "gelu_pytorch_tanh" {
        return Err(ConfigError::UnsupportedArchitecture(format!(
            "{MODEL_TYPE} with hidden_activation {}",
            text.hidden_activation
        )));
    }
    let vocabulary_size = positive("vocab_size", text.vocab_size)?;
    let hidden_size = positive("hidden_size", text.hidden_size)?;
    let intermediate_size = positive("intermediate_size", text.intermediate_size)?;
    let layer_count = positive("num_hidden_layers", text.num_hidden_layers)?;
    let per_layer_width = positive(
        "hidden_size_per_layer_input",
        text.hidden_size_per_layer_input,
    )?;
    let embedding_dim = positive("embedding_dim", text.embedding_dim)?;
    let sliding_window = positive("sliding_window", text.sliding_window)?;
    let max_positions = positive("max_position_embeddings", text.max_position_embeddings)?;
    // Packed per-layer inputs are one [hidden, layers * width] projection.
    layer_count
        .checked_mul(per_layer_width)
        .filter(|&n| i32::try_from(n).is_ok())
        .ok_or_else(|| invalid("hidden_size_per_layer_input", "layers * width exceeds i32"))?;
    let rms_norm_eps = text.rms_norm_eps as f32;
    if !rms_norm_eps.is_finite() || rms_norm_eps <= 0.0 {
        return Err(invalid("rms_norm_eps", "must be finite and positive"));
    }
    if text.pad_token_id as usize >= vocabulary_size {
        return Err(invalid("pad_token_id", "outside the vocabulary"));
    }
    if text.layer_types.len() != layer_count {
        return Err(ConfigError::InvalidLayerPattern {
            expected: layer_count,
            actual: text.layer_types.len(),
        });
    }
    // The reference encoder forces the final layer to full attention; reject rather than rewrite.
    if text.layer_types.last().map(String::as_str) != Some("full_attention") {
        return Err(invalid(
            "layer_types",
            "the final layer must be full_attention",
        ));
    }
    let mut overrides = BTreeMap::new();
    for (key, value) in text.per_layer_config {
        let index = key
            .parse::<usize>()
            .ok()
            .filter(|&index| index < layer_count)
            .ok_or_else(|| invalid("per_layer_config", format!("invalid layer index {key:?}")))?;
        if overrides.insert(index, value).is_some() {
            return Err(invalid(
                "per_layer_config",
                format!("duplicate layer index {key:?}"),
            ));
        }
    }
    let mut thetas = BTreeMap::new();
    for (kind, rope) in &text.rope_parameters {
        if rope.rope_type != "default" {
            return Err(ConfigError::UnsupportedRope(format!(
                "{kind}: {}",
                rope.rope_type
            )));
        }
        let theta = rope.rope_theta as f32;
        if !theta.is_finite() || theta <= 0.0 {
            return Err(invalid(
                format!("rope_parameters.{kind}.rope_theta"),
                "must be finite and positive",
            ));
        }
        thetas.insert(kind.as_str(), theta);
    }
    let mut layers = Vec::with_capacity(layer_count);
    for (index, kind) in text.layer_types.iter().enumerate() {
        let sliding = match kind.as_str() {
            "sliding_attention" => true,
            "full_attention" => false,
            other => {
                return Err(ConfigError::UnsupportedArchitecture(format!(
                    "{MODEL_TYPE} layer type {other}"
                )))
            }
        };
        let rope_theta = *thetas
            .get(kind.as_str())
            .ok_or_else(|| ConfigError::MissingField {
                field: format!("rope_parameters.{kind}"),
            })?;
        let layer = overrides.get(&index);
        let field = |name: &str| format!("per_layer_config.{index:02}.{name}");
        let heads = positive(
            &field("num_attention_heads"),
            layer
                .and_then(|o| o.num_attention_heads)
                .unwrap_or(text.num_attention_heads),
        )?;
        let kv_heads = positive(
            &field("num_key_value_heads"),
            layer
                .and_then(|o| o.num_key_value_heads)
                .unwrap_or(text.num_key_value_heads),
        )?;
        let head_dim = positive(
            &field("head_dim"),
            layer.and_then(|o| o.head_dim).unwrap_or(text.head_dim),
        )?;
        if heads % kv_heads != 0 {
            return Err(invalid(
                field("num_key_value_heads"),
                "must divide num_attention_heads",
            ));
        }
        if head_dim % 2 != 0 {
            return Err(invalid(field("head_dim"), "RoPE requires an even head_dim"));
        }
        heads
            .checked_mul(head_dim)
            .filter(|&n| i32::try_from(n).is_ok())
            .ok_or_else(|| invalid(field("head_dim"), "heads * head_dim exceeds i32"))?;
        layers.push(LayerShape {
            heads,
            kv_heads,
            head_dim,
            sliding,
            rope_theta,
        });
    }
    let media_tokens = [raw.image_token_id, raw.audio_token_id, raw.video_token_id]
        .into_iter()
        .flatten()
        .collect();
    Ok(EncoderConfig {
        vocabulary_size,
        hidden_size,
        intermediate_size,
        per_layer_width,
        embedding_dim,
        rms_norm_eps,
        sliding_window,
        max_positions,
        pad_token_id: text.pad_token_id,
        media_tokens,
        layers,
    })
}
