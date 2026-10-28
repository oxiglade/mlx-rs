use std::{collections::BTreeMap, num::NonZeroUsize};

use serde::Deserialize;
use serde_json::Value;

use crate::{
    config::RawConfig, AffineQuantization, AttentionKind, Config, ConfigError, LayerQuantization,
    ModelType, ParameterPath, QuantizationConfig, RopeConfig, RopeScaling, TransformerDimensions,
};

#[derive(Deserialize)]
struct WireConfig {
    hidden_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    intermediate_size: Option<usize>,
    num_attention_heads: Option<usize>,
    rms_norm_eps: Option<f32>,
    vocab_size: Option<usize>,
    head_dim: Option<usize>,
    max_position_embeddings: Option<usize>,
    num_key_value_heads: Option<usize>,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    rope_theta: Option<f32>,
    #[serde(default)]
    rope_traditional: bool,
    rope_scaling: Option<Value>,
    tie_word_embeddings: Option<bool>,
    layer_types: Option<Vec<String>>,
    sliding_window: Option<usize>,
    quantization: Option<Value>,
    quantization_config: Option<Value>,
}

pub(crate) struct ParsedConfig {
    pub(crate) config: Config,
}

fn invalid(field: &str, reason: &str) -> ConfigError {
    ConfigError::InvalidNumericField {
        field: field.to_owned(),
        reason: reason.to_owned(),
    }
}

fn required<T>(value: Option<T>, field: &str) -> Result<T, ConfigError> {
    value.ok_or_else(|| ConfigError::MissingField {
        field: field.into(),
    })
}

pub(super) fn dimension(value: usize, field: &str) -> Result<i32, ConfigError> {
    if value == 0 {
        return Err(invalid(field, "must be positive"));
    }
    i32::try_from(value).map_err(|_| invalid(field, "exceeds MLX dimension range"))
}

fn positive(value: f32, field: &str) -> Result<f32, ConfigError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid(field, "must be finite and positive"));
    }
    Ok(value)
}

pub(super) fn parse(raw: &RawConfig) -> Result<ParsedConfig, ConfigError> {
    if raw.model_type != "llama" {
        return Err(ConfigError::UnsupportedArchitecture(raw.model_type.clone()));
    }
    let wire: WireConfig = serde_json::from_value(serde_json::to_value(&raw.fields)?)?;
    let hidden_size = required(wire.hidden_size, "hidden_size")?;
    let attention_heads = required(wire.num_attention_heads, "num_attention_heads")?;
    dimension(hidden_size, "hidden_size")?;
    dimension(attention_heads, "num_attention_heads")?;
    let head_dim = match wire.head_dim {
        Some(value) => value,
        None if hidden_size % attention_heads == 0 => hidden_size / attention_heads,
        None => {
            return Err(invalid(
                "hidden_size",
                "must divide evenly into attention heads",
            ))
        }
    };
    let kv_heads = wire.num_key_value_heads.unwrap_or(attention_heads);
    dimension(kv_heads, "num_key_value_heads")?;
    dimension(head_dim, "head_dim")?;
    if attention_heads % kv_heads != 0 {
        return Err(invalid(
            "num_key_value_heads",
            "must divide attention heads",
        ));
    }
    if head_dim % 2 != 0 {
        return Err(invalid("head_dim", "RoPE requires an even dimension"));
    }
    for (heads, field) in [(attention_heads, "query width"), (kv_heads, "KV width")] {
        dimension(
            heads
                .checked_mul(head_dim)
                .ok_or_else(|| invalid(field, "overflow"))?,
            field,
        )?;
    }
    let dimensions = TransformerDimensions {
        hidden_size,
        attention_heads,
        head_dim,
        kv_heads,
        layer_count: required(wire.num_hidden_layers, "num_hidden_layers")?,
        intermediate_size: required(wire.intermediate_size, "intermediate_size")?,
        vocabulary_size: required(wire.vocab_size, "vocab_size")?,
        max_positions: wire.max_position_embeddings,
        rms_norm_epsilon: positive(required(wire.rms_norm_eps, "rms_norm_eps")?, "rms_norm_eps")?,
    };
    for (value, field) in [
        (dimensions.layer_count, "num_hidden_layers"),
        (dimensions.intermediate_size, "intermediate_size"),
        (dimensions.vocabulary_size, "vocab_size"),
    ] {
        dimension(value, field)?;
    }
    if let Some(value) = dimensions.max_positions {
        dimension(value, "max_position_embeddings")?;
    }
    if let Some(value) = wire.sliding_window {
        dimension(value, "sliding_window")?;
    }
    let attention = match wire.layer_types {
        None => vec![AttentionKind::Full; dimensions.layer_count],
        Some(types) => {
            if types.len() != dimensions.layer_count {
                return Err(ConfigError::InvalidLayerPattern {
                    expected: dimensions.layer_count,
                    actual: types.len(),
                });
            }
            types
                .into_iter()
                .map(|kind| match kind.as_str() {
                    "full_attention" => Ok(AttentionKind::Full),
                    "sliding_attention" => {
                        let window = required(wire.sliding_window, "sliding_window")?;
                        Ok(AttentionKind::Sliding {
                            window: NonZeroUsize::new(window)
                                .ok_or_else(|| invalid("sliding_window", "must be positive"))?,
                        })
                    }
                    _ => Err(ConfigError::UnsupportedArchitecture(format!(
                        "llama attention type {kind}"
                    ))),
                })
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    if wire.quantization.is_none() {
        if let Some(legacy) = &wire.quantization_config {
            let method = legacy
                .get("quant_method")
                .and_then(Value::as_str)
                .unwrap_or("missing quant_method");
            return Err(ConfigError::UnsupportedQuantization(format!(
                "legacy quantization_config: {method}"
            )));
        }
    }
    let config = Config {
        model_type: ModelType::new("llama"),
        rope: RopeConfig {
            dimensions: head_dim,
            theta: positive(wire.rope_theta.unwrap_or(10000.0), "rope_theta")?,
            traditional: wire.rope_traditional,
            scaling: parse_rope(wire.rope_scaling.as_ref())?,
        },
        dimensions,
        attention,
        tie_word_embeddings: wire.tie_word_embeddings.unwrap_or(true),
        attention_bias: wire.attention_bias,
        mlp_bias: wire.mlp_bias,
        quantization: wire
            .quantization
            .as_ref()
            .map(parse_quantization)
            .transpose()?,
    };
    if let Some(quantization) = &config.quantization {
        let layout = matrix_layout(&config);
        for (key, setting) in &quantization.layers {
            let (input, _, _) = layout.get(key.as_str()).ok_or_else(|| {
                ConfigError::UnsupportedQuantization(format!("unknown layer {}", key.as_str()))
            })?;
            if let LayerQuantization::Affine(options) = setting {
                if !input.is_multiple_of(options.group_size.get()) {
                    return Err(ConfigError::UnsupportedQuantization(format!(
                        "{} input width is not divisible by group_size",
                        key.as_str()
                    )));
                }
            }
        }
    }
    super::rope::frequencies(&config.rope)?;
    Ok(ParsedConfig { config })
}

fn rope_number(value: &Value, field: &str, default: Option<f32>) -> Result<f32, ConfigError> {
    let number = match value.get(field) {
        Some(value) => value
            .as_f64()
            .map(|v| v as f32)
            .ok_or_else(|| invalid(field, "must be numeric"))?,
        None => required(default, field)?,
    };
    positive(number, field)
}

fn parse_rope(value: Option<&Value>) -> Result<RopeScaling, ConfigError> {
    let Some(value) = value else {
        return Ok(RopeScaling::None);
    };
    if !value.is_object() {
        return Err(ConfigError::UnsupportedRope(
            "rope_scaling must be an object".into(),
        ));
    }
    let kind = value.get("type").or_else(|| value.get("rope_type"));
    let kind = match kind {
        None => "default",
        Some(kind) => kind
            .as_str()
            .ok_or_else(|| ConfigError::UnsupportedRope("invalid rope type".into()))?,
    };
    match kind {
        "default" => Ok(RopeScaling::None),
        "linear" => {
            let factor = rope_number(value, "factor", None)?;
            positive(factor.recip(), "inverse rope factor")?;
            Ok(RopeScaling::Linear { factor })
        }
        "llama3" => {
            let factor = rope_number(value, "factor", None)?;
            let low_frequency_factor = rope_number(value, "low_freq_factor", Some(1.0))?;
            let high_frequency_factor = rope_number(value, "high_freq_factor", Some(4.0))?;
            if high_frequency_factor <= low_frequency_factor {
                return Err(invalid("high_freq_factor", "must exceed low_freq_factor"));
            }
            let original_max_positions = match value.get("original_max_position_embeddings") {
                Some(value) => serde_json::from_value(value.clone())?,
                None => 8192,
            };
            dimension(original_max_positions, "original_max_position_embeddings")?;
            Ok(RopeScaling::Llama3 {
                factor,
                low_frequency_factor,
                high_frequency_factor,
                original_max_positions,
            })
        }
        _ => Err(ConfigError::UnsupportedRope(kind.into())),
    }
}

fn affine(value: &Value) -> Result<AffineQuantization, ConfigError> {
    let bad = || ConfigError::UnsupportedQuantization(value.to_string());
    if value
        .get("mode")
        .is_some_and(|mode| mode.as_str() != Some("affine"))
    {
        return Err(bad());
    }
    let bits = value.get("bits").and_then(Value::as_u64).ok_or_else(bad)?;
    let group_size = value
        .get("group_size")
        .and_then(Value::as_u64)
        .ok_or_else(bad)?;
    if !matches!(bits, 2 | 3 | 4 | 6 | 8) || !matches!(group_size, 32 | 64 | 128) {
        return Err(bad());
    }
    Ok(AffineQuantization {
        bits: u8::try_from(bits).map_err(|_| bad())?,
        group_size: NonZeroUsize::new(usize::try_from(group_size).map_err(|_| bad())?)
            .ok_or_else(bad)?,
    })
}

fn parse_quantization(value: &Value) -> Result<QuantizationConfig, ConfigError> {
    let default = affine(value)?;
    let mut layers = BTreeMap::new();
    let entries = value
        .as_object()
        .ok_or_else(|| ConfigError::UnsupportedQuantization(value.to_string()))?;
    for (key, value) in entries {
        if matches!(key.as_str(), "bits" | "group_size" | "mode") {
            continue;
        }
        if !super::is_matrix_key(&format!("{key}.weight")) {
            return Err(ConfigError::UnsupportedQuantization(format!(
                "unknown layer {key}"
            )));
        }
        let setting = if value == &Value::Bool(false) {
            LayerQuantization::Unquantized
        } else {
            LayerQuantization::Affine(affine(value)?)
        };
        layers.insert(ParameterPath::new(key.as_str()), setting);
    }
    Ok(QuantizationConfig { default, layers })
}

pub(super) fn matrix_layout(config: &Config) -> BTreeMap<String, (usize, usize, bool)> {
    let dims = &config.dimensions;
    let mut layout = BTreeMap::new();
    layout.insert(
        "model.embed_tokens".into(),
        (dims.hidden_size, dims.vocabulary_size, false),
    );
    if !config.tie_word_embeddings {
        layout.insert(
            "lm_head".into(),
            (dims.hidden_size, dims.vocabulary_size, false),
        );
    }
    for layer in 0..dims.layer_count {
        let prefix = format!("model.layers.{layer}");
        let query_width = dims.attention_heads * dims.head_dim;
        let kv_width = dims.kv_heads * dims.head_dim;
        for (name, input, output) in [
            ("q_proj", dims.hidden_size, query_width),
            ("k_proj", dims.hidden_size, kv_width),
            ("v_proj", dims.hidden_size, kv_width),
            ("o_proj", query_width, dims.hidden_size),
        ] {
            layout.insert(
                format!("{prefix}.self_attn.{name}"),
                (input, output, config.attention_bias),
            );
        }
        for (name, input, output) in [
            ("gate_proj", dims.hidden_size, dims.intermediate_size),
            ("up_proj", dims.hidden_size, dims.intermediate_size),
            ("down_proj", dims.intermediate_size, dims.hidden_size),
        ] {
            layout.insert(
                format!("{prefix}.mlp.{name}"),
                (input, output, config.mlp_bias),
            );
        }
    }
    layout
}
