use std::{collections::BTreeMap, num::NonZeroUsize};

use serde::Deserialize;
use serde_json::Value;

use crate::{
    config::RawConfig, AffineQuantization, AttentionKind, Config, ConfigError, LayerQuantization,
    ModelType, ParameterPath, QuantizationConfig, RopeConfig, RopeScaling, TransformerDimensions,
};

#[derive(Deserialize)]
struct WireConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    vocab_size: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default)]
    max_position_embeddings: Option<usize>,
    #[serde(default = "epsilon")]
    rms_norm_eps: f32,
    #[serde(default = "theta")]
    rope_theta: f32,
    #[serde(default)]
    rope_traditional: bool,
    #[serde(default)]
    rope_scaling: Option<Value>,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default)]
    sliding_window: Option<usize>,
    #[serde(default)]
    quantization: Option<Value>,
    #[serde(default)]
    quantization_config: Option<Value>,
    #[serde(default)]
    hidden_act: Option<String>,
    #[serde(default)]
    partial_rotary_factor: Option<f32>,
}

fn epsilon() -> f32 {
    1e-6
}
fn theta() -> f32 {
    1_000_000.0
}

pub(crate) struct ParsedConfig {
    pub(crate) config: Config,
}

pub(super) fn invalid(field: &str, reason: &str) -> ConfigError {
    ConfigError::InvalidNumericField {
        field: field.into(),
        reason: reason.into(),
    }
}

pub(super) fn dimension(field: &str, value: usize) -> Result<i32, ConfigError> {
    if value == 0 {
        return Err(invalid(field, "must be positive"));
    }
    i32::try_from(value).map_err(|_| invalid(field, "exceeds MLX dimension range"))
}

fn positive(field: &str, value: f32) -> Result<f32, ConfigError> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(invalid(field, "must be finite and positive"))
    }
}

pub(super) fn parse(raw: &RawConfig) -> Result<ParsedConfig, ConfigError> {
    if raw.model_type != "qwen3" {
        return Err(ConfigError::UnsupportedArchitecture(raw.model_type.clone()));
    }
    for field in [
        "hidden_size",
        "num_hidden_layers",
        "intermediate_size",
        "num_attention_heads",
        "vocab_size",
    ] {
        if !raw.fields.contains_key(field) {
            return Err(ConfigError::MissingField {
                field: field.into(),
            });
        }
    }
    let wire: WireConfig = serde_json::from_value(serde_json::to_value(&raw.fields)?)?;
    let heads = wire.num_attention_heads;
    dimension("num_attention_heads", heads)?;
    let kv_heads = wire.num_key_value_heads.unwrap_or(heads);
    dimension("num_key_value_heads", kv_heads)?;
    if !heads.is_multiple_of(kv_heads) {
        return Err(invalid(
            "num_key_value_heads",
            "must divide query head count",
        ));
    }
    if wire.head_dim.is_none() && !wire.hidden_size.is_multiple_of(heads) {
        return Err(invalid(
            "hidden_size",
            "must divide evenly into attention heads when head_dim is absent",
        ));
    }
    let head_dim = wire.head_dim.unwrap_or(wire.hidden_size / heads);
    for (field, value) in [
        ("hidden_size", wire.hidden_size),
        ("num_hidden_layers", wire.num_hidden_layers),
        ("intermediate_size", wire.intermediate_size),
        ("head_dim", head_dim),
        ("vocab_size", wire.vocab_size),
    ] {
        dimension(field, value)?;
    }
    dimension(
        "query projection width",
        heads
            .checked_mul(head_dim)
            .ok_or_else(|| invalid("head_dim", "projection width overflow"))?,
    )?;
    if let Some(max) = wire.max_position_embeddings {
        dimension("max_position_embeddings", max)?;
    }
    if !head_dim.is_multiple_of(2) {
        return Err(invalid("head_dim", "RoPE requires an even dimension"));
    }
    if wire.hidden_act.as_deref().is_some_and(|act| act != "silu") {
        return Err(ConfigError::UnsupportedArchitecture(format!(
            "qwen3 activation {:?}",
            wire.hidden_act
        )));
    }
    if wire
        .partial_rotary_factor
        .is_some_and(|factor| factor != 1.0)
    {
        return Err(ConfigError::UnsupportedRope(
            "qwen3 requires full head rotation".into(),
        ));
    }
    let rope = RopeConfig {
        dimensions: head_dim,
        theta: positive("rope_theta", wire.rope_theta)?,
        traditional: wire.rope_traditional,
        scaling: parse_rope(wire.rope_scaling.as_ref())?,
    };
    let kinds = wire
        .layer_types
        .unwrap_or_else(|| vec!["full_attention".into(); wire.num_hidden_layers]);
    if kinds.len() != wire.num_hidden_layers {
        return Err(ConfigError::InvalidLayerPattern {
            expected: wire.num_hidden_layers,
            actual: kinds.len(),
        });
    }
    let attention = kinds
        .iter()
        .map(|kind| match kind.as_str() {
            "full_attention" => Ok(AttentionKind::Full),
            "sliding_attention" => {
                let size = wire
                    .sliding_window
                    .ok_or_else(|| ConfigError::MissingField {
                        field: "sliding_window".into(),
                    })?;
                dimension("sliding_window", size)?;
                let window = NonZeroUsize::new(size)
                    .ok_or_else(|| invalid("sliding_window", "must be positive"))?;
                Ok(AttentionKind::Sliding { window })
            }
            _ => Err(ConfigError::UnsupportedArchitecture(format!(
                "qwen3 attention {kind}"
            ))),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if wire.quantization.is_none() {
        if let Some(legacy) = &wire.quantization_config {
            let method = legacy
                .get("quant_method")
                .and_then(Value::as_str)
                .unwrap_or("missing quant_method");
            return Err(ConfigError::UnsupportedQuantization(format!(
                "legacy quantization_config {method}"
            )));
        }
    }
    let quantization = wire
        .quantization
        .as_ref()
        .map(parse_quantization)
        .transpose()?;
    let config = Config {
        model_type: ModelType::new("qwen3"),
        dimensions: TransformerDimensions {
            hidden_size: wire.hidden_size,
            layer_count: wire.num_hidden_layers,
            intermediate_size: wire.intermediate_size,
            attention_heads: heads,
            kv_heads,
            head_dim,
            vocabulary_size: wire.vocab_size,
            max_positions: wire.max_position_embeddings,
            rms_norm_epsilon: positive("rms_norm_eps", wire.rms_norm_eps)?,
        },
        rope,
        attention,
        tie_word_embeddings: wire.tie_word_embeddings,
        attention_bias: wire.attention_bias,
        mlp_bias: wire.mlp_bias,
        quantization,
    };
    if let Some(quantization) = &config.quantization {
        for path in quantization.layers.keys() {
            if !super::weights::matrix_dimensions(&config, path.as_str())
                .is_some_and(|_| !(config.tie_word_embeddings && path.as_str() == "lm_head"))
            {
                return Err(ConfigError::UnsupportedQuantization(format!(
                    "unknown parameter group {}",
                    path.as_str()
                )));
            }
        }
    }
    Ok(ParsedConfig { config })
}

fn number(value: &Value, field: &str, default: Option<f32>) -> Result<f32, ConfigError> {
    let number = match value.get(field) {
        Some(value) => value
            .as_f64()
            .map(|value| value as f32)
            .ok_or_else(|| invalid(field, "must be numeric"))?,
        None => default.ok_or_else(|| ConfigError::MissingField {
            field: field.into(),
        })?,
    };
    positive(field, number)
}

fn parse_rope(value: Option<&Value>) -> Result<RopeScaling, ConfigError> {
    let Some(value) = value else {
        return Ok(RopeScaling::None);
    };
    if !value.is_object() {
        return Err(ConfigError::UnsupportedRope("expected an object".into()));
    }
    let kind = value
        .get("rope_type")
        .or_else(|| value.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("default");
    if let (Some(a), Some(b)) = (value.get("rope_type"), value.get("type")) {
        if a != b {
            return Err(ConfigError::UnsupportedRope(
                "conflicting rope_type and type".into(),
            ));
        }
    }
    match kind {
        "default" => Ok(RopeScaling::None),
        "linear" => Ok(RopeScaling::Linear {
            factor: number(value, "factor", None)?,
        }),
        "llama3" => {
            let factor = number(value, "factor", None)?;
            let low_frequency_factor = number(value, "low_freq_factor", Some(1.0))?;
            let high_frequency_factor = number(value, "high_freq_factor", Some(4.0))?;
            if high_frequency_factor <= low_frequency_factor {
                return Err(invalid("high_freq_factor", "must exceed low_freq_factor"));
            }
            let original_max_positions = match value.get("original_max_position_embeddings") {
                Some(value) => value
                    .as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| {
                        invalid(
                            "original_max_position_embeddings",
                            "must be a positive integer",
                        )
                    })?,
                None => 8192,
            };
            dimension("original_max_position_embeddings", original_max_positions)?;
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
    if value.get("mode").is_some_and(|mode| mode != "affine") {
        return Err(ConfigError::UnsupportedQuantization(
            "only affine mode is supported".into(),
        ));
    }
    let bits = value.get("bits").and_then(Value::as_u64).unwrap_or(0);
    let group_size = value.get("group_size").and_then(Value::as_u64).unwrap_or(0);
    if !matches!(bits, 2 | 4 | 8) || !matches!(group_size, 32 | 64 | 128) {
        return Err(ConfigError::UnsupportedQuantization(format!(
            "affine bits={bits}, group_size={group_size}"
        )));
    }
    Ok(AffineQuantization {
        bits: u8::try_from(bits)
            .map_err(|_| ConfigError::UnsupportedQuantization("bits overflow".into()))?,
        group_size: NonZeroUsize::new(
            usize::try_from(group_size)
                .map_err(|_| ConfigError::UnsupportedQuantization("group size overflow".into()))?,
        )
        .ok_or_else(|| ConfigError::UnsupportedQuantization("zero group size".into()))?,
    })
}

fn parse_quantization(value: &Value) -> Result<QuantizationConfig, ConfigError> {
    let fields = value
        .as_object()
        .ok_or_else(|| ConfigError::UnsupportedQuantization("expected an object".into()))?;
    let default = affine(value)?;
    let mut layers = BTreeMap::new();
    for (key, value) in fields {
        if matches!(key.as_str(), "bits" | "group_size" | "mode") {
            continue;
        }
        let path = key.strip_prefix("model.").unwrap_or(key);
        let choice = if value == &Value::Bool(false) {
            LayerQuantization::Unquantized
        } else if value == &Value::Bool(true) {
            LayerQuantization::Affine(default.clone())
        } else {
            LayerQuantization::Affine(affine(value)?)
        };
        if layers.insert(ParameterPath::new(path), choice).is_some() {
            return Err(ConfigError::UnsupportedQuantization(format!(
                "duplicate override {path}"
            )));
        }
    }
    Ok(QuantizationConfig { default, layers })
}
