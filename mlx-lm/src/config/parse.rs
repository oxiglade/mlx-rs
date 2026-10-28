use super::*;
use serde::de::DeserializeOwned;
use serde_json::Value;

impl RawConfig {
    pub(crate) fn resolve(&self) -> Result<Config, ConfigError> {
        crate::arch::factory(&self.model_type)?;
        let quantization = self
            .fields
            .get("quantization")
            .filter(|value| !value.is_null());
        if quantization.is_none() {
            if let Some(legacy) = self
                .fields
                .get("quantization_config")
                .filter(|value| !value.is_null())
            {
                let message = match legacy.get("quant_method").and_then(Value::as_str) {
                    Some(method) => format!("legacy quantization_config quant_method {method}"),
                    None => "legacy quantization_config".into(),
                };
                return Err(ConfigError::UnsupportedQuantization(message));
            }
        }
        let hidden_size = self.size("hidden_size")?;
        let attention_heads = self.size("num_attention_heads")?;
        positive_size("num_attention_heads", attention_heads)?;
        let head_dim = self
            .optional_size("head_dim")?
            .unwrap_or(hidden_size / attention_heads);
        let layer_count = self.size("num_hidden_layers")?;
        let config = Config {
            model_type: ModelType::new(self.model_type.as_str()),
            dimensions: TransformerDimensions {
                hidden_size,
                layer_count,
                intermediate_size: self.size("intermediate_size")?,
                attention_heads,
                kv_heads: self
                    .optional_size("num_key_value_heads")?
                    .unwrap_or(attention_heads),
                head_dim,
                vocabulary_size: self.size("vocab_size")?,
                max_positions: self.optional_size("max_position_embeddings")?,
                rms_norm_epsilon: number(self.fields.get("rms_norm_eps"), "rms_norm_eps")?,
            },
            rope: RopeConfig {
                dimensions: head_dim,
                theta: match self.fields.get("rope_theta") {
                    Some(value) => number(Some(value), "rope_theta")?,
                    None if self.model_type == "llama" => 10_000.0,
                    None => {
                        return Err(ConfigError::MissingField {
                            field: "rope_theta".into(),
                        })
                    }
                },
                traditional: self.optional("rope_traditional")?.unwrap_or(false),
                scaling: parse_rope(self.fields.get("rope_scaling"))?,
            },
            attention: self.attention(layer_count)?,
            tie_word_embeddings: self
                .optional("tie_word_embeddings")?
                .or_else(|| (self.model_type == "llama").then_some(true))
                .ok_or_else(|| ConfigError::MissingField {
                    field: "tie_word_embeddings".into(),
                })?,
            attention_bias: self.optional("attention_bias")?.unwrap_or(false),
            mlp_bias: self.optional("mlp_bias")?.unwrap_or(false),
            quantization: quantization.map(parse_quantization).transpose()?,
        };
        config.validate()?;
        Ok(config)
    }

    fn size(&self, field: &str) -> Result<usize, ConfigError> {
        size(self.fields.get(field), field)
    }

    fn optional_size(&self, field: &str) -> Result<Option<usize>, ConfigError> {
        self.fields
            .get(field)
            .filter(|value| !value.is_null())
            .map(|value| size(Some(value), field))
            .transpose()
    }

    fn optional<T: DeserializeOwned>(&self, field: &str) -> Result<Option<T>, ConfigError> {
        self.fields
            .get(field)
            .filter(|value| !value.is_null())
            .map(|value| serde_json::from_value(value.clone()).map_err(ConfigError::from))
            .transpose()
    }

    fn attention(&self, layer_count: usize) -> Result<Vec<AttentionKind>, ConfigError> {
        let window = self
            .optional_size("sliding_window")?
            .map(|window| {
                NonZeroUsize::new(window)
                    .ok_or_else(|| invalid("sliding_window", "must be positive"))
            })
            .transpose()?;
        let sliding = || {
            window
                .map(|window| AttentionKind::Sliding { window })
                .ok_or_else(|| ConfigError::MissingField {
                    field: "sliding_window".into(),
                })
        };
        match self.optional::<Vec<String>>("layer_types")? {
            Some(layers) => {
                if layers.len() != layer_count {
                    return Err(ConfigError::InvalidLayerPattern {
                        expected: layer_count,
                        actual: layers.len(),
                    });
                }
                layers
                    .iter()
                    .map(|kind| match kind.as_str() {
                        "full_attention" => Ok(AttentionKind::Full),
                        "sliding_attention" => sliding(),
                        _ => Err(ConfigError::UnsupportedArchitecture(format!(
                            "{} attention kind {kind}",
                            self.model_type
                        ))),
                    })
                    .collect()
            }
            None => {
                let mut layers = Vec::new();
                layers.try_reserve_exact(layer_count).map_err(|_| {
                    invalid(
                        "num_hidden_layers",
                        "attention pattern allocation exceeds available capacity",
                    )
                })?;
                layers.resize(layer_count, AttentionKind::Full);
                Ok(layers)
            }
        }
    }
}

fn required<'a>(value: Option<&'a Value>, field: &str) -> Result<&'a Value, ConfigError> {
    value.ok_or_else(|| ConfigError::MissingField {
        field: field.into(),
    })
}

fn size(value: Option<&Value>, field: &str) -> Result<usize, ConfigError> {
    required(value, field)?
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| invalid(field, "must be a nonnegative host-sized integer"))
}

fn number(value: Option<&Value>, field: &str) -> Result<f32, ConfigError> {
    let value = required(value, field)?
        .as_f64()
        .ok_or_else(|| invalid(field, "must be a finite positive number"))? as f32;
    positive_number(field, value)?;
    Ok(value)
}

fn parse_rope(value: Option<&Value>) -> Result<RopeScaling, ConfigError> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(RopeScaling::None);
    };
    let default = Value::String("default".into());
    let kind = value
        .get("type")
        .filter(|kind| !kind.is_null() && kind.as_str() != Some(""))
        .or_else(|| value.get("rope_type"))
        .unwrap_or(&default);
    match kind.as_str() {
        Some("default") => Ok(RopeScaling::None),
        Some("linear") => Ok(RopeScaling::Linear {
            factor: number(value.get("factor"), "rope_scaling.factor")?,
        }),
        Some("llama3") => Ok(RopeScaling::Llama3 {
            factor: number(value.get("factor"), "rope_scaling.factor")?,
            low_frequency_factor: number(
                value.get("low_freq_factor"),
                "rope_scaling.low_freq_factor",
            )?,
            high_frequency_factor: number(
                value.get("high_freq_factor"),
                "rope_scaling.high_freq_factor",
            )?,
            original_max_positions: size(
                value.get("original_max_position_embeddings"),
                "rope_scaling.original_max_position_embeddings",
            )?,
        }),
        _ => Err(ConfigError::UnsupportedRope(kind.to_string())),
    }
}

fn parse_affine(value: &Value) -> Result<AffineQuantization, ConfigError> {
    if let Some(mode) = value.get("mode") {
        if mode.as_str() != Some("affine") {
            return Err(ConfigError::UnsupportedQuantization(mode.to_string()));
        }
    }
    let group_size = size(value.get("group_size"), "quantization.group_size")?;
    let group_size = NonZeroUsize::new(group_size)
        .ok_or_else(|| invalid("quantization.group_size", "must be positive"))?;
    let bits = size(value.get("bits"), "quantization.bits")?;
    let bits = u8::try_from(bits)
        .map_err(|_| ConfigError::UnsupportedQuantization(format!("bits {bits}")))?;
    let affine = AffineQuantization { group_size, bits };
    validate_affine(&affine)?;
    Ok(affine)
}

fn parse_quantization(value: &Value) -> Result<QuantizationConfig, ConfigError> {
    let fields = value
        .as_object()
        .ok_or_else(|| ConfigError::UnsupportedQuantization(value.to_string()))?;
    let default = parse_affine(value)?;
    let mut layers = BTreeMap::new();
    for (key, value) in fields {
        if matches!(key.as_str(), "group_size" | "bits" | "mode") {
            continue;
        }
        validate_parameter_path(key)?;
        let layer = match value {
            Value::Bool(false) => LayerQuantization::Unquantized,
            Value::Bool(true) => LayerQuantization::Affine(default.clone()),
            Value::Object(fields) => {
                if fields
                    .keys()
                    .any(|field| !matches!(field.as_str(), "group_size" | "bits" | "mode"))
                {
                    return Err(ConfigError::UnsupportedQuantization(format!(
                        "unsupported affine fields for {key}"
                    )));
                }
                LayerQuantization::Affine(parse_affine(value)?)
            }
            _ => {
                return Err(ConfigError::UnsupportedQuantization(format!(
                    "invalid override for {key}"
                )))
            }
        };
        layers.insert(ParameterPath::new(key.as_str()), layer);
    }
    Ok(QuantizationConfig { default, layers })
}
