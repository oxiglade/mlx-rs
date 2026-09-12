pub use crate::error::ConfigError;
use std::{collections::BTreeMap, num::NonZeroUsize};

mod parse;
#[cfg(test)]
mod tests;

/// Resolved architecture facts validated before model allocation.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Architecture identifier.
    pub model_type: ModelType,
    /// Resolved transformer sizes and normalization epsilon.
    pub dimensions: TransformerDimensions,
    /// Resolved rotary embedding settings.
    pub rope: RopeConfig,
    /// Attention policy in decoder-layer order.
    pub attention: Vec<AttentionKind>,
    /// Whether output projection shares input embeddings.
    pub tie_word_embeddings: bool,
    /// Whether attention projections contain bias slots.
    pub attention_bias: bool,
    /// Whether MLP projections contain bias slots.
    pub mlp_bias: bool,
    /// Optional packed affine weight layout.
    pub quantization: Option<QuantizationConfig>,
}

/// An architecture identifier independent of the built-in registry.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelType(Box<str>);

/// Resolved transformer sizes in host-sized units.
#[derive(Debug, Clone, PartialEq)]
pub struct TransformerDimensions {
    /// Width of each hidden token representation.
    pub hidden_size: usize,
    /// Number of transformer layers.
    pub layer_count: usize,
    /// Width of the MLP intermediate representation.
    pub intermediate_size: usize,
    /// Number of query heads.
    pub attention_heads: usize,
    /// Number of key and value heads.
    pub kv_heads: usize,
    /// Width of each attention head.
    pub head_dim: usize,
    /// Number of token logits.
    pub vocabulary_size: usize,
    /// Optional model position limit.
    pub max_positions: Option<usize>,
    /// Stabilizing epsilon used by RMS normalization.
    pub rms_norm_epsilon: f32,
}

/// The attention visibility policy for one layer.
#[derive(Debug, Clone, PartialEq)]
pub enum AttentionKind {
    /// Attends to every previous token.
    Full,
    /// Attends within a bounded recent window.
    Sliding {
        /// Number of visible recent token positions.
        window: NonZeroUsize,
    },
}

/// Resolved rotary embedding dimensions and scaling.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeConfig {
    /// Number of rotary dimensions.
    pub dimensions: usize,
    /// Positive rotary frequency base.
    pub theta: f32,
    /// Whether rotary pairs use traditional adjacent layout.
    pub traditional: bool,
    /// Resolved position scaling scheme.
    pub scaling: RopeScaling,
}

/// Admitted rotary position scaling schemes.
#[derive(Debug, Clone, PartialEq)]
pub enum RopeScaling {
    /// Uses unscaled rotary positions.
    None,
    /// Scales positions by a fixed factor.
    Linear {
        /// Positive position scaling factor.
        factor: f32,
    },
    /// Uses frequency-dependent Llama 3 scaling.
    Llama3 {
        /// Positive position scaling factor.
        factor: f32,
        /// Low-frequency wavelength cutoff factor.
        low_frequency_factor: f32,
        /// High-frequency wavelength cutoff factor.
        high_frequency_factor: f32,
        /// Original context length used to derive wavelength cutoffs.
        original_max_positions: usize,
    },
}

/// Default affine settings and keyed per-layer overrides.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationConfig {
    /// Default affine settings for quantized parameter groups.
    pub default: AffineQuantization,
    /// Overrides indexed by stable parameter paths.
    pub layers: std::collections::BTreeMap<ParameterPath, LayerQuantization>,
}

/// The resolved quantization choice for one parameter group.
#[derive(Debug, Clone, PartialEq)]
pub enum LayerQuantization {
    /// Keeps the parameter group in its original floating dtype.
    Unquantized,
    /// Stores packed affine weight, scale, and bias tensors.
    Affine(AffineQuantization),
}

/// Packed affine quantization parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct AffineQuantization {
    /// Number of values sharing affine scale and bias.
    pub group_size: NonZeroUsize,
    /// Packed bits per quantized value.
    pub bits: u8,
}

/// A stable dotted parameter key independent of checkpoint order.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParameterPath(Box<str>);

impl ParameterPath {
    /// Creates a key for a named parameter slot.
    pub fn new(path: impl Into<Box<str>>) -> Self {
        Self(path.into())
    }
    /// Borrows the dotted key.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl ModelType {
    /// Creates an architecture identifier.
    pub fn new(name: impl Into<Box<str>>) -> Self {
        Self(name.into())
    }
    /// Borrows the architecture identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
#[derive(Debug, serde::Deserialize)]
pub(crate) struct RawConfig {
    pub(crate) model_type: String,
    #[serde(flatten)]
    pub(crate) fields: BTreeMap<String, serde_json::Value>,
}
impl RawConfig {
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, ConfigError> {
        let value: serde_json::Value = serde_json::from_slice(bytes)?;
        if value
            .as_object()
            .is_some_and(|fields| !fields.contains_key("model_type"))
        {
            return Err(ConfigError::MissingField {
                field: "model_type".into(),
            });
        }
        Ok(serde_json::from_value(value)?)
    }
}
impl Config {
    /// Validates resolved dimensions and supported architecture options before allocation.
    pub fn validate(&self) -> Result<(), ConfigError> {
        crate::arch::factory(self.model_type.as_str())?;
        let d = &self.dimensions;
        for (field, value) in [
            ("hidden_size", d.hidden_size),
            ("num_hidden_layers", d.layer_count),
            ("intermediate_size", d.intermediate_size),
            ("num_attention_heads", d.attention_heads),
            ("num_key_value_heads", d.kv_heads),
            ("head_dim", d.head_dim),
            ("vocab_size", d.vocabulary_size),
        ] {
            positive_size(field, value)?;
        }
        if !d.hidden_size.is_multiple_of(d.attention_heads) {
            return Err(invalid(
                "hidden_size",
                "must be divisible by attention heads",
            ));
        }
        if !d.attention_heads.is_multiple_of(d.kv_heads) {
            return Err(invalid(
                "num_key_value_heads",
                "must divide attention heads",
            ));
        }
        if let Some(positions) = d.max_positions {
            positive_size("max_position_embeddings", positions)?;
        }
        positive_number("rms_norm_eps", d.rms_norm_epsilon)?;
        if self.attention.len() != d.layer_count {
            return Err(ConfigError::InvalidLayerPattern {
                expected: d.layer_count,
                actual: self.attention.len(),
            });
        }
        positive_number("rope_theta", self.rope.theta)?;
        if self.rope.dimensions == 0
            || self.rope.dimensions > d.head_dim
            || !self.rope.dimensions.is_multiple_of(2)
        {
            return Err(invalid(
                "rope.dimensions",
                "must be positive, even, and at most head_dim",
            ));
        }
        match self.rope.scaling {
            RopeScaling::None => {}
            RopeScaling::Linear { factor } => positive_number("rope_scaling.factor", factor)?,
            RopeScaling::Llama3 {
                factor,
                low_frequency_factor,
                high_frequency_factor,
                original_max_positions,
            } => {
                positive_number("rope_scaling.factor", factor)?;
                positive_number("rope_scaling.low_freq_factor", low_frequency_factor)?;
                positive_number("rope_scaling.high_freq_factor", high_frequency_factor)?;
                positive_size(
                    "rope_scaling.original_max_position_embeddings",
                    original_max_positions,
                )?;
                if high_frequency_factor <= low_frequency_factor {
                    return Err(invalid(
                        "rope_scaling.high_freq_factor",
                        "must exceed low_freq_factor",
                    ));
                }
            }
        }
        if let Some(quantization) = &self.quantization {
            validate_affine(&quantization.default)?;
            for (path, layer) in &quantization.layers {
                validate_parameter_path(path.as_str())?;
                if let LayerQuantization::Affine(affine) = layer {
                    validate_affine(affine)?;
                }
            }
        }
        Ok(())
    }
}

fn invalid(field: &str, reason: &str) -> ConfigError {
    ConfigError::InvalidNumericField {
        field: field.into(),
        reason: reason.into(),
    }
}

fn positive_size(field: &str, value: usize) -> Result<(), ConfigError> {
    if value == 0 {
        return Err(invalid(field, "must be positive"));
    }
    Ok(())
}

fn positive_number(field: &str, value: f32) -> Result<(), ConfigError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid(field, "must be finite and positive"));
    }
    Ok(())
}

fn validate_affine(affine: &AffineQuantization) -> Result<(), ConfigError> {
    if !matches!(affine.bits, 2 | 4 | 8) || !matches!(affine.group_size.get(), 32 | 64 | 128) {
        return Err(ConfigError::UnsupportedQuantization(format!(
            "affine bits {}, group_size {}",
            affine.bits, affine.group_size
        )));
    }
    Ok(())
}

fn validate_parameter_path(path: &str) -> Result<(), ConfigError> {
    if path.split('.').any(str::is_empty) {
        return Err(ConfigError::UnsupportedQuantization(format!(
            "invalid parameter path {path:?}"
        )));
    }
    Ok(())
}
