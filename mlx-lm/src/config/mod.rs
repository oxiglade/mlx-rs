pub use crate::error::ConfigError;
use std::{collections::BTreeMap, num::NonZeroUsize};

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
        Ok(serde_json::from_slice(bytes)?)
    }
}
impl Config {
    /// Validates resolved dimensions and supported architecture options before allocation.
    pub fn validate(&self) -> Result<(), ConfigError> {
        Err(ConfigError::UnsupportedArchitecture(
            crate::NotYetImplemented("config validation").to_string(),
        ))
    }
}
