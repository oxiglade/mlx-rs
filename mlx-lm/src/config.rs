//! Typed `config.json` schema. Parsed once at load via
//! [`ModelConfig::from_dir`]; the `model_type` field drives the
//! [`Family`] discriminant through serde's internally-tagged enum, so
//! there is no second parse and no stringly-typed dispatch.

use std::path::Path;

use serde::Deserialize;

use crate::error::Error;
use crate::gemma4::text::config as gemma4;
use crate::gemma4_unified::config as gemma4_unified;
use crate::llama::text::config as llama;
use crate::quantization::QuantizationConfig;
use crate::qwen3::text::config as qwen3;
use crate::qwen3_5::text::config as qwen35;

/// Parsed `config.json` for any supported family.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    /// Family-tagged config body, dispatched on `model_type` by serde.
    #[serde(flatten)]
    pub family: Family,

    /// Some mlx-community checkpoints emit both `quantization` and a
    /// sibling `quantization_config`; the legacy field is folded into
    /// `quantization` by [`Self::from_dir`] when the primary is absent.
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
    #[serde(default, rename = "quantization_config")]
    quantization_legacy: Option<QuantizationConfig>,
}

impl ModelConfig {
    /// Parse `<dir>/config.json` once into the typed schema.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, Error> {
        let path = dir.as_ref().join("config.json");
        let raw = std::fs::read_to_string(&path)?;
        let mut cfg: Self = serde_json::from_str(&raw)?;
        if cfg.quantization.is_none() {
            cfg.quantization = cfg.quantization_legacy.take();
        }
        Ok(cfg)
    }

    /// Effective quantisation settings (modern field, else legacy).
    pub fn quantization(&self) -> Option<&QuantizationConfig> {
        self.quantization
            .as_ref()
            .or(self.quantization_legacy.as_ref())
    }
}

/// Per-family config body. `tag = "model_type"` reads the discriminant
/// at deserialize; an unknown value fails with `unknown variant`
/// rather than falling through a loader.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "model_type")]
pub enum Family {
    #[serde(rename = "llama")]
    Llama(llama::ModelArgs),
    #[serde(rename = "qwen3")]
    Qwen3(qwen3::ModelArgs),
    #[serde(
        rename = "qwen3_5",
        alias = "qwen3_5_text",
        alias = "qwen3_5forconditionalgeneration"
    )]
    Qwen35(qwen35::ModelConfig),
    #[serde(
        rename = "qwen3_5_moe",
        alias = "qwen3_5_moe_text",
        alias = "qwen3_next"
    )]
    Qwen35Moe(qwen35::ModelConfig),
    #[serde(
        rename = "qwen3_5_vl",
        alias = "qwen3_5_vl_text",
        alias = "qwen3_5vlforconditionalgeneration"
    )]
    Qwen35Vl(qwen35::ModelConfig),
    #[serde(
        rename = "gemma4",
        alias = "gemma4_text",
        alias = "gemma4textmodel",
        alias = "gemma4forcausallm"
    )]
    Gemma4(gemma4::ModelConfig),
    #[serde(
        rename = "gemma4_unified",
        alias = "gemma4_unified_text",
        alias = "gemma4unifiedforconditionalgeneration"
    )]
    Gemma4Unified(gemma4_unified::ModelConfig),
}

impl Family {
    /// Canonical `model_type` string.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Llama(_) => "llama",
            Self::Qwen3(_) => "qwen3",
            Self::Qwen35(_) => "qwen3_5",
            Self::Qwen35Moe(_) => "qwen3_5_moe",
            Self::Qwen35Vl(_) => "qwen3_5_vl",
            Self::Gemma4(_) => "gemma4",
            Self::Gemma4Unified(_) => "gemma4_unified",
        }
    }

    pub fn as_llama(&self) -> Option<&llama::ModelArgs> {
        match self {
            Self::Llama(args) => Some(args),
            _ => None,
        }
    }

    pub fn as_qwen3(&self) -> Option<&qwen3::ModelArgs> {
        match self {
            Self::Qwen3(args) => Some(args),
            _ => None,
        }
    }

    pub fn as_qwen35(&self) -> Option<&qwen35::ModelConfig> {
        match self {
            Self::Qwen35(env) | Self::Qwen35Moe(env) | Self::Qwen35Vl(env) => Some(env),
            _ => None,
        }
    }

    pub fn as_gemma4(&self) -> Option<&gemma4::ModelConfig> {
        match self {
            Self::Gemma4(env) => Some(env),
            _ => None,
        }
    }

    pub fn as_gemma4_unified(&self) -> Option<&gemma4_unified::ModelConfig> {
        match self {
            Self::Gemma4Unified(env) => Some(env),
            _ => None,
        }
    }
}
