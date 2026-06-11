//! `quantization_config` parsing for MLX checkpoints.
//!
//! Two checkpoint shapes:
//! - **Uniform:** `{group_size, bits, mode}` — every quantisable param
//!   uses the body settings.
//! - **Per-tensor overrides:** body settings plus path-keyed entries
//!   like `"…layers.0.mlp.gate": {group_size, bits}`. Qwen3.6-MoE
//!   ships the router + shared-expert gates at 8-bit even when the
//!   body is 4-bit; loaders consult [`QuantizationConfig::for_path`].

use std::collections::HashMap;

use mlx_rs::builder::Builder;
use mlx_rs::nn;
use mlx_rs::quantization::{MaybeQuantized, Quantizable};
use serde::Deserialize;

use crate::error::Error;

/// Bits packed per `uint32` inner-weight element: a quantised `Linear`'s
/// packed weight has shape `[out, in / (BITS_PER_U32 / bits)]`.
const BITS_PER_U32: i32 = 32;

/// Quantisation mode from `quantization.mode`. Every production
/// checkpoint ships `affine`; unknown values reject at deserialize.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantMode {
    #[default]
    Affine,
}

/// Body quantisation + optional per-key overrides.
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
    pub mode: QuantMode,
    /// Per-key overrides keyed by the raw safetensors prefix, e.g.
    /// `language_model.model.layers.0.mlp.gate`.
    pub overrides: HashMap<String, (i32, i32)>,
}

impl QuantizationConfig {
    /// `(group_size, bits)` for `path` — its override if any, else the
    /// body defaults.
    pub fn for_path(&self, path: &str) -> (i32, i32) {
        self.overrides
            .get(path)
            .copied()
            .unwrap_or((self.group_size, self.bits))
    }
}

/// Re-quantise one body-quantised linear slot at an override `(group_size,
/// bits)`. Rebuilds a fresh `Linear` at the original `[out, in]` dims
/// (recovered from the packed inner weight shape) then quantises at the
/// override; the caller overwrites the values afterward, so only the slot's
/// `(group_size, bits)` + shape contract matters here.
///
/// Used by loaders for mixed-quant checkpoints where some tensors ship at a
/// different bit width than the body (e.g. mlx-community 4-bit models keep
/// gate/MLP projections at 8-bit).
pub fn requantise_linear(
    slot: &mut MaybeQuantized<nn::Linear>,
    group_size: i32,
    bits: i32,
) -> Result<(), Error> {
    let dummy = nn::LinearBuilder::new(1, 1).bias(false).build()?;
    let linear = match std::mem::replace(slot, MaybeQuantized::Original(dummy)) {
        MaybeQuantized::Original(l) => l,
        MaybeQuantized::Quantized(q) => {
            let shape = q.inner.weight.as_ref().shape();
            let out_features = shape[0];
            let in_features = shape[1] * (BITS_PER_U32 / q.bits);
            nn::LinearBuilder::new(in_features, out_features)
                .bias(false)
                .build()?
        }
    };
    *slot = MaybeQuantized::Original(linear).try_into_quantized(group_size, bits)?;
    Ok(())
}

/// Body knobs deserialize directly; every other entry flows into
/// `extras` so per-key override objects fall out of the residual.
#[derive(Deserialize)]
struct Raw {
    group_size: i32,
    bits: i32,
    #[serde(default)]
    mode: QuantMode,
    #[serde(flatten)]
    extras: HashMap<String, serde_json::Value>,
}

#[derive(Deserialize)]
struct OverrideValue {
    group_size: i32,
    bits: i32,
}

impl<'de> Deserialize<'de> for QuantizationConfig {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let raw = Raw::deserialize(d)?;
        let mut overrides = HashMap::with_capacity(raw.extras.len());
        for (k, v) in raw.extras {
            // Non-object entries are advisory fields transformers may
            // emit — skip. An object is meant to be an override, so a
            // strict-parse failure is an error, not a silent body-bits
            // fallback.
            if v.is_object() {
                let ov: OverrideValue =
                    serde_json::from_value(v).map_err(serde::de::Error::custom)?;
                overrides.insert(k, (ov.group_size, ov.bits));
            }
        }
        Ok(Self {
            group_size: raw.group_size,
            bits: raw.bits,
            mode: raw.mode,
            overrides,
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    use super::*;

    #[test]
    fn parses_uniform_config() {
        let q: QuantizationConfig =
            serde_json::from_str(r#"{"group_size": 64, "bits": 8, "mode": "affine"}"#).unwrap();
        assert_eq!((q.group_size, q.bits), (64, 8));
        assert_eq!(q.mode, QuantMode::Affine);
        assert!(q.overrides.is_empty());
        assert_eq!(q.for_path("anything"), (64, 8));
    }

    #[test]
    fn parses_per_tensor_overrides() {
        let q: QuantizationConfig = serde_json::from_str(
            r#"{
                "group_size": 64,
                "bits": 4,
                "mode": "affine",
                "language_model.model.layers.0.mlp.gate": {"group_size": 64, "bits": 8}
            }"#,
        )
        .unwrap();
        assert_eq!(q.bits, 4);
        assert_eq!(q.overrides.len(), 1);
        assert_eq!(
            q.for_path("language_model.model.layers.0.mlp.gate"),
            (64, 8)
        );
        assert_eq!(
            q.for_path("language_model.model.layers.0.self_attn.q_proj"),
            (64, 4)
        );
    }

    #[test]
    fn default_mode_filled() {
        let q: QuantizationConfig =
            serde_json::from_str(r#"{"group_size": 32, "bits": 4}"#).unwrap();
        assert_eq!(q.mode, QuantMode::Affine);
    }

    #[test]
    fn malformed_override_object_errors() {
        // Override object missing `group_size` must fail, not silently
        // fall back to body bits.
        let r = serde_json::from_str::<QuantizationConfig>(
            r#"{"group_size": 64, "bits": 4, "model.layers.0.mlp.gate": {"bits": 8}}"#,
        );
        assert!(r.is_err());
    }
}
