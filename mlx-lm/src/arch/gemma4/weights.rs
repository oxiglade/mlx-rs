use mlx_rs::{
    module::{Module, Param},
    nn, ops,
    utils::StateProjection,
    Array, Dtype,
};

use super::config::{dimension, ParsedConfig};
use crate::{
    config::ParameterPath,
    weights::{WeightDisposition, WeightEntry, WeightManifest},
    AffineQuantization, AttentionKind, ConfigError, InferenceError, LayerQuantization, WeightError,
};

pub(super) fn map_key(
    external: &str,
    tied: bool,
    first_kv_shared: Option<usize>,
) -> WeightDisposition {
    if external == "model.rotary_emb.inv_freq"
        || external.ends_with(".self_attn.rotary_emb.inv_freq")
        || external.contains("self_attn.rotary_emb.")
    {
        return WeightDisposition::Ignore {
            reason: "RoPE frequencies are derived from the validated configuration",
        };
    }
    if external.ends_with(".input_max")
        || external.ends_with(".input_min")
        || external.ends_with(".output_max")
        || external.ends_with(".output_min")
    {
        return WeightDisposition::Ignore {
            reason: "activation calibration bounds are not used during inference",
        };
    }

    let without_model = external.strip_prefix("model.").unwrap_or(external);
    if without_model.starts_with("vision_tower.")
        || without_model.starts_with("multi_modal_projector.")
        || without_model.starts_with("audio_tower.")
        || without_model.starts_with("embed_audio.")
        || without_model.starts_with("embed_vision.")
        || without_model.starts_with("vision_embedder.")
        || without_model.starts_with("mtp.")
        || without_model.starts_with("language_model.mtp.")
    {
        return WeightDisposition::Ignore {
            reason: "multimodal tower and MTP speculative head weights are ignored by text decoder",
        };
    }

    let Some(internal) = strip_backbone_prefix(external) else {
        return WeightDisposition::Reject;
    };

    if tied && internal == "lm_head.weight" {
        return WeightDisposition::Ignore {
            reason: "tied output projection uses embed_tokens.weight",
        };
    }

    if let Some((layer, tail)) = layer_group(internal) {
        if tail == "layer_scalar" || tail == "layer_scalar.weight" {
            return WeightDisposition::Parameter(ParameterPath::new(format!(
                "layers.{layer}.layer_scalar"
            )));
        }
        if let Some(first_shared) = first_kv_shared {
            if layer >= first_shared
                && (tail.starts_with("self_attn.k_proj.")
                    || tail.starts_with("self_attn.v_proj.")
                    || tail.starts_with("self_attn.k_norm.")
                    || tail.starts_with("self_attn.v_norm."))
            {
                return WeightDisposition::Ignore {
                    reason: "KV-shared layers reuse K/V from earlier layers",
                };
            }
        }
    }

    let Some((group, suffix)) = internal.rsplit_once('.') else {
        return WeightDisposition::Reject;
    };
    let accepted = if is_matrix(group) {
        matches!(suffix, "weight" | "scales" | "biases")
            || (suffix == "bias"
                && !matches!(group, "embed_tokens" | "embed_tokens_per_layer" | "lm_head"))
    } else {
        is_norm(group) && suffix == "weight"
    };
    if accepted {
        WeightDisposition::Parameter(ParameterPath::new(internal))
    } else {
        WeightDisposition::Reject
    }
}

fn strip_backbone_prefix(external: &str) -> Option<&str> {
    if external.starts_with("lm_head.") {
        return Some(external);
    }
    for prefix in [
        "model.language_model.model.",
        "language_model.model.",
        "model.language_model.",
        "language_model.",
        "model.",
    ] {
        if let Some(rest) = external.strip_prefix(prefix) {
            return (!rest.is_empty()).then_some(rest);
        }
    }
    None
}

fn canonical_layer(layer: &str) -> Option<usize> {
    let number = layer.parse::<usize>().ok()?;
    (number.to_string() == layer).then_some(number)
}

fn layer_group(group: &str) -> Option<(usize, &str)> {
    let (layer, tail) = group.strip_prefix("layers.")?.split_once('.')?;
    Some((canonical_layer(layer)?, tail))
}

fn is_matrix(group: &str) -> bool {
    matches!(
        group,
        "embed_tokens" | "embed_tokens_per_layer" | "per_layer_model_projection" | "lm_head"
    ) || layer_group(group).is_some_and(|(_, tail)| {
        matches!(
            tail,
            "self_attn.q_proj"
                | "self_attn.k_proj"
                | "self_attn.v_proj"
                | "self_attn.o_proj"
                | "mlp.gate_proj"
                | "mlp.up_proj"
                | "mlp.down_proj"
                | "per_layer_input_gate"
                | "per_layer_projection"
        )
    })
}

fn is_norm(group: &str) -> bool {
    matches!(group, "norm" | "per_layer_projection_norm")
        || layer_group(group).is_some_and(|(_, tail)| {
            matches!(
                tail,
                "input_layernorm"
                    | "post_attention_layernorm"
                    | "pre_feedforward_layernorm"
                    | "post_feedforward_layernorm"
                    | "post_per_layer_input_norm"
                    | "self_attn.q_norm"
                    | "self_attn.k_norm"
            )
        })
}

pub(super) fn matrix_dimensions(parsed: &ParsedConfig, group: &str) -> Option<(usize, usize)> {
    let d = &parsed.config.dimensions;
    match group {
        "embed_tokens" | "lm_head" => return Some((d.vocabulary_size, d.hidden_size)),
        "embed_tokens_per_layer" => {
            if parsed.hidden_size_per_layer_input == 0 {
                return None;
            }
            return Some((
                parsed.vocab_size_per_layer_input,
                d.layer_count
                    .checked_mul(parsed.hidden_size_per_layer_input)?,
            ));
        }
        "per_layer_model_projection" => {
            if parsed.hidden_size_per_layer_input == 0 {
                return None;
            }
            return Some((
                d.layer_count
                    .checked_mul(parsed.hidden_size_per_layer_input)?,
                d.hidden_size,
            ));
        }
        _ => {}
    }
    let (layer, tail) = layer_group(group)?;
    if layer >= d.layer_count {
        return None;
    }
    let head_dim = *parsed.layer_head_dims.get(layer)?;
    let kv_heads = *parsed.layer_kv_heads.get(layer)?;
    let inter = *parsed.layer_intermediate_sizes.get(layer)?;
    let is_sliding = matches!(
        parsed.config.attention.get(layer),
        Some(AttentionKind::Sliding { .. })
    );
    let use_k_eq_v = parsed.attention_k_eq_v && !is_sliding;

    match tail {
        "self_attn.q_proj" => Some((d.attention_heads.checked_mul(head_dim)?, d.hidden_size)),
        "self_attn.k_proj" if layer < parsed.first_kv_shared_layer_idx => {
            Some((kv_heads.checked_mul(head_dim)?, d.hidden_size))
        }
        "self_attn.v_proj" if layer < parsed.first_kv_shared_layer_idx && !use_k_eq_v => {
            Some((kv_heads.checked_mul(head_dim)?, d.hidden_size))
        }
        "self_attn.o_proj" => Some((d.hidden_size, d.attention_heads.checked_mul(head_dim)?)),
        "mlp.gate_proj" | "mlp.up_proj" => Some((inter, d.hidden_size)),
        "mlp.down_proj" => Some((d.hidden_size, inter)),
        "per_layer_input_gate" if parsed.hidden_size_per_layer_input > 0 => {
            Some((parsed.hidden_size_per_layer_input, d.hidden_size))
        }
        "per_layer_projection" if parsed.hidden_size_per_layer_input > 0 => {
            Some((d.hidden_size, parsed.hidden_size_per_layer_input))
        }
        _ => None,
    }
}

pub(super) struct Slots<'a> {
    pub(super) parsed: &'a ParsedConfig,
    pub(super) manifest: &'a WeightManifest,
}

impl<'a> Slots<'a> {
    fn candidate_keys(group: &str, suffix: Option<&str>) -> Vec<String> {
        let leaf = match suffix {
            Some(s) => format!("{group}.{s}"),
            None => group.to_string(),
        };
        if group == "lm_head" {
            vec![
                leaf.clone(),
                format!("language_model.{leaf}"),
                format!("model.language_model.{leaf}"),
                format!("model.{leaf}"),
            ]
        } else {
            vec![
                format!("model.{leaf}"),
                format!("language_model.model.{leaf}"),
                format!("model.language_model.model.{leaf}"),
                format!("model.language_model.{leaf}"),
                format!("language_model.{leaf}"),
            ]
        }
    }

    fn find_entry(&self, group: &str, suffix: Option<&str>) -> Option<(String, &'a WeightEntry)> {
        for key in Self::candidate_keys(group, suffix) {
            if let Some(entry) = self.manifest.tensors.get(&key) {
                return Some((key, entry));
            }
        }
        None
    }

    pub(super) fn array(
        &self,
        group: &str,
        suffix: Option<&str>,
        shape: &[usize],
        packed: bool,
    ) -> Result<Array, WeightError> {
        let default_key = Self::candidate_keys(group, suffix)
            .into_iter()
            .next()
            .unwrap_or_else(|| group.to_string());
        let (key, entry) = self
            .find_entry(group, suffix)
            .ok_or(WeightError::MissingKey(default_key))?;
        if entry.shape != shape {
            return Err(WeightError::ShapeMismatch {
                key,
                expected: shape.to_vec(),
                actual: entry.shape.clone(),
            });
        }
        let dtype = match (packed, entry.dtype) {
            (true, Dtype::Uint32) => Dtype::Uint32,
            (false, Dtype::Float32) => Dtype::Float32,
            (false, Dtype::Float16) => Dtype::Float16,
            (false, Dtype::Bfloat16) => Dtype::Bfloat16,
            _ => {
                return Err(WeightError::UnsupportedDtype {
                    key,
                    dtype: format!("{:?}", entry.dtype),
                })
            }
        };
        let shape = shape
            .iter()
            .map(|&n| {
                i32::try_from(n).map_err(|_| {
                    WeightError::UnsupportedFormat("dimension exceeds MLX range".into())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ops::zeros_dtype(&shape, dtype)?)
    }

    pub(super) fn norm(&self, group: &str, size: usize) -> Result<Array, WeightError> {
        self.array(group, Some("weight"), &[size], false)
    }

    pub(super) fn optional_layer_scalar(&self, index: usize) -> Result<Option<Array>, WeightError> {
        let group = format!("layers.{index}.layer_scalar");
        if self.find_entry(&group, None).is_some() {
            Ok(Some(self.array(&group, None, &[1], false)?))
        } else if self.find_entry(&group, Some("weight")).is_some() {
            Ok(Some(self.array(&group, Some("weight"), &[1], false)?))
        } else {
            Ok(None)
        }
    }

    fn quantization(&self, group: &str) -> Result<Option<AffineQuantization>, ConfigError> {
        let packed = ["scales", "biases"]
            .iter()
            .any(|suffix| self.find_entry(group, Some(suffix)).is_some())
            || self
                .find_entry(group, Some("weight"))
                .is_some_and(|(_, entry)| entry.dtype == Dtype::Uint32);
        let Some(config) = &self.parsed.config.quantization else {
            if packed {
                return Err(ConfigError::UnsupportedQuantization(format!(
                    "packed {group} without quantization configuration"
                )));
            }
            return Ok(None);
        };
        match config.layers.get(&ParameterPath::new(group)) {
            Some(LayerQuantization::Unquantized) if packed => {
                Err(ConfigError::UnsupportedQuantization(format!(
                    "packed {group} is marked unquantized"
                )))
            }
            Some(LayerQuantization::Unquantized) => Ok(None),
            Some(LayerQuantization::Affine(options)) => Ok(Some(options.clone())),
            None => Ok(packed.then(|| config.default.clone())),
        }
    }

    pub(super) fn linear(&self, group: &str, bias: bool) -> Result<Linear, crate::LoadError> {
        let (output, input) = matrix_dimensions(self.parsed, group)
            .ok_or_else(|| WeightError::UnexpectedKey(group.into()))?;
        let bias = if bias {
            Some(self.array(group, Some("bias"), &[output], false)?)
        } else {
            None
        };
        match self.quantization(group)? {
            None => Ok(Linear::Float(nn::Linear {
                weight: Param::new(self.array(group, Some("weight"), &[output, input], false)?),
                bias: Param::new(bias),
            })),
            Some(options) => {
                if !input.is_multiple_of(options.group_size.get()) {
                    return Err(ConfigError::UnsupportedQuantization(format!(
                        "{group} input width is not divisible by group size"
                    ))
                    .into());
                }
                let packed = input
                    .checked_mul(usize::from(options.bits))
                    .filter(|bits| bits.is_multiple_of(32))
                    .map(|bits| bits / 32)
                    .ok_or_else(|| {
                        ConfigError::UnsupportedQuantization(format!(
                            "{group} packed width is not whole uint32 words"
                        ))
                    })?;
                Ok(Linear::Affine(nn::QuantizedLinear {
                    group_size: dimension("group_size", options.group_size.get())?,
                    bits: i32::from(options.bits),
                    scales: Param::new(self.array(
                        group,
                        Some("scales"),
                        &[output, input / options.group_size.get()],
                        false,
                    )?),
                    biases: Param::new(self.array(
                        group,
                        Some("biases"),
                        &[output, input / options.group_size.get()],
                        false,
                    )?),
                    inner: nn::Linear {
                        weight: Param::new(self.array(
                            group,
                            Some("weight"),
                            &[output, packed],
                            true,
                        )?),
                        bias: Param::new(bias),
                    },
                }))
            }
        }
    }
}

pub(super) enum Linear {
    Float(nn::Linear),
    Affine(nn::QuantizedLinear),
}

impl Linear {
    pub(super) fn dtype(&self) -> Dtype {
        match self {
            Self::Float(layer) => layer.weight.dtype(),
            Self::Affine(layer) => layer.scales.dtype(),
        }
    }
    pub(super) fn forward(&mut self, x: &Array) -> Result<Array, InferenceError> {
        match self {
            Self::Float(layer) => {
                let output = x.matmul(layer.weight.transpose_axes(&[1, 0])?)?;
                Ok(match &layer.bias.value {
                    Some(bias) => output.add(bias)?,
                    None => output,
                })
            }
            Self::Affine(layer) => Ok(layer.forward(x)?),
        }
    }
    pub(super) fn project<'a>(
        &'a mut self,
        group: &str,
        projection: &mut StateProjection<'a>,
    ) -> Result<(), WeightError> {
        let inner = match self {
            Self::Float(inner) => inner,
            Self::Affine(layer) => {
                projection.required(format!("{group}.scales"), &mut layer.scales.value)?;
                projection.required(format!("{group}.biases"), &mut layer.biases.value)?;
                &mut layer.inner
            }
        };
        projection.required(format!("{group}.weight"), &mut inner.weight.value)?;
        projection.optional(format!("{group}.bias"), &mut inner.bias.value)?;
        Ok(())
    }
}

pub(super) enum Embedding {
    Float(nn::Embedding),
    Affine(nn::QuantizedEmbedding),
}

impl Embedding {
    pub(super) fn new(slots: &Slots<'_>, group: &str) -> Result<Self, crate::LoadError> {
        Ok(match slots.linear(group, false)? {
            Linear::Float(layer) => Self::Float(nn::Embedding {
                weight: layer.weight,
            }),
            Linear::Affine(layer) => Self::Affine(nn::QuantizedEmbedding {
                group_size: layer.group_size,
                bits: layer.bits,
                scales: layer.scales,
                biases: layer.biases,
                inner: nn::Embedding {
                    weight: layer.inner.weight,
                },
            }),
        })
    }
    pub(super) fn forward(&self, tokens: &Array) -> Result<Array, InferenceError> {
        match self {
            Self::Float(layer) => Ok(layer.weight.take_axis(tokens, 0)?),
            Self::Affine(layer) => {
                let flat = tokens.reshape(&[-1])?;
                let weight = layer.inner.weight.take_axis(&flat, 0)?;
                let scales = layer.scales.take_axis(&flat, 0)?;
                let biases = layer.biases.take_axis(&flat, 0)?;
                let output = ops::dequantize(
                    weight,
                    scales,
                    &biases,
                    ops::DequantizeOptions {
                        group_size: Some(layer.group_size),
                        bits: Some(layer.bits),
                        ..Default::default()
                    },
                )
                .map_err(mlx_rs::error::Exception::from)?;
                let shape = tokens
                    .shape()
                    .iter()
                    .copied()
                    .chain(std::iter::once(-1))
                    .collect::<Vec<_>>();
                Ok(output.reshape(&shape)?)
            }
        }
    }
    pub(super) fn as_linear(&self, x: &Array) -> Result<Array, InferenceError> {
        Ok(match self {
            Self::Float(layer) => x.matmul(layer.weight.transpose_axes(&[1, 0])?)?,
            Self::Affine(layer) => layer.as_linear(x)?,
        })
    }
    pub(super) fn project<'a>(
        &'a mut self,
        group: &str,
        projection: &mut StateProjection<'a>,
    ) -> Result<(), WeightError> {
        let inner = match self {
            Self::Float(inner) => inner,
            Self::Affine(layer) => {
                projection.required(format!("{group}.scales"), &mut layer.scales.value)?;
                projection.required(format!("{group}.biases"), &mut layer.biases.value)?;
                &mut layer.inner
            }
        };
        projection.required(format!("{group}.weight"), &mut inner.weight.value)?;
        Ok(())
    }
}
