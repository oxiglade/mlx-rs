use mlx_rs::{
    module::{Module, Param},
    nn, ops,
    utils::StateProjection,
    Array, Dtype,
};

use super::config::dimension;
use crate::{
    config::ParameterPath,
    weights::{WeightDisposition, WeightManifest},
    AffineQuantization, Config, ConfigError, InferenceError, LayerQuantization, WeightError,
};

pub(super) fn map_key(external: &str, tied: bool) -> WeightDisposition {
    if external == "model.rotary_emb.inv_freq"
        || external.strip_prefix("model.layers.").is_some_and(|rest| {
            rest.split_once('.').is_some_and(|(layer, suffix)| {
                canonical_layer(layer).is_some() && suffix == "self_attn.rotary_emb.inv_freq"
            })
        })
    {
        return WeightDisposition::Ignore {
            reason: "RoPE frequencies are derived from the validated configuration",
        };
    }
    if tied && external == "lm_head.weight" {
        return WeightDisposition::Ignore {
            reason: "tied output projection uses model.embed_tokens.weight",
        };
    }
    let internal = if external.starts_with("lm_head.") {
        external
    } else if let Some(internal) = external.strip_prefix("model.") {
        internal
    } else {
        return WeightDisposition::Reject;
    };
    let Some((group, suffix)) = internal.rsplit_once('.') else {
        return WeightDisposition::Reject;
    };
    let accepted = if is_matrix(group) {
        matches!(suffix, "weight" | "scales" | "biases")
            || (suffix == "bias" && group != "embed_tokens" && group != "lm_head")
    } else {
        is_norm(group) && suffix == "weight"
    };
    if accepted {
        WeightDisposition::Parameter(ParameterPath::new(internal))
    } else {
        WeightDisposition::Reject
    }
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
    matches!(group, "embed_tokens" | "lm_head")
        || layer_group(group).is_some_and(|(_, tail)| {
            matches!(
                tail,
                "self_attn.q_proj"
                    | "self_attn.k_proj"
                    | "self_attn.v_proj"
                    | "self_attn.o_proj"
                    | "mlp.gate_proj"
                    | "mlp.up_proj"
                    | "mlp.down_proj"
            )
        })
}

fn is_norm(group: &str) -> bool {
    group == "norm"
        || layer_group(group).is_some_and(|(_, tail)| {
            matches!(
                tail,
                "input_layernorm"
                    | "post_attention_layernorm"
                    | "self_attn.q_norm"
                    | "self_attn.k_norm"
            )
        })
}

pub(super) fn matrix_dimensions(config: &Config, group: &str) -> Option<(usize, usize)> {
    let d = &config.dimensions;
    if matches!(group, "embed_tokens" | "lm_head") {
        return Some((d.vocabulary_size, d.hidden_size));
    }
    let (layer, tail) = layer_group(group)?;
    if layer >= d.layer_count {
        return None;
    }
    match tail {
        "self_attn.q_proj" => Some((d.attention_heads.checked_mul(d.head_dim)?, d.hidden_size)),
        "self_attn.k_proj" | "self_attn.v_proj" => {
            Some((d.kv_heads.checked_mul(d.head_dim)?, d.hidden_size))
        }
        "self_attn.o_proj" => Some((d.hidden_size, d.attention_heads.checked_mul(d.head_dim)?)),
        "mlp.gate_proj" | "mlp.up_proj" => Some((d.intermediate_size, d.hidden_size)),
        "mlp.down_proj" => Some((d.hidden_size, d.intermediate_size)),
        _ => None,
    }
}

pub(super) struct Slots<'a> {
    pub(super) config: &'a Config,
    pub(super) manifest: &'a WeightManifest,
}

impl Slots<'_> {
    fn external(group: &str, suffix: &str) -> String {
        if group == "lm_head" {
            format!("{group}.{suffix}")
        } else {
            format!("model.{group}.{suffix}")
        }
    }

    pub(super) fn array(
        &self,
        group: &str,
        suffix: &str,
        shape: &[usize],
        packed: bool,
    ) -> Result<Array, WeightError> {
        let key = Self::external(group, suffix);
        let entry = self
            .manifest
            .tensors
            .get(&key)
            .ok_or_else(|| WeightError::MissingKey(key.clone()))?;
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
        self.array(group, "weight", &[size], false)
    }

    fn quantization(&self, group: &str) -> Result<Option<AffineQuantization>, ConfigError> {
        let packed = ["scales", "biases"].iter().any(|suffix| {
            self.manifest
                .tensors
                .contains_key(&Self::external(group, suffix))
        }) || self
            .manifest
            .tensors
            .get(&Self::external(group, "weight"))
            .is_some_and(|entry| entry.dtype == Dtype::Uint32);
        let Some(config) = &self.config.quantization else {
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
        let (output, input) = matrix_dimensions(self.config, group)
            .ok_or_else(|| WeightError::UnexpectedKey(group.into()))?;
        let bias = if bias {
            Some(self.array(group, "bias", &[output], false)?)
        } else {
            None
        };
        match self.quantization(group)? {
            None => Ok(Linear::Float(nn::Linear {
                weight: Param::new(self.array(group, "weight", &[output, input], false)?),
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
                        "scales",
                        &[output, input / options.group_size.get()],
                        false,
                    )?),
                    biases: Param::new(self.array(
                        group,
                        "biases",
                        &[output, input / options.group_size.get()],
                        false,
                    )?),
                    inner: nn::Linear {
                        weight: Param::new(self.array(group, "weight", &[output, packed], true)?),
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
        // The core floating Linear uses the panic-based transpose convenience method.
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
    pub(super) fn new(slots: &Slots<'_>) -> Result<Self, crate::LoadError> {
        Ok(match slots.linear("embed_tokens", false)? {
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
                let output =
                    ops::dequantize(weight, scales, &biases, layer.group_size, layer.bits)?;
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
        projection: &mut StateProjection<'a>,
    ) -> Result<(), WeightError> {
        let inner = match self {
            Self::Float(inner) => inner,
            Self::Affine(layer) => {
                projection.required("embed_tokens.scales", &mut layer.scales.value)?;
                projection.required("embed_tokens.biases", &mut layer.biases.value)?;
                &mut layer.inner
            }
        };
        projection.required("embed_tokens.weight", &mut inner.weight.value)?;
        Ok(())
    }
}
