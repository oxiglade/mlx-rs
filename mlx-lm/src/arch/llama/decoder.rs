use std::collections::BTreeMap;

use mlx_rs::{fast, ops, utils::StateProjection, Array, Dtype};

use super::{config::dimension, rope::Rope, ParsedConfig};
use crate::{
    arch::DecoderModel,
    cache::{CacheStep, LayerCacheSpec},
    weights::WeightManifest,
    AffineQuantization, AttentionKind, CacheError, Config, ConfigError, InferenceError,
    LayerQuantization, LoadError, ParameterPath, WeightError,
};

struct Matrix {
    weight: Array,
    affine: Option<(AffineQuantization, Array, Array)>,
    bias: Option<Array>,
}

fn float_dtype(weights: &WeightManifest, key: &str) -> Result<Dtype, WeightError> {
    let entry = weights
        .tensors
        .get(key)
        .ok_or_else(|| WeightError::MissingKey(key.into()))?;
    match entry.dtype {
        dtype @ (Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16) => Ok(dtype),
        dtype => Err(WeightError::UnsupportedDtype {
            key: key.into(),
            dtype: format!("{dtype:?}"),
        }),
    }
}

fn slot(
    weights: &WeightManifest,
    key: &str,
    shape: &[usize],
    dtype: Dtype,
) -> Result<Array, WeightError> {
    let entry = weights
        .tensors
        .get(key)
        .ok_or_else(|| WeightError::MissingKey(key.into()))?;
    let actual_dtype = entry.dtype;
    if actual_dtype != dtype {
        return Err(WeightError::DtypeMismatch {
            key: key.into(),
            expected: dtype,
            actual: actual_dtype,
        });
    }
    if entry.shape != shape {
        return Err(WeightError::ShapeMismatch {
            key: key.into(),
            expected: shape.to_vec(),
            actual: entry.shape.clone(),
        });
    }
    let shape = shape
        .iter()
        .map(|&size| {
            i32::try_from(size).map_err(|_| {
                WeightError::UnsupportedFormat("parameter dimension exceeds i32".into())
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ops::zeros_dtype(&shape, dtype)?)
}

impl Matrix {
    fn new(
        config: &Config,
        weights: &WeightManifest,
        key: &str,
        input: usize,
        output: usize,
        biased: bool,
    ) -> Result<Self, LoadError> {
        let weight_key = format!("{key}.weight");
        let scales_key = format!("{key}.scales");
        let packed = weights.tensors.contains_key(&scales_key)
            || weights.tensors.contains_key(&format!("{key}.biases"))
            || weights
                .tensors
                .get(&weight_key)
                .is_some_and(|entry| entry.dtype == Dtype::Uint32);
        if packed && config.quantization.is_none() {
            return Err(ConfigError::UnsupportedQuantization(format!(
                "{key} packed weights require quantization settings"
            ))
            .into());
        }
        let setting = config.quantization.as_ref().and_then(|quant| {
            quant
                .layers
                .get(&ParameterPath::new(key))
                .cloned()
                .or_else(|| packed.then(|| LayerQuantization::Affine(quant.default.clone())))
        });
        let (weight, affine) = match setting {
            Some(LayerQuantization::Affine(options)) => {
                if !input.is_multiple_of(options.group_size.get()) {
                    return Err(ConfigError::UnsupportedQuantization(format!(
                        "{key} input width is not divisible by group_size"
                    ))
                    .into());
                }
                let packed_width = input
                    .checked_mul(usize::from(options.bits))
                    .filter(|width| width % 32 == 0)
                    .ok_or_else(|| {
                        ConfigError::UnsupportedQuantization(format!(
                            "{key} packed width is not integral"
                        ))
                    })?
                    / 32;
                let dtype = float_dtype(weights, &scales_key)?;
                let shape = [output, input / options.group_size.get()];
                (
                    slot(weights, &weight_key, &[output, packed_width], Dtype::Uint32)?,
                    Some((
                        options,
                        slot(weights, &scales_key, &shape, dtype)?,
                        slot(weights, &format!("{key}.biases"), &shape, dtype)?,
                    )),
                )
            }
            _ => (
                slot(
                    weights,
                    &weight_key,
                    &[output, input],
                    float_dtype(weights, &weight_key)?,
                )?,
                None,
            ),
        };
        let bias = if biased {
            let key = format!("{key}.bias");
            Some(slot(weights, &key, &[output], float_dtype(weights, &key)?)?)
        } else {
            None
        };
        Ok(Self {
            weight,
            affine,
            bias,
        })
    }

    fn project<'a>(
        &'a mut self,
        key: &str,
        projection: &mut StateProjection<'a>,
    ) -> Result<(), WeightError> {
        projection.required(format!("{key}.weight"), &mut self.weight)?;
        if let Some((_, scales, biases)) = &mut self.affine {
            projection.required(format!("{key}.scales"), scales)?;
            projection.required(format!("{key}.biases"), biases)?;
        }
        projection.optional(format!("{key}.bias"), &mut self.bias)?;
        Ok(())
    }

    fn linear(&self, input: &Array) -> Result<Array, InferenceError> {
        let result = match &self.affine {
            None => input.matmul(self.weight.transpose()?)?,
            Some((options, scales, biases)) => ops::quantized_matmul(
                input,
                &self.weight,
                scales,
                biases,
                true,
                options.group_size.get() as i32,
                i32::from(options.bits),
            )?,
        };
        Ok(match &self.bias {
            Some(bias) => result.add(bias)?,
            None => result,
        })
    }

    fn embed(&self, tokens: &Array) -> Result<Array, InferenceError> {
        let weight = self.weight.take_axis(tokens, 0)?;
        Ok(match &self.affine {
            None => weight,
            Some((options, scales, biases)) => {
                let [batch, length] = tokens.shape() else {
                    return Err(CacheError::InvalidState(
                        "embedding requires [batch, T] tokens".into(),
                    )
                    .into());
                };
                let weight = weight.flatten(0, -2)?;
                let scales = scales.take_axis(tokens, 0)?.flatten(0, -2)?;
                let biases = biases.take_axis(tokens, 0)?.flatten(0, -2)?;
                ops::dequantize(
                    weight,
                    scales,
                    &biases,
                    options.group_size.get() as i32,
                    i32::from(options.bits),
                )?
                .reshape(&[*batch, *length, -1])?
            }
        })
    }
}

pub(super) struct Decoder {
    config: Config,
    layout: Vec<LayerCacheSpec>,
    matrices: BTreeMap<String, Matrix>,
    norms: BTreeMap<String, Array>,
    rope: Rope,
}

impl Decoder {
    pub(super) fn new(parsed: ParsedConfig, weights: &WeightManifest) -> Result<Self, LoadError> {
        let config = parsed.config;
        let dims = &config.dimensions;
        let mut matrices = BTreeMap::new();
        let mut norms = BTreeMap::new();
        for (key, (input, output, bias)) in super::config::matrix_layout(&config) {
            matrices.insert(
                key.clone(),
                Matrix::new(&config, weights, &key, input, output, bias)?,
            );
        }
        let mut add_norm = |key: String| -> Result<(), LoadError> {
            let dtype = float_dtype(weights, &key)?;
            norms.insert(
                key.clone(),
                slot(weights, &key, &[dims.hidden_size], dtype)?,
            );
            Ok(())
        };
        add_norm("model.norm.weight".into())?;
        for layer in 0..dims.layer_count {
            let prefix = format!("model.layers.{layer}");
            add_norm(format!("{prefix}.input_layernorm.weight"))?;
            add_norm(format!("{prefix}.post_attention_layernorm.weight"))?;
        }
        let layout = config
            .attention
            .iter()
            .enumerate()
            .map(|(layer, attention)| {
                let prefix = format!("model.layers.{layer}");
                let norm_key = format!("{prefix}.input_layernorm.weight");
                let norm = norms
                    .get(&norm_key)
                    .ok_or_else(|| WeightError::MissingKey(norm_key.clone()))?;
                let projection_key = format!("{prefix}.self_attn.k_proj");
                let projection = matrices
                    .get(&projection_key)
                    .ok_or_else(|| WeightError::MissingKey(format!("{projection_key}.weight")))?;
                let projection_dtype = match &projection.affine {
                    Some((_, scales, _)) => scales.dtype(),
                    None => projection.weight.dtype(),
                };
                Ok(LayerCacheSpec {
                    attention: attention.clone(),
                    batch_size: 1,
                    kv_heads: dims.kv_heads,
                    head_dim: dims.head_dim,
                    dtype: Dtype::from_promoting_types(norm.dtype(), projection_dtype),
                })
            })
            .collect::<Result<Vec<_>, WeightError>>()?;
        let rope = Rope::new(config.rope.clone())?;
        Ok(Self {
            config,
            layout,
            matrices,
            norms,
            rope,
        })
    }

    fn matrix(&self, key: &str) -> Result<&Matrix, InferenceError> {
        self.matrices.get(key).ok_or_else(|| {
            InferenceError::UnsupportedArchitecture(format!("missing llama matrix {key}"))
        })
    }

    fn norm(&self, input: &Array, key: &str) -> Result<Array, InferenceError> {
        let weight = self.norms.get(key).ok_or_else(|| {
            InferenceError::UnsupportedArchitecture(format!("missing llama norm {key}"))
        })?;
        Ok(fast::rms_norm(
            input,
            Some(weight),
            self.config.dimensions.rms_norm_epsilon,
        )?)
    }

    fn forward_with(
        &self,
        tokens: &Array,
        offset: usize,
        mut update: impl FnMut(usize, Array, Array) -> Result<(Array, Array, Vec<usize>), CacheError>,
    ) -> Result<Array, InferenceError> {
        let [batch, length] = tokens.shape() else {
            return Err(
                CacheError::InvalidState("llama tokens must have shape [batch, T]".into()).into(),
            );
        };
        if *batch <= 0 || *length <= 0 || !matches!(tokens.dtype(), Dtype::Int32 | Dtype::Uint32) {
            return Err(CacheError::InvalidState(
                "llama tokens require nonempty integer dimensions".into(),
            )
            .into());
        }
        let dims = &self.config.dimensions;
        let heads = dimension(dims.attention_heads, "attention heads")
            .map_err(|e| InferenceError::UnsupportedArchitecture(e.to_string()))?;
        let kv_heads = dimension(dims.kv_heads, "KV heads")
            .map_err(|e| InferenceError::UnsupportedArchitecture(e.to_string()))?;
        let mut hidden = self.matrix("model.embed_tokens")?.embed(tokens)?;
        for (layer, attention) in self.config.attention.iter().enumerate() {
            let prefix = format!("model.layers.{layer}");
            let normalized = self.norm(&hidden, &format!("{prefix}.input_layernorm.weight"))?;
            let project = |name: &str, heads| -> Result<Array, InferenceError> {
                Ok(self
                    .matrix(&format!("{prefix}.self_attn.{name}"))?
                    .linear(&normalized)?
                    .reshape(&[*batch, *length, heads, -1])?
                    .transpose_axes(&[0, 2, 1, 3])?)
            };
            let queries = self.rope.apply(&project("q_proj", heads)?, offset)?;
            let keys = self.rope.apply(&project("k_proj", kv_heads)?, offset)?;
            let values = project("v_proj", kv_heads)?;
            let (keys, values, positions) = update(layer, keys, values)?;
            if keys.shape().get(2).copied() != i32::try_from(positions.len()).ok() {
                return Err(CacheError::InvalidState(
                    "cache positions disagree with returned keys".into(),
                )
                .into());
            }
            let mask = attention_mask(offset, *length as usize, &positions, attention)?;
            let output = fast::scaled_dot_product_attention(
                queries,
                keys,
                values,
                (dims.head_dim as f32).sqrt().recip(),
                Some(fast::ScaledDotProductAttentionMask::Array(&mask)),
                None,
            )?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[*batch, *length, -1])?;
            hidden = hidden.add(
                self.matrix(&format!("{prefix}.self_attn.o_proj"))?
                    .linear(&output)?,
            )?;
            let normalized = self.norm(
                &hidden,
                &format!("{prefix}.post_attention_layernorm.weight"),
            )?;
            let gate = mlx_rs::nn::silu(
                self.matrix(&format!("{prefix}.mlp.gate_proj"))?
                    .linear(&normalized)?,
            )?;
            let up = self
                .matrix(&format!("{prefix}.mlp.up_proj"))?
                .linear(&normalized)?;
            hidden = hidden.add(
                self.matrix(&format!("{prefix}.mlp.down_proj"))?
                    .linear(&gate.multiply(up)?)?,
            )?;
        }
        let output = self.norm(&hidden, "model.norm.weight")?;
        self.matrix(if self.config.tie_word_embeddings {
            "model.embed_tokens"
        } else {
            "lm_head"
        })?
        .linear(&output)
    }
}

fn attention_mask(
    offset: usize,
    length: usize,
    positions: &[usize],
    attention: &AttentionKind,
) -> Result<Array, CacheError> {
    let rows = i32::try_from(length)
        .map_err(|_| CacheError::InvalidState("attention length exceeds i32".into()))?;
    let columns = i32::try_from(positions.len())
        .map_err(|_| CacheError::InvalidState("cache length exceeds i32".into()))?;
    let values = mask_data(offset, length, positions, attention)?;
    Ok(Array::from_slice(&values, &[rows, columns]))
}

pub(super) fn mask_data(
    offset: usize,
    length: usize,
    positions: &[usize],
    attention: &AttentionKind,
) -> Result<Vec<bool>, CacheError> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| CacheError::InvalidState("attention position overflow".into()))?;
    if positions.len() < length || positions.len() > end {
        return Err(CacheError::InvalidState(
            "cache length disagrees with processed tokens".into(),
        ));
    }
    Ok((offset..end)
        .flat_map(|query| {
            positions.iter().map(move |&key| {
                key <= query
                    && match attention {
                        AttentionKind::Full => true,
                        AttentionKind::Sliding { window } => query - key < window.get(),
                    }
            })
        })
        .collect::<Vec<_>>())
}

impl DecoderModel for Decoder {
    fn config(&self) -> &Config {
        &self.config
    }
    fn cache_layout(&self) -> &[LayerCacheSpec] {
        &self.layout
    }

    fn forward(
        &mut self,
        tokens: &Array,
        cache: &mut CacheStep<'_>,
    ) -> Result<Array, InferenceError> {
        let offset = cache.info(0)?.processed_tokens;
        self.forward_with(tokens, offset, |layer, keys, values| {
            if cache.info(layer)?.processed_tokens != offset {
                return Err(CacheError::InvalidState(
                    "llama cache layers have different offsets".into(),
                ));
            }
            let (keys, values) = cache.update_and_fetch(layer, keys, values)?;
            let info = cache.info(layer)?;
            let positions = info
                .retained_prefix
                .chain(info.retained_positions)
                .collect();
            Ok((keys, values, positions))
        })
    }

    fn weight_projection(&mut self) -> Result<StateProjection<'_>, WeightError> {
        let mut projection = StateProjection::new();
        for (key, matrix) in &mut self.matrices {
            matrix.project(key, &mut projection)?;
        }
        for (key, weight) in &mut self.norms {
            projection.required(key.as_str(), weight)?;
        }
        Ok(projection)
    }
}
