//! EmbeddingGemma 2 text embeddings.

mod config;
#[cfg(test)]
mod tests;

use crate::{
    config::ParameterPath,
    weights::{WeightDisposition, WeightManifest},
    EmbeddingError, LoadError, TokenId, Tokenizer, WeightError,
};
use config::{EncoderConfig, LayerShape};
use mlx_rs::{fast, nn, ops, utils::StateProjection, Array, Dtype};
use std::{marker::PhantomData, path::Path, rc::Rc};

/// External prefix of the text encoder's tensors in MLX-layout checkpoints.
const TEXT_PREFIX: &str = "language_model.";
/// Media towers this text-only embedder leaves unloaded.
const MEDIA_PREFIXES: [&str; 4] = [
    "vision_tower.",
    "embed_vision.",
    "audio_tower.",
    "embed_audio.",
];

/// A loaded EmbeddingGemma 2 text encoder and its tokenizer.
///
/// [`Embedder::embed`] tokenizes each text with the tokenizer's post-processor
/// (`<bos>` and `<eos>` for the published checkpoints), right-pads the batch with
/// the configured pad token, runs the bidirectional encoder, mean-pools the
/// projected token states over non-padding positions in `f32`, and L2-normalizes.
/// Prompts such as `task: search result | query: ` from the checkpoint's
/// `config_sentence_transformers.json` are part of the caller's text.
///
/// Only the text encoder is loaded. Vision and audio tensors in the checkpoint are
/// ignored, and texts that tokenize to media placeholder tokens are rejected.
/// The encoder's weights must be floating point; quantized checkpoints are rejected.
///
/// `Embedder` is `!Send` and `!Sync`: create, use and drop it on one OS thread inside
/// the chosen device or stream scope.
pub struct Embedder {
    encoder: Encoder,
    tokenizer: Tokenizer,
    _thread_bound: PhantomData<Rc<()>>,
}

impl Embedder {
    /// Loads `config.json`, the tokenizer and strictly matched safetensors weights
    /// from a local EmbeddingGemma 2 checkpoint directory.
    ///
    /// Every `language_model.*` tensor must fill exactly one encoder parameter with
    /// the expected shape. `vision_tower.*`, `embed_vision.*`, `audio_tower.*` and
    /// `embed_audio.*` tensors are ignored; any other tensor is an error.
    pub fn from_dir(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        let path = path.as_ref();
        let config_path = path.join("config.json");
        let bytes = std::fs::read(&config_path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                LoadError::MissingFile(config_path)
            } else {
                LoadError::Io(error)
            }
        })?;
        let config = config::parse(&bytes)?;
        let tokenizer = Tokenizer::from_dir(path).map_err(|error| match error {
            crate::TokenizerError::MissingFile(missing) => LoadError::MissingFile(missing),
            other => LoadError::Tokenizer(other),
        })?;
        let manifest = WeightManifest::discover(path)?;
        let mut encoder = Encoder::new(config, &manifest)?;
        manifest.load_with(&mut encoder.weight_projection()?, map_key)?;
        Ok(Self {
            encoder,
            tokenizer,
            _thread_bound: PhantomData,
        })
    }

    /// Borrows the checkpoint tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Width of each embedding row.
    pub fn dimension(&self) -> usize {
        self.encoder.config.embedding_dim
    }

    /// Embeds a batch of texts into an evaluated `f32` array of shape
    /// `[texts.len(), dimension()]` whose rows have unit L2 norm.
    ///
    /// Rows are independent of batch composition up to floating-point rounding.
    /// Evaluation happens before returning, so runtime MLX failures surface here as
    /// [`EmbeddingError::Exception`].
    pub fn embed(&self, texts: &[&str]) -> Result<Array, EmbeddingError> {
        if texts.is_empty() {
            return Err(EmbeddingError::EmptyBatch);
        }
        let config = &self.encoder.config;
        let mut encoded = Vec::with_capacity(texts.len());
        for (input, text) in texts.iter().enumerate() {
            let ids = self.tokenizer.encode_with_special_tokens(text, true)?;
            if ids.len() > config.max_positions {
                return Err(EmbeddingError::SequenceTooLong {
                    input,
                    length: ids.len(),
                    limit: config.max_positions,
                });
            }
            for &token_id in &ids {
                let id = u32::from(token_id);
                if id as usize >= config.vocabulary_size {
                    return Err(EmbeddingError::TokenOutOfRange {
                        input,
                        token_id,
                        vocabulary_size: config.vocabulary_size,
                    });
                }
                if config.media_tokens.contains(&id) {
                    return Err(EmbeddingError::MediaToken { input, token_id });
                }
            }
            encoded.push(ids);
        }
        let embeddings = self.encoder.embed(&encoded)?;
        embeddings.eval()?;
        Ok(embeddings)
    }
}

fn map_key(external: &str) -> WeightDisposition {
    if let Some(internal) = external.strip_prefix(TEXT_PREFIX) {
        WeightDisposition::Parameter(ParameterPath::new(internal))
    } else if MEDIA_PREFIXES
        .iter()
        .any(|prefix| external.starts_with(prefix))
    {
        WeightDisposition::Ignore {
            reason: "media towers are not loaded by the text embedder",
        }
    } else {
        WeightDisposition::Reject
    }
}

/// Allocates a placeholder with the checkpoint's floating dtype and the expected shape.
fn slot(manifest: &WeightManifest, internal: &str, shape: &[usize]) -> Result<Array, WeightError> {
    let key = format!("{TEXT_PREFIX}{internal}");
    let entry = manifest
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
    if !matches!(
        entry.dtype,
        Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
    ) {
        return Err(WeightError::UnsupportedDtype {
            key,
            dtype: format!("{:?}", entry.dtype),
        });
    }
    // Config validation bounds every dimension by i32::MAX.
    let shape = shape.iter().map(|&n| n as i32).collect::<Vec<_>>();
    Ok(ops::zeros_dtype(&shape, entry.dtype)?)
}

struct Layer {
    q_proj: Array,
    k_proj: Array,
    v_proj: Array,
    o_proj: Array,
    q_norm: Array,
    k_norm: Array,
    gate_proj: Array,
    up_proj: Array,
    down_proj: Array,
    input_layernorm: Array,
    post_attention_layernorm: Array,
    pre_feedforward_layernorm: Array,
    post_feedforward_layernorm: Array,
    per_layer_input_gate: Array,
    per_layer_projection: Array,
    post_per_layer_input_norm: Array,
    layer_scalar: Array,
}

impl Layer {
    const MATRICES: [&'static str; 9] = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "ple_block.per_layer_input_gate",
        "ple_block.per_layer_projection",
    ];
    const VECTORS: [&'static str; 8] = [
        "self_attn.q_norm",
        "self_attn.k_norm",
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
        "ple_block.post_per_layer_input_norm",
        "layer_scalar",
    ];

    fn new(
        config: &EncoderConfig,
        shape: &LayerShape,
        manifest: &WeightManifest,
        index: usize,
    ) -> Result<Self, WeightError> {
        let hidden = config.hidden_size;
        let queries = shape.heads * shape.head_dim;
        let kv = shape.kv_heads * shape.head_dim;
        let matrix_shapes = [
            [queries, hidden],
            [kv, hidden],
            [kv, hidden],
            [hidden, queries],
            [config.intermediate_size, hidden],
            [config.intermediate_size, hidden],
            [hidden, config.intermediate_size],
            [config.per_layer_width, hidden],
            [hidden, config.per_layer_width],
        ];
        let vector_sizes = [
            shape.head_dim,
            shape.head_dim,
            hidden,
            hidden,
            hidden,
            hidden,
            hidden,
            1,
        ];
        let mut matrices = Self::MATRICES
            .iter()
            .zip(&matrix_shapes)
            .map(|(name, dims)| slot(manifest, &format!("layers.{index}.{name}.weight"), dims));
        let mut vectors = Self::VECTORS.iter().zip(vector_sizes).map(|(name, size)| {
            let internal = if *name == "layer_scalar" {
                format!("layers.{index}.{name}")
            } else {
                format!("layers.{index}.{name}.weight")
            };
            slot(manifest, &internal, &[size])
        });
        let mut matrix = || matrices.next().expect("nine matrix slots");
        let mut vector = || vectors.next().expect("eight vector slots");
        Ok(Self {
            q_proj: matrix()?,
            k_proj: matrix()?,
            v_proj: matrix()?,
            o_proj: matrix()?,
            gate_proj: matrix()?,
            up_proj: matrix()?,
            down_proj: matrix()?,
            per_layer_input_gate: matrix()?,
            per_layer_projection: matrix()?,
            q_norm: vector()?,
            k_norm: vector()?,
            input_layernorm: vector()?,
            post_attention_layernorm: vector()?,
            pre_feedforward_layernorm: vector()?,
            post_feedforward_layernorm: vector()?,
            post_per_layer_input_norm: vector()?,
            layer_scalar: vector()?,
        })
    }

    fn project<'a>(
        &'a mut self,
        index: usize,
        projection: &mut StateProjection<'a>,
    ) -> Result<(), WeightError> {
        let prefix = format!("layers.{index}");
        for (name, array) in [
            ("self_attn.q_proj", &mut self.q_proj),
            ("self_attn.k_proj", &mut self.k_proj),
            ("self_attn.v_proj", &mut self.v_proj),
            ("self_attn.o_proj", &mut self.o_proj),
            ("mlp.gate_proj", &mut self.gate_proj),
            ("mlp.up_proj", &mut self.up_proj),
            ("mlp.down_proj", &mut self.down_proj),
            (
                "ple_block.per_layer_input_gate",
                &mut self.per_layer_input_gate,
            ),
            (
                "ple_block.per_layer_projection",
                &mut self.per_layer_projection,
            ),
            ("self_attn.q_norm", &mut self.q_norm),
            ("self_attn.k_norm", &mut self.k_norm),
            ("input_layernorm", &mut self.input_layernorm),
            (
                "post_attention_layernorm",
                &mut self.post_attention_layernorm,
            ),
            (
                "pre_feedforward_layernorm",
                &mut self.pre_feedforward_layernorm,
            ),
            (
                "post_feedforward_layernorm",
                &mut self.post_feedforward_layernorm,
            ),
            (
                "ple_block.post_per_layer_input_norm",
                &mut self.post_per_layer_input_norm,
            ),
        ] {
            projection.required(format!("{prefix}.{name}.weight"), array)?;
        }
        projection.required(format!("{prefix}.layer_scalar"), &mut self.layer_scalar)?;
        Ok(())
    }
}

/// Rotation tables shared by every layer with the same head width and base.
struct RopeKey {
    head_dim: usize,
    theta: f32,
}

struct Encoder {
    config: EncoderConfig,
    embed_tokens: Array,
    per_layer_model_projection: Array,
    per_layer_projection_norm: Array,
    layers: Vec<Layer>,
    norm: Array,
    embedding_projection: Array,
    /// Distinct RoPE tables and each layer's index into them.
    rope_keys: Vec<RopeKey>,
    layer_rope: Vec<usize>,
}

impl Encoder {
    fn new(config: EncoderConfig, manifest: &WeightManifest) -> Result<Self, WeightError> {
        let hidden = config.hidden_size;
        let layers = config
            .layers
            .iter()
            .enumerate()
            .map(|(index, shape)| Layer::new(&config, shape, manifest, index))
            .collect::<Result<Vec<_>, _>>()?;
        let mut rope_keys: Vec<RopeKey> = Vec::new();
        let mut layer_rope = Vec::with_capacity(config.layers.len());
        for shape in &config.layers {
            let position = rope_keys
                .iter()
                .position(|key| key.head_dim == shape.head_dim && key.theta == shape.rope_theta)
                .unwrap_or_else(|| {
                    rope_keys.push(RopeKey {
                        head_dim: shape.head_dim,
                        theta: shape.rope_theta,
                    });
                    rope_keys.len() - 1
                });
            layer_rope.push(position);
        }
        Ok(Self {
            embed_tokens: slot(
                manifest,
                "embed_tokens.weight",
                &[config.vocabulary_size, hidden],
            )?,
            per_layer_model_projection: slot(
                manifest,
                "ple.per_layer_model_projection.weight",
                &[config.layers.len() * config.per_layer_width, hidden],
            )?,
            per_layer_projection_norm: slot(
                manifest,
                "ple.per_layer_projection_norm.weight",
                &[config.per_layer_width],
            )?,
            norm: slot(manifest, "norm.weight", &[hidden])?,
            embedding_projection: slot(
                manifest,
                "embedding_projection.weight",
                &[config.embedding_dim, hidden],
            )?,
            layers,
            rope_keys,
            layer_rope,
            config,
        })
    }

    fn weight_projection(&mut self) -> Result<StateProjection<'_>, WeightError> {
        let mut projection = StateProjection::new();
        projection.required("embed_tokens.weight", &mut self.embed_tokens)?;
        projection.required(
            "ple.per_layer_model_projection.weight",
            &mut self.per_layer_model_projection,
        )?;
        projection.required(
            "ple.per_layer_projection_norm.weight",
            &mut self.per_layer_projection_norm,
        )?;
        for (index, layer) in self.layers.iter_mut().enumerate() {
            layer.project(index, &mut projection)?;
        }
        projection.required("norm.weight", &mut self.norm)?;
        projection.required(
            "embedding_projection.weight",
            &mut self.embedding_projection,
        )?;
        Ok(projection)
    }

    /// Pools and normalizes a nonempty batch of in-vocabulary token sequences.
    fn embed(&self, batch: &[Vec<TokenId>]) -> Result<Array, mlx_rs::error::Exception> {
        let length = batch.iter().map(Vec::len).max().unwrap_or(0);
        let mut ids = Vec::with_capacity(batch.len() * length);
        let mut valid = Vec::with_capacity(batch.len() * length);
        for sequence in batch {
            ids.extend(sequence.iter().map(|&id| u32::from(id) as i32));
            ids.extend(std::iter::repeat_n(
                self.config.pad_token_id as i32,
                length - sequence.len(),
            ));
            valid.extend(std::iter::repeat_n(1.0f32, sequence.len()));
            valid.extend(std::iter::repeat_n(0.0f32, length - sequence.len()));
        }
        // Lengths are bounded by max_position_embeddings, which fits i32.
        let shape = [batch.len() as i32, length as i32];
        let ids = Array::from_slice(&ids, &shape);
        let valid = Array::from_slice(&valid, &shape);
        let hidden = self.forward(&ids, &valid)?;
        // Mean pooling promotes the projected states to f32 through the mask.
        let mask = valid.expand_dims(-1)?;
        let summed = hidden.multiply(&mask)?.sum_axis(1, false)?;
        let counts = ops::maximum(mask.sum_axis(1, false)?, Array::from_f32(1e-9))?;
        let pooled = summed.divide(&counts)?;
        let norm = pooled.square()?.sum_axis(-1, true)?.sqrt()?;
        pooled.divide(ops::maximum(norm, Array::from_f32(1e-9))?)
    }

    /// Projected token states `[batch, length, embedding_dim]` for padded ids and a
    /// `{0, 1}` validity mask of the same shape.
    fn forward(&self, ids: &Array, valid: &Array) -> Result<Array, mlx_rs::error::Exception> {
        let config = &self.config;
        let eps = config.rms_norm_eps;
        let (batch, length) = (ids.shape()[0], ids.shape()[1]);
        let mut x = self.embed_tokens.take_axis(ids, 0)?;
        let dtype = x.dtype();
        x = x.multiply(scalar((config.hidden_size as f64).sqrt(), dtype)?)?;

        let per_layer = linear(&x, &self.per_layer_model_projection)?
            .multiply(scalar((config.hidden_size as f64).powf(-0.5), dtype)?)?
            .reshape(&[
                batch,
                length,
                config.layers.len() as i32,
                config.per_layer_width as i32,
            ])?;
        let per_layer = fast::rms_norm(per_layer, Some(&self.per_layer_projection_norm), eps)?
            .split_equal(config.layers.len() as i32, 2)?;

        let keys_valid = valid
            .reshape(&[batch, 1, 1, length])?
            .as_dtype(Dtype::Bool)?;
        let positions = ops::arange::<_, i32>(None, length, None)?;
        let distance = positions
            .reshape(&[length, 1])?
            .subtract(positions.reshape(&[1, length])?)?
            .abs()?;
        let sliding_mask =
            keys_valid.logical_and(distance.le(Array::from_int(config.sliding_window as i32))?)?;
        let ropes = self
            .rope_keys
            .iter()
            .map(|key| RopeTable::new(key, length, dtype))
            .collect::<Result<Vec<_>, _>>()?;

        for (index, layer) in self.layers.iter().enumerate() {
            let shape = &config.layers[index];
            let rope = &ropes[self.layer_rope[index]];
            let mask = if shape.sliding {
                &sliding_mask
            } else {
                &keys_valid
            };
            let heads = |array: Array, count: usize| {
                array.reshape(&[batch, length, count as i32, shape.head_dim as i32])
            };
            let h = fast::rms_norm(&x, Some(&layer.input_layernorm), eps)?;
            let queries = fast::rms_norm(
                heads(linear(&h, &layer.q_proj)?, shape.heads)?,
                Some(&layer.q_norm),
                eps,
            )?
            .transpose_axes(&[0, 2, 1, 3])?;
            let keys = fast::rms_norm(
                heads(linear(&h, &layer.k_proj)?, shape.kv_heads)?,
                Some(&layer.k_norm),
                eps,
            )?
            .transpose_axes(&[0, 2, 1, 3])?;
            let values = fast::rms_norm(
                heads(linear(&h, &layer.v_proj)?, shape.kv_heads)?,
                None,
                eps,
            )?
            .transpose_axes(&[0, 2, 1, 3])?;
            let attended = fast::scaled_dot_product_attention(
                rope.apply(&queries)?,
                rope.apply(&keys)?,
                values,
                1.0,
                mask,
                None,
            )?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, length, -1])?;
            x = x.add(fast::rms_norm(
                linear(&attended, &layer.o_proj)?,
                Some(&layer.post_attention_layernorm),
                eps,
            )?)?;

            let h = fast::rms_norm(&x, Some(&layer.pre_feedforward_layernorm), eps)?;
            let gated = nn::gelu_approximate(linear(&h, &layer.gate_proj)?)?
                .multiply(linear(&h, &layer.up_proj)?)?;
            x = x.add(fast::rms_norm(
                linear(&gated, &layer.down_proj)?,
                Some(&layer.post_feedforward_layernorm),
                eps,
            )?)?;

            let per_layer_input = per_layer[index].squeeze_axes(&[2])?;
            let gated = nn::gelu_approximate(linear(&x, &layer.per_layer_input_gate)?)?
                .multiply(per_layer_input)?;
            x = x.add(fast::rms_norm(
                linear(&gated, &layer.per_layer_projection)?,
                Some(&layer.post_per_layer_input_norm),
                eps,
            )?)?;
            x = x.multiply(&layer.layer_scalar)?;
        }
        let x = fast::rms_norm(x, Some(&self.norm), eps)?;
        linear(&x, &self.embedding_projection)
    }
}

/// A Python-float operand: MLX rounds it to the other operand's dtype.
fn scalar(value: f64, dtype: Dtype) -> Result<Array, mlx_rs::error::Exception> {
    Array::from_f32(value as f32).as_dtype(dtype)
}

fn linear(x: &Array, weight: &Array) -> Result<Array, mlx_rs::error::Exception> {
    x.matmul(weight.transpose_axes(&[1, 0])?)
}

/// Rotate-half RoPE with `f32` angles rounded to the activation dtype, matching the
/// reference encoder rather than `fast::rope`'s internal precision.
struct RopeTable {
    cos: Array,
    sin: Array,
}

impl RopeTable {
    fn new(key: &RopeKey, length: i32, dtype: Dtype) -> Result<Self, mlx_rs::error::Exception> {
        let dims = key.head_dim as i32;
        let exponents =
            ops::arange::<_, f32>(None, dims, 2)?.divide(Array::from_f32(dims as f32))?;
        let frequencies =
            Array::from_f32(1.0).divide(Array::from_f32(key.theta).power(&exponents)?)?;
        let angles = ops::arange::<_, f32>(None, length, None)?
            .reshape(&[length, 1])?
            .multiply(frequencies.reshape(&[1, -1])?)?;
        let angles = ops::concatenate(&[&angles, &angles], -1)?;
        Ok(Self {
            cos: angles.cos()?.as_dtype(dtype)?,
            sin: angles.sin()?.as_dtype(dtype)?,
        })
    }

    /// Rotates `[batch, heads, length, head_dim]`.
    fn apply(&self, x: &Array) -> Result<Array, mlx_rs::error::Exception> {
        let halves = x.split_equal(2, -1)?;
        let rotated = ops::concatenate(&[&halves[1].negative()?, &halves[0]], -1)?;
        x.multiply(&self.cos)?.add(rotated.multiply(&self.sin)?)
    }
}
