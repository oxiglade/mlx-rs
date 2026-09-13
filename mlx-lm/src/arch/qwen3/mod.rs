use super::{ArchitectureFactory, DecoderModel, ParsedArchitecture};
use crate::{
    cache::{CacheStep, LayerCacheSpec},
    config::RawConfig,
    weights::{WeightDisposition, WeightManifest},
    AttentionKind, CacheError, Config, ConfigError, InferenceError, LoadError, WeightError,
};
use mlx_rs::{fast, nn, ops, utils::StateProjection, Array, Dtype};

mod config;
mod rope;
#[cfg(test)]
mod tests;
mod weights;
pub(crate) use config::ParsedConfig;
use weights::{Embedding, Linear, Slots};

pub(crate) struct Factory;
impl ArchitectureFactory for Factory {
    fn parse_config(&self, raw: &RawConfig) -> Result<ParsedArchitecture, ConfigError> {
        Ok(ParsedArchitecture::Qwen3(config::parse(raw)?))
    }
    fn build(
        &self,
        parsed: ParsedArchitecture,
        weights: &WeightManifest,
    ) -> Result<Box<dyn DecoderModel>, LoadError> {
        let ParsedArchitecture::Qwen3(parsed) = parsed else {
            return Err(ConfigError::UnsupportedArchitecture(
                "Qwen3 factory received another architecture".into(),
            )
            .into());
        };
        Ok(Box::new(build_decoder(parsed, weights)?))
    }
    #[cfg(test)]
    fn map_safetensors_key(&self, external: &str) -> WeightDisposition {
        weights::map_key(external, false)
    }
    fn map_gguf_key(&self, _external: &str) -> WeightDisposition {
        WeightDisposition::Reject
    }
}

fn build_decoder(parsed: ParsedConfig, weights: &WeightManifest) -> Result<Decoder, LoadError> {
    let mut decoder = Decoder::new(parsed, weights)?;
    let tied = decoder.config.tie_word_embeddings;
    weights.load_with(&mut decoder.weight_projection()?, |key| {
        weights::map_key(key, tied)
    })?;
    Ok(decoder)
}

struct Layer {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    input_norm: Array,
    post_norm: Array,
    q_norm: Array,
    k_norm: Array,
}

impl Layer {
    fn new(slots: &Slots<'_>, index: usize) -> Result<Self, LoadError> {
        let prefix = format!("layers.{index}");
        let d = &slots.config.dimensions;
        let attention_bias = slots.config.attention_bias;
        let mlp_bias = slots.config.mlp_bias;
        Ok(Self {
            q_proj: slots.linear(&format!("{prefix}.self_attn.q_proj"), attention_bias)?,
            k_proj: slots.linear(&format!("{prefix}.self_attn.k_proj"), attention_bias)?,
            v_proj: slots.linear(&format!("{prefix}.self_attn.v_proj"), attention_bias)?,
            o_proj: slots.linear(&format!("{prefix}.self_attn.o_proj"), attention_bias)?,
            gate_proj: slots.linear(&format!("{prefix}.mlp.gate_proj"), mlp_bias)?,
            up_proj: slots.linear(&format!("{prefix}.mlp.up_proj"), mlp_bias)?,
            down_proj: slots.linear(&format!("{prefix}.mlp.down_proj"), mlp_bias)?,
            input_norm: slots.norm(&format!("{prefix}.input_layernorm"), d.hidden_size)?,
            post_norm: slots.norm(&format!("{prefix}.post_attention_layernorm"), d.hidden_size)?,
            q_norm: slots.norm(&format!("{prefix}.self_attn.q_norm"), d.head_dim)?,
            k_norm: slots.norm(&format!("{prefix}.self_attn.k_norm"), d.head_dim)?,
        })
    }
    fn project<'a>(
        &'a mut self,
        index: usize,
        projection: &mut StateProjection<'a>,
    ) -> Result<(), WeightError> {
        let prefix = format!("layers.{index}");
        for (suffix, linear) in [
            ("self_attn.q_proj", &mut self.q_proj),
            ("self_attn.k_proj", &mut self.k_proj),
            ("self_attn.v_proj", &mut self.v_proj),
            ("self_attn.o_proj", &mut self.o_proj),
            ("mlp.gate_proj", &mut self.gate_proj),
            ("mlp.up_proj", &mut self.up_proj),
            ("mlp.down_proj", &mut self.down_proj),
        ] {
            linear.project(&format!("{prefix}.{suffix}"), projection)?;
        }
        for (suffix, array) in [
            ("input_layernorm", &mut self.input_norm),
            ("post_attention_layernorm", &mut self.post_norm),
            ("self_attn.q_norm", &mut self.q_norm),
            ("self_attn.k_norm", &mut self.k_norm),
        ] {
            projection.required(format!("{prefix}.{suffix}.weight"), array)?;
        }
        Ok(())
    }
}

struct Decoder {
    config: Config,
    layout: Vec<LayerCacheSpec>,
    embedding: Embedding,
    layers: Vec<Layer>,
    norm: Array,
    lm_head: Option<Linear>,
    rope: rope::Rope,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
}
impl Decoder {
    fn new(parsed: ParsedConfig, manifest: &WeightManifest) -> Result<Self, LoadError> {
        let config = parsed.config;
        let slots = Slots {
            config: &config,
            manifest,
        };
        let embedding = Embedding::new(&slots)?;
        let d = &config.dimensions;
        let layers = (0..d.layer_count)
            .map(|i| Layer::new(&slots, i))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = slots.norm("norm", d.hidden_size)?;
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(slots.linear("lm_head", false)?)
        };
        let layout = config
            .attention
            .iter()
            .map(|attention| LayerCacheSpec {
                attention: attention.clone(),
                batch_size: 1,
                kv_heads: d.kv_heads,
                head_dim: d.head_dim,
                dtype: embedding.dtype(),
            })
            .collect();
        let rope = rope::Rope::new(&config.rope)?;
        let heads = config::dimension("num_attention_heads", d.attention_heads)?;
        let kv_heads = config::dimension("num_key_value_heads", d.kv_heads)?;
        let head_dim = config::dimension("head_dim", d.head_dim)?;
        Ok(Self {
            config,
            layout,
            embedding,
            layers,
            norm,
            lm_head,
            rope,
            heads,
            kv_heads,
            head_dim,
        })
    }
}

impl DecoderModel for Decoder {
    fn config(&self) -> &Config {
        &self.config
    }
    fn cache_layout(&self) -> &[LayerCacheSpec] {
        &self.layout
    }
    fn weight_projection(&mut self) -> Result<StateProjection<'_>, WeightError> {
        let mut projection = StateProjection::new();
        self.embedding.project(&mut projection)?;
        for (index, layer) in self.layers.iter_mut().enumerate() {
            layer.project(index, &mut projection)?;
        }
        projection.required("norm.weight", &mut self.norm)?;
        if let Some(head) = &mut self.lm_head {
            head.project("lm_head", &mut projection)?;
        }
        Ok(projection)
    }
    fn forward(
        &mut self,
        tokens: &Array,
        cache: &mut CacheStep<'_>,
    ) -> Result<Array, InferenceError> {
        let &[batch, length] = tokens.shape() else {
            return Err(
                CacheError::InvalidState("tokens must have shape [batch, length]".into()).into(),
            );
        };
        if batch <= 0 || length <= 0 || !matches!(tokens.dtype(), Dtype::Int32 | Dtype::Uint32) {
            return Err(
                CacheError::InvalidState("tokens must be nonempty int32 or uint32".into()).into(),
            );
        }
        let mut x = self.embedding.forward(tokens)?;
        let eps = self.config.dimensions.rms_norm_epsilon;
        for (index, (layer, spec)) in self.layers.iter_mut().zip(&self.layout).enumerate() {
            let offset = cache.info(index)?.processed_tokens;
            let h = fast::rms_norm(&x, Some(&layer.input_norm), eps)?;
            let queries = layer
                .q_proj
                .forward(&h)?
                .reshape(&[batch, length, self.heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let keys = layer
                .k_proj
                .forward(&h)?
                .reshape(&[batch, length, self.kv_heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let values = layer
                .v_proj
                .forward(&h)?
                .reshape(&[batch, length, self.kv_heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let queries = self
                .rope
                .apply(&fast::rms_norm(queries, Some(&layer.q_norm), eps)?, offset)?;
            let keys = self
                .rope
                .apply(&fast::rms_norm(keys, Some(&layer.k_norm), eps)?, offset)?;
            let (keys, values) = cache.update_and_fetch(index, keys, values)?;
            let info = cache.info(index)?;
            let positions = info
                .retained_prefix
                .chain(info.retained_positions)
                .collect::<Vec<_>>();
            let position_count = i32::try_from(positions.len())
                .map_err(|_| CacheError::InvalidState("too many key positions".into()))?;
            if keys.shape().get(2) != Some(&position_count) || keys.shape() != values.shape() {
                return Err(CacheError::InvalidState(
                    "cache positions and K/V shape disagree".into(),
                )
                .into());
            }
            let mask = attention_mask(offset, length, &positions, &spec.attention)?;
            let attended = fast::scaled_dot_product_attention(
                &queries,
                keys,
                values,
                (self.head_dim as f32).sqrt().recip(),
                &mask,
                None,
            )?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, length, -1])?;
            x = x.add(layer.o_proj.forward(&attended)?)?;
            let h = fast::rms_norm(&x, Some(&layer.post_norm), eps)?;
            let gated =
                nn::silu(layer.gate_proj.forward(&h)?)?.multiply(layer.up_proj.forward(&h)?)?;
            x = x.add(layer.down_proj.forward(&gated)?)?;
        }
        let x = fast::rms_norm(x, Some(&self.norm), eps)?;
        match &mut self.lm_head {
            Some(head) => head.forward(&x),
            None => self.embedding.as_linear(&x),
        }
    }
}

fn attention_mask(
    offset: usize,
    length: i32,
    positions: &[usize],
    attention: &AttentionKind,
) -> Result<Array, InferenceError> {
    let offset = i32::try_from(offset)
        .map_err(|_| CacheError::InvalidState("query offset exceeds MLX range".into()))?;
    let end = offset
        .checked_add(length)
        .ok_or_else(|| CacheError::InvalidState("query position overflow".into()))?;
    let keys = positions
        .iter()
        .map(|&p| {
            i32::try_from(p)
                .map_err(|_| CacheError::InvalidState("key position exceeds MLX range".into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let count = i32::try_from(keys.len())
        .map_err(|_| CacheError::InvalidState("too many key positions".into()))?;
    let keys = Array::from_slice(&keys, &[1, count]);
    let queries = ops::arange::<_, i32>(Some(offset), end, None)?.reshape(&[length, 1])?;
    let causal = queries.ge(&keys)?;
    Ok(match attention {
        AttentionKind::Full => causal,
        AttentionKind::Sliding { window } => {
            let window = i32::try_from(window.get())
                .map_err(|_| CacheError::InvalidState("window exceeds MLX range".into()))?;
            causal.logical_and(queries.subtract(&keys)?.lt(Array::from_int(window))?)?
        }
    })
}
