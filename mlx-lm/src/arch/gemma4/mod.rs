use super::{ArchitectureFactory, DecoderModel, ParsedArchitecture};
use crate::{
    cache::{CacheStep, LayerCacheSpec},
    config::RawConfig,
    weights::{WeightDisposition, WeightManifest},
    AttentionKind, CacheError, Config, ConfigError, InferenceError, LoadError, WeightError,
};
use mlx_rs::{
    fast, nn,
    ops::{self, indexing::TryIndexOp},
    utils::StateProjection,
    Array, Dtype,
};

mod config;
mod rope;
#[cfg(test)]
mod tests;
mod weights;

pub(crate) use config::{parse as parse_config, ParsedConfig};
use weights::{Embedding, Linear, Slots};

pub(crate) struct Factory;

impl ArchitectureFactory for Factory {
    fn parse_config(&self, raw: &RawConfig) -> Result<ParsedArchitecture, ConfigError> {
        Ok(ParsedArchitecture::Gemma4(config::parse(raw)?))
    }
    fn build(
        &self,
        parsed: ParsedArchitecture,
        weights: &WeightManifest,
    ) -> Result<Box<dyn DecoderModel>, LoadError> {
        let ParsedArchitecture::Gemma4(parsed) = parsed else {
            return Err(ConfigError::UnsupportedArchitecture(
                "Gemma4 factory received another architecture".into(),
            )
            .into());
        };
        Ok(Box::new(build_decoder(parsed, weights)?))
    }
    #[cfg(test)]
    fn map_safetensors_key(&self, external: &str) -> WeightDisposition {
        weights::map_key(external, false, None)
    }
    fn map_gguf_key(&self, external: &str) -> WeightDisposition {
        super::gguf::map(external, true)
    }
}

fn build_decoder(parsed: ParsedConfig, weights: &WeightManifest) -> Result<Decoder, LoadError> {
    let first_kv_shared =
        (parsed.num_kv_shared_layers > 0).then_some(parsed.first_kv_shared_layer_idx);
    let mut decoder = Decoder::new(parsed, weights)?;
    let tied = decoder.config.tie_word_embeddings;
    weights.load_with(&mut decoder.weight_projection()?, |key| {
        weights::map_key(key, tied, first_kv_shared)
    })?;
    Ok(decoder)
}

struct PerLayerInputGate {
    gate: Linear,
    projection: Linear,
    norm: Array,
}

struct PerLayerInputModel {
    embed_tokens_per_layer: Embedding,
    model_projection: Linear,
    projection_norm: Array,
    embed_scale: f32,
    projection_scale: f32,
    input_scale: f32,
    pli_dim: i32,
}

struct Layer {
    q_proj: Linear,
    k_proj: Option<Linear>,
    v_proj: Option<Linear>,
    o_proj: Linear,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    input_norm: Array,
    post_attention_norm: Array,
    pre_feedforward_norm: Array,
    post_feedforward_norm: Array,
    q_norm: Array,
    k_norm: Option<Array>,
    pli_gate: Option<PerLayerInputGate>,
    layer_scalar: Option<Array>,
    rope: rope::LayerRope,
    attention: AttentionKind,
    head_dim: i32,
    kv_heads: i32,
    has_kv: bool,
    use_k_eq_v: bool,
}

impl Layer {
    fn new(slots: &Slots<'_>, index: usize) -> Result<Self, LoadError> {
        let prefix = format!("layers.{index}");
        let parsed = slots.parsed;
        let d = &parsed.config.dimensions;
        let attention_bias = parsed.config.attention_bias;
        let mlp_bias = parsed.config.mlp_bias;
        let head_dim_usize = parsed.layer_head_dims[index];
        let kv_heads_usize = parsed.layer_kv_heads[index];
        let attention = parsed.config.attention[index].clone();
        let is_sliding = matches!(attention, AttentionKind::Sliding { .. });
        let has_kv = index < parsed.first_kv_shared_layer_idx;
        let use_k_eq_v = parsed.attention_k_eq_v && !is_sliding;

        let (theta, factor) = if is_sliding {
            (
                parsed.sliding_rope_theta,
                parsed.sliding_partial_rotary_factor,
            )
        } else {
            (parsed.full_rope_theta, parsed.full_partial_rotary_factor)
        };
        let rope = rope::LayerRope::new(
            head_dim_usize,
            factor,
            theta,
            parsed.config.rope.traditional,
        )?;

        let k_proj = if has_kv {
            Some(slots.linear(&format!("{prefix}.self_attn.k_proj"), attention_bias)?)
        } else {
            None
        };
        let v_proj = if has_kv && !use_k_eq_v {
            Some(slots.linear(&format!("{prefix}.self_attn.v_proj"), attention_bias)?)
        } else {
            None
        };
        let k_norm = if has_kv {
            Some(slots.norm(&format!("{prefix}.self_attn.k_norm"), head_dim_usize)?)
        } else {
            None
        };

        let pli_gate = if parsed.hidden_size_per_layer_input > 0 {
            Some(PerLayerInputGate {
                gate: slots.linear(&format!("{prefix}.per_layer_input_gate"), false)?,
                projection: slots.linear(&format!("{prefix}.per_layer_projection"), false)?,
                norm: slots.norm(
                    &format!("{prefix}.post_per_layer_input_norm"),
                    d.hidden_size,
                )?,
            })
        } else {
            None
        };

        Ok(Self {
            q_proj: slots.linear(&format!("{prefix}.self_attn.q_proj"), attention_bias)?,
            k_proj,
            v_proj,
            o_proj: slots.linear(&format!("{prefix}.self_attn.o_proj"), attention_bias)?,
            gate_proj: slots.linear(&format!("{prefix}.mlp.gate_proj"), mlp_bias)?,
            up_proj: slots.linear(&format!("{prefix}.mlp.up_proj"), mlp_bias)?,
            down_proj: slots.linear(&format!("{prefix}.mlp.down_proj"), mlp_bias)?,
            input_norm: slots.norm(&format!("{prefix}.input_layernorm"), d.hidden_size)?,
            post_attention_norm: slots
                .norm(&format!("{prefix}.post_attention_layernorm"), d.hidden_size)?,
            pre_feedforward_norm: slots.norm(
                &format!("{prefix}.pre_feedforward_layernorm"),
                d.hidden_size,
            )?,
            post_feedforward_norm: slots.norm(
                &format!("{prefix}.post_feedforward_layernorm"),
                d.hidden_size,
            )?,
            q_norm: slots.norm(&format!("{prefix}.self_attn.q_norm"), head_dim_usize)?,
            k_norm,
            pli_gate,
            layer_scalar: slots.optional_layer_scalar(index)?,
            rope,
            attention,
            head_dim: config::dimension("head_dim", head_dim_usize)?,
            kv_heads: config::dimension("num_key_value_heads", kv_heads_usize)?,
            has_kv,
            use_k_eq_v,
        })
    }

    fn project<'a>(
        &'a mut self,
        index: usize,
        projection: &mut StateProjection<'a>,
    ) -> Result<(), WeightError> {
        let prefix = format!("layers.{index}");
        self.q_proj
            .project(&format!("{prefix}.self_attn.q_proj"), projection)?;
        if let Some(k_proj) = &mut self.k_proj {
            k_proj.project(&format!("{prefix}.self_attn.k_proj"), projection)?;
        }
        if let Some(v_proj) = &mut self.v_proj {
            v_proj.project(&format!("{prefix}.self_attn.v_proj"), projection)?;
        }
        self.o_proj
            .project(&format!("{prefix}.self_attn.o_proj"), projection)?;
        self.gate_proj
            .project(&format!("{prefix}.mlp.gate_proj"), projection)?;
        self.up_proj
            .project(&format!("{prefix}.mlp.up_proj"), projection)?;
        self.down_proj
            .project(&format!("{prefix}.mlp.down_proj"), projection)?;

        for (suffix, array) in [
            ("input_layernorm", &mut self.input_norm),
            ("post_attention_layernorm", &mut self.post_attention_norm),
            ("pre_feedforward_layernorm", &mut self.pre_feedforward_norm),
            (
                "post_feedforward_layernorm",
                &mut self.post_feedforward_norm,
            ),
            ("self_attn.q_norm", &mut self.q_norm),
        ] {
            projection.required(format!("{prefix}.{suffix}.weight"), array)?;
        }
        if let Some(k_norm) = &mut self.k_norm {
            projection.required(format!("{prefix}.self_attn.k_norm.weight"), k_norm)?;
        }
        if let Some(pli) = &mut self.pli_gate {
            pli.gate
                .project(&format!("{prefix}.per_layer_input_gate"), projection)?;
            pli.projection
                .project(&format!("{prefix}.per_layer_projection"), projection)?;
            projection.required(
                format!("{prefix}.post_per_layer_input_norm.weight"),
                &mut pli.norm,
            )?;
        }
        projection.optional(format!("{prefix}.layer_scalar"), &mut self.layer_scalar)?;
        Ok(())
    }
}

struct Decoder {
    config: Config,
    layout: Vec<LayerCacheSpec>,
    embedding: Embedding,
    embed_scale: f32,
    pli_model: Option<PerLayerInputModel>,
    layers: Vec<Layer>,
    previous_kvs: Vec<usize>,
    norm: Array,
    lm_head: Option<Linear>,
    final_logit_softcapping: Option<f32>,
    heads: i32,
}

impl Decoder {
    fn new(parsed: ParsedConfig, manifest: &WeightManifest) -> Result<Self, LoadError> {
        let slots = Slots {
            parsed: &parsed,
            manifest,
        };
        let embedding = Embedding::new(&slots, "embed_tokens")?;
        let d = &parsed.config.dimensions;
        let embed_scale = (d.hidden_size as f32).sqrt();

        let pli_model = if parsed.hidden_size_per_layer_input > 0 {
            let pli_dim_usize = parsed.hidden_size_per_layer_input;
            Some(PerLayerInputModel {
                embed_tokens_per_layer: Embedding::new(&slots, "embed_tokens_per_layer")?,
                model_projection: slots.linear("per_layer_model_projection", false)?,
                projection_norm: slots.norm("per_layer_projection_norm", pli_dim_usize)?,
                embed_scale: (pli_dim_usize as f32).sqrt(),
                projection_scale: (d.hidden_size as f32).powf(-0.5),
                input_scale: 2.0_f32.powf(-0.5),
                pli_dim: config::dimension("hidden_size_per_layer_input", pli_dim_usize)?,
            })
        } else {
            None
        };

        let layers = (0..d.layer_count)
            .map(|i| Layer::new(&slots, i))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = slots.norm("norm", d.hidden_size)?;
        let lm_head = if parsed.config.tie_word_embeddings {
            None
        } else {
            Some(slots.linear("lm_head", false)?)
        };

        let layout = layers
            .iter()
            .take(parsed.first_kv_shared_layer_idx)
            .enumerate()
            .map(|(idx, layer)| {
                let k_dtype = layer
                    .k_proj
                    .as_ref()
                    .map(Linear::dtype)
                    .unwrap_or_else(|| layer.q_proj.dtype());
                LayerCacheSpec {
                    attention: layer.attention.clone(),
                    batch_size: 1,
                    kv_heads: parsed.layer_kv_heads[idx],
                    head_dim: parsed.layer_head_dims[idx],
                    dtype: Dtype::from_promoting_types(layer.input_norm.dtype(), k_dtype),
                }
            })
            .collect();

        let heads = config::dimension("num_attention_heads", d.attention_heads)?;
        Ok(Self {
            config: parsed.config,
            layout,
            embedding,
            embed_scale,
            pli_model,
            layers,
            previous_kvs: parsed.previous_kvs,
            norm,
            lm_head,
            final_logit_softcapping: parsed.final_logit_softcapping,
            heads,
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
        self.embedding.project("embed_tokens", &mut projection)?;
        if let Some(pli) = &mut self.pli_model {
            pli.embed_tokens_per_layer
                .project("embed_tokens_per_layer", &mut projection)?;
            pli.model_projection
                .project("per_layer_model_projection", &mut projection)?;
            projection.required("per_layer_projection_norm.weight", &mut pli.projection_norm)?;
        }
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

        let eps = self.config.dimensions.rms_norm_epsilon;
        let raw_embed = self.embedding.forward(tokens)?;
        let scale_arr = Array::from_slice(&[self.embed_scale], &[]).as_dtype(raw_embed.dtype())?;
        let mut x = raw_embed.multiply(&scale_arr)?;

        let per_layer_inputs = if let Some(pli) = &mut self.pli_model {
            let n_layers = i32::try_from(self.layers.len())
                .map_err(|_| CacheError::InvalidState("layer count exceeds MLX range".into()))?;
            let raw_pli = pli.embed_tokens_per_layer.forward(tokens)?;
            let emb_scale = Array::from_slice(&[pli.embed_scale], &[]).as_dtype(raw_pli.dtype())?;
            let raw_pli =
                raw_pli
                    .multiply(&emb_scale)?
                    .reshape(&[batch, length, n_layers, pli.pli_dim])?;

            let projected = pli.model_projection.forward(&x)?;
            let proj_scale =
                Array::from_slice(&[pli.projection_scale], &[]).as_dtype(projected.dtype())?;
            let projected = projected.multiply(&proj_scale)?.reshape(&[
                batch,
                length,
                n_layers,
                pli.pli_dim,
            ])?;
            let projected = fast::rms_norm(projected, Some(&pli.projection_norm), eps)?;

            let combined = projected.add(&raw_pli)?;
            let in_scale = Array::from_slice(&[pli.input_scale], &[]).as_dtype(combined.dtype())?;
            let combined = combined.multiply(&in_scale)?;
            let mut slices = Vec::with_capacity(self.layers.len());
            for i in 0..n_layers {
                slices.push(combined.try_index((.., .., i, ..))?);
            }
            Some(slices)
        } else {
            None
        };

        type SharedKvEntry = (Array, Array, usize, Vec<usize>);
        let mut intermediates: Vec<Option<SharedKvEntry>> = vec![None; self.layers.len()];

        for (index, layer) in self.layers.iter_mut().enumerate() {
            let h = fast::rms_norm(&x, Some(&layer.input_norm), eps)?;
            let queries =
                layer
                    .q_proj
                    .forward(&h)?
                    .reshape(&[batch, length, self.heads, layer.head_dim])?;
            let queries =
                fast::rms_norm(queries, Some(&layer.q_norm), eps)?.transpose_axes(&[0, 2, 1, 3])?;

            let (keys, values, offset, positions) = if layer.has_kv {
                let offset = cache.info(index)?.processed_tokens;
                let k_proj = layer.k_proj.as_mut().ok_or_else(|| {
                    CacheError::InvalidState("missing k_proj on KV-owning layer".into())
                })?;
                let k_norm = layer.k_norm.as_ref().ok_or_else(|| {
                    CacheError::InvalidState("missing k_norm on KV-owning layer".into())
                })?;
                let keys_raw = k_proj.forward(&h)?.reshape(&[
                    batch,
                    length,
                    layer.kv_heads,
                    layer.head_dim,
                ])?;
                let values_raw = if layer.use_k_eq_v {
                    keys_raw.clone()
                } else {
                    let v_proj = layer.v_proj.as_mut().ok_or_else(|| {
                        CacheError::InvalidState("missing v_proj on non-k_eq_v layer".into())
                    })?;
                    v_proj
                        .forward(&h)?
                        .reshape(&[batch, length, layer.kv_heads, layer.head_dim])?
                };

                let keys =
                    fast::rms_norm(keys_raw, Some(k_norm), eps)?.transpose_axes(&[0, 2, 1, 3])?;
                let keys = layer.rope.apply(&keys, offset)?;

                // Gemma 4 v_norm is RMSNormNoScale (no learnable weight)
                let values =
                    fast::rms_norm(values_raw, None, eps)?.transpose_axes(&[0, 2, 1, 3])?;

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
                intermediates[index] =
                    Some((keys.clone(), values.clone(), offset, positions.clone()));
                (keys, values, offset, positions)
            } else {
                let prev_idx = self.previous_kvs[index];
                intermediates[prev_idx].clone().ok_or_else(|| {
                    CacheError::InvalidState(
                        "KV-shared layer referenced unpopulated ancestor layer".into(),
                    )
                })?
            };

            let queries = layer.rope.apply(&queries, offset)?;
            let mask = attention_mask(offset, length, &positions, &layer.attention)?;
            // Gemma 4 uses scale = 1.0 because Q, K, and V are RMS-normalized
            let attended =
                fast::scaled_dot_product_attention(&queries, keys, values, 1.0, &mask, None)?
                    .transpose_axes(&[0, 2, 1, 3])?
                    .reshape(&[batch, length, -1])?;

            let attn_out = layer.o_proj.forward(&attended)?;
            let attn_out = fast::rms_norm(attn_out, Some(&layer.post_attention_norm), eps)?;
            x = x.add(attn_out)?;

            let ffn_in = fast::rms_norm(&x, Some(&layer.pre_feedforward_norm), eps)?;
            let gate = nn::gelu_approximate(layer.gate_proj.forward(&ffn_in)?)?;
            let up = layer.up_proj.forward(&ffn_in)?;
            let ffn_out = layer.down_proj.forward(&gate.multiply(up)?)?;
            let ffn_out = fast::rms_norm(ffn_out, Some(&layer.post_feedforward_norm), eps)?;
            x = x.add(ffn_out)?;

            if let (Some(pli_gate), Some(plis)) = (&mut layer.pli_gate, &per_layer_inputs) {
                if let Some(pli_slice) = plis.get(index) {
                    let g = nn::gelu_approximate(pli_gate.gate.forward(&x)?)?;
                    let g = g.multiply(pli_slice)?;
                    let g = pli_gate.projection.forward(&g)?;
                    let g = fast::rms_norm(g, Some(&pli_gate.norm), eps)?;
                    x = x.add(g)?;
                }
            }

            if let Some(scalar) = &layer.layer_scalar {
                let s = scalar.as_dtype(x.dtype())?;
                x = x.multiply(&s)?;
            }
        }

        let x = fast::rms_norm(x, Some(&self.norm), eps)?;
        let mut logits = match &mut self.lm_head {
            Some(head) => head.forward(&x)?,
            None => self.embedding.as_linear(&x)?,
        };
        if let Some(softcap) = self.final_logit_softcapping {
            let cap = Array::from_slice(&[softcap], &[]).as_dtype(logits.dtype())?;
            logits = ops::tanh(&logits.divide(&cap)?)?.multiply(&cap)?;
        }
        Ok(logits)
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
                .map_err(|_| CacheError::InvalidState("sliding window exceeds MLX range".into()))?;
            causal.logical_and(queries.subtract(&keys)?.lt(Array::from_int(window))?)?
        }
    })
}
