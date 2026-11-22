use super::*;
use crate::{
    AttentionKind, ModelType, ParameterPath, RopeConfig, RopeScaling, TransformerDimensions,
};
use mlx_rs::{fast, ops, Dtype};
use std::{collections::HashMap, num::NonZeroUsize};
mod forward;

pub(super) struct Factory;

/// The `text_config` values the forward pass hard-codes.
const PROFILE: &str = r#"{"attn_res_normalize_values": true, "d_rel": 16,
"final_logit_softcapping": 30.0, "head_dim": 128, "hidden_size": 2304, "intermediate_size": 1152,
"local_layer_ids": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22],
"log_scaling_alpha": 0.1, "log_scaling_n_floor": 8192,
"loop_config": {"core_end": 18, "core_start": 6, "gate_rank": 64, "is_varloop": false, "num_passes": 3},
"mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "model_type": "inkling2_loop_text",
"n_layers_per_attn_res_block": 6, "n_routed_experts": 128, "n_shared_experts": 0,
"num_attention_heads": 24, "num_experts_per_tok": 8, "num_hidden_layers": 24,
"num_key_value_heads": 8, "o_bias": false, "q_bias": false,
"qk_l2_clip_threshold": 22.627416997969522, "rel_extent": 1024, "rel_l2_clip_threshold": 8.0,
"rms_norm_eps": 1e-06, "route_scale": 8.0, "sconv_kernel_size": 4, "sliding_window_size": 1024,
"swa_head_dim": 128, "swa_num_attention_heads": 24, "swa_num_key_value_heads": 8,
"unpadded_vocab_size": 200058, "use_gate_bias": true, "use_value_norm_no_gamma": true,
"vocab_size": 201024}"#;

fn map_key(key: &str) -> WeightDisposition {
    if key.starts_with("model.llm.") {
        WeightDisposition::Parameter(ParameterPath::new(key))
    } else if key.starts_with("model.visual.") {
        WeightDisposition::Ignore {
            reason: "text-only decoder",
        }
    } else {
        WeightDisposition::Reject
    }
}

impl ArchitectureFactory for Factory {
    fn parse_config(&self, raw: &RawConfig) -> Result<ParsedArchitecture, ConfigError> {
        let t = raw
            .fields
            .get("text_config")
            .ok_or_else(|| ConfigError::MissingField {
                field: "text_config".into(),
            })?;
        let profile: serde_json::Value = serde_json::from_str(PROFILE)?;
        for (key, value) in profile.as_object().unwrap() {
            if &t[key] != value {
                return Err(ConfigError::UnsupportedArchitecture(format!(
                    "unsupported text_config.{key}"
                )));
            }
        }
        let config = Config {
            model_type: ModelType::new(raw.model_type.as_str()),
            dimensions: TransformerDimensions {
                hidden_size: 2304,
                layer_count: 24,
                intermediate_size: 1152,
                attention_heads: 24,
                kv_heads: 8,
                head_dim: 128,
                vocabulary_size: 200058,
                max_positions: Some(serde_json::from_value(t["model_max_length"].clone())?),
                rms_norm_epsilon: 1e-6,
            },
            // Inkling has no rotary embedding; `Config` requires one, so this is inert.
            rope: RopeConfig {
                dimensions: 128,
                theta: 10_000.,
                traditional: false,
                scaling: RopeScaling::None,
            },
            attention: (0..24)
                .map(|i| {
                    if i % 6 == 5 {
                        AttentionKind::Full
                    } else {
                        AttentionKind::Sliding {
                            window: NonZeroUsize::new(1024).unwrap(),
                        }
                    }
                })
                .collect(),
            tie_word_embeddings: false,
            attention_bias: false,
            mlp_bias: false,
            quantization: None,
        };
        config.validate()?;
        Ok(ParsedArchitecture::Hybrid(config))
    }
    fn build(
        &self,
        parsed: ParsedArchitecture,
        manifest: &WeightManifest,
    ) -> Result<Box<dyn DecoderModel>, LoadError> {
        let ParsedArchitecture::Hybrid(config) = parsed else {
            return Err(
                ConfigError::UnsupportedArchitecture("expected hybrid config".into()).into(),
            );
        };
        // Sliding windows are recurrent state: a rotating cache would turn off lookahead decoding.
        let layout = forward::visits()
            .flat_map(|i| {
                let window = (i % 6 != 5).then(|| (vec![1, 8, 0, 128], Dtype::Bfloat16));
                let conv = |width| Some((vec![1, 3, width], Dtype::Float32));
                [window, conv(1024), conv(2304)].map(|state| LayerCacheSpec {
                    attention: match state {
                        None => AttentionKind::Full,
                        Some(_) => AttentionKind::Recurrent,
                    },
                    batch_size: 1,
                    kv_heads: 8,
                    head_dim: 128,
                    dtype: Dtype::Bfloat16,
                    recurrent_state: state.map(|state| [state.clone(), state]),
                })
            })
            .collect();
        let mut decoder = Decoder {
            config,
            layout,
            weights: forward::Weights(
                slots()?,
                fast::MetalKernel::new(
                    "inkling_banded_mask_v2",
                    &["rel", "proj", "kshape"],
                    &["out"],
                    include_str!("mask.metal"),
                    Default::default(),
                )?,
            ),
        };
        manifest.load_with(&mut decoder.weight_projection()?, map_key)?;
        decoder.weights.prepare()?;
        Ok(Box::new(decoder))
    }
    #[cfg(test)]
    fn map_safetensors_key(&self, key: &str) -> WeightDisposition {
        map_key(key)
    }
    fn map_gguf_key(&self, _: &str) -> WeightDisposition {
        WeightDisposition::Reject
    }
}

struct Decoder {
    config: Config,
    layout: Vec<LayerCacheSpec>,
    weights: forward::Weights,
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
        self.weights.forward(tokens, cache)
    }
    fn weight_projection(&mut self) -> Result<StateProjection<'_>, WeightError> {
        let mut p = StateProjection::new();
        for (k, v) in &mut self.weights.0 {
            p.required(k.as_str(), v)?;
        }
        Ok(p)
    }
}

fn slots() -> Result<HashMap<String, Array>, LoadError> {
    use Dtype::{Bfloat16 as B, Float32 as F, Uint8 as U};
    let mut m = HashMap::new();
    let mut slot = |name: String, shape: &[i32], dtype: Dtype| -> Result<(), LoadError> {
        m.insert(name, ops::zeros_dtype(shape, dtype)?);
        Ok(())
    };
    let root = "model.llm";
    for name in ["embed", "unembed"] {
        slot(format!("{root}.{name}.weight"), &[201024, 2304], B)?;
    }
    for name in ["embed_norm", "norm"] {
        slot(format!("{root}.{name}.weight"), &[2304], B)?;
    }
    for name in ["queries", "kv_queries"] {
        slot(
            format!("{root}.middle_cycle_cross_pass_{name}"),
            &[24, 2304],
            F,
        )?;
    }
    slot(format!("{root}.attn_res.output_query"), &[2304], F)?;
    for i in 0..24 {
        let global = i % 6 == 5;
        if i > 0 {
            let shape: &[i32] = if global { &[3, 2304] } else { &[2304] };
            slot(format!("{root}.attn_res.attn_queries.{}", i - 1), shape, F)?;
        }
        slot(format!("{root}.attn_res.mlp_queries.{i}"), &[2304], F)?;
        let p = format!("{root}.layers.{i}");
        for (name, shape) in [
            ("attn_sink_proj.weight", [24, 2304]),
            ("gate_attn.weight", [3072, 2304]),
            ("q_l2_clip.scale", [24, 128]),
            ("rel_l2_clip.scale", [24, 16]),
            ("rel_logits_proj.proj", [16, 1024]),
            ("wk_dv.weight", [1024, 2304]),
            ("wo_ud.weight", [2304, 3072]),
            ("wq_du.weight", [3072, 2304]),
            ("wr_du.weight", [384, 2304]),
            ("wv_dv.weight", [1024, 2304]),
        ] {
            slot(format!("{p}.attn.{name}"), &shape, B)?;
        }
        for side in ["k", "v"] {
            if global {
                slot(format!("{p}.attn.{side}_sconv.static_w"), &[4, 1024], B)?;
                slot(format!("{p}.attn.{side}_sconv.weight_proj"), &[32, 2304], B)?;
            } else {
                slot(format!("{p}.attn.{side}_sconv.weight"), &[1024, 1, 4], B)?;
            }
        }
        let norms: &[&str] = if global {
            &["q_input_norm", "k_input_norm", "v_input_norm", "mlp_norm"]
        } else {
            &["attn_norm", "mlp_norm"]
        };
        for name in norms {
            slot(format!("{p}.{name}.weight"), &[2304], B)?;
        }
        for name in ["attn_sconv", "mlp_sconv"] {
            slot(format!("{p}.{name}.weight"), &[2304, 1, 4], B)?;
        }
        slot(format!("{p}.mlp.gate.weight"), &[128, 2304], B)?;
        slot(format!("{p}.mlp.gate.bias"), &[128], F)?;
        if (6..18).contains(&i) {
            for part in ["attn", "k", "v", "mlp"] {
                for (kind, shape) in [
                    ("down", [64, 4608]),
                    ("forget", [2304, 64]),
                    ("gain", [2304, 64]),
                ] {
                    slot(format!("{p}.controllers.{part}.{kind}_weight"), &shape, B)?;
                }
            }
        }
        // The first twenty layers pack experts as NVFP4, the last four as MXFP8.
        let nvfp4 = i < 20;
        for (name, bytes) in [("w13_weight", 1152), ("w2_weight", 576)] {
            let e = format!("{p}.mlp.experts.{name}");
            if nvfp4 {
                slot(e.clone(), &[128, 2304, bytes], U)?;
                slot(format!("{e}_scale"), &[128, 2304, bytes / 8], U)?;
                slot(format!("{e}_amax"), &[128], F)?;
                slot(format!("{e}_outer_scale_denominator"), &[1], F)?;
            } else {
                slot(e.clone(), &[128, 2304, bytes * 2], U)?;
                slot(format!("{e}_scale"), &[128, 2304, bytes / 16], U)?;
            }
        }
    }
    Ok(m)
}
