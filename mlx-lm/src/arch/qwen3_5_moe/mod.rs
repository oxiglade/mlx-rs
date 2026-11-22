use super::*;
use crate::{
    AttentionKind, ModelType, ParameterPath, RopeConfig, RopeScaling, TransformerDimensions,
};
use mlx_rs::{fast, ops, Dtype};
use serde_json::json;
use std::collections::HashMap;
mod forward;

pub(super) struct Factory;

fn map_key(key: &str) -> WeightDisposition {
    if key.starts_with("language_model.") {
        WeightDisposition::Parameter(ParameterPath::new(key))
    } else if key.starts_with("vision_tower.") {
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
        let profile = json!({"model_type":"qwen3_5_moe_text","hidden_size":2048,"num_hidden_layers":40,"head_dim":256,"num_attention_heads":16,"num_key_value_heads":2,"num_experts":256,"num_experts_per_tok":8,"moe_intermediate_size":512,"shared_expert_intermediate_size":512,"full_attention_interval":4,"linear_conv_kernel_dim":4,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_num_key_heads":16,"linear_num_value_heads":32,"rms_norm_eps":1e-6,"attn_output_gate":true,"attention_bias":false,"dtype":"bfloat16","hidden_act":"silu","mamba_ssm_dtype":"float32","tie_word_embeddings":false,"partial_rotary_factor":0.25,"vocab_size":248320});
        for (key, value) in profile.as_object().unwrap() {
            if &t[key] != value {
                return Err(ConfigError::UnsupportedArchitecture(format!(
                    "unsupported text_config.{key}"
                )));
            }
        }
        let rope = &t["rope_parameters"];
        if rope["rope_theta"].as_f64() != Some(10_000_000.)
            || rope["partial_rotary_factor"] != 0.25
            || rope["rope_type"] != "default"
        {
            return Err(ConfigError::UnsupportedArchitecture(
                "unsupported hybrid RoPE".into(),
            ));
        }
        let kinds: Vec<_> = (0..40)
            .map(|i| {
                if i % 4 == 3 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect();
        if t["layer_types"] != json!(kinds) {
            return Err(ConfigError::UnsupportedArchitecture(
                "unsupported hybrid layer pattern".into(),
            ));
        }
        let quant = crate::config::parse_quantization(raw.fields.get("quantization").ok_or_else(
            || ConfigError::MissingField {
                field: "quantization".into(),
            },
        )?)?;
        if quant.default.bits != 4
            || quant.default.group_size.get() != 64
            || quant.layers.len() != 80
        {
            return Err(ConfigError::UnsupportedQuantization(
                "expected stock affine 4-bit layout".into(),
            ));
        }
        for i in 0..40 {
            for name in ["gate", "shared_expert_gate"] {
                let key = ParameterPath::new(format!("language_model.model.layers.{i}.mlp.{name}"));
                if !matches!(quant.layers.get(&key), Some(crate::LayerQuantization::Affine(a)) if a.bits==8 && a.group_size.get()==64)
                {
                    return Err(ConfigError::UnsupportedQuantization(key.as_str().into()));
                }
            }
        }
        let config = Config {
            model_type: ModelType::new(raw.model_type.as_str()),
            dimensions: TransformerDimensions {
                hidden_size: 2048,
                layer_count: 40,
                intermediate_size: 512,
                attention_heads: 16,
                kv_heads: 2,
                head_dim: 256,
                vocabulary_size: 248320,
                max_positions: Some(serde_json::from_value(
                    t["max_position_embeddings"].clone(),
                )?),
                rms_norm_epsilon: 1e-6,
            },
            rope: RopeConfig {
                dimensions: 64,
                theta: 10_000_000.,
                traditional: false,
                scaling: RopeScaling::None,
            },
            attention: kinds
                .iter()
                .map(|k| {
                    if *k == "full_attention" {
                        AttentionKind::Full
                    } else {
                        AttentionKind::Recurrent
                    }
                })
                .collect(),
            tie_word_embeddings: false,
            attention_bias: false,
            mlp_bias: false,
            quantization: Some(quant),
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
        let kernel = |name, inputs: &[&str], outputs: &[&str], src| {
            fast::MetalKernel::new(
                name,
                inputs,
                outputs,
                src,
                fast::MetalKernelOptions {
                    ensure_row_contiguous: true,
                    ..Default::default()
                },
            )
        };
        let layout = config
            .attention
            .iter()
            .map(|attention| LayerCacheSpec {
                attention: attention.clone(),
                batch_size: 1,
                kv_heads: 2,
                head_dim: 256,
                dtype: Dtype::Bfloat16,
                recurrent_state: (*attention == AttentionKind::Recurrent).then(|| {
                    [
                        (vec![1, 3, 8192], Dtype::Bfloat16),
                        (vec![32, 128, 128], Dtype::Float32),
                    ]
                }),
            })
            .collect();
        let mut decoder = Decoder {
            config,
            layout,
            weights: forward::Weights(
                slots()?,
                kernel(
                    "routed_sum",
                    &["projected", "order", "scores"],
                    &["result"],
                    include_str!("routed_sum.metal"),
                )?,
            ),
            decode: kernel(
                "hybrid_delta",
                &["q", "k", "v", "g", "beta", "state"],
                &["out", "next"],
                include_str!("delta.metal"),
            )?,
            prefill: kernel(
                "hybrid_packed",
                &["q", "k", "v", "g", "beta", "state_in"],
                &["y", "state_out"],
                include_str!("prefill.metal"),
            )?,
        };
        manifest.load_with(&mut decoder.weight_projection()?, map_key)?;
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
    decode: fast::MetalKernel,
    prefill: fast::MetalKernel,
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
        self.weights.forward(
            tokens,
            cache,
            if tokens.size() == 1 {
                &self.decode
            } else {
                &self.prefill
            },
        )
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
    let mut m = HashMap::new();
    let mut slot = |name: String, shape: &[i32], dtype: Dtype| -> Result<(), LoadError> {
        m.insert(name, ops::zeros_dtype(shape, dtype)?);
        Ok(())
    };
    let mut matrices = vec![
        (
            "language_model.model.embed_tokens".into(),
            0,
            248320,
            2048,
            4,
        ),
        ("language_model.lm_head".into(), 0, 248320, 2048, 4),
    ];
    slot(
        "language_model.model.norm.weight".into(),
        &[2048],
        Dtype::Bfloat16,
    )?;
    for i in 0..40 {
        let p = format!("language_model.model.layers.{i}");
        for n in ["input_layernorm", "post_attention_layernorm"] {
            slot(format!("{p}.{n}.weight"), &[2048], Dtype::Bfloat16)?;
        }
        for (n, e, o, d, bits) in [
            ("gate", 0, 256, 2048, 8),
            ("shared_expert_gate", 0, 1, 2048, 8),
            ("shared_expert.gate_proj", 0, 512, 2048, 4),
            ("shared_expert.up_proj", 0, 512, 2048, 4),
            ("shared_expert.down_proj", 0, 2048, 512, 4),
            ("switch_mlp.gate_proj", 256, 512, 2048, 4),
            ("switch_mlp.up_proj", 256, 512, 2048, 4),
            ("switch_mlp.down_proj", 256, 2048, 512, 4),
        ] {
            matrices.push((format!("{p}.mlp.{n}"), e, o, d, bits));
        }
        if i % 4 == 3 {
            for n in ["q", "k"] {
                slot(
                    format!("{p}.self_attn.{n}_norm.weight"),
                    &[256],
                    Dtype::Bfloat16,
                )?;
            }
            for (n, o, d) in [
                ("q", 8192, 2048),
                ("k", 512, 2048),
                ("v", 512, 2048),
                ("o", 2048, 4096),
            ] {
                matrices.push((format!("{p}.self_attn.{n}_proj"), 0, o, d, 4));
            }
        } else {
            for (n, s) in [
                ("A_log", &[32][..]),
                ("dt_bias", &[32]),
                ("norm.weight", &[128]),
                ("conv1d.weight", &[8192, 4, 1]),
            ] {
                slot(format!("{p}.linear_attn.{n}"), s, Dtype::Bfloat16)?;
            }
            for (n, o, d) in [
                ("in_proj_a", 32, 2048),
                ("in_proj_b", 32, 2048),
                ("in_proj_qkv", 8192, 2048),
                ("in_proj_z", 4096, 2048),
                ("out_proj", 2048, 4096),
            ] {
                matrices.push((format!("{p}.linear_attn.{n}"), 0, o, d, 4));
            }
        }
    }
    for (name, e, o, d, bits) in matrices {
        for (s, w, ty) in [
            ("weight", d * bits / 32, Dtype::Uint32),
            ("scales", d / 64, Dtype::Bfloat16),
            ("biases", d / 64, Dtype::Bfloat16),
        ] {
            let dims = [e, o, w];
            slot(
                format!("{name}.{s}"),
                if e == 0 { &dims[1..] } else { &dims },
                ty,
            )?;
        }
    }
    Ok(m)
}
