use super::*;
use serde::de::DeserializeOwned;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../conformance/mlx-lm/fixtures")
}

fn read_json(path: &Path) -> Result<Value, ConfigError> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

fn parse(value: &Value) -> Result<Config, ConfigError> {
    RawConfig::from_bytes(&serde_json::to_vec(value)?)?.resolve()
}

fn expected<T: DeserializeOwned>(value: &Value, field: &str) -> Result<T, ConfigError> {
    let value = value.get(field).ok_or_else(|| ConfigError::MissingField {
        field: field.into(),
    })?;
    Ok(serde_json::from_value(value.clone())?)
}

fn base() -> Result<Value, ConfigError> {
    read_json(&fixtures().join("llama-base/config.json"))
}

#[test]
fn fixture_resolved_configs() -> Result<(), ConfigError> {
    let mut count = 0;
    for entry in std::fs::read_dir(fixtures())? {
        let path = entry?.path();
        // GGUF fixtures carry no config.json; they are covered by the gguf module tests.
        if !path.is_dir() || !path.join("config.json").exists() {
            continue;
        }
        let input = read_json(&path.join("config.json"))?;
        let config = parse(&input)?;
        let oracle = read_json(&path.join("expectations.json"))?;
        let resolved = &oracle["config"]["resolved"];
        let d = &config.dimensions;
        assert_eq!(d.hidden_size, expected::<usize>(resolved, "hidden_size")?);
        assert_eq!(d.layer_count, expected::<usize>(resolved, "layer_count")?);
        assert_eq!(
            d.intermediate_size,
            expected::<usize>(resolved, "intermediate_size")?
        );
        assert_eq!(
            d.attention_heads,
            expected::<usize>(resolved, "attention_heads")?
        );
        assert_eq!(d.kv_heads, expected::<usize>(resolved, "kv_heads")?);
        assert_eq!(d.head_dim, expected::<usize>(resolved, "head_dim")?);
        assert_eq!(
            d.vocabulary_size,
            expected::<usize>(resolved, "vocabulary_size")?
        );
        assert_eq!(
            d.rms_norm_epsilon,
            expected::<f32>(resolved, "rms_norm_epsilon")?
        );
        assert_eq!(
            d.max_positions,
            Some(expected::<usize>(&input, "max_position_embeddings")?)
        );
        assert_eq!(
            config.model_type.as_str(),
            expected::<String>(&input, "model_type")?
        );
        assert_eq!(
            config.tie_word_embeddings,
            expected::<bool>(resolved, "tie_word_embeddings")?
        );
        assert!(!config.attention_bias);
        assert!(!config.mlp_bias);
        let attention: Vec<_> = config
            .attention
            .iter()
            .map(|kind| match kind {
                AttentionKind::Full => json!("full"),
                AttentionKind::Sliding { window } => json!({"sliding": window.get()}),
            })
            .collect();
        assert_eq!(json!(attention), resolved["attention_kinds"]);
        assert_eq!(
            config.rope.dimensions,
            expected::<usize>(&resolved["rope"], "dimensions")?
        );
        assert_eq!(
            config.rope.theta,
            expected::<f32>(&resolved["rope"], "theta")?
        );
        assert_eq!(
            config.rope.traditional,
            expected::<bool>(&resolved["rope"], "traditional")?
        );
        let scaling = match config.rope.scaling {
            RopeScaling::None => Value::Null,
            RopeScaling::Linear { factor } => json!({"rope_type": "linear", "factor": factor}),
            RopeScaling::Llama3 {
                factor,
                low_frequency_factor,
                high_frequency_factor,
                original_max_positions,
            } => json!({
                "rope_type": "llama3", "factor": factor, "low_freq_factor": low_frequency_factor,
                "high_freq_factor": high_frequency_factor, "original_max_position_embeddings": original_max_positions,
            }),
        };
        assert_eq!(scaling, resolved["rope"]["scaling"]);
        let quantization = config.quantization.map(|quantization| {
            assert!(quantization.layers.is_empty());
            json!({"group_size": quantization.default.group_size.get(), "bits": quantization.default.bits})
        });
        assert_eq!(json!(quantization), resolved["quantization"]);
        count += 1;
    }
    assert_eq!(count, 6);
    Ok(())
}

#[test]
fn fixture_config_rejections() -> Result<(), ConfigError> {
    let mut count = 0;
    for entry in std::fs::read_dir(fixtures())? {
        let path = entry?.path();
        // GGUF fixtures carry no config.json; they are covered by the gguf module tests.
        if !path.is_dir() || !path.join("config.json").exists() {
            continue;
        }
        let input = read_json(&path.join("config.json"))?;
        let oracle = read_json(&path.join("expectations.json"))?;
        let cases: BTreeMap<String, Value> = expected(&oracle, "errors")?;
        for (name, case) in cases {
            let class: String = expected(&case, "expected_rust_error")?;
            if !class.starts_with("ConfigError::") {
                continue;
            }
            let mut mutated = input.clone();
            let mutation: String = expected(&case, "mutation")?;
            match mutation.as_str() {
                "set quantization to affine, group_size 32, bits 5" => {
                    mutated["quantization"] =
                        json!({"mode": "affine", "group_size": 32, "bits": 5});
                }
                "set model_type to unsupported_oracle_model" => {
                    mutated["model_type"] = json!("unsupported_oracle_model");
                }
                "set rope_scaling.rope_type to unsupported_oracle_rope" => {
                    if mutated["rope_scaling"].is_null() {
                        mutated["rope_scaling"] = json!({});
                    }
                    mutated["rope_scaling"]["rope_type"] = json!("unsupported_oracle_rope");
                }
                _ => {
                    return Err(ConfigError::UnsupportedArchitecture(format!(
                        "unhandled fixture mutation {name}: {mutation}"
                    )))
                }
            }
            let result = parse(&mutated);
            let actual = match &result {
                Err(ConfigError::UnsupportedQuantization(_)) => {
                    "ConfigError::UnsupportedQuantization"
                }
                Err(ConfigError::UnsupportedArchitecture(_)) => {
                    "ConfigError::UnsupportedArchitecture"
                }
                Err(ConfigError::UnsupportedRope(_)) => "ConfigError::UnsupportedRope",
                _ => "unexpected result",
            };
            assert_eq!(actual, class, "{} {name}: {result:?}", path.display());
            count += 1;
        }
    }
    assert_eq!(count, 18);
    Ok(())
}

#[test]
fn defaults_unknown_keys_and_explicit_head_width() -> Result<(), ConfigError> {
    let mut raw = RawConfig::from_bytes(&serde_json::to_vec(&base()?)?)?;
    for key in [
        "head_dim",
        "num_key_value_heads",
        "rope_theta",
        "max_position_embeddings",
        "tie_word_embeddings",
    ] {
        raw.fields.remove(key);
    }
    raw.fields
        .insert("future_metadata".into(), json!({"anything": [1, 2]}));
    let config = raw.resolve()?;
    assert_eq!(config.dimensions.head_dim, 16);
    assert_eq!(config.dimensions.kv_heads, 4);
    assert_eq!(config.dimensions.max_positions, None);
    assert_eq!(config.rope.theta, 10_000.0);
    assert!(config.tie_word_embeddings);
    raw.model_type = "qwen3".into();
    assert!(
        matches!(raw.resolve(), Err(ConfigError::MissingField { field }) if field == "rope_theta")
    );
    raw.fields.insert("rope_theta".into(), json!(1_000_000.0));
    raw.fields
        .insert("tie_word_embeddings".into(), json!(false));
    raw.fields.insert("head_dim".into(), json!(32));
    assert_eq!(raw.resolve()?.dimensions.head_dim, 32);
    assert_eq!(raw.resolve()?.rope.theta, 1_000_000.0);
    assert!(
        matches!(RawConfig::from_bytes(b"{}"), Err(ConfigError::MissingField { field }) if field == "model_type")
    );
    raw.fields.remove("hidden_size");
    assert!(
        matches!(raw.resolve(), Err(ConfigError::MissingField { field }) if field == "hidden_size")
    );
    Ok(())
}

#[test]
fn invalid_sizes_and_numeric_fields() -> Result<(), ConfigError> {
    for (field, value) in [
        ("hidden_size", json!(63)),
        ("num_attention_heads", json!(0)),
        ("num_key_value_heads", json!(3)),
        ("head_dim", json!(3)),
        ("head_dim", json!(0)),
        ("num_hidden_layers", json!(0)),
        ("num_hidden_layers", json!(usize::MAX)),
        ("intermediate_size", json!(-1)),
        ("vocab_size", json!(1.5)),
        ("max_position_embeddings", json!(0)),
        ("rms_norm_eps", json!(0)),
        ("rope_theta", json!(1e100)),
        ("rope_theta", json!(1e-100)),
        ("rope_theta", json!("10000")),
        ("sliding_window", json!(0)),
    ] {
        let mut value_json = base()?;
        value_json[field] = value;
        assert!(
            matches!(
                parse(&value_json),
                Err(ConfigError::InvalidNumericField { .. })
            ),
            "{field}"
        );
    }
    let mut config = parse(&base()?)?;
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0, 0.0] {
        config.rope.theta = value;
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidNumericField { .. })
        ));
    }
    Ok(())
}

#[test]
fn attention_patterns_require_valid_windows_and_lengths() -> Result<(), ConfigError> {
    let mut input = base()?;
    input["layer_types"] = json!(["sliding_attention", "full_attention"]);
    assert!(
        matches!(parse(&input), Err(ConfigError::MissingField { field }) if field == "sliding_window")
    );
    input["sliding_window"] = json!(7);
    assert_eq!(
        parse(&input)?.attention,
        vec![
            AttentionKind::Sliding {
                window: NonZeroUsize::new(7).ok_or_else(|| invalid("window", "zero"))?
            },
            AttentionKind::Full
        ]
    );
    input["layer_types"] = json!(["full_attention"]);
    assert!(matches!(
        parse(&input),
        Err(ConfigError::InvalidLayerPattern {
            expected: 2,
            actual: 1
        })
    ));
    input["layer_types"] = json!(["unknown", "full_attention"]);
    assert!(matches!(
        parse(&input),
        Err(ConfigError::UnsupportedArchitecture(_))
    ));
    input["layer_types"] = Value::Null;
    assert!(parse(&input)?
        .attention
        .iter()
        .all(|kind| matches!(kind, AttentionKind::Full)));
    Ok(())
}

#[test]
fn rope_modes_and_parameters() -> Result<(), ConfigError> {
    let mut input = base()?;
    for kind in ["yarn", "longrope", "proportional", "mrope", "unknown"] {
        input["rope_scaling"] = json!({"rope_type": kind});
        assert!(matches!(
            parse(&input),
            Err(ConfigError::UnsupportedRope(_))
        ));
    }
    input["rope_scaling"] = json!({"rope_type": "default"});
    assert_eq!(parse(&input)?.rope.scaling, RopeScaling::None);
    input["rope_scaling"] = json!({"factor": 2.0});
    assert_eq!(parse(&input)?.rope.scaling, RopeScaling::None);
    input["rope_scaling"] = json!({"type": "linear", "rope_type": "yarn", "factor": 2.0});
    assert_eq!(
        parse(&input)?.rope.scaling,
        RopeScaling::Linear { factor: 2.0 }
    );
    input["rope_scaling"] = json!({"type": "linear", "factor": 2.0});
    assert_eq!(
        parse(&input)?.rope.scaling,
        RopeScaling::Linear { factor: 2.0 }
    );
    input["rope_scaling"]["factor"] = json!(0);
    assert!(matches!(
        parse(&input),
        Err(ConfigError::InvalidNumericField { .. })
    ));
    let llama3 = json!({"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 64});
    for (field, value) in [
        ("factor", json!(-1)),
        ("low_freq_factor", json!(0)),
        ("high_freq_factor", json!(1)),
        ("original_max_position_embeddings", json!(0)),
    ] {
        input["rope_scaling"] = llama3.clone();
        input["rope_scaling"][field] = value;
        assert!(
            matches!(parse(&input), Err(ConfigError::InvalidNumericField { .. })),
            "{field}"
        );
    }
    let mut config = parse(&base()?)?;
    config.rope.scaling = RopeScaling::Linear { factor: f32::NAN };
    assert!(matches!(
        config.validate(),
        Err(ConfigError::InvalidNumericField { .. })
    ));
    Ok(())
}

#[test]
fn affine_overrides_and_rejections() -> Result<(), ConfigError> {
    let mut quantized = read_json(&fixtures().join("llama-quant4/config.json"))?;
    for legacy in [
        quantized["quantization"].clone(),
        json!({"quant_method": "awq"}),
    ] {
        quantized["quantization_config"] = legacy;
        assert_eq!(
            parse(&quantized)?.quantization,
            Some(QuantizationConfig {
                default: AffineQuantization {
                    group_size: NonZeroUsize::new(32)
                        .ok_or_else(|| invalid("group_size", "zero"))?,
                    bits: 4,
                },
                layers: BTreeMap::new(),
            })
        );
    }
    let mut input = base()?;
    input["quantization"] = json!({
        "group_size": 32, "bits": 4,
        "model.layers.0.self_attn.q_proj": {"group_size": 64, "bits": 8, "mode": "affine"},
        "model.embed_tokens": false, "lm_head": true,
    });
    let config = parse(&input)?;
    let q = config
        .quantization
        .ok_or_else(|| invalid("quantization", "absent"))?;
    assert_eq!(
        q.layers.get(&ParameterPath::new("model.embed_tokens")),
        Some(&LayerQuantization::Unquantized)
    );
    assert_eq!(
        q.layers.get(&ParameterPath::new("lm_head")),
        Some(&LayerQuantization::Affine(q.default.clone()))
    );
    assert!(
        matches!(q.layers.get(&ParameterPath::new("model.layers.0.self_attn.q_proj")), Some(LayerQuantization::Affine(affine)) if affine.bits == 8 && affine.group_size.get() == 64)
    );
    for bits in [2, 4, 8] {
        for group_size in [32, 64, 128] {
            input["quantization"] = json!({"bits": bits, "group_size": group_size});
            parse(&input)?;
        }
    }
    for quantization in [
        json!({"bits": 5, "group_size": 32}),
        json!({"bits": 4, "group_size": 16}),
        json!({"bits": 4, "group_size": 32, "mode": "mxfp4"}),
        json!({"bits": 4, "group_size": 32, "lm_head": {"bits": 5, "group_size": 32}}),
    ] {
        input["quantization"] = quantization;
        assert!(matches!(
            parse(&input),
            Err(ConfigError::UnsupportedQuantization(_))
        ));
    }
    input["quantization"] = Value::Null;
    for method in [
        "mxfp4",
        "bitnet",
        "awq",
        "gptq",
        "compressed-tensors",
        "activation",
    ] {
        input["quantization_config"] = json!({"quant_method": method});
        assert!(matches!(
            parse(&input),
            Err(ConfigError::UnsupportedQuantization(message))
                if message == format!("legacy quantization_config quant_method {method}")
        ));
    }
    for legacy in [json!({}), json!({"quant_method": 4})] {
        input["quantization_config"] = legacy;
        assert!(matches!(
            parse(&input),
            Err(ConfigError::UnsupportedQuantization(message))
                if message == "legacy quantization_config"
        ));
    }
    let mut raw = RawConfig::from_bytes(&serde_json::to_vec(&input)?)?;
    raw.fields.remove("quantization");
    assert!(matches!(
        raw.resolve(),
        Err(ConfigError::UnsupportedQuantization(message))
            if message == "legacy quantization_config"
    ));
    raw.fields.insert("quantization_config".into(), Value::Null);
    assert_eq!(raw.resolve()?.quantization, None);
    Ok(())
}
