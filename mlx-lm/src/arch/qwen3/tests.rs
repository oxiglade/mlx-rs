use super::*;
use crate::LayerQuantization;
use serde_json::{json, Value};

fn fixture(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../conformance/mlx-lm/fixtures")
        .join(name)
}

fn parsed(value: Value) -> Result<ParsedConfig, ConfigError> {
    config::parse(&RawConfig::from_bytes(&serde_json::to_vec(&value)?)?)
}

fn base() -> Value {
    json!({ "model_type": "qwen3", "hidden_size": 64, "num_hidden_layers": 2,
        "intermediate_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 16, "vocab_size": 64, "tie_word_embeddings": true })
}

#[test]
fn fixture_configs_and_error_classes() -> Result<(), Box<dyn std::error::Error>> {
    for name in ["qwen3-base", "qwen3-quant4"] {
        let root = fixture(name);
        let input: Value = serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
        let expectations: Value =
            serde_json::from_slice(&std::fs::read(root.join("expectations.json"))?)?;
        let actual = parsed(input.clone())?.config;
        let expected = &expectations["config"]["resolved"];
        let d = &actual.dimensions;
        assert_eq!(d.hidden_size, expected["hidden_size"]);
        assert_eq!(d.layer_count, expected["layer_count"]);
        assert_eq!(d.intermediate_size, expected["intermediate_size"]);
        assert_eq!(d.attention_heads, expected["attention_heads"]);
        assert_eq!(d.kv_heads, expected["kv_heads"]);
        assert_eq!(d.head_dim, expected["head_dim"]);
        assert_eq!(d.vocabulary_size, expected["vocabulary_size"]);
        assert_eq!(
            f64::from(d.rms_norm_epsilon),
            expected["rms_norm_epsilon"].as_f64().unwrap_or_default() as f32 as f64
        );
        assert_eq!(actual.tie_word_embeddings, expected["tie_word_embeddings"]);
        assert_eq!(actual.rope.dimensions, expected["rope"]["dimensions"]);
        assert_eq!(actual.rope.theta as f64, expected["rope"]["theta"]);
        assert_eq!(actual.rope.traditional, expected["rope"]["traditional"]);
        assert_eq!(actual.rope.scaling, crate::RopeScaling::None);
        assert!(actual
            .attention
            .iter()
            .all(|kind| kind == &AttentionKind::Full));
        match &actual.quantization {
            Some(q) => {
                assert_eq!(q.default.bits, expected["quantization"]["bits"]);
                assert_eq!(
                    q.default.group_size.get(),
                    expected["quantization"]["group_size"]
                );
            }
            None => assert!(expected["quantization"].is_null()),
        }
        let errors = expectations["errors"]
            .as_object()
            .ok_or("missing fixture errors")?;
        for (case, error) in errors {
            let mut mutated = input.clone();
            match error["expected_rust_error"]
                .as_str()
                .ok_or("missing error class")?
            {
                "ConfigError::UnsupportedQuantization" => {
                    assert_eq!(case, "bad_quantization_bits");
                    mutated["quantization"] =
                        json!({ "mode": "affine", "group_size": 32, "bits": 5 });
                    assert!(matches!(
                        parsed(mutated),
                        Err(ConfigError::UnsupportedQuantization(_))
                    ));
                }
                "ConfigError::UnsupportedRope" => {
                    assert_eq!(case, "unsupported_rope");
                    mutated["rope_scaling"] = json!({ "rope_type": "unsupported_oracle_rope" });
                    assert!(matches!(
                        parsed(mutated),
                        Err(ConfigError::UnsupportedRope(_))
                    ));
                }
                "ConfigError::UnsupportedArchitecture" => {
                    assert_eq!(case, "unknown_model_type");
                    mutated["model_type"] = json!("unsupported_oracle_model");
                    assert!(matches!(
                        parsed(mutated),
                        Err(ConfigError::UnsupportedArchitecture(_))
                    ));
                }
                "WeightError::MissingShard"
                | "WeightError::DuplicateTensor"
                | "WeightError::ShapeMismatch" => {}
                _ => return Err(format!("unhandled fixture error {case}").into()),
            }
        }
    }
    Ok(())
}

#[test]
fn defaults_and_rejections() -> Result<(), ConfigError> {
    let mut input = base();
    let fields = input
        .as_object_mut()
        .ok_or_else(|| config::invalid("test config", "expected object"))?;
    fields.remove("head_dim");
    fields.remove("num_key_value_heads");
    let config = parsed(input)?.config;
    assert_eq!(config.dimensions.head_dim, 16);
    assert_eq!(config.dimensions.kv_heads, 4);
    for (field, value) in [
        ("num_attention_heads", json!(0)),
        ("num_key_value_heads", json!(3)),
        ("head_dim", json!(15)),
        ("hidden_size", json!(0)),
        ("rms_norm_eps", json!(-1)),
        ("rope_theta", json!(0)),
        ("max_position_embeddings", json!(0)),
        ("head_dim", json!(i32::MAX as u64 + 1)),
    ] {
        let mut input = base();
        input[field] = value;
        assert!(
            matches!(parsed(input), Err(ConfigError::InvalidNumericField { .. })),
            "{field}"
        );
    }
    let mut input = base();
    input["layer_types"] = json!(["full_attention"]);
    assert!(matches!(
        parsed(input),
        Err(ConfigError::InvalidLayerPattern { .. })
    ));
    for (legacy, message) in [
        (json!({"quant_method":"gptq"}), "gptq"),
        (json!({}), "missing quant_method"),
    ] {
        let mut input = base();
        input["quantization_config"] = legacy;
        for null_quantization in [false, true] {
            if null_quantization {
                input["quantization"] = Value::Null;
            }
            assert!(matches!(
                parsed(input.clone()),
                Err(ConfigError::UnsupportedQuantization(reason)) if reason.contains(message)
            ));
        }
    }
    let affine = json!({"bits":4, "group_size":32, "mode":"affine"});
    for legacy in [affine.clone(), json!({"quant_method":"gptq"}), json!(false)] {
        let mut input = base();
        input["quantization"] = affine.clone();
        input["quantization_config"] = legacy;
        let quantization = parsed(input)?
            .config
            .quantization
            .ok_or_else(|| config::invalid("test quantization", "missing"))?;
        assert_eq!(quantization.default.bits, 4);
        assert_eq!(quantization.default.group_size.get(), 32);
        assert!(quantization.layers.is_empty());
    }
    Ok(())
}

#[test]
fn sliding_layers_and_quantization_overrides() -> Result<(), ConfigError> {
    let mut input = base();
    input["layer_types"] = json!(["full_attention", "sliding_attention"]);
    input["sliding_window"] = json!(4);
    input["quantization"] = json!({"bits":4, "group_size":32, "model.embed_tokens":false,
        "model.layers.0.self_attn.q_proj":{"bits":8,"group_size":64}});
    let config = parsed(input.clone())?.config;
    assert!(
        matches!(config.attention.get(1), Some(AttentionKind::Sliding {window}) if window.get() == 4)
    );
    let q = config
        .quantization
        .ok_or_else(|| config::invalid("test quantization", "missing"))?;
    assert_eq!(
        q.layers.get(&crate::ParameterPath::new("embed_tokens")),
        Some(&LayerQuantization::Unquantized)
    );
    input["sliding_window"] = json!(0);
    assert!(matches!(
        parsed(input),
        Err(ConfigError::InvalidNumericField { .. })
    ));
    Ok(())
}

#[test]
fn implicit_layers_are_full() -> Result<(), ConfigError> {
    for max_window_layers in [1, 99] {
        let mut input = base();
        input["use_sliding_window"] = json!(true);
        input["sliding_window"] = json!(4);
        input["max_window_layers"] = json!(max_window_layers);
        assert_eq!(
            parsed(input)?.config.attention,
            vec![AttentionKind::Full; 2]
        );
    }
    let mut input = base();
    input["sliding_window"] = json!(4);
    assert_eq!(
        parsed(input)?.config.attention,
        vec![AttentionKind::Full; 2]
    );
    Ok(())
}

#[test]
fn affine_admission_matches_foundation() -> Result<(), ConfigError> {
    for bits in [0, 1, 2, 3, 4, 5, 6, 7, 8, 16] {
        for group_size in [0, 16, 32, 64, 128, 256] {
            for override_layer in [false, true] {
                let mut input = base();
                input["rms_norm_eps"] = json!(1e-6);
                input["rope_theta"] = json!(1_000_000);
                let affine = json!({"bits": bits, "group_size": group_size});
                input["quantization"] = if override_layer {
                    json!({"bits":4, "group_size":32, "model.layers.0.self_attn.q_proj":affine})
                } else {
                    affine
                };
                let raw = RawConfig::from_bytes(&serde_json::to_vec(&input)?)?;
                let actual = config::parse(&raw);
                let expected = raw.resolve();
                assert_eq!(actual.is_ok(), expected.is_ok(), "{input}: {expected:?}");
                if expected.is_err() {
                    assert!(matches!(
                        actual,
                        Err(ConfigError::UnsupportedQuantization(_))
                    ));
                }
            }
        }
    }
    Ok(())
}

#[test]
fn exact_checkpoint_dispositions() {
    for key in [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.12.self_attn.k_proj.scales",
        "model.layers.1.mlp.down_proj.biases",
        "model.layers.0.mlp.gate_proj.bias",
    ] {
        assert!(
            matches!(Factory.map_safetensors_key(key), WeightDisposition::Parameter(path) if path.as_str() == key.strip_prefix("model.").unwrap_or(key))
        );
    }
    for key in [
        "embed_tokens.weight",
        "model.layers.00.self_attn.q_proj.weight",
        "model.layers.-1.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_norm.scales",
        "model.norm.bias",
        "model.layers.0.mlp.expert.weight",
        "lm_head.bias",
    ] {
        assert!(
            matches!(Factory.map_safetensors_key(key), WeightDisposition::Reject),
            "{key}"
        );
    }
    for key in [
        "model.rotary_emb.inv_freq",
        "model.layers.0.self_attn.rotary_emb.inv_freq",
    ] {
        assert!(
            matches!(Factory.map_safetensors_key(key), WeightDisposition::Ignore { reason } if !reason.is_empty())
        );
    }
    assert!(matches!(
        weights::map_key("lm_head.weight", true),
        WeightDisposition::Ignore { .. }
    ));
    assert!(matches!(
        weights::map_key("lm_head.scales", true),
        WeightDisposition::Parameter(_)
    ));
    assert!(matches!(
        Factory.map_gguf_key("token_embd.weight"),
        WeightDisposition::Reject
    ));
}

#[test]
fn rope_frequency_bands_and_linear_scaling() -> Result<(), ConfigError> {
    let mut input = base();
    input["rope_theta"] = json!(10000);
    input["rope_scaling"] = json!({"rope_type":"llama3", "factor":8, "low_freq_factor":1,
        "high_freq_factor":4, "original_max_position_embeddings":128});
    let config = parsed(input.clone())?.config;
    let frequencies = rope::frequency_table(&config.rope);
    for (index, expected) in [(0, 1.0), (2, 23.391_169), (4, 800.0)] {
        let actual = frequencies
            .get(index)
            .ok_or_else(|| config::invalid("test frequency", "missing"))?;
        assert!((actual - expected).abs() < 2e-5, "{index}: {actual}");
    }
    input["rope_scaling"] = json!({"type":"linear", "factor":2});
    let config = parsed(input.clone())?.config;
    assert_eq!(rope::frequency_table(&config.rope).first(), Some(&2.0));
    input["rope_scaling"] =
        json!({"rope_type":"llama3", "factor":8, "low_freq_factor":4, "high_freq_factor":4});
    assert!(matches!(
        parsed(input),
        Err(ConfigError::InvalidNumericField { .. })
    ));
    Ok(())
}

fn assert_tensor(
    actual: &Array,
    expected: &Array,
    tolerance: &Value,
    name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(actual.shape(), expected.shape(), "{name}: shape");
    assert_eq!(actual.dtype(), expected.dtype(), "{name}: dtype");
    let actual = actual.contiguous()?;
    let expected = expected.contiguous()?;
    actual.eval()?;
    expected.eval()?;
    let actual = actual
        .to_vec_exact::<f32>()
        .map_err(|error| format!("{name}: actual: {error}"))?;
    let expected = expected
        .to_vec_exact::<f32>()
        .map_err(|error| format!("{name}: expected: {error}"))?;
    assert_eq!(actual.len(), expected.len(), "{name}: element count");
    let atol = tolerance["atol"].as_f64().ok_or("missing atol")?;
    let rtol = tolerance["rtol"].as_f64().ok_or("missing rtol")?;
    for (index, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
        let error = f64::from((a - b).abs());
        assert!(
            a.is_finite() && b.is_finite() && error <= atol + rtol * f64::from(b.abs()),
            "{name}[{index}]: actual {a}, expected {b}, error {error}"
        );
    }
    Ok(())
}

fn prefill_fixture(name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let root = fixture(name);
    let raw = RawConfig::from_bytes(&std::fs::read(root.join("config.json"))?)?;
    let manifest = WeightManifest::discover(&root)?;
    let expectations: Value =
        serde_json::from_slice(&std::fs::read(root.join("expectations.json"))?)?;
    let arrays = Array::load_safetensors(root.join("expectations.safetensors"))?;
    let tokens: Vec<u32> = serde_json::from_value(expectations["prefill"]["token_ids"].clone())?;
    let length = tokens.len();
    assert_eq!(length, expectations["prefill"]["T"]);
    let full = arrays
        .get("prefill.full.logits")
        .ok_or("missing full logits")?;
    for chunk in [1, 3, length] {
        let mut model = Factory.build(Factory.parse_config(&raw)?, &manifest)?;
        let mut cache = crate::cache::Cache::new(
            crate::ModelType::new("qwen3"),
            model.cache_layout(),
            &crate::cache::CacheOptions::default(),
            None,
        )?;
        let mut logits = Vec::new();
        for ids in tokens.chunks(chunk) {
            let width = i32::try_from(ids.len())?;
            let input = Array::from_slice(ids, &[1, width]);
            let mut step = cache.step()?;
            let output = model.forward(&input, &mut step)?;
            step.evaluate_and_commit(&[&output])?;
            logits.push(output);
        }
        let actual = ops::concatenate(&logits, 1)?;
        let key = format!("prefill.chunk{chunk}.logits");
        let expected = arrays.get(&key).ok_or("missing chunk logits")?;
        assert_tensor(
            &actual,
            expected,
            &expectations["tolerances"]["logits"],
            &key,
        )?;
        assert_tensor(
            &actual,
            full,
            &expectations["tolerances"]["logits"],
            "prefill.full.logits",
        )?;
        for info in cache.info() {
            let expected = &expectations["cache"]["after_prefill"][info.layer];
            assert_eq!(info.processed_tokens, expected["offset"]);
            assert_eq!(info.retained_positions.start, expected["retained_range"][0]);
            assert_eq!(info.retained_positions.end, expected["retained_range"][1]);
        }
        let views = cache.logical_layers()?;
        assert_eq!(views.len(), model.config().dimensions.layer_count);
        for view in views {
            assert_eq!(view.positions, 0..length);
            for (suffix, actual) in [("keys", &view.keys), ("values", &view.values)] {
                let key = format!("cache.after_prefill.layer{}.{suffix}", view.layer);
                assert_tensor(
                    actual,
                    arrays.get(&key).ok_or("missing cache expectation")?,
                    &expectations["tolerances"]["cache"],
                    &key,
                )?;
            }
        }
    }

    Ok(())
}

#[test]
fn qwen3_base_prefill_and_cache() -> Result<(), Box<dyn std::error::Error>> {
    prefill_fixture("qwen3-base")
}

#[test]
fn qwen3_quant4_prefill_and_cache() -> Result<(), Box<dyn std::error::Error>> {
    prefill_fixture("qwen3-quant4")
}

#[test]
fn fixture_keys_and_shape_rejection() -> Result<(), Box<dyn std::error::Error>> {
    for name in ["qwen3-base", "qwen3-quant4"] {
        let root = fixture(name);
        let data = std::fs::read(root.join("model.safetensors"))?;
        let tensors = safetensors::SafeTensors::deserialize(&data)?;
        let mut internal = std::collections::BTreeSet::new();
        for external in tensors.names() {
            let disposition = Factory.map_safetensors_key(external);
            let WeightDisposition::Parameter(key) = disposition else {
                return Err(format!("fixture tensor {external} was not mapped").into());
            };
            assert!(internal.insert(key));
        }
        let config: Value = serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
        let parsed = parsed(config)?;
        let tensor = tensors.tensor("model.layers.0.self_attn.q_proj.weight")?;
        let expected_shape = tensor.shape().to_vec();
        let mut wrong_shape = expected_shape.clone();
        let rows = wrong_shape
            .first_mut()
            .ok_or("fixture weight has no rows")?;
        *rows = rows.checked_sub(1).ok_or("fixture weight has zero rows")?;
        let target = tempfile::tempdir()?;
        let views = tensors
            .tensors()
            .into_iter()
            .map(|(key, tensor)| {
                let view = if key == "model.layers.0.self_attn.q_proj.weight" {
                    let row_bytes = tensor.data().len() / expected_shape[0];
                    safetensors::tensor::TensorView::new(
                        tensor.dtype(),
                        wrong_shape.clone(),
                        &tensor.data()[..tensor.data().len() - row_bytes],
                    )?
                } else {
                    tensor
                };
                Ok((key, view))
            })
            .collect::<Result<Vec<_>, safetensors::SafeTensorError>>()?;
        safetensors::tensor::serialize_to_file(
            views,
            None,
            &target.path().join("model.safetensors"),
        )?;
        let manifest = WeightManifest::discover(target.path())?;
        assert!(matches!(
            Factory.build(ParsedArchitecture::Qwen3(parsed), &manifest),
            Err(LoadError::Weights(WeightError::ShapeMismatch { .. }))
        ));
    }
    Ok(())
}

#[test]
fn fixture_shard_errors() -> Result<(), Box<dyn std::error::Error>> {
    for name in ["qwen3-base", "qwen3-quant4"] {
        let data = std::fs::read(fixture(name).join("model.safetensors"))?;
        let tensors = safetensors::SafeTensors::deserialize(&data)?;
        let mut names = tensors.names();
        names.sort();
        let midpoint = names.len() / 2;
        let target = tempfile::tempdir()?;
        let shards = [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ];
        let weight_map = names
            .iter()
            .enumerate()
            .map(|(index, key)| (*key, shards[usize::from(index >= midpoint)]))
            .collect::<std::collections::BTreeMap<_, _>>();
        std::fs::write(
            target.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({"weight_map":weight_map}))?,
        )?;
        let write_shard = |shard: &str, keys: &[&str]| -> Result<(), Box<dyn std::error::Error>> {
            let views = keys
                .iter()
                .map(|key| Ok((*key, tensors.tensor(key)?)))
                .collect::<Result<Vec<_>, safetensors::SafeTensorError>>()?;
            safetensors::tensor::serialize_to_file(views, None, &target.path().join(shard))?;
            Ok(())
        };
        write_shard(shards[0], &names[..midpoint])?;
        write_shard(shards[1], &names[midpoint..])?;
        assert_eq!(
            WeightManifest::discover(target.path())?.tensors.len(),
            names.len()
        );
        std::fs::remove_file(target.path().join(shards[1]))?;
        assert!(matches!(
            WeightManifest::discover(target.path()),
            Err(WeightError::MissingShard(_))
        ));
        let mut duplicate = names[midpoint..].to_vec();
        duplicate.push(names[0]);
        write_shard(shards[1], &duplicate)?;
        assert!(matches!(
            WeightManifest::discover(target.path()),
            Err(WeightError::DuplicateTensor(_))
        ));
    }
    Ok(())
}

#[test]
fn causal_and_sliding_mask_with_prefix_positions() -> Result<(), Box<dyn std::error::Error>> {
    let positions = [0, 1, 6, 7, 8, 9];
    let full = attention_mask(8, 2, &positions, &AttentionKind::Full)?;
    full.eval()?;
    assert_eq!(
        full.to_vec_exact::<bool>()?,
        [true, true, true, true, true, false, true, true, true, true, true, true]
    );
    let window = std::num::NonZeroUsize::new(3).ok_or("zero test window")?;
    let sliding = attention_mask(8, 2, &positions, &AttentionKind::Sliding { window })?;
    sliding.eval()?;
    assert_eq!(
        sliding.to_vec_exact::<bool>()?,
        [false, false, true, true, true, false, false, false, false, true, true, true]
    );
    Ok(())
}
