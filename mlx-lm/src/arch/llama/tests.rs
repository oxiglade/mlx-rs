use super::*;
use crate::{AttentionKind, RopeScaling};
use serde_json::{json, Value};

fn fixture(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../conformance/mlx-lm/fixtures")
        .join(name)
}

fn raw(name: &str) -> Result<RawConfig, crate::ConfigError> {
    RawConfig::from_bytes(&std::fs::read(fixture(name).join("config.json"))?)
}

#[test]
fn pure_fixture_configs_match_resolved_facts() -> Result<(), Box<dyn std::error::Error>> {
    for name in [
        "llama-base",
        "llama-sliding",
        "llama-quant4",
        "llama-sharded",
    ] {
        let parsed = config::parse(&raw(name)?)?;
        let expected: Value =
            serde_json::from_slice(&std::fs::read(fixture(name).join("expectations.json"))?)?;
        let expected = &expected["config"]["resolved"];
        let config = parsed.config;
        for (name, actual) in [
            ("hidden_size", config.dimensions.hidden_size),
            ("layer_count", config.dimensions.layer_count),
            ("intermediate_size", config.dimensions.intermediate_size),
            ("attention_heads", config.dimensions.attention_heads),
            ("kv_heads", config.dimensions.kv_heads),
            ("head_dim", config.dimensions.head_dim),
            ("vocabulary_size", config.dimensions.vocabulary_size),
        ] {
            assert_eq!(json!(actual), expected[name], "{name}");
        }
        assert_eq!(
            json!(config.tie_word_embeddings),
            expected["tie_word_embeddings"]
        );
        assert_eq!(
            config.rope.theta,
            expected["rope"]["theta"].as_f64().ok_or("theta")? as f32
        );
        assert_eq!(
            json!(config.rope.dimensions),
            expected["rope"]["dimensions"]
        );
        assert_eq!(
            json!(config.rope.traditional),
            expected["rope"]["traditional"]
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
                "rope_type": "llama3",
                "factor": factor,
                "low_freq_factor": low_frequency_factor,
                "high_freq_factor": high_frequency_factor,
                "original_max_position_embeddings": original_max_positions,
            }),
        };
        assert_eq!(scaling, expected["rope"]["scaling"]);
        assert_eq!(
            config.dimensions.rms_norm_epsilon,
            expected["rms_norm_epsilon"].as_f64().ok_or("epsilon")? as f32
        );
        let attention: Vec<Value> = config
            .attention
            .iter()
            .map(|kind| match kind {
                AttentionKind::Full => json!("full"),
                AttentionKind::Sliding { window } => json!({"sliding": window.get()}),
            })
            .collect();
        assert_eq!(json!(attention), expected["attention_kinds"]);
        if let Some(quant) = config.quantization {
            assert_eq!(json!(quant.default.bits), expected["quantization"]["bits"]);
            assert_eq!(
                json!(quant.default.group_size.get()),
                expected["quantization"]["group_size"]
            );
        } else {
            assert!(expected["quantization"].is_null());
        }
    }
    Ok(())
}

#[test]
fn pure_fixture_architecture_errors_are_typed() -> Result<(), Box<dyn std::error::Error>> {
    for name in [
        "llama-base",
        "llama-sliding",
        "llama-quant4",
        "llama-sharded",
    ] {
        let expected: Value =
            serde_json::from_slice(&std::fs::read(fixture(name).join("expectations.json"))?)?;
        for (case, expectation) in expected["errors"].as_object().ok_or("errors")? {
            let Some(class) = expectation["expected_rust_error"].as_str() else {
                return Err("error class".into());
            };
            if !class.starts_with("ConfigError::") {
                if !matches!(
                    case.as_str(),
                    "missing_shard" | "duplicate_tensor" | "wrong_shape"
                ) {
                    return Err(format!("unhandled fixture error {case}").into());
                }
                continue;
            }
            let mut raw = raw(name)?;
            match case.as_str() {
                "unknown_model_type" => raw.model_type = "unsupported_oracle_model".into(),
                "bad_quantization_bits" => {
                    raw.fields.insert(
                        "quantization".into(),
                        json!({"mode":"affine", "group_size":32, "bits":5}),
                    );
                }
                "unsupported_rope" => {
                    raw.fields.insert(
                        "rope_scaling".into(),
                        json!({"rope_type":"unsupported_oracle_rope"}),
                    );
                }
                _ => return Err(format!("unhandled architecture fixture error {case}").into()),
            }
            let error = match Factory.parse_config(&raw) {
                Err(error) => error,
                Ok(_) => return Err(format!("accepted {case}").into()),
            };
            let actual = match error {
                ConfigError::UnsupportedArchitecture(_) => "ConfigError::UnsupportedArchitecture",
                ConfigError::UnsupportedQuantization(_) => "ConfigError::UnsupportedQuantization",
                ConfigError::UnsupportedRope(_) => "ConfigError::UnsupportedRope",
                error => return Err(error.into()),
            };
            assert_eq!(actual, class, "{name}/{case}");
        }
    }
    Ok(())
}

fn mutated_weights(
    name: &str,
    case: &str,
) -> Result<tempfile::TempDir, Box<dyn std::error::Error>> {
    use safetensors::{
        tensor::{serialize_to_file, TensorView},
        SafeTensors,
    };
    use std::collections::{BTreeMap, BTreeSet};
    let manifest = WeightManifest::discover(&fixture(name))?;
    let shards: BTreeSet<_> = manifest
        .tensors
        .values()
        .filter_map(crate::weights::WeightEntry::safetensors_shard)
        .collect();
    let bytes = shards
        .into_iter()
        .map(std::fs::read)
        .collect::<Result<Vec<_>, _>>()?;
    let mut converted: Vec<Vec<u8>> = Vec::new();
    let mut tensors = BTreeMap::new();
    for bytes in &bytes {
        tensors.extend(SafeTensors::deserialize(bytes)?.tensors());
    }
    if case == "bf16" {
        converted.extend(
            tensors
                .values()
                .filter(|tensor| tensor.dtype() == safetensors::Dtype::F32)
                .map(|tensor| {
                    tensor
                        .data()
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .flat_map(|&chunk| {
                            let bits = f32::from_le_bytes(chunk).to_bits();
                            let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) >> 16;
                            (rounded as u16).to_le_bytes()
                        })
                        .collect()
                }),
        );
        for (tensor, bytes) in tensors
            .values_mut()
            .filter(|tensor| tensor.dtype() == safetensors::Dtype::F32)
            .zip(&converted)
        {
            *tensor = TensorView::new(safetensors::Dtype::BF16, tensor.shape().to_vec(), bytes)?;
        }
    }
    if case == "wrong_shape" {
        let tensor = tensors
            .get_mut("model.layers.0.self_attn.q_proj.weight")
            .ok_or("q weight")?;
        let mut shape = tensor.shape().to_vec();
        let row_bytes = tensor.data().len() / shape[0];
        shape[0] -= 1;
        *tensor = TensorView::new(
            tensor.dtype(),
            shape,
            &tensor.data()[..tensor.data().len() - row_bytes],
        )?;
    }
    let tensors: Vec<_> = tensors.into_iter().collect();
    let (first, second) = tensors.split_at(tensors.len() / 2);
    let mut second = second.to_vec();
    if case == "duplicate_tensor" {
        second.push(first.first().ok_or("first tensor")?.clone());
    }
    let target = tempfile::tempdir()?;
    let mut weight_map = BTreeMap::new();
    for (index, tensors) in [first, second.as_slice()].into_iter().enumerate() {
        let shard = format!("model-{:05}-of-00002.safetensors", index + 1);
        for (key, _) in tensors {
            weight_map.entry(key).or_insert_with(|| shard.clone());
        }
        serialize_to_file(
            tensors.iter().map(|(key, tensor)| (key, tensor)),
            None,
            &target.path().join(&shard),
        )?;
    }
    std::fs::write(
        target.path().join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({"weight_map": weight_map}))?,
    )?;
    if case == "missing_shard" {
        std::fs::remove_file(target.path().join("model-00002-of-00002.safetensors"))?;
    }
    Ok(target)
}

fn assert_weight_error(
    name: &str,
    case: &str,
    error: crate::WeightError,
) -> Result<(), Box<dyn std::error::Error>> {
    let expected: Value =
        serde_json::from_slice(&std::fs::read(fixture(name).join("expectations.json"))?)?;
    let class = match error {
        crate::WeightError::MissingShard(_) => "WeightError::MissingShard",
        crate::WeightError::DuplicateTensor(_) => "WeightError::DuplicateTensor",
        crate::WeightError::ShapeMismatch { .. } => "WeightError::ShapeMismatch",
        other => return Err(other.into()),
    };
    assert_eq!(
        class, expected["errors"][case]["expected_rust_error"],
        "{name}/{case}"
    );
    Ok(())
}

#[test]
fn pure_fixture_shard_errors() -> Result<(), Box<dyn std::error::Error>> {
    for name in [
        "llama-base",
        "llama-sliding",
        "llama-quant4",
        "llama-sharded",
    ] {
        for case in ["missing_shard", "duplicate_tensor"] {
            let copy = mutated_weights(name, case)?;
            match WeightManifest::discover(copy.path()) {
                Err(error) => assert_weight_error(name, case, error)?,
                Ok(_) => return Err(format!("accepted {name}/{case}").into()),
            }
        }
    }
    Ok(())
}

#[test]
fn fixture_wrong_shape_errors() -> Result<(), Box<dyn std::error::Error>> {
    for name in [
        "llama-base",
        "llama-sliding",
        "llama-quant4",
        "llama-sharded",
    ] {
        let copy = mutated_weights(name, "wrong_shape")?;
        let manifest = WeightManifest::discover(copy.path())?;
        match Factory.build(Factory.parse_config(&raw(name)?)?, &manifest) {
            Err(LoadError::Weights(error)) => assert_weight_error(name, "wrong_shape", error)?,
            Err(error) => return Err(error.into()),
            Ok(_) => return Err(format!("accepted {name}/wrong_shape").into()),
        }
    }
    Ok(())
}

#[test]
fn pure_defaults_and_numeric_validation() -> Result<(), Box<dyn std::error::Error>> {
    let mut raw = raw("llama-base")?;
    for key in [
        "head_dim",
        "num_key_value_heads",
        "rope_theta",
        "tie_word_embeddings",
        "max_position_embeddings",
    ] {
        raw.fields.remove(key);
    }
    let parsed = config::parse(&raw)?.config;
    assert_eq!(parsed.dimensions.head_dim, 16);
    assert_eq!(parsed.dimensions.kv_heads, 4);
    assert!(parsed.tie_word_embeddings);
    assert_eq!(parsed.dimensions.max_positions, None);
    assert_eq!(parsed.rope.scaling, RopeScaling::None);
    for (key, value) in [
        ("num_attention_heads", json!(0)),
        ("hidden_size", json!(63)),
        ("num_key_value_heads", json!(3)),
        ("head_dim", json!(3)),
        ("vocab_size", json!(0)),
        ("rms_norm_eps", json!(0)),
        ("num_hidden_layers", json!(2147483648u64)),
    ] {
        let mut candidate = RawConfig {
            model_type: raw.model_type.clone(),
            fields: raw.fields.clone(),
        };
        candidate.fields.insert(key.into(), value);
        assert!(
            matches!(
                config::parse(&candidate),
                Err(ConfigError::InvalidNumericField { .. })
            ),
            "{key}"
        );
    }
    raw.fields
        .insert("layer_types".into(), json!(["full_attention"]));
    assert!(matches!(
        config::parse(&raw),
        Err(ConfigError::InvalidLayerPattern { .. })
    ));
    Ok(())
}

#[test]
fn pure_key_dispositions_are_exact() -> Result<(), Box<dyn std::error::Error>> {
    for key in [
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.layers.0.self_attn.q_proj.scales",
        "model.layers.12.mlp.down_proj.biases",
        "model.layers.1.input_layernorm.weight",
    ] {
        match Factory.map_safetensors_key(key) {
            WeightDisposition::Parameter(path) => assert_eq!(path.as_str(), key),
            _ => return Err(format!("rejected {key}").into()),
        }
    }
    for key in [
        "model.layers.-1.self_attn.q_proj.weight",
        "model.layers.01.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.norm.bias",
        "model.embed_tokens.bias",
        "lm_head.bias",
        "model.layers.0.self_attn.rotary_emb.inv_freq.extra",
    ] {
        assert!(
            matches!(Factory.map_safetensors_key(key), WeightDisposition::Reject),
            "{key}"
        );
    }
    assert!(matches!(
        map_key("lm_head.weight", true),
        WeightDisposition::Ignore {
            reason: "llama.redundant_tied_lm_head"
        }
    ));
    assert!(matches!(
        Factory.map_safetensors_key("model.layers.0.self_attn.rotary_emb.inv_freq"),
        WeightDisposition::Ignore {
            reason: "llama.recomputed_rotary_frequencies"
        }
    ));
    assert!(matches!(
        Factory.map_gguf_key("token_embd.weight"),
        WeightDisposition::Parameter(path) if path.as_str() == "model.embed_tokens.weight"
    ));
    Ok(())
}

#[test]
fn pure_rope_frequency_bands_and_linear_scaling() -> Result<(), Box<dyn std::error::Error>> {
    let config = config::parse(&raw("llama-sliding")?)?.config.rope;
    let frequencies = rope::frequencies(&config)?.ok_or("llama3 frequencies")?;
    assert_eq!(frequencies.len(), 8);
    assert!((frequencies[0] - 1.0).abs() < 1e-6);
    assert!((frequencies[2] - 76.673_85).abs() < 1e-3);
    assert!((frequencies[4] - 800.0).abs() < 1e-3);
    let mut raw = raw("llama-base")?;
    raw.fields
        .insert("rope_scaling".into(), json!({"type":"linear", "factor":2}));
    assert_eq!(
        config::parse(&raw)?.config.rope.scaling,
        RopeScaling::Linear { factor: 2.0 }
    );
    for value in [
        json!({"rope_type":"llama3", "factor":8, "high_freq_factor":1}),
        json!({"type":"linear", "factor":0}),
    ] {
        raw.fields.insert("rope_scaling".into(), value);
        assert!(matches!(
            config::parse(&raw),
            Err(ConfigError::InvalidNumericField { .. })
        ));
    }
    Ok(())
}

fn compare_array(
    actual: &mlx_rs::Array,
    expected: &safetensors::tensor::TensorView<'_>,
    tolerance: &Value,
) -> Result<(), Box<dyn std::error::Error>> {
    let actual = actual.contiguous()?;
    actual.eval()?;
    assert_eq!(actual.dtype(), mlx_rs::Dtype::Float32);
    assert_eq!(expected.dtype(), safetensors::Dtype::F32);
    let shape = actual
        .shape()
        .iter()
        .map(|&n| usize::try_from(n))
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(shape, expected.shape());
    let actual = actual.to_vec_exact::<f32>()?;
    let expected = expected
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    assert_eq!(actual.len(), expected.len());
    let atol = tolerance["atol"].as_f64().ok_or("atol")? as f32;
    let rtol = tolerance["rtol"].as_f64().ok_or("rtol")? as f32;
    for (position, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite()
                && expected.is_finite()
                && (actual - expected).abs() <= atol + rtol * expected.abs(),
            "position {position}: {actual} != {expected}"
        );
    }
    Ok(())
}

#[test]
fn fixture_prefill_all_positions_cache_and_chunks() -> Result<(), Box<dyn std::error::Error>> {
    use mlx_rs::{ops::concatenate, Array};
    for name in [
        "llama-base",
        "llama-sliding",
        "llama-quant4",
        "llama-sharded",
    ] {
        let directory = fixture(name);
        let metadata: Value =
            serde_json::from_slice(&std::fs::read(directory.join("expectations.json"))?)?;
        let bytes = std::fs::read(directory.join("expectations.safetensors"))?;
        let expected = safetensors::SafeTensors::deserialize(&bytes)?;
        let tokens: Vec<u32> = serde_json::from_value(metadata["prefill"]["token_ids"].clone())?;
        let manifest = WeightManifest::discover(&directory)?;
        let mut decoder = Factory.build(Factory.parse_config(&raw(name)?)?, &manifest)?;
        for chunk_size in [tokens.len(), 1, 3] {
            let mut cache = crate::Cache::new(
                decoder.config().model_type.clone(),
                decoder.cache_layout(),
                &crate::CacheOptions::default(),
                std::num::NonZeroUsize::new(tokens.len()),
            )?;
            let mut outputs = Vec::new();
            for ids in tokens.chunks(chunk_size) {
                let input = Array::from_slice(ids, &[1, i32::try_from(ids.len())?]);
                let mut step = cache.step()?;
                let output = decoder.forward(&input, &mut step)?;
                step.evaluate_and_commit(&[&output])?;
                outputs.push(output);
            }
            let output = concatenate(&outputs, 1)?;
            compare_array(
                &output,
                &expected.tensor("prefill.full.logits")?,
                &metadata["tolerances"]["logits"],
            )?;
            compare_array(
                &output,
                &expected.tensor(&format!("prefill.chunk{chunk_size}.logits"))?,
                &metadata["tolerances"]["logits"],
            )?;
            if chunk_size == tokens.len() {
                compare_cache(&cache, &expected, &metadata, "after_prefill")?;
            }
            let greedy: Vec<u32> =
                serde_json::from_value(metadata["decode"]["greedy_ids"].clone())?;
            let mut logits = output;
            for (index, token) in greedy.into_iter().enumerate() {
                use mlx_rs::ops::indexing::TryIndexOp;
                let last = logits.try_index((.., -1, ..))?;
                assert_eq!(
                    mlx_rs::ops::indexing::argmax_axis(&last, -1, false)?
                        .try_item_exact::<u32>()?,
                    token,
                    "{name}/chunk{chunk_size}/decode{index}"
                );
                let mut step = cache.step()?;
                logits = decoder.forward(&Array::from_slice(&[token], &[1, 1]), &mut step)?;
                step.evaluate_and_commit(&[&logits])?;
                compare_array(
                    &logits.squeeze_axes(&[1])?,
                    &expected.tensor(&format!("decode.step{index}.logits"))?,
                    &metadata["tolerances"]["logits"],
                )?;
            }
            compare_cache(&cache, &expected, &metadata, "after_decode")?;
        }
    }
    Ok(())
}

#[test]
fn fixture_half_precision_forward() -> Result<(), Box<dyn std::error::Error>> {
    use mlx_rs::{Array, Dtype};
    for name in ["llama-base", "llama-sliding"] {
        let directory = fixture(name);
        let metadata: Value =
            serde_json::from_slice(&std::fs::read(directory.join("expectations.json"))?)?;
        let bytes = std::fs::read(directory.join("expectations.safetensors"))?;
        let expected = safetensors::SafeTensors::deserialize(&bytes)?;
        let tokens: Vec<u32> = serde_json::from_value(metadata["prefill"]["token_ids"].clone())?;
        let copy = mutated_weights(name, "bf16")?;
        let manifest = WeightManifest::discover(copy.path())?;
        let mut decoder = Factory.build(Factory.parse_config(&raw(name)?)?, &manifest)?;
        let vocabulary = i32::try_from(decoder.config().dimensions.vocabulary_size)?;
        let length = i32::try_from(tokens.len())?;
        let mut cache = crate::Cache::new(
            decoder.config().model_type.clone(),
            decoder.cache_layout(),
            &crate::CacheOptions::default(),
            std::num::NonZeroUsize::new(tokens.len() + 1),
        )?;
        let input = Array::from_slice(&tokens, &[1, length]);
        let mut step = cache.step()?;
        let output = decoder.forward(&input, &mut step)?;
        step.evaluate_and_commit(&[&output])?;
        assert_eq!(output.dtype(), Dtype::Bfloat16);
        assert_eq!(output.shape(), &[1, length, vocabulary]);
        compare_array(
            &output.as_dtype(Dtype::Float32)?,
            &expected.tensor("prefill.full.logits")?,
            &json!({"atol": 0.25, "rtol": 0.0}),
        )?;
        let token: u32 = serde_json::from_value(metadata["decode"]["greedy_ids"][0].clone())?;
        let mut step = cache.step()?;
        let logits = decoder.forward(&Array::from_slice(&[token], &[1, 1]), &mut step)?;
        step.evaluate_and_commit(&[&logits])?;
        assert_eq!(logits.dtype(), Dtype::Bfloat16);
        assert_eq!(logits.shape(), &[1, 1, vocabulary]);
        assert!(logits
            .as_dtype(Dtype::Float32)?
            .to_vec_exact::<f32>()?
            .iter()
            .all(|value| value.is_finite()));
    }
    Ok(())
}

fn compare_cache(
    cache: &crate::Cache,
    expected: &safetensors::SafeTensors<'_>,
    metadata: &Value,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let states = metadata["cache"][label].as_array().ok_or("cache states")?;
    assert_eq!(cache.info().len(), states.len());
    let views = cache.logical_layers()?;
    assert_eq!(views.len(), states.len());
    for (info, view) in cache.info().zip(views) {
        let state = &states[info.layer];
        assert_eq!(view.layer, info.layer);
        assert!(info.retained_prefix.is_empty());
        assert_eq!(json!(info.processed_tokens), state["offset"]);
        assert_eq!(
            json!([info.retained_positions.start, info.retained_positions.end]),
            state["retained_range"]
        );
        assert_eq!(view.positions, info.retained_positions);
        for (kind, actual) in [("keys", &view.keys), ("values", &view.values)] {
            compare_array(
                actual,
                &expected.tensor(&format!("cache.{label}.layer{}.{kind}", info.layer))?,
                &metadata["tolerances"]["cache"],
            )?;
        }
    }
    Ok(())
}

#[test]
fn pure_masks_use_absolute_positions_and_window_boundaries() -> Result<(), crate::CacheError> {
    let full = decoder::mask_data(5, 2, &[0, 3, 4, 5, 6], &AttentionKind::Full)?;
    assert_eq!(
        full,
        [true, true, true, true, false, true, true, true, true, true]
    );
    let sliding = AttentionKind::Sliding {
        window: std::num::NonZeroUsize::new(3)
            .ok_or_else(|| crate::CacheError::InvalidState("window".into()))?,
    };
    let masked = decoder::mask_data(5, 2, &[0, 3, 4, 5, 6], &sliding)?;
    assert_eq!(
        masked,
        [false, true, true, true, false, false, false, true, true, true]
    );
    assert_eq!(
        decoder::mask_data(5, 2, &[3, 4, 5, 6], &sliding)?,
        [true, true, true, false, false, true, true, true]
    );
    assert!(matches!(
        decoder::mask_data(usize::MAX, 1, &[0], &AttentionKind::Full),
        Err(crate::CacheError::InvalidState(_))
    ));
    for (offset, length, positions) in [(5, 2, vec![5]), (0, 1, vec![0, 1])] {
        assert!(matches!(
            decoder::mask_data(offset, length, &positions, &AttentionKind::Full),
            Err(crate::CacheError::InvalidState(_))
        ));
    }
    Ok(())
}

#[test]
fn pure_affine_overrides_are_resolved_before_allocation() -> Result<(), Box<dyn std::error::Error>>
{
    let mut raw = raw("llama-base")?;
    raw.fields.insert(
        "quantization".into(),
        json!({
            "bits":4, "group_size":32, "mode":"affine",
            "model.embed_tokens": false,
            "model.layers.1.self_attn.q_proj": {"bits":8, "group_size":64}
        }),
    );
    let quantization = config::parse(&raw)?
        .config
        .quantization
        .ok_or("quantization")?;
    raw.fields.insert(
        "quantization_config".into(),
        raw.fields["quantization"].clone(),
    );
    assert_eq!(
        config::parse(&raw)?.config.quantization.as_ref(),
        Some(&quantization)
    );
    raw.fields.insert(
        "quantization_config".into(),
        json!({"quant_method": "gptq"}),
    );
    assert_eq!(
        config::parse(&raw)?.config.quantization.as_ref(),
        Some(&quantization)
    );
    assert!(matches!(
        quantization
            .layers
            .get(&ParameterPath::new("model.embed_tokens")),
        Some(crate::LayerQuantization::Unquantized)
    ));
    assert!(
        matches!(quantization.layers.get(&ParameterPath::new("model.layers.1.self_attn.q_proj")), Some(crate::LayerQuantization::Affine(options)) if options.bits == 8 && options.group_size.get() == 64)
    );
    for value in [
        json!({"bits":4, "group_size":32, "model.layers.9.self_attn.q_proj":false}),
        json!({"bits":4, "group_size":32, "model.layers.0.self_attn.q_proj":{"bits":4,"group_size":128}}),
        json!({"bits":4, "group_size":32, "mode":"mxfp4"}),
    ] {
        raw.fields.insert("quantization".into(), value);
        assert!(matches!(
            config::parse(&raw),
            Err(ConfigError::UnsupportedQuantization(_))
        ));
    }
    raw.fields.remove("quantization");
    raw.fields
        .insert("quantization_config".into(), json!({"quant_method":"gptq"}));
    assert!(matches!(
        config::parse(&raw),
        Err(ConfigError::UnsupportedQuantization(method)) if method.contains("gptq")
    ));
    raw.fields.insert("quantization".into(), Value::Null);
    assert!(
        matches!(config::parse(&raw), Err(ConfigError::UnsupportedQuantization(method)) if method.contains("gptq"))
    );
    raw.fields.remove("quantization");
    raw.fields.insert("quantization_config".into(), json!({}));
    assert!(
        matches!(config::parse(&raw), Err(ConfigError::UnsupportedQuantization(method)) if method.contains("missing quant_method"))
    );
    Ok(())
}
