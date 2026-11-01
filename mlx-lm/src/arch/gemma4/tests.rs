use super::*;
use serde_json::{json, Value};
use std::collections::BTreeMap;

fn parsed(value: Value) -> Result<ParsedConfig, ConfigError> {
    config::parse(&RawConfig::from_bytes(&serde_json::to_vec(&value)?)?)
}

fn base_gemma4_text() -> Value {
    json!({
        "model_type": "gemma4_text",
        "hidden_size": 32,
        "num_hidden_layers": 4,
        "intermediate_size": 64,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "global_head_dim": 16,
        "vocab_size": 64,
        "vocab_size_per_layer_input": 64,
        "hidden_size_per_layer_input": 8,
        "num_kv_shared_layers": 2,
        "sliding_window": 4,
        "sliding_window_pattern": 2,
        "layer_types": [
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention"
        ],
        "attention_k_eq_v": true,
        "use_double_wide_mlp": true,
        "tie_word_embeddings": true
    })
}

#[test]
fn config_resolves_text_and_nested_multimodal_and_per_layer_config() -> Result<(), ConfigError> {
    let p = parsed(base_gemma4_text())?;
    assert_eq!(p.first_kv_shared_layer_idx, 2);
    assert_eq!(p.previous_kvs, vec![0, 1, 0, 1]);
    assert_eq!(p.layer_intermediate_sizes, vec![64, 64, 128, 128]);
    assert_eq!(p.full_rope_theta, 1_000_000.0);
    assert_eq!(p.full_partial_rotary_factor, 0.25);

    let multimodal = json!({
        "model_type": "gemma4",
        "vocab_size": 128,
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "num_kv_shared_layers": 0,
            "sliding_window": 4,
            "sliding_window_pattern": 2,
            "per_layer_config": {
                "00": {"head_dim": 16, "num_key_value_heads": 1},
                "01": {"head_dim": 32, "num_key_value_heads": 2}
            }
        }
    });
    let raw = RawConfig::from_bytes(&serde_json::to_vec(&multimodal)?)?;
    assert_eq!(raw.resolve()?.dimensions.vocabulary_size, 128);
    let pm = config::parse(&raw)?;
    assert_eq!(pm.layer_head_dims, vec![16, 32]);
    assert_eq!(pm.layer_kv_heads, vec![1, 2]);
    Ok(())
}

#[test]
fn safetensors_key_mapping_handles_vlm_mtp_and_shared_kv_extras() {
    for (external, expected) in [
        ("model.embed_tokens.weight", "embed_tokens.weight"),
        (
            "language_model.model.embed_tokens_per_layer.weight",
            "embed_tokens_per_layer.weight",
        ),
        (
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.q_proj.weight",
        ),
        (
            "model.layers.1.post_feedforward_layernorm.weight",
            "layers.1.post_feedforward_layernorm.weight",
        ),
        ("model.layers.0.layer_scalar", "layers.0.layer_scalar"),
    ] {
        assert!(
            matches!(
                weights::map_key(external, true, Some(2)),
                WeightDisposition::Parameter(path) if path.as_str() == expected
            ),
            "{external}"
        );
    }

    for ignored in [
        "model.rotary_emb.inv_freq",
        "model.layers.0.self_attn.rotary_emb.inv_freq",
        "model.vision_tower.encoder.layers.0.weight",
        "model.multi_modal_projector.linear.weight",
        "model.audio_tower.conv.weight",
        "model.mtp.proj.weight",
        "language_model.mtp.norm.weight",
        "model.layers.2.self_attn.k_proj.weight",
        "model.layers.3.self_attn.v_norm.weight",
        "lm_head.weight",
    ] {
        assert!(
            matches!(
                weights::map_key(ignored, true, Some(2)),
                WeightDisposition::Ignore { .. }
            ),
            "{ignored}"
        );
    }
}

#[test]
fn proportional_rope_frequencies_mark_nope_pairs_infinite() {
    let freqs = rope::proportional_frequency_table(16, 0.25, 1_000_000.0);
    assert_eq!(freqs.len(), 8);
    assert!((freqs[0] - 1.0).abs() < 1e-6);
    assert!(freqs[1].is_finite());
    for &f in &freqs[2..] {
        assert!(f.is_infinite());
    }
}

#[test]
fn decoder_forward_with_kv_sharing_and_per_layer_inputs() -> Result<(), Box<dyn std::error::Error>>
{
    let cfg_json = base_gemma4_text();
    let raw = RawConfig::from_bytes(&serde_json::to_vec(&cfg_json)?)?;
    let parsed_cfg = Factory.parse_config(&raw)?;

    let mut tensors: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    let mut shapes: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut add_f32 = |name: &str, shape: &[usize], val: f32| {
        let len: usize = shape.iter().product();
        tensors.insert(name.to_string(), vec![val; len]);
        shapes.insert(name.to_string(), shape.to_vec());
    };

    add_f32("model.embed_tokens.weight", &[64, 32], 0.02);
    add_f32("model.embed_tokens_per_layer.weight", &[64, 4 * 8], 0.01);
    add_f32(
        "model.per_layer_model_projection.weight",
        &[4 * 8, 32],
        0.02,
    );
    add_f32("model.per_layer_projection_norm.weight", &[8], 1.0);
    add_f32("model.norm.weight", &[32], 1.0);

    for layer in 0..4 {
        let p = format!("model.layers.{layer}");
        let has_kv = layer < 2;
        let is_sliding = layer % 2 == 0;
        let inter = if layer >= 2 { 128 } else { 64 };

        add_f32(&format!("{p}.input_layernorm.weight"), &[32], 1.0);
        add_f32(&format!("{p}.post_attention_layernorm.weight"), &[32], 1.0);
        add_f32(&format!("{p}.pre_feedforward_layernorm.weight"), &[32], 1.0);
        add_f32(
            &format!("{p}.post_feedforward_layernorm.weight"),
            &[32],
            1.0,
        );
        add_f32(&format!("{p}.self_attn.q_proj.weight"), &[32, 32], 0.02);
        add_f32(&format!("{p}.self_attn.o_proj.weight"), &[32, 32], 0.02);
        add_f32(&format!("{p}.self_attn.q_norm.weight"), &[16], 1.0);
        if has_kv {
            add_f32(&format!("{p}.self_attn.k_proj.weight"), &[16, 32], 0.02);
            add_f32(&format!("{p}.self_attn.k_norm.weight"), &[16], 1.0);
            if is_sliding {
                add_f32(&format!("{p}.self_attn.v_proj.weight"), &[16, 32], 0.02);
            }
        }
        add_f32(&format!("{p}.mlp.gate_proj.weight"), &[inter, 32], 0.02);
        add_f32(&format!("{p}.mlp.up_proj.weight"), &[inter, 32], 0.02);
        add_f32(&format!("{p}.mlp.down_proj.weight"), &[32, inter], 0.02);
        add_f32(&format!("{p}.per_layer_input_gate.weight"), &[8, 32], 0.02);
        add_f32(&format!("{p}.per_layer_projection.weight"), &[32, 8], 0.02);
        add_f32(&format!("{p}.post_per_layer_input_norm.weight"), &[32], 1.0);
        add_f32(&format!("{p}.layer_scalar"), &[1], 1.0);
    }

    let dir = tempfile::tempdir()?;
    let mut byte_map: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for (k, vals) in &tensors {
        let mut bytes = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        byte_map.insert(k.clone(), bytes);
    }
    let views = byte_map
        .iter()
        .map(|(k, b)| {
            Ok((
                k.as_str(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    shapes[k].clone(),
                    b.as_slice(),
                )?,
            ))
        })
        .collect::<Result<Vec<_>, safetensors::SafeTensorError>>()?;
    safetensors::tensor::serialize_to_file(views, None, &dir.path().join("model.safetensors"))?;

    let manifest = WeightManifest::discover(dir.path())?;
    let mut model = Factory.build(parsed_cfg, &manifest)?;
    assert_eq!(model.cache_layout().len(), 2);

    let mut cache = crate::cache::Cache::new(
        crate::ModelType::new("gemma4_text"),
        model.cache_layout(),
        &crate::cache::CacheOptions::default(),
        None,
    )?;

    let input = Array::from_slice(&[1_u32, 2, 3], &[1, 3]);
    let mut step = cache.step()?;
    let logits = model.forward(&input, &mut step)?;
    step.evaluate_and_commit(&[&logits])?;
    assert_eq!(logits.shape(), &[1, 3, 64]);

    let next_input = Array::from_slice(&[4_u32], &[1, 1]);
    let mut step2 = cache.step()?;
    let next_logits = model.forward(&next_input, &mut step2)?;
    step2.evaluate_and_commit(&[&next_logits])?;
    assert_eq!(next_logits.shape(), &[1, 1, 64]);
    let vals = next_logits.to_vec_exact::<f32>()?;
    assert!(vals.iter().all(|v| v.is_finite() && v.abs() <= 30.001));
    Ok(())
}
