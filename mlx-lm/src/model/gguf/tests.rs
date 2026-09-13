use super::*;
use crate::arch::ArchitectureFactory;
use anyhow::{ensure, Context, Result};
use mlx_rs::ops::indexing::TryIndexOp;
use serde_json::Value;
use std::{
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../conformance/mlx-lm")
}
fn fixture(name: &str) -> PathBuf {
    root().join("fixtures").join(name)
}
fn document(name: &str) -> Result<Value> {
    Ok(serde_json::from_slice(&std::fs::read(
        fixture(name).join("expectations.json"),
    )?)?)
}
fn tokenizer(name: &str) -> Result<Tokenizer> {
    let doc = document(name)?;
    let path = fixture(name).join(
        doc["gguf"]["tokenizer_files"]["tokenizer.json"]["path"]
            .as_str()
            .context("tokenizer path")?,
    );
    Ok(Tokenizer::from_dir(
        path.parent().context("tokenizer directory")?,
    )?)
}
fn load_fixture(name: &str) -> Result<Model> {
    let file = mlx_rs::with_stream(&mlx_rs::Stream::cpu(), || {
        GgufFile::load(fixture(name).join("model.gguf"))
    })?;
    Ok(Model::from_gguf(file, tokenizer(name)?)?)
}
fn compare(
    actual: &Array,
    expected: &safetensors::SafeTensors<'_>,
    doc: &Value,
    key: &str,
) -> Result<()> {
    let tensor = expected.tensor(key)?;
    ensure!(
        actual
            .shape()
            .iter()
            .map(|&n| n as usize)
            .collect::<Vec<_>>()
            == tensor.shape(),
        "shape {key}"
    );
    let dtype = match actual.dtype() {
        Dtype::Float32 => "float32",
        Dtype::Float16 => "float16",
        other => anyhow::bail!("unexpected native dtype {other:?}"),
    };
    ensure!(doc["native_dtypes"][key] == dtype, "native dtype {key}");
    let (atol, rtol) = match doc["tolerance_policy"].as_str().context("policy")? {
        "f32-v1" => (2e-4_f64, 2e-4_f64),
        "gguf-f16-v1" => (5e-3, 5e-3),
        "gguf-affine-v1" => (2e-2, 2e-3),
        other => anyhow::bail!("unknown tolerance {other}"),
    };
    let actual = actual.to_vec_cast::<f32>()?;
    let wanted: Vec<_> = tensor
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    ensure!(actual.len() == wanted.len(), "length {key}");
    for (i, (a, b)) in actual.into_iter().zip(wanted).enumerate() {
        ensure!(
            a.is_finite()
                && b.is_finite()
                && f64::from((a - b).abs()) <= atol + rtol * f64::from(b.abs()),
            "{key}[{i}]: {a} != {b}"
        );
    }
    Ok(())
}
fn compare_cache(
    cache: &crate::Cache,
    expected: &safetensors::SafeTensors<'_>,
    doc: &Value,
    label: &str,
) -> Result<()> {
    let states = doc["cache"][label].as_array().context("cache states")?;
    let views = cache.logical_layers()?;
    ensure!(views.len() == states.len(), "layer count");
    for (info, view) in cache.info().zip(views) {
        let state = &states[info.layer];
        ensure!(json!(info.processed_tokens) == state["offset"], "offset");
        ensure!(
            json!([view.positions.start, view.positions.end]) == state["retained_range"],
            "retained range"
        );
        for (kind, array) in [("keys", view.keys), ("values", view.values)] {
            compare(
                &array,
                expected,
                doc,
                &format!("cache.{label}.layer{}.{kind}", info.layer),
            )?;
        }
    }
    Ok(())
}
fn forward_fixture(name: &str) -> Result<()> {
    let mut model = load_fixture(name)?;
    let doc = document(name)?;
    let bytes = std::fs::read(fixture(name).join("expectations.safetensors"))?;
    let expected = safetensors::SafeTensors::deserialize(&bytes)?;
    let inputs: Value = serde_json::from_slice(&std::fs::read(fixture(name).join("inputs.json"))?)?;
    ensure!(
        inputs["token_ids"]["canonical"] == doc["prefill"]["token_ids"],
        "canonical inputs"
    );
    let tokens: Vec<u32> = serde_json::from_value(doc["prefill"]["token_ids"].clone())?;
    let greedy: Vec<u32> = serde_json::from_value(doc["decode"]["greedy_ids"].clone())?;
    ensure!(greedy.len() == 8, "eight decode steps");
    for (label, chunk) in [
        ("full", tokens.len()),
        ("chunk1", 1),
        ("chunk3", 3),
        ("chunk8", 8),
    ] {
        let decoder = &mut model.decoder;
        let mut cache = crate::Cache::new(
            decoder.config().model_type.clone(),
            decoder.cache_layout(),
            &crate::CacheOptions::default(),
            NonZeroUsize::new(tokens.len()),
        )?;
        let mut outputs = Vec::new();
        for ids in tokens.chunks(chunk) {
            let mut step = cache.step()?;
            let logits =
                decoder.forward(&Array::from_slice(ids, &[1, ids.len() as i32]), &mut step)?;
            step.evaluate_and_commit(&[&logits])?;
            outputs.push(logits);
        }
        let mut logits = mlx_rs::ops::concatenate(&outputs, 1)?;
        compare(&logits, &expected, &doc, &format!("prefill.{label}.logits"))?;
        compare(&logits, &expected, &doc, "prefill.full.logits")?;
        compare_cache(&cache, &expected, &doc, "after_prefill")?;
        for (index, &id) in greedy.iter().enumerate() {
            let last = logits.try_index((.., -1, ..))?;
            ensure!(
                mlx_rs::ops::indexing::argmax_axis(&last, -1, false)?.try_item_exact::<u32>()?
                    == id,
                "{name}/{label}/greedy{index}"
            );
            let mut step = cache.step()?;
            logits = decoder.forward(&Array::from_slice(&[id], &[1, 1]), &mut step)?;
            step.evaluate_and_commit(&[&logits])?;
            compare(
                &logits.squeeze_axes(&[1])?,
                &expected,
                &doc,
                &format!("decode.step{index}.logits"),
            )?;
        }
        compare_cache(&cache, &expected, &doc, "after_decode")?;
    }
    let prompt: Vec<_> = tokens.into_iter().map(TokenId::from).collect();
    let options = crate::GenerationOptions {
        max_tokens: NonZeroUsize::new(8).unwrap(),
        stop: crate::StopPolicy {
            tokens: crate::StopTokenPolicy::Exact(vec![]),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut observed = Vec::new();
    for event in model.generate(crate::Prompt::Tokens(&prompt), options)? {
        if let crate::GenerationEvent::Token { token_id, .. } = event? {
            observed.push(u32::from(token_id));
        }
    }
    ensure!(observed == greedy, "public greedy prefix");
    Ok(())
}
macro_rules! forwards {
    ($($test:ident => $name:literal),* $(,)?) => { $(#[test] fn $test() -> Result<()> { forward_fixture($name) })* };
}
forwards! {
    gguf_llama_f32 => "gguf-llama-f32", gguf_llama_f16 => "gguf-llama-f16",
    gguf_llama_q4_0 => "gguf-llama-q4_0", gguf_llama_q4_1 => "gguf-llama-q4_1", gguf_llama_q8_0 => "gguf-llama-q8_0",
    gguf_qwen3_f32 => "gguf-qwen3-f32", gguf_qwen3_f16 => "gguf-qwen3-f16",
    gguf_qwen3_q4_0 => "gguf-qwen3-q4_0", gguf_qwen3_q4_1 => "gguf-qwen3-q4_1", gguf_qwen3_q8_0 => "gguf-qwen3-q8_0",
}

#[test]
fn loading_inside_caller_gpu_stream() -> Result<()> {
    for name in ["gguf-llama-q4_1", "gguf-qwen3-q4_1"] {
        let file = mlx_rs::with_stream(&mlx_rs::Stream::cpu(), || {
            GgufFile::load(fixture(name).join("model.gguf"))
        })?;
        mlx_rs::with_stream(&mlx_rs::Stream::gpu(), || {
            Model::from_gguf(file, tokenizer(name).unwrap())
        })?;
    }
    Ok(())
}

fn error_json(error: LoadError) -> Value {
    match error {
        LoadError::Config(ConfigError::MissingField { field }) => {
            json!({"variant":"LoadError::Config(ConfigError::MissingField)","field":field})
        }
        LoadError::Config(ConfigError::InvalidGgufMetadata { key, .. }) => {
            json!({"variant":"LoadError::Config(ConfigError::InvalidGgufMetadata)","key":key})
        }
        LoadError::Config(ConfigError::InvalidNumericField { field, .. }) => {
            json!({"variant":"LoadError::Config(ConfigError::InvalidNumericField)","field":field})
        }
        LoadError::Config(ConfigError::UnsupportedGgufMetadata { key, .. }) => {
            json!({"variant":"LoadError::Config(ConfigError::UnsupportedGgufMetadata)","key":key})
        }
        LoadError::Config(ConfigError::UnsupportedArchitecture(value)) => {
            json!({"variant":"LoadError::Config(ConfigError::UnsupportedArchitecture)","value":value})
        }
        LoadError::Config(ConfigError::UnsupportedRope(_)) => {
            json!({"variant":"LoadError::Config(ConfigError::UnsupportedRope)"})
        }
        LoadError::Weights(crate::WeightError::MissingKey(key)) => {
            json!({"variant":"LoadError::Weights(WeightError::MissingKey)","key":key})
        }
        LoadError::Weights(crate::WeightError::UnexpectedKey(key)) => {
            json!({"variant":"LoadError::Weights(WeightError::UnexpectedKey)","key":key})
        }
        LoadError::Weights(crate::WeightError::ShapeMismatch {
            key,
            expected,
            actual,
        }) => {
            json!({"variant":"LoadError::Weights(WeightError::ShapeMismatch)","key":key,"expected":expected,"actual":actual})
        }
        LoadError::TokenizerVocabularyOutOfRange {
            token_id,
            vocabulary_size,
        } => {
            json!({"variant":"LoadError::TokenizerVocabularyOutOfRange","token_id":u32::from(token_id),"vocabulary_size":vocabulary_size})
        }
        LoadError::TokenizerMetadataMismatch { key, .. } => {
            json!({"variant":"LoadError::TokenizerMetadataMismatch","key":key})
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn malformed_recipes_exact_typed_errors() -> Result<()> {
    let cases: Value = serde_json::from_slice(&std::fs::read(root().join("gguf_cases.json"))?)?;
    let temp = tempfile::tempdir()?;
    let script = r#"
import json, pathlib, sys
from gguf_recipes import materialize
root, output = map(pathlib.Path, sys.argv[1:])
for recipe in json.loads((root / 'gguf_cases.json').read_text())['cases']:
    if recipe['operation'] != 'load_core':
        materialize(root / 'fixtures' / recipe['base'] / 'model.gguf', output / (recipe['id'] + '.gguf'), recipe)
"#;
    let python = std::env::var_os("MLX_LM_GGUF_RECIPE_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|| root().join("../.venv-mlx-lm/bin/python"));
    let output = std::process::Command::new(python)
        .arg("-c")
        .arg(script)
        .arg(root())
        .arg(temp.path())
        .env("PYTHONPATH", root())
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .output()?;
    ensure!(
        output.status.success(),
        "recipe materialization: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    for case in cases["cases"].as_array().context("recipes")? {
        let id = case["id"].as_str().context("id")?;
        let actual = if case["operation"] == "load_core" {
            let error = GgufFile::load(fixture(case["base"].as_str().unwrap()))
                .err()
                .context("expected core rejection")?;
            ensure!(
                matches!(error, mlx_rs::io::GgufError::Exception(_)),
                "opaque core exception"
            );
            json!({"variant":"GgufError::Exception"})
        } else {
            let file = mlx_rs::with_stream(&mlx_rs::Stream::cpu(), || {
                GgufFile::load(temp.path().join(format!("{id}.gguf")))
            })?;
            let error = Model::from_gguf(file, tokenizer(case["base"].as_str().unwrap())?)
                .err()
                .context("expected model rejection")?;
            error_json(error)
        };
        ensure!(
            actual == case["expected"],
            "{id}: expected {}, got {actual}",
            case["expected"]
        );
    }
    Ok(())
}

#[test]
fn metadata_required_order_and_exact_scalar_types() -> Result<()> {
    for field in [
        "block_count",
        "embedding_length",
        "feed_forward_length",
        "attention.head_count",
        "attention.head_count_kv",
        "attention.layer_norm_rms_epsilon",
        "rope.freq_base",
    ] {
        let name = format!("llama.{field}");
        let base = GgufFile::load(fixture("gguf-llama-f32").join("model.gguf"))?;
        let doc = document("gguf-llama-f32")?;
        let weights = WeightManifest::from_gguf(&base)?
            .normalize_gguf(|key| crate::arch::llama::Factory.map_gguf_key(key))?;
        for replacement in [
            None,
            Some(GgufMetadataValue::String("wrong".into())),
            Some(Array::from_slice(&[2_u32], &[1]).into()),
            Some(Array::from_bool(true).into()),
        ] {
            let mut file = GgufFile::new()?;
            for key in doc["gguf"]["metadata"]
                .as_object()
                .context("metadata")?
                .keys()
            {
                if key != &name {
                    file.insert_metadata(key, base.get_metadata(key)?.context("metadata value")?)?;
                }
            }
            if let Some(value) = replacement.clone() {
                file.insert_metadata(&name, value)?;
            }
            let error = raw_config(&file, &weights)
                .err()
                .context("expected metadata rejection")?;
            match (replacement, error) {
                (None, LoadError::Config(ConfigError::MissingField { field })) => {
                    ensure!(field == name)
                }
                (Some(_), LoadError::Config(ConfigError::InvalidGgufMetadata { key, .. })) => {
                    ensure!(key == name)
                }
                (_, other) => anyhow::bail!("{name}: {other:?}"),
            }
        }
    }
    Ok(())
}

#[test]
fn metadata_optional_profile_rejections() -> Result<()> {
    let base = GgufFile::load(fixture("gguf-llama-f32").join("model.gguf"))?;
    let doc = document("gguf-llama-f32")?;
    let weights = WeightManifest::from_gguf(&base)?
        .normalize_gguf(|key| crate::arch::llama::Factory.map_gguf_key(key))?;
    for (key, value, variant) in [
        (
            "llama.rope.scaling.factor",
            Array::from_f32(2.0).into(),
            "unsupported",
        ),
        (
            "llama.use_parallel_residual",
            Array::from_bool(true).into(),
            "unsupported",
        ),
        (
            "llama.attention.sliding_window",
            Array::from_int(8).into(),
            "unsupported",
        ),
        (
            "llama.tensor_data_layout",
            GgufMetadataValue::String("custom".into()),
            "unsupported",
        ),
        (
            "llama.attention.scale",
            Array::from_f32(1.0).into(),
            "unsupported",
        ),
        (
            "llama.rope.dimension_count",
            Array::from_int(8).into(),
            "rope",
        ),
        (
            "llama.rope.scaling.finetuned",
            Array::from_int(1).into(),
            "mistyped",
        ),
        ("general.file_type", Array::from_f32(1.0).into(), "mistyped"),
        (
            "llama.context_length",
            Array::from_int(-1).into(),
            "numeric",
        ),
    ] {
        let mut file = GgufFile::new()?;
        for source in doc["gguf"]["metadata"]
            .as_object()
            .context("metadata")?
            .keys()
        {
            if source != key {
                file.insert_metadata(
                    source,
                    base.get_metadata(source)?.context("metadata value")?,
                )?;
            }
        }
        file.insert_metadata(key, value)?;
        let error = raw_config(&file, &weights)
            .err()
            .context("expected profile rejection")?;
        match (variant, error) {
            (
                "unsupported",
                LoadError::Config(ConfigError::UnsupportedGgufMetadata { key: observed, .. }),
            ) => ensure!(observed == key),
            (
                "mistyped",
                LoadError::Config(ConfigError::InvalidGgufMetadata { key: observed, .. }),
            ) => ensure!(observed == key),
            ("numeric", LoadError::Config(ConfigError::InvalidNumericField { field, .. })) => {
                ensure!(field == key)
            }
            ("rope", LoadError::Config(ConfigError::UnsupportedRope(_))) => {}
            (_, other) => anyhow::bail!("{key}: {other:?}"),
        }
    }
    Ok(())
}

#[test]
fn evaluation_failure_returns_no_model() -> Result<()> {
    mlx_rs::with_stream(&mlx_rs::Stream::cpu(), || -> Result<()> {
        let name = "gguf-llama-f32";
        let base = GgufFile::load(fixture(name).join("model.gguf"))?;
        let doc = document(name)?;
        let mut file = GgufFile::new()?;
        for key in doc["gguf"]["metadata"]
            .as_object()
            .context("metadata")?
            .keys()
        {
            file.insert_metadata(key, base.get_metadata(key)?.context("metadata value")?)?;
        }
        let singular = mlx_rs::linalg::inv(Array::from_slice(&[0_f32], &[1, 1]))?;
        for (key, array) in base.arrays()? {
            if key == "output.weight" {
                file.insert_array(key, &mlx_rs::ops::broadcast_to(&singular, array.shape())?)?;
            } else {
                file.insert_array(key, &array)?;
            }
        }
        ensure!(matches!(
            Model::from_gguf(file, tokenizer(name)?),
            Err(LoadError::Weights(crate::WeightError::Exception(_)))
        ));
        Ok(())
    })
}
