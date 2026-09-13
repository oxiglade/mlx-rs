use std::{collections::BTreeMap, fs, num::NonZeroUsize, path::Path};

use anyhow::{bail, ensure, Context, Result};
use mlx_lm::{
    oracle_hooks, FinishReason, GenerationEvent, GenerationOptions, LayerQuantization, Model,
    ParameterPath, Prompt, StopTokenPolicy, TokenId, Tokenizer,
};
use mlx_rs::{
    io::{GgufFile, GgufMetadataKind},
    Array, Device, Dtype,
};
use safetensors::SafeTensors;
use serde_json::{json, Value};

use super::{
    comparator,
    observation::{CacheState, Expectations, Observation, Policy, Tensor},
    prototype, reader, surfaces,
};

const FORMATS: [&str; 5] = ["f32", "f16", "q4_0", "q4_1", "q8_0"];

fn document(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(
        path.join("expectations.json"),
    )?)?)
}

fn tensors(path: &Path) -> Result<BTreeMap<String, Tensor>> {
    let bytes = fs::read(path)?;
    Ok(SafeTensors::deserialize(&bytes)?
        .tensors()
        .into_iter()
        .map(|(key, value)| {
            (
                key,
                Tensor {
                    shape: value.shape().to_vec(),
                    dtype: value.dtype(),
                    bytes: value.data().to_vec(),
                },
            )
        })
        .collect())
}

fn expected(path: &Path, doc: &Value) -> Result<Expectations> {
    let tensors = tensors(&path.join("expectations.safetensors"))?;
    let mut policies = BTreeMap::new();
    for key in tensors.keys() {
        let family = if key.starts_with("cache.") {
            "cache"
        } else {
            "logits"
        };
        policies.insert(
            key.clone(),
            Policy::Float(serde_json::from_value(doc["tolerances"][family].clone())?),
        );
    }
    let mut observation = Observation {
        tensors,
        ..Observation::default()
    };
    for stage in ["after_prefill", "after_decode"] {
        for (i, layer) in doc["cache"][stage]
            .as_array()
            .context("cache states")?
            .iter()
            .enumerate()
        {
            observation.caches.insert(
                format!("{stage}.layer{i}"),
                CacheState {
                    offset: layer["offset"].as_u64().context("offset")? as usize,
                    retained: layer["retained_range"][0].as_u64().context("range")? as usize
                        ..layer["retained_range"][1].as_u64().context("range")? as usize,
                },
            );
        }
    }
    observation.greedy_ids = Some(serde_json::from_value(doc["decode"]["greedy_ids"].clone())?);
    Ok(Expectations {
        observation,
        policies,
    })
}

pub fn validate_fixture(path: &Path) -> Result<()> {
    let doc = document(path)?;
    ensure!(
        doc["schema_version"] == 1 && doc["source"] == "gguf",
        "GGUF schema"
    );
    ensure!(doc["provenance"]["mlx_lm"] == "0.31.3", "upstream pin");
    ensure!(
        doc["reference_qualified"] == true,
        "Python/NumPy qualification missing"
    );
    let (policy, atol, rtol) = match doc["gguf"]["storage"].as_str() {
        Some("f32") => ("f32-v1", 2e-4, 2e-4),
        Some("f16") => ("gguf-f16-v1", 5e-3, 5e-3),
        Some("q4_0" | "q4_1" | "q8_0") => ("gguf-affine-v1", 2e-2, 2e-3),
        other => bail!("unknown GGUF fixture storage: {other:?}"),
    };
    ensure!(
        doc["tolerance_policy"] == policy
            && doc["tolerances"]
                == json!({"logits": {"atol": atol, "rtol": rtol}, "cache": {"atol": atol, "rtol": rtol}}),
        "fixed GGUF tolerance changed"
    );
    ensure!(
        doc["decode"]["greedy_ids"]
            .as_array()
            .context("greedy IDs")?
            .len()
            == 8,
        "eight IDs required"
    );
    let expected = expected(path, &doc)?;
    ensure!(
        expected.observation.tensors.len() == 20,
        "GGUF tensor family incomplete"
    );
    ensure!(
        doc["native_dtypes"]
            .as_object()
            .context("native dtypes")?
            .keys()
            .eq(expected.observation.tensors.keys()),
        "native dtype keys"
    );
    ensure!(
        !doc["mutations"]
            .as_object()
            .context("mutations")?
            .is_empty(),
        "fixture lacks effective value mutation"
    );
    prototype::compare("GGUF self comparison", &expected, &expected.observation)
}

fn dtype_name(dtype: Dtype) -> Result<&'static str> {
    match dtype {
        Dtype::Float32 => Ok("float32"),
        Dtype::Float16 => Ok("float16"),
        other => bail!("unexpected native output dtype {other:?}"),
    }
}

fn record(doc: &Value, observed: &mut Observation, key: String, array: &Array) -> Result<()> {
    ensure!(
        doc["native_dtypes"][&key] == dtype_name(array.dtype())?,
        "native dtype: {key}"
    );
    observed
        .tensors
        .insert(key, prototype::observe(&array.as_dtype(Dtype::Float32)?)?);
    Ok(())
}

fn cache(
    doc: &Value,
    observed: &mut Observation,
    session: &oracle_hooks::OracleSession<'_>,
    stage: &str,
) -> Result<()> {
    for layer in session.cache_view()?.layers {
        observed.caches.insert(
            format!("{stage}.layer{}", layer.layer),
            CacheState {
                offset: layer.positions.end,
                retained: layer.positions,
            },
        );
        for (kind, array) in [("keys", layer.keys), ("values", layer.values)] {
            record(
                doc,
                observed,
                format!("cache.{stage}.layer{}.{kind}", layer.layer),
                &array,
            )?;
        }
    }
    Ok(())
}

fn converted(path: &Path, doc: &Value, file: &GgufFile) -> Result<()> {
    let expected = tensors(&path.join("converted.safetensors"))?;
    ensure!(
        file.array_keys()? == expected.keys().cloned().collect::<Vec<_>>(),
        "converted keys"
    );
    for (key, tensor) in expected {
        let array = file
            .get_array(&key)?
            .context("converted array")?
            .contiguous()?;
        ensure!(
            array
                .shape()
                .iter()
                .map(|&n| n as usize)
                .eq(tensor.shape.iter().copied()),
            "converted shape: {key}"
        );
        match tensor.dtype {
            safetensors::Dtype::U32 => {
                ensure!(array.dtype() == Dtype::Uint32, "converted dtype: {key}");
                let bytes: Vec<_> = array
                    .to_vec_exact::<u32>()?
                    .iter()
                    .flat_map(|n| n.to_le_bytes())
                    .collect();
                ensure!(bytes == tensor.bytes, "converted words: {key}");
            }
            safetensors::Dtype::F16 | safetensors::Dtype::F32 => {
                let dtype = if tensor.dtype == safetensors::Dtype::F16 {
                    Dtype::Float16
                } else {
                    Dtype::Float32
                };
                ensure!(array.dtype() == dtype, "converted dtype: {key}");
                let bytes: Vec<_> = if dtype == Dtype::Float16 {
                    array
                        .view::<u16>()?
                        .to_vec_exact::<u16>()?
                        .iter()
                        .flat_map(|n| n.to_le_bytes())
                        .collect()
                } else {
                    array
                        .view::<u32>()?
                        .to_vec_exact::<u32>()?
                        .iter()
                        .flat_map(|n| n.to_le_bytes())
                        .collect()
                };
                ensure!(bytes == tensor.bytes, "converted floating bits: {key}");
            }
            other => bail!("converted unsupported dtype {other:?}"),
        }
    }
    for (key, expected) in doc["gguf"]["metadata"].as_object().context("metadata")? {
        let kind = doc["gguf"]["metadata_types"][key]
            .as_u64()
            .context("metadata type")?;
        if kind == 8 {
            ensure!(
                file.metadata_kind(key)? == Some(GgufMetadataKind::String),
                "metadata kind {key}"
            );
            ensure!(
                file.get_metadata_string(key)?.as_deref() == expected.as_str(),
                "metadata string {key}"
            );
        } else {
            ensure!(
                file.metadata_kind(key)? == Some(GgufMetadataKind::Array),
                "metadata kind {key}"
            );
            let array = file.get_metadata_array(key)?.context("metadata scalar")?;
            ensure!(array.shape().is_empty(), "metadata scalar shape {key}");
            let dtype = if kind == 6 {
                Dtype::Float32
            } else {
                Dtype::Uint32
            };
            ensure!(array.dtype() == dtype, "metadata scalar dtype {key}");
            ensure!(
                f64::from(array.to_vec_cast::<f32>()?[0])
                    == expected.as_f64().context("metadata number")?,
                "metadata value {key}"
            );
        }
    }
    Ok(())
}

fn run(path: &Path) -> Result<()> {
    validate_fixture(path)?;
    let doc = document(path)?;
    let file = mlx_rs::with_device(Device::cpu(), || GgufFile::load(path.join("model.gguf")))?;
    converted(path, &doc, &file)?;
    let tokenizer_path = path.join(
        doc["gguf"]["tokenizer_files"]["tokenizer.json"]["path"]
            .as_str()
            .context("tokenizer reference")?,
    );
    let tokenizer = Tokenizer::from_dir(tokenizer_path.parent().context("tokenizer directory")?)?;
    let mut model = Model::from_gguf(file, tokenizer)?;
    let mut wanted = doc["config"]["resolved"].clone();
    surfaces::normalize_config_floats(&mut wanted);
    let mut actual = surfaces::config(model.config());
    actual["quantization"] = Value::Null;
    ensure!(
        wanted == actual,
        "resolved GGUF config: expected {wanted}, got {actual}"
    );
    ensure!(
        model.config().model_type.as_str()
            == doc["gguf"]["architecture"]
                .as_str()
                .context("architecture")?,
        "source architecture"
    );
    ensure!(
        !model.config().attention_bias && !model.config().mlp_bias,
        "unexpected bias config"
    );
    ensure!(
        model.config().dimensions.max_positions
            == Some(
                doc["config"]["bridge"]["max_position_embeddings"]
                    .as_u64()
                    .context("positions")? as usize
            ),
        "context length"
    );
    let groups = doc["config"]["quantization"]
        .as_object()
        .context("keyed quantization")?;
    if groups.is_empty() {
        ensure!(
            model.config().quantization.is_none(),
            "unexpected quantization config"
        );
    } else {
        let config = model
            .config()
            .quantization
            .as_ref()
            .context("missing quantization config")?;
        ensure!(
            config.layers.len() == groups.len(),
            "quantization key count"
        );
        let (_, first) = groups
            .iter()
            .find(|(_, value)| value.is_object())
            .context("first affine group")?;
        ensure!(
            config.default.bits == first["bits"].as_u64().context("default bits")? as u8
                && config.default.group_size.get() == 32,
            "first canonical group default"
        );
    }
    for (path, value) in doc["config"]["quantization"]
        .as_object()
        .context("keyed quantization")?
    {
        let config = model
            .config()
            .quantization
            .as_ref()
            .context("missing quantization")?;
        let canonical = if model.config().model_type.as_str() == "qwen3" {
            path.strip_prefix("model.").unwrap_or(path)
        } else {
            path
        };
        let choice = config.layers.get(&ParameterPath::new(canonical));
        if value == &Value::Bool(false) {
            ensure!(
                matches!(choice, Some(LayerQuantization::Unquantized)),
                "floating override {path}"
            );
            continue;
        }
        let q = match choice {
            Some(LayerQuantization::Affine(q)) => q,
            None => bail!("missing keyed override {path}"),
            Some(LayerQuantization::Unquantized) => {
                bail!("quantized group marked floating: {path}")
            }
        };
        ensure!(
            q.bits == value["bits"].as_u64().context("bits")? as u8 && q.group_size.get() == 32,
            "keyed affine layout {path}"
        );
    }
    let inputs: Value = serde_json::from_slice(&fs::read(path.join("inputs.json"))?)?;
    for (name, text) in inputs["prompts"].as_object().context("prompts")? {
        let ids: Vec<u32> = model
            .tokenizer()
            .encode(text.as_str().context("prompt")?)?
            .into_iter()
            .map(u32::from)
            .collect();
        ensure!(
            json!(ids) == doc["tokenizer"]["encodings"][name],
            "tokenizer prompt {name}"
        );
    }
    ensure!(
        json!(model
            .tokenizer()
            .eos_tokens()
            .iter()
            .copied()
            .map(u32::from)
            .collect::<Vec<_>>())
            == doc["tokenizer"]["eos_tokens"],
        "tokenizer EOS"
    );
    let bos = model.tokenizer().bos_token().context("tokenizer BOS")?;
    let bos_ids: Vec<u32> = model
        .tokenizer()
        .encode(bos)?
        .into_iter()
        .map(u32::from)
        .collect();
    ensure!(
        bos_ids
            == [doc["tokenizer"]["bos_token_id"]
                .as_u64()
                .context("BOS id")? as u32],
        "tokenizer BOS id"
    );
    let ids: Vec<u32> = serde_json::from_value(doc["prefill"]["token_ids"].clone())?;
    let prompt: Vec<_> = ids.into_iter().map(TokenId::from).collect();
    let decode: Vec<u32> = serde_json::from_value(doc["decode"]["greedy_ids"].clone())?;
    let mut observed = Observation::default();
    for chunk in [
        None,
        NonZeroUsize::new(1),
        NonZeroUsize::new(3),
        NonZeroUsize::new(8),
    ] {
        let (logits, mut session) = oracle_hooks::prefill_logits(&mut model, &prompt, chunk)?;
        let label = chunk.map_or_else(|| "full".to_owned(), |n| format!("chunk{n}"));
        record(
            &doc,
            &mut observed,
            format!("prefill.{label}.logits"),
            &logits,
        )?;
        if chunk.is_none() {
            cache(&doc, &mut observed, &session, "after_prefill")?;
            for (step, &token) in decode.iter().enumerate() {
                let logits = session
                    .decode_step(TokenId::from(token))?
                    .squeeze_axes(&[1])?;
                record(
                    &doc,
                    &mut observed,
                    format!("decode.step{step}.logits"),
                    &logits,
                )?;
            }
            cache(&doc, &mut observed, &session, "after_decode")?;
        }
    }
    let mut options = GenerationOptions::default();
    options.max_tokens = NonZeroUsize::new(8).unwrap();
    options.stop.tokens = StopTokenPolicy::Exact(vec![]);
    let mut generated = model.generate(Prompt::Tokens(&prompt), options)?;
    let mut greedy = Vec::new();
    for event in generated.by_ref() {
        if let GenerationEvent::Token {
            token_id,
            finish_reason,
            ..
        } = event?
        {
            greedy.push(u32::from(token_id));
            ensure!(
                finish_reason.is_some() == (greedy.len() == 8),
                "generation terminal boundary"
            );
            if greedy.len() == 8 {
                ensure!(
                    matches!(finish_reason, Some(FinishReason::Length)),
                    "generation finish reason"
                );
            }
        }
    }
    ensure!(
        generated.cache().tokens().len() == prompt.len() + 7,
        "generation one-token lag"
    );
    observed.greedy_ids = Some(greedy);
    prototype::compare("GGUF model", &expected(path, &doc)?, &observed)
}

pub fn run_all(device: Device) -> Result<()> {
    let q5 = reader::fixture_root().join("../../fixtures/gguf/q5-0-unsupported.gguf");
    let rejected = mlx_rs::with_device(Device::cpu(), || GgufFile::load(q5));
    ensure!(
        matches!(rejected, Err(mlx_rs::io::GgufError::Exception(_))),
        "Q5_0 must fail in core before Model construction"
    );
    for family in ["llama", "qwen3"] {
        for storage in FORMATS {
            let name = format!("gguf-{family}-{storage}");
            let path = reader::fixture_root().join(&name);
            ensure!(
                path.is_dir(),
                "NOT RUN: GGUF corpus missing {name}; host generation required"
            );
            let selected = Device::new(device.get_type()?, device.get_index()?);
            mlx_rs::with_device(selected, || run(&path)).with_context(|| name)?;
        }
    }
    Ok(())
}

pub fn qualify_mutations() -> Result<()> {
    for family in ["llama", "qwen3"] {
        for storage in FORMATS {
            let path = reader::fixture_root().join(format!("gguf-{family}-{storage}"));
            validate_fixture(&path)?;
            let doc = document(&path)?;
            let expected = expected(&path, &doc)?;
            let mutations = tensors(&path.join("mutations.safetensors"))?;
            for (name, recipe) in doc["mutations"].as_object().context("mutations")? {
                let witness = recipe["witness"].as_str().context("mutation witness")?;
                let mut observed = expected.observation.clone();
                observed.tensors.insert(
                    witness.into(),
                    mutations
                        .get(&format!("{name}::{witness}"))
                        .context("mutated output")?
                        .clone(),
                );
                let failures = comparator::compare(&expected, &observed);
                ensure!(
                    failures
                        .iter()
                        .any(|f| f.class.as_str() == recipe["class"].as_str().unwrap()),
                    "inert effective value mutation: {}/{name}",
                    path.display()
                );
                if doc["tolerance_policy"] == "gguf-affine-v1" {
                    let Policy::Float(tolerance) = expected.policies[witness] else {
                        bail!("affine mutation needs a floating policy");
                    };
                    let original = &expected.observation.tensors[witness];
                    let changed = &observed.tensors[witness];
                    ensure!(
                        original.shape == changed.shape
                            && original.dtype == changed.dtype
                            && original.valid_layout()
                            && changed.valid_layout(),
                        "affine mutation witness layout"
                    );
                    let mut margin = 0.0_f64;
                    for (a, b) in original.floats()?.into_iter().zip(changed.floats()?) {
                        ensure!(a.is_finite() && b.is_finite(), "nonfinite affine witness");
                        margin =
                            margin.max((a - b).abs() / (tolerance.atol + tolerance.rtol * a.abs()));
                    }
                    println!("mutation: gguf-{family}-{storage}/{name} witness={witness} margin={margin:.9}x");
                    ensure!(
                        margin >= 10.0,
                        "weak affine mutation gguf-{family}-{storage}/{name}: {margin}x < 10x; redesign witnesses"
                    );
                }
            }
            let mut actual = expected.observation.clone();
            actual
                .caches
                .get_mut("after_decode.layer0")
                .context("cache")?
                .offset += 1;
            ensure!(
                comparator::compare(&expected, &actual)
                    .iter()
                    .any(|f| f.class.as_str() == "cache_offset"),
                "cache offset mutation"
            );
            let mut actual = expected.observation.clone();
            actual
                .caches
                .get_mut("after_decode.layer0")
                .context("cache")?
                .retained
                .start += 1;
            ensure!(
                comparator::compare(&expected, &actual)
                    .iter()
                    .any(|f| f.class.as_str() == "cache_range"),
                "cache range mutation"
            );
            let converted = tensors(&path.join("converted.safetensors"))?;
            let converted_expected = Expectations {
                policies: converted
                    .keys()
                    .map(|key| (key.clone(), Policy::ExactBits))
                    .collect(),
                observation: Observation {
                    tensors: converted,
                    ..Observation::default()
                },
            };
            for (name, recipe) in doc["converted_mutations"]
                .as_object()
                .context("converted mutations")?
            {
                let key = recipe["key"].as_str().context("converted mutation key")?;
                let mut changed = converted_expected.observation.clone();
                if name == "drop_companion" {
                    changed.tensors.remove(key);
                } else {
                    let array = changed
                        .tensors
                        .get_mut(key)
                        .context("converted mutation tensor")?;
                    match name.as_str() {
                        "packed_lane" => array.bytes[0] ^= 1,
                        "signed_rebasing" => array.bytes[0] ^= 128,
                        "bias_sign" => {
                            for half in array.bytes.chunks_exact_mut(2) {
                                half[1] ^= 128;
                            }
                        }
                        "scale_row" => {
                            let stride = array.bytes.len() / array.shape[0];
                            array.bytes.rotate_right(stride);
                        }
                        other => bail!("unknown converted mutation {other}"),
                    }
                }
                ensure!(
                    comparator::compare(&converted_expected, &changed)
                        .iter()
                        .any(|failure| failure.class.as_str() == recipe["class"].as_str().unwrap()),
                    "inert converted mutation {name}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn contract_comparator_qualification() -> Result<()> {
    let cases: Value =
        serde_json::from_str(include_str!("../../../conformance/mlx-lm/gguf_cases.json"))?;
    let mut observation = Observation {
        config: Some(
            json!({"llama.block_count": {"kind": "array", "shape": [], "dtype": "uint32", "value": 2}}),
        ),
        tokenizer: Some(json!({"bos": 4, "eos": [2, 3], "canonical": [12, 14, 15]})),
        ..Observation::default()
    };
    for case in cases["cases"].as_array().context("GGUF error recipes")? {
        observation.errors.insert(
            case["id"].as_str().context("recipe id")?.into(),
            serde_json::to_string(&case["expected"])?,
        );
    }
    let expected = Expectations {
        observation,
        policies: BTreeMap::new(),
    };
    for (field, value) in [
        ("kind", json!("string")),
        ("shape", json!([1])),
        ("dtype", json!("float32")),
        ("value", json!(3)),
    ] {
        let mut changed = expected.observation.clone();
        changed.config.as_mut().unwrap()["llama.block_count"][field] = value;
        ensure!(
            comparator::compare(&expected, &changed)
                .iter()
                .any(|f| f.class.as_str() == "config"),
            "metadata comparator {field}"
        );
    }
    for (field, value) in [
        ("bos", json!(5)),
        ("eos", json!([4])),
        ("canonical", json!([13, 14, 15])),
    ] {
        let mut changed = expected.observation.clone();
        changed.tokenizer.as_mut().unwrap()[field] = value;
        ensure!(
            comparator::compare(&expected, &changed)
                .iter()
                .any(|f| f.class.as_str() == "tokenizer"),
            "tokenizer comparator {field}"
        );
    }
    for (key, error) in &expected.observation.errors {
        let mut changed = expected.observation.clone();
        let mut wrong: Value = serde_json::from_str(error)?;
        wrong["variant"] = json!("wrong_class");
        changed
            .errors
            .insert(key.clone(), serde_json::to_string(&wrong)?);
        ensure!(
            comparator::compare(&expected, &changed)
                .iter()
                .any(|f| f.class.as_str() == "error_class"),
            "error variant comparator {key}"
        );
        for field in ["key", "field", "token_id", "expected", "actual"] {
            let mut wrong: Value = serde_json::from_str(error)?;
            if wrong.get(field).is_some() {
                wrong[field] = json!("wrong_field");
                changed
                    .errors
                    .insert(key.clone(), serde_json::to_string(&wrong)?);
                ensure!(
                    comparator::compare(&expected, &changed)
                        .iter()
                        .any(|f| f.class.as_str() == "error_class"),
                    "error field comparator {key}/{field}"
                );
            }
        }
    }
    Ok(())
}
