use std::{collections::BTreeMap, fs, path::Path};

use anyhow::{bail, ensure, Context, Result};
use mlx_lm::{
    AttentionKind, ChatContinuation, ChatTemplateOptions, Config, ConfigError, GenerationEvent,
    GenerationOptions, LoadError, Message, Model, Prompt, Role, RopeScaling, TokenId, WeightError,
};
use safetensors::{tensor::TensorView, SafeTensors};
use serde_json::{json, Value};

use super::{
    observation::{Expectations, Observation, Tensor},
    reader::Fixture,
};

pub fn expectations(fixture: &Fixture) -> Result<Expectations> {
    let mut expected = fixture.expected.clone();
    let observation = &mut expected.observation;
    observation.processing = None;
    observation.processor_histories.clear();
    observation.text_cases = None;
    observation.trim_after_wrap = None;
    observation.progress = None;
    let tokenizer = observation.tokenizer.as_mut().context("tokenizer")?;
    tokenizer
        .as_object_mut()
        .context("tokenizer object")?
        .remove("eos_source");
    tokenizer["decodings"]
        .as_object_mut()
        .context("decodings")?
        .remove("special_skipped");
    // Resolved config scalars are f32 in the public API, including the normalization epsilon.
    normalize_config_floats(observation.config.as_mut().context("config")?);
    Ok(expected)
}

fn normalize_config_floats(value: &mut Value) {
    match value {
        Value::Number(number) if number.is_f64() => *value = json!(number.as_f64().unwrap() as f32),
        Value::Object(fields) => fields.values_mut().for_each(normalize_config_floats),
        Value::Array(values) => values.iter_mut().for_each(normalize_config_floats),
        _ => (),
    }
}

fn config(config: &Config) -> Value {
    let d = &config.dimensions;
    let rope = &config.rope;
    let scaling = match rope.scaling {
        RopeScaling::None => Value::Null,
        RopeScaling::Linear { factor } => json!({"rope_type": "linear", "factor": factor}),
        RopeScaling::Llama3 {
            factor,
            low_frequency_factor,
            high_frequency_factor,
            original_max_positions,
        } => {
            json!({"rope_type": "llama3", "factor": factor, "low_freq_factor": low_frequency_factor, "high_freq_factor": high_frequency_factor, "original_max_position_embeddings": original_max_positions})
        }
    };
    let attention: Vec<_> = config
        .attention
        .iter()
        .map(|kind| match kind {
            AttentionKind::Full => json!("full"),
            AttentionKind::Sliding { window } => json!({"sliding": window.get()}),
        })
        .collect();
    json!({
        "hidden_size": d.hidden_size, "layer_count": d.layer_count,
        "intermediate_size": d.intermediate_size, "attention_heads": d.attention_heads,
        "kv_heads": d.kv_heads, "head_dim": d.head_dim, "vocabulary_size": d.vocabulary_size,
        "rms_norm_epsilon": d.rms_norm_epsilon, "attention_kinds": attention,
        "tie_word_embeddings": config.tie_word_embeddings,
        "rope": {"dimensions": rope.dimensions, "theta": rope.theta, "traditional": rope.traditional, "scaling": scaling},
        "quantization": config.quantization.as_ref().map(|q| json!({"group_size": q.default.group_size.get(), "bits": q.default.bits})),
    })
}

fn ids(tokens: &[TokenId]) -> Vec<u32> {
    tokens.iter().copied().map(u32::from).collect()
}

fn tokenizer(model: &mut Model, fixture: &Fixture) -> Result<Value> {
    let mut encodings = serde_json::Map::new();
    let mut decodings = serde_json::Map::new();
    for (name, prompt) in fixture.inputs["prompts"].as_object().context("prompts")? {
        let encoded = model
            .tokenizer()
            .encode(prompt.as_str().context("prompt text")?)?;
        encodings.insert(name.clone(), json!(ids(&encoded)));
        let input_ids: Vec<u32> =
            serde_json::from_value(fixture.inputs["token_ids"][name].clone())?;
        let input_ids: Vec<_> = input_ids.into_iter().map(TokenId::from).collect();
        decodings.insert(name.clone(), json!(model.tokenizer().decode(&input_ids)?));
    }
    let special = fixture.inputs["prompts"]["special"]
        .as_str()
        .context("special prompt")?;
    encodings.insert(
        "special_with_defaults".into(),
        json!(ids(&model
            .tokenizer()
            .encode_with_special_tokens(special, true)?)),
    );
    let mut observed = json!({"encodings": encodings, "decodings": decodings, "eos_tokens": ids(model.tokenizer().eos_tokens())});
    if fixture
        .expected
        .observation
        .tokenizer
        .as_ref()
        .context("tokenizer")?
        .get("prompt_encoding")
        .is_some()
    {
        let canonical = fixture.inputs["prompts"][&fixture.prefill.prompt]
            .as_str()
            .context("canonical prompt")?;
        let bos = model
            .tokenizer()
            .bos_token()
            .context("BOS token")?
            .to_owned();
        let mut encodings = serde_json::Map::new();
        for (name, text) in [
            ("canonical", canonical.to_owned()),
            ("canonical_with_bos", format!("{bos}{canonical}")),
            ("whitespace_before_bos", format!(" {bos}{canonical}")),
        ] {
            let mut generation =
                model.generate(Prompt::Text(&text), GenerationOptions::default())?;
            loop {
                match generation.next().context("missing prompt prefill")?? {
                    GenerationEvent::Prefill {
                        processed, total, ..
                    } if processed == total => break,
                    GenerationEvent::Prefill { .. } => (),
                    other => bail!("token before final prompt prefill: {other:?}"),
                }
            }
            encodings.insert(name.into(), json!(ids(generation.cache().tokens())));
        }
        observed["prompt_encoding"] = Value::Object(encodings);
    }
    Ok(observed)
}

fn chat(model: &Model, fixture: &Fixture) -> Result<Value> {
    let mut observed = serde_json::Map::new();
    for (name, case) in fixture
        .expected
        .observation
        .chat
        .as_ref()
        .context("chat")?
        .as_object()
        .context("chat object")?
    {
        let messages = case["messages"]
            .as_array()
            .context("messages")?
            .iter()
            .map(|message| {
                let role = match message["role"].as_str().context("message role")? {
                    "system" => Role::System,
                    "user" => Role::User,
                    "assistant" => Role::Assistant,
                    "tool" => Role::Tool,
                    other => Role::Other(other.into()),
                };
                Ok(Message {
                    role,
                    content: message["content"]
                        .as_str()
                        .context("message content")?
                        .into(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let continuation = match case["continuation"].as_str().context("continuation")? {
            "closed" => ChatContinuation::Closed,
            "start_assistant" => ChatContinuation::StartAssistant,
            "continue_last" => ChatContinuation::ContinueLast,
            other => bail!("unknown continuation {other}"),
        };
        let rendered = model.tokenizer().render_chat(
            &messages,
            ChatTemplateOptions {
                continuation,
                enable_thinking: case["template_options"]["enable_thinking"].as_bool(),
            },
        )?;
        let hex: String = rendered
            .as_bytes()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let mut result = json!({"messages": messages, "continuation": case["continuation"], "rendered_utf8_hex": hex, "token_ids": ids(&model.tokenizer().encode(&rendered)?)});
        if let Some(options) = case.get("template_options") {
            result["template_options"] = options.clone();
        }
        observed.insert(name.clone(), result);
    }
    Ok(Value::Object(observed))
}

fn read_tensors(path: &Path) -> Result<BTreeMap<String, Tensor>> {
    let bytes = fs::read(path)?;
    Ok(SafeTensors::deserialize(&bytes)?
        .tensors()
        .into_iter()
        .map(|(name, tensor)| {
            (
                name,
                Tensor {
                    shape: tensor.shape().to_vec(),
                    dtype: tensor.dtype(),
                    bytes: tensor.data().to_vec(),
                },
            )
        })
        .collect())
}

fn write_tensors(path: &Path, tensors: &BTreeMap<String, Tensor>) -> Result<()> {
    let views = tensors
        .iter()
        .map(|(name, tensor)| {
            Ok((
                name.as_str(),
                TensorView::new(tensor.dtype, tensor.shape.clone(), &tensor.bytes)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    safetensors::serialize_to_file(views, None, path)?;
    Ok(())
}

fn split_checkpoint(path: &Path) -> Result<()> {
    let single = path.join("model.safetensors");
    if !single.exists() {
        return Ok(());
    }
    let tensors = read_tensors(&single)?;
    let mut shards = [BTreeMap::new(), BTreeMap::new()];
    let names = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ];
    let mut index = serde_json::Map::new();
    for (i, (name, tensor)) in tensors.into_iter().enumerate() {
        index.insert(name.clone(), json!(names[i % 2]));
        shards[i % 2].insert(name, tensor);
    }
    for (name, shard) in names.into_iter().zip(&shards) {
        write_tensors(&path.join(name), shard)?;
    }
    fs::write(
        path.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({"weight_map": index}))?,
    )?;
    fs::remove_file(single)?;
    Ok(())
}

fn errors(path: &Path, fixture: &Fixture) -> Result<BTreeMap<String, String>> {
    let mut observed = BTreeMap::new();
    for name in fixture.expected.observation.errors.keys() {
        let copy = tempfile::tempdir()?;
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                fs::copy(entry.path(), copy.path().join(entry.file_name()))?;
            }
        }
        if matches!(name.as_str(), "missing_shard" | "duplicate_tensor") {
            split_checkpoint(copy.path())?;
        }
        let config_path = copy.path().join("config.json");
        let mut config: Value = serde_json::from_slice(&fs::read(&config_path)?)?;
        match name.as_str() {
            "unknown_model_type" => config["model_type"] = json!("unsupported_oracle_model"),
            "unsupported_rope" => {
                config["rope_scaling"] = json!({"rope_type": "unsupported_oracle_rope"})
            }
            "bad_quantization_bits" => {
                config["quantization"] = json!({"group_size": 32, "bits": 5})
            }
            "missing_shard" => {
                fs::remove_file(copy.path().join("model-00002-of-00002.safetensors"))?
            }
            "duplicate_tensor" => {
                let first = read_tensors(&copy.path().join("model-00001-of-00002.safetensors"))?;
                let second_path = copy.path().join("model-00002-of-00002.safetensors");
                let mut second = read_tensors(&second_path)?;
                let (key, tensor) = first.into_iter().next().context("first tensor")?;
                second.insert(key, tensor);
                write_tensors(&second_path, &second)?;
            }
            "wrong_shape" => {
                let key = "model.layers.0.self_attn.q_proj.weight";
                let mut changed = false;
                for entry in fs::read_dir(copy.path())? {
                    let path = entry?.path();
                    if path.extension().is_some_and(|ext| ext == "safetensors")
                        && path
                            .file_name()
                            .is_some_and(|name| name.to_string_lossy().starts_with("model"))
                    {
                        let mut tensors = read_tensors(&path)?;
                        if let Some(tensor) = tensors.get_mut(key) {
                            let row_bytes = tensor.bytes.len() / tensor.shape[0];
                            tensor.shape[0] -= 1;
                            tensor.bytes.truncate(tensor.bytes.len() - row_bytes);
                            write_tensors(&path, &tensors)?;
                            changed = true;
                        }
                    }
                }
                ensure!(changed, "missing weight to mutate");
            }
            other => bail!("unknown error recipe {other}"),
        }
        fs::write(config_path, serde_json::to_vec(&config)?)?;
        let error = match Model::from_dir(copy.path()) {
            Ok(_) => bail!("accepted {name} mutation"),
            Err(error) => error,
        };
        let class = match error {
            LoadError::Config(ConfigError::UnsupportedArchitecture(_)) => {
                "ConfigError::UnsupportedArchitecture"
            }
            LoadError::Config(ConfigError::UnsupportedRope(_)) => "ConfigError::UnsupportedRope",
            LoadError::Config(ConfigError::UnsupportedQuantization(_)) => {
                "ConfigError::UnsupportedQuantization"
            }
            LoadError::Weights(WeightError::MissingShard(_)) => "WeightError::MissingShard",
            LoadError::Weights(WeightError::DuplicateTensor(_)) => "WeightError::DuplicateTensor",
            LoadError::Weights(WeightError::ShapeMismatch { .. }) => "WeightError::ShapeMismatch",
            other => bail!("unexpected {name} error: {other:?}"),
        };
        observed.insert(name.clone(), class.into());
    }
    Ok(observed)
}

pub fn run(path: &Path, fixture: &Fixture, model: &mut Model) -> Result<Observation> {
    Ok(Observation {
        config: Some(config(model.config())),
        tokenizer: Some(tokenizer(model, fixture)?),
        chat: Some(chat(model, fixture)?),
        errors: errors(path, fixture)?,
        ..Observation::default()
    })
}
