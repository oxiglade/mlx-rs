use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{bail, ensure, Context, Result};
use safetensors::SafeTensors;
use serde::Deserialize;
use serde_json::Value;

use super::observation::{CacheState, Expectations, Observation, Policy, Tensor, Tolerance};

#[derive(Deserialize)]
struct Document {
    schema_version: u32,
    provenance: Value,
    config: Config,
    tokenizer: Value,
    chat: Value,
    prefill: Prefill,
    decode: Decode,
    sampling: BTreeMap<String, Sampling>,
    errors: BTreeMap<String, ErrorCase>,
    tolerances: BTreeMap<String, Tolerance>,
    #[serde(default)]
    processing: Option<Value>,
    #[serde(default)]
    cache: Value,
}

#[derive(Deserialize)]
struct Config {
    resolved: Value,
}

#[derive(Deserialize)]
pub struct Prefill {
    pub prompt: String,
    pub token_ids: Vec<u32>,
    #[serde(rename = "T")]
    pub length: usize,
    #[serde(default)]
    pub progress: Option<Value>,
}

#[derive(Deserialize)]
struct Decode {
    greedy_ids: Vec<u32>,
    text_deltas: Vec<String>,
    finish_reason: String,
    stop_token: Option<u32>,
}

#[derive(Deserialize)]
struct Sampling {
    options: Value,
    seed: u64,
    cpu_ids: Vec<u32>,
    #[serde(default)]
    processor_token_histories: Vec<Vec<u32>>,
}

#[derive(Deserialize)]
struct ErrorCase {
    mutation: String,
    expected_rust_error: String,
}

pub struct Fixture {
    pub expected: Expectations,
    pub prefill: Prefill,
    pub inputs: Value,
}

pub fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../conformance/mlx-lm/fixtures")
}

pub fn present(path: &Path) -> Result<bool> {
    match fs::metadata(path) {
        Ok(metadata) => {
            ensure!(
                metadata.is_dir(),
                "fixture path is not a directory: {}",
                path.display()
            );
            Ok(true)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            // Direct stderr survives libtest capture so an absent corpus cannot look like parity evidence.
            use std::io::Write;
            writeln!(
                std::io::stderr(),
                "NOT RUN: mlx-lm fixture test; missing {}",
                path.display()
            )?;
            Ok(false)
        }
        Err(error) => Err(error.into()),
    }
}

pub fn read(path: &Path) -> Result<Fixture> {
    let document: Document = serde_json::from_slice(&fs::read(path.join("expectations.json"))?)?;
    let inputs: Value = serde_json::from_slice(&fs::read(path.join("inputs.json"))?)?;
    ensure!(document.schema_version == 1, "unsupported fixture schema");
    ensure!(
        document.provenance["mlx_lm"] == "0.31.3",
        "unrecognized mlx_lm pin"
    );
    ensure!(inputs.is_object(), "inputs must be an object");
    ensure!(
        document.prefill.length > 0 && document.prefill.length == document.prefill.token_ids.len(),
        "prefill T disagrees with token IDs"
    );
    ensure!(
        !document.prefill.prompt.is_empty(),
        "missing canonical prompt name"
    );
    ensure!(
        document.decode.greedy_ids.len() == 8,
        "expected eight greedy decode IDs"
    );
    ensure!(
        matches!(document.decode.finish_reason.as_str(), "length" | "stop"),
        "invalid finish reason"
    );
    for name in ["logits", "cache", "logprobs"] {
        let policy = document
            .tolerances
            .get(name)
            .with_context(|| format!("missing tolerance {name}"))?;
        ensure!(
            policy.atol.is_finite()
                && policy.rtol.is_finite()
                && policy.atol >= 0.0
                && policy.rtol >= 0.0,
            "invalid tolerance {name}"
        );
    }
    let text_path = path.join("text_cases.json");
    let text_cases = match fs::read(&text_path) {
        Ok(bytes) => {
            let mut text: Value = serde_json::from_slice(&bytes)?;
            ensure!(text["schema_version"] == 1, "unsupported text schema");
            ensure!(
                text["stop_strings"].is_object() && text["decoders"].is_object(),
                "missing text cohorts"
            );
            text.as_object_mut()
                .context("text document must be an object")?
                .remove("provenance");
            Some(text)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    let mut observation = Observation {
        config: Some(document.config.resolved.clone()),
        tokenizer: Some(document.tokenizer),
        chat: Some(document.chat),
        processing: document.processing,
        progress: document.prefill.progress.clone(),
        trim_after_wrap: document.cache.get("trim_after_wrap").cloned(),
        text_cases,
        greedy_ids: Some(document.decode.greedy_ids),
        text_deltas: Some(document.decode.text_deltas),
        finish: Some((document.decode.finish_reason, document.decode.stop_token)),
        ..Observation::default()
    };
    for (name, case) in document.sampling {
        ensure!(
            case.options.is_object(),
            "sampling options must be an object: {name}"
        );
        let _seed = case.seed;
        ensure!(
            case.cpu_ids.len() == 8,
            "sampling {name} requires eight CPU IDs"
        );
        if !case.processor_token_histories.is_empty() {
            ensure!(
                case.processor_token_histories.len() == 8,
                "sampling {name} requires eight processor histories"
            );
            observation
                .processor_histories
                .insert(name.clone(), case.processor_token_histories);
        }
        observation.sampled_ids.insert(name, case.cpu_ids);
    }
    for (name, case) in document.errors {
        ensure!(
            !case.mutation.is_empty() && !case.expected_rust_error.is_empty(),
            "incomplete error case {name}"
        );
        observation.errors.insert(name, case.expected_rust_error);
    }
    let bytes = fs::read(path.join("expectations.safetensors"))?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let mut policies = BTreeMap::new();
    for (key, tensor) in tensors.tensors() {
        let policy = if key.starts_with("quant.") || key.starts_with("cache.trim_after_wrap.") {
            Policy::ExactBits
        } else {
            let name = if key.starts_with("cache.") {
                "cache"
            } else if key.starts_with("sampling.") && key.ends_with(".filtered_logprobs") {
                "logprobs"
            } else if (key.starts_with("prefill.")
                || key.starts_with("decode.step")
                || key.starts_with("processing."))
                && key.ends_with(".logits")
            {
                "logits"
            } else {
                bail!("unknown expectation tensor {key}");
            };
            Policy::Float(document.tolerances[name])
        };
        policies.insert(key.clone(), policy);
        observation.tensors.insert(
            key,
            Tensor {
                shape: tensor.shape().to_vec(),
                dtype: tensor.dtype(),
                bytes: tensor.data().to_vec(),
            },
        );
    }
    if let Some(processing) = &observation.processing {
        for (name, case) in processing
            .as_object()
            .context("processing must be an object")?
        {
            let rows = case["histories"]
                .as_array()
                .context("missing processor histories")?
                .len();
            let columns = case["input_logits"]
                .as_array()
                .context("missing processor logits")?
                .len();
            let key = format!("processing.{name}.logits");
            let tensor = observation
                .tensors
                .get(&key)
                .with_context(|| format!("missing {key}"))?;
            ensure!(
                rows > 0 && columns > 0 && tensor.shape == [rows, columns],
                "invalid processor table {key}"
            );
        }
    }
    if observation.trim_after_wrap.is_some() {
        for stage in ["before_trim", "after_trim", "after_append"] {
            for kind in ["raw_keys", "raw_values", "temporal_keys", "temporal_values"] {
                let key = format!("cache.trim_after_wrap.{stage}.{kind}");
                let tensor = observation
                    .tensors
                    .get(&key)
                    .with_context(|| format!("missing {key}"))?;
                ensure!(
                    tensor.shape == [1, 1, 5, 1],
                    "invalid wrapped cache tensor {key}"
                );
            }
        }
    }
    let kinds = document.config.resolved["attention_kinds"]
        .as_array()
        .context("missing attention kinds")?;
    let layers = document.config.resolved["layer_count"]
        .as_u64()
        .context("missing layer count")? as usize;
    ensure!(
        kinds.len() == layers && layers > 0,
        "attention kinds disagree with layer count"
    );
    for (layer, kind) in kinds.iter().enumerate() {
        let window = if kind == "full" {
            None
        } else {
            let window = kind["sliding"].as_u64().context("invalid attention kind")? as usize;
            ensure!(window > 0, "sliding window must be positive");
            Some(window)
        };
        for (stage, offset) in [
            ("after_prefill", document.prefill.length),
            ("after_decode", document.prefill.length + 8),
        ] {
            let retained = if stage == "after_prefill" {
                offset
            } else {
                window.unwrap_or(offset).min(offset)
            };
            let key = format!("cache.{stage}.layer{layer}");
            for suffix in ["keys", "values"] {
                let tensor = observation
                    .tensors
                    .get(&format!("{key}.{suffix}"))
                    .with_context(|| format!("missing {key}.{suffix}"))?;
                ensure!(
                    tensor.shape.len() == 4 && tensor.shape[2] == retained,
                    "invalid logical cache range for {key}.{suffix}"
                );
            }
            observation.caches.insert(
                key,
                CacheState {
                    offset,
                    retained: offset - retained..offset,
                },
            );
        }
    }
    for key in std::iter::once("prefill.full.logits".to_owned())
        .chain([1, 3, document.prefill.length].map(|n| format!("prefill.chunk{n}.logits")))
        .chain((0..8).map(|n| format!("decode.step{n}.logits")))
        .chain(
            observation
                .sampled_ids
                .keys()
                .map(|name| format!("sampling.{name}.filtered_logprobs")),
        )
    {
        ensure!(
            observation.tensors.contains_key(&key),
            "missing tensor {key}"
        );
    }
    Ok(Fixture {
        expected: Expectations {
            observation,
            policies,
        },
        prefill: document.prefill,
        inputs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::{tensor::TensorView, Dtype};
    use serde_json::json;

    #[test]
    fn real_layout_reads_sliding_prefill_and_temporal_decode_ranges() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path();
        let mut document = json!({
            "schema_version": 1, "provenance": {"mlx_lm": "0.31.3"},
            "config": {"resolved": {"layer_count": 2, "attention_kinds": ["full", {"sliding": 4}]}},
            "tokenizer": {"encodings": {"prompt": [1, 2, 3, 4, 5]}, "decodings": {"prompt": "a"}, "eos_tokens": [0], "eos_source": "generation_config"},
            "chat": {}, "prefill": {"prompt": "prompt", "token_ids": [1, 2, 3, 4, 5], "T": 5},
            "decode": {"greedy_ids": [1, 1, 1, 1, 1, 1, 1, 1], "text_deltas": ["a", ""], "finish_reason": "length", "stop_token": null},
            "sampling": {}, "errors": {},
            "processing": {"presence_positive": {"input_logits": [0, 2], "histories": [[1]], "options": {"presence_penalty": 0.5, "presence_context_size": 3}}},
            "cache": {"trim_after_wrap": {"capacity": 5, "keep": 2, "trim_return": 0}},
            "tolerances": {"logits": {"atol": 0.0002, "rtol": 0.0002}, "cache": {"atol": 0.0002, "rtol": 0.0002}, "logprobs": {"atol": 0.0001, "rtol": 0.0001}}
        });
        fs::write(path.join("inputs.json"), b"{\"seeds\": [0]}")?;
        fs::write(
            path.join("expectations.json"),
            serde_json::to_vec(&document)?,
        )?;
        let mut arrays: BTreeMap<String, (Vec<usize>, Vec<u8>)> = BTreeMap::new();
        arrays.insert(
            "processing.presence_positive.logits".into(),
            (vec![1, 2], vec![0; 8]),
        );
        for stage in ["before_trim", "after_trim", "after_append"] {
            for kind in ["raw_keys", "raw_values", "temporal_keys", "temporal_values"] {
                arrays.insert(
                    format!("cache.trim_after_wrap.{stage}.{kind}"),
                    (vec![1, 1, 5, 1], vec![0; 20]),
                );
            }
        }
        for key in [
            "prefill.full.logits",
            "prefill.chunk1.logits",
            "prefill.chunk3.logits",
            "prefill.chunk5.logits",
        ] {
            arrays.insert(key.into(), (vec![1, 5, 2], vec![0; 40]));
        }
        for step in 0..8 {
            arrays.insert(
                format!("decode.step{step}.logits"),
                (vec![1, 2], vec![0; 8]),
            );
        }
        for layer in 0..2 {
            for (stage, length) in [
                ("after_prefill", 5),
                ("after_decode", if layer == 0 { 13 } else { 4 }),
            ] {
                for suffix in ["keys", "values"] {
                    arrays.insert(
                        format!("cache.{stage}.layer{layer}.{suffix}"),
                        (vec![1, 1, length, 1], vec![0; length * 4]),
                    );
                }
            }
        }
        let views = arrays
            .iter()
            .map(|(key, (shape, bytes))| {
                Ok((
                    key.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), bytes)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        fs::write(
            path.join("expectations.safetensors"),
            safetensors::serialize(views, None)?,
        )?;
        let fixture = read(path)?;
        assert!(fixture.expected.observation.processing.is_some());
        assert!(fixture.expected.observation.trim_after_wrap.is_some());
        assert!(matches!(
            fixture.expected.policies["cache.trim_after_wrap.after_trim.raw_keys"],
            Policy::ExactBits
        ));
        let caches = &fixture.expected.observation.caches;
        assert_eq!(
            caches["cache.after_prefill.layer1"],
            CacheState {
                offset: 5,
                retained: 0..5
            }
        );
        assert_eq!(
            caches["cache.after_decode.layer1"],
            CacheState {
                offset: 13,
                retained: 9..13
            }
        );
        assert_eq!(
            caches["cache.after_decode.layer0"],
            CacheState {
                offset: 13,
                retained: 0..13
            }
        );
        assert!(super::super::comparator::compare(
            &fixture.expected,
            &fixture.expected.observation
        )
        .is_empty());
        document["tolerances"]
            .as_object_mut()
            .unwrap()
            .remove("cache");
        fs::write(
            path.join("expectations.json"),
            serde_json::to_vec(&document)?,
        )?;
        assert!(read(path).is_err());
        Ok(())
    }
}
