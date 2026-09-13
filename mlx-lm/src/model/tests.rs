use super::*;

#[test]
fn public_surface_defaults() {
    let options = GenerationOptions::default();
    assert_eq!(options.max_tokens.get(), 256);
    assert_eq!(options.prefill_chunk_size.get(), 2048);
    assert_eq!(options.sampling.temperature, 0.0);
    assert!(options.sampling.top_p.is_none());
    assert!(options.sampling.top_k.is_none());
    assert!(options.sampling.min_p.is_none());
    assert!(options.sampling.seed.is_none());
    assert!(options.repetition_penalty.is_none());
    assert!(options.presence_penalty.is_none());
    assert!(options.frequency_penalty.is_none());
    assert!(matches!(options.stop.tokens, StopTokenPolicy::Tokenizer));
    assert!(options.stop.stop_strings.is_empty());
    assert!(matches!(
        options.cache.policy,
        crate::CachePolicy::ModelDefault
    ));
    let stop = StopPolicy::default();
    assert!(matches!(stop.tokens, StopTokenPolicy::Tokenizer));
    assert!(stop.stop_strings.is_empty());
}

#[test]
fn options_and_events_can_cross_threads() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<GenerationOptions>();
    assert_send_sync::<GenerationEvent>();
}

#[test]
fn eos_precedence_generation_config_over_config_over_tokenizer() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../conformance/mlx-lm/fixtures/llama-base");
    for name in ["tokenizer.json", "tokenizer_config.json"] {
        std::fs::copy(fixture.join(name), directory.path().join(name))?;
    }
    let config = directory.path().join("config.json");
    let generation_config = directory.path().join("generation_config.json");
    std::fs::write(&config, br#"{"model_type":"llama","eos_token_id":9}"#)?;
    std::fs::write(&generation_config, br#"{"eos_token_id":[3,2]}"#)?;
    assert_eq!(
        Tokenizer::from_dir(directory.path())?.eos_tokens(),
        &[TokenId::from(2), TokenId::from(3)]
    );
    std::fs::remove_file(&generation_config)?;
    assert_eq!(
        Tokenizer::from_dir(directory.path())?.eos_tokens(),
        &[TokenId::from(9)]
    );
    std::fs::remove_file(&config)?;
    let tokenizer = Tokenizer::from_dir(directory.path())?;
    let metadata: serde_json::Value = serde_json::from_slice(&std::fs::read(
        directory.path().join("tokenizer_config.json"),
    )?)?;
    let eos = metadata["eos_token"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("fixture eos_token must be a string"))?;
    let expected = tokenizer.encode(eos)?;
    assert_eq!(expected.len(), 1);
    assert_eq!(tokenizer.eos_tokens(), expected);

    std::fs::write(&config, br#"{"model_type":"llama","eos_token_id":9}"#)?;
    std::fs::write(&generation_config, br#"{"eos_token_id":null}"#)?;
    // mlx_lm treats a falsy generation_config eos_token_id as absent and falls
    // through to config.json.
    assert_eq!(
        Tokenizer::from_dir(directory.path())?.eos_tokens(),
        &[TokenId::from(9)]
    );
    Ok(())
}

#[test]
fn local_loading_matches_fixture_expectations() -> anyhow::Result<()> {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../conformance/mlx-lm/fixtures");
    let mut count = 0;
    let mut not_run = 0;
    for entry in std::fs::read_dir(fixtures)? {
        let path = entry?.path();
        // GGUF fixtures carry no config.json; they are covered by the gguf module tests.
        if !path.is_dir() || !path.join("config.json").exists() {
            continue;
        }
        count += 1;
        let model = match Model::from_dir(&path) {
            Ok(model) => model,
            Err(LoadError::Weights(crate::WeightError::UnsupportedFormat(message)))
            | Err(LoadError::Config(crate::ConfigError::UnsupportedArchitecture(message)))
                if message.ends_with("not yet implemented in this tranche") =>
            {
                eprintln!(
                    "NOT RUN: Model::from_dir happy path for {}: {message}",
                    path.file_name().unwrap_or_default().to_string_lossy()
                );
                not_run += 1;
                continue;
            }
            Err(error) => return Err(error.into()),
        };
        let config =
            crate::config::RawConfig::from_bytes(&std::fs::read(path.join("config.json"))?)?
                .resolve()?;
        assert_eq!(model.config(), &config, "{}", path.display());
        let expectations: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.join("expectations.json"))?)?;
        let mut expected: Vec<u32> =
            serde_json::from_value(expectations["tokenizer"]["eos_tokens"].clone())?;
        let mut actual: Vec<u32> = model
            .tokenizer()
            .eos_tokens()
            .iter()
            .copied()
            .map(u32::from)
            .collect();
        expected.sort_unstable();
        actual.sort_unstable();
        assert_eq!(actual, expected, "{}", path.display());
    }
    assert!(count > 0, "no fixture directories found");
    if not_run > 0 {
        eprintln!("NOT RUN: Model::from_dir happy path for {not_run}/{count} fixtures");
        anyhow::bail!(
            "NOT RUN: {not_run} fixtures still require architecture or loader implementation"
        );
    }
    Ok(())
}

#[test]
fn local_loading_rejects_missing_and_invalid_config_before_allocation() -> Result<(), LoadError> {
    let directory = tempfile::tempdir()?;
    assert!(
        matches!(Model::from_dir(directory.path()), Err(LoadError::MissingFile(path)) if path == directory.path().join("config.json"))
    );
    let path = directory.path().join("config.json");
    std::fs::write(&path, b"{")?;
    assert!(matches!(
        Model::from_dir(directory.path()),
        Err(LoadError::Config(crate::ConfigError::Json(_)))
    ));
    std::fs::write(&path, b"{}")?;
    assert!(
        matches!(Model::from_dir(directory.path()), Err(LoadError::Config(crate::ConfigError::MissingField { field })) if field == "model_type")
    );
    std::fs::write(&path, br#"{"model_type":"unsupported_oracle_model"}"#)?;
    assert!(
        matches!(Model::from_dir(directory.path()), Err(LoadError::Config(crate::ConfigError::UnsupportedArchitecture(name))) if name == "unsupported_oracle_model")
    );
    let mut config: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../../conformance/mlx-lm/fixtures/llama-base/config.json"
    ))
    .map_err(crate::ConfigError::from)?;
    config["num_attention_heads"] = serde_json::json!(0);
    std::fs::write(
        path,
        serde_json::to_vec(&config).map_err(crate::ConfigError::from)?,
    )?;
    assert!(
        matches!(Model::from_dir(directory.path()), Err(LoadError::Config(crate::ConfigError::InvalidNumericField { field, .. })) if field == "num_attention_heads")
    );
    Ok(())
}

#[test]
fn local_loading_reads_tokenizer_metadata_before_building() -> Result<(), LoadError> {
    let directory = tempfile::tempdir()?;
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../conformance/mlx-lm/fixtures/llama-base");
    std::fs::copy(
        fixture.join("config.json"),
        directory.path().join("config.json"),
    )?;
    assert!(
        matches!(Model::from_dir(directory.path()), Err(LoadError::MissingFile(path)) if path == directory.path().join("tokenizer.json"))
    );
    std::fs::copy(
        fixture.join("tokenizer.json"),
        directory.path().join("tokenizer.json"),
    )?;
    std::fs::write(directory.path().join("tokenizer_config.json"), b"{")?;
    assert!(matches!(
        Model::from_dir(directory.path()),
        Err(LoadError::Tokenizer(crate::TokenizerError::Json(_)))
    ));
    std::fs::copy(
        fixture.join("tokenizer_config.json"),
        directory.path().join("tokenizer_config.json"),
    )?;
    std::fs::write(
        directory.path().join("generation_config.json"),
        br#"{"eos_token_id":-1}"#,
    )?;
    assert!(matches!(
        Model::from_dir(directory.path()),
        Err(LoadError::Tokenizer(crate::TokenizerError::InvalidEos(_)))
    ));
    std::fs::write(directory.path().join("generation_config.json"), b"{")?;
    assert!(matches!(
        Model::from_dir(directory.path()),
        Err(LoadError::Tokenizer(crate::TokenizerError::Json(_)))
    ));
    Ok(())
}

#[test]
fn repetition_penalty_range() -> Result<(), crate::SamplingError> {
    for penalty in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0, 0.0] {
        let options = RepetitionPenaltyOptions {
            penalty,
            context_size: NonZeroUsize::MIN,
        };
        assert!(matches!(
            options.validate(),
            Err(crate::SamplingError::InvalidRepetitionPenalty(_))
        ));
    }
    for penalty in [f32::MIN_POSITIVE, 0.5, 1.0, 2.0, f32::MAX] {
        RepetitionPenaltyOptions {
            penalty,
            context_size: NonZeroUsize::MIN,
        }
        .validate()?;
    }
    Ok(())
}

use crate::{
    cache::{CacheStep, LayerCacheSpec},
    InferenceError, SamplingError, TokenizerError,
};
use mlx_rs::{with_device, Device, Dtype};
use serde_json::{json, Value};

type TestResult = anyhow::Result<()>;
const FIXTURES: [&str; 6] = [
    "llama-base",
    "llama-sliding",
    "llama-quant4",
    "qwen3-base",
    "llama-sharded",
    "qwen3-quant4",
];
fn fixture(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../conformance/mlx-lm/fixtures")
        .join(name)
}
fn read_json(path: impl AsRef<Path>) -> anyhow::Result<Value> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}
fn ids(value: &Value) -> Vec<TokenId> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|id| TokenId::from(id.as_u64().unwrap() as u32))
        .collect()
}
fn nz(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).unwrap()
}
fn length_options(length: usize) -> GenerationOptions {
    GenerationOptions {
        max_tokens: nz(length),
        stop: StopPolicy {
            tokens: StopTokenPolicy::Exact(vec![]),
            stop_strings: vec![],
        },
        ..Default::default()
    }
}
fn collect_tokens(generation: &mut Generation<'_>) -> anyhow::Result<Vec<GenerationEvent>> {
    let mut tokens = vec![];
    let mut terminal = false;
    while let Some(event) = generation.next() {
        let event = event?;
        assert!(!terminal);
        if let GenerationEvent::Token {
            token_id,
            finish_reason,
            ..
        } = &event
        {
            let expected = generation.prompt.len() + tokens.len();
            assert_eq!(generation.cache().tokens().len(), expected);
            assert!(generation
                .cache()
                .info()
                .all(|info| info.processed_tokens == expected));
            assert_eq!(
                &generation.cache().tokens()[..generation.prompt.len()],
                &generation.prompt
            );
            for (offset, previous) in tokens.iter().enumerate() {
                let GenerationEvent::Token { token_id, .. } = previous else {
                    unreachable!()
                };
                assert_eq!(
                    generation.cache().tokens()[generation.prompt.len() + offset],
                    *token_id
                );
            }
            assert_eq!(generation.history.last(), Some(token_id));
            terminal = finish_reason.is_some();
            tokens.push(event);
        }
    }
    assert!(terminal);
    assert!(generation.next().is_none());
    Ok(tokens)
}
fn token_ids(events: &[GenerationEvent]) -> Vec<TokenId> {
    events
        .iter()
        .filter_map(|event| match event {
            GenerationEvent::Token { token_id, .. } => Some(*token_id),
            _ => None,
        })
        .collect()
}

#[test]
fn fixture_default_length_and_stop_streams() -> TestResult {
    with_device(Device::cpu(), || {
        for name in FIXTURES {
            let expected = read_json(fixture(name).join("expectations.json"))?;
            let input = read_json(fixture(name).join("inputs.json"))?;
            let prompt = ids(&expected["prefill"]["token_ids"]);
            let mut model = Model::from_dir(fixture(name))?;
            for case in ["default", "length_case", "stop_case"] {
                let golden = if case == "default" {
                    &expected["decode"]
                } else {
                    &expected["decode"][case]
                };
                let mut options = GenerationOptions {
                    max_tokens: nz(input["decode_steps"].as_u64().unwrap() as usize),
                    ..Default::default()
                };
                if case != "default" {
                    options.stop.tokens = StopTokenPolicy::Exact(ids(&golden["eos_tokens"]));
                }
                let mut generation = model.generate(Prompt::Tokens(&prompt), options)?;
                let events = collect_tokens(&mut generation)?;
                assert_eq!(
                    token_ids(&events),
                    ids(&golden["token_ids"]),
                    "{name}/{case}"
                );
                let deltas: Vec<_> = events
                    .iter()
                    .map(|event| match event {
                        GenerationEvent::Token { text, .. } => text.as_str(),
                        _ => unreachable!(),
                    })
                    .collect();
                assert_eq!(json!(deltas), golden["text_deltas"], "{name}/{case}");
                let GenerationEvent::Token {
                    token_id,
                    finish_reason,
                    ..
                } = events.last().unwrap()
                else {
                    unreachable!()
                };
                let reason = if golden["finish_reason"] == "stop" {
                    FinishReason::Stop
                } else {
                    FinishReason::Length
                };
                assert_eq!(*finish_reason, Some(reason), "{name}/{case}");
                let stop_token = (reason == FinishReason::Stop).then_some(u32::from(*token_id));
                assert_eq!(json!(stop_token), golden["stop_token"], "{name}/{case}");
            }
        }
        Ok(())
    })
}
fn sampling_options(case: &Value) -> GenerationOptions {
    let raw = &case["options"];
    let mut options = length_options(case["cpu_ids"].as_array().unwrap().len());
    options.sampling.temperature = raw["temperature"].as_f64().unwrap() as f32;
    options.sampling.seed = case["seed"].as_u64();
    options.sampling.top_p = raw["top_p"].as_f64().filter(|p| *p > 0.0).map(|p| p as f32);
    options.sampling.top_k = NonZeroUsize::new(raw["top_k"].as_u64().unwrap() as usize);
    options.sampling.min_p =
        raw["min_p"]
            .as_f64()
            .filter(|p| *p > 0.0)
            .map(|p| crate::MinPOptions {
                probability: p as f32,
                min_tokens_to_keep: nz(raw["min_tokens_to_keep"].as_u64().unwrap() as usize),
            });
    options.repetition_penalty =
        raw["repetition_penalty"]
            .as_f64()
            .map(|penalty| RepetitionPenaltyOptions {
                penalty: penalty as f32,
                context_size: nz(raw["repetition_context_size"].as_u64().unwrap() as usize),
            });
    for (field, target) in [
        ("presence", &mut options.presence_penalty),
        ("frequency", &mut options.frequency_penalty),
    ] {
        *target = raw[format!("{field}_penalty")]
            .as_f64()
            .map(|penalty| AdditivePenaltyOptions {
                penalty: penalty as f32,
                context_size: nz(raw[format!("{field}_context_size")].as_u64().unwrap() as usize),
            });
    }
    options
}
#[test]
fn fixture_45_seeded_sampling_trajectories() -> TestResult {
    with_device(Device::cpu(), || {
        let mut count = 0;
        for name in FIXTURES {
            let expected = read_json(fixture(name).join("expectations.json"))?;
            let prompt = ids(&expected["prefill"]["token_ids"]);
            let mut model = Model::from_dir(fixture(name))?;
            for (case_name, case) in expected["sampling"].as_object().unwrap() {
                let mut generation =
                    model.generate(Prompt::Tokens(&prompt), sampling_options(case))?;
                let mut observed = vec![];
                loop {
                    let history = generation.history.clone();
                    let Some(event) = generation.next() else {
                        break;
                    };
                    if let GenerationEvent::Token { token_id, .. } = event? {
                        if let Some(histories) = case["processor_token_histories"].as_array() {
                            assert_eq!(
                                history,
                                ids(&histories[observed.len()]),
                                "{name}/{case_name}"
                            );
                        }
                        observed.push(token_id);
                    }
                }
                assert_eq!(observed, ids(&case["cpu_ids"]), "{name}/{case_name}");
                count += 1;
            }
        }
        assert_eq!(count, 45);
        Ok(())
    })
}
#[test]
fn fixture_prefill_progress_and_drop_boundaries() -> TestResult {
    with_device(Device::cpu(), || {
        let expected = read_json(fixture("llama-base").join("expectations.json"))?;
        let mut model = Model::from_dir(fixture("llama-base"))?;
        for (_, case) in expected["prefill"]["progress"].as_object().unwrap() {
            let prompt = ids(&case["token_ids"]);
            let mut options = length_options(1);
            options.prefill_chunk_size = nz(case["ceiling"].as_u64().unwrap() as usize);
            for stop_at in 0..=case["pairs"].as_array().unwrap().len() {
                let mut cache = model.new_cache(Default::default())?;
                let mut generation = model.generate_with_cache(
                    Prompt::Tokens(&prompt),
                    options.clone(),
                    &mut cache,
                )?;
                let mut represented = 0;
                for pair in case["pairs"].as_array().unwrap().iter().take(stop_at) {
                    represented = pair[0].as_u64().unwrap() as usize;
                    assert_eq!(
                        generation.next().transpose()?,
                        Some(GenerationEvent::Prefill {
                            processed: represented,
                            total: prompt.len()
                        })
                    );
                    assert_eq!(generation.cache().tokens(), &prompt[..represented]);
                    assert!(generation.rng.is_none());
                }
                drop(generation);
                assert_eq!(cache.tokens(), &prompt[..represented]);
            }
        }
        Ok(())
    })
}

struct ScriptedDecoder {
    config: Config,
    layout: Vec<LayerCacheSpec>,
    output: Vec<u32>,
    calls: usize,
    fail_at: Option<usize>,
    empty_at: Option<usize>,
}
impl DecoderModel for ScriptedDecoder {
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
        let call = self.calls;
        self.calls += 1;
        for layer in 0..self.layout.len() {
            let values = tokens.as_dtype(Dtype::Float32)?.reshape(&[1, 1, -1, 1])?;
            cache.update_and_fetch(layer, values.clone(), values)?;
            if self.fail_at == Some(call) {
                return Err(InferenceError::Exception(mlx_rs::error::Exception::from(
                    "injected forward failure",
                )));
            }
        }
        if self.output[call.min(self.output.len() - 1)] == u32::MAX {
            return Ok(mlx_rs::ops::broadcast_to(
                failing_row(self.config.dimensions.vocabulary_size)?,
                &[
                    1,
                    tokens.dim(1),
                    self.config.dimensions.vocabulary_size as i32,
                ],
            )?);
        }
        let mut row = vec![f32::NEG_INFINITY; self.config.dimensions.vocabulary_size];
        if self.empty_at != Some(call) {
            row[self.output[call.min(self.output.len() - 1)] as usize] = 0.0;
        }
        let row = Array::from_slice(&row, &[1, 1, row.len() as i32]);
        Ok(mlx_rs::ops::broadcast_to(
            row,
            &[
                1,
                tokens.dim(1),
                self.config.dimensions.vocabulary_size as i32,
            ],
        )?)
    }
    fn weight_projection(
        &mut self,
    ) -> Result<mlx_rs::utils::StateProjection<'_>, crate::WeightError> {
        unreachable!("scripted decoder does not load weights")
    }
}
fn scripted(
    output: Vec<u32>,
    fail_at: Option<usize>,
    empty_at: Option<usize>,
    tokenizer: Option<Tokenizer>,
) -> anyhow::Result<Model> {
    let config = crate::config::RawConfig::from_bytes(&std::fs::read(
        fixture("llama-base").join("config.json"),
    )?)?
    .resolve()?;
    let layout = (0..2)
        .map(|_| LayerCacheSpec {
            attention: crate::AttentionKind::Full,
            batch_size: 1,
            kv_heads: 1,
            head_dim: 1,
            dtype: Dtype::Float32,
        })
        .collect();
    Ok(Model {
        decoder: Box::new(ScriptedDecoder {
            config: config.clone(),
            layout,
            output,
            calls: 0,
            fail_at,
            empty_at,
        }),
        config,
        tokenizer: match tokenizer {
            Some(tokenizer) => tokenizer,
            None => Tokenizer::from_dir(fixture("llama-base"))?,
        },
        identity: Rc::new(()),
    })
}
fn values(array: &Array) -> Vec<f32> {
    if array.size() == 0 {
        Vec::new()
    } else {
        array.as_slice::<f32>().to_vec()
    }
}

fn cache_evidence(cache: &Cache) -> anyhow::Result<Value> {
    let mut layers = vec![];
    for view in cache.logical_layers()? {
        layers.push(json!([
            view.layer,
            view.positions,
            values(&view.keys),
            values(&view.values)
        ]));
    }
    let info: Vec<_> = cache
        .info()
        .map(|i| {
            json!([
                i.processed_tokens,
                i.capacity,
                i.retained_prefix,
                i.retained_positions
            ])
        })
        .collect();
    Ok(
        json!({"tokens": cache.tokens().iter().copied().map(u32::from).collect::<Vec<_>>(), "layers": layers, "info": info}),
    )
}
#[test]
fn forward_failure_after_prefill_chunk_restores_boundary() -> TestResult {
    with_device(Device::cpu(), || {
        let mut model = scripted(vec![12], Some(1), None, None)?;
        let prompt = vec![TokenId::from(12); 8];
        let mut options = length_options(4);
        options.prefill_chunk_size = nz(3);
        let mut cache = model.new_cache(Default::default())?;
        let mut generation =
            model.generate_with_cache(Prompt::Tokens(&prompt), options, &mut cache)?;
        assert!(matches!(
            generation.next().transpose()?,
            Some(GenerationEvent::Prefill { processed: 0, .. })
        ));
        assert!(matches!(
            generation.next().transpose()?,
            Some(GenerationEvent::Prefill { processed: 3, .. })
        ));
        let before = cache_evidence(generation.cache())?;
        assert!(
            matches!(generation.next(), Some(Err(GenerationError::Inference(InferenceError::Exception(source)))) if source.what() == "injected forward failure")
        );
        assert_eq!(cache_evidence(generation.cache())?, before);
        assert_eq!(generation.accepted, 0);
        assert!(generation.next().is_none());
        assert!(generation.next().is_none());
        drop(generation);
        assert_eq!(cache_evidence(&cache)?, before);
        Ok(())
    })
}
#[test]
fn empty_support_at_first_sample_and_decode_preserves_boundary() -> TestResult {
    with_device(Device::cpu(), || {
        for fail_at in [0, 1] {
            let mut model = scripted(vec![12], None, Some(fail_at), None)?;
            let mut options = length_options(3);
            options.sampling.temperature = 0.7;
            options.sampling.seed = Some(1729);
            let mut generation = model.generate(Prompt::Tokens(&[TokenId::from(12)]), options)?;
            generation.next().transpose()?;
            generation.next().transpose()?;
            if fail_at == 1 {
                generation.next().transpose()?;
            }
            let before = cache_evidence(generation.cache())?;
            let rng = generation
                .rng
                .as_ref()
                .map(|r| r.as_array().as_slice::<u32>().to_vec());
            let history = generation.history.clone();
            assert!(matches!(
                generation.next(),
                Some(Err(GenerationError::Sampling(SamplingError::EmptySupport)))
            ));
            assert_eq!(cache_evidence(generation.cache())?, before);
            assert_eq!(generation.history, history);
            assert_eq!(generation.accepted, fail_at);
            assert_eq!(
                generation
                    .rng
                    .as_ref()
                    .map(|r| r.as_array().as_slice::<u32>().to_vec()),
                rng
            );
            assert!(generation.next().is_none());
            assert!(generation.next().is_none());
        }
        Ok(())
    })
}
#[test]
fn tokenizer_failure_after_evaluation_rolls_back_decode() -> TestResult {
    with_device(Device::cpu(), || {
        let cases = read_json(fixture("llama-base").join("text_cases.json"))?;
        let case = &cases["decoders"]["byte_fallback_rewrite"];
        let output: Vec<_> = ids(&case["token_ids"]).into_iter().map(u32::from).collect();
        let tokenizer = Tokenizer::from_bytes(&serde_json::to_vec(&case["tokenizer"])?)?;
        let mut model = scripted(output.clone(), None, None, Some(tokenizer))?;
        let mut generation = model.generate(
            Prompt::Tokens(&[TokenId::from(0)]),
            length_options(output.len()),
        )?;
        generation.next().transpose()?;
        generation.next().transpose()?;
        for _ in 0..output.len() - 1 {
            generation.next().transpose()?;
        }
        let before = cache_evidence(generation.cache())?;
        let Some(Err(GenerationError::Tokenizer(TokenizerError::Tokenizer(source)))) =
            generation.next()
        else {
            panic!("expected tokenizer failure")
        };
        let Some(tokenizers::tokenizer::DecodeStreamError::InvalidPrefix {
            token_id,
            expected_prefix,
            actual_string,
        }) = source.downcast_ref::<tokenizers::tokenizer::DecodeStreamError>()
        else {
            panic!("lost InvalidPrefix")
        };
        assert_eq!(Some(token_id), output.last());
        assert_eq!(expected_prefix, case["emitted_prefix"].as_str().unwrap());
        let hex: String = actual_string
            .as_bytes()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(hex, case["decoded_utf8_hex"]);
        assert_eq!(cache_evidence(generation.cache())?, before);
        assert_eq!(generation.accepted, output.len() - 1);
        assert!(generation.next().is_none());
        assert!(generation.next().is_none());
        Ok(())
    })
}
#[test]
fn stop_precedence_split_matches_and_final_flush() -> TestResult {
    with_device(Device::cpu(), || {
        for policy in [
            StopTokenPolicy::Tokenizer,
            StopTokenPolicy::TokenizerPlus(vec![TokenId::from(12)]),
            StopTokenPolicy::Exact(vec![TokenId::from(12)]),
        ] {
            let token = if matches!(policy, StopTokenPolicy::Tokenizer) {
                3
            } else {
                12
            };
            let mut model = scripted(vec![token], None, None, None)?;
            let mut options = length_options(1);
            options.stop.tokens = policy;
            let events = collect_tokens(
                &mut model.generate(Prompt::Tokens(&[TokenId::from(12)]), options)?,
            )?;
            assert_eq!(
                events,
                vec![GenerationEvent::Token {
                    token_id: TokenId::from(token),
                    text: String::new(),
                    finish_reason: Some(FinishReason::Stop)
                }]
            );
        }
        for (stop, expected) in [
            ("hello world", ""),
            ("hello!", "hello world"),
            ("world!", "hello world"),
        ] {
            let mut model = scripted(vec![12, 13], None, None, None)?;
            let mut options = length_options(2);
            options.stop.stop_strings = vec![stop.into()];
            let events = collect_tokens(
                &mut model.generate(Prompt::Tokens(&[TokenId::from(12)]), options)?,
            )?;
            let text: String = events
                .iter()
                .map(|event| match event {
                    GenerationEvent::Token { text, .. } => text.as_str(),
                    _ => unreachable!(),
                })
                .collect();
            assert_eq!(text, expected);
            assert!(
                matches!(events.last(), Some(GenerationEvent::Token { finish_reason: Some(reason), .. }) if *reason == if stop == "hello world" { FinishReason::Stop } else { FinishReason::Length })
            );
        }
        let cases = read_json(fixture("llama-base").join("text_cases.json"))?;
        for (name, stop, expected, reason) in [
            ("byte_fallback_incomplete", "�", "", FinishReason::Stop),
            ("byte_level_incomplete_tail", "é�", "", FinishReason::Stop),
            (
                "byte_level_incomplete_tail",
                "é!",
                "é�",
                FinishReason::Length,
            ),
        ] {
            let case = &cases["decoders"][name];
            let output: Vec<_> = ids(&case["token_ids"]).into_iter().map(u32::from).collect();
            let tokenizer = Tokenizer::from_bytes(&serde_json::to_vec(&case["tokenizer"])?)?;
            let mut model = scripted(output.clone(), None, None, Some(tokenizer))?;
            let mut options = length_options(output.len());
            options.stop.stop_strings = vec![stop.into()];
            let events =
                collect_tokens(&mut model.generate(Prompt::Tokens(&[TokenId::from(0)]), options)?)?;
            let text: String = events
                .iter()
                .map(|event| match event {
                    GenerationEvent::Token { text, .. } => text.as_str(),
                    _ => unreachable!(),
                })
                .collect();
            assert_eq!(text, expected);
            assert!(
                matches!(events.last(), Some(GenerationEvent::Token { finish_reason: Some(actual), .. }) if *actual == reason)
            );
        }
        Ok(())
    })
}

#[test]
fn reuse_strict_extension_matches_fresh_generation() -> TestResult {
    with_device(Device::cpu(), || {
        for name in FIXTURES {
            let expected = read_json(fixture(name).join("expectations.json"))?;
            let mut prompt = ids(&expected["prefill"]["token_ids"]);
            for policy in [crate::CachePolicy::ModelDefault, crate::CachePolicy::Full] {
                let mut model = Model::from_dir(fixture(name))?;
                let mut options = length_options(4);
                options.cache.policy = policy;
                options.prefill_chunk_size = nz(1);
                options.presence_penalty = Some(AdditivePenaltyOptions {
                    penalty: 0.6,
                    context_size: nz(3),
                });
                let mut cache = model.new_cache(options.cache.clone())?;
                let mut generation = model.generate_with_cache(
                    Prompt::Tokens(&prompt),
                    options.clone(),
                    &mut cache,
                )?;
                let events = collect_tokens(&mut generation)?;
                let snapshot = generation.snapshot()?;
                drop(generation);
                let old_len = cache.tokens().len();
                prompt.extend(token_ids(&events));
                prompt.extend([TokenId::from(12), TokenId::from(13)]);
                let fresh =
                    collect_tokens(&mut model.generate(Prompt::Tokens(&prompt), options.clone())?)?;
                let mut reused = model.generate_with_cache(
                    Prompt::Tokens(&prompt),
                    options.clone(),
                    &mut cache,
                )?;
                assert_eq!(reused.history, vec![*prompt.last().unwrap()]);
                assert_eq!(
                    reused.next().transpose()?,
                    Some(GenerationEvent::Prefill {
                        processed: 0,
                        total: prompt.len() - old_len
                    })
                );
                assert_eq!(collect_tokens(&mut reused)?, fresh, "{name}");
                drop(reused);
                cache.restore(snapshot)?;
                assert_eq!(cache.tokens(), &prompt[..old_len]);
            }
        }
        Ok(())
    })
}
#[test]
fn interleaved_seeded_generators_equal_isolated_runs() -> TestResult {
    with_device(Device::cpu(), || {
        let expected = read_json(fixture("llama-base").join("expectations.json"))?;
        let prompt = ids(&expected["prefill"]["token_ids"]);
        let mut options = sampling_options(&expected["sampling"]["temperature"]);
        let mut left = Model::from_dir(fixture("llama-base"))?;
        let mut right = Model::from_dir(fixture("qwen3-base"))?;
        let isolated_left = left
            .generate(Prompt::Tokens(&prompt), options.clone())?
            .collect::<Result<Vec<_>, _>>()?;
        options.sampling.seed = Some(42);
        let isolated_right = right
            .generate(Prompt::Tokens(&prompt), options.clone())?
            .collect::<Result<Vec<_>, _>>()?;
        let mut right = right.generate(Prompt::Tokens(&prompt), options.clone())?;
        options.sampling.seed = Some(1729);
        let mut left = left.generate(Prompt::Tokens(&prompt), options)?;
        let (mut observed_left, mut observed_right) = (vec![], vec![]);
        loop {
            let l = left.next().transpose()?;
            let r = right.next().transpose()?;
            if l.is_none() && r.is_none() {
                break;
            }
            if let Some(event) = l {
                observed_left.push(event);
            }
            if let Some(event) = r {
                observed_right.push(event);
            }
        }
        assert_eq!(observed_left, isolated_left);
        assert_eq!(observed_right, isolated_right);
        Ok(())
    })
}
#[test]
fn validation_precedence_before_allocation() -> TestResult {
    let mut model = scripted(vec![12], None, None, None)?;
    let mut options = length_options(usize::MAX);
    options.sampling.temperature = -1.0;
    options.repetition_penalty = Some(RepetitionPenaltyOptions {
        penalty: -1.0,
        context_size: nz(1),
    });
    options.presence_penalty = Some(AdditivePenaltyOptions {
        penalty: f32::NAN,
        context_size: nz(1),
    });
    options.frequency_penalty = options.presence_penalty.clone();
    options.stop.stop_strings = vec!["ok".into(), "".into(), "".into()];
    options.stop.tokens = StopTokenPolicy::Exact(vec![TokenId::from(100), TokenId::from(64)]);
    let prompt = [TokenId::from(100), TokenId::from(64)];
    assert!(matches!(
        model.generate(Prompt::Tokens(&prompt), options.clone()),
        Err(GenerationError::Sampling(
            SamplingError::InvalidTemperature(_)
        ))
    ));
    options.sampling.temperature = 0.0;
    assert!(matches!(
        model.generate(Prompt::Tokens(&prompt), options.clone()),
        Err(GenerationError::Sampling(
            SamplingError::InvalidRepetitionPenalty(_)
        ))
    ));
    options.repetition_penalty = None;
    assert!(matches!(
        model.generate(Prompt::Tokens(&prompt), options.clone()),
        Err(GenerationError::Sampling(
            SamplingError::InvalidAdditivePenalty(_)
        ))
    ));
    options.presence_penalty = None;
    assert!(matches!(
        model.generate(Prompt::Tokens(&prompt), options.clone()),
        Err(GenerationError::Sampling(
            SamplingError::InvalidAdditivePenalty(_)
        ))
    ));
    options.frequency_penalty = None;
    assert!(matches!(
        model.generate(Prompt::Tokens(&prompt), options.clone()),
        Err(GenerationError::EmptyStopString { index: 1 })
    ));
    options.stop.stop_strings.clear();
    assert!(
        matches!(model.generate(Prompt::Tokens(&prompt), options.clone()), Err(GenerationError::StopTokenOutOfRange { token_id, vocabulary_size: 64 }) if token_id == TokenId::from(64))
    );
    options.stop.tokens = StopTokenPolicy::Exact(vec![]);
    assert!(matches!(
        model.generate(Prompt::Tokens(&[]), options.clone()),
        Err(GenerationError::EmptyPrompt)
    ));
    assert!(
        matches!(model.generate(Prompt::Tokens(&prompt), options.clone()), Err(GenerationError::PromptTokenOutOfRange { index: 0, token_id, vocabulary_size: 64 }) if token_id == TokenId::from(100))
    );
    assert!(matches!(
        model.generate(Prompt::Tokens(&[TokenId::from(12)]), options.clone()),
        Err(GenerationError::LengthOverflow)
    ));
    options.max_tokens = nz(i32::MAX as usize);
    assert!(
        matches!(model.generate(Prompt::Tokens(&[TokenId::from(12)]), options), Err(GenerationError::SequenceTooLong { length, limit }) if length == i32::MAX as usize + 1 && limit == i32::MAX as usize)
    );
    Ok(())
}
#[test]
fn borrowed_validation_order_preserves_cache() -> TestResult {
    with_device(Device::cpu(), || {
        let mut model = scripted(vec![12], None, None, None)?;
        let foreign = scripted(vec![12], None, None, None)?;
        let prompt = [TokenId::from(12), TokenId::from(13)];
        let mut cache = foreign.new_cache(Default::default())?;
        let before = cache_evidence(&cache)?;
        let mut options = length_options(usize::MAX);
        options.cache.policy = crate::CachePolicy::Rotating {
            capacity: nz(5),
            keep_prefix: 2,
        };
        assert!(matches!(
            model.generate_with_cache(Prompt::Tokens(&prompt), options.clone(), &mut cache),
            Err(GenerationError::Cache(CacheError::FingerprintMismatch))
        ));
        assert_eq!(cache_evidence(&cache)?, before);
        let mut cache = model.new_cache(Default::default())?;
        assert!(matches!(
            model.generate_with_cache(Prompt::Tokens(&prompt), options.clone(), &mut cache),
            Err(GenerationError::Cache(CacheError::PolicyMismatch))
        ));
        collect_tokens(&mut model.generate_with_cache(
            Prompt::Tokens(&prompt),
            length_options(1),
            &mut cache,
        )?)?;
        let before = cache_evidence(&cache)?;
        options.cache = Default::default();
        assert!(matches!(
            model.generate_with_cache(Prompt::Tokens(&prompt[..1]), options.clone(), &mut cache),
            Err(GenerationError::CachePrefixMismatch {
                matched: 1,
                cached: 2,
                prompt: 1
            })
        ));
        assert!(matches!(
            model.generate_with_cache(
                Prompt::Tokens(&[TokenId::from(13)]),
                options.clone(),
                &mut cache
            ),
            Err(GenerationError::CachePrefixMismatch { matched: 0, .. })
        ));
        assert!(matches!(
            model.generate_with_cache(Prompt::Tokens(&prompt), options.clone(), &mut cache),
            Err(GenerationError::NoUncachedTokens)
        ));
        assert_eq!(cache_evidence(&cache)?, before);
        Ok(())
    })
}
#[test]
fn text_prompt_bos_rule_uses_original_string() -> TestResult {
    with_device(Device::cpu(), || {
        let expected = read_json(fixture("llama-base").join("expectations.json"))?;
        let cases = &expected["tokenizer"]["prompt_encoding"];
        let input = read_json(fixture("llama-base").join("inputs.json"))?;
        let canonical = input["prompts"]["canonical"].as_str().unwrap();
        for (name, text) in [
            ("canonical", canonical.to_owned()),
            (
                "canonical_with_bos",
                format!("<|begin_of_text|>{canonical}"),
            ),
            (
                "whitespace_before_bos",
                format!(" <|begin_of_text|>{canonical}"),
            ),
        ] {
            let mut model = scripted(vec![12], None, None, None)?;
            let generation = model.generate(Prompt::Text(&text), length_options(1))?;
            assert_eq!(generation.prompt, ids(&cases[name]));
            assert!(generation.cache().tokens().is_empty());
        }
        Ok(())
    })
}
#[test]
fn evaluation_error_mapping_preserves_source() {
    let error = evaluation_error(CacheError::Exception(mlx_rs::error::Exception::from(
        "injected evaluation failure",
    )));
    assert!(
        matches!(error, GenerationError::Exception(source) if source.what() == "injected evaluation failure")
    );
    assert!(matches!(
        evaluation_error(CacheError::FingerprintMismatch),
        GenerationError::Cache(CacheError::FingerprintMismatch)
    ));
}

fn failing_row(vocabulary: usize) -> Result<Array, mlx_rs::error::Exception> {
    let singular = Array::from_slice(&[0.0f32], &[1, 1]);
    mlx_rs::ops::broadcast_to(mlx_rs::linalg::inv(singular)?, &[1, vocabulary as i32])
}
#[test]
fn runtime_evaluation_failures_at_prefill_first_sample_and_decode() -> TestResult {
    with_device(Device::cpu(), || {
        let original = failing_row(64)?
            .eval()
            .expect_err("singular inverse must fail during evaluation")
            .what()
            .to_owned();
        let original = original
            .split(" at /")
            .next()
            .expect("exception text is nonempty")
            .to_owned();
        for boundary in ["prefill", "first_sample", "decode"] {
            let output = if boundary == "prefill" {
                vec![12, u32::MAX]
            } else if boundary == "decode" {
                vec![12, 12, u32::MAX]
            } else {
                vec![12]
            };
            let mut model = scripted(output, None, None, None)?;
            let mut options = length_options(3);
            options.prefill_chunk_size = nz(1);
            options.sampling.temperature = 0.7;
            options.sampling.seed = Some(1729);
            let mut generation = model.generate(
                Prompt::Tokens(&[TokenId::from(12), TokenId::from(13)]),
                options,
            )?;
            generation.next().transpose()?;
            generation.next().transpose()?;
            if boundary != "prefill" {
                generation.next().transpose()?;
            }
            if boundary == "first_sample" {
                generation.state = GenerationState::FirstSample(failing_row(64)?);
            }
            if boundary == "decode" {
                generation.next().transpose()?;
            }
            let before = cache_evidence(generation.cache())?;
            let accepted = generation.accepted;
            let history = generation.history.clone();
            let rng = generation
                .rng
                .as_ref()
                .map(|r| r.as_array().as_slice::<u32>().to_vec());
            assert!(
                matches!(generation.next(), Some(Err(GenerationError::Exception(source))) if source.what().starts_with(&original)),
                "{boundary}"
            );
            assert_eq!(cache_evidence(generation.cache())?, before, "{boundary}");
            assert_eq!(generation.accepted, accepted);
            assert_eq!(generation.history, history);
            assert_eq!(
                generation
                    .rng
                    .as_ref()
                    .map(|r| r.as_array().as_slice::<u32>().to_vec()),
                rng
            );
            assert!(generation.next().is_none());
            assert!(generation.next().is_none());
        }
        Ok(())
    })
}

#[test]
fn encoding_error_precedes_length_validation() -> TestResult {
    let mut raw = read_json(fixture("llama-base").join("tokenizer.json"))?;
    raw["model"]["unk_token"] = json!("missing unknown token");
    let tokenizer = Tokenizer::from_bytes(&serde_json::to_vec(&raw)?)?;
    let mut model = scripted(vec![12], None, None, Some(tokenizer))?;
    let source = model
        .tokenizer
        .encode_with_special_tokens("not-in-vocabulary", true)
        .unwrap_err()
        .to_string();
    let error = model
        .generate(
            Prompt::Text("not-in-vocabulary"),
            length_options(usize::MAX),
        )
        .err()
        .unwrap();
    assert!(
        matches!(error, GenerationError::Tokenizer(TokenizerError::Tokenizer(error)) if error.to_string() == source)
    );
    Ok(())
}

#[test]
fn text_preparation_terminal_flush_and_stop_exclusion_without_mlx() -> TestResult {
    let cases = read_json(fixture("llama-base").join("text_cases.json"))?;
    for (name, stop, expected, reason) in [
        ("byte_fallback_incomplete", "�", "", FinishReason::Stop),
        ("byte_level_incomplete_tail", "é�", "", FinishReason::Stop),
        (
            "byte_level_incomplete_tail",
            "é!",
            "é�",
            FinishReason::Length,
        ),
    ] {
        let case = &cases["decoders"][name];
        let tokenizer = Tokenizer::from_bytes(&serde_json::to_vec(&case["tokenizer"])?)?;
        let mut state = Some(TextState {
            decoder: tokenizer.decode_stream(),
            filter: StopStringFilter::new(vec![stop.into()])?,
        });
        let tokens = ids(&case["token_ids"]);
        let mut observed = String::new();
        for (index, token) in tokens.iter().enumerate() {
            let last = index + 1 == tokens.len();
            let (next, text, finish) = state.take().unwrap().prepare(*token, false, last)?;
            observed.push_str(&text);
            assert_eq!(finish, last.then_some(reason));
            state = next;
        }
        assert!(state.is_none());
        assert_eq!(observed, expected);
    }
    let tokenizer = Tokenizer::from_dir(fixture("llama-base"))?;
    let state = TextState {
        decoder: tokenizer.decode_stream(),
        filter: StopStringFilter::new(vec![])?,
    };
    let (next, text, finish) = state.prepare(TokenId::from(12), true, true)?;
    assert!(next.is_none());
    assert_eq!(text, "");
    assert_eq!(finish, Some(FinishReason::Stop));
    Ok(())
}

#[cfg(feature = "oracle-hooks")]
#[test]
fn oracle_logprobs_capture_uses_the_same_generation_stream() -> TestResult {
    with_device(Device::cpu(), || {
        let expected = read_json(fixture("llama-base").join("expectations.json"))?;
        let prompt = ids(&expected["prefill"]["token_ids"]);
        let options = sampling_options(&expected["sampling"]["combined"]);
        let mut model = Model::from_dir(fixture("llama-base"))?;
        let normal = model
            .generate(Prompt::Tokens(&prompt), options.clone())?
            .collect::<Result<Vec<_>, _>>()?;
        let mut generation = crate::oracle_hooks::generate_with_logprobs(
            &mut model,
            Prompt::Tokens(&prompt),
            options,
        )?;
        let mut events = vec![];
        assert!(crate::oracle_hooks::first_filtered_logprobs(&generation).is_none());
        while let Some(event) = generation.next() {
            let event = event?;
            if matches!(event, GenerationEvent::Token { .. }) {
                events.push(event);
                break;
            }
            assert!(crate::oracle_hooks::first_filtered_logprobs(&generation).is_none());
            events.push(event);
        }
        let first = crate::oracle_hooks::first_filtered_logprobs(&generation)
            .unwrap()
            .to_vec_exact::<f32>()?;
        let tensors =
            Array::load_safetensors(fixture("llama-base").join("expectations.safetensors"))?;
        let golden = tensors["sampling.combined.filtered_logprobs"].to_vec_exact::<f32>()?;
        let atol = expected["tolerances"]["logprobs"]["atol"].as_f64().unwrap() as f32;
        let rtol = expected["tolerances"]["logprobs"]["rtol"].as_f64().unwrap() as f32;
        assert_eq!(first.len(), golden.len());
        for (actual, expected) in first.iter().zip(golden) {
            if expected == f32::NEG_INFINITY {
                assert_eq!(*actual, expected);
            } else {
                assert!((*actual - expected).abs() <= atol + rtol * expected.abs());
            }
        }
        assert_eq!(first.len(), 64);
        assert_eq!(first.iter().filter(|value| value.is_finite()).count(), 5);
        assert!(!first
            .iter()
            .any(|value| value.is_nan() || *value == f32::INFINITY));
        events.extend(generation.by_ref().collect::<Result<Vec<_>, _>>()?);
        assert_eq!(events, normal);
        assert_eq!(
            crate::oracle_hooks::first_filtered_logprobs(&generation)
                .unwrap()
                .to_vec_exact::<f32>()?,
            first
        );
        Ok(())
    })
}
