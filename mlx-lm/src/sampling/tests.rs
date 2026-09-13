use super::*;

#[test]
fn additive_penalty_range() -> Result<(), SamplingError> {
    for penalty in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            AdditivePenaltyOptions {
                penalty,
                context_size: NonZeroUsize::MIN,
            }
            .validate(),
            Err(SamplingError::InvalidAdditivePenalty(_))
        ));
    }
    for penalty in [f32::MIN, -3.0, -0.0, 0.0, f32::MIN_POSITIVE, 3.0, f32::MAX] {
        AdditivePenaltyOptions {
            penalty,
            context_size: NonZeroUsize::MIN,
        }
        .validate()?;
    }
    Ok(())
}

#[test]
fn seed_accepts_the_full_u64_range_even_for_greedy() -> Result<(), SamplingError> {
    for seed in [None, Some(0), Some(u64::MAX)] {
        SamplerOptions {
            seed,
            ..Default::default()
        }
        .validate(64)?;
    }
    Ok(())
}

#[test]
fn sampling_ranges_apply_even_to_greedy() -> Result<(), SamplingError> {
    SamplerOptions::default().validate(64)?;
    assert!(matches!(
        SamplerOptions::default().validate(0),
        Err(SamplingError::EmptyVocabulary)
    ));
    for temperature in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1] {
        let options = SamplerOptions {
            temperature,
            ..Default::default()
        };
        assert!(matches!(
            options.validate(64),
            Err(SamplingError::InvalidTemperature(_))
        ));
    }
    for top_p in [f32::NAN, f32::INFINITY, -0.1, 0.0, 1.1] {
        let options = SamplerOptions {
            top_p: Some(top_p),
            ..Default::default()
        };
        assert!(
            matches!(options.validate(64), Err(SamplingError::InvalidProbability { field, .. }) if field == "top_p")
        );
    }
    for top_p in [f32::MIN_POSITIVE, 0.5, 1.0] {
        SamplerOptions {
            top_p: Some(top_p),
            ..Default::default()
        }
        .validate(64)?;
    }
    let options = SamplerOptions {
        top_k: NonZeroUsize::new(64),
        ..Default::default()
    };
    options.validate(64)?;
    assert!(matches!(
        options.validate(63),
        Err(SamplingError::TopKExceedsVocabulary {
            top_k: 64,
            vocabulary_size: 63
        })
    ));
    let min_p = MinPOptions {
        probability: f32::NAN,
        min_tokens_to_keep: NonZeroUsize::MIN,
    };
    assert!(matches!(
        SamplerOptions {
            min_p: Some(min_p),
            ..Default::default()
        }
        .validate(64),
        Err(SamplingError::InvalidProbability { .. })
    ));
    Ok(())
}

#[test]
fn min_p_range_and_retained_support() -> Result<(), SamplingError> {
    let mut options = MinPOptions {
        probability: 0.0,
        min_tokens_to_keep: NonZeroUsize::MIN,
    };
    assert!(matches!(
        options.validate(0),
        Err(SamplingError::EmptyVocabulary)
    ));
    for probability in [0.0, 0.5, 1.0] {
        options.probability = probability;
        options.validate(1)?;
    }
    for probability in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1, 1.1] {
        options.probability = probability;
        assert!(
            matches!(options.validate(64), Err(SamplingError::InvalidProbability { field, .. }) if field == "min_p")
        );
    }
    options.probability = 0.1;
    options.min_tokens_to_keep = NonZeroUsize::new(64).ok_or(SamplingError::EmptyVocabulary)?;
    options.validate(64)?;
    assert!(matches!(
        options.validate(63),
        Err(SamplingError::MinTokensToKeepExceedsVocabulary {
            min_tokens_to_keep: 64,
            vocabulary_size: 63
        })
    ));
    Ok(())
}

use crate::{GenerationError, RepetitionPenaltyOptions, TokenId};
use mlx_rs::{with_device, Array, Device, Dtype};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn nz(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).unwrap()
}

fn key_values(row: &Array) -> Vec<u32> {
    row.eval().unwrap();
    row.as_slice::<u32>().to_vec()
}

fn values(row: &Array) -> Vec<f32> {
    let row = row.as_dtype(Dtype::Float32).unwrap();
    row.eval().unwrap();
    row.as_slice::<f32>().to_vec()
}

#[test]
fn processors_preserve_duplicates_signs_and_independent_windows() -> TestResult {
    with_device(Device::cpu(), || {
        let history: Vec<_> = [1, 2, 1, 2, 4, 1, 3].map(TokenId::from).into();
        let row = Array::from_slice(&[0.0f32, 2.0, -3.0, 1.0, -0.5], &[1, 5]);
        let repetition = RepetitionPenaltyOptions {
            penalty: 2.0,
            context_size: nz(3),
        };
        assert_eq!(
            values(&repetition.process(&history, row.clone())?),
            [0.0, 1.0, -3.0, 0.5, -1.0]
        );
        for coefficient in [-0.5, 0.0, 0.5] {
            let options = AdditivePenaltyOptions {
                penalty: coefficient,
                context_size: nz(3),
            };
            let prefix = &history[..3];
            assert_eq!(
                values(&PresencePenalty(options.clone()).process(prefix, row.clone())?),
                [0.0, 2.0 - coefficient, -3.0 - coefficient, 1.0, -0.5]
            );
            assert_eq!(
                values(&FrequencyPenalty(options).process(prefix, row.clone())?),
                [0.0, 2.0 - 2.0 * coefficient, -3.0 - coefficient, 1.0, -0.5]
            );
        }
        assert_eq!(values(&repetition.process(&[], row.clone())?), values(&row));
        Ok(())
    })
}

#[test]
fn greedy_bypasses_filters_but_validates_options() -> TestResult {
    with_device(Device::cpu(), || {
        let row = Array::from_slice(&[1.0f32, 3.0, 3.0], &[1, 3]);
        let mut options = SamplerOptions {
            top_k: Some(nz(1)),
            seed: Some(1729),
            ..Default::default()
        };
        let sampler = SamplingEngine::new(options.clone(), 3, None, None, None)?;
        let sample = sampler.sample(&[], row.clone(), None, true)?;
        assert_eq!(sample.token.try_item_exact::<u32>()?, 1);
        assert!(sample.rng.is_none());
        assert!(values(sample.filtered_logprobs.as_ref().unwrap())
            .iter()
            .all(|x| x.is_finite()));
        options.top_p = Some(0.0);
        assert!(matches!(
            SamplingEngine::new(options, 3, None, None, None),
            Err(SamplingError::InvalidProbability { .. })
        ));
        Ok(())
    })
}

#[test]
fn filter_equalities_rescue_and_identities() -> TestResult {
    with_device(Device::cpu(), || {
        let row = Array::from_slice(&[0.25f32.ln(), 0.25f32.ln(), 0.5f32.ln()], &[1, 3]);
        let filtered = top_p(&row, 0.5)?;
        assert_eq!(
            values(&filtered),
            [f32::NEG_INFINITY, f32::NEG_INFINITY, 0.5f32.ln()]
        );
        assert_eq!(values(&top_p(&row, 1.0)?), values(&row));
        assert_eq!(values(&top_k(&row, 3)?), values(&row));
        assert_eq!(
            values(&min_p(
                &row,
                &MinPOptions {
                    probability: 0.0,
                    min_tokens_to_keep: nz(1)
                }
            )?),
            values(&row)
        );
        let row = Array::from_slice(&[-2.0f32, -1.0, 0.0], &[1, 3]);
        let options = MinPOptions {
            probability: (-1.0f32).exp(),
            min_tokens_to_keep: nz(1),
        };
        assert_eq!(
            values(&min_p(&row, &options)?),
            [f32::NEG_INFINITY, -1.0, 0.0]
        );
        let options = MinPOptions {
            probability: 1.0,
            min_tokens_to_keep: nz(2),
        };
        assert_eq!(
            values(&min_p(&row, &options)?),
            [f32::NEG_INFINITY, -1.0, 0.0]
        );
        Ok(())
    })
}

#[test]
fn invalid_rows_and_overflows_do_not_advance_rng() -> TestResult {
    with_device(Device::cpu(), || {
        let options = SamplerOptions {
            temperature: 0.7,
            seed: Some(1729),
            ..Default::default()
        };
        let sampler = SamplingEngine::new(options, 2, None, None, None)?;
        let rng = RandomState::with_seed(1729)?;
        let before = rng.as_array().clone();
        for entries in [
            [f32::NAN, 0.0],
            [f32::INFINITY, 0.0],
            [f32::NAN, f32::NEG_INFINITY],
        ] {
            assert!(matches!(
                sampler.sample(&[], Array::from_slice(&entries, &[1, 2]), Some(&rng), false),
                Err(GenerationError::Sampling(SamplingError::InvalidLogitsValue))
            ));
        }
        assert!(matches!(
            sampler.sample(
                &[],
                Array::from_slice(&[f32::NEG_INFINITY; 2], &[1, 2]),
                Some(&rng),
                false
            ),
            Err(GenerationError::Sampling(SamplingError::EmptySupport))
        ));
        assert_eq!(key_values(rng.as_array()), key_values(&before));
        let sampler = SamplingEngine::new(
            SamplerOptions::default(),
            2,
            None,
            Some(AdditivePenaltyOptions {
                penalty: -f32::MAX,
                context_size: nz(1),
            }),
            None,
        )?;
        assert!(matches!(
            sampler.sample(
                &[TokenId::from(0)],
                Array::from_slice(&[f32::MAX, 0.0], &[1, 2]),
                None,
                false
            ),
            Err(GenerationError::Sampling(SamplingError::InvalidLogitsValue))
        ));
        let sampler = SamplingEngine::new(
            SamplerOptions {
                temperature: f32::from_bits(1),
                ..Default::default()
            },
            2,
            None,
            None,
            None,
        )?;
        assert!(matches!(
            sampler.sample(&[], Array::from_slice(&[0.0f32, 0.0], &[1, 2]), None, false),
            Err(GenerationError::Sampling(SamplingError::EmptySupport))
        ));
        Ok(())
    })
}

#[test]
fn candidate_rng_splits_once_and_retries_without_commit() -> TestResult {
    with_device(Device::cpu(), || {
        let sampler = SamplingEngine::new(
            SamplerOptions {
                temperature: 1.0,
                seed: Some(1729),
                ..Default::default()
            },
            3,
            None,
            None,
            None,
        )?;
        let row = Array::from_slice(&[0.1f32.ln(), 0.2f32.ln(), 0.7f32.ln()], &[1, 3]);
        let rng = RandomState::with_seed(1729)?;
        let mut expected_rng = rng.clone();
        let key = expected_rng.next_key()?;
        let expected = mlx_rs::random::categorical(&row, -1, None, &key)?;
        let pending = sampler.sample(&[], row.clone(), Some(&rng), true)?;
        assert_eq!(pending.token.shape(), &[1]);
        assert_eq!(pending.filtered_logprobs.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(
            pending.token.try_item_exact::<u32>()?,
            expected.try_item_exact::<u32>()?
        );
        assert_eq!(
            key_values(pending.rng.as_ref().unwrap().as_array()),
            key_values(expected_rng.as_array())
        );
        let retried = sampler.sample(&[], row, Some(&rng), false)?;
        assert_eq!(
            pending.token.try_item_exact::<u32>()?,
            retried.token.try_item_exact::<u32>()?
        );
        Ok(())
    })
}

#[test]
fn logits_shape_and_dtype_are_checked_before_evaluation() -> TestResult {
    with_device(Device::cpu(), || {
        let sampler = SamplingEngine::new(SamplerOptions::default(), 3, None, None, None)?;
        for shape in [&[3][..], &[3, 1], &[1, 1, 3], &[1, 2]] {
            let row = mlx_rs::ops::zeros::<f32>(shape)?;
            assert!(matches!(
                sampler.sample(&[], row, None, false),
                Err(GenerationError::Sampling(
                    SamplingError::InvalidLogitsShape { .. }
                ))
            ));
        }
        let row = Array::from_slice(&[0i32, 1, 2], &[1, 3]);
        assert!(matches!(
            sampler.sample(&[], row, None, false),
            Err(GenerationError::Sampling(
                SamplingError::InvalidLogitsDtype(Dtype::Int32)
            ))
        ));
        Ok(())
    })
}

#[test]
fn runtime_failures_keep_the_original_exception_at_the_sampling_boundary() -> TestResult {
    with_device(Device::cpu(), || {
        let sampler = SamplingEngine::new(
            SamplerOptions {
                temperature: 1.0,
                seed: Some(1729),
                ..Default::default()
            },
            3,
            None,
            None,
            None,
        )?;
        for fail_on_sample in [false, true] {
            let mut evaluate = |arrays: &[&Array]| {
                if !fail_on_sample
                    || arrays
                        .iter()
                        .any(|array| array.dtype() == Dtype::Uint32 && array.shape() == [1])
                {
                    Err(mlx_rs::error::Exception::from(
                        "injected sampling evaluation failure",
                    ))
                } else {
                    mlx_rs::transforms::eval(arrays.iter().copied())
                }
            };
            let error = sampler
                .sample_with_evaluator(
                    &[],
                    Array::from_slice(&[0.0f32, 1.0, 2.0], &[1, 3]),
                    None,
                    false,
                    &mut evaluate,
                )
                .err()
                .unwrap();
            assert!(
                matches!(error, GenerationError::Exception(source) if source.what() == "injected sampling evaluation failure")
            );
        }
        Ok(())
    })
}

fn fixture(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../conformance/mlx-lm/fixtures")
        .join(name)
}

fn json_file(path: impl AsRef<std::path::Path>) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn engine_from_case(options: &serde_json::Value, seed: u64, vocabulary: usize) -> SamplingEngine {
    let number = |key: &str| options[key].as_f64().unwrap_or(0.0) as f32;
    let additive = |prefix: &str| {
        options[format!("{prefix}_penalty")]
            .as_f64()
            .map(|penalty| AdditivePenaltyOptions {
                penalty: penalty as f32,
                context_size: nz(options[format!("{prefix}_context_size")]
                    .as_u64()
                    .unwrap_or(20) as usize),
            })
    };
    SamplingEngine::new(SamplerOptions {
        temperature: number("temperature"),
        top_p: (number("top_p") != 0.0).then(|| number("top_p")),
        top_k: NonZeroUsize::new(options["top_k"].as_u64().unwrap_or(0) as usize),
        min_p: (number("min_p") != 0.0).then(|| MinPOptions {
            probability: number("min_p"),
            min_tokens_to_keep: nz(options["min_tokens_to_keep"].as_u64().unwrap_or(1) as usize),
        }),
        seed: Some(seed),
    }, vocabulary,
    options["repetition_penalty"].as_f64().filter(|penalty| *penalty != 0.0).map(|penalty| RepetitionPenaltyOptions {
        penalty: penalty as f32,
        context_size: nz(options["repetition_context_size"].as_u64().unwrap_or(20) as usize),
    }), additive("presence"), additive("frequency")).unwrap()
}

fn assert_golden(
    actual: &Array,
    tensors: &safetensors::SafeTensors<'_>,
    key: &str,
    tolerance: &serde_json::Value,
) {
    let tensor = tensors.tensor(key).unwrap();
    assert_eq!(tensor.dtype(), safetensors::Dtype::F32);
    assert_eq!(actual.dtype(), Dtype::Float32, "{key}");
    let expected: Vec<_> = tensor
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    let actual = values(actual);
    assert_eq!(actual.len(), expected.len(), "{key}");
    let atol = tolerance["atol"].as_f64().unwrap() as f32;
    let rtol = tolerance["rtol"].as_f64().unwrap() as f32;
    for (index, (&actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            !actual.is_nan() && actual != f32::INFINITY,
            "{key}[{index}] invalid value"
        );
        assert_eq!(
            actual.is_finite(),
            expected.is_finite(),
            "{key}[{index}] support"
        );
        if expected.is_finite() {
            assert!(
                (actual - expected).abs() <= atol + rtol * expected.abs(),
                "{key}[{index}]: {actual} != {expected}"
            );
        }
    }
}

fn fixture_sampling(name: &str, only_additive: bool) -> TestResult {
    let path = fixture(name);
    let inputs = json_file(path.join("inputs.json"));
    let expected = json_file(path.join("expectations.json"));
    let bytes = std::fs::read(path.join("expectations.safetensors"))?;
    let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
    let prompt: Vec<u32> = inputs["token_ids"][inputs["canonical_prompt"].as_str().unwrap()]
        .as_array()
        .unwrap()
        .iter()
        .map(|id| id.as_u64().unwrap() as u32)
        .collect();
    let raw = crate::config::RawConfig::from_bytes(&std::fs::read(path.join("config.json"))?)?;
    let factory = crate::arch::factory(&raw.model_type)?;
    let manifest = crate::weights::WeightManifest::discover(&path)?;
    let mut decoder = factory.build(factory.parse_config(&raw)?, &manifest)?;
    let vocabulary = decoder.config().dimensions.vocabulary_size;
    let cases = expected["sampling"].as_object().unwrap();
    let mut checked = 0;
    for (case, golden) in cases {
        let additive = [
            "presence_penalty",
            "frequency_penalty",
            "penalties_combined",
        ]
        .contains(&case.as_str());
        if additive != only_additive {
            continue;
        }
        let sampler = engine_from_case(
            &golden["options"],
            golden["seed"].as_u64().unwrap(),
            vocabulary,
        );
        let mut cache = crate::Cache::new(
            decoder.config().model_type.clone(),
            decoder.cache_layout(),
            &crate::CacheOptions::default(),
            None,
        )?;
        let mut step = cache.step()?;
        let prefill = decoder.forward(
            &Array::from_slice(&prompt[..prompt.len() - 1], &[1, prompt.len() as i32 - 1]),
            &mut step,
        )?;
        step.evaluate_and_commit(&[&prefill])?;
        let mut history = vec![TokenId::from(*prompt.last().unwrap())];
        let mut rng = None;
        let expected_ids = golden["cpu_ids"].as_array().unwrap();
        assert_eq!(expected_ids.len(), 8, "{name}/{case}");
        // Empty stop set: every sample becomes the next input, including EOS IDs.
        for (position, expected_id) in expected_ids.iter().enumerate() {
            let input = u32::from(*history.last().unwrap());
            let mut step = cache.step()?;
            let logits = decoder.forward(&Array::from_slice(&[input], &[1, 1]), &mut step)?;
            let row = logits.try_index((.., -1, ..))?;
            if let Some(histories) = golden["processor_token_histories"].as_array() {
                assert_eq!(
                    serde_json::json!(history.iter().copied().map(u32::from).collect::<Vec<_>>()),
                    histories[position],
                    "{name}/{case} history {position}"
                );
            }
            let pending = sampler.sample(&history, row, rng.as_ref(), true)?;
            if position == 0 {
                assert_golden(
                    pending.filtered_logprobs.as_ref().unwrap(),
                    &tensors,
                    &format!("sampling.{case}.filtered_logprobs"),
                    &expected["tolerances"]["logprobs"],
                );
            }
            let token = pending.token.try_item_exact::<u32>()?;
            assert_eq!(
                u64::from(token),
                expected_id.as_u64().unwrap(),
                "{name}/{case} position {position}"
            );
            assert!(
                values(pending.filtered_logprobs.as_ref().unwrap())[token as usize].is_finite()
            );
            step.evaluate_and_commit(&[&logits, &pending.token])?;
            rng = pending.rng;
            history.push(TokenId::from(token));
        }
        checked += 1;
    }
    assert_eq!(checked, if only_additive { 3 } else { 7 });
    Ok(())
}

#[test]
fn all_six_cpu_fixtures_match_filtered_logprobs_and_eight_ids() -> TestResult {
    with_device(Device::cpu(), || {
        for name in [
            "llama-base",
            "llama-sliding",
            "llama-quant4",
            "llama-sharded",
            "qwen3-base",
            "qwen3-quant4",
        ] {
            fixture_sampling(name, false)?;
        }
        Ok(())
    })
}

#[test]
fn additive_cpu_fixture_trajectories() -> TestResult {
    with_device(Device::cpu(), || fixture_sampling("llama-base", true))
}

#[test]
fn processor_oracle_table() -> TestResult {
    with_device(Device::cpu(), || {
        let path = fixture("llama-base");
        let expected = json_file(path.join("expectations.json"));
        let bytes = std::fs::read(path.join("expectations.safetensors"))?;
        let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
        let cases = expected["processing"].as_object().unwrap();
        assert_eq!(cases.len(), 8);
        for (case, golden) in cases {
            let input_logits: Vec<f32> = golden["input_logits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect();
            let engine = engine_from_case(&golden["options"], 1729, input_logits.len());
            let rows: Vec<_> = golden["histories"]
                .as_array()
                .unwrap()
                .iter()
                .map(|history| {
                    let history: Vec<TokenId> = history
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|id| TokenId::from(id.as_u64().unwrap() as u32))
                        .collect();
                    engine
                        .process(
                            &history,
                            Array::from_slice(&input_logits, &[1, input_logits.len() as i32]),
                        )
                        .unwrap()
                })
                .collect();
            assert_golden(
                &mlx_rs::ops::concatenate(&rows.iter().collect::<Vec<_>>(), 0)?,
                &tensors,
                &format!("processing.{case}.logits"),
                &expected["tolerances"]["logits"],
            );
        }
        Ok(())
    })
}

#[test]
fn half_precision_processors_and_sampling_keep_dtype() -> TestResult {
    with_device(Device::cpu(), || {
        for dtype in [Dtype::Float16, Dtype::Bfloat16, Dtype::Float32] {
            let sampler = SamplingEngine::new(
                SamplerOptions {
                    temperature: 0.7,
                    seed: Some(1729),
                    top_p: Some(0.9),
                    ..Default::default()
                },
                3,
                Some(RepetitionPenaltyOptions {
                    penalty: 1.3,
                    context_size: nz(3),
                }),
                Some(AdditivePenaltyOptions {
                    penalty: 0.6,
                    context_size: nz(3),
                }),
                Some(AdditivePenaltyOptions {
                    penalty: 0.4,
                    context_size: nz(3),
                }),
            )?;
            let row = Array::from_slice(&[-2.0f32, 0.0, 1.0], &[1, 3]).as_dtype(dtype)?;
            let history = [TokenId::from(1), TokenId::from(1), TokenId::from(2)];
            let processed = sampler.process(&history, row.clone())?;
            assert_eq!(processed.dtype(), dtype);
            let exact = SamplingEngine::new(
                SamplerOptions::default(),
                3,
                Some(RepetitionPenaltyOptions {
                    penalty: 2.0,
                    context_size: nz(3),
                }),
                Some(AdditivePenaltyOptions {
                    penalty: 0.5,
                    context_size: nz(3),
                }),
                Some(AdditivePenaltyOptions {
                    penalty: 0.5,
                    context_size: nz(3),
                }),
            )?;
            let exact_processed = exact.process(&history, row.clone())?;
            assert_eq!(exact_processed.dtype(), dtype);
            assert_eq!(values(&exact_processed), [-2.0, -1.5, -0.5]);
            let pending = sampler.sample(&history, row, None, true)?;
            let filtered = pending.filtered_logprobs.as_ref().unwrap();
            assert_eq!(filtered.dtype(), dtype);
            assert!(values(filtered)[pending.token.try_item_exact::<u32>()? as usize].is_finite());
        }
        Ok(())
    })
}

#[test]
fn metal_distribution_and_filtered_support() -> TestResult {
    with_device(Device::gpu(), || {
        let row = Array::from_slice(&[0.1f32.ln(), 0.2f32.ln(), 0.7f32.ln()], &[1, 3]);
        let variants = [
            SamplerOptions {
                temperature: 1.0,
                seed: Some(1729),
                ..Default::default()
            },
            SamplerOptions {
                temperature: 1.0,
                seed: Some(1729),
                top_p: Some(0.75),
                ..Default::default()
            },
            SamplerOptions {
                temperature: 1.0,
                seed: Some(1729),
                min_p: Some(MinPOptions {
                    probability: 0.2,
                    min_tokens_to_keep: nz(1),
                }),
                top_k: Some(nz(2)),
                ..Default::default()
            },
        ];
        // Guard Malloc gives every allocation its own page, so the full draw count takes hours;
        // the bound below scales with the count, so a smaller sample stays a valid guard.
        let draws: usize = if std::env::var("DYLD_INSERT_LIBRARIES")
            .is_ok_and(|value| value.contains("libgmalloc"))
        {
            1_024
        } else {
            16_384
        };
        for (variant, options) in variants.into_iter().enumerate() {
            let sampler = SamplingEngine::new(options, 3, None, None, None)?;
            let mut rng = None;
            let mut counts = [0usize; 3];
            let mut probabilities = Vec::new();
            for draw in 0..draws {
                let pending = sampler.sample(&[], row.clone(), rng.as_ref(), draw == 0)?;
                if let Some(filtered) = &pending.filtered_logprobs {
                    probabilities = values(filtered).into_iter().map(f32::exp).collect();
                    let sum: f32 = probabilities.iter().sum();
                    probabilities.iter_mut().for_each(|p| *p /= sum);
                    let expected = if variant == 0 {
                        [0.1f32, 0.2, 0.7]
                    } else {
                        [0.0, 2.0 / 9.0, 7.0 / 9.0]
                    };
                    for (&actual, expected) in probabilities.iter().zip(expected) {
                        assert!((actual - expected).abs() < 1e-6);
                    }
                }
                let id = pending.token.try_item_exact::<u32>()? as usize;
                assert!(probabilities[id] > 0.0);
                counts[id] += 1;
                rng = pending.rng;
            }
            for (observed, probability) in counts.into_iter().zip(probabilities) {
                if probability == 0.0 {
                    assert_eq!(observed, 0);
                } else {
                    let p = f64::from(probability);
                    let expected = draws as f64 * p;
                    assert!(
                        (observed as f64 - expected).abs()
                            <= 8.0 * (expected * (1.0 - p)).sqrt() + 8.0
                    );
                }
            }
        }
        let greedy = SamplingEngine::new(SamplerOptions::default(), 3, None, None, None)?;
        let pending = greedy.sample(
            &[],
            Array::from_slice(&[0.0f32, 2.0, 2.0], &[1, 3]),
            None,
            false,
        )?;
        assert_eq!(pending.token.try_item_exact::<u32>()?, 1);
        Ok(())
    })
}

#[test]
fn composition_uses_repetition_then_presence_then_frequency_and_expires_each_context() -> TestResult
{
    with_device(Device::cpu(), || {
        let sampler = SamplingEngine::new(
            SamplerOptions::default(),
            5,
            Some(RepetitionPenaltyOptions {
                penalty: 2.0,
                context_size: nz(2),
            }),
            Some(AdditivePenaltyOptions {
                penalty: 0.5,
                context_size: nz(3),
            }),
            Some(AdditivePenaltyOptions {
                penalty: -0.5,
                context_size: nz(4),
            }),
        )?;
        let history: Vec<_> = [1, 2, 1, 2, 4, 1, 3].map(TokenId::from).into();
        let source = [0.0f32, 2.0, -3.0, 1.0, -0.5];
        let processed = sampler.process(&history, Array::from_slice(&source, &[1, 5]))?;
        assert_eq!(values(&processed), [0.0, 1.0, -2.5, 0.5, -0.5]);
        let repeated = RepetitionPenaltyOptions {
            penalty: 2.0,
            context_size: nz(3),
        };
        assert_eq!(
            values(&repeated.process(&history[..3], Array::from_slice(&source, &[1, 5]))?),
            [0.0, 1.0, -6.0, 1.0, -0.5]
        );
        for processor in [
            sampler.presence.as_ref().unwrap() as &dyn LogitsProcessor,
            sampler.frequency.as_ref().unwrap() as &dyn LogitsProcessor,
        ] {
            assert_eq!(
                values(&processor.process(&[], Array::from_slice(&source, &[1, 5]))?),
                source
            );
        }
        Ok(())
    })
}

#[test]
fn partition_ties_use_python_membership_and_filters_do_not_renormalize() -> TestResult {
    with_device(Device::cpu(), || {
        let row = Array::from_slice(&[-1.0f32; 7], &[1, 7]);
        let partitions = mlx_rs::ops::argpartition_axis(row.negative()?, 2, -1)?;
        partitions.eval()?;
        let expected_ids = &partitions.as_slice::<u32>()[..3];
        let filtered = values(&top_k(&row, 3)?);
        for (id, value) in filtered.iter().enumerate() {
            assert_eq!(value.is_finite(), expected_ids.contains(&(id as u32)));
        }
        let sampler = SamplingEngine::new(
            SamplerOptions {
                temperature: 1.0,
                top_p: Some(0.7),
                min_p: Some(MinPOptions {
                    probability: 0.5,
                    min_tokens_to_keep: nz(1),
                }),
                top_k: Some(nz(2)),
                ..Default::default()
            },
            4,
            None,
            None,
            None,
        )?;
        let probabilities = [0.05f32, 0.15, 0.3, 0.5];
        let row = Array::from_slice(&probabilities.map(f32::ln), &[1, 4]);
        let filtered = values(&sampler.filter(row)?);
        assert_eq!(
            filtered,
            [
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.3f32.ln(),
                0.5f32.ln()
            ]
        );
        Ok(())
    })
}

#[test]
fn engine_validates_options_and_penalties_in_contract_order_without_rng() {
    let invalid_additive = AdditivePenaltyOptions {
        penalty: f32::NAN,
        context_size: nz(3),
    };
    let invalid_repetition = RepetitionPenaltyOptions {
        penalty: 0.0,
        context_size: nz(3),
    };
    assert!(matches!(
        SamplingEngine::new(
            SamplerOptions {
                top_p: Some(0.0),
                ..Default::default()
            },
            3,
            Some(invalid_repetition.clone()),
            Some(invalid_additive.clone()),
            Some(invalid_additive.clone())
        ),
        Err(SamplingError::InvalidProbability { .. })
    ));
    assert!(matches!(
        SamplingEngine::new(
            SamplerOptions::default(),
            3,
            Some(invalid_repetition),
            Some(invalid_additive.clone()),
            Some(invalid_additive.clone())
        ),
        Err(SamplingError::InvalidRepetitionPenalty(0.0))
    ));
    assert!(
        matches!(SamplingEngine::new(SamplerOptions::default(), 3, None, Some(invalid_additive), Some(AdditivePenaltyOptions { penalty: f32::INFINITY, context_size: nz(3) })), Err(SamplingError::InvalidAdditivePenalty(penalty)) if penalty.is_nan())
    );
    for seed in [None, Some(0), Some(u64::MAX)] {
        assert!(SamplingEngine::new(
            SamplerOptions {
                seed,
                ..Default::default()
            },
            3,
            None,
            None,
            None
        )
        .is_ok());
    }
}

#[test]
fn filtered_empty_support_and_scaled_nan_are_rejected() -> TestResult {
    with_device(Device::cpu(), || {
        let sampler = SamplingEngine::new(
            SamplerOptions {
                temperature: 1.0,
                top_p: Some(f32::from_bits(1)),
                ..Default::default()
            },
            2,
            None,
            None,
            None,
        )?;
        assert!(matches!(
            sampler.sample(&[], Array::from_slice(&[0.0f32, 0.0], &[1, 2]), None, false),
            Err(GenerationError::Sampling(SamplingError::EmptySupport))
        ));
        let sampler = SamplingEngine::new(
            SamplerOptions {
                temperature: f32::from_bits(1),
                ..Default::default()
            },
            2,
            None,
            None,
            None,
        )?;
        assert!(matches!(
            sampler.sample(
                &[],
                Array::from_slice(&[0.0f32, f32::NEG_INFINITY], &[1, 2]),
                None,
                false
            ),
            Err(GenerationError::Sampling(SamplingError::InvalidLogitsValue))
        ));
        let row = Array::from_slice(&[f32::NEG_INFINITY, -2.0f32, 0.0], &[1, 3]);
        let rescued = min_p(
            &row,
            &MinPOptions {
                probability: 1.0,
                min_tokens_to_keep: nz(3),
            },
        )?;
        assert_eq!(values(&rescued), values(&row));
        Ok(())
    })
}
