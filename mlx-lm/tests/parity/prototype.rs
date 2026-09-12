use std::{num::NonZeroUsize, path::Path};

use anyhow::{bail, ensure, Context, Result};
use mlx_lm::{
    oracle_hooks, AdditivePenaltyOptions, FinishReason, Generation, GenerationEvent,
    GenerationOptions, MinPOptions, Model, Prompt, RepetitionPenaltyOptions, StopTokenPolicy,
    TokenId,
};
use mlx_rs::{ops::indexing::IndexOp, Array, Dtype};
use safetensors::Dtype as SafeDtype;
use serde_json::{json, Value};

use super::{
    comparator,
    observation::{CacheState, Expectations, Observation, Tensor},
    reader::{Fixture, StreamCase},
    surfaces,
};

pub fn observe(array: &Array) -> Result<Tensor> {
    // Cache transposes and sliced logits need row-major materialization before exact host reads.
    let array = array.contiguous()?;
    array.eval()?;
    let shape = array
        .shape()
        .iter()
        .map(|&d| usize::try_from(d))
        .collect::<Result<Vec<_>, _>>()?;
    let (dtype, bytes) = match array.dtype() {
        Dtype::Float32 => (
            SafeDtype::F32,
            array
                .to_vec_exact::<f32>()?
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect(),
        ),
        dtype => bail!("adapter emitted unsupported dtype {dtype:?}"),
    };
    Ok(Tensor {
        shape,
        dtype,
        bytes,
    })
}

pub fn compare(label: &str, expected: &Expectations, observed: &Observation) -> Result<()> {
    let failures = comparator::compare(expected, observed);
    ensure!(
        failures.is_empty(),
        "{label}:\n{}",
        failures
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    );
    Ok(())
}

fn ids(tokens: &[TokenId]) -> Vec<u32> {
    tokens.iter().copied().map(u32::from).collect()
}

fn tokens(ids: &[u32]) -> Vec<TokenId> {
    ids.iter().copied().map(TokenId::from).collect()
}

fn limit(fixture: &Fixture) -> Result<NonZeroUsize> {
    nonzero(&fixture.inputs["decode_steps"])
}

fn nonzero(value: &Value) -> Result<NonZeroUsize> {
    NonZeroUsize::new(usize::try_from(
        value.as_u64().context("expected positive integer")?,
    )?)
    .context("expected nonzero integer")
}

fn options(fixture: &Fixture) -> Result<GenerationOptions> {
    let mut options = GenerationOptions::default();
    options.max_tokens = limit(fixture)?;
    Ok(options)
}

fn schedule(total: usize, ceiling: usize) -> Vec<[usize; 2]> {
    let mut pairs = vec![[0, total]];
    for processed in (ceiling..total - 1).step_by(ceiling) {
        pairs.push([processed, total]);
    }
    if total > 1 {
        pairs.push([total - 1, total]);
    }
    pairs.push([total, total]);
    pairs
}

fn collect(generation: &mut Generation<'_>, total: usize, ceiling: usize) -> Result<Observation> {
    let mut observed = Observation {
        greedy_ids: Some(Vec::new()),
        text_deltas: Some(Vec::new()),
        ..Observation::default()
    };
    let mut progress = Vec::new();
    let mut started = false;
    for event in generation.by_ref() {
        ensure!(observed.finish.is_none(), "event after terminal token");
        match event? {
            GenerationEvent::Prefill {
                processed, total, ..
            } => {
                ensure!(!started, "prefill event after token");
                progress.push([processed, total]);
            }
            GenerationEvent::Token {
                token_id,
                text,
                finish_reason,
                ..
            } => {
                started = true;
                let token = u32::from(token_id);
                observed.greedy_ids.as_mut().unwrap().push(token);
                observed.text_deltas.as_mut().unwrap().push(text);
                observed.finish = match finish_reason {
                    None => None,
                    Some(FinishReason::Length) => Some(("length".into(), None)),
                    Some(FinishReason::Stop) => Some(("stop".into(), Some(token))),
                    Some(other) => bail!("unknown finish reason {other:?}"),
                };
            }
            other => bail!("unknown generation event {other:?}"),
        }
    }
    ensure!(
        observed.finish.is_some(),
        "stream ended without terminal token"
    );
    compare(
        "stream prefill",
        &Expectations {
            observation: Observation {
                progress: Some(json!(schedule(total, ceiling))),
                ..Observation::default()
            },
            policies: Default::default(),
        },
        &Observation {
            progress: Some(json!(progress)),
            ..Observation::default()
        },
    )?;
    Ok(observed)
}

fn stream_expectation(case: &StreamCase) -> Expectations {
    Expectations {
        observation: Observation {
            greedy_ids: Some(case.token_ids.clone()),
            text_deltas: Some(case.text_deltas.clone()),
            finish: Some((case.finish_reason.clone(), case.stop_token)),
            ..Observation::default()
        },
        policies: Default::default(),
    }
}

fn sampling_options(fixture: &Fixture, recipe: &Value, seed: u64) -> Result<GenerationOptions> {
    let mut options = options(fixture)?;
    options.stop.tokens = StopTokenPolicy::Exact(vec![]);
    options.sampling.seed = Some(seed);
    options.sampling.temperature = recipe["temperature"].as_f64().context("temperature")? as f32;
    options.sampling.top_p = recipe["top_p"]
        .as_f64()
        .filter(|&p| p != 0.0)
        .map(|p| p as f32);
    options.sampling.top_k =
        NonZeroUsize::new(usize::try_from(recipe["top_k"].as_u64().context("top_k")?)?);
    options.sampling.min_p = recipe["min_p"]
        .as_f64()
        .filter(|&p| p != 0.0)
        .map(|p| {
            Ok::<_, anyhow::Error>(MinPOptions {
                probability: p as f32,
                min_tokens_to_keep: nonzero(&recipe["min_tokens_to_keep"])?,
            })
        })
        .transpose()?;
    options.repetition_penalty = recipe["repetition_penalty"]
        .as_f64()
        .map(|penalty| {
            Ok::<_, anyhow::Error>(RepetitionPenaltyOptions {
                penalty: penalty as f32,
                context_size: nonzero(&recipe["repetition_context_size"])?,
            })
        })
        .transpose()?;
    for (field, context, destination) in [
        (
            "presence_penalty",
            "presence_context_size",
            &mut options.presence_penalty,
        ),
        (
            "frequency_penalty",
            "frequency_context_size",
            &mut options.frequency_penalty,
        ),
    ] {
        *destination = recipe[field]
            .as_f64()
            .map(|penalty| {
                Ok::<_, anyhow::Error>(AdditivePenaltyOptions {
                    penalty: penalty as f32,
                    context_size: nonzero(&recipe[context])?,
                })
            })
            .transpose()?;
    }
    Ok(options)
}

fn progress(model: &mut Model, fixture: &Fixture) -> Result<()> {
    let cases = if let Some(progress) = &fixture.prefill.progress {
        progress.clone()
    } else {
        let mut cases = serde_json::Map::new();
        for ceiling in fixture.inputs["chunk_sizes"]
            .as_array()
            .context("chunk sizes")?
        {
            let ceiling = nonzero(ceiling)?.get();
            cases.insert(format!("chunk{ceiling}"), json!({"ceiling": ceiling, "token_ids": fixture.prefill.token_ids, "pairs": schedule(fixture.prefill.length, ceiling)}));
        }
        cases.insert("one_chunk1".into(), json!({"ceiling": 1, "token_ids": [fixture.prefill.token_ids[0]], "pairs": schedule(1, 1)}));
        Value::Object(cases)
    };
    let mut actual = serde_json::Map::new();
    for (name, case) in cases.as_object().context("progress cases")? {
        let prompt_ids: Vec<u32> = serde_json::from_value(case["token_ids"].clone())?;
        let prompt = tokens(&prompt_ids);
        let mut options = options(fixture)?;
        options.max_tokens = NonZeroUsize::MIN;
        options.prefill_chunk_size = nonzero(&case["ceiling"])?;
        let mut generation = model.generate(Prompt::Tokens(&prompt), options)?;
        let mut pairs = Vec::new();
        let mut token_count = 0;
        // One sampled token lets us check the entire prefill sequence despite Rust's nonzero limit.
        for event in generation.by_ref() {
            match event? {
                GenerationEvent::Prefill {
                    processed, total, ..
                } => {
                    ensure!(token_count == 0, "prefill after token");
                    pairs.push([processed, total]);
                }
                GenerationEvent::Token { finish_reason, .. } => {
                    ensure!(finish_reason.is_some(), "missing terminal reason");
                    token_count += 1;
                }
                other => bail!("unknown generation event {other:?}"),
            }
        }
        ensure!(token_count == 1, "expected one sampled token");
        actual.insert(name.clone(), json!({"ceiling": case["ceiling"], "token_ids": ids(generation.cache().tokens()), "pairs": pairs}));
    }
    compare(
        "prefill.progress",
        &Expectations {
            observation: Observation {
                progress: Some(cases),
                ..Observation::default()
            },
            policies: Default::default(),
        },
        &Observation {
            progress: Some(Value::Object(actual)),
            ..Observation::default()
        },
    )
}

pub fn run(path: &Path, fixture: &Fixture) -> Result<()> {
    let mut model = Model::from_dir(path)?;
    let prompt = tokens(&fixture.prefill.token_ids);
    let mut observed = surfaces::run(path, fixture, &mut model)?;
    let mut expected = surfaces::expectations(fixture)?;
    let settings = options(fixture)?;
    let mut generation = model.generate(Prompt::Tokens(&prompt), settings.clone())?;
    let stream = collect(
        &mut generation,
        prompt.len(),
        settings.prefill_chunk_size.get(),
    )?;
    drop(generation);
    compare(
        "default decode",
        &Expectations {
            observation: Observation {
                greedy_ids: Some(fixture.decode.token_ids.clone()),
                text_deltas: Some(fixture.decode.text_deltas.clone()),
                finish: Some((
                    fixture.decode.finish_reason.clone(),
                    fixture.decode.stop_token,
                )),
                ..Observation::default()
            },
            policies: Default::default(),
        },
        &stream,
    )?;
    observed.greedy_ids = stream.greedy_ids;
    observed.text_deltas = stream.text_deltas;
    observed.finish = stream.finish;
    for (name, case) in [
        ("length", &fixture.decode.length_case),
        ("stop", &fixture.decode.stop_case),
    ] {
        let case = case
            .as_ref()
            .with_context(|| format!("missing {name} case"))?;
        let mut settings = options(fixture)?;
        settings.stop.tokens = StopTokenPolicy::Exact(tokens(&case.eos_tokens));
        let mut generation = model.generate(Prompt::Tokens(&prompt), settings.clone())?;
        let stream = collect(
            &mut generation,
            prompt.len(),
            settings.prefill_chunk_size.get(),
        )?;
        compare(name, &stream_expectation(case), &stream)?;
    }
    progress(&mut model, fixture)?;
    ensure!(
        fixture.sampling.len() == if path.ends_with("llama-base") { 10 } else { 7 },
        "sampling case count"
    );
    for (name, case) in &fixture.sampling {
        let settings = sampling_options(fixture, &case.options, case.seed)?;
        let mut generation = model.generate(Prompt::Tokens(&prompt), settings.clone())?;
        let stream = collect(
            &mut generation,
            prompt.len(),
            settings.prefill_chunk_size.get(),
        )?;
        ensure!(
            stream.finish == Some(("length".into(), None)),
            "sampling {name} stopped early"
        );
        observed
            .sampled_ids
            .insert(name.clone(), stream.greedy_ids.context("sampled ids")?);
        drop(generation);
        let mut generation =
            oracle_hooks::generate_with_logprobs(&mut model, Prompt::Tokens(&prompt), settings)?;
        loop {
            match generation.next().context("missing first sample")?? {
                GenerationEvent::Prefill { .. } => (),
                GenerationEvent::Token { .. } => break,
                other => bail!("unknown generation event {other:?}"),
            }
        }
        let logprobs = oracle_hooks::first_filtered_logprobs(&generation)
            .context("missing first filtered logprobs")?;
        observed.tensors.insert(
            format!("sampling.{name}.filtered_logprobs"),
            observe(&logprobs.index(0))?,
        );
    }
    for chunk in std::iter::once(None).chain(
        fixture.inputs["chunk_sizes"]
            .as_array()
            .context("chunk sizes")?
            .iter()
            .map(|n| nonzero(n).map(Some))
            .collect::<Result<Vec<_>>>()?,
    ) {
        let (logits, session) = oracle_hooks::prefill_logits(&mut model, &prompt, chunk)?;
        let key = chunk.map_or_else(
            || "prefill.full.logits".into(),
            |n| format!("prefill.chunk{n}.logits"),
        );
        observed.tensors.insert(key, observe(&logits)?);
        if chunk.is_none() {
            for layer in session.cache_view()?.layers {
                let key = format!("cache.after_prefill.layer{}", layer.layer);
                observed
                    .tensors
                    .insert(format!("{key}.keys"), observe(&layer.keys)?);
                observed
                    .tensors
                    .insert(format!("{key}.values"), observe(&layer.values)?);
                observed.caches.insert(
                    key,
                    CacheState {
                        offset: layer.positions.end,
                        retained: layer.positions,
                    },
                );
            }
        }
    }
    expected.observation.tensors.retain(|key, _| {
        key.starts_with("prefill.")
            || key.starts_with("cache.after_prefill.")
            || key.starts_with("sampling.")
    });
    expected
        .policies
        .retain(|key, _| expected.observation.tensors.contains_key(key));
    expected
        .observation
        .caches
        .retain(|key, _| key.starts_with("cache.after_prefill."));
    compare(&path.display().to_string(), &expected, &observed)
}
