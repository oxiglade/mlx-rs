use crate::{Cache, CacheError, CacheOptions, InferenceError, Model, TokenId};
use mlx_rs::{error::Exception, ops::concatenate, Array};
use std::{num::NonZeroUsize, ops::Range};

/// Creates the public generation engine with first-sample distribution capture enabled.
pub fn generate_with_logprobs<'m>(
    model: &'m mut Model,
    prompt: crate::Prompt<'_>,
    options: crate::GenerationOptions,
) -> Result<crate::Generation<'m>, crate::GenerationError> {
    let mut generation = model.generate(prompt, options)?;
    generation.capture_first_logprobs();
    Ok(generation)
}

/// Returns the evaluated [1, V] filtered logprobs after the first successful token event.
pub fn first_filtered_logprobs<'a>(generation: &'a crate::Generation<'_>) -> Option<&'a Array> {
    generation.first_logprobs()
}

pub fn prefill_logits<'m>(
    model: &'m mut Model,
    tokens: &[TokenId],
    chunk: Option<NonZeroUsize>,
) -> Result<(Array, OracleSession<'m>), InferenceError> {
    if tokens.is_empty() {
        return Err(InferenceError::EmptyPrompt);
    }
    let architecture = model.config().model_type.clone();
    let cache = Cache::new(
        architecture,
        model.decoder_mut().cache_layout(),
        &CacheOptions::default(),
        None,
    )?;
    let mut session = OracleSession { model, cache };
    let mut logits = Vec::new();
    for tokens in tokens.chunks(chunk.map_or(tokens.len(), NonZeroUsize::get)) {
        logits.push(session.forward(tokens)?);
    }
    Ok((concatenate(&logits, 1)?, session))
}

pub struct OracleSession<'m> {
    model: &'m mut Model,
    cache: Cache,
}

impl OracleSession<'_> {
    pub fn decode_step(&mut self, token: TokenId) -> Result<Array, InferenceError> {
        self.forward(&[token])
    }

    pub fn cache_view(&self) -> Result<CacheSnapshotView, CacheError> {
        let layers = self
            .cache
            .logical_layers()?
            .into_iter()
            .map(|view| LayerView {
                layer: view.layer,
                keys: view.keys,
                values: view.values,
                positions: view.positions,
            })
            .collect();
        Ok(CacheSnapshotView { layers })
    }

    fn forward(&mut self, tokens: &[TokenId]) -> Result<Array, InferenceError> {
        let length = i32::try_from(tokens.len())
            .map_err(|_| Exception::custom("token chunk length exceeds the MLX dimension limit"))?;
        let tokens: Vec<u32> = tokens.iter().copied().map(u32::from).collect();
        let tokens = Array::from_slice(&tokens, &[1, length]);
        let mut step = self.cache.step()?;
        let logits = self.model.decoder_mut().forward(&tokens, &mut step)?;
        step.evaluate_and_commit(&[&logits])?;
        Ok(logits)
    }
}

pub struct CacheSnapshotView {
    pub layers: Vec<LayerView>,
}

pub struct LayerView {
    pub layer: usize,
    pub keys: Array,
    pub values: Array,
    pub positions: Range<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;
    use mlx_rs::ops::indexing::TryIndexOp;
    use serde::Deserialize;
    use std::{collections::HashMap, path::Path};

    #[derive(Deserialize)]
    struct Expectations {
        prefill: Prefill,
        decode: Decode,
        // The fixture's cache map also carries the trim-after-wrap cohort, whose value is an
        // object rather than a per-layer list; the hooks test reads only the layer lists.
        cache: HashMap<String, serde_json::Value>,
        tolerances: HashMap<String, Tolerance>,
    }

    #[derive(Deserialize)]
    struct Prefill {
        token_ids: Vec<u32>,
    }

    #[derive(Deserialize)]
    struct Decode {
        greedy_ids: Vec<u32>,
    }

    #[derive(Deserialize)]
    struct CacheState {
        offset: usize,
        retained_range: [usize; 2],
    }

    #[derive(Deserialize)]
    struct Inputs {
        chunk_sizes: Vec<NonZeroUsize>,
    }

    #[derive(Deserialize)]
    struct Tolerance {
        atol: f32,
        rtol: f32,
    }

    fn load_fixture(path: &Path) -> anyhow::Result<Option<Model>> {
        match Model::from_dir(path) {
            Ok(model) => Ok(Some(model)),
            Err(crate::LoadError::Weights(crate::WeightError::UnsupportedFormat(message)))
            | Err(crate::LoadError::Config(crate::ConfigError::UnsupportedArchitecture(message)))
                if message.ends_with("not yet implemented in this tranche") =>
            {
                eprintln!("NOT RUN: oracle hooks for {}: {message}", path.display());
                Ok(None)
            }
            Err(error) => Err(error.into()),
        }
    }

    #[test]
    fn empty_prompt_is_rejected_before_mlx() -> anyhow::Result<()> {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../conformance/mlx-lm/fixtures/llama-base");
        let Some(mut model) = load_fixture(&path)? else {
            anyhow::bail!("NOT RUN: empty-prompt test requires a real Model::from_dir");
        };
        for chunk in [None, Some(NonZeroUsize::MIN)] {
            assert!(matches!(
                prefill_logits(&mut model, &[], chunk),
                Err(InferenceError::EmptyPrompt)
            ));
        }
        Ok(())
    }

    fn compare(
        actual: &Array,
        expected: &Array,
        tolerance: &Tolerance,
        name: &str,
    ) -> anyhow::Result<()> {
        assert_eq!(actual.shape(), expected.shape(), "{name}");
        let actual = actual.contiguous()?.to_vec_exact::<f32>()?;
        let expected = expected.contiguous()?.to_vec_exact::<f32>()?;
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            anyhow::ensure!(
                actual.is_finite()
                    && expected.is_finite()
                    && (actual - expected).abs()
                        <= tolerance.atol + tolerance.rtol * expected.abs(),
                "{name}[{index}]: {actual} != {expected}"
            );
        }
        Ok(())
    }

    fn compare_cache(
        session: &OracleSession<'_>,
        expectations: &Expectations,
        tensors: &HashMap<String, Array>,
        stage: &str,
        chunk: Option<NonZeroUsize>,
    ) -> anyhow::Result<()> {
        let states: Vec<CacheState> = serde_json::from_value(
            expectations
                .cache
                .get(stage)
                .context("missing cache state")?
                .clone(),
        )?;
        let tolerance = expectations
            .tolerances
            .get("cache")
            .context("missing cache tolerance")?;
        let view = session.cache_view()?;
        assert_eq!(view.layers.len(), states.len());
        for ((layer, state), info) in view.layers.iter().zip(states).zip(session.cache.info()) {
            assert_eq!(layer.layer, info.layer);
            assert_eq!(info.processed_tokens, state.offset);
            let [start, end] = state.retained_range;
            let expected_start = match (chunk, info.kind) {
                (Some(chunk), crate::CacheKind::Rotating) => {
                    let last_chunk = (state.offset - 1) % chunk.get() + 1;
                    end.saturating_sub(info.capacity + last_chunk - 1)
                }
                _ => start,
            };
            assert_eq!(layer.positions, expected_start..end);
            for (kind, actual) in [("keys", &layer.keys), ("values", &layer.values)] {
                let name = format!("cache.{stage}.layer{}.{kind}", layer.layer);
                let expected = tensors
                    .get(&name)
                    .with_context(|| format!("missing {name}"))?;
                let first = i32::try_from(layer.positions.start - start)?;
                let last = i32::try_from(layer.positions.end - start)?;
                let expected = expected.try_index((.., .., first..last, ..))?;
                compare(actual, &expected, tolerance, &name)?;
            }
        }
        Ok(())
    }

    #[test]
    fn fixture_prefill_matches_expectations() -> anyhow::Result<()> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../conformance/mlx-lm/fixtures");
        let mut count = 0;
        let mut not_run = 0;
        for entry in std::fs::read_dir(root)? {
            let path = entry?.path();
            // GGUF fixtures carry no config.json; the gguf module tests own them.
            if !path.is_dir() || !path.join("config.json").exists() {
                continue;
            }
            count += 1;
            let Some(mut model) = load_fixture(&path)? else {
                not_run += 1;
                continue;
            };
            let expectations: Expectations =
                serde_json::from_slice(&std::fs::read(path.join("expectations.json"))?)?;
            let inputs: Inputs = serde_json::from_slice(&std::fs::read(path.join("inputs.json"))?)?;
            let tensors = Array::load_safetensors(path.join("expectations.safetensors"))?;
            let tokens: Vec<TokenId> = expectations
                .prefill
                .token_ids
                .iter()
                .copied()
                .map(TokenId::from)
                .collect();
            let tolerance = expectations
                .tolerances
                .get("logits")
                .context("missing logits tolerance")?;
            for chunk in std::iter::once(None).chain(inputs.chunk_sizes.into_iter().map(Some)) {
                let (logits, mut session) = prefill_logits(&mut model, &tokens, chunk)?;
                let name = chunk.map_or_else(
                    || "prefill.full.logits".to_owned(),
                    |size| format!("prefill.chunk{size}.logits"),
                );
                compare(
                    &logits,
                    tensors
                        .get(&name)
                        .with_context(|| format!("missing {name}"))?,
                    tolerance,
                    &name,
                )?;
                compare_cache(&session, &expectations, &tensors, "after_prefill", chunk)?;
                if chunk.is_none() {
                    for (step, token) in expectations.decode.greedy_ids.iter().enumerate() {
                        let logits = session.decode_step(TokenId::from(*token))?;
                        let name = format!("decode.step{step}.logits");
                        let expected = tensors
                            .get(&name)
                            .with_context(|| format!("missing {name}"))?
                            .expand_dims(1)?;
                        compare(&logits, &expected, tolerance, &name)?;
                    }
                    compare_cache(&session, &expectations, &tensors, "after_decode", None)?;
                }
            }
        }
        anyhow::ensure!(count > 0, "no fixture directories found");
        if not_run > 0 {
            eprintln!("NOT RUN: oracle hooks for {not_run}/{count} fixtures");
            anyhow::bail!(
                "NOT RUN: {not_run} fixtures still require architecture or loader implementation"
            );
        }
        Ok(())
    }
}
