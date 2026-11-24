use super::{pipeline::assert_scripted_prefix, *};
use crate::arch::DraftContext;
use std::cell::{Cell, RefCell};

/// A target whose token after `t` is `t + 1`, with a drafter of three proposals a round that
/// gets a scripted number of them right.
struct DraftedDecoder {
    config: Config,
    layout: Vec<LayerCacheSpec>,
    right: Vec<usize>,
    /// The token of every row the drafter was shown.
    shown: Rc<RefCell<Vec<f32>>>,
    rounds: Rc<Cell<usize>>,
}
impl DecoderModel for DraftedDecoder {
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
        Ok(self.forward_taps(tokens, None, cache)?.0)
    }
    fn weight_projection(
        &mut self,
    ) -> Result<mlx_rs::utils::StateProjection<'_>, crate::WeightError> {
        unreachable!("scripted decoder does not load weights")
    }
    fn draft_block(&self) -> Option<usize> {
        Some(4)
    }
    fn forward_taps(
        &mut self,
        tokens: &Array,
        _: Option<&Array>,
        cache: &mut CacheStep<'_>,
    ) -> Result<(Array, Array), InferenceError> {
        let rows = tokens.as_dtype(Dtype::Float32)?;
        let stored = rows.reshape(&[1, 1, -1, 1])?;
        let mut dependency = Array::from(0.0f32);
        for layer in 0..self.layout.len() {
            let (keys, values) = cache.update_and_fetch(layer, stored.clone(), stored.clone())?;
            dependency = dependency.add(keys.sum(false)?)?.add(values.sum(false)?)?;
        }
        let dependency = dependency.multiply(Array::from(0.0f32))?;
        let vocabulary = self.config.dimensions.vocabulary_size as i32;
        let next = tokens
            .as_dtype(Dtype::Int32)?
            .add(Array::from_int(1))?
            .expand_dims(-1)?;
        let logits = mlx_rs::ops::select(
            Array::arange::<_, i32>(0, vocabulary, None)?.eq(next)?,
            Array::from(0.0f32),
            Array::from(f32::NEG_INFINITY),
        )?;
        Ok((logits.add(&dependency)?, rows.expand_dims(-1)?))
    }
    fn observe(&self, _: &mut DraftContext, taps: &Array) -> Result<(), InferenceError> {
        self.shown.borrow_mut().extend(values(taps));
        Ok(())
    }
    fn propose(
        &self,
        _: &DraftContext,
        anchor: TokenId,
        count: usize,
    ) -> Result<Array, InferenceError> {
        let round = self.rounds.replace(self.rounds.get() + 1);
        let right = self.right.get(round).copied().unwrap_or(0);
        let ids: Vec<u32> = (0..count as u32)
            .map(|i| {
                if (i as usize) < right {
                    u32::from(anchor) + 1 + i
                } else {
                    0
                }
            })
            .collect();
        Ok(Array::from_slice(&ids, &[1, count as i32]))
    }
}

type Drafted = (Model, Rc<RefCell<Vec<f32>>>, Rc<Cell<usize>>);
fn drafted(right: &[usize]) -> anyhow::Result<Drafted> {
    let mut model = scripted(vec![12], None, None, None)?;
    let shown = Rc::new(RefCell::new(Vec::new()));
    let rounds = Rc::new(Cell::new(0));
    model.decoder = Box::new(DraftedDecoder {
        config: model.config.clone(),
        layout: model.decoder.cache_layout().to_vec(),
        right: right.to_vec(),
        shown: shown.clone(),
        rounds: rounds.clone(),
    });
    Ok((model, shown, rounds))
}
fn range(ids: std::ops::Range<u32>) -> Vec<TokenId> {
    ids.map(TokenId::from).collect()
}

#[test]
fn drafted_rounds_publish_the_agreed_proposals_and_one_token_more() -> TestResult {
    with_device(Device::gpu(), || {
        // How many proposals each round gets right, and the rounds the eight tokens after the
        // first then take.
        for (right, rounds) in [(&[][..], 8), (&[3, 3], 2), (&[1, 3, 0, 2], 4)] {
            let (mut model, shown, count) = drafted(right)?;
            let mut cache = model.new_cache(Default::default())?;
            let mut options = length_options(9);
            options.draft = true;
            let mut generation =
                model.generate_with_cache(Prompt::Tokens(&range(20..22)), options, &mut cache)?;
            let events = collect_tokens(&mut generation)?;
            drop(generation);
            assert_eq!(token_ids(&events), range(22..31), "{right:?}");
            assert_eq!(count.get(), rounds, "{right:?}");
            assert_scripted_prefix(&cache, &range(20..30))?;
            // The drafter's context is the rows that stayed, without the rejected ones.
            let stayed: Vec<_> = (20..30).map(|id| id as f32).collect();
            assert_eq!(*shown.borrow(), stayed, "{right:?}");
        }
        Ok(())
    })
}

#[test]
fn a_drafted_round_cut_short_leaves_only_published_rows() -> TestResult {
    with_device(Device::gpu(), || {
        // The first round verifies 12 to 15; each case ends at 13, inside it.
        for end in ["token", "string", "drop"] {
            let (mut model, _, _) = drafted(&[3])?;
            let mut options = length_options(12);
            options.draft = true;
            match end {
                "token" => options.stop.tokens = StopTokenPolicy::Exact(vec![13.into()]),
                "string" => options.stop.stop_strings = vec!["hello world".into()],
                _ => {}
            }
            let mut cache = model.new_cache(Default::default())?;
            let mut generation =
                model.generate_with_cache(Prompt::Tokens(&[10.into()]), options, &mut cache)?;
            // Two prefill events, then the tokens.
            let events = generation.by_ref().take(5).collect::<Result<Vec<_>, _>>()?;
            assert_eq!(token_ids(&events), range(11..14), "{end}");
            let stop = (end != "drop").then_some(FinishReason::Stop);
            assert!(
                matches!(events.last(), Some(GenerationEvent::Token { finish_reason, .. }) if *finish_reason == stop),
                "{end}"
            );
            if end != "drop" {
                assert!(generation.next().is_none());
            }
            assert_scripted_prefix(generation.cache()?, &range(10..13))?;
            drop(generation);
            assert_scripted_prefix(&cache, &range(10..13))?;
            // The next request extends the cache from the published rows.
            let mut extension = range(10..13);
            extension.push(25.into());
            let mut retry = model.generate_with_cache(
                Prompt::Tokens(&extension),
                length_options(2),
                &mut cache,
            )?;
            collect_tokens(&mut retry)?;
            drop(retry);
            extension.push(26.into());
            assert_scripted_prefix(&cache, &extension)?;
        }
        Ok(())
    })
}

#[test]
fn requests_that_cannot_be_drafted_decode_serially() -> TestResult {
    for case in [
        "off",
        "sampled",
        "repetition",
        "presence",
        "frequency",
        "cpu",
    ] {
        let device = if case == "cpu" {
            Device::cpu()
        } else {
            Device::gpu()
        };
        with_device(device, || -> TestResult {
            let (mut model, _, rounds) = drafted(&[3])?;
            let mut options = length_options(6);
            options.draft = case != "off";
            let additive = Some(AdditivePenaltyOptions {
                penalty: 0.1,
                context_size: nz(4),
            });
            match case {
                "sampled" => {
                    options.sampling.temperature = 0.7;
                    options.sampling.seed = Some(1729);
                }
                "repetition" => {
                    options.repetition_penalty = Some(RepetitionPenaltyOptions {
                        penalty: 1.1,
                        context_size: nz(4),
                    });
                }
                "presence" => options.presence_penalty = additive,
                "frequency" => options.frequency_penalty = additive,
                _ => {}
            }
            let mut generation = model.generate(Prompt::Tokens(&[20.into()]), options)?;
            let events = collect_tokens(&mut generation)?;
            assert_eq!(token_ids(&events), range(21..27), "{case}");
            assert_eq!(rounds.get(), 0, "{case}");
            Ok(())
        })?;
    }
    Ok(())
}
