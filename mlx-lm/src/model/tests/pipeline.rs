use super::*;

struct CountedDecoder {
    inner: Box<dyn DecoderModel>,
    calls: Rc<std::cell::Cell<usize>>,
    streams: Option<Rc<std::cell::RefCell<Vec<mlx_rs::Stream>>>>,
}
impl DecoderModel for CountedDecoder {
    fn config(&self) -> &Config {
        self.inner.config()
    }
    fn cache_layout(&self) -> &[LayerCacheSpec] {
        self.inner.cache_layout()
    }
    fn forward(
        &mut self,
        tokens: &Array,
        cache: &mut CacheStep<'_>,
    ) -> Result<Array, InferenceError> {
        self.calls.set(self.calls.get() + 1);
        if let Some(streams) = &self.streams {
            streams
                .borrow_mut()
                .push(mlx_rs::Stream::thread_local_or_default());
        }
        self.inner.forward(tokens, cache)
    }
    fn weight_projection(
        &mut self,
    ) -> Result<mlx_rs::utils::StateProjection<'_>, crate::WeightError> {
        self.inner.weight_projection()
    }
}
fn count_forwards(model: &mut Model) -> Rc<std::cell::Cell<usize>> {
    let calls = Rc::new(std::cell::Cell::new(0));
    let placeholder = scripted(vec![12], None, None, None).unwrap().decoder;
    model.decoder = Box::new(CountedDecoder {
        inner: std::mem::replace(&mut model.decoder, placeholder),
        calls: calls.clone(),
        streams: None,
    });
    calls
}

#[test]
fn pipeline_gpu_looks_ahead_but_cpu_and_rotating_do_not() -> TestResult {
    for device in [Device::cpu(), Device::gpu()] {
        let gpu = matches!(device.get_type()?, mlx_rs::DeviceType::Gpu);
        let stream = mlx_rs::Stream::new_with_device(&device);
        mlx_rs::with_stream(&stream, || -> TestResult {
            for rotating in [false, true] {
                for limit in [1, 3] {
                    let mut model = scripted(vec![12, 13, 14], None, None, None)?;
                    let calls = count_forwards(&mut model);
                    let mut options = length_options(limit);
                    if rotating {
                        options.cache.policy = crate::CachePolicy::Rotating {
                            capacity: nz(8),
                            keep_prefix: 0,
                        };
                    }
                    let mut generation = model.generate(Prompt::Tokens(&[12.into()]), options)?;
                    generation.next().transpose()?;
                    generation.next().transpose()?;
                    assert_eq!(calls.get(), 1);
                    generation.next().transpose()?;
                    assert_eq!(calls.get(), 1 + usize::from(gpu && !rotating && limit > 1));
                    assert_eq!(generation.cache()?.tokens()?, &[12.into()]);
                    assert_eq!(
                        generation.cache()?.info()?.next().unwrap().processed_tokens,
                        1
                    );
                }
            }
            Ok(())
        })?;
    }
    Ok(())
}

#[test]
fn pipeline_matches_synchronous_events_rng_and_logprobs() -> TestResult {
    with_device(Device::gpu(), || {
        for name in FIXTURES {
            for temperature in [0.0, 0.7] {
                for penalties in [false, true] {
                    let mut model = Model::from_dir(fixture(name))?;
                    let mut options = length_options(12);
                    options.sampling.temperature = temperature;
                    options.sampling.seed = Some(1729);
                    if temperature != 0.0 {
                        options.sampling.top_p = Some(0.9);
                        options.sampling.top_k = Some(nz(12));
                        options.sampling.min_p = Some(crate::MinPOptions {
                            probability: 0.05,
                            min_tokens_to_keep: nz(2),
                        });
                    }
                    if penalties {
                        options.repetition_penalty = Some(RepetitionPenaltyOptions {
                            penalty: 1.1,
                            context_size: nz(4),
                        });
                        options.presence_penalty = Some(AdditivePenaltyOptions {
                            penalty: 0.1,
                            context_size: nz(1),
                        });
                        options.frequency_penalty = Some(AdditivePenaltyOptions {
                            penalty: -0.1,
                            context_size: nz(6),
                        });
                    }
                    let prompt = [12.into(), 13.into(), 12.into()];
                    let mut runs = vec![];
                    for enabled in [false, true] {
                        let mut generation =
                            model.generate(Prompt::Tokens(&prompt), options.clone())?;
                        if !enabled {
                            generation.pipeline_stream = None;
                        }
                        generation.capture_logprobs = true;
                        let events = generation.by_ref().collect::<Result<Vec<_>, _>>()?;
                        let logprobs = generation
                            .first_logprobs
                            .as_ref()
                            .unwrap()
                            .to_vec_cast::<f32>()?;
                        let rng = generation
                            .rng
                            .as_ref()
                            .map(|r| r.as_array().to_vec_exact::<u32>())
                            .transpose()?;
                        runs.push((events, logprobs, rng, cache_evidence(generation.cache()?)?));
                    }
                    assert_eq!(runs[0], runs[1], "{name}/{temperature}/{penalties}");
                }
            }
        }
        Ok(())
    })
}

fn assert_scripted_prefix(cache: &Cache, expected: &[TokenId]) -> TestResult {
    assert_eq!(cache.tokens()?, expected);
    for layer in cache.logical_layers()? {
        assert_eq!(layer.positions, 0..expected.len());
        let expected: Vec<_> = expected.iter().map(|id| u32::from(*id) as f32).collect();
        assert_eq!(values(&layer.keys), expected);
        assert_eq!(values(&layer.values), expected);
    }
    Ok(())
}

#[test]
fn pipeline_stops_and_cancellation_leave_only_committed_prefix() -> TestResult {
    with_device(Device::gpu(), || {
        for stop in ["length", "token", "string", "drop", "guard"] {
            for accepted in [1, 2, 5] {
                let mut model = scripted(vec![12, 13, 14, 15, 16, 17], None, None, None)?;
                let calls = count_forwards(&mut model);
                let mut options = length_options(if stop == "length" { accepted } else { 8 });
                options.sampling.temperature = 0.7;
                options.sampling.seed = Some(42);
                if stop == "token" {
                    options.stop.tokens =
                        StopTokenPolicy::Exact(vec![TokenId::from(11 + accepted as u32)]);
                }
                if stop == "string" {
                    if accepted != 2 {
                        continue;
                    }
                    options.stop.stop_strings = vec!["hello world".into()];
                }
                let mut cache = model.new_cache(Default::default())?;
                let mut generation =
                    model.generate_with_cache(Prompt::Tokens(&[20.into()]), options, &mut cache)?;
                generation.next().transpose()?;
                generation.next().transpose()?;
                for _ in 0..accepted {
                    generation.next().transpose()?;
                }
                assert_eq!(generation.accepted, accepted);
                assert_eq!(calls.get(), accepted + usize::from(stop != "length"));
                let mut expected = vec![20.into()];
                expected.extend((12..11 + accepted as u32).map(TokenId::from));
                assert_scripted_prefix(generation.cache()?, &expected)?;
                let snapshot = generation.snapshot()?;
                if stop == "guard" {
                    drop(generation.pending.take());
                }
                if !matches!(stop, "drop" | "guard") {
                    assert!(generation.next().is_none());
                }
                drop(generation);
                assert_scripted_prefix(&cache, &expected)?;
                cache.restore(snapshot)?;
                assert_scripted_prefix(&cache, &expected)?;
                let mut extension = expected.clone();
                extension.push(25.into());
                let mut retry = model.generate_with_cache(
                    Prompt::Tokens(&extension),
                    length_options(2),
                    &mut cache,
                )?;
                collect_tokens(&mut retry)?;
                drop(retry);
                assert_eq!(cache.tokens()?[expected.len()], 25.into());
                assert_scripted_prefix(&cache, cache.tokens()?)?;
            }
        }
        Ok(())
    })
}

#[test]
fn pipeline_sampling_error_discards_next_step_without_committing_rng() -> TestResult {
    with_device(Device::gpu(), || {
        for fail_at in [0, 1, 3] {
            let mut model = scripted(vec![12], None, Some(fail_at), None)?;
            let calls = count_forwards(&mut model);
            let mut options = length_options(8);
            options.sampling.temperature = 0.7;
            options.sampling.seed = Some(1729);
            let mut generation = model.generate(Prompt::Tokens(&[20.into()]), options)?;
            generation.next().transpose()?;
            generation.next().transpose()?;
            for _ in 0..fail_at {
                generation.next().transpose()?;
            }
            let before = cache_evidence(generation.cache()?)?;
            let rng = generation
                .rng
                .as_ref()
                .map(|r| r.as_array().to_vec_exact::<u32>())
                .transpose()?;
            assert!(matches!(
                generation.next(),
                Some(Err(GenerationError::Sampling(SamplingError::EmptySupport)))
            ));
            assert_eq!(calls.get(), fail_at + 2);
            assert_eq!(generation.accepted, fail_at);
            assert_eq!(
                generation
                    .rng
                    .as_ref()
                    .map(|r| r.as_array().to_vec_exact::<u32>())
                    .transpose()?,
                rng
            );
            assert!(generation.pending.is_none());
            assert_eq!(cache_evidence(generation.cache()?)?, before);
            assert!(generation.next().is_none());
        }
        Ok(())
    })
}

#[test]
fn pipeline_forward_error_is_delivered_at_its_own_token() -> TestResult {
    with_device(Device::gpu(), || {
        let mut model = scripted(vec![12], Some(2), None, None)?;
        let mut generation = model.generate(Prompt::Tokens(&[20.into()]), length_options(5))?;
        generation.next().transpose()?;
        generation.next().transpose()?;
        for _ in 0..2 {
            generation.next().transpose()?;
        }
        let before = cache_evidence(generation.cache()?)?;
        assert!(
            matches!(generation.next(), Some(Err(GenerationError::Inference(InferenceError::Exception(source)))) if source.what() == "injected forward failure")
        );
        assert_eq!(generation.accepted, 2);
        assert_eq!(cache_evidence(generation.cache()?)?, before);
        assert!(generation.next().is_none());
        Ok(())
    })
}

struct FailedSubmissionDecoder {
    inner: Box<dyn DecoderModel>,
    calls: usize,
}
impl DecoderModel for FailedSubmissionDecoder {
    fn config(&self) -> &Config {
        self.inner.config()
    }
    fn cache_layout(&self) -> &[LayerCacheSpec] {
        self.inner.cache_layout()
    }
    fn forward(
        &mut self,
        tokens: &Array,
        cache: &mut CacheStep<'_>,
    ) -> Result<Array, InferenceError> {
        self.calls += 1;
        if self.calls != 3 {
            return self.inner.forward(tokens, cache);
        }
        let bad = failing_metal_value()?;
        let mut dependency = Array::from(0.0f32);
        for layer in 0..self.cache_layout().len() {
            let values = tokens.as_dtype(Dtype::Float32)?.reshape(&[1, 1, -1, 1])?;
            let values = if layer == 1 {
                values.add(&bad)?
            } else {
                values
            };
            let (keys, values) = cache.update_and_fetch(layer, values.clone(), values)?;
            dependency = dependency.add(keys.sum(false)?)?.add(values.sum(false)?)?;
        }
        Ok(mlx_rs::ops::zeros_dtype(
            &[
                1,
                tokens.dim(1),
                self.config().dimensions.vocabulary_size as i32,
            ],
            Dtype::Float32,
        )?
        .add(dependency)?)
    }
    fn weight_projection(
        &mut self,
    ) -> Result<mlx_rs::utils::StateProjection<'_>, crate::WeightError> {
        self.inner.weight_projection()
    }
}

#[test]
fn pipeline_submission_failure_after_publication_invalidates_without_replay() -> TestResult {
    with_device(Device::gpu(), || {
        let mut model = scripted(vec![12], None, None, None)?;
        model.decoder = Box::new(FailedSubmissionDecoder {
            inner: model.decoder,
            calls: 0,
        });
        let mut cache = model.new_cache(Default::default())?;
        let mut generation = model.generate_with_cache(
            Prompt::Tokens(&[20.into()]),
            length_options(5),
            &mut cache,
        )?;
        generation.next().transpose()?;
        generation.next().transpose()?;
        generation.next().transpose()?;
        let snapshot = generation.snapshot()?;
        generation.next().transpose()?;
        assert_eq!(generation.accepted, 2);
        assert!(
            matches!(generation.next(), Some(Err(GenerationError::Exception(source))) if source.what().contains("pipeline_failed_evaluation"))
        );
        assert_eq!(generation.accepted, 2);
        assert!(matches!(generation.cache(), Err(CacheError::Invalidated)));
        assert!(generation.next().is_none());
        drop(generation);
        cache.reset()?;
        cache.restore(snapshot)?;
        assert_scripted_prefix(&cache, &[20.into()])?;
        cache.reset()?;
        let mut generation = model.generate_with_cache(
            Prompt::Tokens(&[21.into()]),
            length_options(2),
            &mut cache,
        )?;
        collect_tokens(&mut generation)?;
        Ok(())
    })
}

#[test]
fn pipeline_discarded_submission_error_invalidates_the_cache() -> TestResult {
    with_device(Device::gpu(), || {
        for ending in ["token", "string", "drop", "sampling"] {
            let mut model = scripted(
                vec![12, 13, 14],
                None,
                (ending == "sampling").then_some(1),
                None,
            )?;
            model.decoder = Box::new(FailedSubmissionDecoder {
                inner: model.decoder,
                calls: 0,
            });
            let mut options = length_options(5);
            if ending == "token" {
                options.stop.tokens = StopTokenPolicy::Exact(vec![13.into()]);
            }
            if ending == "string" {
                options.stop.stop_strings = vec!["hello world".into()];
            }
            let mut cache = model.new_cache(Default::default())?;
            let mut generation =
                model.generate_with_cache(Prompt::Tokens(&[20.into()]), options, &mut cache)?;
            generation.next().transpose()?;
            generation.next().transpose()?;
            generation.next().transpose()?;
            if ending == "sampling" {
                assert!(matches!(
                    generation.next(),
                    Some(Err(GenerationError::Sampling(SamplingError::EmptySupport)))
                ));
            } else {
                generation.next().transpose()?;
            }
            assert!(matches!(
                generation.snapshot(),
                Err(CacheError::Invalidated)
            ));
            drop(generation);
            assert!(matches!(cache.tokens(), Err(CacheError::Invalidated)));
            cache.reset()?;
            collect_tokens(&mut model.generate_with_cache(
                Prompt::Tokens(&[25.into()]),
                length_options(2),
                &mut cache,
            )?)?;
        }
        Ok(())
    })
}

#[test]
fn pipeline_keeps_its_gpu_stream_when_the_caller_changes_scopes() -> TestResult {
    let stream = mlx_rs::Stream::new_with_device(&Device::gpu());
    let mut model = mlx_rs::with_stream(&stream, || Model::from_dir(fixture("qwen3-base")))?;
    let mut options = length_options(8);
    options.prefill_chunk_size = NonZeroUsize::MIN;
    let prompt = [12.into(), 13.into(), 14.into()];
    let mut control = mlx_rs::with_stream(&stream, || {
        model.generate(Prompt::Tokens(&prompt), options.clone())
    })?;
    let expected =
        mlx_rs::with_stream(&stream, || control.by_ref().collect::<Result<Vec<_>, _>>())?;
    drop(control);
    let streams = Rc::new(std::cell::RefCell::new(Vec::new()));
    model.decoder = Box::new(CountedDecoder {
        inner: model.decoder,
        calls: Rc::new(std::cell::Cell::new(0)),
        streams: Some(streams.clone()),
    });
    let mut generation =
        mlx_rs::with_stream(&stream, || model.generate(Prompt::Tokens(&prompt), options))?;
    assert_eq!(generation.pipeline_stream.as_ref(), Some(&stream));
    let actual = with_device(Device::cpu(), || {
        generation.by_ref().collect::<Result<Vec<_>, _>>()
    })?;
    assert_eq!(actual, expected);
    let streams = streams.borrow();
    assert_eq!(streams.len(), 10);
    for observed in streams.iter() {
        assert_eq!(observed, &stream);
        assert!(matches!(
            observed.device()?.get_type()?,
            mlx_rs::DeviceType::Gpu
        ));
    }
    Ok(())
}

#[test]
fn pipeline_failure_leaves_an_existing_snapshot_and_sibling_valid() -> TestResult {
    with_device(Device::gpu(), || {
        let mut model = scripted(vec![12], None, None, None)?;
        model.decoder = Box::new(FailedSubmissionDecoder {
            inner: model.decoder,
            calls: 0,
        });
        let mut other = model.new_cache(Default::default())?;
        let mut generation = model.generate(Prompt::Tokens(&[20.into()]), length_options(5))?;
        for _ in 0..3 {
            generation.next().transpose()?;
        }
        let saved = generation.snapshot()?;
        other.restore(generation.snapshot()?)?;
        generation.next().transpose()?;
        assert!(matches!(generation.cache(), Err(CacheError::Invalidated)));
        assert_scripted_prefix(&other, &[20.into()])?;
        assert!(matches!(
            generation.next(),
            Some(Err(GenerationError::Exception(_)))
        ));
        assert_scripted_prefix(&other, &[20.into()])?;
        other.restore(saved)?;
        assert_scripted_prefix(&other, &[20.into()])?;
        Ok(())
    })
}

#[test]
fn pipeline_first_sampling_failure_leaves_untouched_cache_valid() -> TestResult {
    with_device(Device::gpu(), || {
        for count in [1, 5] {
            let mut model = scripted(vec![12], None, None, None)?;
            let mut generation =
                model.generate(Prompt::Tokens(&[20.into()]), length_options(count))?;
            generation.next().transpose()?;
            generation.next().transpose()?;
            let snapshot = generation.snapshot()?;
            generation.state = GenerationState::FirstSample(mlx_rs::ops::broadcast_to(
                failing_metal_value()?,
                &[1, 64],
            )?);
            assert!(matches!(
                generation.next(),
                Some(Err(GenerationError::Exception(_)))
            ));
            assert_scripted_prefix(generation.cache()?, &[20.into()])?;
            generation.cache.get_mut().restore(snapshot)?;
            assert_scripted_prefix(generation.cache()?, &[20.into()])?;
        }
        Ok(())
    })
}

fn failing_metal_value() -> Result<Array, mlx_rs::error::Exception> {
    use mlx_rs::fast::{
        MetalKernel, MetalKernelApplyOptions, MetalKernelOptions, MetalKernelOutput,
    };
    let kernel = MetalKernel::new(
        "pipeline_failed_evaluation",
        &[],
        &["out"],
        "\n#error pipeline_failed_evaluation\nout[0] = 0;",
        MetalKernelOptions::default(),
    )?;
    // SAFETY: Compilation fails; the sole thread otherwise initializes the scalar output.
    let mut outputs = unsafe {
        kernel.apply(
            &[] as &[Array],
            &[MetalKernelOutput {
                shape: vec![1],
                dtype: Dtype::Float32,
            }],
            (1, 1, 1),
            (1, 1, 1),
            MetalKernelApplyOptions::default(),
        )?
    };
    Ok(outputs.remove(0))
}
