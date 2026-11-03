use super::*;
use safetensors::SafeTensors;
use std::{fs, path::Path};

type TestResult = Result<(), CacheError>;

fn spec() -> LayerCacheSpec {
    LayerCacheSpec {
        attention: crate::AttentionKind::Full,
        batch_size: 1,
        kv_heads: 1,
        head_dim: 1,
        dtype: Dtype::Float32,
    }
}
fn nonzero(n: usize) -> Result<NonZeroUsize, CacheError> {
    NonZeroUsize::new(n).ok_or_else(|| invalid("test expected nonzero value"))
}
fn make_cache(policy: CachePolicy, layers: usize, capacity: usize) -> Result<Cache, CacheError> {
    Cache::new(
        crate::ModelType::new("llama"),
        &vec![spec(); layers],
        &CacheOptions { policy },
        Some(nonzero(capacity)?),
    )
}
fn tensor(data: &[f32], shape: &[usize]) -> Result<Array, CacheError> {
    let dimensions: Vec<i32> = shape
        .iter()
        .map(|&n| dimension(n))
        .collect::<Result<_, _>>()?;
    let count = dimensions
        .iter()
        .try_fold(1i32, |count, &n| count.checked_mul(n))
        .ok_or_else(|| invalid("test tensor size overflow"))?;
    if count as usize != data.len() {
        return Err(invalid("test tensor shape mismatch"));
    }
    Ok(Array::from_slice(data, &dimensions))
}
fn tokens(data: &[f32]) -> Result<Array, CacheError> {
    tensor(data, &[1, 1, data.len(), 1])
}
fn contents(array: &Array) -> Result<Vec<f32>, CacheError> {
    let array = array.contiguous()?;
    array
        .to_vec_exact::<f32>()
        .map_err(|error| invalid(&error.to_string()))
}
fn append(cache: &mut Cache, data: &[f32]) -> Result<(Array, Array), CacheError> {
    let mut step = cache.step()?;
    let mut result = None;
    for layer in 0..step.staged.len() {
        result = Some(step.update_and_fetch(
            layer,
            tokens(data)?,
            tokens(&data.iter().map(|n| -n).collect::<Vec<_>>())?,
        )?);
    }
    step.commit()?;
    result.ok_or_else(|| invalid("no test layers"))
}
fn first_info(cache: &Cache) -> Result<CacheInfo, CacheError> {
    cache
        .info()?
        .next()
        .ok_or_else(|| invalid("no test layers"))
}

fn exact_contents(array: &Array) -> Result<Vec<f32>, CacheError> {
    array
        .to_vec_exact::<f32>()
        .map_err(|error| invalid(&error.to_string()))
}

#[test]
fn logical_full_layers_after_appends() -> TestResult {
    let mut cache = make_cache(CachePolicy::Full, 2, 2)?;
    for view in cache.logical_layers()? {
        assert_eq!(view.positions, 0..0);
        assert_eq!(view.keys.shape(), [1, 1, 0, 1]);
        assert_eq!(view.values.shape(), [1, 1, 0, 1]);
    }
    append(&mut cache, &[0., 1.])?;
    append(&mut cache, &[2., 3., 4.])?;
    let views = cache.logical_layers()?;
    assert_eq!(views.len(), 2);
    for (layer, view) in views.iter().enumerate() {
        assert_eq!(view.layer, layer);
        assert_eq!(view.positions, 0..5);
        assert_eq!(view.keys.shape(), [1, 1, 5, 1]);
        assert_eq!(exact_contents(&view.keys)?, [0., 1., 2., 3., 4.]);
        assert_eq!(exact_contents(&view.values)?, [0., -1., -2., -3., -4.]);
    }
    Ok(())
}

#[test]
fn logical_rotating_layers_after_wrap_and_oversized_prefill() -> TestResult {
    for keep_prefix in [0, 2] {
        let mut cache = make_cache(
            CachePolicy::Rotating {
                capacity: nonzero(5)?,
                keep_prefix,
            },
            1,
            1,
        )?;
        append(&mut cache, &[0., 1., 2., 3., 4., 5., 6., 7.])?;
        for position in 8..16 {
            let views = cache.logical_layers()?;
            assert_eq!(views.len(), if keep_prefix == 0 { 1 } else { 2 });
            if keep_prefix > 0 {
                assert_eq!(views[0].positions, 0..2);
            }
            let tail = views.last().ok_or_else(|| invalid("missing tail"))?;
            let start = if position == 8 {
                keep_prefix
            } else {
                position - (5 - keep_prefix)
            };
            assert_eq!(tail.positions, start..position);
            for view in views {
                assert_eq!(view.layer, 0);
                assert_eq!(view.keys.shape(), [1, 1, view.positions.len() as i32, 1]);
                let expected: Vec<_> = view.positions.map(|n| n as f32).collect();
                assert_eq!(exact_contents(&view.keys)?, expected);
                assert_eq!(
                    exact_contents(&view.values)?,
                    expected.iter().map(|n| -n).collect::<Vec<_>>()
                );
            }
            append(&mut cache, &[position as f32])?;
        }
    }
    Ok(())
}

#[test]
fn pure_policy_validation() -> TestResult {
    let mut layout = spec();
    assert_eq!(resolve_policy(&layout, &CachePolicy::ModelDefault)?, None);
    layout.attention = crate::AttentionKind::Sliding {
        window: nonzero(5)?,
    };
    assert_eq!(
        resolve_policy(&layout, &CachePolicy::ModelDefault)?,
        Some((5, 0))
    );
    assert_eq!(resolve_policy(&layout, &CachePolicy::Full)?, None);
    assert!(matches!(
        resolve_policy(
            &layout,
            &CachePolicy::Rotating {
                capacity: nonzero(4)?,
                keep_prefix: 4
            }
        ),
        Err(CacheError::UnsupportedPolicy(_))
    ));
    assert!(matches!(
        validate_rotation(0, 0),
        Err(CacheError::UnsupportedPolicy(_))
    ));
    layout.batch_size = 0;
    assert!(matches!(
        resolve_policy(&layout, &CachePolicy::Full),
        Err(CacheError::InvalidState(_))
    ));
    assert!(matches!(
        Cache::new(
            crate::ModelType::new("llama"),
            &[],
            &CacheOptions::default(),
            None
        ),
        Err(CacheError::InvalidState(_))
    ));
    Ok(())
}
#[test]
fn pure_growth_boundaries() -> TestResult {
    assert_eq!(grown_capacity(4, 4)?, 4);
    assert_eq!(grown_capacity(4, 5)?, 8);
    assert_eq!(grown_capacity(4, 17)?, 32);
    assert_eq!(grown_capacity(0, 1)?, 1);
    assert!(matches!(
        grown_capacity(4, usize::MAX),
        Err(CacheError::InvalidState(_))
    ));
    assert!(matches!(
        shape(&spec(), usize::MAX),
        Err(CacheError::InvalidState(_))
    ));
    Ok(())
}
#[test]
fn full_append_and_growth() -> TestResult {
    let mut cache = make_cache(CachePolicy::Full, 1, 2)?;
    let (keys, values) = append(&mut cache, &[0., 1.])?;
    assert_eq!(contents(&keys)?, [0., 1.]);
    assert_eq!(contents(&values)?, [0., -1.]);
    assert_eq!(first_info(&cache)?.capacity, 2);
    append(&mut cache, &[2.])?;
    assert_eq!(first_info(&cache)?.capacity, 4);
    let (keys, _) = append(&mut cache, &[3., 4., 5., 6., 7., 8.])?;
    assert_eq!(contents(&keys)?, [0., 1., 2., 3., 4., 5., 6., 7., 8.]);
    let info = first_info(&cache)?;
    assert_eq!(info.capacity, 16);
    assert_eq!(info.processed_tokens, 9);
    assert_eq!(info.retained_positions, 0..9);
    Ok(())
}
#[test]
fn rotating_wrap_prefix_and_chunk_after_wrap() -> TestResult {
    let mut cache = make_cache(
        CachePolicy::Rotating {
            capacity: nonzero(5)?,
            keep_prefix: 2,
        },
        1,
        1,
    )?;
    for n in 0..11 {
        let (keys, values) = append(&mut cache, &[n as f32])?;
        let expected: Vec<f32> = if n < 5 {
            (0..=n).map(|i| i as f32).collect()
        } else {
            [0., 1.]
                .into_iter()
                .chain((n - 2..=n).map(|i| i as f32))
                .collect()
        };
        assert_eq!(contents(&keys)?, expected);
        assert_eq!(
            contents(&values)?,
            expected.iter().map(|n| -n).collect::<Vec<_>>()
        );
    }
    let info = first_info(&cache)?;
    assert_eq!(info.retained_prefix, 0..2);
    assert_eq!(info.retained_positions, 8..11);
    let (keys, _) = append(&mut cache, &[11., 12., 13.])?;
    assert_eq!(contents(&keys)?, [0., 1., 9., 10., 11., 12., 13.]);
    assert_eq!(first_info(&cache)?.retained_positions, 9..14);
    let (keys, _) = append(&mut cache, &[14.])?;
    assert_eq!(contents(&keys)?, [0., 1., 12., 13., 14.]);
    assert_eq!(first_info(&cache)?.retained_positions, 12..15);
    Ok(())
}
#[test]
fn rotating_prefill_over_capacity_and_capacity_one() -> TestResult {
    for keep in [0, 2] {
        let mut cache = make_cache(
            CachePolicy::Rotating {
                capacity: nonzero(5)?,
                keep_prefix: keep,
            },
            1,
            1,
        )?;
        let (keys, _) = append(&mut cache, &[0., 1., 2., 3., 4., 5., 6., 7.])?;
        assert_eq!(contents(&keys)?, [0., 1., 2., 3., 4., 5., 6., 7.]);
        assert_eq!(first_info(&cache)?.retained_positions, keep..8);
        let (keys, _) = append(&mut cache, &[8.])?;
        assert_eq!(
            contents(&keys)?,
            if keep == 0 {
                vec![4., 5., 6., 7., 8.]
            } else {
                vec![0., 1., 6., 7., 8.]
            }
        );
    }
    let mut cache = make_cache(
        CachePolicy::Rotating {
            capacity: nonzero(1)?,
            keep_prefix: 0,
        },
        1,
        1,
    )?;
    append(&mut cache, &[0., 1., 2.])?;
    let (keys, _) = append(&mut cache, &[3.])?;
    assert_eq!(contents(&keys)?, [3.]);
    let (keys, _) = append(&mut cache, &[4., 5.])?;
    assert_eq!(contents(&keys)?, [4., 5.]);
    Ok(())
}
#[test]
fn rotating_trim_before_capacity_and_restore_after_wrap() -> TestResult {
    let mut storage = rotating::RotatingCache::new(&spec(), 5, 1)?;
    storage.update_and_fetch(tokens(&[0., 1., 2.])?, tokens(&[0., -1., -2.])?)?;
    assert_eq!(storage.trim(2)?, 2);
    let (keys, _) = storage.update_and_fetch(tokens(&[3.])?, tokens(&[-3.])?)?;
    assert_eq!(contents(&keys)?, [0., 3.]);
    assert_eq!(storage.trim(99)?, 2);
    assert_eq!(storage.info(0).processed_tokens, 0);
    storage.update_and_fetch(
        tokens(&[0., 1., 2., 3., 4.])?,
        tokens(&[0., -1., -2., -3., -4.])?,
    )?;
    assert!(matches!(storage.trim(1), Err(CacheError::InvalidState(_))));

    let mut cache = make_cache(
        CachePolicy::Rotating {
            capacity: nonzero(5)?,
            keep_prefix: 1,
        },
        1,
        1,
    )?;
    append(&mut cache, &[0., 1., 2., 3., 4.])?;
    append(&mut cache, &[5.])?;
    let snapshot = cache.snapshot()?;
    append(&mut cache, &[6., 7., 8.])?;
    cache.restore(snapshot)?;
    let (keys, _) = append(&mut cache, &[9.])?;
    assert_eq!(contents(&keys)?, [0., 3., 4., 5., 9.]);
    assert_eq!(first_info(&cache)?.retained_positions, 3..7);
    Ok(())
}

#[test]
fn transaction_rollback_and_snapshot_restore() -> TestResult {
    let mut cache = make_cache(CachePolicy::Full, 2, 2)?;
    append(&mut cache, &[1., 2.])?;
    let snapshot = cache.snapshot()?;
    {
        let mut step = cache.step()?;
        step.update_and_fetch(0, tokens(&[3.])?, tokens(&[-3.])?)?;
    }
    assert_eq!(first_info(&cache)?.processed_tokens, 2);
    {
        let mut step = cache.step()?;
        step.update_and_fetch(0, tokens(&[3.])?, tokens(&[-3.])?)?;
        let error = step.update_and_fetch(1, tokens(&[3.])?, tokens(&[-3., -4.])?);
        assert!(matches!(
            error,
            Err(CacheError::ShapeMismatch { layer: 1, .. })
        ));
        assert!(matches!(step.commit(), Err(CacheError::InvalidState(_))));
    }
    assert_eq!(first_info(&cache)?.processed_tokens, 2);
    {
        let mut step = cache.step()?;
        for layer in 0..2 {
            step.update_and_fetch(layer, tokens(&[3.])?, tokens(&[-3.])?)?;
        }
        let output = tokens(&[99.])?;
        drop(step.evaluate(&[&output])?);
    }
    assert_eq!(first_info(&cache)?.processed_tokens, 2);
    let (keys, _) = append(&mut cache, &[4.])?;
    assert_eq!(contents(&keys)?, [1., 2., 4.]);
    cache.restore(snapshot)?;
    assert_eq!(first_info(&cache)?.processed_tokens, 2);
    let (keys, _) = append(&mut cache, &[5.])?;
    assert_eq!(contents(&keys)?, [1., 2., 5.]);
    Ok(())
}
#[test]
fn invalid_steps_and_snapshot_fingerprints() -> TestResult {
    let mut cache = make_cache(CachePolicy::Full, 2, 2)?;
    assert!(matches!(
        cache.step()?.commit(),
        Err(CacheError::InvalidState(_))
    ));
    {
        let mut step = cache.step()?;
        step.update_and_fetch(0, tokens(&[1.])?, tokens(&[1.])?)?;
        assert!(matches!(
            step.update_and_fetch(0, tokens(&[1.])?, tokens(&[1.])?),
            Err(CacheError::InvalidState(_))
        ));
        assert!(step.commit().is_err());
    }
    {
        let mut step = cache.step()?;
        step.update_and_fetch(0, tokens(&[1.])?, tokens(&[1.])?)?;
        step.update_and_fetch(1, tokens(&[1., 2.])?, tokens(&[1., 2.])?)?;
        assert!(matches!(step.commit(), Err(CacheError::InvalidState(_))));
    }
    let mut wrong_arch = make_cache(CachePolicy::Full, 2, 2)?;
    wrong_arch.model_identity = cache.model_identity.clone();
    wrong_arch.architecture = crate::ModelType::new("qwen3");
    assert!(matches!(
        cache.restore(wrong_arch.snapshot()?),
        Err(CacheError::FingerprintMismatch)
    ));
    let mut wrong_count = make_cache(CachePolicy::Full, 1, 2)?;
    wrong_count.model_identity = cache.model_identity.clone();
    assert!(matches!(
        cache.restore(wrong_count.snapshot()?),
        Err(CacheError::FingerprintMismatch)
    ));
    let mut wrong_kind = make_cache(
        CachePolicy::Rotating {
            capacity: nonzero(2)?,
            keep_prefix: 0,
        },
        2,
        2,
    )?;
    assert!(matches!(
        cache.restore({
            wrong_kind.model_identity = cache.model_identity.clone();
            wrong_kind.snapshot()?
        }),
        Err(CacheError::FingerprintMismatch)
    ));
    let mut wrong_capacity = make_cache(
        CachePolicy::Rotating {
            capacity: nonzero(3)?,
            keep_prefix: 0,
        },
        2,
        2,
    )?;
    assert!(matches!(
        wrong_kind.restore({
            wrong_capacity.model_identity = wrong_kind.model_identity.clone();
            wrong_capacity.snapshot()?
        }),
        Err(CacheError::FingerprintMismatch)
    ));
    let mut snapshot = cache.snapshot()?;
    snapshot.layers.remove(&0);
    assert!(matches!(
        cache.restore(snapshot),
        Err(CacheError::FingerprintMismatch)
    ));
    Ok(())
}

fn fixture_array(tensors: &SafeTensors<'_>, name: &str) -> Result<Array, CacheError> {
    let view = tensors
        .tensor(name)
        .map_err(|error| invalid(&error.to_string()))?;
    if view.dtype() != safetensors::Dtype::F32 {
        return Err(invalid("fixture cache must be f32"));
    }
    let (chunks, remainder) = view.data().as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(invalid("invalid f32 bytes"));
    }
    let data: Vec<f32> = chunks
        .iter()
        .map(|&bytes| f32::from_le_bytes(bytes))
        .collect();
    tensor(&data, view.shape())
}
fn equal_array(actual: &Array, expected: &Array) -> TestResult {
    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(actual.dtype(), expected.dtype());
    assert_eq!(contents(actual)?, contents(expected)?);
    Ok(())
}
#[test]
fn committed_fixture_cache_arrays() -> TestResult {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../conformance/mlx-lm/fixtures");
    for name in [
        "llama-base",
        "llama-quant4",
        "llama-sharded",
        "llama-sliding",
        "qwen3-base",
        "qwen3-quant4",
    ] {
        let directory = root.join(name);
        let bytes = fs::read(directory.join("expectations.safetensors"))
            .map_err(|error| invalid(&error.to_string()))?;
        let tensors =
            SafeTensors::deserialize(&bytes).map_err(|error| invalid(&error.to_string()))?;
        let document: serde_json::Value = serde_json::from_slice(
            &fs::read(directory.join("expectations.json"))
                .map_err(|error| invalid(&error.to_string()))?,
        )
        .map_err(|error| invalid(&error.to_string()))?;
        let layer_count = document["config"]["resolved"]["layer_count"]
            .as_u64()
            .ok_or_else(|| invalid("missing layer count"))? as usize;
        for layer in 0..layer_count {
            let pre_keys =
                fixture_array(&tensors, &format!("cache.after_prefill.layer{layer}.keys"))?;
            let pre_values = fixture_array(
                &tensors,
                &format!("cache.after_prefill.layer{layer}.values"),
            )?;
            let final_keys =
                fixture_array(&tensors, &format!("cache.after_decode.layer{layer}.keys"))?;
            let final_values =
                fixture_array(&tensors, &format!("cache.after_decode.layer{layer}.values"))?;
            let sliding =
                document["config"]["resolved"]["attention_kinds"][layer]["sliding"].as_u64();
            let attention = match sliding {
                Some(w) => crate::AttentionKind::Sliding {
                    window: nonzero(w as usize)?,
                },
                None => crate::AttentionKind::Full,
            };
            let dims = pre_keys.shape();
            let layout = LayerCacheSpec {
                attention,
                batch_size: dims[0] as usize,
                kv_heads: dims[1] as usize,
                head_dim: dims[3] as usize,
                dtype: pre_keys.dtype(),
            };
            let prompt = token_length(&pre_keys)?;
            let total = prompt + 8;
            let mut cache = Cache::new(
                crate::ModelType::new(if name.starts_with("llama") {
                    "llama"
                } else {
                    "qwen3"
                }),
                std::slice::from_ref(&layout),
                &CacheOptions::default(),
                Some(nonzero(total)?),
            )?;
            let mut step = cache.step()?;
            let (keys, values) = step.update_and_fetch(0, pre_keys.clone(), pre_values.clone())?;
            step.evaluate_and_commit(&[&keys, &values])?;
            equal_array(&keys, &pre_keys)?;
            equal_array(&values, &pre_values)?;
            assert_eq!(first_info(&cache)?.retained_positions, 0..prompt);
            assert_eq!(first_info(&cache)?.processed_tokens, prompt);
            let views = cache.logical_layers()?;
            assert_eq!(views.len(), 1);
            assert_eq!(views[0].positions, 0..prompt);
            assert_eq!(exact_contents(&views[0].keys)?, contents(&pre_keys)?);
            assert_eq!(exact_contents(&views[0].values)?, contents(&pre_values)?);
            let retained = token_length(&final_keys)?;
            let first_final = total - retained;
            let mut last = None;
            for position in prompt..total {
                let (keys, values) = if position >= first_final {
                    (
                        slice(
                            &final_keys,
                            position - first_final..position - first_final + 1,
                        )?,
                        slice(
                            &final_values,
                            position - first_final..position - first_final + 1,
                        )?,
                    )
                } else {
                    // The fixture omits early decode K/V that are evicted before its final observation.
                    let buffer = Buffer::new(&layout, 1)?;
                    (buffer.keys, buffer.values)
                };
                let mut step = cache.step()?;
                last = Some(step.update_and_fetch(0, keys, values)?);
                step.commit()?;
            }
            let (keys, values) = last.ok_or_else(|| invalid("missing decode output"))?;
            equal_array(&keys, &final_keys)?;
            equal_array(&values, &final_values)?;
            let views = cache.logical_layers()?;
            assert_eq!(views.len(), 1);
            let view = &views[0];
            assert_eq!(view.layer, 0);
            assert_eq!(view.positions, first_final..total);
            assert_eq!(view.keys.shape(), final_keys.shape());
            assert_eq!(view.values.shape(), final_values.shape());
            assert_eq!(exact_contents(&view.keys)?, contents(&final_keys)?);
            assert_eq!(exact_contents(&view.values)?, contents(&final_values)?);
            let info = first_info(&cache)?;
            assert_eq!(info.processed_tokens, total);
            assert_eq!(info.retained_positions, first_final..total);
            assert_eq!(retained, sliding.map_or(total, |w| total.min(w as usize)));
        }
    }
    Ok(())
}

#[test]
fn pure_exact_prefix_and_length_limits() -> Result<(), crate::GenerationError> {
    let cached = [1.into(), 2.into(), 3.into()];
    for (prompt, matched) in [
        (vec![1.into(), 2.into()], 2),
        (vec![1.into(), 9.into(), 3.into(), 4.into()], 1),
    ] {
        assert!(matches!(
            validate_prefix(&cached, &prompt),
            Err(crate::GenerationError::CachePrefixMismatch { matched: n, cached: 3, prompt: p })
                if n == matched && p == prompt.len()
        ));
    }
    assert!(matches!(
        validate_prefix(&cached, &cached),
        Err(crate::GenerationError::NoUncachedTokens)
    ));
    assert_eq!(
        validate_prefix(&cached, &[1.into(), 2.into(), 3.into(), 4.into()])?,
        3
    );
    assert_eq!(checked_capacity(8, nonzero(4)?)?.get(), 12);
    assert!(matches!(
        checked_capacity(usize::MAX, nonzero(1)?),
        Err(crate::GenerationError::LengthOverflow)
    ));
    assert!(matches!(
        checked_capacity(i32::MAX as usize, nonzero(1)?),
        Err(crate::GenerationError::SequenceTooLong { .. })
    ));
    Ok(())
}

#[test]
fn pure_ledger_shares_until_append_and_reserves_commit() -> TestResult {
    let mut ledger = TokenLedger::default();
    ledger.reserve(128)?;
    let first = ledger.prepare(&[1.into(), 2.into()])?;
    assert!(ledger.tokens().is_empty());
    ledger.commit(first);
    let snapshot = ledger.clone();
    assert!(Rc::ptr_eq(&ledger.ids, &snapshot.ids));
    let append = ledger.prepare(&[3.into()])?;
    assert_eq!(ledger.tokens(), snapshot.tokens());
    ledger.commit(append);
    assert_eq!(snapshot.tokens(), &[1.into(), 2.into()]);
    assert_eq!(ledger.tokens(), &[1.into(), 2.into(), 3.into()]);
    assert!(!Rc::ptr_eq(&ledger.ids, &snapshot.ids));
    let allocation = ledger.ids.as_ptr();
    for token in 4..100u32 {
        let append = ledger.prepare(&[token.into()])?;
        ledger.commit(append);
        assert_eq!(allocation, ledger.ids.as_ptr());
    }
    let before = ledger.tokens().to_vec();
    let append = ledger.prepare(&[100.into()])?;
    drop(append);
    assert_eq!(ledger.tokens(), before);
    Ok(())
}

fn append_ids(cache: &mut Cache, ids: &[u32]) -> TestResult {
    let represented: Vec<_> = ids.iter().copied().map(crate::TokenId::from).collect();
    let data: Vec<_> = ids.iter().map(|&id| id as f32).collect();
    let mut step = cache.step_with_tokens(&represented)?;
    for layer in 0..step.staged.len() {
        step.update_and_fetch(
            layer,
            tokens(&data)?,
            tokens(&data.iter().map(|x| -x).collect::<Vec<_>>())?,
        )?;
    }
    step.evaluate(&[])?.commit();
    Ok(())
}

#[derive(Debug, PartialEq)]
struct CacheState {
    fingerprints: Vec<LayerFingerprint>,
    raw: Vec<(Vec<f32>, Vec<f32>)>,
    temporal: Vec<(Vec<f32>, Vec<f32>)>,
    ids: Vec<crate::TokenId>,
}
fn cache_state(cache: &Cache) -> Result<CacheState, CacheError> {
    Ok(CacheState {
        fingerprints: cache
            .layers
            .iter()
            .map(|layer| layer.fingerprint())
            .collect(),
        raw: cache
            .layers
            .iter()
            .map(|layer| {
                let (keys, values) = layer.arrays()?;
                Ok((contents(&keys)?, contents(&values)?))
            })
            .collect::<Result<_, CacheError>>()?,
        temporal: cache
            .layers
            .iter()
            .map(|layer| {
                let (keys, values) = layer.logical_arrays()?;
                Ok((contents(&keys)?, contents(&values)?))
            })
            .collect::<Result<_, CacheError>>()?,
        ids: cache.tokens()?.to_vec(),
    })
}

fn assert_committed(cache: &Cache, before: &CacheState) -> TestResult {
    let after = cache_state(cache)?;
    assert_eq!(after.temporal, before.temporal);
    assert_eq!(after.ids, before.ids);
    for (after, before) in after.fingerprints.iter().zip(&before.fingerprints) {
        assert_eq!(after.processed_tokens, before.processed_tokens);
        assert_eq!(after.length, before.length);
        assert_eq!(after.index, before.index);
    }
    Ok(())
}

#[test]
fn evaluated_guard_rolls_back_layers_and_ledger() -> TestResult {
    let mut cache = Cache::new_for_model(
        crate::ModelType::new("llama"),
        &[spec(), spec()],
        &CacheOptions::default(),
        Some(nonzero(2)?),
        Rc::new(()),
    )?;
    assert!(matches!(cache.step(), Err(CacheError::InvalidState(_))));
    append_ids(&mut cache, &[1, 2])?;
    let snapshot = cache.snapshot()?;
    let before = cache_state(&cache)?;
    for failure in ["drop", "incomplete", "count", "layer", "evaluated"] {
        let mut step = cache.step_with_tokens(&[3.into()])?;
        step.reserve(nonzero(32)?)?;
        step.update_and_fetch(0, tokens(&[3.])?, tokens(&[-3.])?)?;
        match failure {
            "drop" => drop(step),
            "incomplete" => assert!(matches!(
                step.evaluate(&[]),
                Err(CacheError::InvalidState(_))
            )),
            "layer" => {
                assert!(step
                    .update_and_fetch(1, tokens(&[3.])?, tokens(&[-3., -4.])?)
                    .is_err());
                assert!(matches!(
                    step.evaluate(&[]),
                    Err(CacheError::InvalidState(_))
                ));
            }
            "count" => {
                step.update_and_fetch(1, tokens(&[3., 4.])?, tokens(&[-3., -4.])?)?;
                assert!(matches!(
                    step.evaluate(&[]),
                    Err(CacheError::InvalidState(_))
                ));
            }
            _ => {
                step.update_and_fetch(1, tokens(&[3.])?, tokens(&[-3.])?)?;
                let output = tokens(&[99.])?;
                drop(step.evaluate(&[&output])?);
            }
        }
        assert_committed(&cache, &before)?;
        assert!(Rc::ptr_eq(&cache.ledger.ids, &snapshot.ledger.ids));
    }
    let mut step = cache.step_with_tokens(&[3.into(), 4.into()])?;
    for layer in 0..2 {
        step.update_and_fetch(layer, tokens(&[3.])?, tokens(&[-3.])?)?;
    }
    assert!(matches!(
        step.evaluate(&[]),
        Err(CacheError::InvalidState(_))
    ));
    assert_committed(&cache, &before)?;
    append_ids(&mut cache, &[5])?;
    assert_eq!(cache.tokens()?, &[1.into(), 2.into(), 5.into()]);
    Ok(())
}

#[test]
fn wrapped_snapshot_restores_exact_state_and_ledger() -> TestResult {
    let options = CacheOptions {
        policy: CachePolicy::Rotating {
            capacity: nonzero(5)?,
            keep_prefix: 2,
        },
    };
    let identity = Rc::new(());
    let mut cache = Cache::new_for_model(
        crate::ModelType::new("llama"),
        &[spec(), spec()],
        &options,
        None,
        identity.clone(),
    )?;
    let mut reference = Cache::new_for_model(
        crate::ModelType::new("llama"),
        &[spec(), spec()],
        &options,
        None,
        identity,
    )?;
    for id in 0..9 {
        append_ids(&mut cache, &[id])?;
        append_ids(&mut reference, &[id])?;
    }
    let before = cache_state(&cache)?;
    let snapshot = cache.snapshot()?;
    assert!(Rc::ptr_eq(&cache.ledger.ids, &snapshot.ledger.ids));
    for id in 9..20 {
        append_ids(&mut cache, &[id])?;
    }
    assert_eq!(snapshot.ledger.tokens(), before.ids);
    cache.restore(snapshot)?;
    assert_eq!(cache_state(&cache)?, before);
    for ids in [&[21][..], &[22, 23, 24][..], &[25][..]] {
        append_ids(&mut cache, ids)?;
        append_ids(&mut reference, ids)?;
        assert_eq!(cache_state(&cache)?, cache_state(&reference)?);
    }
    Ok(())
}

#[test]
fn reuse_rejects_before_mutation_and_reserves_staged_growth() -> Result<(), crate::GenerationError>
{
    let architecture = crate::ModelType::new("llama");
    let identity = Rc::new(());
    let layout = [spec(), spec()];
    let options = CacheOptions::default();
    let mut cache = Cache::new_for_model(
        architecture.clone(),
        &layout,
        &options,
        Some(nonzero(2)?),
        identity.clone(),
    )?;
    append_ids(&mut cache, &[1, 2])?;
    let before = cache_state(&cache)?;
    let prompt = [1.into(), 2.into(), 3.into()];
    assert!(matches!(
        cache.validate_reuse(&Rc::new(()), &architecture, &layout, &options, &[]),
        Err(crate::GenerationError::Cache(
            CacheError::FingerprintMismatch
        ))
    ));
    let wrong = CacheOptions {
        policy: CachePolicy::Rotating {
            capacity: nonzero(5)?,
            keep_prefix: 2,
        },
    };
    assert!(matches!(
        cache.validate_reuse(&identity, &architecture, &layout, &wrong, &[]),
        Err(crate::GenerationError::Cache(CacheError::PolicyMismatch))
    ));
    for ids in [&[1.into()][..], &[9.into(), 2.into(), 3.into()][..]] {
        assert!(matches!(
            cache.validate_reuse(&identity, &architecture, &layout, &options, ids),
            Err(crate::GenerationError::CachePrefixMismatch { .. })
        ));
    }
    assert!(matches!(
        cache.validate_reuse(&identity, &architecture, &layout, &options, &prompt[..2]),
        Err(crate::GenerationError::NoUncachedTokens)
    ));
    assert_eq!(
        cache.validate_reuse(&identity, &architecture, &layout, &options, &prompt)?,
        2
    );
    let mut foreign =
        Cache::new_for_model(architecture.clone(), &layout, &options, None, Rc::new(()))?;
    assert!(matches!(
        cache.restore(foreign.snapshot()?),
        Err(CacheError::FingerprintMismatch)
    ));
    assert_eq!(cache_state(&cache)?, before);
    let ledger = cache.ledger.clone();
    cache.ledger = TokenLedger::default();
    assert!(matches!(
        cache.validate_reuse(&identity, &architecture, &layout, &options, &prompt),
        Err(crate::GenerationError::Cache(CacheError::InvalidState(_)))
    ));
    cache.ledger = ledger;
    let mut step = cache.step_with_tokens(&prompt[2..])?;
    step.reserve(checked_capacity(prompt.len(), nonzero(20)?)?)?;
    assert_eq!(
        step.cache.info()?.map(|i| i.capacity).collect::<Vec<_>>(),
        [2, 2]
    );
    for layer in 0..2 {
        step.update_and_fetch(layer, tokens(&[3.])?, tokens(&[-3.])?)?;
    }
    step.evaluate_and_commit(&[])?;
    assert_eq!(
        cache.info()?.map(|i| i.capacity).collect::<Vec<_>>(),
        [32, 32]
    );
    assert_eq!(cache.tokens()?, prompt);
    let fresh = Cache::new_for_model(
        architecture,
        &layout,
        &options,
        Some(checked_capacity(3, nonzero(20)?)?),
        identity,
    )?;
    assert_eq!(
        fresh.info()?.map(|i| i.capacity).collect::<Vec<_>>(),
        [23, 23]
    );
    Ok(())
}

#[test]
fn fixture_trim_after_wrap_golden() -> TestResult {
    fn number(value: &serde_json::Value) -> Result<usize, CacheError> {
        value
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| invalid("expected fixture integer"))
    }
    fn positions(value: &serde_json::Value) -> Result<Vec<usize>, CacheError> {
        value
            .as_array()
            .ok_or_else(|| invalid("expected fixture positions"))?
            .iter()
            .map(number)
            .collect()
    }

    let directory =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../conformance/mlx-lm/fixtures/llama-sliding");
    let bytes = fs::read(directory.join("expectations.safetensors"))
        .map_err(|error| invalid(&error.to_string()))?;
    let tensors = SafeTensors::deserialize(&bytes).map_err(|error| invalid(&error.to_string()))?;
    let document: serde_json::Value = serde_json::from_slice(
        &fs::read(directory.join("expectations.json"))
            .map_err(|error| invalid(&error.to_string()))?,
    )
    .map_err(|error| invalid(&error.to_string()))?;
    let golden = &document["cache"]["trim_after_wrap"];
    let capacity = number(&golden["capacity"])?;
    let keep = number(&golden["keep"])?;
    let trim_requested = number(&golden["trim_requested"])?;
    assert_eq!(number(&golden["trim_return"])?, 0);
    let check_stage = |storage: &rotating::RotatingCache,
                       stage: &str|
     -> Result<(Array, Array), CacheError> {
        let metadata = &golden[stage];
        let offset = number(&metadata["offset"])?;
        let fingerprint = storage.fingerprint();
        assert_eq!(fingerprint.processed_tokens, offset, "{stage}");
        assert_eq!(
            fingerprint.index,
            number(&metadata["rotation_index"])?,
            "{stage}"
        );
        assert_eq!(metadata["can_trim"].as_bool(), Some(false), "{stage}");
        assert_eq!(
            metadata["can_trim"].as_bool(),
            Some(offset < capacity),
            "{stage}"
        );
        let info = storage.info(0);
        let prefix = positions(&metadata["retained_prefix"])?;
        let tail = positions(&metadata["retained_tail"])?;
        assert_eq!(info.retained_prefix.collect::<Vec<_>>(), prefix, "{stage}");
        assert_eq!(info.retained_positions.collect::<Vec<_>>(), tail, "{stage}");
        assert_eq!(
            prefix.into_iter().chain(tail).collect::<Vec<_>>(),
            positions(&metadata["temporal_positions"])?,
            "{stage}"
        );
        let array =
            |name: &str| fixture_array(&tensors, &format!("cache.trim_after_wrap.{stage}.{name}"));
        let raw = storage.arrays()?;
        equal_array(&raw.0, &array("raw_keys")?)?;
        equal_array(&raw.1, &array("raw_values")?)?;
        let temporal = storage.logical_arrays()?;
        let expected = (array("temporal_keys")?, array("temporal_values")?);
        equal_array(&temporal.0, &expected.0)?;
        equal_array(&temporal.1, &expected.1)?;
        Ok(expected)
    };

    let mut storage = rotating::RotatingCache::new(&spec(), capacity, keep)?;
    for position in 0..number(&golden["before_trim"]["offset"])? {
        let p = position as f32;
        storage.update_and_fetch(tokens(&[p])?, tokens(&[100.0 + 3.0 * p])?)?;
    }
    check_stage(&storage, "before_trim")?;
    assert!(matches!(
        storage.trim(trim_requested),
        Err(CacheError::InvalidState(_))
    ));
    check_stage(&storage, "after_trim")?;
    let p = number(&golden["after_trim"]["offset"])? as f32;
    let actual = storage.update_and_fetch(tokens(&[p])?, tokens(&[100.0 + 3.0 * p])?)?;
    let expected = check_stage(&storage, "after_append")?;
    equal_array(&actual.0, &expected.0)?;
    equal_array(&actual.1, &expected.1)?;
    Ok(())
}

fn forward_ids(
    decoder: &mut dyn crate::arch::DecoderModel,
    cache: &mut Cache,
    ids: &[crate::TokenId],
    capacity: Option<NonZeroUsize>,
) -> Result<Array, Box<dyn std::error::Error>> {
    let raw: Vec<u32> = ids.iter().copied().map(u32::from).collect();
    let input = Array::from_slice(&raw, &[1, dimension(raw.len())?]);
    let mut step = cache.step_with_tokens(ids)?;
    if let Some(capacity) = capacity {
        step.reserve(capacity)?;
    }
    let output = decoder.forward(&input, &mut step)?;
    step.evaluate_and_commit(&[&output])?;
    Ok(output)
}

fn close_array(
    actual: &Array,
    expected: &Array,
    tolerance: &serde_json::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(actual.shape(), expected.shape());
    let atol = tolerance["atol"].as_f64().ok_or("missing atol")?;
    let rtol = tolerance["rtol"].as_f64().ok_or("missing rtol")?;
    for (a, b) in contents(actual)?.into_iter().zip(contents(expected)?) {
        assert!(
            a.is_finite()
                && b.is_finite()
                && f64::from((a - b).abs()) <= atol + rtol * f64::from(b.abs()),
            "{a} differs from {b}"
        );
    }
    Ok(())
}

#[test]
fn fixture_reuse_matches_fresh_forward() -> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../conformance/mlx-lm/fixtures");
    for name in [
        "llama-base",
        "llama-sliding",
        "llama-quant4",
        "llama-sharded",
        "qwen3-base",
        "qwen3-quant4",
    ] {
        let directory = root.join(name);
        let raw = crate::config::RawConfig::from_bytes(&fs::read(directory.join("config.json"))?)?;
        let factory = crate::arch::factory(&raw.model_type)?;
        let manifest = crate::weights::WeightManifest::discover(&directory)?;
        let mut decoder = factory.build(factory.parse_config(&raw)?, &manifest)?;
        let document: serde_json::Value =
            serde_json::from_slice(&fs::read(directory.join("expectations.json"))?)?;
        let ids: Vec<u32> = serde_json::from_value(document["prefill"]["token_ids"].clone())?;
        let prompt: Vec<_> = ids.into_iter().map(crate::TokenId::from).collect();
        assert!(prompt.len() > 3);
        for policy in [
            CachePolicy::ModelDefault,
            CachePolicy::Full,
            CachePolicy::Rotating {
                capacity: nonzero(5)?,
                keep_prefix: 2,
            },
        ] {
            let identity = Rc::new(());
            let options = CacheOptions { policy };
            let capacity = checked_capacity(prompt.len(), nonzero(4)?)?;
            let mut reused = Cache::new_for_model(
                crate::ModelType::new(raw.model_type.clone()),
                decoder.cache_layout(),
                &options,
                Some(nonzero(2)?),
                identity.clone(),
            )?;
            let mut fresh = Cache::new_for_model(
                crate::ModelType::new(raw.model_type.clone()),
                decoder.cache_layout(),
                &options,
                Some(capacity),
                identity.clone(),
            )?;
            forward_ids(decoder.as_mut(), &mut reused, &prompt[..3], None)?;
            let start = reused.validate_reuse(
                &identity,
                &crate::ModelType::new(raw.model_type.clone()),
                decoder.cache_layout(),
                &options,
                &prompt,
            )?;
            assert_eq!(start, 3);
            let actual = forward_ids(
                decoder.as_mut(),
                &mut reused,
                &prompt[start..],
                Some(capacity),
            )?;
            let full = forward_ids(decoder.as_mut(), &mut fresh, &prompt, None)?;
            let expected = full.try_index((.., dimension(start)?.., ..))?;
            close_array(&actual, &expected, &document["tolerances"]["logits"])?;
            assert_eq!(reused.tokens()?, prompt);
            for id in &prompt[..4] {
                let actual = forward_ids(decoder.as_mut(), &mut reused, &[*id], None)?;
                let expected = forward_ids(decoder.as_mut(), &mut fresh, &[*id], None)?;
                close_array(&actual, &expected, &document["tolerances"]["logits"])?;
                assert_eq!(reused.tokens()?, fresh.tokens()?);
                let actual = reused.logical_layers()?;
                let expected = fresh.logical_layers()?;
                assert_eq!(actual.len(), expected.len());
                for (a, b) in actual.iter().zip(&expected) {
                    assert_eq!(a.layer, b.layer);
                    assert_eq!(a.positions, b.positions);
                    close_array(&a.keys, &b.keys, &document["tolerances"]["cache"])?;
                    close_array(&a.values, &b.values, &document["tolerances"]["cache"])?;
                }
            }
        }
    }
    Ok(())
}

#[test]
fn pure_resolved_policy_agreement() -> TestResult {
    let mut layout = spec();
    layout.attention = crate::AttentionKind::Sliding {
        window: nonzero(5)?,
    };
    let mut fingerprint = LayerFingerprint {
        spec: layout.clone(),
        kind: CacheKind::Rotating,
        capacity: 5,
        keep_prefix: 0,
        processed_tokens: 9,
        length: 5,
        index: 4,
        keys_shape: vec![1, 1, 5, 1],
        values_shape: vec![1, 1, 5, 1],
        keys_dtype: Dtype::Float32,
        values_dtype: Dtype::Float32,
    };
    validate_policy(&fingerprint, &layout, &CachePolicy::ModelDefault)?;
    fingerprint.keep_prefix = 2;
    assert!(matches!(
        validate_policy(&fingerprint, &layout, &CachePolicy::ModelDefault),
        Err(CacheError::PolicyMismatch)
    ));
    validate_policy(
        &fingerprint,
        &layout,
        &CachePolicy::Rotating {
            capacity: nonzero(5)?,
            keep_prefix: 2,
        },
    )?;
    assert!(matches!(
        validate_policy(
            &fingerprint,
            &layout,
            &CachePolicy::Rotating {
                capacity: nonzero(6)?,
                keep_prefix: 2
            }
        ),
        Err(CacheError::PolicyMismatch)
    ));
    fingerprint.kind = CacheKind::Full;
    fingerprint.capacity = 256;
    validate_policy(&fingerprint, &layout, &CachePolicy::Full)?;
    assert!(matches!(
        validate_policy(&fingerprint, &layout, &CachePolicy::ModelDefault),
        Err(CacheError::PolicyMismatch)
    ));
    layout.attention = crate::AttentionKind::Full;
    validate_policy(&fingerprint, &layout, &CachePolicy::ModelDefault)?;
    Ok(())
}

#[test]
fn failed_lazy_append_invalidates_cache() -> Result<(), Box<dyn std::error::Error>> {
    mlx_rs::with_device(
        mlx_rs::Device::cpu(),
        || -> Result<(), Box<dyn std::error::Error>> {
            let mut cache = make_cache(CachePolicy::Full, 1, 16)?;
            append(&mut cache, &[1., 2.])?;
            let bad = mlx_rs::linalg::inv(Array::from_slice(&[0.0f32], &[1, 1]))?
                .reshape(&[1, 1, 1, 1])?;
            let mut step = cache.step()?;
            step.update_and_fetch(0, bad, tokens(&[3.])?)?;
            assert!(matches!(step.evaluate(&[]), Err(CacheError::Exception(_))));
            assert_invalidated(&mut cache)?;
            cache.reset()?;
            let (keys, values) = append(&mut cache, &[4.])?;
            assert_eq!(contents(&keys)?, [4.]);
            assert_eq!(contents(&values)?, [-4.]);
            Ok(())
        },
    )
}

#[test]
fn abandoned_steps_preserve_committed_prefix_on_cpu_and_metal() -> TestResult {
    for device in [mlx_rs::Device::cpu(), mlx_rs::Device::gpu()] {
        mlx_rs::with_device(device, || -> TestResult {
            for policy in [
                CachePolicy::Full,
                CachePolicy::Rotating {
                    capacity: nonzero(5)?,
                    keep_prefix: 2,
                },
            ] {
                for count in [1, 2, 5, 16, 17, 32] {
                    let mut cache = make_cache(policy.clone(), 2, 2)?;
                    for index in 0..count {
                        append(&mut cache, &[index as f32])?;
                    }
                    let before = cache_state(&cache)?;
                    for evaluated in [false, true] {
                        let mut step = cache.step()?;
                        for layer in 0..2 {
                            step.update_and_fetch(layer, tokens(&[999.])?, tokens(&[-999.])?)?;
                        }
                        if evaluated {
                            drop(step.evaluate(&[])?);
                        } else {
                            drop(step);
                        }
                        assert_committed(&cache, &before)?;
                    }
                    let (keys, _) = append(&mut cache, &[123.])?;
                    assert_eq!(contents(&keys)?.last(), Some(&123.));
                }
            }
            Ok(())
        })?;
    }
    Ok(())
}

#[test]
fn failed_metal_append_does_not_retain_the_failed_graph() -> Result<(), Box<dyn std::error::Error>>
{
    use mlx_rs::fast::{
        MetalKernel, MetalKernelApplyOptions, MetalKernelOptions, MetalKernelOutput,
    };
    mlx_rs::with_device(
        mlx_rs::Device::gpu(),
        || -> Result<(), Box<dyn std::error::Error>> {
            for abandon_first in [false, true] {
                let mut cache = make_cache(CachePolicy::Full, 1, 32)?;
                for index in 0..17 {
                    append(&mut cache, &[index as f32])?;
                }
                let kernel = MetalKernel::new(
                    "cache_recovery_failure",
                    &[],
                    &["out"],
                    "\n#error injected_cache_failure\nout[0] = 1.0f;",
                    MetalKernelOptions::default(),
                )?;
                // SAFETY: Compilation fails; the body otherwise writes the sole f32 output once.
                let mut outputs = unsafe {
                    kernel.apply(
                        &[] as &[Array],
                        &[MetalKernelOutput {
                            shape: vec![1, 1, 1, 1],
                            dtype: Dtype::Float32,
                        }],
                        (1, 1, 1),
                        (1, 1, 1),
                        MetalKernelApplyOptions::default(),
                    )?
                };
                let mut step = cache.step()?;
                step.update_and_fetch(0, outputs.remove(0), tokens(&[99.])?)?;
                if abandon_first {
                    drop(step);
                    let snapshot = cache.snapshot()?;
                    let (keys, values) = append(&mut cache, &[123.])?;
                    assert_eq!(contents(&keys)?.last(), Some(&123.));
                    assert_eq!(contents(&values)?.last(), Some(&-123.));
                    cache.reset()?;
                    cache.restore(snapshot)?;
                    assert_eq!(cache.info()?.next().unwrap().processed_tokens, 17);
                } else {
                    assert!(matches!(step.evaluate(&[]), Err(CacheError::Exception(_))));
                    assert_invalidated(&mut cache)?;
                }
                cache.reset()?;
                let (keys, values) = append(&mut cache, &[4.])?;
                assert_eq!(contents(&keys)?, [4.]);
                assert_eq!(contents(&values)?, [-4.]);
            }
            Ok(())
        },
    )
}

fn assert_invalidated(cache: &mut Cache) -> TestResult {
    assert!(matches!(cache.info(), Err(CacheError::Invalidated)));
    assert!(matches!(cache.tokens(), Err(CacheError::Invalidated)));
    assert!(matches!(
        cache.logical_layers(),
        Err(CacheError::Invalidated)
    ));
    assert!(matches!(cache.snapshot(), Err(CacheError::Invalidated)));
    assert!(matches!(cache.step(), Err(CacheError::Invalidated)));
    assert!(matches!(
        cache.step_with_tokens(&[1.into()]),
        Err(CacheError::Invalidated)
    ));
    assert!(matches!(
        cache.validate_reuse(
            &cache.model_identity,
            &cache.architecture,
            &[spec()],
            &CacheOptions::default(),
            &[1.into()]
        ),
        Err(crate::GenerationError::Cache(CacheError::Invalidated))
    ));
    Ok(())
}

#[test]
fn evaluation_failure_invalidates_full_and_rotating_caches() -> TestResult {
    for device in [mlx_rs::Device::cpu(), mlx_rs::Device::gpu()] {
        mlx_rs::with_device(device, || -> TestResult {
            for policy in [
                CachePolicy::Full,
                CachePolicy::Rotating {
                    capacity: nonzero(5)?,
                    keep_prefix: 2,
                },
            ] {
                for after_eval in [false, true] {
                    let mut cache = make_cache(policy.clone(), 2, 2)?;
                    append(&mut cache, &[1., 2., 3., 4., 5.])?;
                    let before = cache_state(&cache)?;
                    let snapshot = cache.snapshot()?;
                    let mut step = cache.step()?;
                    for layer in 0..2 {
                        step.update_and_fetch(layer, tokens(&[99.])?, tokens(&[-99.])?)?;
                    }
                    let error = step
                        .evaluate_with(&[], |arrays| {
                            if after_eval {
                                mlx_rs::transforms::eval(arrays.iter().copied())?;
                            }
                            Err(
                                mlx_rs::error::Exception::from("injected evaluation failure")
                                    .into(),
                            )
                        })
                        .err()
                        .expect("evaluation must fail");
                    assert!(
                        matches!(error, CacheError::Exception(source) if source.what() == "injected evaluation failure")
                    );
                    assert_invalidated(&mut cache)?;
                    cache.reset()?;
                    cache.restore(snapshot)?;
                    assert_committed(&cache, &before)?;
                    cache.reset()?;
                    assert!(cache.tokens()?.is_empty());
                    assert!(cache.info()?.all(|info| info.processed_tokens == 0));
                    let (keys, _) = append(&mut cache, &[6., 7.])?;
                    assert_eq!(contents(&keys)?, [6., 7.]);
                }
            }
            Ok(())
        })?;
    }
    Ok(())
}

#[test]
fn failed_cache_does_not_invalidate_snapshot_siblings() -> TestResult {
    mlx_rs::with_device(mlx_rs::Device::cpu(), || -> TestResult {
        let mut source = make_cache(CachePolicy::Full, 1, 16)?;
        append(&mut source, &[1., 2.])?;
        let saved = source.snapshot()?;
        let mut first = make_cache(CachePolicy::Full, 1, 16)?;
        let mut second = make_cache(CachePolicy::Full, 1, 16)?;
        first.model_identity = source.model_identity.clone();
        second.model_identity = source.model_identity.clone();
        first.restore(source.snapshot()?)?;
        second.restore(source.snapshot()?)?;
        let bad =
            mlx_rs::linalg::inv(Array::from_slice(&[0.0f32], &[1, 1]))?.reshape(&[1, 1, 1, 1])?;
        let mut step = first.step()?;
        step.update_and_fetch(0, bad, tokens(&[3.])?)?;
        assert!(matches!(step.evaluate(&[]), Err(CacheError::Exception(_))));
        assert_invalidated(&mut first)?;
        assert_eq!(contents(&append(&mut second, &[4.])?.0)?, [1., 2., 4.]);
        assert_eq!(contents(&append(&mut source, &[5.])?.0)?, [1., 2., 5.]);
        first.reset()?;
        first.restore(saved)?;
        assert_eq!(contents(&append(&mut first, &[6.])?.0)?, [1., 2., 6.]);
        Ok(())
    })
}

#[test]
fn sampling_construction_failure_drops_unevaluated_cache_graph(
) -> Result<(), Box<dyn std::error::Error>> {
    mlx_rs::with_device(
        mlx_rs::Device::cpu(),
        || -> Result<(), Box<dyn std::error::Error>> {
            let mut cache = make_cache(CachePolicy::Full, 1, 16)?;
            append(&mut cache, &[1., 2.])?;
            let bad = mlx_rs::linalg::inv(Array::from_slice(&[0.0f32], &[1, 1]))?
                .reshape(&[1, 1, 1, 1])?;
            let mut step = cache.step()?;
            step.update_and_fetch(0, bad, tokens(&[3.])?)?;
            let result =
                step.evaluate_outputs::<()>(|_, _| Err(crate::SamplingError::EmptySupport.into()));
            assert!(matches!(
                result,
                Err(crate::GenerationError::Sampling(
                    crate::SamplingError::EmptySupport
                ))
            ));
            drop(result);
            let (keys, _) = append(&mut cache, &[4.])?;
            assert_eq!(contents(&keys)?, [1., 2., 4.]);
            Ok(())
        },
    )
}

#[test]
fn sampling_roots_evaluate_every_decoder_cache_write() -> anyhow::Result<()> {
    mlx_rs::with_device(mlx_rs::Device::gpu(), || -> anyhow::Result<()> {
        for name in [
            "llama-base",
            "llama-sliding",
            "llama-quant4",
            "llama-sharded",
            "qwen3-base",
            "qwen3-quant4",
        ] {
            let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../conformance/mlx-lm/fixtures")
                .join(name);
            let raw =
                crate::config::RawConfig::from_bytes(&fs::read(directory.join("config.json"))?)?;
            let factory = crate::arch::factory(&raw.model_type)?;
            let weights = crate::weights::WeightManifest::discover(&directory)?;
            let mut decoder = factory.build(factory.parse_config(&raw)?, &weights)?;
            for temperature in [0.0, 0.7] {
                for capture in [false, true] {
                    let mut cache = Cache::new_for_model(
                        crate::ModelType::new(raw.model_type.clone()),
                        decoder.cache_layout(),
                        &CacheOptions {
                            policy: CachePolicy::Full,
                        },
                        Some(nonzero(2)?),
                        Rc::new(()),
                    )?;
                    let prompt = [12.into(), 13.into()];
                    let mut step = cache.step_with_tokens(&prompt)?;
                    let logits =
                        decoder.forward(&Array::from_slice(&[12u32, 13], &[1, 2]), &mut step)?;
                    step.evaluate_and_commit(&[&logits])?;
                    drop(logits);
                    let engine = crate::sampling::SamplingEngine::new(
                        crate::SamplerOptions {
                            temperature,
                            seed: Some(1729),
                            ..Default::default()
                        },
                        decoder.config().dimensions.vocabulary_size,
                        None,
                        None,
                        None,
                    )?;
                    for token in [14u32, 15, 16] {
                        let offset = cache.tokens()?.len();
                        let mut step = cache.speculative_step(offset)?;
                        let logits =
                            decoder.forward(&Array::from_slice(&[token], &[1, 1]), &mut step)?;
                        let row = logits.reshape(&[1, -1])?;
                        let plan = engine.prepare(&[], None, row, None, capture)?;
                        drop(logits);
                        step.submit(&plan.outputs(&[]))?;
                        plan.finish()?;
                        for (index, layer) in cache.layers.iter().enumerate() {
                            let (keys, values) = layer.arrays()?;
                            for (kind, array) in [("K", keys), ("V", values)] {
                                let mut available = false;
                                // SAFETY: Both pointers remain valid for this non-evaluating query.
                                let status = unsafe {
                                    mlx_sys::_mlx_array_is_available(&mut available, array.as_ptr())
                                };
                                assert_eq!(status, 0);
                                assert!(available, "{name} layer {index} {kind} at {offset}, temperature={temperature}, capture={capture}");
                            }
                        }
                        cache.commit_input(token.into())?;
                    }
                }
            }
        }
        Ok(())
    })
}

#[test]
fn append_snapshot_storage_is_independent_on_metal() -> TestResult {
    mlx_rs::with_device(mlx_rs::Device::gpu(), || -> TestResult {
        for capacity in [2, 16, 512] {
            let mut source = make_cache(CachePolicy::Full, 1, capacity)?;
            append(&mut source, &[1., 2.])?;
            let before = source.layers[0].arrays()?.0.to_vec_exact::<f32>().unwrap();
            let saved = source.snapshot()?;
            let source_ptr = source.layers[0].arrays()?.0.as_slice::<f32>().as_ptr();
            let saved_ptr = saved.layers[&0].arrays()?.0.as_slice::<f32>().as_ptr();
            if capacity > 2 {
                assert_ne!(source_ptr, saved_ptr);
            }
            let mut sibling = make_cache(CachePolicy::Full, 1, capacity)?;
            sibling.model_identity = source.model_identity.clone();
            sibling.restore(saved)?;
            assert_eq!(
                sibling.layers[0].arrays()?.0.as_slice::<f32>().as_ptr(),
                saved_ptr
            );
            for i in 0..32 {
                append(&mut source, &[10. + i as f32])?;
                append(&mut sibling, &[-10. - i as f32])?;
            }
            let left = contents(&source.layers[0].logical_arrays()?.0)?;
            let right = contents(&sibling.layers[0].logical_arrays()?.0)?;
            assert_eq!(&left[..2], &before[..2]);
            assert_eq!(&right[..2], &before[..2]);
            assert_eq!(left[33], 41.);
            assert_eq!(right[33], -41.);
        }
        Ok(())
    })
}

#[test]
fn rolled_back_gpu_suffix_can_be_reused_with_pending_readers(
) -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("MLX_LM_ROLLBACK_CHILD").is_err() {
        let status = std::process::Command::new(std::env::current_exe()?)
            .args([
                "--exact",
                "cache::tests::rolled_back_gpu_suffix_can_be_reused_with_pending_readers",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("MLX_LM_ROLLBACK_CHILD", "1")
            .env("MLX_MAX_MB_PER_BUFFER", "16")
            .status()?;
        assert!(status.success());
        return Ok(());
    }
    let first = mlx_rs::Stream::new_with_device(&mlx_rs::Device::gpu());
    let second = mlx_rs::Stream::new_with_device(&mlx_rs::Device::gpu());
    for reuse in [&first, &second] {
        for _ in 0..4 {
            let (mut cache, reader) =
                mlx_rs::with_stream(&first, || -> Result<_, Box<dyn std::error::Error>> {
                    let mut cache = make_cache(CachePolicy::Full, 1, 16)?;
                    append(&mut cache, &[1., 2.])?;
                    let mut step = cache.step()?;
                    let (keys, _) = step.update_and_fetch(0, tokens(&[9.])?, tokens(&[-9.])?)?;
                    step.submit(&[&keys])?;
                    // A separate evaluation prevents the append graph's fences from ordering reuse.
                    let matrix = mlx_rs::ops::ones::<f32>(&[2048, 2048])?;
                    let mut slow = matrix.clone();
                    for _ in 0..8 {
                        slow = slow.matmul(&matrix)?.divide(Array::from(2048.0f32))?;
                    }
                    let suffix = keys.try_index((0, 0, 2, 0))?;
                    let reader = slow.add(&suffix)?.sum(false)?;
                    mlx_rs::transforms::async_eval([&reader])?;
                    let mut available = false;
                    assert_eq!(
                        unsafe {
                            mlx_sys::_mlx_array_is_available(&mut available, reader.as_ptr())
                        },
                        0
                    );
                    assert!(
                        !available,
                        "reader must still be running before suffix reuse"
                    );
                    Ok((cache, reader))
                })?;
            mlx_rs::with_stream(reuse, || -> TestResult {
                let (keys, _) = append(&mut cache, &[17.])?;
                assert_eq!(contents(&keys)?, [1., 2., 17.]);
                Ok(())
            })?;
            assert_eq!(reader.try_item_exact::<f32>()?, 10. * 2048. * 2048.);
        }
    }
    Ok(())
}
