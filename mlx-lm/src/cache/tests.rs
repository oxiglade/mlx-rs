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
    cache.info().next().ok_or_else(|| invalid("no test layers"))
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
    assert_eq!(first_info(&cache)?.capacity, 2);
    {
        let mut step = cache.step()?;
        for layer in 0..2 {
            step.update_and_fetch(layer, tokens(&[3.])?, tokens(&[-3.])?)?;
        }
        let output = tokens(&[99.])?;
        let error = step.commit_with(&[&output], |arrays| {
            assert_eq!(arrays.len(), 5);
            assert_eq!(
                arrays.last().map(|array| array.shape()),
                Some(&[1, 1, 1, 1][..])
            );
            Err(invalid("injected evaluation failure"))
        });
        assert!(matches!(error, Err(CacheError::InvalidState(_))));
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
    wrong_arch.architecture = crate::ModelType::new("qwen3");
    assert!(matches!(
        cache.restore(wrong_arch.snapshot()?),
        Err(CacheError::FingerprintMismatch)
    ));
    let mut wrong_count = make_cache(CachePolicy::Full, 1, 2)?;
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
        cache.restore(wrong_kind.snapshot()?),
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
        wrong_kind.restore(wrong_capacity.snapshot()?),
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
