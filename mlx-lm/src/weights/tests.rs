use super::*;
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    SafeTensors,
};
use std::{fs, num::NonZeroUsize};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../conformance/mlx-lm/fixtures")
        .join(name)
}

fn copy_fixture(name: &str) -> Result<tempfile::TempDir, WeightError> {
    let target = tempfile::tempdir()?;
    for entry in fs::read_dir(fixture(name))? {
        let entry = entry?;
        fs::copy(entry.path(), target.path().join(entry.file_name()))?;
    }
    Ok(target)
}

fn assign(key: &str) -> WeightDisposition {
    WeightDisposition::Parameter(ParameterPath::new(key))
}

fn expected(manifest: &WeightManifest) -> Result<ExpectedSlots, WeightError> {
    let shards: BTreeSet<_> = manifest
        .tensors
        .values()
        .map(|entry| &entry.shard)
        .collect();
    let mut slots = BTreeMap::new();
    for shard in shards {
        for (key, tensor) in read_tensors(shard)? {
            let dtype = tensor_dtype(&key, tensor.dtype)?;
            slots.insert(key, Some((dtype, tensor.shape)));
        }
    }
    Ok(slots)
}

fn assert_fixture_error(name: &str, case: &str, error: WeightError) -> Result<(), WeightError> {
    let expectations: serde_json::Value =
        serde_json::from_slice(&fs::read(fixture(name).join("expectations.json"))?)?;
    let class = match error {
        WeightError::MissingShard(_) => "WeightError::MissingShard",
        WeightError::DuplicateTensor(_) => "WeightError::DuplicateTensor",
        WeightError::ShapeMismatch { .. } => "WeightError::ShapeMismatch",
        other => return Err(other),
    };
    assert_eq!(expectations["errors"][case]["expected_rust_error"], class);
    Ok(())
}

struct TensorData {
    dtype: safetensors::Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

fn read_tensors(path: &Path) -> Result<BTreeMap<String, TensorData>, WeightError> {
    let bytes = fs::read(path)?;
    Ok(SafeTensors::deserialize(&bytes)?
        .tensors()
        .into_iter()
        .map(|(name, tensor)| {
            (
                name,
                TensorData {
                    dtype: tensor.dtype(),
                    shape: tensor.shape().to_vec(),
                    data: tensor.data().to_vec(),
                },
            )
        })
        .collect())
}

fn write_tensors(path: &Path, tensors: &BTreeMap<String, TensorData>) -> Result<(), WeightError> {
    let views = tensors
        .iter()
        .map(|(name, tensor)| {
            Ok((
                name.as_str(),
                TensorView::new(tensor.dtype, tensor.shape.clone(), &tensor.data)?,
            ))
        })
        .collect::<Result<Vec<_>, WeightError>>()?;
    serialize_to_file(views, None, path)?;
    Ok(())
}

mod pure {
    use super::*;

    #[test]
    fn fixture_manifests() -> Result<(), WeightError> {
        for (name, count, shard_count) in [
            ("llama-base", 21, 1),
            ("llama-sharded", 21, 2),
            ("llama-quant4", 53, 1),
        ] {
            let manifest = WeightManifest::discover(&fixture(name))?;
            assert_eq!(manifest.tensors.len(), count);
            let shards: BTreeSet<_> = manifest
                .tensors
                .values()
                .map(|entry| &entry.shard)
                .collect();
            assert_eq!(shards.len(), shard_count);
            for shard in shards {
                for (key, tensor) in read_tensors(shard)? {
                    let entry = manifest.entry(&key)?;
                    assert_eq!(entry.shard, *shard);
                    assert_eq!(entry.dtype, tensor_dtype(&key, tensor.dtype)?);
                    assert_eq!(entry.shape, tensor.shape);
                }
            }
            assert_eq!(manifest.plan(&expected(&manifest)?, assign)?.len(), count);
        }
        let base = WeightManifest::discover(&fixture("llama-base"))?;
        let sharded = WeightManifest::discover(&fixture("llama-sharded"))?;
        assert_eq!(expected(&base)?, expected(&sharded)?);
        assert_eq!(base.entry("model.embed_tokens.weight")?.shape, [64, 64]);
        assert_eq!(base.entry("model.norm.weight")?.shape, [64]);
        Ok(())
    }

    #[test]
    fn affine_manifest_and_configured_layout() -> Result<(), WeightError> {
        let manifest = WeightManifest::discover(&fixture("llama-quant4"))?;
        let raw: serde_json::Value =
            serde_json::from_slice(&fs::read(fixture("llama-quant4").join("config.json"))?)?;
        assert_eq!(raw["quantization"]["bits"], 4);
        assert_eq!(raw["quantization"]["group_size"], 32);
        let config = AffineQuantization {
            group_size: NonZeroUsize::new(32)
                .ok_or_else(|| WeightError::UnsupportedFormat("zero group size".into()))?,
            bits: 4,
        };
        let groups = manifest.affine_groups()?;
        assert_eq!(groups.len(), 16);
        assert!(groups.contains("model.embed_tokens"));
        assert!(groups.contains("lm_head"));
        for prefix in groups {
            manifest.validate_affine_group(&prefix, &config)?;
            let weight = manifest.entry(&format!("{prefix}.weight"))?;
            assert_eq!(
                weight.dtype,
                tensor_dtype(&format!("{prefix}.weight"), safetensors::Dtype::U32)?
            );
            assert_eq!(weight.shape[1], 8);
            let scales = manifest.entry(&format!("{prefix}.scales"))?;
            assert_eq!(
                scales.dtype,
                tensor_dtype(&format!("{prefix}.scales"), safetensors::Dtype::F32)?
            );
            assert_eq!(scales.shape, [weight.shape[0], 2]);
        }
        let incompatible = AffineQuantization { bits: 8, ..config };
        assert!(matches!(
            manifest.validate_affine_group("model.embed_tokens", &incompatible),
            Err(WeightError::ShapeMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn missing_shard_mutation() -> Result<(), WeightError> {
        let copy = copy_fixture("llama-sharded")?;
        fs::remove_file(copy.path().join("model-00002-of-00002.safetensors"))?;
        match WeightManifest::discover(copy.path()) {
            Err(error) => assert_fixture_error("llama-sharded", "missing_shard", error),
            Ok(_) => Err(WeightError::UnsupportedFormat(
                "missing shard accepted".into(),
            )),
        }
    }

    #[test]
    fn duplicate_tensor_mutation() -> Result<(), WeightError> {
        let copy = copy_fixture("llama-sharded")?;
        let first = read_tensors(&copy.path().join("model-00001-of-00002.safetensors"))?;
        let second_path = copy.path().join("model-00002-of-00002.safetensors");
        let mut second = read_tensors(&second_path)?;
        let (name, tensor) = first
            .into_iter()
            .next()
            .ok_or_else(|| WeightError::MissingKey("first fixture tensor".into()))?;
        second.insert(name, tensor);
        write_tensors(&second_path, &second)?;
        match WeightManifest::discover(copy.path()) {
            Err(error) => assert_fixture_error("llama-sharded", "duplicate_tensor", error),
            Ok(_) => Err(WeightError::UnsupportedFormat(
                "duplicate tensor accepted".into(),
            )),
        }
    }

    #[test]
    fn wrong_shape_mutations() -> Result<(), WeightError> {
        for name in ["llama-base", "llama-sharded", "llama-quant4"] {
            let original = WeightManifest::discover(&fixture(name))?;
            let copy = copy_fixture(name)?;
            let copied = WeightManifest::discover(copy.path())?;
            let key = "model.layers.0.self_attn.q_proj.weight";
            let path = &copied.entry(key)?.shard;
            let mut tensors = read_tensors(path)?;
            let tensor = tensors
                .get_mut(key)
                .ok_or_else(|| WeightError::MissingKey(key.into()))?;
            let row_bytes = tensor.data.len() / tensor.shape[0];
            tensor.shape[0] -= 1;
            tensor.data.truncate(tensor.data.len() - row_bytes);
            write_tensors(path, &tensors)?;
            match copied.read_tensor(key) {
                Err(error) => assert_fixture_error(name, "wrong_shape", error)?,
                Ok(_) => {
                    return Err(WeightError::UnsupportedFormat(
                        "changed tensor shape accepted".into(),
                    ))
                }
            }
            let mutated = WeightManifest::discover(copy.path())?;
            match mutated.plan(&expected(&original)?, assign) {
                Err(error) => assert_fixture_error(name, "wrong_shape", error)?,
                Ok(_) => {
                    return Err(WeightError::UnsupportedFormat(
                        "wrong shape accepted".into(),
                    ))
                }
            }
        }
        Ok(())
    }

    #[test]
    fn exact_keys_dtypes_and_approved_ignores() -> Result<(), WeightError> {
        let manifest = WeightManifest::discover(&fixture("llama-base"))?;
        let mut slots = expected(&manifest)?;
        slots.remove("lm_head.weight");
        slots.insert("absent.bias".into(), None);
        assert!(matches!(
            manifest.plan(&slots, assign),
            Err(WeightError::UnexpectedKey(_))
        ));
        let tied = |key: &str| {
            if key == "lm_head.weight" {
                WeightDisposition::Ignore {
                    reason: "tied-output-head",
                }
            } else {
                assign(key)
            }
        };
        assert_eq!(manifest.plan(&slots, tied)?.len(), 20);
        assert!(matches!(
            manifest.plan(&slots, |_| WeightDisposition::Reject),
            Err(WeightError::UnexpectedKey(_))
        ));
        assert!(matches!(
            manifest.plan(&slots, |_| WeightDisposition::Ignore { reason: "" }),
            Err(WeightError::UnexpectedKey(_))
        ));
        slots.insert("missing.weight".into(), Some((Dtype::Float32, vec![64])));
        assert!(matches!(
            manifest.plan(&slots, tied),
            Err(WeightError::MissingKey(_))
        ));
        slots.remove("missing.weight");
        slots.insert("model.norm.weight".into(), Some((Dtype::Float16, vec![64])));
        assert!(matches!(
            manifest.plan(&slots, tied),
            Err(WeightError::DtypeMismatch { .. })
        ));
        assert!(matches!(
            manifest.plan(&expected(&manifest)?, |_| assign("model.norm.weight")),
            Err(WeightError::DuplicateTensor(_))
        ));
        Ok(())
    }

    #[test]
    fn index_conflicts_and_precedence() -> Result<(), WeightError> {
        let copy = copy_fixture("llama-sharded")?;
        let path = copy.path().join("model.safetensors.index.json");
        let original: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
        for (key, shard) in [
            ("lm_head.weight", "model-00002-of-00002.safetensors"),
            ("missing.weight", "model-00001-of-00002.safetensors"),
            ("lm_head.weight", "../model.safetensors"),
        ] {
            let mut index = original.clone();
            index["weight_map"][key] = shard.into();
            fs::write(&path, serde_json::to_vec(&index)?)?;
            assert!(matches!(
                WeightManifest::discover(copy.path()),
                Err(WeightError::ConflictingIndex { .. })
            ));
        }
        fs::write(
            &path,
            br#"{"weight_map":{"a":"one.safetensors","a":"two.safetensors"}}"#,
        )?;
        assert!(matches!(
            WeightManifest::discover(copy.path()),
            Err(WeightError::ConflictingIndex { .. })
        ));
        fs::copy(
            fixture("llama-base").join("model.safetensors"),
            copy.path().join("model.safetensors"),
        )?;
        fs::write(&path, serde_json::to_vec(&original)?)?;
        fs::remove_file(copy.path().join("model-00002-of-00002.safetensors"))?;
        assert!(matches!(
            WeightManifest::discover(copy.path()),
            Err(WeightError::MissingShard(_))
        ));
        Ok(())
    }

    #[test]
    fn unsupported_dtype_and_corrupt_headers() -> Result<(), WeightError> {
        let copy = tempfile::tempdir()?;
        let path = copy.path().join("model.safetensors");
        assert!(matches!(
            WeightManifest::discover(copy.path()),
            Err(WeightError::MissingFile(_))
        ));
        for (header, data, duplicate) in [
            (
                r#"{"a":{"dtype":"F8_E5M2","shape":[1],"data_offsets":[0,1]}}"#,
                vec![0],
                false,
            ),
            (
                r#"{"a":{"dtype":"F64","shape":[1],"data_offsets":[0,8]}}"#,
                vec![0; 8],
                false,
            ),
            (
                r#"{"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#,
                vec![],
                true,
            ),
        ] {
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header.as_bytes());
            bytes.extend(data);
            fs::write(&path, bytes)?;
            let result = WeightManifest::discover(copy.path());
            if duplicate {
                assert!(matches!(result, Err(WeightError::DuplicateTensor(_))));
            } else {
                assert!(matches!(result, Err(WeightError::UnsupportedDtype { .. })));
            }
        }
        let source = fs::read(fixture("llama-base").join("model.safetensors"))?;
        fs::write(&path, &source[..source.len() - 1])?;
        assert!(matches!(
            WeightManifest::discover(copy.path()),
            Err(WeightError::Safetensors(_))
        ));
        Ok(())
    }

    #[test]
    fn tied_packed_head_uses_only_the_embedding_slots() -> Result<(), WeightError> {
        let mut manifest = WeightManifest::discover(&fixture("llama-quant4"))?;
        let mut slots = expected(&manifest)?;
        slots.retain(|key, _| !key.starts_with("lm_head."));
        manifest.tensors.remove("lm_head.scales");
        manifest.tensors.remove("lm_head.biases");
        let assignments = manifest.plan(&slots, |key| {
            if key.starts_with("lm_head.") {
                WeightDisposition::Ignore {
                    reason: "tied-output-head",
                }
            } else {
                assign(key)
            }
        })?;
        assert_eq!(assignments.len(), 50);
        assert!(assignments.contains_key("model.embed_tokens.weight"));
        assert!(!assignments.contains_key("lm_head.weight"));
        Ok(())
    }

    #[test]
    fn incomplete_affine_groups() -> Result<(), WeightError> {
        for suffix in ["weight", "scales", "biases"] {
            let mut manifest = WeightManifest::discover(&fixture("llama-quant4"))?;
            let missing = format!("model.embed_tokens.{suffix}");
            manifest.tensors.remove(&missing);
            assert!(
                matches!(manifest.affine_groups(), Err(WeightError::MissingKey(key)) if key == missing)
            );
        }
        Ok(())
    }
}

mod runtime {
    use super::*;
    use mlx_rs::{
        module::Param,
        nn::{Embedding, Linear, QuantizedEmbedding, QuantizedLinear},
    };

    fn values<T: mlx_rs::ArrayElement + Clone>(array: &Array) -> Result<Vec<T>, WeightError> {
        use mlx_rs::error::AsSliceError;
        array.eval()?;
        array
            .try_as_slice::<T>()
            .map(<[T]>::to_vec)
            .map_err(|error| match error {
                AsSliceError::DtypeMismatch { expecting, found } => WeightError::DtypeMismatch {
                    key: "test observation".into(),
                    expected: expecting,
                    actual: found,
                },
                AsSliceError::Exception(error) => WeightError::Exception(error),
                other => WeightError::UnsupportedFormat(other.to_string()),
            })
    }

    #[test]
    fn projection_round_trip_and_atomic_failure() -> Result<(), WeightError> {
        let copy = tempfile::tempdir()?;
        let path = copy.path().join("model.safetensors");
        let mut tensors = BTreeMap::new();
        for (name, values, shape) in [
            ("external.weight", vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]),
            ("external.bias", vec![5.0_f32, 6.0], vec![2]),
        ] {
            tensors.insert(
                name.into(),
                TensorData {
                    dtype: safetensors::Dtype::F32,
                    shape,
                    data: values.into_iter().flat_map(f32::to_le_bytes).collect(),
                },
            );
        }
        write_tensors(&path, &tensors)?;
        let manifest = WeightManifest::discover(copy.path())?;
        let mut module = Linear {
            weight: Param::new(Array::from_slice(&[0_f32; 4], &[2, 2])),
            bias: Param::new(Some(Array::from_slice(&[0_f32; 2], &[2]))),
        };
        let mut absent = None;
        let mut projection = StateProjection::new();
        projection.optional("linear.bias", &mut module.bias.value)?;
        projection.required("linear.weight", &mut module.weight.value)?;
        projection.optional("absent", &mut absent)?;
        let disposition = |key: &str| assign(&key.replace("external.", "linear."));
        let before = projection.snapshot();
        manifest.load_with(&mut projection, disposition)?;
        let loaded = projection.snapshot();
        projection.restore(before, false)?;
        projection.restore(loaded.clone(), false)?;
        for (key, value) in projection.iter() {
            if let Some(array) = value {
                let values = values::<f32>(array)?;
                if key == "linear.weight" {
                    assert_eq!(values, [1., 2., 3., 4.]);
                } else {
                    assert_eq!(values, [5., 6.]);
                }
            } else {
                assert_eq!(key, "absent");
            }
        }
        tensors
            .get_mut("external.weight")
            .ok_or_else(|| WeightError::MissingKey("external.weight".into()))?
            .shape = vec![1, 4];
        write_tensors(&path, &tensors)?;
        assert!(matches!(
            manifest.load_with(&mut projection, disposition),
            Err(WeightError::ShapeMismatch { .. })
        ));
        for (key, value) in projection.iter() {
            let saved = loaded
                .iter()
                .find(|(saved_key, _)| *saved_key == key)
                .and_then(|(_, value)| value);
            assert_eq!(
                value.map(values::<f32>).transpose()?,
                saved.map(values::<f32>).transpose()?
            );
        }
        Ok(())
    }

    #[test]
    fn packed_embedding_and_linear_slots() -> Result<(), WeightError> {
        let tensors = read_tensors(&fixture("llama-quant4").join("model.safetensors"))?;
        let manifest = WeightManifest::discover(&fixture("llama-quant4"))?;
        let mut embedding = QuantizedEmbedding {
            group_size: 32,
            bits: 4,
            inner: Embedding {
                weight: Param::new(Array::from_slice(&[0_u32; 512], &[64, 8])),
            },
            scales: Param::new(Array::from_slice(&[0_f32; 128], &[64, 2])),
            biases: Param::new(Array::from_slice(&[0_f32; 128], &[64, 2])),
        };
        let mut linear = QuantizedLinear {
            group_size: 32,
            bits: 4,
            inner: Linear {
                weight: Param::new(Array::from_slice(&[0_u32; 512], &[64, 8])),
                bias: Param::new(None),
            },
            scales: Param::new(Array::from_slice(&[0_f32; 128], &[64, 2])),
            biases: Param::new(Array::from_slice(&[0_f32; 128], &[64, 2])),
        };
        let mut projection = StateProjection::new();
        projection.required("embedding.inner.weight", &mut embedding.inner.weight.value)?;
        projection.required("embedding.scales", &mut embedding.scales.value)?;
        projection.required("embedding.biases", &mut embedding.biases.value)?;
        projection.required("head.inner.weight", &mut linear.inner.weight.value)?;
        projection.required("head.scales", &mut linear.scales.value)?;
        projection.required("head.biases", &mut linear.biases.value)?;
        projection.optional("head.inner.bias", &mut linear.inner.bias.value)?;
        manifest.load_with(&mut projection, |key| {
            let mapped = key
                .strip_prefix("model.embed_tokens.")
                .map(|suffix| ("embedding", suffix))
                .or_else(|| key.strip_prefix("lm_head.").map(|suffix| ("head", suffix)));
            match mapped {
                Some((prefix, "weight")) => assign(&format!("{prefix}.inner.weight")),
                Some((prefix, suffix)) => assign(&format!("{prefix}.{suffix}")),
                None => WeightDisposition::Ignore {
                    reason: "test-only-submodule",
                },
            }
        })?;
        for (key, value) in projection.iter() {
            if let Some(value) = value {
                let external = key
                    .replace("embedding.", "model.embed_tokens.")
                    .replace("head.", "lm_head.")
                    .replace("inner.", "");
                let tensor = tensors
                    .get(&external)
                    .ok_or_else(|| WeightError::MissingKey(external.clone()))?;
                let (bytes, remainder) = tensor.data.as_chunks::<4>();
                assert!(remainder.is_empty());
                if key.ends_with("weight") {
                    assert_eq!(tensor.dtype, safetensors::Dtype::U32);
                    let expected: Vec<_> = bytes
                        .iter()
                        .map(|&bytes| u32::from_le_bytes(bytes))
                        .collect();
                    assert_eq!(values::<u32>(value)?, expected);
                } else {
                    assert_eq!(tensor.dtype, safetensors::Dtype::F32);
                    let expected: Vec<_> = bytes
                        .iter()
                        .map(|&bytes| f32::from_le_bytes(bytes))
                        .collect();
                    assert_eq!(values::<f32>(value)?, expected);
                }
            }
        }
        Ok(())
    }
}
