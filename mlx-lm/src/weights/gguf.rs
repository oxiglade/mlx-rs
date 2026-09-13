use super::*;
use crate::{config::LayerQuantization, Config};
use mlx_rs::io::GgufFile;
use std::num::NonZeroUsize;

fn is_norm(group: &str) -> bool {
    group == "model.norm" || group.ends_with("layernorm") || group.ends_with("_norm")
}

impl WeightManifest {
    pub(crate) fn from_gguf(file: &GgufFile) -> Result<Self, WeightError> {
        mlx_rs::with_stream(&mlx_rs::Stream::cpu(), || {
            let mut tensors = BTreeMap::new();
            let mut keys = file.array_keys()?;
            keys.sort();
            for key in keys {
                let array = file
                    .get_array(&key)?
                    .ok_or_else(|| WeightError::MissingKey(key.clone()))?;
                let shape = array
                    .shape()
                    .iter()
                    .map(|&n| {
                        usize::try_from(n).map_err(|_| {
                            WeightError::UnsupportedFormat(format!("negative dimension for {key}"))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                checked_shape(&key, &shape)?;
                let dtype = array.dtype();
                if tensors
                    .insert(
                        key.clone(),
                        WeightEntry {
                            source: WeightSource::Gguf(array),
                            shape,
                            dtype,
                        },
                    )
                    .is_some()
                {
                    return Err(WeightError::DuplicateTensor(key));
                }
            }
            Ok(Self { tensors })
        })
    }

    pub(crate) fn normalize_gguf(
        self,
        disposition: impl Fn(&str) -> WeightDisposition,
    ) -> Result<Self, WeightError> {
        let mut tensors = BTreeMap::new();
        for (external, entry) in self.tensors {
            let WeightDisposition::Parameter(path) = disposition(&external) else {
                return Err(WeightError::UnexpectedKey(external));
            };
            if tensors.insert(path.as_str().to_owned(), entry).is_some() {
                return Err(WeightError::DuplicateTensor(external));
            }
        }
        Ok(Self { tensors })
    }

    pub(crate) fn permute_gguf_rows(
        &mut self,
        group: &ParameterPath,
        heads: usize,
        head_dim: usize,
    ) -> Result<(), WeightError> {
        let rows = heads
            .checked_mul(head_dim)
            .filter(|_| heads > 0 && head_dim > 0 && head_dim.is_multiple_of(2))
            .ok_or_else(|| {
                WeightError::UnsupportedFormat(format!(
                    "invalid Q/K head layout {}",
                    group.as_str()
                ))
            })?;
        mlx_rs::with_stream(&mlx_rs::Stream::cpu(), || {
            let mut staged = Vec::new();
            for suffix in ["weight", "scales", "biases"] {
                let key = format!("{}.{suffix}", group.as_str());
                let Some(entry) = self.tensors.get(&key) else {
                    continue;
                };
                if entry.shape.len() != 2 {
                    return Err(WeightError::UnsupportedFormat(format!(
                        "Q/K slot {key} must be a matrix"
                    )));
                }
                validate_shape(&key, &[rows, entry.shape[1]], &entry.shape)?;
                let shape = checked_shape(&key, &[heads, head_dim / 2, 2, entry.shape[1]])?;
                let array = self
                    .read_tensor(&key)?
                    .reshape(&shape)?
                    .transpose_axes(&[0, 2, 1, 3])?
                    .reshape(&checked_shape(&key, &entry.shape)?)?;
                array.eval()?;
                staged.push((key, array));
            }
            for (key, array) in staged {
                self.tensors
                    .get_mut(&key)
                    .expect("staged existing slot")
                    .source = WeightSource::Gguf(array);
            }
            Ok(())
        })
        .map_err(external_error)
    }

    pub(crate) fn validate_gguf_keys(&self, layers: usize, qwen3: bool) -> Result<(), WeightError> {
        let mut groups = BTreeSet::from(["model.embed_tokens".to_owned(), "model.norm".to_owned()]);
        if self.tensors.keys().any(|key| key.starts_with("lm_head.")) {
            groups.insert("lm_head".into());
        }
        for layer in 0..layers {
            for name in [
                "input_layernorm",
                "post_attention_layernorm",
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ] {
                groups.insert(format!("model.layers.{layer}.{name}"));
            }
            if qwen3 {
                for name in ["q_norm", "k_norm"] {
                    groups.insert(format!("model.layers.{layer}.self_attn.{name}"));
                }
            }
        }
        let affine = self.affine_group_prefixes();
        let mut all: BTreeSet<_> = groups.iter().cloned().collect();
        all.extend(
            self.tensors
                .keys()
                .filter_map(|key| key.rsplit_once('.').map(|(group, _)| group.to_owned())),
        );
        let mut all: Vec<_> = all.into_iter().collect();
        all.sort_by_key(|group| crate::arch::gguf::external(&format!("{group}.weight")));
        for group in all {
            if !groups.contains(&group) {
                let key = self
                    .tensors
                    .keys()
                    .find(|key| {
                        key.rsplit_once('.')
                            .is_some_and(|(prefix, _)| prefix == group)
                    })
                    .expect("observed group");
                return Err(WeightError::UnexpectedKey(key.clone()));
            }
            let key = format!("{group}.weight");
            let entry = self.entry(&key)?;
            if is_norm(&group) && !matches!(entry.dtype, Dtype::Float16 | Dtype::Float32) {
                return Err(WeightError::UnsupportedDtype {
                    key,
                    dtype: format!("{:?}", entry.dtype),
                });
            }
            if !is_norm(&group) && affine.contains(&group) {
                self.affine_entries(&group)?;
            }
        }
        Ok(())
    }

    pub(crate) fn gguf_quantization_groups(
        &self,
    ) -> Result<Vec<GgufQuantizationGroup>, WeightError> {
        let affine = self.affine_group_prefixes();
        let mut groups = affine.clone();
        groups.extend(
            self.tensors
                .keys()
                .filter_map(|key| key.strip_suffix(".weight").map(str::to_owned)),
        );
        let mut ordered: Vec<_> = groups.iter().collect();
        ordered.sort_by_key(|group| crate::arch::gguf::external(&format!("{group}.weight")));
        let mut result = Vec::new();
        for group in ordered {
            let key = format!("{group}.weight");
            let entry = self.entry(&key)?;
            if is_norm(group) {
                if !matches!(entry.dtype, Dtype::Float16 | Dtype::Float32) {
                    return Err(WeightError::UnsupportedDtype {
                        key,
                        dtype: format!("{:?}", entry.dtype),
                    });
                }
                continue;
            }
            let quantization = if affine.contains(group) {
                let [weight, scales, biases] = self.affine_entries(group)?;
                validate_dtype(&format!("{group}.scales"), Dtype::Float16, scales.dtype)?;
                validate_dtype(&format!("{group}.biases"), Dtype::Float16, biases.dtype)?;
                let columns = scales.shape[1];
                let bits = if columns > 0 && weight.shape[1].is_multiple_of(columns) {
                    weight.shape[1] / columns
                } else {
                    0
                };
                if !matches!(bits, 4 | 8) {
                    return Err(WeightError::UnsupportedFormat(format!(
                        "affine group {}: group_size 32, bits {bits}",
                        crate::arch::gguf::external(&key)
                    )));
                }
                let options = AffineQuantization {
                    group_size: NonZeroUsize::new(32).unwrap(),
                    bits: bits as u8,
                };
                self.validate_affine_group(group, &options)?;
                LayerQuantization::Affine(options)
            } else {
                if !matches!(entry.dtype, Dtype::Float16 | Dtype::Float32) {
                    return Err(WeightError::UnsupportedDtype {
                        key: key.clone(),
                        dtype: format!("{:?}", entry.dtype),
                    });
                }
                LayerQuantization::Unquantized
            };
            result.push(GgufQuantizationGroup {
                path: ParameterPath::new(group.as_str()),
                quantization,
            });
        }
        result.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(result)
    }

    pub(crate) fn validate_gguf_shapes(&self, config: &Config) -> Result<(), WeightError> {
        let d = &config.dimensions;
        let mut expected = BTreeMap::new();
        let mut matrix = |group: String, output, input| {
            expected.insert(group, vec![output, input]);
        };
        matrix(
            "model.embed_tokens".into(),
            d.vocabulary_size,
            d.hidden_size,
        );
        if !config.tie_word_embeddings {
            matrix("lm_head".into(), d.vocabulary_size, d.hidden_size);
        }
        for i in 0..d.layer_count {
            for (name, output, input) in [
                (
                    "self_attn.q_proj",
                    d.attention_heads * d.head_dim,
                    d.hidden_size,
                ),
                ("self_attn.k_proj", d.kv_heads * d.head_dim, d.hidden_size),
                ("self_attn.v_proj", d.kv_heads * d.head_dim, d.hidden_size),
                (
                    "self_attn.o_proj",
                    d.hidden_size,
                    d.attention_heads * d.head_dim,
                ),
                ("mlp.gate_proj", d.intermediate_size, d.hidden_size),
                ("mlp.up_proj", d.intermediate_size, d.hidden_size),
                ("mlp.down_proj", d.hidden_size, d.intermediate_size),
            ] {
                matrix(format!("model.layers.{i}.{name}"), output, input);
            }
        }
        expected.insert("model.norm".into(), vec![d.hidden_size]);
        for i in 0..d.layer_count {
            for name in ["input_layernorm", "post_attention_layernorm"] {
                expected.insert(format!("model.layers.{i}.{name}"), vec![d.hidden_size]);
            }
            if config.model_type.as_str() == "qwen3" {
                for name in ["q_norm", "k_norm"] {
                    expected.insert(
                        format!("model.layers.{i}.self_attn.{name}"),
                        vec![d.head_dim],
                    );
                }
            }
        }
        let groups = self.gguf_quantization_groups()?;
        let quantized: BTreeMap<_, _> = groups
            .iter()
            .map(|g| (g.path.as_str(), &g.quantization))
            .collect();
        let mut slots = BTreeMap::new();
        for (group, shape) in expected {
            if let Some(LayerQuantization::Affine(q)) = quantized.get(group.as_str()) {
                if shape.len() != 2 || !shape[1].is_multiple_of(32) {
                    return Err(WeightError::UnsupportedFormat(format!(
                        "unaligned affine group {group}"
                    )));
                }
                let packed = shape[1]
                    .checked_mul(usize::from(q.bits))
                    .ok_or(safetensors::SafeTensorError::ValidationOverflow)?
                    / 32;
                slots.insert(
                    format!("{group}.weight"),
                    (vec![shape[0], packed], Some(Dtype::Uint32)),
                );
                for suffix in ["scales", "biases"] {
                    slots.insert(
                        format!("{group}.{suffix}"),
                        (vec![shape[0], shape[1] / 32], Some(Dtype::Float16)),
                    );
                }
            } else {
                slots.insert(format!("{group}.weight"), (shape, None));
            }
        }
        let mut keys: Vec<_> = slots
            .keys()
            .chain(self.tensors.keys())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        keys.sort_by_key(|key| {
            let external = crate::arch::gguf::external(key);
            let (group, suffix) = external.rsplit_once('.').expect("parameter suffix");
            // Validate packed storage before companions, as in affine_entries.
            let slot = match suffix {
                "weight" => 0,
                "scales" => 1,
                _ => 2,
            };
            (group.to_owned(), slot)
        });
        for key in keys {
            let (shape, dtype) = slots
                .get(key)
                .ok_or_else(|| WeightError::UnexpectedKey(key.clone()))?;
            let entry = self.entry(key)?;
            validate_shape(key, shape, &entry.shape)?;
            if let Some(dtype) = dtype {
                validate_dtype(key, *dtype, entry.dtype)?;
            } else if !matches!(entry.dtype, Dtype::Float16 | Dtype::Float32) {
                return Err(WeightError::UnsupportedDtype {
                    key: key.clone(),
                    dtype: format!("{:?}", entry.dtype),
                });
            }
        }
        Ok(())
    }
}

pub(crate) fn external_error(error: WeightError) -> WeightError {
    use crate::arch::gguf::external;
    match error {
        WeightError::MissingKey(key) => WeightError::MissingKey(external(&key)),
        WeightError::UnexpectedKey(key) => WeightError::UnexpectedKey(external(&key)),
        WeightError::DuplicateTensor(key) => WeightError::DuplicateTensor(external(&key)),
        WeightError::ShapeMismatch {
            key,
            expected,
            actual,
        } => WeightError::ShapeMismatch {
            key: external(&key),
            expected,
            actual,
        },
        WeightError::DtypeMismatch {
            key,
            expected,
            actual,
        } => WeightError::DtypeMismatch {
            key: external(&key),
            expected,
            actual,
        },
        WeightError::UnsupportedDtype { key, dtype } => WeightError::UnsupportedDtype {
            key: external(&key),
            dtype,
        },
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(entries: &[(&str, Dtype, &[usize])]) -> WeightManifest {
        WeightManifest {
            tensors: entries
                .iter()
                .map(|(key, dtype, shape)| {
                    (
                        (*key).into(),
                        WeightEntry {
                            source: WeightSource::Safetensors(PathBuf::from("unused")),
                            shape: shape.to_vec(),
                            dtype: *dtype,
                        },
                    )
                })
                .collect(),
        }
    }

    fn fixture_layout(name: &str) -> anyhow::Result<(WeightManifest, Config, serde_json::Value)> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../conformance/mlx-lm/fixtures")
            .join(name);
        let doc: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("expectations.json"))?)?;
        let weights = WeightManifest::from_safetensors(&root.join("converted.safetensors"))?;
        let raw =
            crate::config::RawConfig::from_bytes(&serde_json::to_vec(&doc["config"]["bridge"])?)?;
        let factory = crate::arch::factory(&raw.model_type)?;
        let config = match factory.parse_config(&raw)? {
            crate::arch::ParsedArchitecture::Llama(p) => p.config,
            crate::arch::ParsedArchitecture::Qwen3(p) => p.config,
        };
        Ok((weights, config, doc))
    }

    #[test]
    fn ten_converted_fixture_layouts_and_keyed_quantization() -> anyhow::Result<()> {
        for family in ["llama", "qwen3"] {
            for storage in ["f32", "f16", "q4_0", "q4_1", "q8_0"] {
                let (weights, config, doc) = fixture_layout(&format!("gguf-{family}-{storage}"))?;
                let factory = crate::arch::factory(family)?;
                let weights = weights.normalize_gguf(|key| factory.map_gguf_key(key))?;
                weights.validate_gguf_keys(config.dimensions.layer_count, family == "qwen3")?;
                weights.validate_gguf_shapes(&config)?;
                let groups = weights.gguf_quantization_groups()?;
                let mut actual = serde_json::Map::new();
                for group in groups {
                    actual.insert(group.path.as_str().into(), match group.quantization {
                        LayerQuantization::Unquantized => serde_json::json!(false),
                        LayerQuantization::Affine(q) => serde_json::json!({"group_size":q.group_size.get(),"bits":q.bits,"mode":"affine"}),
                    });
                }
                if matches!(storage, "f32" | "f16") {
                    assert!(actual.values().all(|value| value == false));
                } else {
                    assert_eq!(
                        serde_json::Value::Object(actual),
                        doc["config"]["quantization"]
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn frozen_weight_error_recipes_from_converted_headers() -> anyhow::Result<()> {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../conformance/mlx-lm/gguf_cases.json");
        let recipes: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        for recipe in recipes["cases"].as_array().unwrap() {
            let operation = recipe["operation"].as_str().unwrap();
            if !matches!(
                operation,
                "copy_tensor"
                    | "rename_tensor"
                    | "remove_tensor"
                    | "truncate_rows"
                    | "truncate_columns"
            ) {
                continue;
            }
            let (mut weights, config, _) = fixture_layout(recipe["base"].as_str().unwrap())?;
            let key = recipe["key"].as_str().unwrap();
            let group = key.strip_suffix(".weight").unwrap();
            let names: Vec<_> = weights
                .tensors
                .keys()
                .filter(|name| {
                    *name == key
                        || (weights.tensors[key].dtype == Dtype::Uint32
                            && name.starts_with(&format!("{group}.")))
                })
                .cloned()
                .collect();
            for name in names {
                match operation {
                    "remove_tensor" => {
                        weights.tensors.remove(&name);
                    }
                    "copy_tensor" | "rename_tensor" => {
                        let target = recipe["target"].as_str().unwrap();
                        let target = if name == key {
                            target.to_owned()
                        } else {
                            format!(
                                "{}.{}",
                                target.strip_suffix(".weight").unwrap(),
                                name.rsplit_once('.').unwrap().1
                            )
                        };
                        let entry = weights.tensors[&name].clone();
                        weights.tensors.insert(target, entry);
                        if operation == "rename_tensor" {
                            weights.tensors.remove(&name);
                        }
                    }
                    "truncate_rows" => {
                        weights.tensors.get_mut(&name).unwrap().shape[0] =
                            recipe["rows"].as_u64().unwrap() as usize
                    }
                    "truncate_columns" => weights.tensors.get_mut(&name).unwrap().shape[1] /= 2,
                    _ => unreachable!(),
                }
            }
            let factory = crate::arch::factory(config.model_type.as_str())?;
            let result = weights
                .normalize_gguf(|key| factory.map_gguf_key(key))
                .and_then(|weights| {
                    weights
                        .validate_gguf_keys(
                            config.dimensions.layer_count,
                            config.model_type.as_str() == "qwen3",
                        )
                        .map_err(external_error)?;
                    // The embedding determines vocabulary, including the bad-row recipe.
                    let mut config = config.clone();
                    config.dimensions.vocabulary_size =
                        weights.tensors["model.embed_tokens.weight"].shape[0];
                    weights
                        .validate_gguf_shapes(&config)
                        .map_err(external_error)
                });
            let actual = match result.expect_err("malformed weights must fail") {
                WeightError::MissingKey(key) => {
                    serde_json::json!({"variant":"LoadError::Weights(WeightError::MissingKey)","key":key})
                }
                WeightError::UnexpectedKey(key) => {
                    serde_json::json!({"variant":"LoadError::Weights(WeightError::UnexpectedKey)","key":key})
                }
                WeightError::ShapeMismatch {
                    key,
                    expected,
                    actual,
                } => {
                    serde_json::json!({"variant":"LoadError::Weights(WeightError::ShapeMismatch)","key":key,"expected":expected,"actual":actual})
                }
                other => anyhow::bail!("{}: {other:?}", recipe["id"]),
            };
            assert_eq!(actual, recipe["expected"], "{}", recipe["id"]);
        }
        Ok(())
    }

    #[test]
    fn normalization_collisions_and_rejections() {
        let weights = manifest(&[
            ("a", Dtype::Float32, &[2, 2]),
            ("b", Dtype::Float32, &[2, 2]),
        ]);
        assert!(
            matches!(weights.normalize_gguf(|_| WeightDisposition::Parameter(ParameterPath::new("same"))), Err(WeightError::DuplicateTensor(key)) if key == "b")
        );
        let weights = manifest(&[
            ("z", Dtype::Float32, &[2, 2]),
            ("a", Dtype::Float32, &[2, 2]),
        ]);
        assert!(
            matches!(weights.normalize_gguf(|_| WeightDisposition::Ignore { reason: "not allowed" }), Err(WeightError::UnexpectedKey(key)) if key == "a")
        );
    }
    #[test]
    fn gguf_missing_keys_use_external_order() {
        let weights = manifest(&[]);
        assert!(
            matches!(weights.validate_gguf_keys(1, false).map_err(external_error), Err(WeightError::MissingKey(key)) if key == "blk.0.attn_k.weight")
        );
        let mut weights = manifest(&[(
            "model.layers.0.self_attn.k_proj.weight",
            Dtype::Float32,
            &[32, 64],
        )]);
        assert!(
            matches!(weights.validate_gguf_keys(1, false).map_err(external_error), Err(WeightError::MissingKey(key)) if key == "blk.0.attn_norm.weight")
        );
        weights.tensors.insert(
            "model.layers.0.input_layernorm.weight".into(),
            WeightEntry {
                source: WeightSource::Safetensors(PathBuf::from("unused")),
                shape: vec![64],
                dtype: Dtype::Float32,
            },
        );
        assert!(
            matches!(weights.validate_gguf_keys(1, false).map_err(external_error), Err(WeightError::MissingKey(key)) if key == "blk.0.attn_output.weight")
        );
    }

    #[test]
    fn integer_norms_and_affine_diagnostics_use_external_groups() {
        let weights = manifest(&[("model.norm.weight", Dtype::Uint32, &[64])]);
        assert!(
            matches!(weights.gguf_quantization_groups().map_err(external_error), Err(WeightError::UnsupportedDtype { key, .. }) if key == "output_norm.weight")
        );
        let weights = manifest(&[
            ("lm_head.weight", Dtype::Uint32, &[64, 8]),
            ("lm_head.scales", Dtype::Float32, &[64, 2]),
            ("lm_head.biases", Dtype::Float32, &[64, 2]),
            (
                "model.layers.0.self_attn.k_proj.weight",
                Dtype::Uint32,
                &[32, 8],
            ),
            (
                "model.layers.0.self_attn.k_proj.scales",
                Dtype::Float32,
                &[32, 2],
            ),
            (
                "model.layers.0.self_attn.k_proj.biases",
                Dtype::Float32,
                &[32, 2],
            ),
        ]);
        assert!(
            matches!(weights.gguf_quantization_groups().map_err(external_error), Err(WeightError::DtypeMismatch { key, .. }) if key == "blk.0.attn_k.scales")
        );
    }

    #[test]
    fn converted_affine_profile_is_narrow() {
        for bits in [2, 3, 4, 6, 8] {
            let weights = manifest(&[
                ("x.weight", Dtype::Uint32, &[16, bits * 2]),
                ("x.scales", Dtype::Float16, &[16, 2]),
                ("x.biases", Dtype::Float16, &[16, 2]),
            ]);
            assert_eq!(
                weights.gguf_quantization_groups().is_ok(),
                matches!(bits, 4 | 8)
            );
        }
        for dtype in [Dtype::Float32, Dtype::Bfloat16] {
            let weights = manifest(&[
                ("x.weight", Dtype::Uint32, &[16, 8]),
                ("x.scales", dtype, &[16, 2]),
                ("x.biases", dtype, &[16, 2]),
            ]);
            assert!(
                matches!(weights.gguf_quantization_groups(), Err(WeightError::DtypeMismatch { key, expected: Dtype::Float16, .. }) if key == "x.scales")
            );
        }
    }
    #[test]
    fn affine_companions_precede_dtype_and_orphan_output() {
        let weights = manifest(&[("lm_head.scales", Dtype::Float32, &[64])]);
        assert!(
            matches!(weights.gguf_quantization_groups().map_err(external_error), Err(WeightError::MissingKey(key)) if key == "output.weight")
        );
        let weights = manifest(&[
            (
                "model.layers.0.self_attn.q_proj.weight",
                Dtype::Float32,
                &[64, 64],
            ),
            (
                "model.layers.0.self_attn.q_proj.scales",
                Dtype::Float32,
                &[64],
            ),
        ]);
        assert!(
            matches!(weights.gguf_quantization_groups().map_err(external_error), Err(WeightError::MissingKey(key)) if key == "blk.0.attn_q.biases")
        );
    }
    #[test]
    fn gguf_evaluation_failure_keeps_projection_unchanged() -> anyhow::Result<()> {
        mlx_rs::with_stream(&mlx_rs::Stream::cpu(), || -> anyhow::Result<()> {
            let broken = mlx_rs::linalg::inv(Array::from_slice(&[0_f32], &[1, 1]))?;
            let good = Array::from_slice(&[9_f32], &[1, 1]);
            let weights = WeightManifest {
                tensors: [("a", good), ("b", broken)]
                    .into_iter()
                    .map(|(key, array)| {
                        (
                            key.into(),
                            WeightEntry {
                                source: WeightSource::Gguf(array),
                                shape: vec![1, 1],
                                dtype: Dtype::Float32,
                            },
                        )
                    })
                    .collect(),
            };
            let mut a = Array::from_slice(&[1_f32], &[1, 1]);
            let mut b = Array::from_slice(&[2_f32], &[1, 1]);
            let mut projection = StateProjection::new();
            projection.required("a", &mut a)?;
            projection.required("b", &mut b)?;
            assert!(matches!(
                weights.load_with(&mut projection, |key| WeightDisposition::Parameter(
                    ParameterPath::new(key)
                )),
                Err(WeightError::Exception(_))
            ));
            drop(projection);
            assert_eq!(a.to_vec_exact::<f32>()?, [1.]);
            assert_eq!(b.to_vec_exact::<f32>()?, [2.]);
            Ok(())
        })
    }

    #[test]
    fn llama_literal_orientation_all_affine_slots_and_owned_backing() -> anyhow::Result<()> {
        let mut file = GgufFile::new()?;
        let order = [0_u32, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15];
        // Two heads catch accidental mixing across a head boundary.
        let values: Vec<_> = [0, 16]
            .into_iter()
            .flat_map(|base| order.map(|row| base + row))
            .collect();
        let array = Array::from_slice(&values, &[32, 1]).as_dtype(Dtype::Float32)?;
        for suffix in ["weight", "scales", "biases"] {
            file.insert_array(format!("blk.0.attn_q.{suffix}"), &array)?;
        }
        let mut weights = WeightManifest::from_gguf(&file)?.normalize_gguf(|key| {
            use crate::arch::ArchitectureFactory;
            crate::arch::llama::Factory.map_gguf_key(key)
        })?;
        drop(file);
        weights.permute_gguf_rows(
            &ParameterPath::new("model.layers.0.self_attn.q_proj"),
            2,
            16,
        )?;
        for suffix in ["weight", "scales", "biases"] {
            assert_eq!(
                weights
                    .read_tensor(&format!("model.layers.0.self_attn.q_proj.{suffix}"))?
                    .to_vec_cast::<u32>()?,
                (0..32).collect::<Vec<_>>()
            );
        }
        Ok(())
    }
}
