use crate::config::{AffineQuantization, ParameterPath};
pub use crate::error::WeightError;
use mlx_rs::{error::IoError, io::GgufFile, utils::StateProjection, Array, Dtype};
use safetensors::tensor::{Metadata, TensorInfo};
use serde::{de::MapAccess, Deserialize, Deserializer};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::{Component, Path, PathBuf},
};

pub(crate) struct WeightManifest {
    pub(crate) tensors: BTreeMap<String, WeightEntry>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WeightEntry {
    pub(crate) shard: PathBuf,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: Dtype,
}
pub(crate) enum WeightDisposition {
    Parameter(ParameterPath),
    Ignore { reason: &'static str },
    Reject,
}

type ExpectedSlots = BTreeMap<String, Option<(Dtype, Vec<usize>)>>;

impl WeightManifest {
    pub(crate) fn discover(directory: &Path) -> Result<Self, WeightError> {
        let index = directory.join("model.safetensors.index.json");
        if index.try_exists()? {
            Self::from_sharded_index(&index)
        } else {
            Self::from_safetensors(&directory.join("model.safetensors"))
        }
    }

    pub(crate) fn from_safetensors(path: &Path) -> Result<Self, WeightError> {
        let file = open_weights(path)?;
        let (_, metadata) = read_header(&file)?;
        let mut tensors = BTreeMap::new();
        for (key, info) in metadata.tensors() {
            let dtype = tensor_dtype(&key, info.dtype)?;
            checked_shape(&key, &info.shape)?;
            tensors.insert(
                key,
                WeightEntry {
                    shard: path.to_owned(),
                    shape: info.shape.clone(),
                    dtype,
                },
            );
        }
        Ok(Self { tensors })
    }

    pub(crate) fn from_sharded_index(path: &Path) -> Result<Self, WeightError> {
        #[derive(Deserialize)]
        struct Index {
            weight_map: Entries<String>,
        }
        let index: Index = serde_json::from_reader(open_weights(path)?)?;
        let directory = path.parent().unwrap_or_else(|| Path::new("."));
        let mut weight_map = BTreeMap::new();
        let mut shards = BTreeSet::new();
        for (key, shard) in index.weight_map.0 {
            let relative = Path::new(&shard);
            if relative
                .components()
                .any(|part| !matches!(part, Component::Normal(_)))
                || relative.as_os_str().is_empty()
            {
                return Err(WeightError::ConflictingIndex {
                    key,
                    shard: relative.to_owned(),
                });
            }
            let shard = directory.join(relative);
            if weight_map.insert(key.clone(), shard.clone()).is_some() {
                return Err(WeightError::ConflictingIndex { key, shard });
            }
            shards.insert(shard);
        }
        let mut tensors = BTreeMap::new();
        for shard in shards {
            let manifest = Self::from_safetensors(&shard).map_err(|error| match error {
                WeightError::MissingFile(path) => WeightError::MissingShard(path),
                other => other,
            })?;
            for (key, entry) in manifest.tensors {
                if tensors.insert(key.clone(), entry).is_some() {
                    return Err(WeightError::DuplicateTensor(key));
                }
            }
        }
        // Duplicate detection precedes index reconciliation so an extra copy has one stable error.
        for (key, entry) in &tensors {
            if weight_map.get(key) != Some(&entry.shard) {
                return Err(WeightError::ConflictingIndex {
                    key: key.clone(),
                    shard: entry.shard.clone(),
                });
            }
        }
        for (key, shard) in weight_map {
            if !tensors.contains_key(&key) {
                return Err(WeightError::ConflictingIndex { key, shard });
            }
        }
        Ok(Self { tensors })
    }

    #[allow(dead_code)] // GGUF loading is tranche 4.
    pub(crate) fn from_gguf(_file: &GgufFile) -> Result<Self, WeightError> {
        Err(WeightError::UnsupportedFormat(
            crate::NotYetImplemented("GGUF manifest").to_string(),
        ))
    }

    #[allow(dead_code)] // GGUF loading is tranche 4.
    pub(crate) fn affine_groups(&self) -> Result<BTreeSet<String>, WeightError> {
        let groups = self.affine_group_prefixes();
        for prefix in &groups {
            self.affine_entries(prefix)?;
        }
        Ok(groups)
    }

    fn affine_group_prefixes(&self) -> BTreeSet<String> {
        let mut groups = BTreeSet::new();
        for (key, entry) in &self.tensors {
            if let Some(prefix) = key
                .strip_suffix(".scales")
                .or_else(|| key.strip_suffix(".biases"))
            {
                groups.insert(prefix.to_owned());
            } else if entry.dtype == Dtype::Uint32 {
                if let Some(prefix) = key.strip_suffix(".weight") {
                    groups.insert(prefix.to_owned());
                }
            }
        }
        groups
    }

    #[allow(dead_code)] // GGUF loading is tranche 4.
    pub(crate) fn validate_affine_group(
        &self,
        prefix: &str,
        quantization: &AffineQuantization,
    ) -> Result<(), WeightError> {
        let [weight, scales, _] = self.affine_entries(prefix)?;
        let bits = usize::from(quantization.bits);
        let group_size = quantization.group_size.get();
        if !matches!(bits, 2 | 3 | 4 | 6 | 8) || !matches!(group_size, 32 | 64 | 128) {
            return Err(WeightError::UnsupportedFormat(format!(
                "affine group {prefix}: group_size {group_size}, bits {bits}"
            )));
        }
        let columns = scales.shape[1]
            .checked_mul(group_size)
            .and_then(|width| width.checked_mul(bits))
            .ok_or(safetensors::SafeTensorError::ValidationOverflow)?;
        if columns % 32 != 0 {
            return Err(WeightError::UnsupportedFormat(format!(
                "unaligned affine group {prefix}"
            )));
        }
        validate_shape(
            &format!("{prefix}.weight"),
            &[scales.shape[0], columns / 32],
            &weight.shape,
        )
    }

    fn affine_entries(&self, prefix: &str) -> Result<[&WeightEntry; 3], WeightError> {
        let weight_key = format!("{prefix}.weight");
        let scales_key = format!("{prefix}.scales");
        let biases_key = format!("{prefix}.biases");
        let weight = self.entry(&weight_key)?;
        let scales = self.entry(&scales_key)?;
        let biases = self.entry(&biases_key)?;
        validate_dtype(&weight_key, Dtype::Uint32, weight.dtype)?;
        let dtype = scales.dtype;
        if !matches!(dtype, Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32) {
            return Err(WeightError::UnsupportedDtype {
                key: scales_key,
                dtype: format!("{:?}", scales.dtype),
            });
        }
        validate_dtype(&biases_key, dtype, biases.dtype)?;
        for (key, entry) in [(&weight_key, weight), (&scales_key, scales)] {
            if entry.shape.len() != 2 {
                return Err(WeightError::UnsupportedFormat(format!(
                    "affine tensor {key} must be a matrix"
                )));
            }
        }
        validate_shape(&biases_key, &scales.shape, &biases.shape)?;
        validate_shape(
            &scales_key,
            &[weight.shape[0], scales.shape[1]],
            &scales.shape,
        )?;
        Ok([weight, scales, biases])
    }

    pub(crate) fn load_with(
        &self,
        projection: &mut StateProjection<'_>,
        disposition: impl Fn(&str) -> WeightDisposition,
    ) -> Result<(), WeightError> {
        let expected = projection
            .iter()
            .map(|(key, value)| {
                let metadata = value
                    .map(|array| {
                        let shape = array
                            .shape()
                            .iter()
                            .map(|&dimension| {
                                usize::try_from(dimension).map_err(|_| {
                                    WeightError::UnsupportedFormat(format!(
                                        "negative dimension for {key}"
                                    ))
                                })
                            })
                            .collect::<Result<_, _>>()?;
                        Ok::<_, WeightError>((array.dtype(), shape))
                    })
                    .transpose()?;
                Ok((key.to_owned(), metadata))
            })
            .collect::<Result<BTreeMap<_, _>, WeightError>>()?;
        let assignments = self.plan(&expected, disposition)?;
        let mut shards = BTreeMap::<_, Vec<_>>::new();
        for (key, external) in assignments {
            let entry = self.entry(&external)?;
            shards
                .entry(&entry.shard)
                .or_default()
                .push((key, external, entry));
        }
        let mut loaded: BTreeMap<_, _> = expected.keys().map(|key| (key.clone(), None)).collect();
        for (shard, assignments) in shards {
            let mut arrays =
                Array::load_safetensors(shard).map_err(|error| shard_error(shard, error))?;
            for (key, external, entry) in assignments {
                let array = arrays
                    .remove(&external)
                    .ok_or_else(|| WeightError::MissingKey(external.clone()))?;
                validate_array(&external, entry, &array)?;
                loaded.insert(key, Some(array));
            }
        }
        mlx_rs::transforms::eval(loaded.values().filter_map(Option::as_ref))?;
        let mut staged = StateProjection::new();
        for (key, value) in &mut loaded {
            staged.optional(key.as_str(), value)?;
        }
        projection.restore(staged.snapshot(), false)?;
        Ok(())
    }

    fn plan(
        &self,
        expected: &ExpectedSlots,
        disposition: impl Fn(&str) -> WeightDisposition,
    ) -> Result<BTreeMap<String, String>, WeightError> {
        let mut assignments = BTreeMap::new();
        for external in self.tensors.keys() {
            match disposition(external) {
                WeightDisposition::Parameter(path) => {
                    let key = path.as_str();
                    if !matches!(expected.get(key), Some(Some(_))) {
                        return Err(WeightError::UnexpectedKey(external.clone()));
                    }
                    if assignments
                        .insert(key.to_owned(), external.clone())
                        .is_some()
                    {
                        return Err(WeightError::DuplicateTensor(key.to_owned()));
                    }
                }
                WeightDisposition::Ignore { reason } if !reason.trim().is_empty() => {}
                _ => return Err(WeightError::UnexpectedKey(external.clone())),
            }
        }
        for (key, metadata) in expected {
            if let Some((dtype, shape)) = metadata {
                let external = assignments
                    .get(key)
                    .ok_or_else(|| WeightError::MissingKey(key.clone()))?;
                let entry = self.entry(external)?;
                validate_shape(external, shape, &entry.shape)?;
                validate_dtype(external, *dtype, entry.dtype)?;
            }
        }
        let present = assignments.values().collect::<BTreeSet<_>>();
        for prefix in self.affine_group_prefixes() {
            if ["weight", "scales", "biases"]
                .iter()
                .any(|suffix| present.contains(&format!("{prefix}.{suffix}")))
            {
                self.affine_entries(&prefix)?;
            }
        }
        Ok(assignments)
    }

    fn entry(&self, external: &str) -> Result<&WeightEntry, WeightError> {
        self.tensors
            .get(external)
            .ok_or_else(|| WeightError::MissingKey(external.to_owned()))
    }

    #[allow(dead_code)] // GGUF loading is tranche 4.
    pub(crate) fn read_tensor(&self, external: &str) -> Result<Array, WeightError> {
        let entry = self.entry(external)?;
        let mut arrays = Array::load_safetensors(&entry.shard)
            .map_err(|error| shard_error(&entry.shard, error))?;
        let array = arrays
            .remove(external)
            .ok_or_else(|| WeightError::MissingKey(external.to_owned()))?;
        validate_array(external, entry, &array)?;
        Ok(array)
    }
}

fn validate_array(external: &str, entry: &WeightEntry, array: &Array) -> Result<(), WeightError> {
    let shape = array
        .shape()
        .iter()
        .map(|&dimension| {
            usize::try_from(dimension).map_err(|_| {
                WeightError::UnsupportedFormat(format!("negative dimension for {external}"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    validate_shape(external, &entry.shape, &shape)?;
    validate_dtype(external, entry.dtype, array.dtype())
}

fn shard_error(path: &Path, error: IoError) -> WeightError {
    match error {
        IoError::Exception(error) => WeightError::Exception(error),
        IoError::NotFile | IoError::UnableToOpenFile => WeightError::MissingShard(path.to_owned()),
        other => WeightError::UnsupportedFormat(format!("{}: {other}", path.display())),
    }
}

fn open_weights(path: &Path) -> Result<File, WeightError> {
    File::open(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            WeightError::MissingFile(path.to_owned())
        } else {
            WeightError::Io(error)
        }
    })
}

fn read_header(mut file: &File) -> Result<(u64, Metadata), WeightError> {
    let length = file.metadata()?.len();
    let mut prefix = [0; 8];
    file.read_exact(&mut prefix)?;
    let size = u64::from_le_bytes(prefix);
    if size > 100_000_000 {
        return Err(safetensors::SafeTensorError::HeaderTooLarge.into());
    }
    let offset = size + 8;
    if offset > length {
        return Err(safetensors::SafeTensorError::InvalidHeaderLength.into());
    }
    let mut header = vec![0; size as usize];
    file.read_exact(&mut header)?;
    let entries: Entries<serde_json::Value> = serde_json::from_slice(&header)?;
    let mut names = BTreeSet::new();
    let mut tensors = Vec::new();
    for (key, value) in entries.0 {
        if !names.insert(key.clone()) {
            return Err(WeightError::DuplicateTensor(key));
        }
        if key == "__metadata__" {
            let _: BTreeMap<String, String> = serde_json::from_value(value)?;
        } else {
            let info: TensorInfo = serde_json::from_value(value)?;
            tensor_dtype(&key, info.dtype)?;
            tensors.push((key, info));
        }
    }
    tensors.sort_by_key(|(_, info)| info.data_offsets);
    let end = tensors.last().map_or(0, |(_, info)| info.data_offsets.1);
    let metadata = Metadata::new(None, tensors)?;
    if (end as u64).checked_add(offset) != Some(length) {
        return Err(safetensors::SafeTensorError::MetadataIncompleteBuffer.into());
    }
    Ok((offset, metadata))
}

// JSON maps must retain repeated keys until they can become typed loader errors.
struct Entries<T>(Vec<(String, T)>);
impl<'de, T: Deserialize<'de>> Deserialize<'de> for Entries<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor<T>(std::marker::PhantomData<T>);
        impl<'de, T: Deserialize<'de>> serde::de::Visitor<'de> for Visitor<T> {
            type Value = Entries<T>;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a keyed object")
            }
            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
                let mut entries = Vec::new();
                while let Some(entry) = map.next_entry()? {
                    entries.push(entry);
                }
                Ok(Entries(entries))
            }
        }
        deserializer.deserialize_map(Visitor(std::marker::PhantomData))
    }
}

fn tensor_dtype(key: &str, dtype: safetensors::Dtype) -> Result<Dtype, WeightError> {
    use safetensors::Dtype as Stored;
    Ok(match dtype {
        Stored::BOOL => Dtype::Bool,
        Stored::U8 => Dtype::Uint8,
        Stored::I8 => Dtype::Int8,
        Stored::U16 => Dtype::Uint16,
        Stored::I16 => Dtype::Int16,
        Stored::U32 => Dtype::Uint32,
        Stored::I32 => Dtype::Int32,
        Stored::U64 => Dtype::Uint64,
        Stored::I64 => Dtype::Int64,
        Stored::F16 => Dtype::Float16,
        Stored::BF16 => Dtype::Bfloat16,
        Stored::F32 => Dtype::Float32,
        _ => {
            return Err(WeightError::UnsupportedDtype {
                key: key.to_owned(),
                dtype: format!("{dtype:?}"),
            })
        }
    })
}

fn checked_shape(key: &str, shape: &[usize]) -> Result<Vec<i32>, WeightError> {
    i32::try_from(shape.len())
        .map_err(|_| WeightError::UnsupportedFormat(format!("rank exceeds i32 for {key}")))?;
    shape
        .iter()
        .map(|&dimension| {
            i32::try_from(dimension).map_err(|_| {
                WeightError::UnsupportedFormat(format!("dimension exceeds i32 for {key}"))
            })
        })
        .collect()
}

fn validate_shape(key: &str, expected: &[usize], actual: &[usize]) -> Result<(), WeightError> {
    if expected != actual {
        return Err(WeightError::ShapeMismatch {
            key: key.to_owned(),
            expected: expected.to_owned(),
            actual: actual.to_owned(),
        });
    }
    Ok(())
}

fn validate_dtype(key: &str, expected: Dtype, actual: Dtype) -> Result<(), WeightError> {
    if expected != actual {
        return Err(WeightError::DtypeMismatch {
            key: key.to_owned(),
            expected,
            actual,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
