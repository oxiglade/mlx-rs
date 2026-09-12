pub use crate::error::WeightError;
use crate::{arch::ArchitectureFactory, config::ParameterPath};
use mlx_rs::{io::GgufFile, utils::StateProjection, Array};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

/// Deterministic external keys and their source tensor metadata.
pub(crate) struct WeightManifest {
    pub(crate) tensors: BTreeMap<String, WeightEntry>,
}
pub(crate) struct WeightEntry {
    pub(crate) shard: PathBuf,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: safetensors::Dtype,
}
/// An architecture's explicit disposition for an external tensor key.
pub(crate) enum WeightDisposition {
    Parameter(ParameterPath),
    Ignore { reason: &'static str },
    Reject,
}
impl WeightManifest {
    pub(crate) fn discover(_directory: &Path) -> Result<Self, WeightError> {
        Err(unsupported("safetensors discovery"))
    }
    pub(crate) fn from_safetensors(_path: &Path) -> Result<Self, WeightError> {
        Err(unsupported("single safetensors manifest"))
    }
    pub(crate) fn from_sharded_index(_path: &Path) -> Result<Self, WeightError> {
        Err(unsupported("sharded safetensors manifest"))
    }
    pub(crate) fn from_gguf(_file: &GgufFile) -> Result<Self, WeightError> {
        Err(unsupported("GGUF manifest"))
    }
    pub(crate) fn load_strict(
        &self,
        _factory: &dyn ArchitectureFactory,
        _projection: &mut StateProjection<'_>,
    ) -> Result<(), WeightError> {
        Err(unsupported("strict keyed weight assignment"))
    }
    pub(crate) fn read_tensor(&self, _external: &str) -> Result<Array, WeightError> {
        Err(unsupported("tensor materialization"))
    }
}
fn unsupported(operation: &'static str) -> WeightError {
    WeightError::UnsupportedFormat(crate::NotYetImplemented(operation).to_string())
}
