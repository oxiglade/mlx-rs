//! GGUF container loading, inspection, construction, and saving.

use std::{
    collections::HashMap,
    ffi::{CStr, CString},
    fmt,
    marker::PhantomData,
    path::Path,
    rc::Rc,
};

use crate::{
    error::Exception,
    utils::{
        guard::{Guard, MaybeUninitArray},
        SUCCESS,
    },
    Array, Dtype, Stream,
};

const NOT_FOUND: i32 = 2;
const WRONG_METADATA_KIND: i32 = 3;

/// A value stored in the GGUF metadata namespace.
///
/// ```rust
/// use mlx_rs::{io::{GgufMetadataKind, GgufMetadataValue}, Array};
///
/// let value = GgufMetadataValue::from(Array::from_int(7));
/// assert_eq!(value.kind(), GgufMetadataKind::Array);
/// ```
#[derive(Debug, Clone)]
pub enum GgufMetadataValue {
    /// A scalar or one-dimensional metadata array.
    Array(Array),
    /// A single UTF-8 string.
    String(String),
    /// A list of UTF-8 strings.
    Strings(Vec<String>),
}

/// The kind of a value in the GGUF metadata namespace.
///
/// ```rust
/// use mlx_rs::io::GgufMetadataKind;
///
/// assert_ne!(GgufMetadataKind::String, GgufMetadataKind::Strings);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufMetadataKind {
    /// An MLX array.
    Array,
    /// A single string.
    String,
    /// A list of strings.
    Strings,
}

/// An error from GGUF validation or an upstream MLX operation.
///
/// Stable Rust-side validation failures have dedicated variants. Other native runtime failures
/// remain opaque in [`GgufError::Exception`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum GgufError {
    /// The load path is not an existing local file.
    #[error("path must point to a local file")]
    NotFile,

    /// The path cannot be represented by the C string ABI.
    #[error("path is not valid UTF-8")]
    InvalidPathUtf8,

    /// The path does not end in `.gguf`.
    #[error("path must have a .gguf extension")]
    UnsupportedExtension,

    /// A path, key, or value contains an interior null byte.
    #[error("GGUF text contains an interior null byte")]
    InteriorNul,

    /// Text returned by the native API is not valid UTF-8.
    #[error("GGUF text is not valid UTF-8")]
    InvalidUtf8,

    /// The array namespace already contains this key.
    #[error("array key {key:?} already exists")]
    ArrayKeyAlreadyExists {
        /// The duplicate array key.
        key: String,
    },

    /// The metadata namespace already contains this key.
    #[error("metadata key {key:?} already exists")]
    MetadataKeyAlreadyExists {
        /// The duplicate metadata key.
        key: String,
    },

    /// A typed getter was used for a different metadata kind.
    #[error("metadata {key:?} has kind {actual:?}, expected {expected:?}")]
    WrongMetadataKind {
        /// The metadata key.
        key: String,
        /// The kind requested by the getter.
        expected: GgufMetadataKind,
        /// The kind stored in the container.
        actual: GgufMetadataKind,
    },

    /// The tensor dtype is not supported by the MLX 0.32.2 GGUF writer.
    #[error("tensor dtype {dtype:?} cannot be written as GGUF")]
    UnsupportedTensorDtype {
        /// The rejected dtype.
        dtype: Dtype,
    },

    /// The metadata-array dtype is not supported by the MLX 0.32.2 GGUF writer.
    #[error("metadata array dtype {dtype:?} cannot be written as GGUF")]
    UnsupportedMetadataArrayDtype {
        /// The rejected dtype.
        dtype: Dtype,
    },

    /// Metadata arrays may only be scalar or one-dimensional.
    #[error("metadata arrays must be scalar or one-dimensional, found rank {rank}")]
    InvalidMetadataArrayRank {
        /// The rejected array rank.
        rank: usize,
    },

    /// Metadata arrays must contain at least one element.
    #[error("metadata arrays cannot be empty")]
    EmptyMetadataArray,

    /// An opaque error reported by MLX.
    #[error(transparent)]
    Exception(#[from] Exception),
}

/// A live two-namespace GGUF container.
///
/// Array keys can be enumerated, while the current C ABI does not expose metadata-key
/// enumeration. The handle is intentionally non-cloneable and thread-affine.
///
/// ```rust
/// use mlx_rs::io::GgufFile;
///
/// let file = GgufFile::new()?;
/// assert!(file.array_keys()?.is_empty());
/// # Ok::<(), mlx_rs::io::GgufError>(())
/// ```
pub struct GgufFile {
    handle: GgufHandle,
    thread_affinity: PhantomData<Rc<()>>,
}

impl fmt::Debug for GgufFile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("GgufFile").finish_non_exhaustive()
    }
}

impl GgufMetadataValue {
    /// Returns the value's metadata kind.
    pub fn kind(&self) -> GgufMetadataKind {
        match self {
            Self::Array(_) => GgufMetadataKind::Array,
            Self::String(_) => GgufMetadataKind::String,
            Self::Strings(_) => GgufMetadataKind::Strings,
        }
    }
}

impl From<Array> for GgufMetadataValue {
    fn from(value: Array) -> Self {
        Self::Array(value)
    }
}

impl From<&Array> for GgufMetadataValue {
    fn from(value: &Array) -> Self {
        Self::Array(value.clone())
    }
}

impl From<String> for GgufMetadataValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<&str> for GgufMetadataValue {
    fn from(value: &str) -> Self {
        Self::String(value.to_owned())
    }
}

impl From<Vec<String>> for GgufMetadataValue {
    fn from(value: Vec<String>) -> Self {
        Self::Strings(value)
    }
}

impl GgufFile {
    /// Creates an empty GGUF container.
    pub fn new() -> Result<Self, GgufError> {
        install_error_handler();
        let handle = GgufHandle::new()?;
        Ok(Self {
            handle,
            thread_affinity: PhantomData,
        })
    }

    /// Loads a GGUF file on the thread-local stream, falling back to CPU.
    ///
    /// ```no_run
    /// use mlx_rs::io::GgufFile;
    ///
    /// let file = GgufFile::load("model.gguf")?;
    /// println!("{} tensors", file.array_keys()?.len());
    /// # Ok::<(), mlx_rs::io::GgufError>(())
    /// ```
    pub fn load(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(GgufError::NotFile);
        }
        let path = path_c_string(path)?;
        let mut file = Self::new()?;
        let stream = Stream::thread_local_or_cpu();
        let status =
            unsafe { mlx_sys::mlx_load_gguf(&mut file.handle.raw, path.as_ptr(), stream.as_ptr()) };
        status_result(status, "loading a GGUF file")?;
        Ok(file)
    }

    /// Saves the container to a `.gguf` path.
    ///
    /// Saving evaluates contained arrays and may allocate row-major materializations. It uses the
    /// GGUF writer directly and does not consult the ambient Rust stream.
    ///
    /// ```no_run
    /// use mlx_rs::{io::GgufFile, Array};
    ///
    /// let mut file = GgufFile::new()?;
    /// file.insert_array("weight", &Array::from_f32(1.0))?;
    /// file.save("model.gguf")?;
    /// # Ok::<(), mlx_rs::io::GgufError>(())
    /// ```
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), GgufError> {
        for array in self.arrays()?.into_values() {
            validate_tensor_dtype(array.dtype())?;
        }
        let path = path_c_string(path.as_ref())?;
        let status = unsafe { mlx_sys::mlx_save_gguf(path.as_ptr(), self.handle.raw) };
        status_result(status, "saving a GGUF file")
    }

    /// Returns array keys in deterministic lexical order.
    pub fn array_keys(&self) -> Result<Vec<String>, GgufError> {
        let mut keys = VectorStringGuard::new()?;
        let status = unsafe { mlx_sys::mlx_io_gguf_get_keys(&mut keys.raw, self.handle.raw) };
        status_result(status, "enumerating GGUF array keys")?;
        let mut result = keys.to_vec()?;
        result.sort();
        Ok(result)
    }

    /// Returns a new independently owned MLX array handle for `key`.
    pub fn get_array(&self, key: &str) -> Result<Option<Array>, GgufError> {
        let key = text_c_string(key)?;
        let mut output = MaybeUninitArray::new();
        let status = unsafe {
            mlx_sys::mlx_io_gguf_get_array(output.as_mut_raw_ptr(), self.handle.raw, key.as_ptr())
        };
        match status {
            SUCCESS => {
                output.set_init_success(true);
                Ok(Some(output.try_into_guarded()?))
            }
            NOT_FOUND => Ok(None),
            _ => Err(status_exception(status, "getting a GGUF array")),
        }
    }

    /// Copies the array namespace into independently owned MLX handles.
    pub fn arrays(&self) -> Result<HashMap<String, Array>, GgufError> {
        self.array_keys()?
            .into_iter()
            .map(|key| {
                self.get_array(&key)?
                    .map(|array| (key, array))
                    .ok_or_else(|| status_exception(NOT_FOUND, "reading an enumerated GGUF array"))
            })
            .collect()
    }

    /// Inserts an array, rejecting duplicate keys and unsupported writer dtypes.
    pub fn insert_array(&mut self, key: impl AsRef<str>, value: &Array) -> Result<(), GgufError> {
        let key_text = key.as_ref();
        if self.get_array(key_text)?.is_some() {
            return Err(GgufError::ArrayKeyAlreadyExists {
                key: key_text.to_owned(),
            });
        }
        validate_tensor_dtype(value.dtype())?;
        let key = text_c_string(key_text)?;
        let status = unsafe {
            mlx_sys::mlx_io_gguf_set_array(self.handle.raw, key.as_ptr(), value.as_ptr())
        };
        status_result(status, "inserting a GGUF array")
    }

    /// Returns the kind of a metadata value, or `None` when the key is absent.
    pub fn metadata_kind(&self, key: &str) -> Result<Option<GgufMetadataKind>, GgufError> {
        let key = text_c_string(key)?;
        let probes = [
            (
                GgufMetadataKind::Array,
                mlx_sys::mlx_io_gguf_has_metadata_array
                    as unsafe extern "C" fn(
                        *mut bool,
                        mlx_sys::mlx_io_gguf,
                        *const std::ffi::c_char,
                    ) -> i32,
            ),
            (
                GgufMetadataKind::String,
                mlx_sys::mlx_io_gguf_has_metadata_string,
            ),
            (
                GgufMetadataKind::Strings,
                mlx_sys::mlx_io_gguf_has_metadata_vector_string,
            ),
        ];
        for (kind, probe) in probes {
            let mut present = false;
            let status = unsafe { probe(&mut present, self.handle.raw, key.as_ptr()) };
            match status {
                SUCCESS if present => return Ok(Some(kind)),
                SUCCESS => {}
                NOT_FOUND => return Ok(None),
                _ => return Err(status_exception(status, "checking GGUF metadata kind")),
            }
        }
        Err(status_exception(
            WRONG_METADATA_KIND,
            "identifying GGUF metadata kind",
        ))
    }

    /// Returns a metadata value, or `None` when the key is absent.
    pub fn get_metadata(&self, key: &str) -> Result<Option<GgufMetadataValue>, GgufError> {
        match self.metadata_kind(key)? {
            Some(GgufMetadataKind::Array) => Ok(self.get_metadata_array(key)?.map(Into::into)),
            Some(GgufMetadataKind::String) => Ok(self.get_metadata_string(key)?.map(Into::into)),
            Some(GgufMetadataKind::Strings) => Ok(self.get_metadata_strings(key)?.map(Into::into)),
            None => Ok(None),
        }
    }

    /// Returns array metadata, or `None` when the key is absent.
    pub fn get_metadata_array(&self, key: &str) -> Result<Option<Array>, GgufError> {
        let c_key = text_c_string(key)?;
        let mut output = MaybeUninitArray::new();
        let status = unsafe {
            mlx_sys::mlx_io_gguf_get_metadata_array(
                output.as_mut_raw_ptr(),
                self.handle.raw,
                c_key.as_ptr(),
            )
        };
        match status {
            SUCCESS => {
                output.set_init_success(true);
                Ok(Some(output.try_into_guarded()?))
            }
            NOT_FOUND => Ok(None),
            WRONG_METADATA_KIND => Err(self.wrong_kind(key, GgufMetadataKind::Array)?),
            _ => Err(status_exception(status, "getting GGUF array metadata")),
        }
    }

    /// Returns string metadata, or `None` when the key is absent.
    pub fn get_metadata_string(&self, key: &str) -> Result<Option<String>, GgufError> {
        let c_key = text_c_string(key)?;
        let mut output = StringGuard::new();
        let status = unsafe {
            mlx_sys::mlx_io_gguf_get_metadata_string(
                &mut output.raw,
                self.handle.raw,
                c_key.as_ptr(),
            )
        };
        match status {
            SUCCESS => Ok(Some(output.to_string()?)),
            NOT_FOUND => Ok(None),
            WRONG_METADATA_KIND => Err(self.wrong_kind(key, GgufMetadataKind::String)?),
            _ => Err(status_exception(status, "getting GGUF string metadata")),
        }
    }

    /// Returns string-list metadata, or `None` when the key is absent.
    pub fn get_metadata_strings(&self, key: &str) -> Result<Option<Vec<String>>, GgufError> {
        let c_key = text_c_string(key)?;
        let mut output = VectorStringGuard::new()?;
        let status = unsafe {
            mlx_sys::mlx_io_gguf_get_metadata_vector_string(
                &mut output.raw,
                self.handle.raw,
                c_key.as_ptr(),
            )
        };
        match status {
            SUCCESS => Ok(Some(output.to_vec()?)),
            NOT_FOUND => Ok(None),
            WRONG_METADATA_KIND => Err(self.wrong_kind(key, GgufMetadataKind::Strings)?),
            _ => Err(status_exception(
                status,
                "getting GGUF string-list metadata",
            )),
        }
    }

    /// Inserts metadata, rejecting duplicate keys and unsupported metadata arrays.
    pub fn insert_metadata<V>(&mut self, key: impl AsRef<str>, value: V) -> Result<(), GgufError>
    where
        V: Into<GgufMetadataValue>,
    {
        let value = value.into();
        let key_text = key.as_ref();
        if self.metadata_kind(key_text)?.is_some() {
            return Err(GgufError::MetadataKeyAlreadyExists {
                key: key_text.to_owned(),
            });
        }
        if let GgufMetadataValue::Array(array) = &value {
            validate_metadata_array(array)?;
        }
        let key = text_c_string(key_text)?;
        match value {
            GgufMetadataValue::Array(array) => {
                let status = unsafe {
                    mlx_sys::mlx_io_gguf_set_metadata_array(
                        self.handle.raw,
                        key.as_ptr(),
                        array.as_ptr(),
                    )
                };
                status_result(status, "inserting GGUF array metadata")
            }
            GgufMetadataValue::String(value) => {
                let value = text_c_string(&value)?;
                let status = unsafe {
                    mlx_sys::mlx_io_gguf_set_metadata_string(
                        self.handle.raw,
                        key.as_ptr(),
                        value.as_ptr(),
                    )
                };
                status_result(status, "inserting GGUF string metadata")
            }
            GgufMetadataValue::Strings(values) => {
                let strings = values
                    .iter()
                    .map(|value| text_c_string(value))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut pointers = strings
                    .iter()
                    .map(|value| value.as_ptr())
                    .collect::<Vec<_>>();
                let vector = VectorStringGuard::from_data(&mut pointers)?;
                let status = unsafe {
                    mlx_sys::mlx_io_gguf_set_metadata_vector_string(
                        self.handle.raw,
                        key.as_ptr(),
                        vector.raw,
                    )
                };
                status_result(status, "inserting GGUF string-list metadata")
            }
        }
    }

    fn wrong_kind(&self, key: &str, expected: GgufMetadataKind) -> Result<GgufError, GgufError> {
        let actual = self
            .metadata_kind(key)?
            .ok_or_else(|| status_exception(NOT_FOUND, "resolving a present GGUF metadata key"))?;
        Ok(GgufError::WrongMetadataKind {
            key: key.to_owned(),
            expected,
            actual,
        })
    }
}

struct GgufHandle {
    raw: mlx_sys::mlx_io_gguf,
}

impl GgufHandle {
    fn new() -> Result<Self, GgufError> {
        let raw = unsafe { mlx_sys::mlx_io_gguf_new() };
        if raw.ctx.is_null() {
            Err(status_exception(SUCCESS + 1, "creating a GGUF container"))
        } else {
            Ok(Self { raw })
        }
    }
}

impl Drop for GgufHandle {
    fn drop(&mut self) {
        let _ = unsafe { mlx_sys::mlx_io_gguf_free(self.raw) };
    }
}

struct StringGuard {
    raw: mlx_sys::mlx_string,
}

impl StringGuard {
    fn new() -> Self {
        // mlx_string_new returns an empty handle whose ctx stays null until a
        // successful getter fills it through mlx_string_set_.
        let raw = unsafe { mlx_sys::mlx_string_new() };
        Self { raw }
    }

    fn to_string(&self) -> Result<String, GgufError> {
        let data = unsafe { mlx_sys::mlx_string_data(self.raw) };
        if data.is_null() {
            return Err(status_exception(SUCCESS + 1, "reading an MLX string"));
        }
        unsafe { CStr::from_ptr(data) }
            .to_str()
            .map(str::to_owned)
            .map_err(|_| GgufError::InvalidUtf8)
    }
}

impl Drop for StringGuard {
    fn drop(&mut self) {
        let _ = unsafe { mlx_sys::mlx_string_free(self.raw) };
    }
}

struct VectorStringGuard {
    raw: mlx_sys::mlx_vector_string,
}

impl VectorStringGuard {
    fn new() -> Result<Self, GgufError> {
        let raw = unsafe { mlx_sys::mlx_vector_string_new() };
        Self::from_raw(raw, "creating an MLX string vector")
    }

    fn from_data(data: &mut [*const std::ffi::c_char]) -> Result<Self, GgufError> {
        let raw = unsafe { mlx_sys::mlx_vector_string_new_data(data.as_mut_ptr(), data.len()) };
        Self::from_raw(raw, "creating an MLX string vector from data")
    }

    fn from_raw(raw: mlx_sys::mlx_vector_string, operation: &str) -> Result<Self, GgufError> {
        if raw.ctx.is_null() {
            Err(status_exception(SUCCESS + 1, operation))
        } else {
            Ok(Self { raw })
        }
    }

    fn to_vec(&self) -> Result<Vec<String>, GgufError> {
        let len = unsafe { mlx_sys::mlx_vector_string_size(self.raw) };
        (0..len)
            .map(|index| {
                let mut data = std::ptr::null_mut();
                let status = unsafe { mlx_sys::mlx_vector_string_get(&mut data, self.raw, index) };
                status_result(status, "reading an MLX string vector")?;
                if data.is_null() {
                    return Err(status_exception(
                        SUCCESS + 1,
                        "reading an MLX string vector",
                    ));
                }
                unsafe { CStr::from_ptr(data) }
                    .to_str()
                    .map(str::to_owned)
                    .map_err(|_| GgufError::InvalidUtf8)
            })
            .collect()
    }
}

impl Drop for VectorStringGuard {
    fn drop(&mut self) {
        let _ = unsafe { mlx_sys::mlx_vector_string_free(self.raw) };
    }
}

fn install_error_handler() {
    crate::error::INIT_ERR_HANDLER.call_once(crate::error::setup_mlx_error_handler);
}

fn status_exception(status: i32, operation: &str) -> GgufError {
    GgufError::Exception(crate::error::exception_from_status(status, operation))
}

fn status_result(status: i32, operation: &str) -> Result<(), GgufError> {
    if status == SUCCESS {
        Ok(())
    } else {
        Err(status_exception(status, operation))
    }
}

fn text_c_string(value: &str) -> Result<CString, GgufError> {
    CString::new(value).map_err(|_| GgufError::InteriorNul)
}

fn path_c_string(path: &Path) -> Result<CString, GgufError> {
    if path.extension().and_then(|extension| extension.to_str()) != Some("gguf") {
        return Err(GgufError::UnsupportedExtension);
    }
    let path = path.to_str().ok_or(GgufError::InvalidPathUtf8)?;
    text_c_string(path)
}

fn validate_tensor_dtype(dtype: Dtype) -> Result<(), GgufError> {
    if matches!(
        dtype,
        Dtype::Float32 | Dtype::Float16 | Dtype::Int8 | Dtype::Int16 | Dtype::Int32
    ) {
        Ok(())
    } else {
        Err(GgufError::UnsupportedTensorDtype { dtype })
    }
}

fn validate_metadata_array(array: &Array) -> Result<(), GgufError> {
    let rank = array.ndim();
    if rank > 1 {
        return Err(GgufError::InvalidMetadataArrayRank { rank });
    }
    if array.size() == 0 {
        return Err(GgufError::EmptyMetadataArray);
    }
    let dtype = array.dtype();
    if matches!(
        dtype,
        Dtype::Bool
            | Dtype::Int8
            | Dtype::Int16
            | Dtype::Int32
            | Dtype::Int64
            | Dtype::Uint8
            | Dtype::Uint16
            | Dtype::Uint32
            | Dtype::Uint64
            | Dtype::Float32
    ) {
        Ok(())
    } else {
        Err(GgufError::UnsupportedMetadataArrayDtype { dtype })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writer_prevalidation_rejects_unsupported_tensor_dtypes() {
        for dtype in [
            Dtype::Bool,
            Dtype::Uint8,
            Dtype::Uint16,
            Dtype::Uint32,
            Dtype::Uint64,
            Dtype::Int64,
            Dtype::Bfloat16,
            Dtype::Complex64,
        ] {
            assert!(matches!(
                validate_tensor_dtype(dtype),
                Err(GgufError::UnsupportedTensorDtype { dtype: actual }) if actual == dtype
            ));
        }
    }

    #[test]
    fn metadata_prevalidation_rejects_rank_empty_and_dtype() {
        let rank = Array::from_slice(&[1_i32, 2, 3, 4], &[2, 2]);
        assert!(matches!(
            validate_metadata_array(&rank),
            Err(GgufError::InvalidMetadataArrayRank { rank: 2 })
        ));

        let empty = Array::from_slice::<i32>(&[], &[0]);
        assert!(matches!(
            validate_metadata_array(&empty),
            Err(GgufError::EmptyMetadataArray)
        ));

        for array in [
            Array::from_slice(&[half::f16::from_f32(1.0)], &[1]),
            Array::from_slice(&[half::bf16::from_f32(1.0)], &[1]),
            Array::from_complex(crate::complex64::new(1.0, 0.0)),
        ] {
            assert!(matches!(
                validate_metadata_array(&array),
                Err(GgufError::UnsupportedMetadataArrayDtype { .. })
            ));
        }
    }

    #[test]
    fn duplicate_keys_are_rejected_per_namespace() {
        let mut file = GgufFile::new().unwrap();
        let value = Array::from_int(1);
        file.insert_array("same", &value).unwrap();
        assert!(matches!(
            file.insert_array("same", &value),
            Err(GgufError::ArrayKeyAlreadyExists { key }) if key == "same"
        ));

        file.insert_metadata("same", "first").unwrap();
        assert!(matches!(
            file.insert_metadata("same", vec!["second".to_owned()]),
            Err(GgufError::MetadataKeyAlreadyExists { key }) if key == "same"
        ));
    }

    #[test]
    fn paths_and_text_are_validated_before_ffi_entry() {
        assert!(matches!(
            GgufFile::load("definitely-absent.gguf"),
            Err(GgufError::NotFile)
        ));
        let file = GgufFile::new().unwrap();
        assert!(matches!(
            file.save("wrong-extension.bin"),
            Err(GgufError::UnsupportedExtension)
        ));
        let mut file = GgufFile::new().unwrap();
        assert!(matches!(
            file.insert_array("bad\0key", &Array::from_int(1)),
            Err(GgufError::InteriorNul)
        ));
    }
}
