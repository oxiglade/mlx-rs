use crate::error::IoError;
use crate::utils::guard::Guarded;
use crate::utils::io::SafeTensors;
use crate::utils::SUCCESS;
use crate::{Array, Stream};
use std::collections::HashMap;
use std::ffi::CString;
use std::path::Path;
fn check_file_extension(path: &Path, expected: &str) -> Result<(), IoError> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext == expected => Ok(()),
        _ => Err(IoError::UnsupportedFormat),
    }
}

impl Array {
    /// Load array from a binary file in `.npy` format.
    ///
    /// # Params
    ///
    /// - path: path of file to load
    /// - stream: stream or device to evaluate on
    pub fn load_numpy(path: impl AsRef<Path>) -> Result<Array, IoError> {
        let stream = Stream::thread_local_or_cpu();
        let path = path.as_ref();
        if !path.is_file() {
            return Err(IoError::NotFile);
        }
        let c_path = CString::new(path.to_str().ok_or(IoError::InvalidUtf8)?)?;
        check_file_extension(path, "npy")?;

        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_load(res, c_path.as_ptr(), stream.as_ref().as_ptr())
        })
        .map_err(Into::into)
    }

    /// Compatibility shim for [`load_numpy`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `load_numpy`"
    )]
    pub fn load_numpy_device(
        path: impl AsRef<Path>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, IoError> {
        crate::with_stream(stream.as_ref(), || Self::load_numpy(path))
    }

    /// Load dictionary of ``MLXArray`` from a `safetensors` file.
    ///
    /// # Params
    ///
    /// - path: path of file to load
    /// - stream: stream or device to evaluate on
    ///
    pub fn load_safetensors(path: impl AsRef<Path>) -> Result<HashMap<String, Array>, IoError> {
        let stream = Stream::thread_local_or_cpu();
        let safetensors = SafeTensors::load_device(path.as_ref(), stream)?;
        let data = safetensors.data()?;
        Ok(data)
    }

    /// Compatibility shim for [`load_safetensors`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `load_safetensors`"
    )]
    pub fn load_safetensors_device(
        path: impl AsRef<Path>,
        stream: impl AsRef<Stream>,
    ) -> Result<HashMap<String, Array>, IoError> {
        crate::with_stream(stream.as_ref(), || Self::load_safetensors(path))
    }

    /// Load dictionary of ``MLXArray`` and metadata `[String:String]` from a `safetensors` file.
    ///
    /// # Params
    ///
    /// - path: path of file to load
    /// - stream: stream or device to evaluate on
    #[allow(clippy::type_complexity)]
    pub fn load_safetensors_with_metadata(
        path: impl AsRef<Path>,
    ) -> Result<(HashMap<String, Array>, HashMap<String, String>), IoError> {
        let stream = Stream::thread_local_or_cpu();
        let safetensors = SafeTensors::load_device(path.as_ref(), stream)?;
        let data = safetensors.data()?;
        let metadata = safetensors.metadata()?;

        Ok((data, metadata))
    }

    /// Compatibility shim for [`load_safetensors_with_metadata`].
    #[allow(clippy::type_complexity)]
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `load_safetensors_with_metadata`"
    )]
    pub fn load_safetensors_with_metadata_device(
        path: impl AsRef<Path>,
        stream: impl AsRef<Stream>,
    ) -> Result<(HashMap<String, Array>, HashMap<String, String>), IoError> {
        crate::with_stream(stream.as_ref(), || {
            Self::load_safetensors_with_metadata(path)
        })
    }

    /// Save array to a binary file in `.npy`format.
    ///
    /// # Params
    ///
    /// - array: array to save
    /// - url: URL of file to load
    pub fn save_numpy(&self, path: impl AsRef<Path>) -> Result<(), IoError> {
        let path = path.as_ref();
        check_file_extension(path, "npy")?;
        let c_path = CString::new(path.to_str().ok_or(IoError::InvalidUtf8)?)?;

        unsafe { mlx_sys::mlx_save(c_path.as_ptr(), self.as_ptr()) };

        Ok(())
    }

    /// Save dictionary of arrays in `safetensors` format.
    ///
    /// # Params
    ///
    /// - arrays: arrays to save
    /// - metadata: metadata to save
    /// - path: path of file to save
    /// - stream: stream or device to evaluate on
    pub fn save_safetensors<'a, I, S, V>(
        arrays: I,
        metadata: impl Into<Option<&'a HashMap<String, String>>>,
        path: impl AsRef<Path>,
    ) -> Result<(), IoError>
    where
        I: IntoIterator<Item = (S, V)>,
        S: AsRef<str>,
        V: AsRef<Array>,
    {
        crate::error::INIT_ERR_HANDLER
            .with(|init| init.call_once(crate::error::setup_mlx_error_handler));

        let path = path.as_ref();

        check_file_extension(path, "safetensors")?;

        let arrays = unsafe {
            let data = mlx_sys::mlx_map_string_to_array_new();
            for (key, array) in arrays.into_iter() {
                let key = CString::new(key.as_ref())?;

                let status = mlx_sys::mlx_map_string_to_array_insert(
                    data,
                    key.as_ptr(),
                    array.as_ref().as_ptr(),
                );

                if status != SUCCESS {
                    mlx_sys::mlx_map_string_to_array_free(data);
                    return Err(crate::error::exception_from_status(
                        status,
                        "inserting a safetensors array",
                    )
                    .into());
                }
            }
            data
        };

        let default_metadata = HashMap::new();
        let metadata_ref = metadata.into().unwrap_or(&default_metadata);

        let metadata = unsafe {
            let data = mlx_sys::mlx_map_string_to_string_new();
            for (key, value) in metadata_ref.iter() {
                let key = CString::new(key.as_str())?;
                let value = CString::new(value.as_str())?;

                let status =
                    mlx_sys::mlx_map_string_to_string_insert(data, key.as_ptr(), value.as_ptr());

                if status != SUCCESS {
                    mlx_sys::mlx_map_string_to_string_free(data);
                    return Err(crate::error::exception_from_status(
                        status,
                        "inserting safetensors metadata",
                    )
                    .into());
                }
            }
            data
        };

        let c_path = CString::new(path.to_str().ok_or(IoError::InvalidUtf8)?)?;

        unsafe {
            let status = mlx_sys::mlx_save_safetensors(c_path.as_ptr(), arrays, metadata);

            let last_error = match status {
                SUCCESS => None,
                _ => Some(crate::error::exception_from_status(
                    status,
                    "saving safetensors",
                )),
            };

            mlx_sys::mlx_map_string_to_array_free(arrays);
            mlx_sys::mlx_map_string_to_string_free(metadata);

            if let Some(error) = last_error {
                return Err(error.into());
            }
        };

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::Array;

    #[test]
    fn test_save_arrays() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().join("test.safetensors");

        let mut arrays = std::collections::HashMap::new();
        arrays.insert("foo".to_string(), Array::ones::<i32>(&[1, 2]).unwrap());
        arrays.insert("bar".to_string(), Array::zeros::<i32>(&[2, 1]).unwrap());

        Array::save_safetensors(&arrays, None, &path).unwrap();

        let loaded_arrays = Array::load_safetensors(&path).unwrap();

        // compare values
        let mut loaded_keys: Vec<_> = loaded_arrays.keys().cloned().collect();
        let mut original_keys: Vec<_> = arrays.keys().cloned().collect();
        loaded_keys.sort();
        original_keys.sort();
        assert_eq!(loaded_keys, original_keys);

        for key in loaded_keys {
            let loaded_array = loaded_arrays.get(&key).unwrap();
            let original_array = arrays.get(&key).unwrap();
            assert!(loaded_array
                .all_close(original_array, None, None, None)
                .unwrap());
        }
    }

    #[test]
    fn test_save_array() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().join("test.npy");

        let a = Array::ones::<i32>(&[2, 4]).unwrap();
        a.save_numpy(&path).unwrap();

        let b = Array::load_numpy(&path).unwrap();
        assert!(a.all_close(&b, None, None, None).unwrap());
    }
}
