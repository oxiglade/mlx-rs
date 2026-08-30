use crate::error::{Exception, IoError};
use crate::utils::SUCCESS;
use crate::{Array, Stream};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr::null_mut;

use super::Guarded;

pub(crate) struct SafeTensors {
    pub(crate) c_data: mlx_sys::mlx_map_string_to_array,
    pub(crate) c_metadata: mlx_sys::mlx_map_string_to_string,
}

struct ArrayMapIterator(mlx_sys::mlx_map_string_to_array_iterator);

impl ArrayMapIterator {
    unsafe fn new(map: mlx_sys::mlx_map_string_to_array) -> Self {
        Self(unsafe { mlx_sys::mlx_map_string_to_array_iterator_new(map) })
    }
}

impl Drop for ArrayMapIterator {
    fn drop(&mut self) {
        let status = unsafe { mlx_sys::mlx_map_string_to_array_iterator_free(self.0) };
        debug_assert_eq!(status, SUCCESS);
    }
}

struct StringMapIterator(mlx_sys::mlx_map_string_to_string_iterator);

impl StringMapIterator {
    unsafe fn new(map: mlx_sys::mlx_map_string_to_string) -> Self {
        Self(unsafe { mlx_sys::mlx_map_string_to_string_iterator_new(map) })
    }
}

impl Drop for StringMapIterator {
    fn drop(&mut self) {
        let status = unsafe { mlx_sys::mlx_map_string_to_string_iterator_free(self.0) };
        debug_assert_eq!(status, SUCCESS);
    }
}

impl Drop for SafeTensors {
    fn drop(&mut self) {
        unsafe {
            mlx_sys::mlx_map_string_to_string_free(self.c_metadata);
            mlx_sys::mlx_map_string_to_array_free(self.c_data);
        }
    }
}

impl SafeTensors {
    pub(crate) fn load_device(path: &Path, stream: impl AsRef<Stream>) -> Result<Self, IoError> {
        if !path.is_file() {
            return Err(IoError::NotFile);
        }

        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or(IoError::UnsupportedFormat)?;

        if extension != "safetensors" {
            return Err(IoError::UnsupportedFormat);
        }

        let path_str = path.to_str().ok_or(IoError::InvalidUtf8)?;
        let filepath = CString::new(path_str)?;

        SafeTensors::try_from_op(|(res_0, res_1)| unsafe {
            mlx_sys::mlx_load_safetensors(res_0, res_1, filepath.as_ptr(), stream.as_ref().as_ptr())
        })
        .map_err(Into::into)
    }

    pub(crate) fn data(&self) -> Result<HashMap<String, Array>, Exception> {
        crate::error::INIT_ERR_HANDLER
            .with(|init| init.call_once(crate::error::setup_mlx_error_handler));
        let iterator = unsafe { ArrayMapIterator::new(self.c_data) };
        Self::data_from_iterator(&iterator)
    }

    fn data_from_iterator(
        iterator: &ArrayMapIterator,
    ) -> Result<HashMap<String, Array>, Exception> {
        let mut map = HashMap::new();
        unsafe {
            loop {
                let mut key_ptr: *const ::std::os::raw::c_char = null_mut();
                let mut value = mlx_sys::mlx_array_new();
                let status = mlx_sys::mlx_map_string_to_array_iterator_next(
                    &mut key_ptr as *mut *const _,
                    &mut value,
                    iterator.0,
                );

                match status {
                    SUCCESS => {
                        let key = CStr::from_ptr(key_ptr).to_string_lossy().into_owned();
                        let array = Array::from_ptr(value);
                        map.insert(key, array);
                    }
                    1 => {
                        mlx_sys::mlx_array_free(value);
                        return Err(crate::error::exception_from_status(
                            status,
                            "advancing an array map iterator",
                        ));
                    }
                    2 => {
                        mlx_sys::mlx_array_free(value);
                        break;
                    }
                    _ => unreachable!(),
                }
            }
        }

        Ok(map)
    }

    pub(crate) fn metadata(&self) -> Result<HashMap<String, String>, Exception> {
        crate::error::INIT_ERR_HANDLER
            .with(|init| init.call_once(crate::error::setup_mlx_error_handler));

        let mut map = HashMap::new();
        unsafe {
            let iterator = StringMapIterator::new(self.c_metadata);

            let mut key: *const ::std::os::raw::c_char = null_mut();
            let mut value: *const ::std::os::raw::c_char = null_mut();
            loop {
                let status = mlx_sys::mlx_map_string_to_string_iterator_next(
                    &mut key as *mut *const _,
                    &mut value as *mut *const _,
                    iterator.0,
                );

                match status {
                    SUCCESS => {
                        let key = CStr::from_ptr(key).to_string_lossy().into_owned();
                        let value = CStr::from_ptr(value).to_string_lossy().into_owned();
                        map.insert(key, value);
                    }
                    1 => {
                        return Err(crate::error::exception_from_status(
                            status,
                            "advancing a metadata map iterator",
                        ));
                    }
                    2 => break,
                    _ => unreachable!(),
                }
            }
        }

        Ok(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_iterator_is_freed_after_next_error() {
        crate::error::INIT_ERR_HANDLER
            .with(|init| init.call_once(crate::error::setup_mlx_error_handler));

        for _ in 0..200 {
            let map = unsafe { mlx_sys::mlx_map_string_to_array_new() };
            let mut iterator = unsafe { ArrayMapIterator::new(map) };
            let context = iterator.0.ctx;
            iterator.0.ctx = null_mut();

            let result = SafeTensors::data_from_iterator(&iterator);
            iterator.0.ctx = context;
            assert!(result.is_err());
            drop(iterator);
            assert_eq!(
                unsafe { mlx_sys::mlx_map_string_to_array_free(map) },
                SUCCESS
            );
        }
    }
}
