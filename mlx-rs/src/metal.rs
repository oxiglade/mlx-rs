//! Metal-specific runtime configuration.

use std::ffi::{CStr, CString};

use crate::{
    error::{Exception, Result},
    utils::SUCCESS,
};

struct StringHandle {
    raw: mlx_sys::mlx_string,
}

impl StringHandle {
    fn new() -> Self {
        Self {
            raw: unsafe { mlx_sys::mlx_string_new() },
        }
    }

    fn to_string(&self) -> Result<String> {
        let data = unsafe { mlx_sys::mlx_string_data(self.raw) };
        if data.is_null() {
            return Err(Exception::custom("MLX returned an empty string handle"));
        }
        unsafe { CStr::from_ptr(data) }
            .to_str()
            .map(str::to_owned)
            .map_err(|error| Exception::custom(error.to_string()))
    }
}

impl Drop for StringHandle {
    fn drop(&mut self) {
        let _ = unsafe { mlx_sys::mlx_string_free(self.raw) };
    }
}

fn install_error_handler() {
    crate::error::INIT_ERR_HANDLER.call_once(crate::error::setup_mlx_error_handler);
}

fn status_result(status: i32, operation: &str) -> Result<()> {
    if status == SUCCESS {
        Ok(())
    } else {
        Err(crate::error::exception_from_status(status, operation))
    }
}

/// Returns the path used to load the default Metal library.
pub fn metallib_path() -> Result<String> {
    install_error_handler();
    let mut path = StringHandle::new();
    let status = unsafe { mlx_sys::mlx_metal_get_metallib_path(&mut path.raw) };
    status_result(status, "reading the Metal library path")?;
    path.to_string()
}

/// Sets the path used by subsequent Metal initialization.
///
/// This changes process-global MLX state and can affect every thread that initializes Metal
/// afterward.
pub fn set_metallib_path(path: impl AsRef<str>) -> Result<()> {
    install_error_handler();
    let path = CString::new(path.as_ref()).map_err(|error| Exception::custom(error.to_string()))?;
    let status = unsafe { mlx_sys::mlx_metal_set_metallib_path(path.as_ptr()) };
    status_result(status, "setting the Metal library path")
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::{metallib_path, set_metallib_path};

    static METALLIB_PATH: Mutex<()> = Mutex::new(());

    struct RestorePath(String);

    impl Drop for RestorePath {
        fn drop(&mut self) {
            set_metallib_path(&self.0).expect("restore metallib path");
        }
    }

    #[test]
    fn metallib_path_is_non_empty() {
        let _guard = METALLIB_PATH.lock().expect("lock metallib path");
        let original = metallib_path().unwrap();
        let _restore = RestorePath(original);
        set_metallib_path("mlx-rs-metallib-path-test").unwrap();

        assert!(!metallib_path().unwrap().is_empty());
    }

    #[test]
    fn metallib_path_roundtrips() {
        let _guard = METALLIB_PATH.lock().expect("lock metallib path");
        let original = metallib_path().unwrap();
        let _restore = RestorePath(original.clone());
        let temporary = format!("{original}.mlx-rs-roundtrip");

        set_metallib_path(&temporary).unwrap();

        assert_eq!(metallib_path().unwrap(), temporary);
    }
}
