//! Support shared by the Metal and CUDA custom kernels.

use std::ffi::CString;

use crate::error::{Exception, Result};
use crate::utils::SUCCESS;
use crate::Dtype;

/// A compile-time parameter for a custom kernel (`MetalKernel` or `CudaKernel`).
///
/// The kernel source refers to each argument by name (for example `T`, `N` or `USE_BIAS`).
/// Metal emits them as template parameters of the generated `[[kernel]]` function; CUDA emits
/// them as `using` aliases and `constexpr` constants at the top of the generated body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateArg {
    /// A type parameter, emitted as the matching backend type (for example `float`).
    Dtype(Dtype),
    /// An `int` parameter.
    Int(i32),
    /// A `bool` parameter.
    Bool(bool),
}

pub(super) struct StringVector {
    pub(super) raw: mlx_sys::mlx_vector_string,
}

impl StringVector {
    pub(super) fn new(values: &[&str]) -> Result<Self> {
        let owned = values
            .iter()
            .map(|value| c_string(value))
            .collect::<Result<Vec<_>>>()?;
        let mut pointers: Vec<_> = owned.iter().map(|value| value.as_ptr()).collect();
        // mlx-c copies each string into the vector, so `owned` only needs to outlive this call.
        let raw =
            unsafe { mlx_sys::mlx_vector_string_new_data(pointers.as_mut_ptr(), pointers.len()) };
        if raw.ctx.is_null() {
            return Err(crate::error::exception_from_status(
                SUCCESS + 1,
                "creating an MLX string vector",
            ));
        }
        Ok(Self { raw })
    }
}

impl Drop for StringVector {
    fn drop(&mut self) {
        let _ = unsafe { mlx_sys::mlx_vector_string_free(self.raw) };
    }
}

pub(super) fn c_string(value: &str) -> Result<CString> {
    CString::new(value).map_err(|error| Exception::custom(error.to_string()))
}
