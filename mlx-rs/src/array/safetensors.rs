//! Implement conversion from safetensors TensorView to Array
//! 
//! `F8_*` dtypes are not supported and will return an error.

use std::{ffi::c_void, mem::transmute};

use bytemuck::cast_slice;
use safetensors::tensor::TensorView;

use crate::{error::ConversionError, Dtype};

use super::Array;

impl<'data> TryFrom<TensorView<'data>> for Array {
    type Error = ConversionError;

    fn try_from(value: TensorView<'data>) -> Result<Self, Self::Error> {
        let dtype: Dtype = value.dtype().try_into()?;
        let shape = value.shape()
            .iter()
            .map(|x| *x as i32)
            .collect::<Vec<_>>();

        let data = value.data();

        unsafe {
            Ok(Array::from_raw_data(data.as_ptr() as *const c_void, &shape, dtype))
        }
    }
}

impl<'a> TryFrom<&'a Array> for TensorView<'a> {
    type Error = ConversionError;

    fn try_from(value: &'a Array) -> Result<Self, Self::Error> {
        let dtype: safetensors::tensor::Dtype = value.dtype().try_into()?;
        let shape = value.shape()
            .iter()
            .map(|x| *x as usize)
            .collect::<Vec<_>>();
        let data: &[u8] = unsafe {
            match value.dtype() {
                Dtype::Bool => {
                    let data = value.as_slice::<bool>();
                    cast_slice(data)
                },
                Dtype::Uint8 => {
                    let data = value.as_slice::<u8>();
                    cast_slice(data)
                },
                Dtype::Uint16 => {
                    let data = value.as_slice::<u16>();
                    cast_slice(data)
                },
                Dtype::Uint32 => {
                    let data = value.as_slice::<u32>();
                    cast_slice(data)
                },
                Dtype::Uint64 => {
                    let data = value.as_slice::<u64>();
                    cast_slice(data)
                },
                Dtype::Int8 => {
                    let data = value.as_slice::<i8>();
                    cast_slice(data)
                },
                Dtype::Int16 => {
                    let data = value.as_slice::<i16>();
                    cast_slice(data)
                },
                Dtype::Int32 => {
                    let data = value.as_slice::<i32>();
                    cast_slice(data)
                },
                Dtype::Int64 => {
                    let data = value.as_slice::<i64>();
                    cast_slice(data)
                },
                Dtype::Float16 => {
                    let data = value.as_slice::<half::f16>();
                    let bits: &[u16] = transmute(data);
                    cast_slice(bits)
                },
                Dtype::Float32 => {
                    let data = value.as_slice::<f32>();
                    cast_slice(data)
                },
                Dtype::Float64 => {
                    let data = value.as_slice::<f64>();
                    cast_slice(data)
                },
                Dtype::Bfloat16 => {
                    let data = value.as_slice::<half::bf16>();
                    let bits: &[u16] = transmute(data);
                    cast_slice(bits)
                },
                Dtype::Float64 => {
                    let data = value.as_slice::<f64>();
                    cast_slice(data)
                },
                Dtype::Complex64 => return Err(ConversionError::MlxDtype(Dtype::Complex64)),
            }
        };

        TensorView::new(dtype, shape, data)
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use safetensors::tensor::TensorView;

    use crate::{array, complex64, Array};

    // Helper macro to test conversion between Array and TensorView
    macro_rules! assert_conversion {
        ($arr:expr, $dtype:expr) => {
            let arr = $arr.as_dtype($dtype).unwrap();
            let tensor = TensorView::try_from(&arr).unwrap();
            let arr2 = Array::try_from(tensor).unwrap();

            assert_eq!(arr, arr2);
        };
    }

    #[test]
    fn test_conversion_bool() {
        let arr = array!([[true, false, true], [false, true, false]]);
        assert_conversion!(&arr, crate::Dtype::Bool);
    }

    #[test]
    fn test_conversion_uint8() {
        let arr = array!([[1, 2, 3], [4, 5, 6]]);
        assert_conversion!(&arr, crate::Dtype::Uint8);
    }

    #[test]
    fn test_conversion_uint16() {
        let arr = array!([[1, 2, 3], [4, 5, 6]]);
        assert_conversion!(&arr, crate::Dtype::Uint16);
    }

    #[test]
    fn test_conversion_uint32() {
        let arr = array!([[1, 2, 3], [4, 5, 6]]);
        assert_conversion!(&arr, crate::Dtype::Uint32);
    }

    #[test]
    fn test_conversion_uint64() {
        let arr = array!([[1, 2, 3], [4, 5, 6]]);
        assert_conversion!(&arr, crate::Dtype::Uint64);
    }

    #[test]
    fn test_conversion_int8() {
        let arr = array!([[1, 2, 3], [4, 5, 6]]);
        assert_conversion!(&arr, crate::Dtype::Int8);
    }

    #[test]
    fn test_conversion_int16() {
        let arr = array!([[1, 2, 3], [4, 5, 6]]);
        assert_conversion!(&arr, crate::Dtype::Int16);
    }

    #[test]
    fn test_conversion_int32() {
        let arr = array!([[1, 2, 3], [4, 5, 6]]);
        assert_conversion!(&arr, crate::Dtype::Int32);
    }

    #[test]
    fn test_conversion_int64() {
        let arr = array!([[1, 2, 3], [4, 5, 6]]);
        assert_conversion!(&arr, crate::Dtype::Int64);
    }

    #[test]
    fn test_conversion_float16() {
        let arr = array!([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert_conversion!(&arr, crate::Dtype::Float16);
    }

    #[test]
    fn test_conversion_float32() {
        let arr = array!([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert_conversion!(&arr, crate::Dtype::Float32);
    }

    #[test]
    fn test_conversion_bfloat16() {
        let arr = array!([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert_conversion!(&arr, crate::Dtype::Bfloat16);
    }

    #[test]
    fn test_conversion_complex64() {
        let arr = array!([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).as_type::<complex64>().unwrap();
        let tensor = TensorView::try_from(&arr);
        assert!(tensor.is_err());
    }
}