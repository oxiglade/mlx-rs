use half::{bf16, f16};
use mlx_internal_macros::generate_test_cases;
use strum::EnumIter;

use crate::error::InexactDtypeError;

generate_test_cases! {
    /// Array element type
    #[derive(
        Debug,
        Clone,
        Copy,
        PartialEq,
        Eq,
        num_enum::IntoPrimitive,
        num_enum::TryFromPrimitive,
        EnumIter,
        Hash,
    )]
    #[repr(u32)]
    pub enum Dtype {
        /// bool
        Bool = mlx_sys::mlx_dtype__MLX_BOOL,

        /// u8
        Uint8 = mlx_sys::mlx_dtype__MLX_UINT8,

        /// u16
        Uint16 = mlx_sys::mlx_dtype__MLX_UINT16,

        /// u32
        Uint32 = mlx_sys::mlx_dtype__MLX_UINT32,

        /// u64
        Uint64 = mlx_sys::mlx_dtype__MLX_UINT64,

        /// i8
        Int8 = mlx_sys::mlx_dtype__MLX_INT8,

        /// i16
        Int16 = mlx_sys::mlx_dtype__MLX_INT16,

        /// i32
        Int32 = mlx_sys::mlx_dtype__MLX_INT32,

        /// i64
        Int64 = mlx_sys::mlx_dtype__MLX_INT64,

        /// f16
        Float16 = mlx_sys::mlx_dtype__MLX_FLOAT16,

        /// f32
        Float32 = mlx_sys::mlx_dtype__MLX_FLOAT32,

        /// f64
        Float64 = mlx_sys::mlx_dtype__MLX_FLOAT64,

        /// bfloat16
        Bfloat16 = mlx_sys::mlx_dtype__MLX_BFLOAT16,

        /// complex64
        Complex64 = mlx_sys::mlx_dtype__MLX_COMPLEX64,
    }
}

impl Dtype {
    /// Returns `true` if the data type is complex.
    pub fn is_complex(&self) -> bool {
        matches!(self, Dtype::Complex64)
    }

    /// Returns `true` if the data type is floating point.
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            Dtype::Float16 | Dtype::Float32 | Dtype::Float64 | Dtype::Bfloat16
        )
    }

    /// Returns `true` if the data type is one of `f16`, `f32`, `f64`, `bfloat16`, or `complex64`.
    pub fn is_inexact(&self) -> bool {
        matches!(
            self,
            Dtype::Float16 | Dtype::Float32 | Dtype::Float64 | Dtype::Complex64 | Dtype::Bfloat16
        )
    }

    /// Returns the promotion type of two data types.
    pub fn from_promoting_types(a: Dtype, b: Dtype) -> Self {
        a.promote_with(b)
    }

    /// Minimum value of the float point types. Returns `Err(_)` if the type is not
    /// float point
    pub fn finfo_min(&self) -> Result<f64, InexactDtypeError> {
        match self {
            Dtype::Float16 => Ok(f16::MIN.to_f64_const()),
            Dtype::Float32 => Ok(f32::MIN as f64),
            Dtype::Complex64 => Ok(f32::MIN as f64),
            Dtype::Bfloat16 => Ok(bf16::MIN.to_f64_const()),
            _ => Err(InexactDtypeError(*self)),
        }
    }

    /// Maximum value of the float point types. Returns `Err(_)` if the type is not
    /// float point
    pub fn finfo_max(&self) -> Result<f64, InexactDtypeError> {
        match self {
            Dtype::Float16 => Ok(f16::MAX.to_f64_const()),
            Dtype::Float32 => Ok(f32::MAX as f64),
            Dtype::Complex64 => Ok(f32::MAX as f64),
            Dtype::Bfloat16 => Ok(bf16::MAX.to_f64_const()),
            _ => Err(InexactDtypeError(*self)),
        }
    }
}

pub(crate) trait TypePromotion {
    fn promote_with(self, other: Self) -> Self;
}

impl TypePromotion for Dtype {
    fn promote_with(self, other: Self) -> Self {
        use crate::dtype::Dtype::*;
        match (self, other) {
            // Boolean promotions
            (Bool, Bool) => Bool,
            (Bool, _) | (_, Bool) => {
                if self == Bool {
                    other
                } else {
                    self
                }
            }

            // Uint8 promotions
            (Uint8, Uint8) => Uint8,
            (Uint8, Uint16) | (Uint16, Uint8) => Uint16,
            (Uint8, Uint32) | (Uint32, Uint8) => Uint32,
            (Uint8, Uint64) | (Uint64, Uint8) => Uint64,
            (Uint8, Int8) | (Int8, Uint8) => Int16,
            (Uint8, Int16) | (Int16, Uint8) => Int16,
            (Uint8, Int32) | (Int32, Uint8) => Int32,
            (Uint8, Int64) | (Int64, Uint8) => Int64,

            // Uint16 promotions
            (Uint16, Uint16) => Uint16,
            (Uint16, Uint32) | (Uint32, Uint16) => Uint32,
            (Uint16, Uint64) | (Uint64, Uint16) => Uint64,
            (Uint16, Int8) | (Int8, Uint16) => Int32,
            (Uint16, Int16) | (Int16, Uint16) => Int32,
            (Uint16, Int32) | (Int32, Uint16) => Int32,
            (Uint16, Int64) | (Int64, Uint16) => Int64,

            // Uint32 promotions
            (Uint32, Uint32) => Uint32,
            (Uint32, Uint64) | (Uint64, Uint32) => Uint64,
            (Uint32, Int8) | (Int8, Uint32) => Int64,
            (Uint32, Int16) | (Int16, Uint32) => Int64,
            (Uint32, Int32) | (Int32, Uint32) => Int64,
            (Uint32, Int64) | (Int64, Uint32) => Int64,

            // Uint64 promotions
            (Uint64, Uint64) => Uint64,
            (Uint64, Int8) | (Int8, Uint64) => Float32,
            (Uint64, Int16) | (Int16, Uint64) => Float32,
            (Uint64, Int32) | (Int32, Uint64) => Float32,
            (Uint64, Int64) | (Int64, Uint64) => Float32,

            // Int8 promotions
            (Int8, Int8) => Int8,
            (Int8, Int16) | (Int16, Int8) => Int16,
            (Int8, Int32) | (Int32, Int8) => Int32,
            (Int8, Int64) | (Int64, Int8) => Int64,

            // Int16 promotions
            (Int16, Int16) => Int16,
            (Int16, Int32) | (Int32, Int16) => Int32,
            (Int16, Int64) | (Int64, Int16) => Int64,

            // Int32 promotions
            (Int32, Int32) => Int32,
            (Int32, Int64) | (Int64, Int32) => Int64,

            // Int64 promotions
            (Int64, Int64) => Int64,

            // Float16 promotions
            (Float16, Bfloat16) | (Bfloat16, Float16) => Float32,

            // Complex type
            (Complex64, _) | (_, Complex64) => Complex64,

            // Float64 promotions
            (Float64, _) | (_, Float64) => Float64,

            // Float32 promotions
            (Float32, _) | (_, Float32) => Float32,

            // Float16 promotions
            (Float16, _) | (_, Float16) => Float16,

            // Bfloat16 promotions
            (Bfloat16, _) | (_, Bfloat16) => Bfloat16,
        }
    }
}

cfg_safetensors! {
    impl TryFrom<safetensors::tensor::Dtype> for Dtype {
        type Error = crate::error::ConversionError;

        fn try_from(value: safetensors::tensor::Dtype) -> Result<Self, Self::Error> {
            let out = match value {
                safetensors::Dtype::BOOL => Dtype::Bool,
                safetensors::Dtype::U8 => Dtype::Uint8,
                safetensors::Dtype::I8 => Dtype::Int8,
                safetensors::Dtype::F8_E5M2 => return Err(crate::error::ConversionError::SafeTensorDtype(value)),
                safetensors::Dtype::F8_E4M3 => return Err(crate::error::ConversionError::SafeTensorDtype(value)),
                safetensors::Dtype::I16 => Dtype::Int16,
                safetensors::Dtype::U16 => Dtype::Uint16,
                safetensors::Dtype::F16 => Dtype::Float16,
                safetensors::Dtype::BF16 => Dtype::Bfloat16,
                safetensors::Dtype::I32 => Dtype::Int32,
                safetensors::Dtype::U32 => Dtype::Uint32,
                safetensors::Dtype::F32 => Dtype::Float32,
                safetensors::Dtype::F64 => Dtype::Float64,
                safetensors::Dtype::I64 => Dtype::Int64,
                safetensors::Dtype::U64 => Dtype::Uint64,
                _ => return Err(crate::error::ConversionError::SafeTensorDtype(value)),
            };
            Ok(out)
        }
    }

    impl TryFrom<Dtype> for safetensors::tensor::Dtype {
        type Error = crate::error::ConversionError;

        fn try_from(value: Dtype) -> Result<Self, Self::Error> {
            let out = match value {
                Dtype::Bool => safetensors::Dtype::BOOL,
                Dtype::Uint8 => safetensors::Dtype::U8,
                Dtype::Int8 => safetensors::Dtype::I8,
                Dtype::Int16 => safetensors::Dtype::I16,
                Dtype::Uint16 => safetensors::Dtype::U16,
                Dtype::Float16 => safetensors::Dtype::F16,
                Dtype::Bfloat16 => safetensors::Dtype::BF16,
                Dtype::Int32 => safetensors::Dtype::I32,
                Dtype::Uint32 => safetensors::Dtype::U32,
                Dtype::Float32 => safetensors::Dtype::F32,
                Dtype::Float64 => safetensors::Dtype::F64,
                Dtype::Int64 => safetensors::Dtype::I64,
                Dtype::Uint64 => safetensors::Dtype::U64,
                Dtype::Complex64 => return Err(crate::error::ConversionError::MlxDtype(value)),
            };
            Ok(out)
        }
    }
}
