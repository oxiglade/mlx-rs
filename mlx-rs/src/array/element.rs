use crate::error::Result;
use crate::sealed::Sealed;
use crate::{complex64, Array, Dtype};
use half::{bf16, f16};

/// A marker trait for array elements.
pub trait ArrayElement: Sized + Sealed {
    /// The data type of the element.
    const DTYPE: Dtype;

    /// Access the value of a scalar array. Returns `Err` if the array is not scalar.
    fn array_item(array: &Array) -> Result<Self>;

    /// Access the raw data of an array.
    fn array_data(array: &Array) -> *const Self;
}

/// A marker trait for array element types that can be constructed via the
/// [`Array::from_slice`] method. This trait is implemented for all array
/// element types except for [`f64`].
///
/// [`f64`] is treated specially because it is not supported on GPU devices, but
/// rust defaults floating point literals to `f64`. With this trait, we can
/// limit the default floating point literals to `f32` for constructors
/// functions like [`Array::from_slice`] and [`Array::from_iter`], and macro
/// [`crate::array!`].
pub trait FromSliceElement: ArrayElement {}

macro_rules! impl_array_element {
    ($type:ty, $dtype:expr, $ctype:ident) => {
        paste::paste! {
            impl Sealed for $type {}
            impl ArrayElement for $type {
                const DTYPE: Dtype = $dtype;

                fn array_item(array: &Array) -> Result<Self> {
                    use crate::utils::guard::*;

                    <$type as Guarded>::try_from_op(|ptr| unsafe {
                        mlx_sys::[<mlx_array_item_ $ctype >](ptr, array.as_ptr())
                    })
                }

                fn array_data(array: &Array) -> *const Self {
                    unsafe { mlx_sys::[<mlx_array_data_ $ctype >](array.as_ptr()).cast::<Self>() }
                }

            }
        }
    };
}

impl_array_element!(bool, Dtype::Bool, bool);
impl_array_element!(u8, Dtype::Uint8, uint8);
impl_array_element!(u16, Dtype::Uint16, uint16);
impl_array_element!(u32, Dtype::Uint32, uint32);
impl_array_element!(u64, Dtype::Uint64, uint64);
impl_array_element!(i8, Dtype::Int8, int8);
impl_array_element!(i16, Dtype::Int16, int16);
impl_array_element!(i32, Dtype::Int32, int32);
impl_array_element!(i64, Dtype::Int64, int64);
impl_array_element!(f64, Dtype::Float64, float64);
impl_array_element!(f32, Dtype::Float32, float32);
impl_array_element!(f16, Dtype::Float16, float16);
impl_array_element!(bf16, Dtype::Bfloat16, bfloat16);
impl_array_element!(complex64, Dtype::Complex64, complex64);

macro_rules! impl_from_slice_element {
    ($type:ty) => {
        impl FromSliceElement for $type {}
    };
}

impl_from_slice_element!(bool);
impl_from_slice_element!(u8);
impl_from_slice_element!(u16);
impl_from_slice_element!(u32);
impl_from_slice_element!(u64);
impl_from_slice_element!(i8);
impl_from_slice_element!(i16);
impl_from_slice_element!(i32);
impl_from_slice_element!(i64);
impl_from_slice_element!(f32);
impl_from_slice_element!(f16);
impl_from_slice_element!(bf16);
impl_from_slice_element!(complex64);
