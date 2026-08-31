use crate::array::Array;
use crate::array::ArrayElement;
use crate::error::Result;
use crate::utils::guard::Guarded;
use crate::{Dtype, Stream};
use mlx_internal_macros::generate_macro;
use num_traits::NumCast;

/// Sample count and endpoint behavior for [`linspace`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinspaceOptions {
    /// Number of samples.
    pub count: i32,

    /// Include `stop` as the final sample.
    pub endpoint: bool,
}

impl Default for LinspaceOptions {
    fn default() -> Self {
        Self {
            count: 50,
            endpoint: true,
        }
    }
}

impl Array {
    /// Construct an array of zeros returning an error if shape is invalid.
    ///
    /// # Params
    ///
    /// - shape: Desired shape
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// Array::zeros::<f32>(&[5, 10]).unwrap();
    /// ```
    pub fn zeros<T: ArrayElement>(shape: &[i32]) -> Result<Array> {
        let dtype = T::DTYPE;
        zeros_dtype(shape, dtype)
    }

    /// Compatibility shim for [`zeros`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `zeros`"
    )]
    pub fn zeros_device<T: ArrayElement>(
        shape: &[i32],
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || Self::zeros::<T>(shape))
    }

    /// Construct an array of ones returning an error if shape is invalid.
    ///
    /// # Params
    ///
    /// - shape: Desired shape
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// Array::ones::<f32>(&[5, 10]).unwrap();
    /// ```
    pub fn ones<T: ArrayElement>(shape: &[i32]) -> Result<Array> {
        let dtype = T::DTYPE;
        ones_dtype(shape, dtype)
    }

    /// Compatibility shim for [`ones`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `ones`"
    )]
    pub fn ones_device<T: ArrayElement>(
        shape: &[i32],
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || Self::ones::<T>(shape))
    }

    /// Create an identity matrix or a general diagonal matrix returning an error if params are invalid.
    ///
    /// # Params
    ///
    /// - n: number of rows in the output
    /// - m: number of columns in the output -- equal to `n` if not specified
    /// - k: index of the diagonal - defaults to 0 if not specified
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// //  create [10, 10] array with 1's on the diagonal.
    /// let r = Array::eye::<f32>(10, None, None).unwrap();
    /// ```
    pub fn eye<T: ArrayElement>(n: i32, m: Option<i32>, k: Option<i32>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_eye(
                res,
                n,
                m.unwrap_or(n),
                k.unwrap_or(0),
                T::DTYPE.into(),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compatibility shim for [`eye`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `eye`"
    )]
    pub fn eye_device<T: ArrayElement>(
        n: i32,
        m: Option<i32>,
        k: Option<i32>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || Self::eye::<T>(n, m, k))
    }

    /// Construct an array with the given value returning an error if shape is invalid.
    ///
    /// Constructs an array of size `shape` filled with `values`. If `values`
    /// is an [Array] it must be [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting) to the given `shape`.
    ///
    /// # Params
    ///
    /// - shape: shape of the output array
    /// - values: values to be broadcast into the array
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::{Array, array};
    /// //  create [5, 4] array filled with 7
    /// let r = Array::full::<f32>(&[5, 4], array!(7.0f32)).unwrap();
    /// ```
    pub fn full<T: ArrayElement>(shape: &[i32], values: impl AsRef<Array>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_full(
                res,
                shape.as_ptr(),
                shape.len(),
                values.as_ref().as_ptr(),
                T::DTYPE.into(),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compatibility shim for [`full`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `full`"
    )]
    pub fn full_device<T: ArrayElement>(
        shape: &[i32],
        values: impl AsRef<Array>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || Self::full::<T>(shape, values))
    }

    /// Create a square identity matrix returning an error if params are invalid.
    ///
    /// # Params
    ///
    /// - n: number of rows and columns in the output
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// //  create [10, 10] array with 1's on the diagonal.
    /// let r = Array::identity::<f32>(10).unwrap();
    /// ```
    pub fn identity<T: ArrayElement>(n: i32) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_identity(res, n, T::DTYPE.into(), stream.as_ref().as_ptr())
        })
    }

    /// Compatibility shim for [`identity`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `identity`"
    )]
    pub fn identity_device<T: ArrayElement>(n: i32, stream: impl AsRef<Stream>) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || Self::identity::<T>(n))
    }

    /// Generates ranges of numbers.
    ///
    /// Generate numbers in the half-open interval `[start, stop)` in increments of `step`.
    ///
    /// # Params
    ///
    /// - `start`: Starting value which defaults to `0`.
    /// - `stop`: Stopping value.
    /// - `step`: Increment which defaults to `1`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    ///
    /// // Create a 1-D array with values from 0 to 50
    /// let r = Array::arange::<_, f32>(None, 50, None);
    /// ```
    pub fn arange<U, T>(
        start: impl Into<Option<U>>,
        stop: U,
        step: impl Into<Option<U>>,
    ) -> Result<Array>
    where
        U: NumCast,
        T: ArrayElement,
    {
        let stream = Stream::thread_local_or_default();
        let start: f64 = start.into().and_then(NumCast::from).unwrap_or(0.0);
        let stop: f64 = NumCast::from(stop).unwrap();
        let step: f64 = step.into().and_then(NumCast::from).unwrap_or(1.0);

        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_arange(
                res,
                start,
                stop,
                step,
                T::DTYPE.into(),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compatibility shim for [`arange`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `arange`"
    )]
    pub fn arange_device<U, T>(
        start: impl Into<Option<U>>,
        stop: U,
        step: impl Into<Option<U>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array>
    where
        U: NumCast,
        T: ArrayElement,
    {
        crate::with_stream(stream.as_ref(), || Self::arange::<U, T>(start, stop, step))
    }

    /// Compatibility shim for [`linspace`] with endpoint inclusion.
    #[deprecated(since = "0.26.0", note = "use `ops::linspace` with `LinspaceOptions`")]
    pub fn linspace<U, T>(start: U, stop: U, count: impl Into<Option<i32>>) -> Result<Array>
    where
        U: NumCast,
        T: ArrayElement,
    {
        linspace::<U, T>(
            start,
            stop,
            LinspaceOptions {
                count: count.into().unwrap_or(50),
                endpoint: true,
            },
        )
    }

    /// Compatibility shim for [`linspace`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `ops::linspace`"
    )]
    pub fn linspace_device<U, T>(
        start: U,
        stop: U,
        count: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array>
    where
        U: NumCast,
        T: ArrayElement,
    {
        crate::with_stream(stream.as_ref(), || {
            linspace::<U, T>(
                start,
                stop,
                LinspaceOptions {
                    count: count.into().unwrap_or(50),
                    endpoint: true,
                },
            )
        })
    }

    /// Repeat an array along a specified axis returning an error if params are invalid.
    ///
    /// # Params
    ///
    /// - array: array to repeat
    /// - count: number of times to repeat
    /// - axis: axis to repeat along
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// // repeat a [2, 2] array 4 times along axis 1
    /// let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
    /// let r = Array::repeat_axis::<i32>(source, 4, 1).unwrap();
    /// ```
    pub fn repeat_axis<T: ArrayElement>(array: Array, count: i32, axis: i32) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_repeat_axis(res, array.as_ptr(), count, axis, stream.as_ref().as_ptr())
        })
    }

    /// Compatibility shim for [`repeat_axis`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `repeat_axis`"
    )]
    pub fn repeat_axis_device<T: ArrayElement>(
        array: Array,
        count: i32,
        axis: i32,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || {
            Self::repeat_axis::<T>(array, count, axis)
        })
    }

    /// Repeat a flattened array along axis 0 returning an error if params are invalid.
    ///
    /// # Params
    ///
    /// - array: array to repeat
    /// - count: number of times to repeat
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// // repeat a 4 element array 4 times along axis 0
    /// let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
    /// let r = Array::repeat::<i32>(source, 4).unwrap();
    /// ```
    pub fn repeat<T: ArrayElement>(array: Array, count: i32) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_repeat(res, array.as_ptr(), count, stream.as_ref().as_ptr())
        })
    }

    /// Compatibility shim for [`repeat`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `repeat`"
    )]
    pub fn repeat_device<T: ArrayElement>(
        array: Array,
        count: i32,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || Self::repeat::<T>(array, count))
    }

    /// An array with ones at and below the given diagonal and zeros elsewhere.
    ///
    /// # Params
    ///
    /// - n: number of rows in the output
    /// - m: number of columns in the output -- equal to `n` if not specified
    /// - k: index of the diagonal -- defaults to 0 if not specified
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// // [5, 5] array with the lower triangle filled with 1s
    /// let r = Array::tri::<f32>(5, None, None);
    /// ```
    pub fn tri<T: ArrayElement>(n: i32, m: Option<i32>, k: Option<i32>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_tri(
                res,
                n,
                m.unwrap_or(n),
                k.unwrap_or(0),
                T::DTYPE.into(),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compatibility shim for [`tri`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `tri`"
    )]
    pub fn tri_device<T: ArrayElement>(
        n: i32,
        m: Option<i32>,
        k: Option<i32>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || Self::tri::<T>(n, m, k))
    }
}

/// See [`Array::zeros`]
pub fn zeros<T: ArrayElement>(shape: &[i32]) -> Result<Array> {
    Array::zeros::<T>(shape)
}

/// Compatibility shim for [`zeros`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `zeros`"
)]
pub fn zeros_device<T: ArrayElement>(
    shape: &[i32],
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || zeros::<T>(shape))
}

/// An array of zeros like the input.
pub fn zeros_like(input: impl AsRef<Array>) -> Result<Array> {
    let a = input.as_ref();
    let shape = a.shape();
    let dtype = a.dtype();
    zeros_dtype(shape, dtype)
}

/// Compatibility shim for [`zeros_like`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `zeros_like`"
)]
pub fn zeros_like_device(
    input: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || zeros_like(input))
}

/// Similar to [`Array::zeros`] but with a specified dtype.
pub fn zeros_dtype(shape: &[i32], dtype: Dtype) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_zeros(
            res,
            shape.as_ptr(),
            shape.len(),
            dtype.into(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`zeros_dtype`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `zeros_dtype`"
)]
pub fn zeros_dtype_device(
    shape: &[i32],
    dtype: Dtype,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || zeros_dtype(shape, dtype))
}

/// See [`Array::ones`]
pub fn ones<T: ArrayElement>(shape: &[i32]) -> Result<Array> {
    Array::ones::<T>(shape)
}

/// Compatibility shim for [`ones`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `ones`"
)]
pub fn ones_device<T: ArrayElement>(
    shape: &[i32],
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || ones::<T>(shape))
}

/// An array of ones like the input.
pub fn ones_like(input: impl AsRef<Array>) -> Result<Array> {
    let a = input.as_ref();
    let shape = a.shape();
    let dtype = a.dtype();
    ones_dtype(shape, dtype)
}

/// Compatibility shim for [`ones_like`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `ones_like`"
)]
pub fn ones_like_device(
    input: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || ones_like(input))
}

/// An array filled with the given value, with the same shape as the input.
///
/// # Params
///
/// - `input`: Input array to take shape from
/// - `values`: Value(s) to fill the array with
/// - `dtype`: Optional dtype for the output array. Defaults to the dtype of the input array.
/// - `stream`: Stream to run the operation on
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, Dtype, ops::full_like};
///
/// let a = Array::from_slice(&[1i32, 2, 3], &[3]);
/// // Fill with same dtype as input
/// let b = full_like(&a, &Array::from_f32(7.0), None).unwrap();
/// assert_eq!(b.dtype(), Dtype::Int32);
///
/// // Fill with specified dtype
/// let c = full_like(&a, &Array::from_f32(7.5), Some(Dtype::Float32)).unwrap();
/// assert_eq!(c.dtype(), Dtype::Float32);
/// ```
pub fn full_like(
    input: impl AsRef<Array>,
    values: impl AsRef<Array>,
    dtype: impl Into<Option<Dtype>>,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    let a = input.as_ref();
    let dtype = dtype.into().unwrap_or_else(|| a.dtype());
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_full_like(
            res,
            a.as_ptr(),
            values.as_ref().as_ptr(),
            dtype.into(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`full_like`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `full_like`"
)]
pub fn full_like_device(
    input: impl AsRef<Array>,
    values: impl AsRef<Array>,
    #[optional] dtype: impl Into<Option<Dtype>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || full_like(input, values, dtype))
}

/// Similar to [`Array::ones`] but with a specified dtype.
pub fn ones_dtype(shape: &[i32], dtype: Dtype) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_ones(
            res,
            shape.as_ptr(),
            shape.len(),
            dtype.into(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`ones_dtype`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `ones_dtype`"
)]
pub fn ones_dtype_device(
    shape: &[i32],
    dtype: Dtype,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || ones_dtype(shape, dtype))
}

/// See [`Array::eye`]
pub fn eye<T: ArrayElement>(n: i32, m: Option<i32>, k: Option<i32>) -> Result<Array> {
    Array::eye::<T>(n, m, k)
}

/// Compatibility shim for [`eye`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `eye`"
)]
pub fn eye_device<T: ArrayElement>(
    n: i32,
    #[optional] m: Option<i32>,
    #[optional] k: Option<i32>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || eye::<T>(n, m, k))
}

/// See [`Array::full`]
pub fn full<T: ArrayElement>(shape: &[i32], values: impl AsRef<Array>) -> Result<Array> {
    Array::full::<T>(shape, values)
}

/// Compatibility shim for [`full`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `full`"
)]
pub fn full_device<T: ArrayElement>(
    shape: &[i32],
    values: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || full::<T>(shape, values))
}

/// See [`Array::identity`]
pub fn identity<T: ArrayElement>(n: i32) -> Result<Array> {
    Array::identity::<T>(n)
}

/// Compatibility shim for [`identity`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `identity`"
)]
pub fn identity_device<T: ArrayElement>(
    n: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || identity::<T>(n))
}

/// See [`Array::arange`]
pub fn arange<U, T>(
    start: impl Into<Option<U>>,
    stop: U,
    step: impl Into<Option<U>>,
) -> Result<Array>
where
    U: NumCast,
    T: ArrayElement,
{
    Array::arange::<U, T>(start, stop, step)
}

/// Compatibility shim for [`arange`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `arange`"
)]
pub fn arange_device<U, T>(
    #[optional] start: impl Into<Option<U>>,
    #[named] stop: U,
    #[optional] step: impl Into<Option<U>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array>
where
    U: NumCast,
    T: ArrayElement,
{
    crate::with_stream(stream.as_ref(), || arange::<U, T>(start, stop, step))
}

/// Generate evenly spaced values between `start` and `stop`.
///
/// A negative count errors, zero returns an empty array, and one returns `start`. When `endpoint`
/// is false, the interval is divided by `count` and `stop` is excluded. Endpoints are converted
/// directly to the C ABI's `double` representation.
///
/// ```rust
/// use mlx_rs::{ops::{linspace, LinspaceOptions}, with_stream, Stream};
///
/// // float64 runs on the CPU only.
/// let output = with_stream(&Stream::cpu(), || {
///     linspace::<_, f64>(
///         16_777_217.0_f64,
///         16_777_219.0_f64,
///         LinspaceOptions {
///             count: 3,
///             endpoint: true,
///         },
///     )
/// })
/// .unwrap();
/// assert_eq!(output.shape(), &[3]);
/// ```
pub fn linspace<U, T>(start: U, stop: U, options: LinspaceOptions) -> Result<Array>
where
    U: NumCast,
    T: ArrayElement,
{
    let stream = Stream::thread_local_or_default();
    let start: f64 = NumCast::from(start).unwrap();
    let stop: f64 = NumCast::from(stop).unwrap();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_linspace_endpoint(
            res,
            start,
            stop,
            options.count,
            options.endpoint,
            T::DTYPE.into(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`linspace`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `linspace`"
)]
pub fn linspace_device<U, T>(
    start: U,
    stop: U,
    #[optional] count: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array>
where
    U: NumCast,
    T: ArrayElement,
{
    crate::with_stream(stream.as_ref(), || {
        linspace::<U, T>(
            start,
            stop,
            LinspaceOptions {
                count: count.into().unwrap_or(50),
                endpoint: true,
            },
        )
    })
}

/// See [`Array::repeat`]
pub fn repeat_axis<T: ArrayElement>(array: Array, count: i32, axis: i32) -> Result<Array> {
    Array::repeat_axis::<T>(array, count, axis)
}

/// Compatibility shim for [`repeat_axis`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `repeat_axis`"
)]
pub fn repeat_axis_device<T: ArrayElement>(
    array: Array,
    count: i32,
    axis: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || repeat_axis::<T>(array, count, axis))
}

/// See [`Array::repeat`]
pub fn repeat<T: ArrayElement>(array: Array, count: i32) -> Result<Array> {
    Array::repeat::<T>(array, count)
}

/// Compatibility shim for [`repeat`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `repeat`"
)]
pub fn repeat_device<T: ArrayElement>(
    array: Array,
    count: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || repeat::<T>(array, count))
}

/// See [`Array::tri`]
pub fn tri<T: ArrayElement>(n: i32, m: Option<i32>, k: Option<i32>) -> Result<Array> {
    Array::tri::<T>(n, m, k)
}

/// Compatibility shim for [`tri`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `tri`"
)]
pub fn tri_device<T: ArrayElement>(
    n: i32,
    #[optional] m: Option<i32>,
    #[optional] k: Option<i32>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || tri::<T>(n, m, k))
}

/// Zeros the array above the given diagonal
///
/// # Params
///
/// - `a`: input array
/// - `k`: diagonal of the 2D array. Default to `0`
/// - `stream`: stream to execute on
pub fn tril(a: impl AsRef<Array>, k: impl Into<Option<i32>>) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    let a = a.as_ref();
    let k = k.into().unwrap_or(0);
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_tril(res, a.as_ptr(), k, stream.as_ref().as_ptr())
    })
}

/// Compatibility shim for [`tril`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `tril`"
)]
pub fn tril_device(
    a: impl AsRef<Array>,
    #[optional] k: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || tril(a, k))
}

/// Zeros the array below the given diagonal
///
/// # Params
///
/// - `a`: input array
/// - `k`: diagonal of the 2D array. Default to `0`
pub fn triu(a: impl AsRef<Array>, k: impl Into<Option<i32>>) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    let a = a.as_ref();
    let k = k.into().unwrap_or(0);
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_triu(res, a.as_ptr(), k, stream.as_ref().as_ptr())
    })
}

/// Compatibility shim for [`triu`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `triu`"
)]
pub fn triu_device(
    a: impl AsRef<Array>,
    #[optional] k: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || triu(a, k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        array, dtype::Dtype, test_utils::assert_array_eq, test_utils::tolerances, with_device,
        Device,
    };
    use half::f16;

    #[test]
    fn test_zeros() {
        let array = Array::zeros::<f32>(&[2, 3]).unwrap();
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        let data: &[f32] = array.as_slice();
        assert_eq!(data, &[0.0; 6]);
    }

    #[test]
    fn test_zeros_try() {
        let array = Array::zeros::<f32>(&[2, 3]);
        assert!(array.is_ok());

        let array = Array::zeros::<f32>(&[-1, 3]);
        assert!(array.is_err());
    }

    #[test]
    fn test_ones() {
        let array = Array::ones::<f16>(&[2, 3]).unwrap();
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.dtype(), Dtype::Float16);

        let data: &[f16] = array.as_slice();
        assert_eq!(data, &[f16::from_f32(1.0); 6]);
    }

    #[test]
    fn test_eye() {
        let array = Array::eye::<f32>(3, None, None).unwrap();
        assert_eq!(array.shape(), &[3, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        let data: &[f32] = array.as_slice();
        assert_eq!(data, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_full_scalar() {
        let array = Array::full::<f32>(&[2, 3], array!(7f32)).unwrap();
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        assert_array_eq(
            array,
            Array::from_slice(&[7.0; 6], &[2, 3]),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );
    }

    #[test]
    fn test_full_array() {
        let source = with_device(Device::cpu(), || Array::zeros::<f32>(&[1, 3])).unwrap();
        let array = Array::full::<f32>(&[2, 3], source).unwrap();
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        assert_array_eq(
            array,
            Array::from_slice(&[0.0; 6], &[2, 3]),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );
    }

    #[test]
    fn test_full_try() {
        let source = Array::zeros::<f32>(&[1, 3]).unwrap();
        let array = Array::full::<f32>(&[2, 3], source);
        assert!(array.is_ok());

        let source = Array::zeros::<f32>(&[1, 3]).unwrap();
        let array = Array::full::<f32>(&[-1, 3], source);
        assert!(array.is_err());
    }

    #[test]
    fn test_identity() {
        let array = Array::identity::<f32>(3).unwrap();
        assert_eq!(array.shape(), &[3, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        let data: &[f32] = array.as_slice();
        assert_eq!(data, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_arange() {
        let array = Array::arange::<_, f32>(None, 50, None).unwrap();
        assert_eq!(array.shape(), &[50]);
        assert_eq!(array.dtype(), Dtype::Float32);

        let data: &[f32] = array.as_slice();
        let expected: Vec<f32> = (0..50).map(|x| x as f32).collect();
        assert_eq!(data, expected.as_slice());

        let array = Array::arange::<_, i32>(0, 50, None).unwrap();
        assert_eq!(array.shape(), &[50]);
        assert_eq!(array.dtype(), Dtype::Int32);

        let data: &[i32] = array.as_slice();
        let expected: Vec<i32> = (0..50).collect();
        assert_eq!(data, expected.as_slice());

        let result = Array::arange::<_, bool>(None, 50, None);
        assert!(result.is_err());

        let result = Array::arange::<_, f32>(f64::NEG_INFINITY, 50.0, None);
        assert!(result.is_err());

        let result = Array::arange::<_, f32>(0.0, f64::INFINITY, None);
        assert!(result.is_err());

        let result = Array::arange::<_, f32>(0.0, 50.0, f32::NAN);
        assert!(result.is_err());

        let result = Array::arange::<_, f32>(f32::NAN, 50.0, None);
        assert!(result.is_err());

        let result = Array::arange::<_, f32>(0.0, f32::NAN, None);
        assert!(result.is_err());

        let result = Array::arange::<_, f32>(0, i32::MAX as i64 + 1, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_linspace_int() {
        let array = Array::linspace::<_, f32>(0, 50, None).unwrap();
        assert_eq!(array.shape(), &[50]);
        assert_eq!(array.dtype(), Dtype::Float32);

        let expected_data: Vec<f32> = (0..50).map(|x| x as f32 * (50.0 / 49.0)).collect();
        let expected = Array::from_slice(&expected_data, &[50]);
        assert_eq!(array.shape(), expected.shape());
        assert_array_all_close!(array, expected);
    }

    #[test]
    fn test_linspace_float() {
        let array = Array::linspace::<_, f32>(0., 50., None).unwrap();
        assert_eq!(array.shape(), &[50]);
        assert_eq!(array.dtype(), Dtype::Float32);

        let expected_data: Vec<f32> = (0..50).map(|x| x as f32 * (50.0 / 49.0)).collect();
        let expected = Array::from_slice(&expected_data, &[50]);
        assert_eq!(array.shape(), expected.shape());
        assert_array_all_close!(array, expected);
    }

    #[test]
    fn test_linspace_try() {
        let array = Array::linspace::<_, f32>(0, 50, None);
        assert!(array.is_ok());

        let array = Array::linspace::<_, f32>(0, 50, Some(-1));
        assert!(array.is_err());
    }

    #[test]
    fn test_repeat() {
        let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
        let array = Array::repeat_axis::<i32>(source, 4, 1).unwrap();
        assert_eq!(array.shape(), &[2, 8]);
        assert_eq!(array.dtype(), Dtype::Int32);

        let data: &[i32] = array.as_slice();
        assert_eq!(data, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);
    }

    #[test]
    fn test_repeat_try() {
        let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
        let array = Array::repeat_axis::<i32>(source, 4, 1);
        assert!(array.is_ok());

        let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
        let array = Array::repeat_axis::<i32>(source, -1, 1);
        assert!(array.is_err());
    }

    #[test]
    fn test_repeat_all() {
        let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
        let array = Array::repeat::<i32>(source, 4).unwrap();
        assert_eq!(array.shape(), &[16]);
        assert_eq!(array.dtype(), Dtype::Int32);

        let data: &[i32] = array.as_slice();
        assert_eq!(data, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);
    }

    #[test]
    fn test_repeat_all_try() {
        let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
        let array = Array::repeat::<i32>(source, 4);
        assert!(array.is_ok());

        let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
        let array = Array::repeat::<i32>(source, -1);
        assert!(array.is_err());
    }

    #[test]
    fn test_tri() {
        let array = Array::tri::<f32>(3, None, None).unwrap();
        assert_eq!(array.shape(), &[3, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        let data: &[f32] = array.as_slice();
        assert_eq!(data, &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
    }

    // The tests below are adapted from the C++ unit test `ops_tests.cpp/test full_like`
    #[test]
    fn test_full_like() {
        // Test with explicit dtype (different from input)
        let base_int = Array::from_slice(&[1i16, 2, 3], &[3]);
        let from_array_with_dtype =
            full_like(&base_int, &array!(7.5f32), Some(Dtype::Float16)).unwrap();
        assert_eq!(from_array_with_dtype.dtype(), Dtype::Float16);
        assert_eq!(from_array_with_dtype.shape(), &[3]);

        assert_array_eq(
            from_array_with_dtype,
            Array::from_slice(&[f16::from_f32(7.5); 3], &[3]),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );

        // Test with default dtype (inherits from input)
        let from_array_default_dtype = full_like(&base_int, &array!(4.0f32), None).unwrap();
        assert_eq!(from_array_default_dtype.dtype(), Dtype::Int16);
        assert_array_eq(
            from_array_default_dtype,
            Array::from_slice(&[4_i16, 4, 4], &[3]),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );

        // Test with explicit dtype float32
        let from_scalar_with_dtype =
            full_like(&base_int, &array!(3.25f32), Some(Dtype::Float32)).unwrap();
        assert_eq!(from_scalar_with_dtype.dtype(), Dtype::Float32);
        assert_array_eq(
            from_scalar_with_dtype,
            Array::from_slice(&[3.25_f32; 3], &[3]),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );

        // Test with float base and int value - uses base dtype
        let base_float = Array::from_slice(&[1.0f32, 2.0f32], &[2]);
        let from_scalar_default_dtype = full_like(&base_float, &array!(2i32), None).unwrap();
        assert_eq!(from_scalar_default_dtype.dtype(), Dtype::Float32);
        assert_array_eq(
            from_scalar_default_dtype,
            Array::from_slice(&[2.0_f32; 2], &[2]),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );
    }
}
