//! Implements bindings for the sorting ops.

use mlx_internal_macros::generate_macro;

use crate::{error::Result, utils::guard::Guarded, Array, Stream};

/// Returns a sorted copy of the array. Returns an error if the arguments are invalid.
///
/// # Params
///
/// - `array`: input array
/// - `axis`: axis to sort over
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let axis = 0;
/// let result = sort_axis(&a, axis);
/// ```
pub fn sort_axis(a: impl AsRef<Array>, axis: i32) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_sort_axis(res, a.as_ref().as_ptr(), axis, stream.as_ref().as_ptr())
    })
}

/// Compatibility shim for [`sort_axis`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `sort_axis`"
)]
pub fn sort_axis_device(
    a: impl AsRef<Array>,
    axis: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || sort_axis(a, axis))
}

/// Returns a sorted copy of the flattened array. Returns an error if the arguments are invalid.
///
/// # Params
///
/// - `array`: input array
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let result = sort(&a);
/// ```
pub fn sort(a: impl AsRef<Array>) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_sort(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Compatibility shim for [`sort`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `sort`"
)]
pub fn sort_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || sort(a))
}

/// Returns the indices that sort the array. Returns an error if the arguments are invalid.
///
/// # Params
///
/// - `a`: The array to sort.
/// - `axis`: axis to sort over
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let axis = 0;
/// let result = argsort_axis(&a, axis);
/// ```
pub fn argsort_axis(a: impl AsRef<Array>, axis: i32) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_argsort_axis(res, a.as_ref().as_ptr(), axis, stream.as_ref().as_ptr())
    })
}

/// Compatibility shim for [`argsort_axis`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `argsort_axis`"
)]
pub fn argsort_axis_device(
    a: impl AsRef<Array>,
    axis: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || argsort_axis(a, axis))
}

/// Returns the indices that sort the flattened array. Returns an error if the arguments are
/// invalid.
///
/// # Params
///
/// - `a`: The array to sort.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let result = argsort(&a);
/// ```
pub fn argsort(a: impl AsRef<Array>) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_argsort(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Compatibility shim for [`argsort`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `argsort`"
)]
pub fn argsort_device(
    a: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || argsort(a))
}

/// Returns a partitioned copy of the array such that the smaller `kth` elements are first.
/// Returns an error if the arguments are invalid.
///
/// The ordering of the elements in partitions is undefined.
///
/// # Params
///
/// - `array`: input array
/// - `kth`: Element at the `kth` index will be in its sorted position in the output. All elements
///   before the kth index will be less or equal to the `kth` element and all elements after will be
///   greater or equal to the `kth` element in the output.
/// - `axis`: axis to partition over
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let kth = 1;
/// let axis = 0;
/// let result = partition_axis(&a, kth, axis);
/// ```
pub fn partition_axis(a: impl AsRef<Array>, kth: i32, axis: i32) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_partition_axis(
            res,
            a.as_ref().as_ptr(),
            kth,
            axis,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`partition_axis`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `partition_axis`"
)]
pub fn partition_axis_device(
    a: impl AsRef<Array>,
    kth: i32,
    axis: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || partition_axis(a, kth, axis))
}

/// Returns a partitioned copy of the flattened array such that the smaller `kth` elements are
/// first. Returns an error if the arguments are invalid.
///
/// The ordering of the elements in partitions is undefined.
///
/// # Params
///
/// - `array`: input array
/// - `kth`: Element at the `kth` index will be in its sorted position in the output. All elements
///   before the kth index will be less or equal to the `kth` element and all elements after will be
///   greater or equal to the `kth` element in the output.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let kth = 1;
/// let result = partition(&a, kth);
/// ```
pub fn partition(a: impl AsRef<Array>, kth: i32) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_partition(res, a.as_ref().as_ptr(), kth, stream.as_ref().as_ptr())
    })
}

/// Compatibility shim for [`partition`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `partition`"
)]
pub fn partition_device(
    a: impl AsRef<Array>,
    kth: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || partition(a, kth))
}

/// Returns the indices that partition the array. Returns an error if the arguments are invalid.
///
/// The ordering of the elements within a partition in given by the indices is undefined.
///
/// # Params
///
/// - `a`: The array to sort.
/// - `kth`: element index at the `kth` position in the output will give the sorted position.  All
///   indices before the`kth` position will be of elements less than or equal to the element at the
///   `kth` index and all indices after will be elemenents greater than or equal to the element at
///   the `kth` position.
/// - `axis`: axis to partition over
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let kth = 1;
/// let axis = 0;
/// let result = argpartition_axis(&a, kth, axis);
/// ```
pub fn argpartition_axis(a: impl AsRef<Array>, kth: i32, axis: i32) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_argpartition_axis(
            res,
            a.as_ref().as_ptr(),
            kth,
            axis,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`argpartition_axis`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `argpartition_axis`"
)]
pub fn argpartition_axis_device(
    a: impl AsRef<Array>,
    kth: i32,
    axis: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || argpartition_axis(a, kth, axis))
}

/// Returns the indices that partition the flattened array. Returns an error if the arguments are
/// invalid.
///
/// The ordering of the elements within a partition in given by the indices is undefined.
///
/// # Params
///
/// - `a`: The array to sort.
/// - `kth`: element index at the `kth` position in the output will give the sorted position.  All
///   indices before the`kth` position will be of elements less than or equal to the element at the
///   `kth` index and all indices after will be elemenents greater than or equal to the element at
///   the `kth` position.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let kth = 1;
/// let result = argpartition(&a, kth);
/// ```
pub fn argpartition(a: impl AsRef<Array>, kth: i32) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_argpartition(res, a.as_ref().as_ptr(), kth, stream.as_ref().as_ptr())
    })
}

/// Compatibility shim for [`argpartition`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `argpartition`"
)]
pub fn argpartition_device(
    a: impl AsRef<Array>,
    kth: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || argpartition(a, kth))
}

#[cfg(test)]
mod tests {
    use crate::Array;

    #[test]
    fn test_sort_with_invalid_axis() {
        let a = Array::from_slice(&[1, 2, 3, 4, 5], &[5]);
        let axis = 1;
        let result = super::sort_axis(&a, axis);
        assert!(result.is_err());
    }

    #[test]
    fn test_partition_with_invalid_axis() {
        let a = Array::from_slice(&[1, 2, 3, 4, 5], &[5]);
        let kth = 2;
        let axis = 1;
        let result = super::partition_axis(&a, kth, axis);
        assert!(result.is_err());
    }

    #[test]
    fn test_partition_with_invalid_kth() {
        let a = Array::from_slice(&[1, 2, 3, 4, 5], &[5]);
        let kth = 5;
        let axis = 0;
        let result = super::partition_axis(&a, kth, axis);
        assert!(result.is_err());
    }

    #[test]
    fn test_partition_all_with_invalid_kth() {
        let a = Array::from_slice(&[1, 2, 3, 4, 5], &[5]);
        let kth = 5;
        let result = super::partition(&a, kth);
        assert!(result.is_err());
    }
}
