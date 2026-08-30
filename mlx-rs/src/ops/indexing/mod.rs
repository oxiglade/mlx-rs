//! Indexing Arrays
//!
//! # Overview
//!
//! Due to limitations in the `std::ops::Index` and `std::ops::IndexMut` traits (only references can
//! be returned), the indexing is achieved with the [`IndexOp`] and [`IndexMutOp`] traits where
//! arrays can be indexed with [`IndexOp::index()`] and [`IndexMutOp::index_mut()`] respectively.
//!
//! The following types can be used as indices:
//!
//! | Type | Description |
//! |------|-------------|
//! | [`i32`] | An integer index |
//! | [`Array`] | Use an array to index another array |
//! | `&Array` | Use a reference to an array to index another array |
//! | [`std::ops::Range<i32>`] | A range index |
//! | [`std::ops::RangeFrom<i32>`] | A range index |
//! | [`std::ops::RangeFull`] | A range index |
//! | [`std::ops::RangeInclusive<i32>`] | A range index |
//! | [`std::ops::RangeTo<i32>`] | A range index |
//! | [`std::ops::RangeToInclusive<i32>`] | A range index |
//! | [`StrideBy`] | A range index with stride |
//! | [`NewAxis`] | Add a new axis |
//! | [`Ellipsis`] | Consume all axes |
//!
//! # Single axis indexing
//!
//! | Indexing Operation | `mlx` (python) | `mlx-swift` | `mlx-rs` |
//! |--------------------|--------|-------|------|
//! | integer | `arr[1]` | `arr[1]` | `arr.index(1)` |
//! | range expression | `arr[1:3]` | `arr[1..<3]` | `arr.index(1..3)` |
//! | full range | `arr[:]` | `arr[0...]` | `arr.index(..)` |
//! | range with stride | `arr[::2]` | `arr[.stride(by: 2)]` | `arr.index((..).stride_by(2))` |
//! | ellipsis (consuming all axes) | `arr[...]` | `arr[.ellipsis]` | `arr.index(Ellipsis)` |
//! | newaxis | `arr[None]` | `arr[.newAxis]` | `arr.index(NewAxis)` |
//! | mlx array `i` | `arr[i]` | `arr[i]` | `arr.index(i)` |
//!
//! # Multi-axes indexing
//!
//! Multi-axes indexing with combinations of the above operations is also supported by combining the
//! operations in a tuple with the restriction that `Ellipsis` can only be used once.
//!
//! ## Examples
//!
//! ```rust
//! // See the multi-dimensional example code for mlx python https://ml-explore.github.io/mlx/build/html/usage/indexing.html
//!
//! use mlx_rs::{Array, ops::indexing::*};
//!
//! let a = Array::from_iter(0..8, &[2, 2, 2]);
//!
//! // a[:, :, 0]
//! let mut s1 = a.index((.., .., 0));
//!
//! let expected = Array::from_slice(&[0, 2, 4, 6], &[2, 2]);
//! assert_eq!(s1, expected);
//!
//! // a[..., 0]
//! let mut s2 = a.index((Ellipsis, 0));
//!
//! let expected = Array::from_slice(&[0, 2, 4, 6], &[2, 2]);
//! assert_eq!(s1, expected);
//! ```
//!
//! # Set values with indexing
//!
//! The same indexing operations (single or multiple) can be used to set values in an array using
//! the [`IndexMutOp`] trait.
//!
//! ## Example
//!
//! ```rust
//! use mlx_rs::{Array, ops::indexing::*};
//!
//! let mut a = Array::from_slice(&[1, 2, 3], &[3]);
//! a.index_mut(2, Array::from_int(0));
//!
//! let expected = Array::from_slice(&[1, 2, 0], &[3]);
//! assert_eq!(a, expected);
//! ```
//!
//! ```rust
//! use mlx_rs::{Array, ops::indexing::*};
//!
//! let mut a = Array::from_iter(0i32..20, &[2, 2, 5]);
//!
//! // writing using slices -- this ends up covering two elements
//! a.index_mut((0..1, 1..2, 2..4), Array::from_int(88));
//!
//! let expected = Array::from_slice(
//!     &[
//!         0, 1, 2, 3, 4, 5, 6, 88, 88, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
//!     ],
//!     &[2, 2, 5],
//! );
//! assert_eq!(a, expected);
//! ```

use std::{borrow::Cow, ops::Bound, rc::Rc};

use mlx_internal_macros::generate_macro;

use crate::{error::Result, utils::guard::Guarded, Array, Stream, StreamOrDevice};

pub(crate) mod index_impl;
pub(crate) mod indexmut_impl;

/* -------------------------------------------------------------------------- */
/*                                Custom types                                */
/* -------------------------------------------------------------------------- */

/// New axis indexing operation.
///
/// See the module level documentation for more information.
#[derive(Debug, Clone, Copy)]
pub struct NewAxis;

/// Ellipsis indexing operation.
///
/// See the module level documentation for more information.
#[derive(Debug, Clone, Copy)]
pub struct Ellipsis;

/// Stride indexing operation.
///
/// See the module level documentation for more information.
#[derive(Debug, Clone, Copy)]
pub struct StrideBy<I> {
    /// The inner iterator
    pub inner: I,

    /// The stride
    pub stride: i32,
}

/// Helper trait for creating a stride indexing operation.
pub trait IntoStrideBy: Sized {
    /// Create a stride indexing operation.
    fn stride_by(self, stride: i32) -> StrideBy<Self>;
}

impl<T> IntoStrideBy for T {
    fn stride_by(self, stride: i32) -> StrideBy<Self> {
        StrideBy {
            inner: self,
            stride,
        }
    }
}

/// Range indexing operation.
#[derive(Debug, Clone)]
pub struct RangeIndex {
    start: Bound<i32>,
    stop: Bound<i32>,
    stride: i32,
}

impl RangeIndex {
    pub(crate) fn new(start: Bound<i32>, stop: Bound<i32>, stride: Option<i32>) -> Self {
        let stride = stride.unwrap_or(1);
        Self {
            start,
            stop,
            stride,
        }
    }

    pub(crate) fn is_full(&self) -> bool {
        matches!(self.start, Bound::Unbounded)
            && matches!(self.stop, Bound::Unbounded)
            && self.stride == 1
    }

    pub(crate) fn stride(&self) -> i32 {
        self.stride
    }

    pub(crate) fn start(&self, size: i32) -> i32 {
        match self.start {
            Bound::Included(start) => start,
            Bound::Excluded(start) => start + 1,
            Bound::Unbounded => {
                // ref swift binding
                // _start ?? (stride < 0 ? size - 1 : 0)

                if self.stride.is_negative() {
                    size - 1
                } else {
                    0
                }
            }
        }
    }

    pub(crate) fn absolute_start(&self, size: i32) -> i32 {
        // ref swift binding
        // return start < 0 ? start + size : start

        let start = self.start(size);
        if start.is_negative() {
            start + size
        } else {
            start
        }
    }

    pub(crate) fn end(&self, size: i32) -> i32 {
        match self.stop {
            Bound::Included(stop) => stop + 1,
            Bound::Excluded(stop) => stop,
            Bound::Unbounded => {
                // ref swift binding
                // _end ?? (stride < 0 ? -size - 1 : size)

                if self.stride.is_negative() {
                    -size - 1
                } else {
                    size
                }
            }
        }
    }

    pub(crate) fn absolute_end(&self, size: i32) -> i32 {
        // ref swift binding
        // return end < 0 ? end + size : end

        let end = self.end(size);
        if end.is_negative() {
            end + size
        } else {
            end
        }
    }
}

/// Indexing operation for arrays.
#[derive(Debug, Clone)]
pub enum ArrayIndexOp<'a> {
    /// An `Ellipsis` is used to consume all axes
    ///
    /// This is equivalent to `...` in python
    Ellipsis,

    /// A single index operation
    ///
    /// This is equivalent to `arr[1]` in python
    TakeIndex {
        /// The index to take
        index: i32,
    },

    /// Indexing with an array
    TakeArray {
        /// The indices to take
        indices: Rc<Array>, // TODO: remove `Rc` because `Array` is `Clone`
    },

    /// Indexing with an array reference
    TakeArrayRef {
        /// The indices to take
        indices: &'a Array,
    },

    /// Indexing with a range
    ///
    /// This is equivalent to `arr[1:3]` in python
    Slice(RangeIndex),

    /// New axis operation
    ///
    /// This is equivalent to `arr[None]` in python
    ExpandDims,
}

impl ArrayIndexOp<'_> {
    fn is_array_or_index(&self) -> bool {
        // Using the full match syntax to avoid forgetting to add new variants
        match self {
            ArrayIndexOp::TakeIndex { .. }
            | ArrayIndexOp::TakeArrayRef { .. }
            | ArrayIndexOp::TakeArray { .. } => true,
            ArrayIndexOp::Ellipsis | ArrayIndexOp::Slice(_) | ArrayIndexOp::ExpandDims => false,
        }
    }
    fn is_array(&self) -> bool {
        // Using the full match syntax to avoid forgetting to add new variants
        match self {
            ArrayIndexOp::TakeArray { .. } | ArrayIndexOp::TakeArrayRef { .. } => true,
            ArrayIndexOp::TakeIndex { .. }
            | ArrayIndexOp::Ellipsis
            | ArrayIndexOp::Slice(_)
            | ArrayIndexOp::ExpandDims => false,
        }
    }
}

/* -------------------------------------------------------------------------- */
/*                                Custom traits                               */
/* -------------------------------------------------------------------------- */

/// Trait for custom indexing operations.
///
/// Out of bounds indexing is allowed and wouldn't return an error.
pub trait TryIndexOp<Idx> {
    /// Try to index the array with the given index.
    fn try_index_device(&self, i: Idx, stream: impl AsRef<Stream>) -> Result<Array>;

    /// Try to index the array with the given index.
    fn try_index(&self, i: Idx) -> Result<Array> {
        self.try_index_device(i, StreamOrDevice::default())
    }
}

/// Trait for custom indexing operations.
///
/// This is implemented for all types that implement `TryIndexOp`.
pub trait IndexOp<Idx>: TryIndexOp<Idx> {
    /// Index the array with the given index.
    fn index_device(&self, i: Idx, stream: impl AsRef<Stream>) -> Array {
        self.try_index_device(i, stream).unwrap()
    }

    /// Index the array with the given index.
    fn index(&self, i: Idx) -> Array {
        self.try_index(i).unwrap()
    }
}

impl<T, Idx> IndexOp<Idx> for T where T: TryIndexOp<Idx> {}

/// Trait for custom mutable indexing operations.
pub trait TryIndexMutOp<Idx, Val> {
    /// Try to index the array with the given index and set the value.
    fn try_index_mut_device(&mut self, i: Idx, val: Val, stream: impl AsRef<Stream>) -> Result<()>;

    /// Try to index the array with the given index and set the value.
    fn try_index_mut(&mut self, i: Idx, val: Val) -> Result<()> {
        self.try_index_mut_device(i, val, StreamOrDevice::default())
    }
}
// TODO: should `Val` impl `AsRef<Array>` or `Into<Array>`?

/// Trait for custom mutable indexing operations.
pub trait IndexMutOp<Idx, Val>: TryIndexMutOp<Idx, Val> {
    /// Index the array with the given index and set the value.
    fn index_mut_device(&mut self, i: Idx, val: Val, stream: impl AsRef<Stream>) {
        self.try_index_mut_device(i, val, stream).unwrap()
    }

    /// Index the array with the given index and set the value.
    fn index_mut(&mut self, i: Idx, val: Val) {
        self.try_index_mut(i, val).unwrap()
    }
}

impl<T, Idx, Val> IndexMutOp<Idx, Val> for T where T: TryIndexMutOp<Idx, Val> {}

/// Trait for custom indexing operations.
pub trait ArrayIndex<'a> {
    /// `mlx` allows out of bounds indexing.
    fn index_op(self) -> ArrayIndexOp<'a>;
}

/* -------------------------------------------------------------------------- */
/*                               Implementation                               */
/* -------------------------------------------------------------------------- */

// Implement public bindings
impl Array {
    /// Take elements along an axis.
    ///
    /// The elements are taken from `indices` along the specified axis. If the axis is not specified
    /// the array is treated as a flattened 1-D array prior to performing the take.
    ///
    /// See [`Array::take_all`] for the flattened array.
    ///
    /// # Params
    ///
    /// - `indices`: The indices to take from the array.
    /// - `axis`: The axis along which to take the elements.
    pub fn take_axis(&self, indices: impl AsRef<Array>, axis: i32) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_take_axis(
                res,
                self.as_ptr(),
                indices.as_ref().as_ptr(),
                axis,
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compatibility shim for [`take_axis`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `take_axis`"
    )]
    pub fn take_axis_device(
        &self,
        indices: impl AsRef<Array>,
        axis: i32,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.take_axis(indices, axis))
    }

    /// Take elements from flattened 1-D array.
    ///
    /// # Params
    ///
    /// - `indices`: The indices to take from the array.
    pub fn take(&self, indices: impl AsRef<Array>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_take(
                res,
                self.as_ptr(),
                indices.as_ref().as_ptr(),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compatibility shim for [`take`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `take`"
    )]
    pub fn take_device(
        &self,
        indices: impl AsRef<Array>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.take(indices))
    }

    /// Take values along an axis at the specified indices.
    ///
    /// If no axis is specified, the array is flattened to 1D prior to the indexing operation.
    ///
    /// # Params
    ///
    /// - `indices`: The indices to take from the array.
    /// - `axis`: Axis in the input to take the values from.
    pub fn take_along_axis(
        &self,
        indices: impl AsRef<Array>,
        axis: impl Into<Option<i32>>,
    ) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        let (input, axis) = match axis.into() {
            None => (Cow::Owned(self.reshape(&[-1])?), 0),
            Some(ax) => (Cow::Borrowed(self), ax),
        };

        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_take_along_axis(
                res,
                input.as_ptr(),
                indices.as_ref().as_ptr(),
                axis,
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compatibility shim for [`take_along_axis`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `take_along_axis`"
    )]
    pub fn take_along_axis_device(
        &self,
        indices: impl AsRef<Array>,
        axis: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.take_along_axis(indices, axis))
    }

    /// Put values along an axis at the specified indices.
    ///
    /// If no axis is specified, the array is flattened to 1D prior to the indexing operation.
    ///
    /// # Params
    /// - indices: Indices array. These should be broadcastable with the input array excluding the `axis` dimension.
    /// - values: Values array. These should be broadcastable with the indices.
    /// - axis: Axis in the destination to put the values to.
    /// - stream: stream or device to evaluate on.
    pub fn put_along_axis(
        &self,
        indices: impl AsRef<Array>,
        values: impl AsRef<Array>,
        axis: impl Into<Option<i32>>,
    ) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        match axis.into() {
            None => {
                let input = self.reshape(&[-1])?;
                let array = Array::try_from_op(|res| unsafe {
                    mlx_sys::mlx_put_along_axis(
                        res,
                        input.as_ptr(),
                        indices.as_ref().as_ptr(),
                        values.as_ref().as_ptr(),
                        0,
                        stream.as_ref().as_ptr(),
                    )
                })?;
                let array = array.reshape(self.shape())?;
                Ok(array)
            }
            Some(ax) => Array::try_from_op(|res| unsafe {
                mlx_sys::mlx_put_along_axis(
                    res,
                    self.as_ptr(),
                    indices.as_ref().as_ptr(),
                    values.as_ref().as_ptr(),
                    ax,
                    stream.as_ref().as_ptr(),
                )
            }),
        }
    }

    /// Compatibility shim for [`put_along_axis`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `put_along_axis`"
    )]
    pub fn put_along_axis_device(
        &self,
        indices: impl AsRef<Array>,
        values: impl AsRef<Array>,
        axis: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || {
            self.put_along_axis(indices, values, axis)
        })
    }
}

/// Indices of the maximum values along the axis.
///
/// See [`argmax_all`] for the flattened array.
///
/// # Params
///
/// - `a`: The input array.
/// - `axis`: Axis to reduce over
/// - `keep_dims`: Keep reduced axes as singleton dimensions, defaults to False.
pub fn argmax_axis(
    a: impl AsRef<Array>,
    axis: i32,
    keep_dims: impl Into<Option<bool>>,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    let keep_dims = keep_dims.into().unwrap_or(false);

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_argmax_axis(
            res,
            a.as_ref().as_ptr(),
            axis,
            keep_dims,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`argmax_axis`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::ops::indexing"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `argmax_axis`"
)]
pub fn argmax_axis_device(
    a: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || argmax_axis(a, axis, keep_dims))
}

/// Indices of the maximum value over the entire array.
///
/// # Params
///
/// - `a`: The input array.
/// - `keep_dims`: Keep reduced axes as singleton dimensions, defaults to False.
pub fn argmax(a: impl AsRef<Array>, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    let keep_dims = keep_dims.into().unwrap_or(false);

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_argmax(
            res,
            a.as_ref().as_ptr(),
            keep_dims,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`argmax`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::ops::indexing"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `argmax`"
)]
pub fn argmax_device(
    a: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || argmax(a, keep_dims))
}

/// Indices of the minimum values along the axis.
///
/// See [`argmin_all`] for the flattened array.
///
/// # Params
///
/// - `a`: The input array.
/// - `axis`: Axis to reduce over.
/// - `keep_dims`: Keep reduced axes as singleton dimensions, defaults to False.
pub fn argmin_axis(
    a: impl AsRef<Array>,
    axis: i32,
    keep_dims: impl Into<Option<bool>>,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    let keep_dims = keep_dims.into().unwrap_or(false);

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_argmin_axis(
            res,
            a.as_ref().as_ptr(),
            axis,
            keep_dims,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`argmin_axis`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::ops::indexing"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `argmin_axis`"
)]
pub fn argmin_axis_device(
    a: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || argmin_axis(a, axis, keep_dims))
}

/// Indices of the minimum value over the entire array.
///
/// # Params
///
/// - `a`: The input array.
/// - `keep_dims`: Keep reduced axes as singleton dimensions, defaults to False.
pub fn argmin(a: impl AsRef<Array>, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    let keep_dims = keep_dims.into().unwrap_or(false);

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_argmin(
            res,
            a.as_ref().as_ptr(),
            keep_dims,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`argmin`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::ops::indexing"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `argmin`"
)]
pub fn argmin_device(
    a: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || argmin(a, keep_dims))
}

/// See [`Array::take_along_axis`]
pub fn take_along_axis(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    axis: impl Into<Option<i32>>,
) -> Result<Array> {
    a.as_ref().take_along_axis(indices, axis)
}

/// Compatibility shim for [`take_along_axis`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::ops::indexing"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `take_along_axis`"
)]
pub fn take_along_axis_device(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    #[optional] axis: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || take_along_axis(a, indices, axis))
}

/// See [`Array::put_along_axis`]
pub fn put_along_axis(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    values: impl AsRef<Array>,
    axis: impl Into<Option<i32>>,
) -> Result<Array> {
    a.as_ref().put_along_axis(indices, values, axis)
}

/// Compatibility shim for [`put_along_axis`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::ops::indexing"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `put_along_axis`"
)]
pub fn put_along_axis_device(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    values: impl AsRef<Array>,
    #[optional] axis: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || put_along_axis(a, indices, values, axis))
}

/// See [`Array::take`]
pub fn take_axis(a: impl AsRef<Array>, indices: impl AsRef<Array>, axis: i32) -> Result<Array> {
    a.as_ref().take_axis(indices, axis)
}

/// Compatibility shim for [`take_axis`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::ops::indexing"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `take_axis`"
)]
pub fn take_axis_device(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    axis: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || take_axis(a, indices, axis))
}

/// See [`Array::take_all`]
pub fn take(a: impl AsRef<Array>, indices: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().take(indices)
}

/// Compatibility shim for [`take`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::ops::indexing"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `take`"
)]
pub fn take_device(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || take(a, indices))
}

/// Returns the `k` largest elements from the input along a given axis.
///
/// The elements will not necessarily be in sorted order.
///
/// See [`topk_all`] for the flattened array.
///
/// # Params
///
/// - `a`: The input array.
/// - `k`: The number of elements to return.
/// - `axis`: Axis to sort over. Default to `-1` if not specified.
pub fn topk_axis(a: impl AsRef<Array>, k: i32, axis: i32) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_topk_axis(res, a.as_ref().as_ptr(), k, axis, stream.as_ref().as_ptr())
    })
}

/// Compatibility shim for [`topk_axis`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::ops::indexing"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `topk_axis`"
)]
pub fn topk_axis_device(
    a: impl AsRef<Array>,
    k: i32,
    axis: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || topk_axis(a, k, axis))
}

/// Returns the `k` largest elements from the flattened input array.
pub fn topk(a: impl AsRef<Array>, k: i32) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_topk(res, a.as_ref().as_ptr(), k, stream.as_ref().as_ptr())
    })
}

/// Compatibility shim for [`topk`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::ops::indexing"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `topk`"
)]
pub fn topk_device(
    a: impl AsRef<Array>,
    k: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || topk(a, k))
}

/// Scatter updates to the array at the given indices along a single axis.
///
/// # Params
///
/// - `a`: Input array
/// - `indices`: Indices array specifying positions to scatter into
/// - `updates`: Values to scatter
/// - `axis`: The axis along which to scatter
pub fn scatter_single(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    updates: impl AsRef<Array>,
    axis: i32,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_scatter_single(
            res,
            a.as_ref().as_ptr(),
            indices.as_ref().as_ptr(),
            updates.as_ref().as_ptr(),
            axis,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`scatter_single`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::ops::indexing"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `scatter_single`"
)]
pub fn scatter_single_device(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    updates: impl AsRef<Array>,
    axis: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || {
        scatter_single(a, indices, updates, axis)
    })
}

/// Scatter-add updates to the array at the given indices along a single axis.
///
/// Adds the update values to the existing values at the specified indices.
///
/// # Params
///
/// - `a`: Input array
/// - `indices`: Indices array specifying positions to scatter into
/// - `updates`: Values to add
/// - `axis`: The axis along which to scatter
pub fn scatter_add_single(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    updates: impl AsRef<Array>,
    axis: i32,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_scatter_add_single(
            res,
            a.as_ref().as_ptr(),
            indices.as_ref().as_ptr(),
            updates.as_ref().as_ptr(),
            axis,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`scatter_add_single`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::ops::indexing"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `scatter_add_single`"
)]
pub fn scatter_add_single_device(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    updates: impl AsRef<Array>,
    axis: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || {
        scatter_add_single(a, indices, updates, axis)
    })
}

/// Scatter-max updates to the array at the given indices along a single axis.
///
/// Takes the maximum of the update values and existing values at the specified indices.
///
/// # Params
///
/// - `a`: Input array
/// - `indices`: Indices array specifying positions to scatter into
/// - `updates`: Values to compare
/// - `axis`: The axis along which to scatter
pub fn scatter_max_single(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    updates: impl AsRef<Array>,
    axis: i32,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_scatter_max_single(
            res,
            a.as_ref().as_ptr(),
            indices.as_ref().as_ptr(),
            updates.as_ref().as_ptr(),
            axis,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`scatter_max_single`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::ops::indexing"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `scatter_max_single`"
)]
pub fn scatter_max_single_device(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    updates: impl AsRef<Array>,
    axis: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || {
        scatter_max_single(a, indices, updates, axis)
    })
}

/// Scatter-min updates to the array at the given indices along a single axis.
///
/// Takes the minimum of the update values and existing values at the specified indices.
///
/// # Params
///
/// - `a`: Input array
/// - `indices`: Indices array specifying positions to scatter into
/// - `updates`: Values to compare
/// - `axis`: The axis along which to scatter
pub fn scatter_min_single(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    updates: impl AsRef<Array>,
    axis: i32,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_scatter_min_single(
            res,
            a.as_ref().as_ptr(),
            indices.as_ref().as_ptr(),
            updates.as_ref().as_ptr(),
            axis,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`scatter_min_single`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::ops::indexing"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `scatter_min_single`"
)]
pub fn scatter_min_single_device(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    updates: impl AsRef<Array>,
    axis: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || {
        scatter_min_single(a, indices, updates, axis)
    })
}

/// Scatter-prod updates to the array at the given indices along a single axis.
///
/// Multiplies the update values with existing values at the specified indices.
///
/// # Params
///
/// - `a`: Input array
/// - `indices`: Indices array specifying positions to scatter into
/// - `updates`: Values to multiply
/// - `axis`: The axis along which to scatter
pub fn scatter_prod_single(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    updates: impl AsRef<Array>,
    axis: i32,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_scatter_prod_single(
            res,
            a.as_ref().as_ptr(),
            indices.as_ref().as_ptr(),
            updates.as_ref().as_ptr(),
            axis,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`scatter_prod_single`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::ops::indexing"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `scatter_prod_single`"
)]
pub fn scatter_prod_single_device(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    updates: impl AsRef<Array>,
    axis: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || {
        scatter_prod_single(a, indices, updates, axis)
    })
}

/// Gather elements from the array at the given indices along a single axis.
///
/// # Params
///
/// - `a`: Input array
/// - `indices`: Indices array specifying positions to gather from
/// - `axis`: The axis along which to gather
/// - `slice_sizes`: The sizes of the slices to gather
pub fn gather_single(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    axis: i32,
    slice_sizes: &[i32],
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_gather_single(
            res,
            a.as_ref().as_ptr(),
            indices.as_ref().as_ptr(),
            axis,
            slice_sizes.as_ptr(),
            slice_sizes.len(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`gather_single`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::ops::indexing"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `gather_single`"
)]
pub fn gather_single_device(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    axis: i32,
    slice_sizes: &[i32],
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || {
        gather_single(a, indices, axis, slice_sizes)
    })
}

/// Scatter values into an array at locations where mask is true.
///
/// # Params
///
/// - `a`: Input array
/// - `mask`: Boolean mask array indicating where to scatter
/// - `src`: Source values to scatter
pub fn masked_scatter(
    a: impl AsRef<Array>,
    mask: impl AsRef<Array>,
    src: impl AsRef<Array>,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_masked_scatter(
            res,
            a.as_ref().as_ptr(),
            mask.as_ref().as_ptr(),
            src.as_ref().as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`masked_scatter`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::ops::indexing"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `masked_scatter`"
)]
pub fn masked_scatter_device(
    a: impl AsRef<Array>,
    mask: impl AsRef<Array>,
    src: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || masked_scatter(a, mask, src))
}

/* -------------------------------------------------------------------------- */
/*                              Helper functions                              */
/* -------------------------------------------------------------------------- */
fn count_non_new_axis_operations(operations: &[ArrayIndexOp]) -> usize {
    operations
        .iter()
        .filter(|op| !matches!(op, ArrayIndexOp::ExpandDims))
        .count()
}
fn expand_ellipsis_operations<'a>(
    ndim: usize,
    operations: &'a [ArrayIndexOp<'a>],
) -> Cow<'a, [ArrayIndexOp<'a>]> {
    let ellipsis_count = operations
        .iter()
        .filter(|op| matches!(op, ArrayIndexOp::Ellipsis))
        .count();
    if ellipsis_count == 0 {
        return Cow::Borrowed(operations);
    }

    if ellipsis_count > 1 {
        panic!("Indexing with multiple ellipsis is not supported");
    }

    let ellipsis_pos = operations
        .iter()
        .position(|op| matches!(op, ArrayIndexOp::Ellipsis))
        .unwrap();
    let prefix = &operations[..ellipsis_pos];
    let suffix = &operations[(ellipsis_pos + 1)..];
    let expand_range =
        count_non_new_axis_operations(prefix)..(ndim - count_non_new_axis_operations(suffix));
    let expand = expand_range.map(|_| (..).index_op());

    let mut expanded = Vec::with_capacity(ndim);
    expanded.extend_from_slice(prefix);
    expanded.extend(expand);
    expanded.extend_from_slice(suffix);

    Cow::Owned(expanded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{array, ops::reshape, Array};

    // Tests adapted from C++ `ops_tests.cpp/test scatter`
    #[test]
    fn test_scatter_single() {
        // Single element scatter
        let input = Array::zeros::<f32>(&[4]).unwrap();
        let indices = Array::from_slice(&[0u32, 1], &[2]);
        let updates = Array::ones::<f32>(&[2, 1]).unwrap();
        let out = scatter_single(&input, &indices, &updates, 0).unwrap();
        let expected = array!([1.0f32, 1.0, 0.0, 0.0]);
        assert!(out.all_close(&expected, 1e-5, 1e-5, None).unwrap());
    }

    #[test]
    fn test_scatter_add_single() {
        // Single element scatter add
        let input = Array::ones::<f32>(&[4]).unwrap();
        let indices = Array::from_slice(&[0u32, 0, 3], &[3]);
        let updates = Array::ones::<f32>(&[3, 1]).unwrap();
        let out = scatter_add_single(&input, &indices, &updates, 0).unwrap();
        let expected = array!([3.0f32, 1.0, 1.0, 2.0]);
        assert!(out.all_close(&expected, 1e-5, 1e-5, None).unwrap());
    }

    #[test]
    fn test_scatter_max_single() {
        // Single element scatter max
        let input = Array::ones::<f32>(&[4]).unwrap();
        let indices = Array::from_slice(&[0u32, 0, 3], &[3]);
        let updates = reshape(array!([1.0f32, 6.0, -2.0]), &[3, 1]).unwrap();
        let out = scatter_max_single(&input, &indices, &updates, 0).unwrap();
        let expected = array!([6.0f32, 1.0, 1.0, 1.0]);
        assert!(out.all_close(&expected, 1e-5, 1e-5, None).unwrap());
    }

    #[test]
    fn test_scatter_min_single() {
        // Single element scatter min
        let input = Array::ones::<f32>(&[4]).unwrap();
        let indices = Array::from_slice(&[0u32, 0, 3], &[3]);
        let updates = reshape(array!([1.0f32, -6.0, 2.0]), &[3, 1]).unwrap();
        let out = scatter_min_single(&input, &indices, &updates, 0).unwrap();
        let expected = array!([-6.0f32, 1.0, 1.0, 1.0]);
        assert!(out.all_close(&expected, 1e-5, 1e-5, None).unwrap());
    }

    #[test]
    fn test_scatter_prod_single() {
        // Single element scatter prod
        let input = Array::ones::<f32>(&[4]).unwrap();
        let indices = Array::from_slice(&[0u32, 0, 3], &[3]);
        let updates = Array::full::<f32>(&[3, 1], array!(2.0f32)).unwrap();
        let out = scatter_prod_single(&input, &indices, &updates, 0).unwrap();
        let expected = array!([4.0f32, 1.0, 1.0, 2.0]);
        assert!(out.all_close(&expected, 1e-5, 1e-5, None).unwrap());
    }

    #[test]
    fn test_gather_single() {
        // Simple gather test
        let input = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[4]);
        let indices = Array::from_slice(&[1u32, 3], &[2]);
        let out = gather_single(&input, &indices, 0, &[1]).unwrap();
        let expected = array!([[1.0f32], [3.0]]);
        assert!(out.all_close(&expected, 1e-5, 1e-5, None).unwrap());
    }

    #[test]
    fn test_masked_scatter() {
        // Simple masked scatter test
        let input = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);
        let mask = Array::from_slice(&[true, false, true, false], &[4]);
        let src = Array::from_slice(&[10.0f32, 20.0], &[2]);
        let out = masked_scatter(&input, &mask, &src).unwrap();
        let expected = array!([10.0f32, 2.0, 20.0, 4.0]);
        assert!(out.all_close(&expected, 1e-5, 1e-5, None).unwrap());
    }
}
