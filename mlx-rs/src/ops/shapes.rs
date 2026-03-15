use mlx_internal_macros::{default_device, generate_macro};
use smallvec::SmallVec;

use crate::{
    constants::DEFAULT_STACK_VEC_LEN,
    error::Result,
    utils::{guard::Guarded, IntoOption, VectorArray},
    Array, Stream,
};

impl Array {
    /// See [`expand_dims()`].
    #[default_device]
    pub fn expand_dims_device(&self, axis: i32, stream: impl AsRef<Stream>) -> Result<Array> {
        expand_dims_device(self, axis, stream)
    }

    /// See [`expand_dims_axes()`].
    #[default_device]
    pub fn expand_dims_axes_device(
        &self,
        axes: &[i32],
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        expand_dims_axes_device(self, axes, stream)
    }

    /// See [`flatten`].
    #[default_device]
    pub fn flatten_device(
        &self,
        start_axis: impl Into<Option<i32>>,
        end_axis: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        flatten_device(self, start_axis, end_axis, stream)
    }

    /// See [`reshape`].
    #[default_device]
    pub fn reshape_device(&self, shape: &[i32], stream: impl AsRef<Stream>) -> Result<Array> {
        reshape_device(self, shape, stream)
    }

    /// See [`squeeze_axes()`].
    #[default_device]
    pub fn squeeze_axes_device(&self, axes: &[i32], stream: impl AsRef<Stream>) -> Result<Array> {
        squeeze_axes_device(self, axes, stream)
    }

    /// See [`squeeze()`].
    #[default_device]
    pub fn squeeze_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        squeeze_device(self, stream)
    }

    /// See [`as_strided`]
    #[default_device]
    pub fn as_strided_device<'a>(
        &'a self,
        shape: impl IntoOption<&'a [i32]>,
        strides: impl IntoOption<&'a [i64]>,
        offset: impl Into<Option<usize>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        as_strided_device(self, shape, strides, offset, stream)
    }

    /// See [`at_least_1d`]
    #[default_device]
    pub fn at_least_1d_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        at_least_1d_device(self, stream)
    }

    /// See [`at_least_2d`]
    #[default_device]
    pub fn at_least_2d_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        at_least_2d_device(self, stream)
    }

    /// See [`at_least_3d`]
    #[default_device]
    pub fn at_least_3d_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        at_least_3d_device(self, stream)
    }

    /// See [`contiguous`]
    #[default_device]
    pub fn contiguous_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        contiguous_device(self, stream)
    }

    /// See [`move_axis`]
    #[default_device]
    pub fn move_axis_device(
        &self,
        src: i32,
        dst: i32,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        move_axis_device(self, src, dst, stream)
    }

    /// See [`split`]
    #[default_device]
    pub fn split_axis_device(
        &self,
        indices: &[i32],
        axis: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Vec<Array>> {
        split_sections_device(self, indices, axis, stream)
    }

    /// See [`split`]
    #[default_device]
    pub fn split_device(
        &self,
        num_parts: i32,
        axis: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Vec<Array>> {
        split_device(self, num_parts, axis, stream)
    }

    /// See [`swap_axes`]
    #[default_device]
    pub fn swap_axes_device(
        &self,
        axis1: i32,
        axis2: i32,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        swap_axes_device(self, axis1, axis2, stream)
    }

    /// See [`transpose_axes`]
    #[default_device]
    pub fn transpose_axes_device(&self, axes: &[i32], stream: impl AsRef<Stream>) -> Result<Array> {
        transpose_axes_device(self, axes, stream)
    }

    /// See [`transpose`]
    #[default_device]
    pub fn transpose_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        transpose_device(self, stream)
    }

    /// [`transpose_axes`] and unwrap the result.
    pub fn t(&self) -> Array {
        self.transpose().unwrap()
    }
}

fn resolve_strides(
    shape: &[i32],
    strides: Option<&[i64]>,
) -> SmallVec<[i64; DEFAULT_STACK_VEC_LEN]> {
    match strides {
        Some(strides) => SmallVec::from_slice(strides),
        None => {
            let result = shape
                .iter()
                .rev()
                .scan(1, |acc, &dim| {
                    let result = *acc;
                    *acc *= dim as i64;
                    Some(result)
                })
                .collect::<SmallVec<[i64; DEFAULT_STACK_VEC_LEN]>>();
            result.into_iter().rev().collect()
        }
    }
}

/// Broadcast a vector of arrays against one another. Returns an error if the shapes are
/// broadcastable.
///
/// # Params
///
/// - `arrays`: The arrays to broadcast.
#[generate_macro]
#[default_device]
pub fn broadcast_arrays_device(
    arrays: &[impl AsRef<Array>],
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Vec<Array>> {
    let c_vec = VectorArray::try_from_iter(arrays.iter())?;
    Vec::<Array>::try_from_op(|res| unsafe {
        mlx_sys::mlx_broadcast_arrays(res, c_vec.as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Create a view into the array with the given shape and strides.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let x = Array::from_iter(0..10, &[10]);
/// let y = as_strided(&x, &[3, 3], &[1, 1], 0);
/// ```
#[generate_macro]
#[default_device]
pub fn as_strided_device<'a>(
    a: impl AsRef<Array>,
    #[optional] shape: impl IntoOption<&'a [i32]>,
    #[optional] strides: impl IntoOption<&'a [i64]>,
    #[optional] offset: impl Into<Option<usize>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let shape = shape.into_option().unwrap_or(a.shape());
    let resolved_strides = resolve_strides(shape, strides.into_option());
    let offset = offset.into().unwrap_or(0);

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_as_strided(
            res,
            a.as_ptr(),
            shape.as_ptr(),
            shape.len(),
            resolved_strides.as_ptr(),
            resolved_strides.len(),
            offset,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Broadcast an array to the given shape. Returns an error if the shapes are not broadcastable.
///
/// # Params
///
/// - `a`: The input array.
/// - `shape`: The shape to broadcast to.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let x = Array::from_f32(2.3);
/// let result = broadcast_to(&x, &[1, 1]);
/// ```
#[generate_macro]
#[default_device]
pub fn broadcast_to_device(
    a: impl AsRef<Array>,
    shape: &[i32],
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_broadcast_to(
            res,
            a.as_ref().as_ptr(),
            shape.as_ptr(),
            shape.len(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Concatenate the arrays along the given axis. Returns an error if the shapes are invalid.
///
/// # Params
///
/// - `arrays`: The arrays to concatenate.
/// - `axis`: The axis to concatenate along.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let x = Array::from_iter(0..4, &[2, 2]);
/// let y = Array::from_iter(4..8, &[2, 2]);
/// let result = concatenate_axis(&[x, y], 0);
/// ```
#[generate_macro]
#[default_device]
pub fn concatenate_axis_device(
    arrays: &[impl AsRef<Array>],
    axis: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let c_arrays = VectorArray::try_from_iter(arrays.iter())?;
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_concatenate_axis(res, c_arrays.as_ptr(), axis, stream.as_ref().as_ptr())
    })
}

/// Concatenate the arrays along the first axis. Returns an error if the shapes are invalid.
#[generate_macro]
#[default_device]
pub fn concatenate_device(
    arrays: &[impl AsRef<Array>],
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let c_arrays = VectorArray::try_from_iter(arrays.iter())?;
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_concatenate(res, c_arrays.as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Add a size one dimension at the given axis, returns an error if the axes are invalid.
///
/// # Params
///
/// - `a`: The input array.
/// - `axes`: The index of the inserted dimensions.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let x = Array::zeros::<i32>(&[2, 2]).unwrap();
/// let result = expand_dims_axes(&x, &[0]);
/// ```
#[generate_macro]
#[default_device]
pub fn expand_dims_axes_device(
    a: impl AsRef<Array>,
    axes: &[i32],
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_expand_dims_axes(
            res,
            a.as_ref().as_ptr(),
            axes.as_ptr(),
            axes.len(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Similar to [`expand_dims_axes`], but only takes a single axis.
#[generate_macro]
#[default_device]
pub fn expand_dims_device(
    a: impl AsRef<Array>,
    axis: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_expand_dims(res, a.as_ref().as_ptr(), axis, stream.as_ref().as_ptr())
    })
}

/// Flatten an array. Returns an error if the axes are invalid.
///
/// The axes flattened will be between `start_axis` and `end_axis`, inclusive. Negative axes are
/// supported. After converting negative axis to positive, axes outside the valid range will be
/// clamped to a valid value, `start_axis` to `0` and `end_axis` to `ndim - 1`.
///
/// # Params
///
/// - `a`: The input array.
/// - `start_axis`: The first axis to flatten. Default is `0` if not provided.
/// - `end_axis`: The last axis to flatten. Default is `-1` if not provided.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let x = Array::zeros::<i32>(&[2, 2, 2]).unwrap();
/// let y = flatten(&x, None, None);
/// ```
#[generate_macro]
#[default_device]
pub fn flatten_device(
    a: impl AsRef<Array>,
    #[optional] start_axis: impl Into<Option<i32>>,
    #[optional] end_axis: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let start_axis = start_axis.into().unwrap_or(0);
    let end_axis = end_axis.into().unwrap_or(-1);

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_flatten(
            res,
            a.as_ref().as_ptr(),
            start_axis,
            end_axis,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Unflatten an axis of an array to a shape.
///
/// # Params
///
/// - `a`: input array
/// - `axis`: axis to unflatten
/// - `shape`: shape to unflatten into
#[generate_macro]
#[default_device]
pub fn unflatten_device(
    a: impl AsRef<Array>,
    axis: i32,
    shape: &[i32],
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_unflatten(
            res,
            a.as_ref().as_ptr(),
            axis,
            shape.as_ptr(),
            shape.len(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Reshape an array while preserving the size. Returns an error if the new shape is invalid.
///
/// # Params
///
/// - `a`: The input array.
/// - `shape`: New shape.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let x = Array::zeros::<i32>(&[2, 2]).unwrap();
/// let result = reshape(&x, &[4]);
/// ```
#[generate_macro]
#[default_device]
pub fn reshape_device(
    a: impl AsRef<Array>,
    shape: &[i32],
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_reshape(
            res,
            a.as_ref().as_ptr(),
            shape.as_ptr(),
            shape.len(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Remove length one axes from an array. Returns an error if the axes are invalid.
///
/// # Params
///
/// - `a`: The input array.
/// - `axes`: Axes to remove. If `None`, all length one axes will be removed.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let x = Array::zeros::<i32>(&[1, 2, 1, 3]).unwrap();
/// let result = squeeze(&x);
/// ```
#[generate_macro]
#[default_device]
pub fn squeeze_axes_device(
    a: impl AsRef<Array>,
    axes: &[i32],
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_squeeze_axes(
            res,
            a.as_ptr(),
            axes.as_ptr(),
            axes.len(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Similar to [`squeeze_axes`], but removes all length one axes.
#[generate_macro]
#[default_device]
pub fn squeeze_device(
    a: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_squeeze(res, a.as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Convert array to have at least one dimension.
///
/// # Params
///
/// - `a`: The input array.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let x = Array::from_int(1);
/// let out = at_least_1d(&x);
/// ```
#[generate_macro]
#[default_device]
pub fn at_least_1d_device(
    a: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_atleast_1d(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Convert array to have at least two dimensions.
///
/// # Params
///
/// - `a`: The input array.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let x = Array::from_int(1);
/// let out = at_least_2d(&x);
/// ```
#[generate_macro]
#[default_device]
pub fn at_least_2d_device(
    a: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_atleast_2d(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Convert array to have at least three dimensions.
///
/// # Params
///
/// - `a`: The input array.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let x = Array::from_int(1);
/// let out = at_least_3d(&x);
/// ```
#[generate_macro]
#[default_device]
pub fn at_least_3d_device(
    a: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_atleast_3d(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Move an axis to a new position. Returns an error if the axes are invalid.
///
/// # Params
///
/// - `a`: The input array.
/// - `src`: Specifies the source axis.
/// - `dst`: Specifies the destination axis.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let a = Array::zeros::<i32>(&[2, 3, 4]).unwrap();
/// let result = move_axis(&a, 0, 2);
/// ```
#[generate_macro]
#[default_device]
pub fn move_axis_device(
    a: impl AsRef<Array>,
    src: i32,
    dst: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_moveaxis(res, a.as_ref().as_ptr(), src, dst, stream.as_ref().as_ptr())
    })
}

/// Split an array along a given axis. Returns an error if the indices are invalid.
///
/// # Params
///
/// - `a`: The input array.
/// - `indices`: The indices to split at.
/// - `axis`: The axis to split along. Default is `0` if not provided.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let a = Array::from_iter(0..10, &[10]);
/// let result = split_sections(&a, &[3, 7], 0);
/// ```
#[generate_macro]
#[default_device]
pub fn split_sections_device(
    a: impl AsRef<Array>,
    indices: &[i32],
    #[optional] axis: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Vec<Array>> {
    let axis = axis.into().unwrap_or(0);
    Vec::<Array>::try_from_op(|res| unsafe {
        mlx_sys::mlx_split_sections(
            res,
            a.as_ref().as_ptr(),
            indices.as_ptr(),
            indices.len(),
            axis,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Split an array into equal parts along a given axis. Returns an error if the array cannot be
/// split into equal parts.
///
/// # Params
///
/// - `a`: The input array.
/// - `num_parts`: The number of parts to split into.
/// - `axis`: The axis to split along. Default is `0` if not provided.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let a = Array::from_iter(0..10, &[10]);
/// let result = split(&a, 2, 0);
/// ```
#[generate_macro]
#[default_device]
pub fn split_device(
    a: impl AsRef<Array>,
    num_parts: i32,
    #[optional] axis: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Vec<Array>> {
    let axis = axis.into().unwrap_or(0);
    Vec::<Array>::try_from_op(|res| unsafe {
        mlx_sys::mlx_split(
            res,
            a.as_ref().as_ptr(),
            num_parts,
            axis,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Number of padding values to add to the edges of each axis.
#[derive(Debug)]
pub enum PadWidth<'a> {
    /// (before, after) values for all axes.
    Same((i32, i32)),

    /// List of (before, after) values for each axis.
    Widths(&'a [(i32, i32)]),
}

impl From<i32> for PadWidth<'_> {
    fn from(width: i32) -> Self {
        PadWidth::Same((width, width))
    }
}

impl From<(i32, i32)> for PadWidth<'_> {
    fn from(width: (i32, i32)) -> Self {
        PadWidth::Same(width)
    }
}

impl<'a> From<&'a [(i32, i32)]> for PadWidth<'a> {
    fn from(widths: &'a [(i32, i32)]) -> Self {
        PadWidth::Widths(widths)
    }
}

impl<'a, const N: usize> From<&'a [(i32, i32); N]> for PadWidth<'a> {
    fn from(widths: &'a [(i32, i32); N]) -> Self {
        PadWidth::Widths(widths)
    }
}

impl PadWidth<'_> {
    fn low_pads(&self, ndim: usize) -> SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> {
        match self {
            PadWidth::Same((low, _high)) => (0..ndim).map(|_| *low).collect(),
            PadWidth::Widths(widths) => widths.iter().map(|(low, _high)| *low).collect(),
        }
    }

    fn high_pads(&self, ndim: usize) -> SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> {
        match self {
            PadWidth::Same((_low, high)) => (0..ndim).map(|_| *high).collect(),
            PadWidth::Widths(widths) => widths.iter().map(|(_low, high)| *high).collect(),
        }
    }
}

/// The padding mode.
#[derive(Debug)]
pub enum PadMode {
    /// Pad with a constant value.
    Constant,

    /// Pad with the edge value.
    Edge,
}

impl PadMode {
    unsafe fn as_c_str(&self) -> *const i8 {
        static CONSTANT: &[u8] = b"constant\0";
        static EDGE: &[u8] = b"edge\0";

        match self {
            PadMode::Constant => CONSTANT.as_ptr() as *const _,
            PadMode::Edge => EDGE.as_ptr() as *const _,
        }
    }
}

/// Pad an array with a constant value. Returns an error if the width is invalid.
///
/// # Params
///
/// - `a`: The input array.
/// - `width`: Number of padded values to add to the edges of each axis:`((before_1, after_1),
///   (before_2, after_2), ..., (before_N, after_N))`. If a single pair of integers is passed then
///   `(before_i, after_i)` are all the same. If a single integer or tuple with a single integer is
///   passed then all axes are extended by the same number on each side.
/// - `value`: The value to pad the array with. Default is `0` if not provided.
/// - `mode`: The padding mode. Default is `PadMode::Constant` if not provided.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let a = Array::from_iter(0..4, &[2, 2]);
/// let result = pad(&a, 1, Array::from_int(0), None);
/// ```
#[generate_macro]
#[default_device]
pub fn pad_device<'a>(
    a: impl AsRef<Array>,
    #[optional] width: impl Into<PadWidth<'a>>,
    #[optional] value: impl Into<Option<Array>>,
    #[optional] mode: impl Into<Option<PadMode>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let width = width.into();
    let ndim = a.ndim();
    let axes: SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> = (0..ndim).map(|i| i as i32).collect();
    let low_pads = width.low_pads(ndim);
    let high_pads = width.high_pads(ndim);
    let value = value
        .into()
        .map(Ok)
        .unwrap_or_else(|| Array::from_int(0).as_dtype(a.dtype()))?;
    let mode = mode.into().unwrap_or(PadMode::Constant);

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_pad(
            res,
            a.as_ptr(),
            axes.as_ptr(),
            axes.len(),
            low_pads.as_ptr(),
            low_pads.len(),
            high_pads.as_ptr(),
            high_pads.len(),
            value.as_ptr(),
            mode.as_c_str(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Stacks the arrays along a new axis. Returns an error if the arguments are invalid.
///
/// # Params
///
/// - `arrays`: The input arrays.
/// - `axis`: The axis in the result array along which the input arrays are stacked.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let a = Array::from_iter(0..4, &[2, 2]);
/// let b = Array::from_iter(4..8, &[2, 2]);
/// let result = stack_axis(&[&a, &b], 0);
/// ```
#[generate_macro]
#[default_device]
pub fn stack_axis_device(
    arrays: &[impl AsRef<Array>],
    axis: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let c_vec = VectorArray::try_from_iter(arrays.iter())?;
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_stack_axis(res, c_vec.as_ptr(), axis, stream.as_ref().as_ptr())
    })
}

/// Stacks the arrays along a new axis. Returns an error if the arrays have different shapes.
///
/// # Params
///
/// - `arrays`: The input arrays.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let a = Array::from_iter(0..4, &[2, 2]);
/// let b = Array::from_iter(4..8, &[2, 2]);
/// let result = stack(&[&a, &b]);
/// ```
#[generate_macro]
#[default_device]
pub fn stack_device(
    arrays: &[impl AsRef<Array>],
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let c_vec = VectorArray::try_from_iter(arrays.iter())?;
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_stack(res, c_vec.as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Swap two axes of an array. Returns an error if the axes are invalid.
///
/// # Params
///
/// - `a`: The input array.
/// - `axis1`: The first axis.
/// - `axis2`: The second axis.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let a = Array::from_iter(0..6, &[2, 3]);
/// let result = swap_axes(&a, 0, 1);
/// ```
#[generate_macro]
#[default_device]
pub fn swap_axes_device(
    a: impl AsRef<Array>,
    axis1: i32,
    axis2: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_swapaxes(
            res,
            a.as_ref().as_ptr(),
            axis1,
            axis2,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Construct an array by repeating `a` the number of times given by `reps`.
///
/// # Params
///
/// - `a`: The input array.
/// - `reps`: The number of repetitions along each axis.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let x = Array::from_slice(&[1, 2, 3], &[3]);
/// let y = tile(&x, &[2]);
/// ```
#[generate_macro]
#[default_device]
pub fn tile_device(
    a: impl AsRef<Array>,
    reps: &[i32],
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_tile(
            res,
            a.as_ref().as_ptr(),
            reps.as_ptr(),
            reps.len(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Transpose the dimensions of the array. Returns an error if the axes are invalid.
///
/// # Params
///
/// - `a`: The input array.
/// - `axes`: Specifies the source axis for each axis in the new array. The default is to reverse
///  the axes.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let x = Array::from_slice(&[1, 2, 3, 4, 5, 6], &[2, 3]);
/// let y1 = transpose_axes(&x, &[0, 1]).unwrap();
/// let y2 = transpose(&x).unwrap();
/// ```
///
/// # See also
///
/// - [`transpose`]
#[generate_macro]
#[default_device]
pub fn transpose_axes_device(
    a: impl AsRef<Array>,
    axes: &[i32],
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_transpose_axes(
            res,
            a.as_ref().as_ptr(),
            axes.as_ptr(),
            axes.len(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Transpose with all axes reversed
#[generate_macro]
#[default_device]
pub fn transpose_device(
    a: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_transpose(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Returns a contiguous array with the same data as the input.
///
/// If the array is already contiguous, it is returned as-is. Otherwise,
/// a new contiguous array is created with the data copied.
///
/// This is useful when you need to call `as_slice()` on an array that may
/// have non-contiguous strides (e.g., after `index()` or `transpose_axes()`).
///
/// # Params
///
/// - `a`: The input array.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
/// use mlx_rs::ops::indexing::IndexOp;
///
/// let x = Array::from_slice(&[1i32, 2, 3, 4, 5, 6], &[2, 3]);
/// let sliced = x.index((.., ..2));  // May be non-contiguous
/// let c = contiguous(&sliced).unwrap();
/// // Now c is guaranteed to be contiguous and safe to use with as_slice()
/// ```
#[generate_macro]
#[default_device]
pub fn contiguous_device(
    a: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_contiguous(
            res,
            a.as_ref().as_ptr(),
            false, // allow_col_major
            stream.as_ref().as_ptr(),
        )
    })
}

// The unit tests below are adapted from
// https://github.com/ml-explore/mlx/blob/main/tests/ops_tests.cpp
#[cfg(test)]
mod tests {
    use crate::{array, Array, Dtype};

    use super::*;

    #[test]
    fn test_squeeze() {
        let a = Array::zeros::<i32>(&[2, 1, 2, 1, 2, 1]).unwrap();
        assert_eq!(
            squeeze_axes(&a, &[1, 3, 5][..]).unwrap().shape(),
            &[2, 2, 2]
        );
        assert_eq!(
            squeeze_axes(&a, &[-1, -3, -5][..]).unwrap().shape(),
            &[2, 2, 2]
        );
        assert_eq!(
            squeeze_axes(&a, &[1][..]).unwrap().shape(),
            &[2, 2, 1, 2, 1]
        );
        assert_eq!(
            squeeze_axes(&a, &[-1][..]).unwrap().shape(),
            &[2, 1, 2, 1, 2]
        );

        assert!(squeeze_axes(&a, &[0][..]).is_err());
        assert!(squeeze_axes(&a, &[2][..]).is_err());
        assert!(squeeze_axes(&a, &[1, 3, 1][..]).is_err());
        assert!(squeeze_axes(&a, &[1, 3, -3][..]).is_err());
    }

    #[test]
    fn test_expand_dims() {
        let a = Array::zeros::<i32>(&[2, 2]).unwrap();
        assert_eq!(expand_dims_axes(&a, &[0][..]).unwrap().shape(), &[1, 2, 2]);
        assert_eq!(expand_dims_axes(&a, &[-1][..]).unwrap().shape(), &[2, 2, 1]);
        assert_eq!(expand_dims_axes(&a, &[1][..]).unwrap().shape(), &[2, 1, 2]);
        assert_eq!(
            expand_dims_axes(&a, &[0, 1, 2]).unwrap().shape(),
            &[1, 1, 1, 2, 2]
        );
        assert_eq!(
            expand_dims_axes(&a, &[0, 1, 2, 5, 6, 7]).unwrap().shape(),
            &[1, 1, 1, 2, 2, 1, 1, 1]
        );

        assert!(expand_dims_axes(&a, &[3]).is_err());
        assert!(expand_dims_axes(&a, &[0, 1, 0]).is_err());
        assert!(expand_dims_axes(&a, &[0, 1, -4]).is_err());
    }

    #[test]
    fn test_flatten() {
        let x = Array::zeros::<i32>(&[2, 3, 4]).unwrap();
        assert_eq!(flatten(&x, None, None).unwrap().shape(), &[2 * 3 * 4]);

        assert_eq!(flatten(&x, 1, 1).unwrap().shape(), &[2, 3, 4]);
        assert_eq!(flatten(&x, 1, 2).unwrap().shape(), &[2, 3 * 4]);
        assert_eq!(flatten(&x, 1, 3).unwrap().shape(), &[2, 3 * 4]);
        assert_eq!(flatten(&x, 1, -1).unwrap().shape(), &[2, 3 * 4]);
        assert_eq!(flatten(&x, -2, -1).unwrap().shape(), &[2, 3 * 4]);
        assert_eq!(flatten(&x, -3, -1).unwrap().shape(), &[2 * 3 * 4]);
        assert_eq!(flatten(&x, -4, -1).unwrap().shape(), &[2 * 3 * 4]);

        assert!(flatten(&x, 2, 1).is_err());

        assert!(flatten(&x, 5, 6).is_err());

        assert!(flatten(&x, -5, -4).is_err());

        let x = Array::from_int(1);
        assert_eq!(flatten(&x, -3, -1).unwrap().shape(), &[1]);
        assert_eq!(flatten(&x, 0, 0).unwrap().shape(), &[1]);
    }

    #[test]
    fn test_unflatten() {
        let a = array!([1, 2, 3, 4]);
        let b = unflatten(&a, 0, &[2, -1]).unwrap();
        let expected = array!([[1, 2], [3, 4]]);
        assert_eq!(b, expected);
    }

    #[test]
    fn test_reshape() {
        let x = Array::from_int(1);
        assert!(reshape(&x, &[]).unwrap().shape().is_empty());
        assert!(reshape(&x, &[2]).is_err());
        let y = reshape(&x, &[1, 1, 1]).unwrap();
        assert_eq!(y.shape(), &[1, 1, 1]);
        let y = reshape(&x, &[-1, 1, 1]).unwrap();
        assert_eq!(y.shape(), &[1, 1, 1]);
        let y = reshape(&x, &[1, 1, -1]).unwrap();
        assert_eq!(y.shape(), &[1, 1, 1]);
        assert!(reshape(&x, &[1, -1, -1]).is_err());
        assert!(reshape(&x, &[2, -1]).is_err());

        let x = Array::zeros::<i32>(&[2, 2, 2]).unwrap();
        let y = reshape(&x, &[8]).unwrap();
        assert_eq!(y.shape(), &[8]);
        assert!(reshape(&x, &[7]).is_err());
        let y = reshape(&x, &[-1]).unwrap();
        assert_eq!(y.shape(), &[8]);
        let y = reshape(&x, &[-1, 2]).unwrap();
        assert_eq!(y.shape(), &[4, 2]);
        assert!(reshape(&x, &[-1, 7]).is_err());

        let x = Array::from_slice::<i32>(&[], &[0]);
        let y = reshape(&x, &[0, 0, 0]).unwrap();
        assert_eq!(y.shape(), &[0, 0, 0]);
        y.eval().unwrap();
        assert_eq!(y.size(), 0);
        assert!(reshape(&x, &[]).is_err());
        assert!(reshape(&x, &[1]).is_err());
        let y = reshape(&x, &[1, 5, 0]).unwrap();
        assert_eq!(y.shape(), &[1, 5, 0]);
    }

    #[test]
    fn test_as_strided() {
        let x = Array::from_iter(0..10, &[10]);
        let y = as_strided(&x, &[3, 3][..], &[1, 1][..], 0).unwrap();
        let expected = Array::from_slice(&[0, 1, 2, 1, 2, 3, 2, 3, 4], &[3, 3]);
        assert_eq!(y, expected);

        let y = as_strided(&x, &[3, 3][..], &[0, 3][..], 0).unwrap();
        let expected = Array::from_slice(&[0, 3, 6, 0, 3, 6, 0, 3, 6], &[3, 3]);
        assert_eq!(y, expected);

        let x = x.reshape(&[2, 5]).unwrap();
        let x = x.transpose_axes(&[1, 0][..]).unwrap();
        let y = as_strided(&x, &[3, 3][..], &[2, 1][..], 1).unwrap();
        let expected = Array::from_slice(&[5, 1, 6, 6, 2, 7, 7, 3, 8], &[3, 3]);
        assert_eq!(y, expected);
    }

    #[test]
    fn test_at_least_1d() {
        let x = Array::from_int(1);
        let out = at_least_1d(&x).unwrap();
        assert_eq!(out.ndim(), 1);
        assert_eq!(out.shape(), &[1]);

        let x = Array::from_slice(&[1, 2, 3], &[3]);
        let out = at_least_1d(&x).unwrap();
        assert_eq!(out.ndim(), 1);
        assert_eq!(out.shape(), &[3]);

        let x = Array::from_slice(&[1, 2, 3], &[3, 1]);
        let out = at_least_1d(&x).unwrap();
        assert_eq!(out.ndim(), 2);
        assert_eq!(out.shape(), &[3, 1]);
    }

    #[test]
    fn test_at_least_2d() {
        let x = Array::from_int(1);
        let out = at_least_2d(&x).unwrap();
        assert_eq!(out.ndim(), 2);
        assert_eq!(out.shape(), &[1, 1]);

        let x = Array::from_slice(&[1, 2, 3], &[3]);
        let out = at_least_2d(&x).unwrap();
        assert_eq!(out.ndim(), 2);
        assert_eq!(out.shape(), &[1, 3]);

        let x = Array::from_slice(&[1, 2, 3], &[3, 1]);
        let out = at_least_2d(&x).unwrap();
        assert_eq!(out.ndim(), 2);
        assert_eq!(out.shape(), &[3, 1]);
    }

    #[test]
    fn test_at_least_3d() {
        let x = Array::from_int(1);
        let out = at_least_3d(&x).unwrap();
        assert_eq!(out.ndim(), 3);
        assert_eq!(out.shape(), &[1, 1, 1]);

        let x = Array::from_slice(&[1, 2, 3], &[3]);
        let out = at_least_3d(&x).unwrap();
        assert_eq!(out.ndim(), 3);
        assert_eq!(out.shape(), &[1, 3, 1]);

        let x = Array::from_slice(&[1, 2, 3], &[3, 1]);
        let out = at_least_3d(&x).unwrap();
        assert_eq!(out.ndim(), 3);
        assert_eq!(out.shape(), &[3, 1, 1]);
    }

    #[test]
    fn test_move_axis() {
        let a = Array::from_int(0);
        assert!(move_axis(&a, 0, 0).is_err());

        let a = Array::zeros::<i32>(&[2]).unwrap();
        assert!(move_axis(&a, 0, 1).is_err());
        assert_eq!(move_axis(&a, 0, 0).unwrap().shape(), &[2]);
        assert_eq!(move_axis(&a, -1, -1).unwrap().shape(), &[2]);

        let a = Array::zeros::<i32>(&[2, 3, 4]).unwrap();
        assert!(move_axis(&a, 0, -4).is_err());
        assert!(move_axis(&a, 0, 3).is_err());
        assert!(move_axis(&a, 3, 0).is_err());
        assert!(move_axis(&a, -4, 0).is_err());
        assert_eq!(move_axis(&a, 0, 2).unwrap().shape(), &[3, 4, 2]);
        assert_eq!(move_axis(&a, 0, 1).unwrap().shape(), &[3, 2, 4]);
        assert_eq!(move_axis(&a, 0, -1).unwrap().shape(), &[3, 4, 2]);
        assert_eq!(move_axis(&a, -2, 2).unwrap().shape(), &[2, 4, 3]);
    }

    #[test]
    fn test_split_equal() {
        let x = Array::from_int(3);
        assert!(split(&x, 0, 0).is_err());

        let x = Array::from_slice(&[0, 1, 2], &[3]);
        assert!(split(&x, 3, 1).is_err());
        assert!(split(&x, -2, 1).is_err());

        let out = split(&x, 3, 0).unwrap();
        assert_eq!(out.len(), 3);

        let mut out = split(&x, 3, -1).unwrap();
        assert_eq!(out.len(), 3);
        for (i, a) in out.iter_mut().enumerate() {
            assert_eq!(a.shape(), &[1]);
            assert_eq!(a.dtype(), Dtype::Int32);
            assert_eq!(a.item::<i32>(), i as i32);
        }

        let x = Array::from_slice(&[0, 1, 2, 3, 4, 5], &[2, 3]);
        let out = split(&x, 2, None).unwrap();
        assert_eq!(out[0], Array::from_slice(&[0, 1, 2], &[1, 3]));
        assert_eq!(out[1], Array::from_slice(&[3, 4, 5], &[1, 3]));

        let out = split(&x, 3, 1).unwrap();
        assert_eq!(out[0], Array::from_slice(&[0, 3], &[2, 1]));
        assert_eq!(out[1], Array::from_slice(&[1, 4], &[2, 1]));
        assert_eq!(out[2], Array::from_slice(&[2, 5], &[2, 1]));

        let x = Array::zeros::<i32>(&[8, 12]).unwrap();
        let out = split(&x, 2, None).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].shape(), &[4, 12]);
        assert_eq!(out[1].shape(), &[4, 12]);

        let out = split(&x, 3, 1).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].shape(), &[8, 4]);
        assert_eq!(out[1].shape(), &[8, 4]);
        assert_eq!(out[2].shape(), &[8, 4]);
    }

    #[test]
    fn test_split() {
        let x = Array::zeros::<i32>(&[8, 12]).unwrap();

        let out = split_sections(&x, &[], None).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].shape(), x.shape());

        let out = split_sections(&x, &[3, 7], None).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].shape(), &[3, 12]);
        assert_eq!(out[1].shape(), &[4, 12]);
        assert_eq!(out[2].shape(), &[1, 12]);

        let out = split_sections(&x, &[20], None).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].shape(), &[8, 12]);
        assert_eq!(out[1].shape(), &[0, 12]);

        let out = split_sections(&x, &[-5], None).unwrap();
        assert_eq!(out[0].shape(), &[3, 12]);
        assert_eq!(out[1].shape(), &[5, 12]);

        let out = split_sections(&x, &[2, 8], Some(1)).unwrap();
        assert_eq!(out[0].shape(), &[8, 2]);
        assert_eq!(out[1].shape(), &[8, 6]);
        assert_eq!(out[2].shape(), &[8, 4]);

        let x = Array::from_iter(0i32..5, &[5]);
        let out = split_sections(&x, &[2, 1, 2], None).unwrap();
        assert_eq!(out[0], Array::from_slice(&[0, 1], &[2]));
        assert_eq!(out[1], Array::from_slice::<i32>(&[], &[0]));
        assert_eq!(out[2], Array::from_slice(&[1], &[1]));
        assert_eq!(out[3], Array::from_slice(&[2, 3, 4], &[3]));
    }

    #[test]
    fn test_pad() {
        let x = Array::zeros::<f32>(&[1, 2, 3]).unwrap();
        assert_eq!(pad(&x, 1, None, None).unwrap().shape(), &[3, 4, 5]);
        assert_eq!(pad(&x, (0, 1), None, None).unwrap().shape(), &[2, 3, 4]);
        assert_eq!(
            pad(&x, &[(1, 1), (1, 2), (3, 1)], None, None)
                .unwrap()
                .shape(),
            &[3, 5, 7]
        );
    }

    #[test]
    fn test_stack() {
        let x = Array::from_slice::<f32>(&[], &[0]);
        let x = vec![x];
        assert_eq!(stack_axis(&x, 0).unwrap().shape(), &[1, 0]);
        assert_eq!(stack_axis(&x, 1).unwrap().shape(), &[0, 1]);

        let x = Array::from_slice(&[1, 2, 3], &[3]);
        let x = vec![x];
        assert_eq!(stack_axis(&x, 0).unwrap().shape(), &[1, 3]);
        assert_eq!(stack_axis(&x, 1).unwrap().shape(), &[3, 1]);

        let y = Array::from_slice(&[4, 5, 6], &[3]);
        let mut z = x;
        z.push(y);
        assert_eq!(stack(&z).unwrap().shape(), &[2, 3]);
        assert_eq!(stack_axis(&z, 1).unwrap().shape(), &[3, 2]);
        assert_eq!(stack_axis(&z, -1).unwrap().shape(), &[3, 2]);
        assert_eq!(stack_axis(&z, -2).unwrap().shape(), &[2, 3]);

        let empty: Vec<Array> = Vec::new();
        assert!(stack_axis(&empty, 0).is_err());

        let x = Array::from_slice(&[1, 2, 3], &[3])
            .as_dtype(Dtype::Float16)
            .unwrap();
        let y = Array::from_slice(&[4, 5, 6], &[3])
            .as_dtype(Dtype::Int32)
            .unwrap();
        assert_eq!(stack_axis(&[x, y], 0).unwrap().dtype(), Dtype::Float16);

        let x = Array::from_slice(&[1, 2, 3], &[3])
            .as_dtype(Dtype::Int32)
            .unwrap();
        let y = Array::from_slice(&[4, 5, 6, 7], &[4])
            .as_dtype(Dtype::Int32)
            .unwrap();
        assert!(stack_axis(&[x, y], 0).is_err());
    }

    #[test]
    fn test_swap_axes() {
        let a = Array::from_int(0);
        assert!(swap_axes(&a, 0, 0).is_err());

        let a = Array::zeros::<i32>(&[2]).unwrap();
        assert!(swap_axes(&a, 0, 1).is_err());
        assert_eq!(swap_axes(&a, 0, 0).unwrap().shape(), &[2]);
        assert_eq!(swap_axes(&a, -1, -1).unwrap().shape(), &[2]);

        let a = Array::zeros::<i32>(&[2, 3, 4]).unwrap();
        assert!(swap_axes(&a, 0, -4).is_err());
        assert!(swap_axes(&a, 0, 3).is_err());
        assert!(swap_axes(&a, 3, 0).is_err());
        assert!(swap_axes(&a, -4, 0).is_err());
        assert_eq!(swap_axes(&a, 0, 2).unwrap().shape(), &[4, 3, 2]);
        assert_eq!(swap_axes(&a, 0, 1).unwrap().shape(), &[3, 2, 4]);
        assert_eq!(swap_axes(&a, 0, -1).unwrap().shape(), &[4, 3, 2]);
        assert_eq!(swap_axes(&a, -2, 2).unwrap().shape(), &[2, 4, 3]);
    }

    #[test]
    fn test_tile() {
        let x = Array::from_slice(&[1, 2, 3], &[3]);
        let y = tile(&x, &[2]).unwrap();
        let expected = Array::from_slice(&[1, 2, 3, 1, 2, 3], &[6]);
        assert_eq!(y, expected);

        let x = Array::from_slice(&[1, 2, 3, 4], &[2, 2]);
        let y = tile(&x, &[2]).unwrap();
        let expected = Array::from_slice(&[1, 2, 1, 2, 3, 4, 3, 4], &[2, 4]);
        assert_eq!(y, expected);

        let x = Array::from_slice(&[1, 2, 3, 4], &[2, 2]);
        let y = tile(&x, &[4, 1]).unwrap();
        let expected =
            Array::from_slice(&[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], &[8, 2]);
        assert_eq!(y, expected);

        let x = Array::from_slice(&[1, 2, 3, 4], &[2, 2]);
        let y = tile(&x, &[2, 2]).unwrap();
        let expected =
            Array::from_slice(&[1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4], &[4, 4]);
        assert_eq!(y, expected);

        let x = Array::from_slice(&[1, 2, 3], &[3]);
        let y = tile(&x, &[2, 2, 2]).unwrap();
        let expected = Array::from_slice(
            &[
                1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
            ],
            &[2, 2, 6],
        );
        assert_eq!(y, expected);
    }

    #[test]
    fn test_transpose() {
        let x = Array::from_int(1);
        let y = transpose(&x).unwrap();
        assert!(y.shape().is_empty());
        assert_eq!(y.item::<i32>(), 1);
        assert!(transpose_axes(&x, &[0][..]).is_err());
        assert!(transpose_axes(&x, &[1][..]).is_err());

        let x = Array::from_slice(&[1], &[1]);
        let y = transpose(&x).unwrap();
        assert_eq!(y.shape(), &[1]);
        assert_eq!(y.item::<i32>(), 1);

        let y = transpose_axes(&x, &[-1][..]).unwrap();
        assert_eq!(y.shape(), &[1]);
        assert_eq!(y.item::<i32>(), 1);

        assert!(transpose_axes(&x, &[1][..]).is_err());
        assert!(transpose_axes(&x, &[0, 0][..]).is_err());

        let x = Array::from_slice::<i32>(&[], &[0]);
        let y = transpose(&x).unwrap();
        assert_eq!(y.shape(), &[0]);
        y.eval().unwrap();
        assert_eq!(y.size(), 0);

        let x = Array::from_slice(&[1, 2, 3, 4, 5, 6], &[2, 3]);
        let mut y = transpose(&x).unwrap();
        assert_eq!(y.shape(), &[3, 2]);
        y = transpose_axes(&x, &[-1, 0][..]).unwrap();
        assert_eq!(y.shape(), &[3, 2]);
        y = transpose_axes(&x, &[-1, -2][..]).unwrap();
        assert_eq!(y.shape(), &[3, 2]);
        y.eval().unwrap();
        assert_eq!(y, Array::from_slice(&[1, 4, 2, 5, 3, 6], &[3, 2]));

        let y = transpose_axes(&x, &[0, 1][..]).unwrap();
        assert_eq!(y.shape(), &[2, 3]);
        assert_eq!(y, x);

        let y = transpose_axes(&x, &[0, -1][..]).unwrap();
        assert_eq!(y.shape(), &[2, 3]);
        assert_eq!(y, x);

        assert!(transpose_axes(&x, &[][..]).is_err());
        assert!(transpose_axes(&x, &[0][..]).is_err());
        assert!(transpose_axes(&x, &[0, 0][..]).is_err());
        assert!(transpose_axes(&x, &[0, 0, 0][..]).is_err());
        assert!(transpose_axes(&x, &[0, 1, 1][..]).is_err());

        let x = Array::from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], &[2, 3, 2]);
        let y = transpose(&x).unwrap();
        assert_eq!(y.shape(), &[2, 3, 2]);
        let expected = Array::from_slice(&[1, 7, 3, 9, 5, 11, 2, 8, 4, 10, 6, 12], &[2, 3, 2]);
        assert_eq!(y, expected);

        let y = transpose_axes(&x, &[0, 1, 2][..]).unwrap();
        assert_eq!(y.shape(), &[2, 3, 2]);
        assert_eq!(y, x);

        let y = transpose_axes(&x, &[1, 0, 2][..]).unwrap();
        assert_eq!(y.shape(), &[3, 2, 2]);
        let expected = Array::from_slice(&[1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12], &[3, 2, 2]);
        assert_eq!(y, expected);

        let y = transpose_axes(&x, &[0, 2, 1][..]).unwrap();
        assert_eq!(y.shape(), &[2, 2, 3]);
        let expected = Array::from_slice(&[1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12], &[2, 2, 3]);
        assert_eq!(y, expected);

        let mut x = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7], &[4, 2]);
        x = reshape(transpose(&x).unwrap(), &[2, 2, 2]).unwrap();
        let expected = Array::from_slice(&[0, 2, 4, 6, 1, 3, 5, 7], &[2, 2, 2]);
        assert_eq!(x, expected);

        let mut x = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7], &[1, 4, 1, 2]);
        // assert!(x.flags().row_contiguous);
        x = transpose_axes(&x, &[2, 1, 0, 3][..]).unwrap();
        x.eval().unwrap();
        // assert!(x.flags().row_contiguous);
    }

    #[test]
    fn test_contiguous() {
        // A freshly created array should be contiguous
        let x = Array::from_slice(&[1i32, 2, 3, 4, 5, 6], &[2, 3]);
        assert!(x.is_contiguous());

        // After transpose, the array may not be contiguous
        let t = transpose(&x).unwrap();
        t.eval().unwrap();
        // Transposed array is typically not contiguous (strides are swapped)
        assert!(!t.is_contiguous());

        // contiguous() should make it contiguous again
        let c = contiguous(&t).unwrap();
        c.eval().unwrap();
        assert!(c.is_contiguous());

        // The values should be the same (but in contiguous memory order)
        assert_eq!(c.shape(), &[3, 2]);
        let data: Vec<i32> = c.try_as_slice().unwrap().to_vec();
        assert_eq!(data, vec![1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn test_is_contiguous_and_contiguous() {
        let x = Array::from_slice(&[1i32, 2, 3, 4, 5, 6], &[2, 3]);

        // Contiguous array: is_contiguous() returns true
        assert!(x.is_contiguous());
        let slice = x.try_as_slice::<i32>();
        assert!(slice.is_ok());
        assert_eq!(slice.unwrap(), &[1, 2, 3, 4, 5, 6]);

        // Transposed array: is_contiguous() returns false
        let t = transpose(&x).unwrap();
        t.eval().unwrap();
        assert!(!t.is_contiguous());

        // After contiguous(), is_contiguous() returns true and data is correct
        let c = contiguous(&t).unwrap();
        c.eval().unwrap();
        assert!(c.is_contiguous());
        let slice = c.try_as_slice::<i32>();
        assert!(slice.is_ok());
        assert_eq!(slice.unwrap(), &[1, 4, 2, 5, 3, 6]);
    }
}
