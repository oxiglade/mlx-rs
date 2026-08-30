use std::borrow::Cow;

use smallvec::{smallvec, SmallVec};

use crate::{
    constants::DEFAULT_STACK_VEC_LEN,
    error::Result,
    ops::{
        broadcast_arrays, broadcast_to,
        indexing::{count_non_new_axis_operations, expand_ellipsis_operations},
        reshape,
    },
    utils::{resolve_index_signed_unchecked, VectorArray},
    with_stream, Array, Stream,
};

use super::{ArrayIndex, ArrayIndexOp, Guarded, RangeIndex, TryIndexMutOp};

impl Array {
    pub(crate) fn slice_update_device(
        &self,
        update: &Array,
        starts: &[i32],
        ends: &[i32],
        strides: &[i32],
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_slice_update(
                res,
                self.as_ptr(),
                update.as_ptr(),
                starts.as_ptr(),
                starts.len(),
                ends.as_ptr(),
                ends.len(),
                strides.as_ptr(),
                strides.len(),
                stream.as_ref().as_ptr(),
            )
        })
    }
}

// See `updateSlice` in the swift binding or `mlx_slice_update` in the python binding
fn update_slice(
    src: &Array,
    operations: &[ArrayIndexOp],
    update: &Array,
    stream: impl AsRef<Stream>,
) -> Result<Option<Array>> {
    let ndim = src.ndim();
    if ndim == 0 || operations.is_empty() {
        return Ok(None);
    }

    // Remove leading singletons dimensions from the update
    let mut update = remove_leading_singleton_dimensions(update, &stream)?;

    // Build slice update params
    let mut starts: SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> = smallvec![0; ndim];
    let mut ends: SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> = SmallVec::from_slice(src.shape());
    let mut strides: SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> = smallvec![1; ndim];

    // If it's just a simple slice, just do a slice update and return
    if operations.len() == 1 {
        if let ArrayIndexOp::Slice(range_index) = &operations[0] {
            let size = src.dim(0);
            starts[0] = range_index.start(size);
            ends[0] = range_index.end(size);
            strides[0] = range_index.stride();

            return Ok(Some(src.slice_update_device(
                &update, &starts, &ends, &strides, &stream,
            )?));
        }
    }

    // Can't route to slice update if any arrays are present
    if operations.iter().any(|op| op.is_array()) {
        return Ok(None);
    }

    // Expand ellipses into a series of ':' (range full) slices
    let operations = expand_ellipsis_operations(ndim, operations);

    // If no non-None indices return the broadcasted update
    let non_new_axis_operation_count = count_non_new_axis_operations(&operations);
    if non_new_axis_operation_count == 0 {
        return Ok(Some(with_stream(stream.as_ref(), || {
            broadcast_to(&update, src.shape())
        })?));
    }

    // Process entries
    // let mut update_expand_dims: SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> = SmallVec::new();
    let mut update_reshape: SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> = smallvec![0; ndim];
    let mut axis = src.ndim() - 1;
    let mut update_axis = update.ndim() as i32 - 1;

    while axis >= non_new_axis_operation_count {
        if update_axis >= 0 {
            update_reshape[axis] = update.dim(update_axis);
            update_axis -= 1;
        } else {
            update_reshape[axis] = 1;
        }
        axis -= 1;
    }

    for item in operations.iter().rev() {
        use ArrayIndexOp::*;

        match item {
            TakeIndex { index } => {
                let size = src.dim(axis as i32);
                let index = if index.is_negative() {
                    size + index
                } else {
                    *index
                };
                // SAFETY: axis is always non-negative
                starts[axis] = index;
                ends[axis] = index.saturating_add(1);

                update_reshape[axis] = 1;
                axis = axis.saturating_sub(1);
            }
            Slice(slice) => {
                let size = src.dim(axis as i32);
                // SAFETY: axis is always non-negative
                starts[axis] = slice.start(size);
                ends[axis] = slice.end(size);
                strides[axis] = slice.stride();

                if update_axis >= 0 {
                    update_reshape[axis] = update.dim(update_axis);
                    update_axis = update_axis.saturating_sub(1);
                } else {
                    update_reshape[axis] = 1;
                }
                axis = axis.saturating_sub(1);
            }
            ExpandDims => {}
            Ellipsis | TakeArray { indices: _ } | TakeArrayRef { indices: _ } => {
                panic!("unexpected item in operations")
            }
        }
    }

    if update.shape() != &update_reshape[..] {
        update = Cow::Owned(with_stream(stream.as_ref(), || {
            reshape(update, &update_reshape)
        })?);
    }

    Ok(Some(src.slice_update_device(
        &update, &starts, &ends, &strides, &stream,
    )?))
}

// See `leadingSingletonDimensionsRemoved` in the swift binding
fn remove_leading_singleton_dimensions(
    a: &Array,
    stream: impl AsRef<Stream>,
) -> Result<Cow<'_, Array>> {
    let shape = a.shape();
    let mut new_shape: Vec<_> = shape.iter().skip_while(|&&dim| dim == 1).cloned().collect();
    if shape != new_shape {
        if new_shape.is_empty() {
            new_shape = vec![1];
        }
        Ok(Cow::Owned(with_stream(stream.as_ref(), || {
            a.reshape(&new_shape)
        })?))
    } else {
        Ok(Cow::Borrowed(a))
    }
}

struct ScatterArgs<'a> {
    indices: SmallVec<[Cow<'a, Array>; DEFAULT_STACK_VEC_LEN]>,
    update: Array,
    axes: SmallVec<[i32; DEFAULT_STACK_VEC_LEN]>,
}

/// See `scatterArguments` in the swift binding
fn scatter_args<'a>(
    src: &'a Array,
    operations: &'a [ArrayIndexOp],
    update: &Array,
    stream: impl AsRef<Stream>,
) -> Result<ScatterArgs<'a>> {
    use ArrayIndexOp::*;

    if operations.len() == 1 {
        return match &operations[0] {
            TakeIndex { index } => scatter_args_index(src, *index, update, stream),
            TakeArray { indices } => {
                scatter_args_array(src, Cow::Borrowed(indices), update, stream)
            }
            TakeArrayRef { indices } => {
                scatter_args_array(src, Cow::Borrowed(indices), update, stream)
            }
            Slice(range_index) => scatter_args_slice(src, range_index, update, stream),
            ExpandDims => Ok(ScatterArgs {
                indices: smallvec![],
                update: with_stream(stream.as_ref(), || broadcast_to(update, src.shape()))?,
                axes: smallvec![],
            }),
            Ellipsis => panic!("Unable to update array with ellipsis argument"),
        };
    }

    scatter_args_nd(src, operations, update, stream)
}

fn scatter_args_index<'a>(
    src: &'a Array,
    index: i32,
    update: &Array,
    stream: impl AsRef<Stream>,
) -> Result<ScatterArgs<'a>> {
    // mlx_scatter_args_index

    // Remove any leading singleton dimensions from the update
    // and then broadcast update to shape of src[0, ...]
    let update = remove_leading_singleton_dimensions(update, &stream)?;

    let mut shape: SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> = SmallVec::from_slice(src.shape());
    shape[0] = 1;

    Ok(ScatterArgs {
        indices: smallvec![Cow::Owned(Array::from_int(resolve_index_signed_unchecked(
            index,
            src.dim(0)
        )))],
        update: with_stream(stream.as_ref(), || broadcast_to(&update, &shape))?,
        axes: smallvec![0],
    })
}

fn scatter_args_array<'a>(
    src: &'a Array,
    a: Cow<'a, Array>,
    update: &Array,
    stream: impl AsRef<Stream>,
) -> Result<ScatterArgs<'a>> {
    // mlx_scatter_args_array

    // trim leading singleton dimensions
    let update = remove_leading_singleton_dimensions(update, &stream)?;

    // The update shape must broadcast with indices.shape + [1] + src.shape[1:]
    let mut update_shape: SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> = a
        .shape()
        .iter()
        .chain(src.shape().iter().skip(1))
        .cloned()
        .collect();
    let update = with_stream(stream.as_ref(), || broadcast_to(&update, &update_shape))?;

    update_shape.insert(a.ndim(), 1);
    let update = with_stream(stream.as_ref(), || update.reshape(&update_shape))?;

    Ok(ScatterArgs {
        indices: smallvec![a],
        update,
        axes: smallvec![0],
    })
}

fn scatter_args_slice<'a>(
    src: &'a Array,
    range_index: &'a RangeIndex,
    update: &Array,
    stream: impl AsRef<Stream>,
) -> Result<ScatterArgs<'a>> {
    // mlx_scatter_args_slice

    // if none slice is requested braodcast the update to the src size and return it
    if range_index.is_full() {
        let update = remove_leading_singleton_dimensions(update, &stream)?;

        return Ok(ScatterArgs {
            indices: smallvec![],
            update: with_stream(stream.as_ref(), || broadcast_to(&update, src.shape()))?,
            axes: smallvec![],
        });
    }

    let size = src.dim(0);
    let start = range_index.start(size);
    let end = range_index.end(size);
    let stride = range_index.stride();

    // If simple stride
    if stride == 1 {
        let update = remove_leading_singleton_dimensions(update, &stream)?;

        // Broadcast update to slice size
        let update_broadcast_shape: SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> = (1..end - start)
            .chain(src.shape().iter().skip(1).cloned())
            .collect();
        let update = with_stream(stream.as_ref(), || {
            broadcast_to(&update, &update_broadcast_shape)
        })?;

        let indices = Array::from_slice(&[start], &[1]);
        Ok(ScatterArgs {
            indices: smallvec![Cow::Owned(indices)],
            update,
            axes: smallvec![0],
        })
    } else {
        // stride != 1, convert the slice to an array
        let a_vals = strided_range_to_vec(start, end, stride);
        let a = Array::from_slice(&a_vals, &[a_vals.len() as i32]);

        scatter_args_array(src, Cow::Owned(a), update, stream)
    }
}

fn scatter_args_nd<'a>(
    src: &'a Array,
    operations: &[ArrayIndexOp],
    update: &Array,
    stream: impl AsRef<Stream>,
) -> Result<ScatterArgs<'a>> {
    use ArrayIndexOp::*;

    // mlx_scatter_args_nd

    let shape = src.shape();

    let operations = expand_ellipsis_operations(src.ndim(), operations);
    let update = remove_leading_singleton_dimensions(update, &stream)?;

    // If no non-newAxis indices return the broadcasted update
    let non_new_axis_operation_count = count_non_new_axis_operations(&operations);
    if non_new_axis_operation_count == 0 {
        return Ok(ScatterArgs {
            indices: smallvec![],
            update: with_stream(stream.as_ref(), || broadcast_to(&update, shape))?,
            axes: smallvec![],
        });
    }

    // Analyse the types of the indices
    let mut max_dims = 0;
    let mut arrays_first = false;
    let mut count_new_axis: i32 = 0;
    let mut count_slices: i32 = 0;
    let mut count_arrays: i32 = 0;
    let mut count_strided_slices: i32 = 0;
    let mut count_simple_slices_post: i32 = 0;

    let mut have_array = false;
    let mut have_non_array = false;

    macro_rules! analyze_indices_take_array {
        ($indices:ident) => {
            have_array = true;
            if have_array && have_non_array {
                arrays_first = true;
            }
            max_dims = $indices.ndim().max(max_dims);
            count_arrays = count_arrays.saturating_add(1);
            count_simple_slices_post = 0;
        };
    }

    for item in operations.iter() {
        match item {
            TakeIndex { index: _ } => {
                // ignore
            }
            Slice(range_index) => {
                have_non_array = have_array;
                count_slices = count_slices.saturating_add(1);
                if range_index.stride() != 1 {
                    count_strided_slices = count_strided_slices.saturating_add(1);
                    count_simple_slices_post = 0;
                } else {
                    count_simple_slices_post = count_simple_slices_post.saturating_add(1);
                }
            }
            TakeArray { indices } => {
                analyze_indices_take_array!(indices);
            }
            TakeArrayRef { indices } => {
                analyze_indices_take_array!(indices);
            }
            ExpandDims => {
                have_non_array = true;
                count_new_axis = count_new_axis.saturating_add(1);
            }
            Ellipsis => panic!("Unexpected item ellipsis in scatter_args_nd"),
        }
    }

    // We have index dims for the arrays, strided slices (implemented as arrays), none
    let mut index_dims = (max_dims + count_new_axis as usize + count_slices as usize)
        .saturating_sub(count_simple_slices_post as usize);

    // If we have simple non-strided slices, we also attach an index for that
    if index_dims == 0 {
        index_dims = 1;
    }

    // Go over each index type and translate to the needed scatter args
    let mut array_indices: SmallVec<[Array; DEFAULT_STACK_VEC_LEN]> =
        SmallVec::with_capacity(operations.len());
    let mut slice_number: i32 = 0;
    let mut array_number: i32 = 0;
    let mut axis: i32 = 0;

    // We collect the shapes of the slices and updates during this process
    let mut update_shape = vec![1; non_new_axis_operation_count];
    let mut slice_shapes: SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> = SmallVec::new();

    macro_rules! update_shapes_take_array {
        ($indices:ident) => {
            // Place the arrays in the correct dimension
            let start = if arrays_first {
                max_dims - $indices.ndim()
            } else {
                // SAFETY: slice_number is never decremented and should be non-negative
                slice_number as usize + max_dims - $indices.ndim()
            };
            let mut new_shape = vec![1; index_dims];

            for j in 0..$indices.ndim() {
                new_shape[start + j] = $indices.dim(j as i32);
            }

            array_indices.push(with_stream(stream.as_ref(), || {
                $indices.reshape(&new_shape)
            })?);
            array_number = array_number.saturating_add(1);

            if !arrays_first && array_number == count_arrays {
                slice_number = slice_number.saturating_add_unsigned(max_dims as u32);
            }

            // Add the shape to the update
            update_shape[axis as usize] = 1;
            axis = axis.saturating_add(1);
        };
    }

    for item in operations.iter() {
        match item {
            TakeIndex { index } => {
                let resolved_index = resolve_index_signed_unchecked(*index, src.dim(axis));
                array_indices.push(Array::from_int(resolved_index));
                // SAFETY: axis is always non-negative
                update_shape[axis as usize] = 1;
                axis = axis.saturating_add(1);
            }
            Slice(range_index) => {
                let size = src.dim(axis);
                let start = range_index.absolute_start(size);
                let end = range_index.absolute_end(size);
                let stride = range_index.stride();

                let mut index_shape = vec![1; index_dims];

                // If it's a simple slice, we only need to add the start index
                if array_number >= count_arrays && count_strided_slices <= 0 && stride == 1 {
                    let index = with_stream(stream.as_ref(), || {
                        Array::from_int(start).reshape(&index_shape)
                    })?;
                    let slice_shape_entry = end - start;
                    slice_shapes.push(slice_shape_entry);
                    array_indices.push(index);

                    // Add the shape to the update
                    update_shape[axis as usize] = slice_shape_entry;
                } else {
                    // Otherwise we expand the slice into indices using arange
                    let index_vals = strided_range_to_vec(start, end, stride);
                    let index = Array::from_slice(&index_vals, &[index_vals.len() as i32]);
                    let location = if arrays_first {
                        slice_number.saturating_add(max_dims as i32)
                    } else {
                        slice_number
                    };
                    index_shape[location as usize] = index.size() as i32;
                    array_indices.push(with_stream(stream.as_ref(), || {
                        index.reshape(&index_shape)
                    })?);

                    slice_number = slice_number.saturating_add(1);
                    count_strided_slices = count_strided_slices.saturating_sub(1);

                    // Add the shape to the update
                    update_shape[axis as usize] = 1;
                }

                axis = axis.saturating_add(1);
            }
            TakeArray { indices } => {
                update_shapes_take_array!(indices);
            }
            TakeArrayRef { indices } => {
                update_shapes_take_array!(indices);
            }
            ExpandDims => slice_number = slice_number.saturating_add(1),
            Ellipsis => panic!("Unexpected item ellipsis in scatter_args_nd"),
        }
    }

    // Broadcast the update to the indices and slices
    let array_indices = with_stream(stream.as_ref(), || broadcast_arrays(&array_indices))?;
    let update_shape_broadcast: Vec<_> = array_indices[0]
        .shape()
        .iter()
        .chain(slice_shapes.iter())
        .chain(src.shape().iter().skip(non_new_axis_operation_count))
        .cloned()
        .collect();
    let update = with_stream(stream.as_ref(), || {
        broadcast_to(&update, &update_shape_broadcast)
    })?;

    // Reshape the update with the size-1 dims for the int and array indices
    let update_reshape: Vec<_> = array_indices[0]
        .shape()
        .iter()
        .chain(update_shape.iter())
        .chain(src.shape().iter().skip(non_new_axis_operation_count))
        .cloned()
        .collect();

    let update = with_stream(stream.as_ref(), || update.reshape(&update_reshape))?;

    let array_indices_len = array_indices.len();

    let indices = array_indices.into_iter().map(Cow::Owned).collect();
    Ok(ScatterArgs {
        indices,
        update,
        axes: (0..array_indices_len as i32).collect(),
    })
}

fn strided_range_to_vec(start: i32, exclusive_end: i32, stride: i32) -> Vec<i32> {
    let estimated_capacity = (exclusive_end - start).abs() / stride.abs();
    let mut vec = Vec::with_capacity(estimated_capacity as usize);
    let mut current = start;

    assert_ne!(stride, 0, "Stride cannot be zero");

    if stride.is_negative() {
        while current > exclusive_end {
            vec.push(current);
            current += stride;
        }
    } else {
        while current < exclusive_end {
            vec.push(current);
            current += stride;
        }
    }

    vec
}

unsafe fn scatter_device(
    a: &Array,
    indices: &[impl AsRef<Array>],
    updates: &Array,
    axes: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let indices_vector = VectorArray::try_from_iter(indices.iter())?;

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_scatter(
            res,
            a.as_ptr(),
            indices_vector.as_ptr(),
            updates.as_ptr(),
            axes.as_ptr(),
            axes.len(),
            stream.as_ref().as_ptr(),
        )
    })
}

impl Array {
    fn try_index_mut_device_inner(
        &mut self,
        operations: &[ArrayIndexOp],
        update: &Array,
        stream: impl AsRef<Stream>,
    ) -> Result<()> {
        if let Some(result) = update_slice(self, operations, update, &stream)? {
            *self = result;
            return Ok(());
        }

        let ScatterArgs {
            indices,
            update,
            axes,
        } = scatter_args(self, operations, update, &stream)?;
        if !indices.is_empty() {
            let result = unsafe { scatter_device(self, &indices, &update, &axes, stream)? };
            drop(indices);
            *self = result;
        } else {
            drop(indices);
            *self = update;
        }
        Ok(())
    }
}

impl<'a, Val> TryIndexMutOp<&'a [ArrayIndexOp<'a>], Val> for Array
where
    Val: AsRef<Array>,
{
    fn try_index_mut_device(
        &mut self,
        i: &'a [ArrayIndexOp<'a>],
        val: Val,
        stream: impl AsRef<Stream>,
    ) -> Result<()> {
        let update = val.as_ref();
        self.try_index_mut_device_inner(i, update, stream)
    }
}

impl<A, Val> TryIndexMutOp<A, Val> for Array
where
    for<'a> A: ArrayIndex<'a>,
    Val: AsRef<Array>,
{
    fn try_index_mut_device(&mut self, i: A, val: Val, stream: impl AsRef<Stream>) -> Result<()> {
        let operations = [i.index_op()];
        let update = val.as_ref();
        self.try_index_mut_device_inner(&operations, update, stream)
    }
}

impl<'a, A, Val> TryIndexMutOp<(A,), Val> for Array
where
    A: ArrayIndex<'a>,
    Val: AsRef<Array>,
{
    fn try_index_mut_device(
        &mut self,
        (i,): (A,),
        val: Val,
        stream: impl AsRef<Stream>,
    ) -> Result<()> {
        let operations = [i.index_op()];
        let update = val.as_ref();
        self.try_index_mut_device_inner(&operations, update, stream)
    }
}

impl<'a, 'b, A, B, Val> TryIndexMutOp<(A, B), Val> for Array
where
    A: ArrayIndex<'a>,
    B: ArrayIndex<'b>,
    Val: AsRef<Array>,
{
    fn try_index_mut_device(
        &mut self,
        i: (A, B),
        val: Val,
        stream: impl AsRef<Stream>,
    ) -> Result<()> {
        let operations = [i.0.index_op(), i.1.index_op()];
        let update = val.as_ref();
        self.try_index_mut_device_inner(&operations, update, stream)
    }
}

impl<'a, 'b, 'c, A, B, C, Val> TryIndexMutOp<(A, B, C), Val> for Array
where
    A: ArrayIndex<'a>,
    B: ArrayIndex<'b>,
    C: ArrayIndex<'c>,
    Val: AsRef<Array>,
{
    fn try_index_mut_device(
        &mut self,
        i: (A, B, C),
        val: Val,
        stream: impl AsRef<Stream>,
    ) -> Result<()> {
        let operations = [i.0.index_op(), i.1.index_op(), i.2.index_op()];
        let update = val.as_ref();
        self.try_index_mut_device_inner(&operations, update, stream)
    }
}

impl<'a, 'b, 'c, 'd, A, B, C, D, Val> TryIndexMutOp<(A, B, C, D), Val> for Array
where
    A: ArrayIndex<'a>,
    B: ArrayIndex<'b>,
    C: ArrayIndex<'c>,
    D: ArrayIndex<'d>,
    Val: AsRef<Array>,
{
    fn try_index_mut_device(
        &mut self,
        i: (A, B, C, D),
        val: Val,
        stream: impl AsRef<Stream>,
    ) -> Result<()> {
        let operations = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
        ];
        let update = val.as_ref();
        self.try_index_mut_device_inner(&operations, update, stream)
    }
}

impl<'a, 'b, 'c, 'd, 'e, A, B, C, D, E, Val> TryIndexMutOp<(A, B, C, D, E), Val> for Array
where
    A: ArrayIndex<'a>,
    B: ArrayIndex<'b>,
    C: ArrayIndex<'c>,
    D: ArrayIndex<'d>,
    E: ArrayIndex<'e>,
    Val: AsRef<Array>,
{
    fn try_index_mut_device(
        &mut self,
        i: (A, B, C, D, E),
        val: Val,
        stream: impl AsRef<Stream>,
    ) -> Result<()> {
        let operations = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
        ];
        let update = val.as_ref();
        self.try_index_mut_device_inner(&operations, update, stream)
    }
}

impl<'a, 'b, 'c, 'd, 'e, 'f, A, B, C, D, E, F, Val> TryIndexMutOp<(A, B, C, D, E, F), Val> for Array
where
    A: ArrayIndex<'a>,
    B: ArrayIndex<'b>,
    C: ArrayIndex<'c>,
    D: ArrayIndex<'d>,
    E: ArrayIndex<'e>,
    F: ArrayIndex<'f>,
    Val: AsRef<Array>,
{
    fn try_index_mut_device(
        &mut self,
        i: (A, B, C, D, E, F),
        val: Val,
        stream: impl AsRef<Stream>,
    ) -> Result<()> {
        let operations = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
            i.5.index_op(),
        ];
        let update = val.as_ref();
        self.try_index_mut_device_inner(&operations, update, stream)
    }
}

impl<'a, 'b, 'c, 'd, 'e, 'f, 'g, A, B, C, D, E, F, G, Val> TryIndexMutOp<(A, B, C, D, E, F, G), Val>
    for Array
where
    A: ArrayIndex<'a>,
    B: ArrayIndex<'b>,
    C: ArrayIndex<'c>,
    D: ArrayIndex<'d>,
    E: ArrayIndex<'e>,
    F: ArrayIndex<'f>,
    G: ArrayIndex<'g>,
    Val: AsRef<Array>,
{
    fn try_index_mut_device(
        &mut self,
        i: (A, B, C, D, E, F, G),
        val: Val,
        stream: impl AsRef<Stream>,
    ) -> Result<()> {
        let operations = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
            i.5.index_op(),
            i.6.index_op(),
        ];
        let update = val.as_ref();
        self.try_index_mut_device_inner(&operations, update, stream)
    }
}

impl<'a, 'b, 'c, 'd, 'e, 'f, 'g, 'h, A, B, C, D, E, F, G, H, Val>
    TryIndexMutOp<(A, B, C, D, E, F, G, H), Val> for Array
where
    A: ArrayIndex<'a>,
    B: ArrayIndex<'b>,
    C: ArrayIndex<'c>,
    D: ArrayIndex<'d>,
    E: ArrayIndex<'e>,
    F: ArrayIndex<'f>,
    G: ArrayIndex<'g>,
    H: ArrayIndex<'h>,
    Val: AsRef<Array>,
{
    fn try_index_mut_device(
        &mut self,
        i: (A, B, C, D, E, F, G, H),
        val: Val,
        stream: impl AsRef<Stream>,
    ) -> Result<()> {
        let operations = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
            i.5.index_op(),
            i.6.index_op(),
            i.7.index_op(),
        ];
        let update = val.as_ref();
        self.try_index_mut_device_inner(&operations, update, stream)
    }
}

impl<'a, 'b, 'c, 'd, 'e, 'f, 'g, 'h, 'i, A, B, C, D, E, F, G, H, I, Val>
    TryIndexMutOp<(A, B, C, D, E, F, G, H, I), Val> for Array
where
    A: ArrayIndex<'a>,
    B: ArrayIndex<'b>,
    C: ArrayIndex<'c>,
    D: ArrayIndex<'d>,
    E: ArrayIndex<'e>,
    F: ArrayIndex<'f>,
    G: ArrayIndex<'g>,
    H: ArrayIndex<'h>,
    I: ArrayIndex<'i>,
    Val: AsRef<Array>,
{
    fn try_index_mut_device(
        &mut self,
        i: (A, B, C, D, E, F, G, H, I),
        val: Val,
        stream: impl AsRef<Stream>,
    ) -> Result<()> {
        let operations = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
            i.5.index_op(),
            i.6.index_op(),
            i.7.index_op(),
            i.8.index_op(),
        ];
        let update = val.as_ref();
        self.try_index_mut_device_inner(&operations, update, stream)
    }
}

impl<'a, 'b, 'c, 'd, 'e, 'f, 'g, 'h, 'i, 'j, A, B, C, D, E, F, G, H, I, J, Val>
    TryIndexMutOp<(A, B, C, D, E, F, G, H, I, J), Val> for Array
where
    A: ArrayIndex<'a>,
    B: ArrayIndex<'b>,
    C: ArrayIndex<'c>,
    D: ArrayIndex<'d>,
    E: ArrayIndex<'e>,
    F: ArrayIndex<'f>,
    G: ArrayIndex<'g>,
    H: ArrayIndex<'h>,
    I: ArrayIndex<'i>,
    J: ArrayIndex<'j>,
    Val: AsRef<Array>,
{
    fn try_index_mut_device(
        &mut self,
        i: (A, B, C, D, E, F, G, H, I, J),
        val: Val,
        stream: impl AsRef<Stream>,
    ) -> Result<()> {
        let operations = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
            i.5.index_op(),
            i.6.index_op(),
            i.7.index_op(),
            i.8.index_op(),
            i.9.index_op(),
        ];
        let update = val.as_ref();
        self.try_index_mut_device_inner(&operations, update, stream)
    }
}

impl<'a, 'b, 'c, 'd, 'e, 'f, 'g, 'h, 'i, 'j, 'k, A, B, C, D, E, F, G, H, I, J, K, Val>
    TryIndexMutOp<(A, B, C, D, E, F, G, H, I, J, K), Val> for Array
where
    A: ArrayIndex<'a>,
    B: ArrayIndex<'b>,
    C: ArrayIndex<'c>,
    D: ArrayIndex<'d>,
    E: ArrayIndex<'e>,
    F: ArrayIndex<'f>,
    G: ArrayIndex<'g>,
    H: ArrayIndex<'h>,
    I: ArrayIndex<'i>,
    J: ArrayIndex<'j>,
    K: ArrayIndex<'k>,
    Val: AsRef<Array>,
{
    fn try_index_mut_device(
        &mut self,
        i: (A, B, C, D, E, F, G, H, I, J, K),
        val: Val,
        stream: impl AsRef<Stream>,
    ) -> Result<()> {
        let operations = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
            i.5.index_op(),
            i.6.index_op(),
            i.7.index_op(),
            i.8.index_op(),
            i.9.index_op(),
            i.10.index_op(),
        ];
        let update = val.as_ref();
        self.try_index_mut_device_inner(&operations, update, stream)
    }
}

impl<'a, 'b, 'c, 'd, 'e, 'f, 'g, 'h, 'i, 'j, 'k, 'l, A, B, C, D, E, F, G, H, I, J, K, L, Val>
    TryIndexMutOp<(A, B, C, D, E, F, G, H, I, J, K, L), Val> for Array
where
    A: ArrayIndex<'a>,
    B: ArrayIndex<'b>,
    C: ArrayIndex<'c>,
    D: ArrayIndex<'d>,
    E: ArrayIndex<'e>,
    F: ArrayIndex<'f>,
    G: ArrayIndex<'g>,
    H: ArrayIndex<'h>,
    I: ArrayIndex<'i>,
    J: ArrayIndex<'j>,
    K: ArrayIndex<'k>,
    L: ArrayIndex<'l>,
    Val: AsRef<Array>,
{
    fn try_index_mut_device(
        &mut self,
        i: (A, B, C, D, E, F, G, H, I, J, K, L),
        val: Val,
        stream: impl AsRef<Stream>,
    ) -> Result<()> {
        let operations = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
            i.5.index_op(),
            i.6.index_op(),
            i.7.index_op(),
            i.8.index_op(),
            i.9.index_op(),
            i.10.index_op(),
            i.11.index_op(),
        ];
        let update = val.as_ref();
        self.try_index_mut_device_inner(&operations, update, stream)
    }
}

impl<
        'a,
        'b,
        'c,
        'd,
        'e,
        'f,
        'g,
        'h,
        'i,
        'j,
        'k,
        'l,
        'm,
        A,
        B,
        C,
        D,
        E,
        F,
        G,
        H,
        I,
        J,
        K,
        L,
        M,
        Val,
    > TryIndexMutOp<(A, B, C, D, E, F, G, H, I, J, K, L, M), Val> for Array
where
    A: ArrayIndex<'a>,
    B: ArrayIndex<'b>,
    C: ArrayIndex<'c>,
    D: ArrayIndex<'d>,
    E: ArrayIndex<'e>,
    F: ArrayIndex<'f>,
    G: ArrayIndex<'g>,
    H: ArrayIndex<'h>,
    I: ArrayIndex<'i>,
    J: ArrayIndex<'j>,
    K: ArrayIndex<'k>,
    L: ArrayIndex<'l>,
    M: ArrayIndex<'m>,
    Val: AsRef<Array>,
{
    fn try_index_mut_device(
        &mut self,
        i: (A, B, C, D, E, F, G, H, I, J, K, L, M),
        val: Val,
        stream: impl AsRef<Stream>,
    ) -> Result<()> {
        let operations = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
            i.5.index_op(),
            i.6.index_op(),
            i.7.index_op(),
            i.8.index_op(),
            i.9.index_op(),
            i.10.index_op(),
            i.11.index_op(),
            i.12.index_op(),
        ];
        let update = val.as_ref();
        self.try_index_mut_device_inner(&operations, update, stream)
    }
}

impl<
        'a,
        'b,
        'c,
        'd,
        'e,
        'f,
        'g,
        'h,
        'i,
        'j,
        'k,
        'l,
        'm,
        'n,
        A,
        B,
        C,
        D,
        E,
        F,
        G,
        H,
        I,
        J,
        K,
        L,
        M,
        N,
        Val,
    > TryIndexMutOp<(A, B, C, D, E, F, G, H, I, J, K, L, M, N), Val> for Array
where
    A: ArrayIndex<'a>,
    B: ArrayIndex<'b>,
    C: ArrayIndex<'c>,
    D: ArrayIndex<'d>,
    E: ArrayIndex<'e>,
    F: ArrayIndex<'f>,
    G: ArrayIndex<'g>,
    H: ArrayIndex<'h>,
    I: ArrayIndex<'i>,
    J: ArrayIndex<'j>,
    K: ArrayIndex<'k>,
    L: ArrayIndex<'l>,
    M: ArrayIndex<'m>,
    N: ArrayIndex<'n>,
    Val: AsRef<Array>,
{
    fn try_index_mut_device(
        &mut self,
        i: (A, B, C, D, E, F, G, H, I, J, K, L, M, N),
        val: Val,
        stream: impl AsRef<Stream>,
    ) -> Result<()> {
        let operations = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
            i.5.index_op(),
            i.6.index_op(),
            i.7.index_op(),
            i.8.index_op(),
            i.9.index_op(),
            i.10.index_op(),
            i.11.index_op(),
            i.12.index_op(),
            i.13.index_op(),
        ];
        let update = val.as_ref();
        self.try_index_mut_device_inner(&operations, update, stream)
    }
}

impl<
        'a,
        'b,
        'c,
        'd,
        'e,
        'f,
        'g,
        'h,
        'i,
        'j,
        'k,
        'l,
        'm,
        'n,
        'o,
        A,
        B,
        C,
        D,
        E,
        F,
        G,
        H,
        I,
        J,
        K,
        L,
        M,
        N,
        O,
        Val,
    > TryIndexMutOp<(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O), Val> for Array
where
    A: ArrayIndex<'a>,
    B: ArrayIndex<'b>,
    C: ArrayIndex<'c>,
    D: ArrayIndex<'d>,
    E: ArrayIndex<'e>,
    F: ArrayIndex<'f>,
    G: ArrayIndex<'g>,
    H: ArrayIndex<'h>,
    I: ArrayIndex<'i>,
    J: ArrayIndex<'j>,
    K: ArrayIndex<'k>,
    L: ArrayIndex<'l>,
    M: ArrayIndex<'m>,
    N: ArrayIndex<'n>,
    O: ArrayIndex<'o>,
    Val: AsRef<Array>,
{
    fn try_index_mut_device(
        &mut self,
        i: (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O),
        val: Val,
        stream: impl AsRef<Stream>,
    ) -> Result<()> {
        let operations = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
            i.5.index_op(),
            i.6.index_op(),
            i.7.index_op(),
            i.8.index_op(),
            i.9.index_op(),
            i.10.index_op(),
            i.11.index_op(),
            i.12.index_op(),
            i.13.index_op(),
            i.14.index_op(),
        ];
        let update = val.as_ref();
        self.try_index_mut_device_inner(&operations, update, stream)
    }
}

impl<
        'a,
        'b,
        'c,
        'd,
        'e,
        'f,
        'g,
        'h,
        'i,
        'j,
        'k,
        'l,
        'm,
        'n,
        'o,
        'p,
        A,
        B,
        C,
        D,
        E,
        F,
        G,
        H,
        I,
        J,
        K,
        L,
        M,
        N,
        O,
        P,
        Val,
    > TryIndexMutOp<(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P), Val> for Array
where
    A: ArrayIndex<'a>,
    B: ArrayIndex<'b>,
    C: ArrayIndex<'c>,
    D: ArrayIndex<'d>,
    E: ArrayIndex<'e>,
    F: ArrayIndex<'f>,
    G: ArrayIndex<'g>,
    H: ArrayIndex<'h>,
    I: ArrayIndex<'i>,
    J: ArrayIndex<'j>,
    K: ArrayIndex<'k>,
    L: ArrayIndex<'l>,
    M: ArrayIndex<'m>,
    N: ArrayIndex<'n>,
    O: ArrayIndex<'o>,
    P: ArrayIndex<'p>,
    Val: AsRef<Array>,
{
    fn try_index_mut_device(
        &mut self,
        i: (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P),
        val: Val,
        stream: impl AsRef<Stream>,
    ) -> Result<()> {
        let operations = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
            i.5.index_op(),
            i.6.index_op(),
            i.7.index_op(),
            i.8.index_op(),
            i.9.index_op(),
            i.10.index_op(),
            i.11.index_op(),
            i.12.index_op(),
            i.13.index_op(),
            i.14.index_op(),
            i.15.index_op(),
        ];
        let update = val.as_ref();
        self.try_index_mut_device_inner(&operations, update, stream)
    }
}

/// The unit tests below are adapted from the Swift binding tests
#[cfg(test)]
mod tests {
    use crate::{
        ops::{indexing::*, ones, zeros},
        Array,
    };

    #[test]
    fn test_array_mutate_single_index() {
        let mut a = Array::from_iter(0i32..12, &[3, 4]);
        let new_value = Array::from_int(77);
        a.index_mut(1, new_value);

        let expected = Array::from_slice(&[0, 1, 2, 3, 77, 77, 77, 77, 8, 9, 10, 11], &[3, 4]);
        assert_array_all_close!(a, expected);
    }

    #[test]
    fn test_array_mutate_broadcast_multi_index() {
        let mut a = Array::from_iter(0i32..20, &[2, 2, 5]);

        // broadcast to a row
        a.index_mut((1, 0), Array::from_int(77));

        // assign to a row
        a.index_mut((0, 0), Array::from_slice(&[55i32, 66, 77, 88, 99], &[5]));

        // single element
        a.index_mut((0, 1, 3), Array::from_int(123));

        let expected = Array::from_slice(
            &[
                55, 66, 77, 88, 99, 5, 6, 7, 123, 9, 77, 77, 77, 77, 77, 15, 16, 17, 18, 19,
            ],
            &[2, 2, 5],
        );
        assert_array_all_close!(a, expected);
    }

    #[test]
    fn test_array_mutate_broadcast_slice() {
        let mut a = Array::from_iter(0i32..20, &[2, 2, 5]);

        // writing using slices -- this ends up covering two elements
        a.index_mut((0..1, 1..2, 2..4), Array::from_int(88));

        let expected = Array::from_slice(
            &[
                0, 1, 2, 3, 4, 5, 6, 88, 88, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            ],
            &[2, 2, 5],
        );
        assert_array_all_close!(a, expected);
    }

    #[test]
    fn test_array_mutate_advanced() {
        let mut a = Array::from_iter(0i32..35, &[5, 7]);

        let i1 = Array::from_slice(&[0, 2, 4], &[3]);
        let i2 = Array::from_slice(&[0, 1, 2], &[3]);

        a.index_mut((i1, i2), Array::from_slice(&[100, 200, 300], &[3]));

        assert_eq!(a.index((0, 0)).item::<i32>(), 100i32);
        assert_eq!(a.index((2, 1)).item::<i32>(), 200i32);
        assert_eq!(a.index((4, 2)).item::<i32>(), 300i32);
    }

    #[test]
    fn test_full_index_write_single() {
        fn check<I>(index: I, expected_sum: i32)
        where
            for<'a> I: ArrayIndex<'a>,
        {
            let mut a = Array::from_iter(0..60, &[3, 4, 5]);

            a.index_mut(index, Array::from_int(1));
            let sum = a.sum(None).unwrap().item::<i32>();
            assert_eq!(sum, expected_sum);
        }

        // a[...]
        // not valid

        // a[None]
        check(NewAxis, 60);

        // a[0]
        check(0, 1600);

        // a[1:3]
        check(1..3, 230);

        // i = mx.array([2, 1])
        let i = Array::from_slice(&[2, 1], &[2]);

        // a[i]
        check(i, 230);
    }

    #[test]
    fn test_full_index_write_no_array() {
        macro_rules! check {
            (($( $i:expr ),*), $sum:expr ) => {
                {
                    let mut a = Array::from_iter(0..360, &[2, 3, 4, 5, 3]);

                    a.index_mut(($($i),*), Array::from_int(1));
                    let sum = a.sum(None).unwrap().item::<i32>();
                    assert_eq!(sum, $sum);
                }
            };
        }

        // a[..., 0] = 1
        check!((Ellipsis, 0), 43320);

        // a[0, ...] = 1
        check!((0, Ellipsis), 48690);

        // a[0, ..., 0] = 1
        check!((0, Ellipsis, 0), 59370);

        // a[..., ::2, :] = 1
        check!((Ellipsis, (..).stride_by(2), ..), 26064);

        // a[..., None, ::2, -1]
        check!((Ellipsis, NewAxis, (..).stride_by(2), -1), 51696);

        // a[:, 2:, 0] = 1
        check!((.., 2.., 0), 58140);

        // a[::-1, :2, 2:, ..., None, ::2] = 1
        check!(
            (
                (..).stride_by(-1),
                ..2,
                2..,
                Ellipsis,
                NewAxis,
                (..).stride_by(2)
            ),
            51540
        );
    }

    #[test]
    fn test_full_index_write_array() {
        // these have an Array as a source of indices and go through the gather path

        macro_rules! check {
            (($( $i:expr ),*), $sum:expr ) => {
                {
                    let mut a = Array::from_iter(0..540, &[3, 3, 4, 5, 3]);

                    a.index_mut(($($i),*), Array::from_int(1));
                    let sum = a.sum(None).unwrap().item::<i32>();
                    assert_eq!(sum, $sum);
                }
            };
        }

        // i = mx.array([2, 1])
        let i = Array::from_slice(&[2, 1], &[2]);

        // a[0, i] = 1
        check!((0, &i), 131310);

        // a[..., i, 0] = 1
        check!((Ellipsis, &i, 0), 126378);

        // a[i, 0, ...] = 1
        check!((&i, 0, Ellipsis), 109710);

        // a[i, ..., i] = 1
        check!((&i, Ellipsis, &i), 102450);

        // a[i, ..., ::2, :] = 1
        check!((&i, Ellipsis, (..).stride_by(2), ..), 68094);

        // a[..., i, None, ::2, -1] = 1
        check!((Ellipsis, &i, NewAxis, (..).stride_by(2), -1), 130977);

        // a[:, 2:, i] = 1
        check!((.., 2.., &i), 115965);

        // a[::-1, :2, i, 2:, ..., None, ::2] = 1
        check!(
            (
                (..).stride_by(-1),
                ..2,
                i,
                2..,
                Ellipsis,
                NewAxis,
                (..).stride_by(2)
            ),
            128142
        );
    }

    #[test]
    fn test_slice_update_with_broadcast() {
        let mut xs = zeros::<f32>(&[4, 3, 2]).unwrap();
        let x = ones::<f32>(&[4, 2]).unwrap();

        let result = xs.try_index_mut((.., 0, ..), x);
        assert!(
            result.is_ok(),
            "Failed to update slice with broadcast: {result:?}"
        );
    }
}
