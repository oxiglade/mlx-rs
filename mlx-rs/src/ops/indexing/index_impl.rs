use std::{
    borrow::Cow,
    ops::{Bound, Deref, RangeBounds},
    rc::Rc,
};

use smallvec::{smallvec, SmallVec};

use crate::{
    array,
    constants::DEFAULT_STACK_VEC_LEN,
    error::Result,
    ops::indexing::expand_ellipsis_operations,
    utils::{resolve_index_unchecked, VectorArray},
    Array, Stream,
};

use super::{
    ArrayIndex, ArrayIndexOp, Ellipsis, Guarded, NewAxis, RangeIndex, StrideBy, TryIndexOp,
};

/* -------------------------------------------------------------------------- */
/*                               Implementation                               */
/* -------------------------------------------------------------------------- */

impl<'a> ArrayIndex<'a> for i32 {
    fn index_op(self) -> ArrayIndexOp<'a> {
        ArrayIndexOp::TakeIndex { index: self }
    }
}

impl<'a> ArrayIndex<'a> for NewAxis {
    fn index_op(self) -> ArrayIndexOp<'a> {
        ArrayIndexOp::ExpandDims
    }
}

impl<'a> ArrayIndex<'a> for Ellipsis {
    fn index_op(self) -> ArrayIndexOp<'a> {
        ArrayIndexOp::Ellipsis
    }
}

impl<'a> ArrayIndex<'a> for Array {
    fn index_op(self) -> ArrayIndexOp<'a> {
        ArrayIndexOp::TakeArray {
            indices: Rc::new(self),
        }
    }
}

impl<'a> ArrayIndex<'a> for &'a Array {
    fn index_op(self) -> ArrayIndexOp<'a> {
        ArrayIndexOp::TakeArrayRef { indices: self }
    }
}

impl<'a> ArrayIndex<'a> for ArrayIndexOp<'a> {
    fn index_op(self) -> ArrayIndexOp<'a> {
        self
    }
}

macro_rules! impl_array_index_for_bounded_range {
    ($t:ty) => {
        impl<'a> ArrayIndex<'a> for $t {
            fn index_op(self) -> ArrayIndexOp<'a> {
                ArrayIndexOp::Slice(RangeIndex::new(
                    self.start_bound().cloned(),
                    self.end_bound().cloned(),
                    Some(1),
                ))
            }
        }
    };
}

impl_array_index_for_bounded_range!(std::ops::Range<i32>);
impl_array_index_for_bounded_range!(std::ops::RangeFrom<i32>);
impl_array_index_for_bounded_range!(std::ops::RangeInclusive<i32>);
impl_array_index_for_bounded_range!(std::ops::RangeTo<i32>);
impl_array_index_for_bounded_range!(std::ops::RangeToInclusive<i32>);

impl<'a> ArrayIndex<'a> for std::ops::RangeFull {
    fn index_op(self) -> ArrayIndexOp<'a> {
        ArrayIndexOp::Slice(RangeIndex::new(Bound::Unbounded, Bound::Unbounded, Some(1)))
    }
}

macro_rules! impl_array_index_for_stride_by_bounded_range {
    ($t:ty) => {
        impl<'a> ArrayIndex<'a> for StrideBy<$t> {
            fn index_op(self) -> ArrayIndexOp<'a> {
                ArrayIndexOp::Slice(RangeIndex::new(
                    self.inner.start_bound().cloned(),
                    self.inner.end_bound().cloned(),
                    Some(self.stride),
                ))
            }
        }
    };
}

impl_array_index_for_stride_by_bounded_range!(std::ops::Range<i32>);
impl_array_index_for_stride_by_bounded_range!(std::ops::RangeFrom<i32>);
impl_array_index_for_stride_by_bounded_range!(std::ops::RangeInclusive<i32>);
impl_array_index_for_stride_by_bounded_range!(std::ops::RangeTo<i32>);
impl_array_index_for_stride_by_bounded_range!(std::ops::RangeToInclusive<i32>);

impl<'a> ArrayIndex<'a> for StrideBy<std::ops::RangeFull> {
    fn index_op(self) -> ArrayIndexOp<'a> {
        ArrayIndexOp::Slice(RangeIndex::new(
            Bound::Unbounded,
            Bound::Unbounded,
            Some(self.stride),
        ))
    }
}

impl<'a, T> TryIndexOp<T> for Array
where
    T: ArrayIndex<'a>,
{
    fn try_index_device(&self, i: T, stream: impl AsRef<Stream>) -> Result<Array> {
        get_item(self, i, stream)
    }
}

impl<'a> TryIndexOp<&'a [ArrayIndexOp<'a>]> for Array {
    fn try_index_device(
        &self,
        i: &'a [ArrayIndexOp<'a>],
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        get_item_nd(self, i, stream)
    }
}

impl<'a, A> TryIndexOp<(A,)> for Array
where
    A: ArrayIndex<'a>,
{
    fn try_index_device(&self, i: (A,), stream: impl AsRef<Stream>) -> Result<Array> {
        let i = [i.0.index_op()];
        get_item_nd(self, &i, stream)
    }
}

impl<'a, 'b, A, B> TryIndexOp<(A, B)> for Array
where
    A: ArrayIndex<'a>,
    B: ArrayIndex<'b>,
{
    fn try_index_device(&self, i: (A, B), stream: impl AsRef<Stream>) -> Result<Array> {
        let i = [i.0.index_op(), i.1.index_op()];
        get_item_nd(self, &i, stream)
    }
}

impl<'a, 'b, 'c, A, B, C> TryIndexOp<(A, B, C)> for Array
where
    A: ArrayIndex<'a>,
    B: ArrayIndex<'b>,
    C: ArrayIndex<'c>,
{
    fn try_index_device(&self, i: (A, B, C), stream: impl AsRef<Stream>) -> Result<Array> {
        let i = [i.0.index_op(), i.1.index_op(), i.2.index_op()];
        get_item_nd(self, &i, stream)
    }
}

impl<'a, 'b, 'c, 'd, A, B, C, D> TryIndexOp<(A, B, C, D)> for Array
where
    A: ArrayIndex<'a>,
    B: ArrayIndex<'b>,
    C: ArrayIndex<'c>,
    D: ArrayIndex<'d>,
{
    fn try_index_device(&self, i: (A, B, C, D), stream: impl AsRef<Stream>) -> Result<Array> {
        let i = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
        ];
        get_item_nd(self, &i, stream)
    }
}

impl<'a, 'b, 'c, 'd, 'e, A, B, C, D, E> TryIndexOp<(A, B, C, D, E)> for Array
where
    A: ArrayIndex<'a>,
    B: ArrayIndex<'b>,
    C: ArrayIndex<'c>,
    D: ArrayIndex<'d>,
    E: ArrayIndex<'e>,
{
    fn try_index_device(&self, i: (A, B, C, D, E), stream: impl AsRef<Stream>) -> Result<Array> {
        let i = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
        ];
        get_item_nd(self, &i, stream)
    }
}

impl<'a, 'b, 'c, 'd, 'e, 'f, A, B, C, D, E, F> TryIndexOp<(A, B, C, D, E, F)> for Array
where
    A: ArrayIndex<'a>,
    B: ArrayIndex<'b>,
    C: ArrayIndex<'c>,
    D: ArrayIndex<'d>,
    E: ArrayIndex<'e>,
    F: ArrayIndex<'f>,
{
    fn try_index_device(&self, i: (A, B, C, D, E, F), stream: impl AsRef<Stream>) -> Result<Array> {
        let i = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
            i.5.index_op(),
        ];
        get_item_nd(self, &i, stream)
    }
}

impl<'a, 'b, 'c, 'd, 'e, 'f, 'g, A, B, C, D, E, F, G> TryIndexOp<(A, B, C, D, E, F, G)> for Array
where
    A: ArrayIndex<'a>,
    B: ArrayIndex<'b>,
    C: ArrayIndex<'c>,
    D: ArrayIndex<'d>,
    E: ArrayIndex<'e>,
    F: ArrayIndex<'f>,
    G: ArrayIndex<'g>,
{
    fn try_index_device(
        &self,
        i: (A, B, C, D, E, F, G),
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        let i = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
            i.5.index_op(),
            i.6.index_op(),
        ];
        get_item_nd(self, &i, stream)
    }
}

impl<'a, 'b, 'c, 'd, 'e, 'f, 'g, 'h, A, B, C, D, E, F, G, H> TryIndexOp<(A, B, C, D, E, F, G, H)>
    for Array
where
    A: ArrayIndex<'a>,
    B: ArrayIndex<'b>,
    C: ArrayIndex<'c>,
    D: ArrayIndex<'d>,
    E: ArrayIndex<'e>,
    F: ArrayIndex<'f>,
    G: ArrayIndex<'g>,
    H: ArrayIndex<'h>,
{
    fn try_index_device(
        &self,
        i: (A, B, C, D, E, F, G, H),
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        let i = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
            i.5.index_op(),
            i.6.index_op(),
            i.7.index_op(),
        ];
        get_item_nd(self, &i, stream)
    }
}

impl<'a, 'b, 'c, 'd, 'e, 'f, 'g, 'h, 'i, A, B, C, D, E, F, G, H, I>
    TryIndexOp<(A, B, C, D, E, F, G, H, I)> for Array
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
{
    fn try_index_device(
        &self,
        i: (A, B, C, D, E, F, G, H, I),
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        let i = [
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
        get_item_nd(self, &i, stream)
    }
}

impl<'a, 'b, 'c, 'd, 'e, 'f, 'g, 'h, 'i, 'j, A, B, C, D, E, F, G, H, I, J>
    TryIndexOp<(A, B, C, D, E, F, G, H, I, J)> for Array
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
{
    fn try_index_device(
        &self,
        i: (A, B, C, D, E, F, G, H, I, J),
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        let i = [
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
        get_item_nd(self, &i, stream)
    }
}

impl<'a, 'b, 'c, 'd, 'e, 'f, 'g, 'h, 'i, 'j, 'k, A, B, C, D, E, F, G, H, I, J, K>
    TryIndexOp<(A, B, C, D, E, F, G, H, I, J, K)> for Array
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
{
    fn try_index_device(
        &self,
        i: (A, B, C, D, E, F, G, H, I, J, K),
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        let i = [
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
        get_item_nd(self, &i, stream)
    }
}

impl<'a, 'b, 'c, 'd, 'e, 'f, 'g, 'h, 'i, 'j, 'k, 'l, A, B, C, D, E, F, G, H, I, J, K, L>
    TryIndexOp<(A, B, C, D, E, F, G, H, I, J, K, L)> for Array
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
{
    fn try_index_device(
        &self,
        i: (A, B, C, D, E, F, G, H, I, J, K, L),
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        let i = [
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
        get_item_nd(self, &i, stream)
    }
}

impl<'a, 'b, 'c, 'd, 'e, 'f, 'g, 'h, 'i, 'j, 'k, 'l, 'm, A, B, C, D, E, F, G, H, I, J, K, L, M>
    TryIndexOp<(A, B, C, D, E, F, G, H, I, J, K, L, M)> for Array
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
{
    fn try_index_device(
        &self,
        i: (A, B, C, D, E, F, G, H, I, J, K, L, M),
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        let i = [
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
        get_item_nd(self, &i, stream)
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
    > TryIndexOp<(A, B, C, D, E, F, G, H, I, J, K, L, M, N)> for Array
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
{
    fn try_index_device(
        &self,
        i: (A, B, C, D, E, F, G, H, I, J, K, L, M, N),
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        let i = [
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
        get_item_nd(self, &i, stream)
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
    > TryIndexOp<(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O)> for Array
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
{
    fn try_index_device(
        &self,
        i: (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O),
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        let i = [
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
        get_item_nd(self, &i, stream)
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
    > TryIndexOp<(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P)> for Array
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
{
    fn try_index_device(
        &self,
        i: (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P),
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        let i = [
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
        get_item_nd(self, &i, stream)
    }
}

// Implement private bindings
impl Array {
    // This is exposed in the c api but not found in the swift or python api
    //
    // Thie is not the same as rust slice. Slice in python is more like `StepBy` iterator in rust
    pub(crate) fn slice_device(
        &self,
        start: &[i32],
        stop: &[i32],
        strides: &[i32],
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_slice(
                res,
                self.as_ptr(),
                start.as_ptr(),
                start.len(),
                stop.as_ptr(),
                stop.len(),
                strides.as_ptr(),
                strides.len(),
                stream.as_ref().as_ptr(),
            )
        })
    }
}

/* -------------------------------------------------------------------------- */
/*                              Helper functions                              */
/* -------------------------------------------------------------------------- */

fn absolute_indices(absolute_start: i32, absolute_end: i32, stride: i32) -> Vec<i32> {
    let mut indices = Vec::new();
    let mut i = absolute_start;
    while (stride > 0 && i < absolute_end) || (stride < 0 && i > absolute_end) {
        indices.push(i);
        i += stride;
    }
    indices
}

enum GatherIndexItem<'a> {
    Owned(Array),
    Borrowed(&'a Array),
    Rc(Rc<Array>),
}

impl AsRef<Array> for GatherIndexItem<'_> {
    fn as_ref(&self) -> &Array {
        match self {
            GatherIndexItem::Owned(array) => array,
            GatherIndexItem::Borrowed(array) => array,
            GatherIndexItem::Rc(array) => array,
        }
    }
}

impl Deref for GatherIndexItem<'_> {
    type Target = Array;

    fn deref(&self) -> &Self::Target {
        match self {
            GatherIndexItem::Owned(array) => array,
            GatherIndexItem::Borrowed(array) => array,
            GatherIndexItem::Rc(array) => array,
        }
    }
}

// Implement additional public APIs
//
// TODO: rewrite this in a more rusty way
#[inline]
fn gather_nd<'a>(
    src: &Array,
    operations: impl Iterator<Item = &'a ArrayIndexOp<'a>>,
    gather_first: bool,
    last_array_or_index: usize,
    stream: impl AsRef<Stream>,
) -> Result<(usize, Array)> {
    use ArrayIndexOp::*;

    let mut max_dims = 0;
    let mut slice_count = 0;
    let mut is_slice: Vec<bool> = Vec::with_capacity(last_array_or_index);
    let mut gather_indices: Vec<GatherIndexItem> = Vec::with_capacity(last_array_or_index);

    let shape = src.shape();

    // prepare the gather indices
    let mut axes = Vec::with_capacity(last_array_or_index);
    let mut operation_len: usize = 0;
    let mut slice_sizes = shape.to_vec();
    for (i, op) in operations.enumerate() {
        axes.push(i as i32);
        operation_len += 1;
        slice_sizes[i] = 1;
        match op {
            TakeIndex { index } => {
                let item = Array::from_int(resolve_index_unchecked(
                    *index,
                    src.dim(i as i32) as usize,
                ) as i32);
                gather_indices.push(GatherIndexItem::Owned(item));
                is_slice.push(false);
            }
            Slice(range) => {
                slice_count += 1;
                is_slice.push(true);

                let size = shape[i];
                let absolute_start = range.absolute_start(size);
                let absolute_end = range.absolute_end(size);
                let indices = absolute_indices(absolute_start, absolute_end, range.stride());

                let item = Array::from_slice(&indices, &[indices.len() as i32]);

                gather_indices.push(GatherIndexItem::Owned(item));
            }
            TakeArray { indices } => {
                is_slice.push(false);
                max_dims = max_dims.max(indices.ndim());
                // Cloning is just incrementing the reference count
                gather_indices.push(GatherIndexItem::Rc(indices.clone()));
            }
            TakeArrayRef { indices } => {
                is_slice.push(false);
                max_dims = max_dims.max(indices.ndim());
                // Cloning is just incrementing the reference count
                gather_indices.push(GatherIndexItem::Borrowed(indices));
            }
            Ellipsis | ExpandDims => {
                unreachable!("Unexpected operation in gather_nd")
            }
        }
    }

    // reshape them so that the int/array indices are first
    if gather_first {
        if slice_count > 0 {
            let mut slice_index = 0;
            for (i, item) in gather_indices.iter_mut().enumerate() {
                if is_slice[i] {
                    let mut new_shape = vec![1; max_dims + slice_count];
                    new_shape[max_dims + slice_index] = item.dim(0);
                    *item = GatherIndexItem::Owned(item.reshape(&new_shape)?);
                    slice_index += 1;
                } else {
                    let mut new_shape = item.shape().to_vec();
                    new_shape.extend((0..slice_count).map(|_| 1));
                    *item = GatherIndexItem::Owned(item.reshape(&new_shape)?);
                }
            }
        }
    } else {
        // reshape them so that the int/array indices are last
        for (i, item) in gather_indices[..slice_count].iter_mut().enumerate() {
            let mut new_shape = vec![1; max_dims + slice_count];
            new_shape[i] = item.dim(0);
            *item = GatherIndexItem::Owned(item.reshape(&new_shape)?);
        }
    }

    // Do the gather
    // let indices = new_mlx_vector_array(gather_indices);
    // SAFETY: indices will be freed at the end of this function. The lifetime of the items in
    // `gather_indices` is managed by the `gather_indices` vector.
    let indices = VectorArray::try_from_iter(gather_indices.iter())?;

    let gathered = Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_gather(
            res,
            src.as_ptr(),
            indices.as_ptr(),
            axes.as_ptr(),
            axes.len(),
            slice_sizes.as_ptr(),
            slice_sizes.len(),
            stream.as_ref().as_ptr(),
        )
    })?;
    let gathered_shape = gathered.shape();

    // Squeeze the dims
    let output_shape: Vec<i32> = gathered_shape[0..(max_dims + slice_count)]
        .iter()
        .chain(gathered_shape[(max_dims + slice_count + operation_len)..].iter())
        .copied()
        .collect();
    let result = gathered.reshape(&output_shape)?;

    Ok((max_dims, result))
}

#[inline]
fn get_item_index(src: &Array, index: i32, axis: i32, stream: impl AsRef<Stream>) -> Result<Array> {
    let index = resolve_index_unchecked(index, src.dim(axis) as usize) as i32;
    src.take_axis_device(array!(index), axis, stream)
}

#[inline]
fn get_item_array(
    src: &Array,
    indices: &Array,
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    src.take_axis_device(indices, axis, stream)
}

#[inline]
fn get_item_slice(src: &Array, range: RangeIndex, stream: impl AsRef<Stream>) -> Result<Array> {
    let ndim = src.ndim();
    let mut starts: SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> = smallvec![0; ndim];
    let mut ends: SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> = SmallVec::from_slice(src.shape());
    let mut strides: SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> = smallvec![1; ndim];

    let size = ends[0];
    starts[0] = range.start(size);
    ends[0] = range.end(size);
    strides[0] = range.stride();

    src.slice_device(&starts, &ends, &strides, stream)
}

// See `mlx_get_item` in python/src/indexing.cpp and `getItem` in
// mlx-swift/Sources/MLX/MLXArray+Indexing.swift
fn get_item<'a>(
    src: &Array,
    index: impl ArrayIndex<'a>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    use ArrayIndexOp::*;

    match index.index_op() {
        Ellipsis => Ok(src.deep_clone()),
        TakeIndex { index } => get_item_index(src, index, 0, stream),
        TakeArray { indices } => get_item_array(src, &indices, 0, stream),
        TakeArrayRef { indices } => get_item_array(src, indices, 0, stream),
        Slice(range) => get_item_slice(src, range, stream),
        ExpandDims => src.expand_dims_device(0, stream),
    }
}

// See `mlx_get_item_nd` in python/src/indexing.cpp and `getItemNd` in
// mlx-swift/Sources/MLX/MLXArray+Indexing.swift
fn get_item_nd(
    src: &Array,
    operations: &[ArrayIndexOp],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    use ArrayIndexOp::*;

    let mut src = Cow::Borrowed(src);

    // The plan is as follows:
    // 1. Replace the ellipsis with a series of slice(None)
    // 2. Loop over the indices and calculate the gather indices
    // 3. Calculate the remaining slices and reshapes

    let operations = expand_ellipsis_operations(src.ndim(), operations);

    // Gather handling

    // compute gatherFirst -- this will be true if there is:
    // - a leading array or index operation followed by
    // - a non index/array (e.g. a slice)
    // - an int/array operation
    //
    // - and there is at least one array operation (hanled below with haveArray)
    let mut gather_first = false;
    let mut have_array_or_index = false; // This is `haveArrayOrIndex` in the swift binding
    let mut have_non_array = false;
    for item in operations.iter() {
        if item.is_array_or_index() {
            if have_array_or_index && have_non_array {
                gather_first = true;
                break;
            }
            have_array_or_index = true;
        } else {
            have_non_array = have_non_array || have_array_or_index;
        }
    }

    let array_count = operations.iter().filter(|op| op.is_array()).count();
    let have_array = array_count > 0;

    let mut remaining_indices: Vec<ArrayIndexOp> = Vec::new();
    if have_array {
        // apply all the operations (except for .newAxis) up to and including the
        // final .array operation (array operations are implemented via gather)
        let last_array_or_index = operations
            .iter()
            .rposition(|op| op.is_array_or_index())
            .unwrap(); // safe because we know there is at least one array operation
        let gather_indices = operations[..=last_array_or_index]
            .iter()
            .filter(|op| !matches!(op, Ellipsis | ExpandDims));
        let (max_dims, gathered) = gather_nd(
            &src,
            gather_indices,
            gather_first,
            last_array_or_index,
            &stream,
        )?;

        src = Cow::Owned(gathered);

        // Reassemble the indices for the slicing or reshaping if there are any
        if gather_first {
            remaining_indices.extend((0..max_dims).map(|_| (..).index_op()));

            // copy any newAxis in the gatherIndices through.  any slices get
            // copied in as full range (already applied)
            for item in &operations[..=last_array_or_index] {
                // Using full match syntax to avoid forgetting to add new cases
                match item {
                    ExpandDims => remaining_indices.push(item.clone()),
                    Slice { .. } => remaining_indices.push((..).index_op()),
                    Ellipsis
                    | TakeIndex { index: _ }
                    | TakeArray { indices: _ }
                    | TakeArrayRef { indices: _ } => {}
                }
            }

            // append the remaining operations
            remaining_indices.extend(operations[(last_array_or_index + 1)..].iter().cloned());
        } else {
            // !gather_first
            for item in operations.iter() {
                // Using full match syntax to avoid forgetting to add new cases
                match item {
                    TakeIndex { .. } | TakeArray { .. } | TakeArrayRef { .. } => break,
                    ExpandDims => remaining_indices.push(item.clone()),
                    Ellipsis | Slice(_) => remaining_indices.push((..).index_op()),
                }
            }

            // handle the trailing gathers
            remaining_indices.extend((0..max_dims).map(|_| (..).index_op()));

            // and the remaining operations
            remaining_indices.extend(operations[(last_array_or_index + 1)..].iter().cloned());
        }
    }

    if have_array && remaining_indices.is_empty() {
        return Ok(src.into_owned());
    }

    if remaining_indices.is_empty() {
        remaining_indices = operations.to_vec();
    }

    // Slice handling
    let ndim = src.ndim();
    let mut starts: SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> = smallvec![0; ndim];
    let mut ends: SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> = SmallVec::from_slice(src.shape());
    let mut strides: SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> = smallvec![1; ndim];
    let mut squeeze_needed = false;
    let mut axis = 0;

    for item in remaining_indices.iter() {
        match item {
            ExpandDims => continue,
            TakeIndex { mut index } => {
                if !have_array {
                    index = resolve_index_unchecked(index, src.dim(axis as i32) as usize) as i32;
                    starts[axis] = index;
                    ends[axis] = index + 1;
                    squeeze_needed = true;
                }
            }
            Slice(range) => {
                let size = src.dim(axis as i32);
                starts[axis] = range.start(size);
                ends[axis] = range.end(size);
                strides[axis] = range.stride();
            }
            Ellipsis | TakeArray { .. } | TakeArrayRef { .. } => {
                unreachable!("Unexpected item in remaining_indices: {:?}", item)
            }
        }
        axis += 1;
    }

    src = Cow::Owned(src.slice_device(&starts, &ends, &strides, stream)?);

    // Unsqueeze handling
    if remaining_indices.len() > ndim || squeeze_needed {
        let mut new_shape = SmallVec::<[i32; DEFAULT_STACK_VEC_LEN]>::new();
        let mut axis_ = 0;
        for item in remaining_indices {
            // using full match syntax to avoid forgetting to add new cases
            match item {
                ExpandDims => new_shape.push(1),
                TakeIndex { .. } => {
                    if squeeze_needed {
                        axis_ += 1;
                    }
                }
                Ellipsis | TakeArray { .. } | TakeArrayRef { .. } | Slice(_) => {
                    new_shape.push(src.dim(axis_));
                    axis_ += 1;
                }
            }
        }
        new_shape.extend(src.shape()[(axis_ as usize)..].iter().cloned());

        src = Cow::Owned(src.reshape(&new_shape)?);
    }

    Ok(src.into_owned())
}

/* -------------------------------------------------------------------------- */
/*                                 Unit tests                                 */
/* -------------------------------------------------------------------------- */

#[cfg(test)]
mod tests {
    use crate::{
        ops::indexing::{Ellipsis, IndexOp, IntoStrideBy, NewAxis},
        test_utils::{assert_array_eq, tolerances},
        Array,
    };

    #[test]
    fn test_array_index_negative_int() {
        let a = Array::from_iter(0i32..8, &[8]);

        let s = a.index(-1);

        assert_eq!(s.ndim(), 0);
        assert_eq!(s.item::<i32>(), 7);

        let s = a.index(-8);

        assert_eq!(s.ndim(), 0);
        assert_eq!(s.item::<i32>(), 0);
    }

    #[test]
    fn test_array_index_new_axis() {
        let a = Array::from_iter(0..60, &[3, 4, 5]);
        let s = a.index(NewAxis);

        assert_eq!(s.ndim(), 4);
        assert_eq!(s.shape(), &[1, 3, 4, 5]);

        let expected = Array::from_iter(0..60, &[1, 3, 4, 5]);
        assert_array_eq(s, expected, tolerances::EXACT.rtol, tolerances::EXACT.atol);
    }

    #[test]
    fn test_array_index_ellipsis() {
        let a = Array::from_iter(0i32..8, &[2, 2, 2]);

        let s1 = a.index((.., .., 0));
        let expected = Array::from_slice(&[0, 2, 4, 6], &[2, 2]);
        assert_array_eq(s1, expected, tolerances::EXACT.rtol, tolerances::EXACT.atol);

        let s2 = a.index((Ellipsis, 0));

        let expected = Array::from_slice(&[0, 2, 4, 6], &[2, 2]);
        assert_array_eq(s2, expected, tolerances::EXACT.rtol, tolerances::EXACT.atol);

        let s3 = a.index(Ellipsis);

        let expected = Array::from_iter(0i32..8, &[2, 2, 2]);
        assert_array_eq(s3, expected, tolerances::EXACT.rtol, tolerances::EXACT.atol);
    }

    #[test]
    fn test_array_index_stride() {
        let arr = Array::from_iter(0..10, &[10]);
        let s = arr.index((2..8).stride_by(2));

        let expected = Array::from_slice(&[2, 4, 6], &[3]);
        assert_array_eq(s, expected, tolerances::EXACT.rtol, tolerances::EXACT.atol);
    }

    // The unit tests below are ported from the swift binding.
    // See `mlx-swift/Tests/MLXTests/MLXArray+IndexingTests.swift`

    #[test]
    fn test_array_subscript_int() {
        let a = Array::from_iter(0i32..512, &[8, 8, 8]);

        let s = a.index(1);

        assert_eq!(s.ndim(), 2);
        assert_eq!(s.shape(), &[8, 8]);

        let expected = Array::from_iter(64..128, &[8, 8]);
        assert_array_eq(s, expected, tolerances::EXACT.rtol, tolerances::EXACT.atol);
    }

    #[test]
    fn test_array_subscript_int_array() {
        // squeeze output dimensions as needed
        let a = Array::from_iter(0i32..512, &[8, 8, 8]);

        let s1 = a.index((1, 2));

        assert_eq!(s1.ndim(), 1);
        assert_eq!(s1.shape(), &[8]);

        let expected = Array::from_iter(80..88, &[8]);
        assert_array_eq(s1, expected, tolerances::EXACT.rtol, tolerances::EXACT.atol);

        let s2 = a.index((1, 2, 3));

        assert_eq!(s2.ndim(), 0);
        assert!(s2.shape().is_empty());
        assert_eq!(s2.item::<i32>(), 64 + 2 * 8 + 3);
    }

    #[test]
    fn test_array_subscript_int_array_2() {
        // last dimension should not be squeezed
        let a = Array::from_iter(0i32..512, &[8, 8, 8, 1]);

        let s = a.index(1);

        assert_eq!(s.ndim(), 3);
        assert_eq!(s.shape(), &[8, 8, 1]);

        let s1 = a.index((1, 2));

        assert_eq!(s1.ndim(), 2);
        assert_eq!(s1.shape(), &[8, 1]);

        let s2 = a.index((1, 2, 3));

        assert_eq!(s2.ndim(), 1);
        assert_eq!(s2.shape(), &[1]);
    }

    #[test]
    fn test_array_subscript_from_end() {
        let a = Array::from_iter(0i32..12, &[3, 4]);

        let s = a.index((-1, -2));

        assert_eq!(s.ndim(), 0);
        assert_eq!(s.item::<i32>(), 10);
    }

    #[test]
    fn test_array_subscript_range() {
        let a = Array::from_iter(0i32..512, &[8, 8, 8]);

        let s1 = a.index(1..3);

        assert_eq!(s1.ndim(), 3);
        assert_eq!(s1.shape(), &[2, 8, 8]);
        let expected = Array::from_iter(64..192, &[2, 8, 8]);
        assert_array_eq(s1, expected, tolerances::EXACT.rtol, tolerances::EXACT.atol);

        // even though the first dimension is 1 we do not squeeze it
        let s2 = a.index(1..=1);

        assert_eq!(s2.ndim(), 3);
        assert_eq!(s2.shape(), &[1, 8, 8]);
        let expected = Array::from_iter(64..128, &[1, 8, 8]);
        assert_array_eq(s2, expected, tolerances::EXACT.rtol, tolerances::EXACT.atol);

        // multiple ranges, resolving RangeExpressions vs the dimensions
        let s3 = a.index((1..2, ..3, 3..));

        assert_eq!(s3.ndim(), 3);
        assert_eq!(s3.shape(), &[1, 3, 5]);
        let expected = Array::from_slice(
            &[67, 68, 69, 70, 71, 75, 76, 77, 78, 79, 83, 84, 85, 86, 87],
            &[1, 3, 5],
        );
        assert_array_eq(s3, expected, tolerances::EXACT.rtol, tolerances::EXACT.atol);

        let s4 = a.index((-2..-1, ..-3, -3..));

        assert_eq!(s4.ndim(), 3);
        assert_eq!(s4.shape(), &[1, 5, 3]);
        let expected = Array::from_slice(
            &[
                389, 390, 391, 397, 398, 399, 405, 406, 407, 413, 414, 415, 421, 422, 423,
            ],
            &[1, 5, 3],
        );
        assert_array_eq(s4, expected, tolerances::EXACT.rtol, tolerances::EXACT.atol);
    }

    #[test]
    fn test_array_subscript_advanced() {
        // advanced subscript examples taken from
        // https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing

        let a = Array::from_iter(0..35, &[5, 7]).as_type::<i32>().unwrap();

        let i1 = Array::from_slice(&[0, 2, 4], &[3]);
        let i2 = Array::from_slice(&[0, 1, 2], &[3]);

        let s1 = a.index((i1, i2));

        assert_eq!(s1.ndim(), 1);
        assert_eq!(s1.shape(), &[3]);

        let expected = Array::from_slice(&[0i32, 15, 30], &[3]);
        assert_array_eq(s1, expected, tolerances::EXACT.rtol, tolerances::EXACT.atol);
    }

    #[test]
    fn test_array_subscript_advanced_with_ref() {
        let a = Array::from_iter(0..35, &[5, 7]).as_type::<i32>().unwrap();

        let i1 = Array::from_slice(&[0, 2, 4], &[3]);
        let i2 = Array::from_slice(&[0, 1, 2], &[3]);

        let s1 = a.index((i1, &i2));

        assert_eq!(s1.ndim(), 1);
        assert_eq!(s1.shape(), &[3]);

        let expected = Array::from_slice(&[0i32, 15, 30], &[3]);
        assert_array_eq(s1, expected, tolerances::EXACT.rtol, tolerances::EXACT.atol);
    }

    #[test]
    fn test_array_subscript_advanced_2() {
        let a = Array::from_iter(0..12, &[6, 2]).as_type::<i32>().unwrap();

        let i1 = Array::from_slice(&[0, 2, 4], &[3]);
        let s2 = a.index(i1);

        let expected = Array::from_slice(&[0i32, 1, 4, 5, 8, 9], &[3, 2]);
        assert_array_eq(s2, expected, tolerances::EXACT.rtol, tolerances::EXACT.atol);
    }

    #[test]
    fn test_collection() {
        let a = Array::from_iter(0i32..20, &[2, 2, 5]);

        // enumerate "rows"
        for i in 0..2 {
            let row = a.index(i);
            let expected = Array::from_iter((i * 10)..(i * 10 + 10), &[2, 5]);
            assert_array_all_close!(row, expected);
        }
    }

    #[test]
    fn test_array_subscript_advanced_2d() {
        let a = Array::from_iter(0..12, &[4, 3]).as_type::<i32>().unwrap();

        let rows = Array::from_slice(&[0, 0, 3, 3], &[2, 2]);
        let cols = Array::from_slice(&[0, 2, 0, 2], &[2, 2]);

        let s = a.index((rows, cols));

        let expected = Array::from_slice(&[0, 2, 9, 11], &[2, 2]);
        assert_array_eq(s, expected, tolerances::EXACT.rtol, tolerances::EXACT.atol);
    }

    #[test]
    fn test_array_subscript_advanced_2d_2() {
        let a = Array::from_iter(0..12, &[4, 3]).as_type::<i32>().unwrap();

        let rows = Array::from_slice(&[0, 3], &[2, 1]);
        let cols = Array::from_slice(&[0, 2], &[2]);

        let s = a.index((rows, cols));

        let expected = Array::from_slice(&[0, 2, 9, 11], &[2, 2]);
        assert_array_eq(s, expected, tolerances::EXACT.rtol, tolerances::EXACT.atol);
    }

    fn check(result: impl AsRef<Array>, shape: &[i32], expected_sum: i32) {
        let result = result.as_ref();
        assert_eq!(result.shape(), shape);

        let sum = result.sum(None).unwrap();

        assert_eq!(sum.item::<i32>(), expected_sum);
    }

    #[test]
    fn test_full_index_read_single() {
        let a = Array::from_iter(0..60, &[3, 4, 5]);

        // a[...]
        check(a.index(Ellipsis), &[3, 4, 5], 1770);

        // a[None]
        check(a.index(NewAxis), &[1, 3, 4, 5], 1770);

        // a[0]
        check(a.index(0), &[4, 5], 190);

        // a[1:3]
        check(a.index(1..3), &[2, 4, 5], 1580);

        // i = mx.array([2, 1])
        let i = Array::from_slice(&[2, 1], &[2]);

        // a[i]
        check(a.index(i), &[2, 4, 5], 1580);
    }

    #[test]
    fn test_full_index_read_no_array() {
        let a = Array::from_iter(0..360, &[2, 3, 4, 5, 3]);

        // a[..., 0]
        check(a.index((Ellipsis, 0)), &[2, 3, 4, 5], 21420);

        // a[0, ...]
        check(a.index((0, Ellipsis)), &[3, 4, 5, 3], 16110);

        // a[0, ..., 0]
        check(a.index((0, Ellipsis, 0)), &[3, 4, 5], 5310);

        // a[..., ::2, :]
        let result = a.index((Ellipsis, (..).stride_by(2), ..));
        check(result, &[2, 3, 4, 3, 3], 38772);

        // a[..., None, ::2, -1]
        let result = a.index((Ellipsis, NewAxis, (..).stride_by(2), -1));
        check(result, &[2, 3, 4, 1, 3], 12996);

        // a[:, 2:, 0]
        check(a.index((.., 2.., 0)), &[2, 1, 5, 3], 6510);

        // a[::-1, :2, 2:, ..., None, ::2]
        let result = a.index((
            (..).stride_by(-1),
            ..2,
            2..,
            Ellipsis,
            NewAxis,
            (..).stride_by(2),
        ));
        check(result, &[2, 2, 2, 5, 1, 2], 13160);
    }

    #[test]
    fn test_full_index_read_array() {
        // these have an `Array` as a source of indices and go through the gather path

        // a = mx.arange(540).reshape(3, 3, 4, 5, 3)
        let a = Array::from_iter(0..540, &[3, 3, 4, 5, 3]);

        // i = mx.array([2, 1])
        let i = Array::from_slice(&[2, 1], &[2]);

        // a[0, i]
        check(a.index((0, &i)), &[2, 4, 5, 3], 14340);

        // a[..., i, 0]
        check(a.index((Ellipsis, &i, 0)), &[3, 3, 4, 2], 19224);

        // a[i, 0, ...]
        check(a.index((&i, 0, Ellipsis)), &[2, 4, 5, 3], 35940);

        // gatherFirst path
        // a[i, ..., i]
        check(a.index((&i, Ellipsis, &i)), &[2, 3, 4, 5], 43200);

        // a[i, ..., ::2, :]
        let result = a.index((&i, Ellipsis, (..).stride_by(2), ..));
        check(result, &[2, 3, 4, 3, 3], 77652);

        // gatherFirst path
        // a[..., i, None, ::2, -1]
        let result = a.index((Ellipsis, &i, NewAxis, (..).stride_by(2), -1));
        check(result, &[2, 3, 3, 1, 3], 14607);

        // a[:, 2:, i]
        check(a.index((.., 2.., &i)), &[3, 1, 2, 5, 3], 29655);

        // a[::-1, :2, i, 2:, ..., None, ::2]
        let result = a.index((
            (..).stride_by(-1),
            ..2,
            i,
            2..,
            Ellipsis,
            NewAxis,
            (..).stride_by(2),
        ));
        check(result, &[3, 2, 2, 3, 1, 2], 17460);
    }
}
