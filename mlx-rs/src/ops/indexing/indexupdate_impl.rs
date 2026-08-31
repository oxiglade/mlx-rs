use crate::Array;

use super::{
    indexmut_impl::try_index_update_operations, ArrayIndex, ArrayIndexOp, IndexUpdateError,
    TryIndexUpdateOp, UpdateMode,
};

impl<'a, Value> TryIndexUpdateOp<&'a [ArrayIndexOp<'a>], Value> for Array
where
    Value: AsRef<Array>,
{
    fn try_index_update(
        &self,
        index: &'a [ArrayIndexOp<'a>],
        update: Value,
        mode: UpdateMode,
    ) -> Result<Array, IndexUpdateError> {
        try_index_update_operations(self, index, update.as_ref(), mode)
    }
}

impl<'a, Index, Value> TryIndexUpdateOp<Index, Value> for Array
where
    Index: ArrayIndex<'a>,
    Value: AsRef<Array>,
{
    fn try_index_update(
        &self,
        index: Index,
        update: Value,
        mode: UpdateMode,
    ) -> Result<Array, IndexUpdateError> {
        try_index_update_operations(self, &[index.index_op()], update.as_ref(), mode)
    }
}

macro_rules! impl_tuple_update {
    ($(($lifetime:lifetime, $index:ident, $field:tt)),+ $(,)?) => {
        impl<$($lifetime,)+ $($index,)+ Value> TryIndexUpdateOp<($($index,)+), Value> for Array
        where
            $($index: ArrayIndex<$lifetime>,)+
            Value: AsRef<Array>,
        {
            fn try_index_update(
                &self,
                index: ($($index,)+),
                update: Value,
                mode: UpdateMode,
            ) -> Result<Array, IndexUpdateError> {
                let operations = [$(index.$field.index_op(),)+];
                try_index_update_operations(self, &operations, update.as_ref(), mode)
            }
        }
    };
}

impl_tuple_update!(('a, A, 0));
impl_tuple_update!(('a, A, 0), ('b, B, 1));
impl_tuple_update!(('a, A, 0), ('b, B, 1), ('c, C, 2));
impl_tuple_update!(('a, A, 0), ('b, B, 1), ('c, C, 2), ('d, D, 3));
impl_tuple_update!(('a, A, 0), ('b, B, 1), ('c, C, 2), ('d, D, 3), ('e, E, 4));
impl_tuple_update!(('a, A, 0), ('b, B, 1), ('c, C, 2), ('d, D, 3), ('e, E, 4), ('f, F, 5));
impl_tuple_update!(('a, A, 0), ('b, B, 1), ('c, C, 2), ('d, D, 3), ('e, E, 4), ('f, F, 5), ('g, G, 6));
impl_tuple_update!(('a, A, 0), ('b, B, 1), ('c, C, 2), ('d, D, 3), ('e, E, 4), ('f, F, 5), ('g, G, 6), ('h, H, 7));
impl_tuple_update!(('a, A, 0), ('b, B, 1), ('c, C, 2), ('d, D, 3), ('e, E, 4), ('f, F, 5), ('g, G, 6), ('h, H, 7), ('i, I, 8));
impl_tuple_update!(('a, A, 0), ('b, B, 1), ('c, C, 2), ('d, D, 3), ('e, E, 4), ('f, F, 5), ('g, G, 6), ('h, H, 7), ('i, I, 8), ('j, J, 9));
impl_tuple_update!(('a, A, 0), ('b, B, 1), ('c, C, 2), ('d, D, 3), ('e, E, 4), ('f, F, 5), ('g, G, 6), ('h, H, 7), ('i, I, 8), ('j, J, 9), ('k, K, 10));
impl_tuple_update!(('a, A, 0), ('b, B, 1), ('c, C, 2), ('d, D, 3), ('e, E, 4), ('f, F, 5), ('g, G, 6), ('h, H, 7), ('i, I, 8), ('j, J, 9), ('k, K, 10), ('l, L, 11));
impl_tuple_update!(('a, A, 0), ('b, B, 1), ('c, C, 2), ('d, D, 3), ('e, E, 4), ('f, F, 5), ('g, G, 6), ('h, H, 7), ('i, I, 8), ('j, J, 9), ('k, K, 10), ('l, L, 11), ('m, M, 12));
impl_tuple_update!(('a, A, 0), ('b, B, 1), ('c, C, 2), ('d, D, 3), ('e, E, 4), ('f, F, 5), ('g, G, 6), ('h, H, 7), ('i, I, 8), ('j, J, 9), ('k, K, 10), ('l, L, 11), ('m, M, 12), ('n, N, 13));
impl_tuple_update!(('a, A, 0), ('b, B, 1), ('c, C, 2), ('d, D, 3), ('e, E, 4), ('f, F, 5), ('g, G, 6), ('h, H, 7), ('i, I, 8), ('j, J, 9), ('k, K, 10), ('l, L, 11), ('m, M, 12), ('n, N, 13), ('o, O, 14));
impl_tuple_update!(('a, A, 0), ('b, B, 1), ('c, C, 2), ('d, D, 3), ('e, E, 4), ('f, F, 5), ('g, G, 6), ('h, H, 7), ('i, I, 8), ('j, J, 9), ('k, K, 10), ('l, L, 11), ('m, M, 12), ('n, N, 13), ('o, O, 14), ('p, P, 15));
