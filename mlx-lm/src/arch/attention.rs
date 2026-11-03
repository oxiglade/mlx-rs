use crate::{AttentionKind, CacheKind, InferenceError};
use mlx_rs::{fast::ScaledDotProductAttentionMask, Array};

#[derive(Default)]
pub(super) struct AttentionMasks {
    entries: Vec<Mask>,
}
struct Mask {
    length: usize,
    offset: usize,
    positions: Vec<usize>,
    attention: AttentionKind,
    array: Array,
}
impl AttentionMasks {
    pub(super) fn get_or_create(
        &mut self,
        offset: usize,
        length: usize,
        positions: &[usize],
        attention: &AttentionKind,
        kind: CacheKind,
        create: impl FnOnce() -> Result<Array, InferenceError>,
    ) -> Result<Option<ScaledDotProductAttentionMask<'_>>, InferenceError> {
        if length == 1
            && *attention == AttentionKind::Full
            && positions.iter().all(|&position| position <= offset)
        {
            return Ok(None);
        }
        if *attention == AttentionKind::Full
            && kind == CacheKind::Full
            && length > 1
            && positions.len() >= length
            && positions.last().copied() == offset.checked_add(length - 1)
            && positions
                .windows(2)
                .all(|pair| pair[0].checked_add(1) == Some(pair[1]))
        {
            return Ok(Some(ScaledDotProductAttentionMask::Causal));
        }
        let index = self.entries.iter().position(|mask| {
            mask.length == length
                && mask.offset == offset
                && mask.positions == positions
                && mask.attention == *attention
        });
        let index = match index {
            Some(index) => index,
            None => {
                self.entries.push(Mask {
                    length,
                    offset,
                    positions: positions.to_vec(),
                    attention: attention.clone(),
                    array: create()?,
                });
                self.entries.len() - 1
            }
        };
        Ok(Some(ScaledDotProductAttentionMask::Array(
            &self.entries[index].array,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    #[test]
    fn causal_requires_full_contiguous_query_suffix() -> Result<(), InferenceError> {
        let sliding = AttentionKind::Sliding {
            window: NonZeroUsize::new(3).unwrap(),
        };
        for (offset, length, positions, attention, kind, causal) in [
            (
                0,
                3,
                vec![0, 1, 2],
                &AttentionKind::Full,
                CacheKind::Full,
                true,
            ),
            (
                2,
                2,
                vec![0, 1, 2, 3],
                &AttentionKind::Full,
                CacheKind::Full,
                true,
            ),
            (
                8,
                2,
                vec![6, 7, 8, 9],
                &AttentionKind::Full,
                CacheKind::Full,
                true,
            ),
            (
                8,
                2,
                vec![0, 7, 8, 9],
                &AttentionKind::Full,
                CacheKind::Full,
                false,
            ),
            (2, 2, vec![0, 1, 2, 3], &sliding, CacheKind::Full, false),
            (
                2,
                2,
                vec![0, 1, 2, 3],
                &AttentionKind::Full,
                CacheKind::Rotating,
                false,
            ),
            (
                8,
                2,
                vec![0, 7, 8, 9],
                &AttentionKind::Full,
                CacheKind::Rotating,
                false,
            ),
            (
                8,
                2,
                vec![6, 7, 8],
                &AttentionKind::Full,
                CacheKind::Full,
                false,
            ),
            (
                8,
                2,
                vec![7, 8, 9, 10],
                &AttentionKind::Full,
                CacheKind::Full,
                false,
            ),
            (8, 2, vec![9], &AttentionKind::Full, CacheKind::Full, false),
            (
                8,
                2,
                vec![7, 9, 8, 9],
                &AttentionKind::Full,
                CacheKind::Full,
                false,
            ),
            (8, 2, vec![], &AttentionKind::Full, CacheKind::Full, false),
        ] {
            let mut masks = AttentionMasks::default();
            let mut created = false;
            let mask = masks.get_or_create(offset, length, &positions, attention, kind, || {
                created = true;
                Ok(Array::from(true))
            })?;
            assert_eq!(
                matches!(mask, Some(ScaledDotProductAttentionMask::Causal)),
                causal
            );
            assert_eq!(created, !causal);
        }
        Ok(())
    }

    #[test]
    fn full_decode_omits_masks_and_other_steps_share_by_positions_and_policy(
    ) -> Result<(), InferenceError> {
        let mut masks = AttentionMasks::default();
        assert!(masks
            .get_or_create(
                8,
                1,
                &[0, 6, 7, 8],
                &AttentionKind::Full,
                CacheKind::Full,
                || panic!("full decode needs no mask")
            )?
            .is_none());
        let mut calls = 0;
        let sliding = AttentionKind::Sliding {
            window: NonZeroUsize::new(3).unwrap(),
        };
        for (length, attention, positions) in [
            (2, &AttentionKind::Full, &[0, 7, 8, 9][..]),
            (1, &sliding, &[0, 7, 8][..]),
            (1, &sliding, &[6, 7, 8][..]),
        ] {
            for _ in 0..3 {
                masks
                    .get_or_create(8, length, positions, attention, CacheKind::Full, || {
                        calls += 1;
                        Ok(Array::from(true))
                    })?
                    .unwrap();
            }
        }
        assert_eq!(calls, 3);
        assert!(masks
            .get_or_create(8, 1, &[8, 9], &AttentionKind::Full, CacheKind::Full, || Ok(
                Array::from(false)
            ))?
            .is_some());
        Ok(())
    }
}
