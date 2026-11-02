use crate::{AttentionKind, InferenceError};
use mlx_rs::Array;

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
        create: impl FnOnce() -> Result<Array, InferenceError>,
    ) -> Result<Option<&Array>, InferenceError> {
        if length == 1
            && *attention == AttentionKind::Full
            && positions.iter().all(|&position| position <= offset)
        {
            return Ok(None);
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
        Ok(Some(&self.entries[index].array))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    #[test]
    fn full_decode_omits_masks_and_other_steps_share_by_positions_and_policy(
    ) -> Result<(), InferenceError> {
        let mut masks = AttentionMasks::default();
        assert!(masks
            .get_or_create(8, 1, &[0, 6, 7, 8], &AttentionKind::Full, || panic!(
                "full decode needs no mask"
            ))?
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
                    .get_or_create(8, length, positions, attention, || {
                        calls += 1;
                        Ok(Array::from(true))
                    })?
                    .unwrap();
            }
        }
        assert_eq!(calls, 3);
        assert!(masks
            .get_or_create(8, 1, &[8, 9], &AttentionKind::Full, || Ok(Array::from(
                false
            )))?
            .is_some());
        Ok(())
    }
}
