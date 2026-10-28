use smallvec::SmallVec;

use crate::{
    constants::DEFAULT_STACK_VEC_LEN,
    error::{Exception, Result},
    utils::resolve_index_unchecked,
    Array, Axes,
};

type LengthsAndAxes = (
    SmallVec<[i32; DEFAULT_STACK_VEC_LEN]>,
    SmallVec<[i32; DEFAULT_STACK_VEC_LEN]>,
);

pub(super) fn require_real_axis(axes: &[i32]) -> Result<()> {
    if axes.is_empty() {
        return Err(Exception::custom("real FFT requires at least one axis"));
    }
    Ok(())
}

pub(super) fn resolve_inverse_real_length(length: i32) -> Result<i32> {
    if length < 1 {
        return Err(Exception::custom(
            "inverse real FFT input length must be positive",
        ));
    }
    length
        .checked_sub(1)
        .and_then(|length| length.checked_mul(2))
        .ok_or_else(|| Exception::custom("inverse real FFT output length exceeds i32::MAX"))
}

#[inline]
pub(super) fn resolve_size_and_axis_unchecked(
    a: &Array,
    n: Option<i32>,
    axis: Option<i32>,
) -> (i32, i32) {
    let axis = axis.unwrap_or(-1);
    let n = n.unwrap_or_else(|| {
        let axis_index = resolve_index_unchecked(axis, a.ndim());
        a.shape()[axis_index]
    });
    (n, axis)
}

// Use Cow or SmallVec?
#[inline]
pub(super) fn resolve_sizes_and_axes_unchecked<'a>(
    a: &Array,
    s: Option<&'a [i32]>,
    axes: Option<&'a [i32]>,
) -> LengthsAndAxes {
    match (s, axes) {
        (Some(s), Some(axes)) => {
            let valid_s = SmallVec::<[i32; DEFAULT_STACK_VEC_LEN]>::from_slice(s);
            let valid_axes = SmallVec::<[i32; DEFAULT_STACK_VEC_LEN]>::from_slice(axes);
            (valid_s, valid_axes)
        }
        (Some(s), None) => {
            let valid_s = SmallVec::<[i32; DEFAULT_STACK_VEC_LEN]>::from_slice(s);
            let valid_axes = (-(valid_s.len() as i32)..0).collect();
            (valid_s, valid_axes)
        }
        (None, Some(axes)) => {
            let valid_s = axes
                .iter()
                .map(|&axis| {
                    let axis_index = resolve_index_unchecked(axis, a.ndim());
                    a.shape()[axis_index]
                })
                .collect();
            let valid_axes = SmallVec::<[i32; DEFAULT_STACK_VEC_LEN]>::from_slice(axes);
            (valid_s, valid_axes)
        }
        (None, None) => {
            let valid_s: SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> =
                (0..a.ndim()).map(|axis| a.shape()[axis]).collect();
            let valid_axes = (-(valid_s.len() as i32)..0).collect();
            (valid_s, valid_axes)
        }
    }
}

pub(super) fn resolve_lengths_and_axes(
    shape: &[i32],
    lengths: Option<&[i32]>,
    axes: &Axes,
) -> Result<LengthsAndAxes> {
    let ndim =
        i32::try_from(shape.len()).map_err(|_| Exception::custom("FFT rank exceeds i32::MAX"))?;
    let axes = match axes {
        Axes::All => (0..ndim).collect::<SmallVec<_>>(),
        Axes::Axis(axis) => SmallVec::from_slice(&[*axis]),
        Axes::Axes(axes) => SmallVec::from_slice(axes),
    };

    for &axis in &axes {
        if axis < -ndim || axis >= ndim {
            return Err(Exception::custom(format!(
                "FFT axis {axis} is out of bounds for rank {ndim}"
            )));
        }
    }

    let lengths = match lengths {
        Some(lengths) => {
            if lengths.len() != axes.len() {
                return Err(Exception::custom("FFT options require one length per axis"));
            }
            if lengths.iter().any(|&length| length <= 0) {
                return Err(Exception::custom("FFT lengths must be positive"));
            }
            SmallVec::from_slice(lengths)
        }
        None => axes
            .iter()
            .map(|&axis| {
                let index = if axis < 0 { ndim + axis } else { axis };
                shape[index as usize]
            })
            .collect(),
    };

    Ok((lengths, axes))
}

pub(super) fn legacy_fftn_options(
    lengths: Option<&[i32]>,
    axes: Option<&[i32]>,
) -> Result<super::FftnOptions> {
    let axes = match (lengths, axes) {
        (_, Some(axes)) => Axes::from(axes),
        (Some(lengths), None) => {
            let count = i32::try_from(lengths.len())
                .map_err(|_| Exception::custom("FFT axis count exceeds i32::MAX"))?;
            Axes::Axes((-count..0).collect())
        }
        (None, None) => Axes::All,
    };
    Ok(super::FftnOptions {
        lengths: lengths.map(<[i32]>::to_vec),
        axes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Axes;

    #[test]
    fn correlated_fft_options_reject_mismatched_lengths_and_axes() {
        let error =
            resolve_lengths_and_axes(&[2, 3, 4], Some(&[8, 9]), &Axes::Axis(1)).unwrap_err();
        assert!(error.what().contains("one length per axis"));
    }

    #[test]
    fn axes_all_resolves_every_dimension() {
        let (lengths, axes) = resolve_lengths_and_axes(&[2, 3, 4], None, &Axes::All).unwrap();
        assert_eq!(lengths.as_slice(), &[2, 3, 4]);
        assert_eq!(axes.as_slice(), &[0, 1, 2]);
    }

    #[test]
    fn inverse_real_length_rejects_empty_and_overflowing_dimensions() {
        assert!(resolve_inverse_real_length(0).is_err());
        assert!(resolve_inverse_real_length(i32::MAX).is_err());
        assert_eq!(resolve_inverse_real_length(4).unwrap(), 6);
    }
}
