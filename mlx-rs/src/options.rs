/// Axis selection for operations that accept all, one, or several axes.
///
/// Convert an `i32`, `Vec<i32>`, slice, or array with `.into()`. Operations with additional
/// independent defaults use a concrete `FooOptions: Default` type; for example,
/// [`crate::fft::FftnOptions`] combines FFT lengths with this selection.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Axes {
    /// Select every axis.
    #[default]
    All,

    /// Select one axis.
    Axis(i32),

    /// Select several axes in the given order.
    Axes(Vec<i32>),
}

impl From<i32> for Axes {
    fn from(axis: i32) -> Self {
        Self::Axis(axis)
    }
}

impl From<Vec<i32>> for Axes {
    fn from(axes: Vec<i32>) -> Self {
        Self::Axes(axes)
    }
}

impl From<&[i32]> for Axes {
    fn from(axes: &[i32]) -> Self {
        Self::Axes(axes.to_vec())
    }
}

impl From<&Vec<i32>> for Axes {
    fn from(axes: &Vec<i32>) -> Self {
        Self::Axes(axes.clone())
    }
}

impl<const N: usize> From<[i32; N]> for Axes {
    fn from(axes: [i32; N]) -> Self {
        Self::Axes(axes.into())
    }
}

impl<const N: usize> From<&[i32; N]> for Axes {
    fn from(axes: &[i32; N]) -> Self {
        Self::Axes(axes.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ergonomic_conversions_preserve_axis_selection() {
        assert_eq!(Axes::from(2), Axes::Axis(2));
        assert_eq!(Axes::from(vec![0, -1]), Axes::Axes(vec![0, -1]));
        assert_eq!(Axes::from(&vec![5, 6]), Axes::Axes(vec![5, 6]));
        assert_eq!(Axes::from(&[1, 2]), Axes::Axes(vec![1, 2]));
        assert_eq!(Axes::from(&[3, 4][..]), Axes::Axes(vec![3, 4]));
    }
}
