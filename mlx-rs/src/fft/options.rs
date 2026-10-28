use crate::Axes;

/// Options shared by the n-dimensional FFT and real FFT transforms.
///
/// `lengths` and `axes` are correlated: when lengths are provided, there must be exactly one
/// length per selected axis. An all-axis transform may provide one length per input dimension.
/// A partial transform must set both fields, for example:
///
/// ```
/// # use mlx_rs::{fft::FftnOptions, Axes};
/// let options = FftnOptions {
///     lengths: Some(vec![8, 16]),
///     axes: Axes::from([-2, -1]),
/// };
/// # let _ = options;
/// ```
///
/// Axis-only selection may use `FftnOptions { axes: (-1).into(), ..Default::default() }`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FftnOptions {
    /// Sizes of the transformed axes, or the input sizes when omitted.
    pub lengths: Option<Vec<i32>>,

    /// Axes to transform.
    pub axes: Axes,
}
