use crate::module::{Module, Param};
use crate::{
    error::Exception,
    ops::{conv_transpose1d, conv_transpose2d, conv_transpose3d, zeros},
    random::uniform,
    Array,
};
use mlx_internal_macros::{Buildable, Builder};
use mlx_macros::ModuleParameters;

use crate::utils::{SingleOrPair, SingleOrTriple};

/// Builder for the `ConvTranspose1d` module.
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_conv_transpose_1d,
    err = Exception,
)]
pub struct ConvTranspose1dBuilder {
    /// The number of input channels.
    pub input_channels: i32,

    /// The number of output channels.
    pub output_channels: i32,

    /// The size of the convolution filters.
    pub kernel_size: i32,

    /// If `true`, add a learnable bias to the output. Default to [`ConvTranspose1d::DEFAULT_BIAS`] if not
    /// specified.
    #[builder(optional, default = ConvTranspose1d::DEFAULT_BIAS)]
    pub bias: bool,

    /// Padding. Default to [`ConvTranspose1d::DEFAULT_PADDING`] if not specified.
    #[builder(optional, default = ConvTranspose1d::DEFAULT_PADDING)]
    pub padding: i32,

    /// Output padding. Default to [`ConvTranspose1d::DEFAULT_OUTPUT_PADDING`] if not specified.
    #[builder(optional, default = ConvTranspose1d::DEFAULT_OUTPUT_PADDING)]
    pub output_padding: i32,

    /// Stride. Default to [`ConvTranspose1d::DEFAULT_STRIDE`] if not specified.
    #[builder(optional, default = ConvTranspose1d::DEFAULT_STRIDE)]
    pub stride: i32,

    /// Dilation. Default to [`ConvTranspose1d::DEFAULT_DILATION`] if not specified.
    #[builder(optional, default = ConvTranspose1d::DEFAULT_DILATION)]
    pub dilation: i32,
}

fn build_conv_transpose_1d(builder: ConvTranspose1dBuilder) -> Result<ConvTranspose1d, Exception> {
    let input_channels = builder.input_channels;
    let output_channels = builder.output_channels;
    let kernel_size = builder.kernel_size;

    let bias = builder.bias;
    let padding = builder.padding;
    let output_padding = builder.output_padding;
    let stride = builder.stride;

    let scale = f32::sqrt(1.0f32 / (input_channels * kernel_size) as f32);
    let weight = uniform::<_, f32>(
        -scale,
        scale,
        &[output_channels, kernel_size, input_channels],
        None,
    )?;
    let bias = if bias {
        Some(zeros::<f32>(&[output_channels])?)
    } else {
        None
    };

    Ok(ConvTranspose1d {
        weight: Param::new(weight),
        bias: Param::new(bias),
        padding,
        output_padding,
        stride,
        dilation: builder.dilation,
    })
}

/// Applies a 1-dimensional convolution over the multi-channel input sequence.
///
/// The channels are expected to be last i.e. the input shape should be `NLC` where:
///
/// - `N` is the batch dimension
/// - `L` is the sequence length
/// - `C` is the number of input channels
#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
pub struct ConvTranspose1d {
    /// The weight of the convolution layer.
    #[param]
    pub weight: Param<Array>,

    /// The bias of the convolution layer.
    #[param]
    pub bias: Param<Option<Array>>,

    /// Padding. Default to 0 if not specified.
    pub padding: i32,

    /// Output padding. Default to 0 if not specified.
    pub output_padding: i32,

    /// Stride. Default to 1 if not specified.
    pub stride: i32,

    /// Dilation. Default to [`ConvTranspose1d::DEFAULT_DILATION`] if not specified.
    pub dilation: i32,
}

impl ConvTranspose1d {
    /// Default value for `bias` if not specified.
    pub const DEFAULT_BIAS: bool = true;

    /// Default value for `padding` if not specified.
    pub const DEFAULT_PADDING: i32 = 0;

    /// Default value for `output_padding` if not specified.
    pub const DEFAULT_OUTPUT_PADDING: i32 = 0;

    /// Default value for `stride` if not specified.
    pub const DEFAULT_STRIDE: i32 = 1;

    /// Default value for `dilation` if not specified.
    pub const DEFAULT_DILATION: i32 = 1;
}

impl Module<&Array> for ConvTranspose1d {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let mut y = conv_transpose1d(
            x,
            self.weight.as_ref(),
            self.stride,
            self.padding,
            self.dilation,
            self.output_padding,
            None,
        )?;
        if let Some(bias) = &self.bias.value {
            y += bias;
        }
        Ok(y)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Builder for the `ConvTranspose2d` module.
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_conv_transpose_2d,
    err = Exception,
)]
pub struct ConvTranspose2dBuilder {
    /// The number of input channels.
    pub input_channels: i32,

    /// The number of output channels.
    pub output_channels: i32,

    /// The size of the convolution filters.
    pub kernel_size: SingleOrPair<i32>,

    /// If `true`, add a learnable bias to the output. Default to [`ConvTranspose2d::DEFAULT_BIAS`] if not
    /// specified.
    #[builder(optional, default = ConvTranspose2d::DEFAULT_BIAS)]
    bias: bool,

    /// Padding. Default to [`ConvTranspose2d::DEFAULT_PADDING`] if not specified.
    #[builder(optional, default = ConvTranspose2d::DEFAULT_PADDING)]
    padding: SingleOrPair<i32>,

    /// Output padding. Default to [`ConvTranspose2d::DEFAULT_OUTPUT_PADDING`] if not specified.
    #[builder(optional, default = ConvTranspose2d::DEFAULT_OUTPUT_PADDING)]
    output_padding: SingleOrPair<i32>,

    /// Stride. Default to [`ConvTranspose2d::DEFAULT_STRIDE`] if not specified.
    #[builder(optional, default = ConvTranspose2d::DEFAULT_STRIDE)]
    stride: SingleOrPair<i32>,

    /// Dilation. Default to [`ConvTranspose2d::DEFAULT_DILATION`] if not specified.
    #[builder(optional, default = ConvTranspose2d::DEFAULT_DILATION)]
    dilation: SingleOrPair<i32>,
}

fn build_conv_transpose_2d(builder: ConvTranspose2dBuilder) -> Result<ConvTranspose2d, Exception> {
    let input_channels = builder.input_channels;
    let output_channels = builder.output_channels;
    let kernel_size: (i32, i32) = builder.kernel_size.into();

    let bias = builder.bias;
    let padding = builder.padding.into();
    let output_padding = builder.output_padding.into();
    let stride = builder.stride.into();

    let scale = f32::sqrt(1.0 / (input_channels * kernel_size.0 * kernel_size.1) as f32);
    let weight = uniform::<_, f32>(
        -scale,
        scale,
        &[
            output_channels,
            kernel_size.0,
            kernel_size.1,
            input_channels,
        ],
        None,
    )?;
    let bias = if bias {
        Some(zeros::<f32>(&[output_channels])?)
    } else {
        None
    };

    Ok(ConvTranspose2d {
        weight: Param::new(weight),
        bias: Param::new(bias),
        padding,
        output_padding,
        stride,
        dilation: builder.dilation.into(),
    })
}

/// Applies a 2-dimensional convolution over the multi-channel input image.
///
/// The channels are expected to be last i.e. the input shape should be `NHWC` where:
///
/// - `N` is the batch dimension
/// - `H` is the input image height
/// - `W` is the input image width
/// - `C` is the number of input channels
#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
pub struct ConvTranspose2d {
    /// The weight of the convolution layer.
    #[param]
    pub weight: Param<Array>,

    /// The bias of the convolution layer.
    #[param]
    pub bias: Param<Option<Array>>,

    /// Padding. Default to `(0, 0)` if not specified.
    pub padding: (i32, i32),

    /// Output padding. Default to `(0, 0)` if not specified.
    pub output_padding: (i32, i32),

    /// Stride. Default to `(1, 1)` if not specified.
    pub stride: (i32, i32),

    /// Dilation. Default to [`ConvTranspose2d::DEFAULT_DILATION`] if not specified.
    pub dilation: (i32, i32),
}

impl ConvTranspose2d {
    /// Default value for `bias` if not specified.
    pub const DEFAULT_BIAS: bool = true;

    /// Default value for `padding` if not specified.
    pub const DEFAULT_PADDING: SingleOrPair<i32> = SingleOrPair::Pair(0, 0);

    /// Default value for `output_padding` if not specified.
    pub const DEFAULT_OUTPUT_PADDING: SingleOrPair<i32> = SingleOrPair::Pair(0, 0);

    /// Default value for `stride` if not specified.
    pub const DEFAULT_STRIDE: SingleOrPair<i32> = SingleOrPair::Pair(1, 1);

    /// Default value for `dilation` if not specified.
    pub const DEFAULT_DILATION: SingleOrPair<i32> = SingleOrPair::Pair(1, 1);
}

impl Module<&Array> for ConvTranspose2d {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let mut y = conv_transpose2d(
            x,
            self.weight.as_ref(),
            self.stride,
            self.padding,
            self.dilation,
            self.output_padding,
            None,
        )?;
        if let Some(bias) = &self.bias.value {
            y += bias;
        }
        Ok(y)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Builder for the `ConvTranspose3d` module.
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_conv_transpose_3d,
    err = Exception,
)]
pub struct ConvTranspose3dBuilder {
    /// The number of input channels.
    pub input_channels: i32,

    /// The number of output channels.
    pub output_channels: i32,

    /// The size of the convolution filters.
    pub kernel_size: SingleOrTriple<i32>,

    /// If `true`, add a learnable bias to the output. Default to [`ConvTranspose3d::DEFAULT_BIAS`] if not
    /// specified.
    #[builder(optional, default = ConvTranspose3d::DEFAULT_BIAS)]
    pub bias: bool,

    /// Padding. Default to [`ConvTranspose3d::DEFAULT_PADDING`] if not specified.
    #[builder(optional, default = ConvTranspose3d::DEFAULT_PADDING)]
    pub padding: SingleOrTriple<i32>,

    /// Output padding. Default to [`ConvTranspose3d::DEFAULT_OUTPUT_PADDING`] if not specified.
    #[builder(optional, default = ConvTranspose3d::DEFAULT_OUTPUT_PADDING)]
    pub output_padding: SingleOrTriple<i32>,

    /// Stride. Default to [`ConvTranspose3d::DEFAULT_STRIDE`] if not specified.
    #[builder(optional, default = ConvTranspose3d::DEFAULT_STRIDE)]
    pub stride: SingleOrTriple<i32>,

    /// Dilation. Default to [`ConvTranspose3d::DEFAULT_DILATION`] if not specified.
    #[builder(optional, default = ConvTranspose3d::DEFAULT_DILATION)]
    pub dilation: SingleOrTriple<i32>,
}

fn build_conv_transpose_3d(builder: ConvTranspose3dBuilder) -> Result<ConvTranspose3d, Exception> {
    let input_channels = builder.input_channels;
    let output_channels = builder.output_channels;
    let kernel_size: (i32, i32, i32) = builder.kernel_size.into();

    let bias = builder.bias;
    let padding = builder.padding.into();
    let output_padding = builder.output_padding.into();
    let stride = builder.stride.into();

    let scale =
        f32::sqrt(1.0 / (input_channels * kernel_size.0 * kernel_size.1 * kernel_size.2) as f32);
    let weight = uniform::<_, f32>(
        -scale,
        scale,
        &[
            output_channels,
            kernel_size.0,
            kernel_size.1,
            kernel_size.2,
            input_channels,
        ],
        None,
    )?;
    let bias = if bias {
        Some(zeros::<f32>(&[output_channels])?)
    } else {
        None
    };

    Ok(ConvTranspose3d {
        weight: Param::new(weight),
        bias: Param::new(bias),
        padding,
        output_padding,
        stride,
        dilation: builder.dilation.into(),
    })
}

/// Applies a 3-dimensional convolution over the multi-channel input image.
///
/// The channels are expected to be last i.e. the input shape should be `NHWC` where:
///
/// - `N` is the batch dimension
/// - `H` is the input image height
/// - `W` is the input image width
/// - `C` is the number of input channels
#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
pub struct ConvTranspose3d {
    /// The weight of the convolution layer.
    #[param]
    pub weight: Param<Array>,

    /// The bias of the convolution layer.
    #[param]
    pub bias: Param<Option<Array>>,

    /// Padding. Default to `(0, 0, 0)` if not specified.
    pub padding: (i32, i32, i32),

    /// Output padding. Default to `(0, 0, 0)` if not specified.
    pub output_padding: (i32, i32, i32),

    /// Stride. Default to `(1, 1, 1)` if not specified.
    pub stride: (i32, i32, i32),

    /// Dilation. Default to [`ConvTranspose3d::DEFAULT_DILATION`] if not specified.
    pub dilation: (i32, i32, i32),
}

impl ConvTranspose3d {
    /// Default value for `bias` if not specified.
    pub const DEFAULT_BIAS: bool = true;

    /// Default value for `padding` if not specified.
    pub const DEFAULT_PADDING: SingleOrTriple<i32> = SingleOrTriple::Triple(0, 0, 0);

    /// Default value for `output_padding` if not specified.
    pub const DEFAULT_OUTPUT_PADDING: SingleOrTriple<i32> = SingleOrTriple::Triple(0, 0, 0);

    /// Default value for `stride` if not specified.
    pub const DEFAULT_STRIDE: SingleOrTriple<i32> = SingleOrTriple::Triple(1, 1, 1);

    /// Default value for `dilation` if not specified.
    pub const DEFAULT_DILATION: SingleOrTriple<i32> = SingleOrTriple::Triple(1, 1, 1);
}

impl Module<&Array> for ConvTranspose3d {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let mut y = conv_transpose3d(
            x,
            self.weight.as_ref(),
            self.stride,
            self.padding,
            self.dilation,
            self.output_padding,
            None,
        )?;
        if let Some(bias) = &self.bias.value {
            y += bias;
        }
        Ok(y)
    }

    fn training_mode(&mut self, _: bool) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        builder::Builder,
        ops::arange,
        test_utils::{assert_array_eq, tolerances},
    };

    #[test]
    fn test_conv_transpose1d_dilation_matches_mlx() {
        let default = ConvTranspose1d::new(2, 3, 3).unwrap();
        assert_eq!(default.dilation, 1);
        assert_eq!(default.output_padding, 0);
        assert_eq!(default.weight.shape(), &[3, 3, 2]);
        let x = arange::<_, f32>(None, 8, None)
            .unwrap()
            .reshape(&[1, 4, 2])
            .unwrap()
            / 10.0;
        for (dilation, bias) in [(1, false), (2, true)] {
            let mut conv = ConvTranspose1dBuilder::new(2, 3, 3)
                .stride(2)
                .padding(1)
                .output_padding(1)
                .dilation(dilation)
                .bias(bias)
                .build()
                .unwrap();
            assert_eq!(conv.weight.shape(), &[3, 3, 2]);
            conv.weight.value = arange::<_, f32>(None, 18, None)
                .unwrap()
                .reshape(&[3, 3, 2])
                .unwrap()
                / 7.0;
            if bias {
                conv.bias.value = Some(Array::from_slice(&[0.25f32, -0.5, 1.0], &[3]));
            }
            let y = conv.forward(&x).unwrap();
            assert_eq!(y.shape(), &[1, 6 + 2 * dilation, 3]);
            let mut expected =
                conv_transpose1d(&x, conv.weight.as_ref(), 2, 1, dilation, 1, None).unwrap();
            if let Some(bias) = &conv.bias.value {
                expected += bias;
            }
            assert_array_eq(
                y,
                expected,
                tolerances::MLX_DEFAULT.rtol,
                tolerances::MLX_DEFAULT.atol,
            );
        }
    }

    #[test]
    fn test_conv_transpose2d_dilation_matches_mlx() {
        let default = ConvTranspose2d::new(2, 3, (2, 3)).unwrap();
        assert_eq!(default.dilation, (1, 1));
        assert_eq!(default.output_padding, (0, 0));
        assert_eq!(default.weight.shape(), &[3, 2, 3, 2]);
        let x = arange::<_, f32>(None, 24, None)
            .unwrap()
            .reshape(&[1, 3, 4, 2])
            .unwrap()
            / 10.0;
        for (dilation, bias) in [((1, 1), false), ((2, 3), true)] {
            let mut conv = ConvTranspose2dBuilder::new(2, 3, (2, 3))
                .stride((2, 1))
                .padding((1, 0))
                .output_padding((1, 0))
                .dilation(dilation)
                .bias(bias)
                .build()
                .unwrap();
            assert_eq!(conv.weight.shape(), &[3, 2, 3, 2]);
            conv.weight.value = arange::<_, f32>(None, 36, None)
                .unwrap()
                .reshape(&[3, 2, 3, 2])
                .unwrap()
                / 7.0;
            if bias {
                conv.bias.value = Some(Array::from_slice(&[0.25f32, -0.5, 1.0], &[3]));
            }
            let y = conv.forward(&x).unwrap();
            assert_eq!(y.shape(), &[1, 4 + dilation.0, 4 + 2 * dilation.1, 3]);
            let mut expected = conv_transpose2d(
                &x,
                conv.weight.as_ref(),
                (2, 1),
                (1, 0),
                dilation,
                (1, 0),
                None,
            )
            .unwrap();
            if let Some(bias) = &conv.bias.value {
                expected += bias;
            }
            assert_array_eq(
                y,
                expected,
                tolerances::MLX_DEFAULT.rtol,
                tolerances::MLX_DEFAULT.atol,
            );
        }
    }

    #[test]
    fn test_conv_transpose3d_dilation_matches_mlx() {
        let default = ConvTranspose3d::new(2, 3, (2, 2, 2)).unwrap();
        assert_eq!(default.dilation, (1, 1, 1));
        assert_eq!(default.output_padding, (0, 0, 0));
        assert_eq!(default.weight.shape(), &[3, 2, 2, 2, 2]);
        let x = arange::<_, f32>(None, 48, None)
            .unwrap()
            .reshape(&[1, 2, 3, 4, 2])
            .unwrap()
            / 10.0;
        for (dilation, bias) in [((1, 1, 1), false), ((2, 3, 2), true)] {
            let mut conv = ConvTranspose3dBuilder::new(2, 3, (2, 2, 2))
                .stride((2, 1, 2))
                .padding((0, 1, 0))
                .output_padding((1, 0, 1))
                .dilation(dilation)
                .bias(bias)
                .build()
                .unwrap();
            assert_eq!(conv.weight.shape(), &[3, 2, 2, 2, 2]);
            conv.weight.value = arange::<_, f32>(None, 48, None)
                .unwrap()
                .reshape(&[3, 2, 2, 2, 2])
                .unwrap()
                / 7.0;
            if bias {
                conv.bias.value = Some(Array::from_slice(&[0.25f32, -0.5, 1.0], &[3]));
            }
            let y = conv.forward(&x).unwrap();
            assert_eq!(
                y.shape(),
                &[1, 4 + dilation.0, 1 + dilation.1, 8 + dilation.2, 3]
            );
            let mut expected = conv_transpose3d(
                &x,
                conv.weight.as_ref(),
                (2, 1, 2),
                (0, 1, 0),
                dilation,
                (1, 0, 1),
                None,
            )
            .unwrap();
            if let Some(bias) = &conv.bias.value {
                expected += bias;
            }
            assert_array_eq(
                y,
                expected,
                tolerances::MLX_DEFAULT.rtol,
                tolerances::MLX_DEFAULT.atol,
            );
        }
    }
}
