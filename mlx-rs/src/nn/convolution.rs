use crate::module::{Module, Param};
use crate::{
    error::Exception,
    ops::{conv1d, conv2d, zeros},
    random::uniform,
    Array,
};
use mlx_internal_macros::{Buildable, Builder};
use mlx_macros::ModuleParameters;

use crate::utils::{SingleOrPair, SingleOrTriple};

/// Builder for the `Conv1d` module.
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_conv1d,
    err = Exception,
)]
pub struct Conv1dBuilder {
    /// Number of input channels.
    pub input_channels: i32,

    /// Number of output channels.
    pub output_channels: i32,

    /// Size of the convolution filters.
    pub kernel_size: i32,

    /// If `true`, add a learnable bias to the output. Default to [`Conv1d::DEFAULT_BIAS`] if not
    /// specified.
    #[builder(optional, default = Conv1d::DEFAULT_BIAS)]
    pub bias: bool,

    /// Stride. Default to [`Conv1d::DEFAULT_STRIDE`] if not specified.
    #[builder(optional, default = Conv1d::DEFAULT_STRIDE)]
    pub stride: i32,

    /// Padding. Default to [`Conv1d::DEFAULT_PADDING`] if not specified.
    #[builder(optional, default = Conv1d::DEFAULT_PADDING)]
    pub padding: i32,

    /// Dilation. Default to [`Conv1d::DEFAULT_DILATION`] if not specified.
    #[builder(optional, default = Conv1d::DEFAULT_DILATION)]
    pub dilation: i32,

    /// Groups. Default to [`Conv1d::DEFAULT_GROUPS`] if not specified.
    #[builder(optional, default = Conv1d::DEFAULT_GROUPS)]
    pub groups: i32,
}

fn build_conv1d(builder: Conv1dBuilder) -> Result<Conv1d, Exception> {
    let input_channels = builder.input_channels;
    let output_channels = builder.output_channels;
    let kernel_size = builder.kernel_size;
    let with_bias = builder.bias;

    let scale = f32::sqrt(1.0f32 / (input_channels * kernel_size) as f32);
    let weight = uniform::<_, f32>(
        -scale,
        scale,
        &[output_channels, kernel_size, input_channels],
        None,
    )?;
    let bias = if with_bias {
        Some(zeros::<f32>(&[output_channels])?)
    } else {
        None
    };

    Ok(Conv1d {
        weight: Param::new(weight),
        bias: Param::new(bias),
        stride: builder.stride,
        padding: builder.padding,
        dilation: builder.dilation,
        groups: builder.groups,
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
pub struct Conv1d {
    /// The weight of the convolution layer.
    #[param]
    pub weight: Param<Array>,

    /// The bias of the convolution layer.
    #[param]
    pub bias: Param<Option<Array>>,

    /// Stride. Default to [`Conv1d::DEFAULT_STRIDE`] if not specified.
    pub stride: i32,

    /// Padding. Default to [`Conv1d::DEFAULT_PADDING`] if not specified.
    pub padding: i32,

    /// Dilation. Default to [`Conv1d::DEFAULT_DILATION`] if not specified.
    pub dilation: i32,

    /// Groups. Default to [`Conv1d::DEFAULT_GROUPS`] if not specified.
    pub groups: i32,
}

impl Conv1d {
    /// Default value for `with_bias` if not specified.
    pub const DEFAULT_BIAS: bool = true;

    /// Default value for `stride` if not specified.
    pub const DEFAULT_STRIDE: i32 = 1;

    /// Default value for `padding` if not specified.
    pub const DEFAULT_PADDING: i32 = 0;

    /// Default value for `dilation` if not specified.
    pub const DEFAULT_DILATION: i32 = 1;

    /// Default value for `groups` if not specified.
    pub const DEFAULT_GROUPS: i32 = 1;
}

impl Module<&Array> for Conv1d {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let mut y = conv1d(
            x,
            self.weight.as_ref(),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )?;
        if let Some(bias) = &self.bias.value {
            y += bias;
        }
        Ok(y)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Builder for the `Conv2d` module.
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_conv2d,
    err = Exception,
)]
pub struct Conv2dBuilder {
    /// Number of input channels.
    pub input_channels: i32,

    /// Number of output channels.
    pub output_channels: i32,

    /// Size of the convolution filters.
    pub kernel_size: SingleOrPair<i32>,

    /// If `true`, add a learnable bias to the output. Default to [`Conv2d::DEFAULT_BIAS`] if not
    /// specified.
    #[builder(optional, default = Conv2d::DEFAULT_BIAS)]
    pub bias: bool,

    /// Stride. Default to [`Conv2d::DEFAULT_STRIDE`] if not specified.
    #[builder(optional, default = Conv2d::DEFAULT_STRIDE)]
    pub stride: SingleOrPair<i32>,

    /// Padding. Default to [`Conv2d::DEFAULT_PADDING`] if not specified.
    #[builder(optional, default = Conv2d::DEFAULT_PADDING)]
    pub padding: SingleOrPair<i32>,

    /// Dilation. Default to [`Conv2d::DEFAULT_DILATION`] if not specified.
    #[builder(optional, default = Conv2d::DEFAULT_DILATION)]
    pub dilation: SingleOrPair<i32>,

    /// Groups. Default to [`Conv2d::DEFAULT_GROUPS`] if not specified.
    #[builder(optional, default = Conv2d::DEFAULT_GROUPS)]
    pub groups: i32,
}

fn build_conv2d(builder: Conv2dBuilder) -> Result<Conv2d, Exception> {
    let input_channels = builder.input_channels;
    let output_channels = builder.output_channels;
    let kernel_size: (i32, i32) = builder.kernel_size.into();
    let with_bias = builder.bias;
    let padding = builder.padding.into();
    let stride = builder.stride.into();
    let dilation = builder.dilation.into();

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
    let bias = if with_bias {
        Some(zeros::<f32>(&[output_channels])?)
    } else {
        None
    };

    Ok(Conv2d {
        weight: Param::new(weight),
        bias: Param::new(bias),
        stride,
        padding,
        dilation,
        groups: builder.groups,
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
pub struct Conv2d {
    /// The weight of the convolution layer.
    #[param]
    pub weight: Param<Array>,

    /// The bias of the convolution layer.
    #[param]
    pub bias: Param<Option<Array>>,

    /// Stride. Default to [`Conv2d::DEFAULT_STRIDE`] if not specified.
    pub stride: (i32, i32),

    /// Padding. Default to [`Conv2d::DEFAULT_PADDING`] if not specified.
    pub padding: (i32, i32),

    /// Dilation. Default to [`Conv2d::DEFAULT_DILATION`] if not specified.
    pub dilation: (i32, i32),

    /// Groups. Default to [`Conv2d::DEFAULT_GROUPS`] if not specified.
    pub groups: i32,
}

impl Conv2d {
    /// Default value for `with_bias` if not specified.
    pub const DEFAULT_BIAS: bool = true;

    /// Default value for `stride` if not specified.
    pub const DEFAULT_STRIDE: SingleOrPair = SingleOrPair::Pair(1, 1);

    /// Default value for `padding` if not specified.
    pub const DEFAULT_PADDING: SingleOrPair = SingleOrPair::Pair(0, 0);

    /// Default value for `dilation` if not specified.
    pub const DEFAULT_DILATION: SingleOrPair = SingleOrPair::Pair(1, 1);

    /// Default value for `groups` if not specified.
    pub const DEFAULT_GROUPS: i32 = 1;
}

impl Module<&Array> for Conv2d {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let mut y = conv2d(
            x,
            self.weight.as_ref(),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )?;
        if let Some(bias) = &self.bias.value {
            y += bias;
        }
        Ok(y)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Builder for the `Conv3d` module.
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_conv3d,
    err = Exception,
)]
pub struct Conv3dBuilder {
    /// Number of input channels.
    pub input_channels: i32,

    /// Number of output channels.
    pub output_channels: i32,

    /// Size of the convolution filters.
    pub kernel_size: SingleOrTriple<i32>,

    /// If `true`, add a learnable bias to the output. Default to [`Conv3d::DEFAULT_BIAS`] if not
    /// specified.
    #[builder(optional, default = Conv3d::DEFAULT_BIAS)]
    pub bias: bool,

    /// Stride. Default to [`Conv3d::DEFAULT_STRIDE`] if not specified.
    #[builder(optional, default = Conv3d::DEFAULT_STRIDE)]
    pub stride: SingleOrTriple<i32>,

    /// Padding. Default to [`Conv3d::DEFAULT_PADDING`] if not specified.
    #[builder(optional, default = Conv3d::DEFAULT_PADDING)]
    pub padding: SingleOrTriple<i32>,

    /// Dilation. Default to [`Conv3d::DEFAULT_DILATION`] if not specified.
    #[builder(optional, default = Conv3d::DEFAULT_DILATION)]
    pub dilation: SingleOrTriple<i32>,

    /// Groups. Default to [`Conv3d::DEFAULT_GROUPS`] if not specified.
    #[builder(optional, default = Conv3d::DEFAULT_GROUPS)]
    pub groups: i32,
}

fn build_conv3d(builder: Conv3dBuilder) -> Result<Conv3d, Exception> {
    let input_channels = builder.input_channels;
    let output_channels = builder.output_channels;
    let kernel_size: (i32, i32, i32) = builder.kernel_size.into();
    let with_bias = builder.bias;
    let padding = builder.padding.into();
    let stride = builder.stride.into();
    let dilation = builder.dilation.into();

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
    let bias = if with_bias {
        Some(zeros::<f32>(&[output_channels])?)
    } else {
        None
    };

    Ok(Conv3d {
        weight: Param::new(weight),
        bias: Param::new(bias),
        stride,
        padding,
        dilation,
        groups: builder.groups,
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
pub struct Conv3d {
    /// The weight of the convolution layer.
    #[param]
    pub weight: Param<Array>,

    /// The bias of the convolution layer.
    #[param]
    pub bias: Param<Option<Array>>,

    /// Stride. Default to `(1, 1, 1)` if not specified.
    pub stride: (i32, i32, i32),

    /// Padding. Default to `(0, 0, 0)` if not specified.
    pub padding: (i32, i32, i32),

    /// Dilation. Default to `(1, 1, 1)` if not specified.
    pub dilation: (i32, i32, i32),

    /// Groups. Default to 1 if not specified.
    pub groups: i32,
}

impl Conv3d {
    /// Default value for `with_bias` if not specified.
    pub const DEFAULT_BIAS: bool = true;

    /// Default value for `stride` if not specified.
    pub const DEFAULT_STRIDE: SingleOrTriple<i32> = SingleOrTriple::Triple(1, 1, 1);

    /// Default value for `padding` if not specified.
    pub const DEFAULT_PADDING: SingleOrTriple<i32> = SingleOrTriple::Triple(0, 0, 0);

    /// Default value for `dilation` if not specified.
    pub const DEFAULT_DILATION: SingleOrTriple<i32> = SingleOrTriple::Triple(1, 1, 1);

    /// Default value for `groups` if not specified.
    pub const DEFAULT_GROUPS: i32 = 1;
}

impl Module<&Array> for Conv3d {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let mut y = crate::ops::conv3d(
            x,
            self.weight.as_ref(),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )?;
        if let Some(bias) = &self.bias.value {
            y += bias;
        }
        Ok(y)
    }

    fn training_mode(&mut self, _: bool) {}
}

// The following tests are ported from the swift bindings:
// mlx-swift/Tests/MLXTests/IntegrationTests.swift
#[cfg(test)]
mod tests {
    use crate::module::Module;
    use crate::{random::uniform, Dtype};
    use float_eq::assert_float_eq;

    use crate::nn::Conv1d;

    #[test]
    fn test_conv1d() {
        crate::random::seed(819).unwrap();
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item_exact::<f32>(),
            0.512_987_5,
            abs <= 0.010_259_75
        );
        assert_float_eq!(
            a.sum(None).unwrap().item_exact::<f32>(),
            131.324_8,
            abs <= 2.626_496
        );
        let result = Conv1d::new(16, 2, 8).unwrap().forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 1, 2]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            0.264_865_2,
            abs <= 0.005_297_303_7
        );
        assert_float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            1.059_460_8,
            abs <= 0.021_189_215
        );
    }

    #[test]
    fn test_conv2d() {
        crate::random::seed(62).unwrap();
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 8, 4], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 8, 4]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item_exact::<f32>(),
            0.522_504_27,
            abs <= 0.010_450_086
        );
        assert_float_eq!(
            a.sum(None).unwrap().item_exact::<f32>(),
            267.522_2,
            abs <= 5.350_444
        );
        let result = crate::nn::Conv2d::new(4, 2, (8, 8))
            .unwrap()
            .forward(&a)
            .unwrap();
        assert_eq!(result.shape(), &[2, 1, 1, 2]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            -0.279_321_5,
            abs <= 0.005_586_43
        );
        assert_float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            -1.117_286,
            abs <= 0.022_345_72
        );
    }
}
