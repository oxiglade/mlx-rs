use crate::module::{Module, Param};
use crate::{
    error::{ConvBuildError, Exception},
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
    err = ConvBuildError,
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

fn build_conv1d(builder: Conv1dBuilder) -> Result<Conv1d, ConvBuildError> {
    let input_channels = builder.input_channels;
    let output_channels = builder.output_channels;
    let kernel_size = builder.kernel_size;
    let with_bias = builder.bias;
    let groups = checked_groups(input_channels, output_channels, builder.groups)?;

    let scale = f32::sqrt(1.0f32 / (input_channels * kernel_size) as f32);
    let weight = uniform::<_, f32>(
        -scale,
        scale,
        &[output_channels, kernel_size, input_channels / groups],
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
        groups,
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
    err = ConvBuildError,
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

fn build_conv2d(builder: Conv2dBuilder) -> Result<Conv2d, ConvBuildError> {
    let input_channels = builder.input_channels;
    let output_channels = builder.output_channels;
    let kernel_size: (i32, i32) = builder.kernel_size.into();
    let with_bias = builder.bias;
    let padding = builder.padding.into();
    let stride = builder.stride.into();
    let dilation = builder.dilation.into();
    let groups = checked_groups(input_channels, output_channels, builder.groups)?;

    let scale = f32::sqrt(1.0 / (input_channels * kernel_size.0 * kernel_size.1) as f32);
    let weight = uniform::<_, f32>(
        -scale,
        scale,
        &[
            output_channels,
            kernel_size.0,
            kernel_size.1,
            input_channels / groups,
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
        groups,
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
    err = ConvBuildError,
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

fn build_conv3d(builder: Conv3dBuilder) -> Result<Conv3d, ConvBuildError> {
    let input_channels = builder.input_channels;
    let output_channels = builder.output_channels;
    let kernel_size: (i32, i32, i32) = builder.kernel_size.into();
    let with_bias = builder.bias;
    let padding = builder.padding.into();
    let stride = builder.stride.into();
    let dilation = builder.dilation.into();
    if builder.groups != 1 {
        return Err(ConvBuildError::GroupedConv3d(builder.groups));
    }

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

/// Validates `groups` for a grouped convolution: it must be positive and divide both channel
/// counts, matching the weight shape `[C_out, ..., C_in / groups]` that MLX expects.
fn checked_groups(
    input_channels: i32,
    output_channels: i32,
    groups: i32,
) -> Result<i32, ConvBuildError> {
    if groups < 1 || input_channels % groups != 0 || output_channels % groups != 0 {
        return Err(ConvBuildError::InvalidGroups {
            groups,
            input_channels,
            output_channels,
        });
    }
    Ok(groups)
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

    #[test]
    fn test_conv1d_depthwise_matches_mlx() {
        use crate::{builder::Builder, nn::Conv1dBuilder, ops::arange};

        let x = &arange::<_, f32>(None, 48, None)
            .unwrap()
            .reshape(&[2, 6, 4])
            .unwrap()
            / 10.0;
        let w = &(&arange::<_, f32>(None, 12, None)
            .unwrap()
            .reshape(&[4, 3, 1])
            .unwrap()
            / 7.0)
            - 0.5;

        let mut conv = Conv1dBuilder::new(4, 4, 3)
            .padding(1)
            .groups(4)
            .build()
            .unwrap();
        assert_eq!(conv.weight.shape(), &[4, 3, 1]);
        conv.weight.value = w;
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 6, 4]);

        // Python MLX 0.32.2: `mx.conv1d(x, w, padding=1, groups=4)`
        #[rustfmt::skip]
        let expected: [f32; 48] = [
            -0.08571428, 0.1142857, 0.4857143, 1.028571,
            -0.3142857, 0.2214286, 1.014286, 2.064286,
            -0.7428572, 0.3071429, 1.614286, 3.178571,
            -1.171429, 0.3928572, 2.214286, 4.292857,
            -1.6, 0.4785715, 2.814286, 5.407143,
            -1.514286, 0.02857148, 1.742857, 3.628572,
            -1.457143, 0.8000001, 3.228571, 5.828571,
            -2.885714, 0.7357144, 4.614286, 8.75,
            -3.314286, 0.8214287, 5.214286, 9.864285,
            -3.742857, 0.907143, 5.814286, 10.97857,
            -4.171429, 0.9928573, 6.414286, 12.09286,
            -3.571429, 0.02857158, 3.8, 7.742857,
        ];
        for (got, want) in y.to_vec_exact::<f32>().unwrap().iter().zip(expected) {
            assert_float_eq!(*got, want, abs <= 1e-5 * (1.0 + want.abs()));
        }
    }

    #[test]
    fn test_conv2d_grouped_weight_shape() {
        use crate::{builder::Builder, nn::Conv2dBuilder};

        let mut conv = Conv2dBuilder::new(4, 8, 3).groups(2).build().unwrap();
        assert_eq!(conv.weight.shape(), &[8, 3, 3, 2]);
        let x = uniform::<_, f32>(0.0, 1.0, &[1, 5, 5, 4], None).unwrap();
        assert_eq!(conv.forward(&x).unwrap().shape(), &[1, 3, 3, 8]);
    }

    #[test]
    fn test_conv_groups_must_divide_channels() {
        use crate::{
            builder::Builder,
            error::ConvBuildError,
            nn::{Conv1dBuilder, Conv2dBuilder, Conv3dBuilder},
        };

        let invalid = |groups, input_channels, output_channels| ConvBuildError::InvalidGroups {
            groups,
            input_channels,
            output_channels,
        };
        assert_eq!(
            Conv1dBuilder::new(6, 4, 3).groups(4).build().unwrap_err(),
            invalid(4, 6, 4)
        );
        assert_eq!(
            Conv1dBuilder::new(4, 6, 3).groups(4).build().unwrap_err(),
            invalid(4, 4, 6)
        );
        assert_eq!(
            Conv1dBuilder::new(4, 4, 3).groups(0).build().unwrap_err(),
            invalid(0, 4, 4)
        );
        assert_eq!(
            Conv2dBuilder::new(4, 6, 3).groups(4).build().unwrap_err(),
            invalid(4, 4, 6)
        );
        assert_eq!(
            Conv3dBuilder::new(4, 4, 3).groups(2).build().unwrap_err(),
            ConvBuildError::GroupedConv3d(2)
        );
    }
}
