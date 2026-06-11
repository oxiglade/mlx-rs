use std::borrow::Cow;

use crate::{
    array,
    error::Exception,
    module::{Module, Param},
    ops::{ones, rsqrt, zeros},
    Array,
};
use mlx_internal_macros::{Buildable, Builder};
use mlx_macros::ModuleParameters;

fn instance_norm(x: &Array, axes: &[i32], eps: &Array) -> Result<Array, Exception> {
    // Compute stats
    let mean = x.mean_axes(axes, true)?;
    let variance = x.var_axes(axes, true, None)?;

    // Normalize
    let x = x.subtract(&mean)?.multiply(rsqrt(&variance.add(eps)?)?)?;

    Ok(x)
}

/// Builder for [`InstanceNorm`].
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_instance_norm,
    err = Exception,
)]
pub struct InstanceNormBuilder {
    /// Number of features in the input
    pub dimensions: i32,

    /// Value added to the denominator for numerical stability. Default to
    /// [`InstanceNorm::DEFAULT_EPS`].
    #[builder(optional, default = InstanceNorm::DEFAULT_EPS)]
    pub eps: f32,

    /// If `true`, addes a trainable `weight` and `bias`. Default to
    /// [`InstanceNorm::DEFAULT_AFFINE`].
    #[builder(optional, default = InstanceNorm::DEFAULT_AFFINE)]
    pub affine: bool,
}

fn build_instance_norm(builder: InstanceNormBuilder) -> Result<InstanceNorm, Exception> {
    let eps = builder.eps;
    let affine = builder.affine;

    let (weight, bias) = if affine {
        (
            Some(ones::<f32>(&[builder.dimensions])?),
            Some(zeros::<f32>(&[builder.dimensions])?),
        )
    } else {
        (None, None)
    };

    Ok(InstanceNorm {
        dimensions: builder.dimensions,
        eps: array!(eps),
        weight: Param::new(weight),
        bias: Param::new(bias),
    })
}

/// Applies instance normalization [1] on the inputs.
///
/// ### References
///
/// 1. [https://arxiv.org/abs/1607.08022](https://arxiv.org/abs/1607.08022)
#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
pub struct InstanceNorm {
    /// Number of features in the input
    pub dimensions: i32,

    /// Value added to the denominator for numerical stability.
    pub eps: Array,

    /// An optional trainable weight
    #[param]
    pub weight: Param<Option<Array>>,

    /// An optional trainable bias
    #[param]
    pub bias: Param<Option<Array>>,
}

impl InstanceNorm {
    /// Default value for `eps`.
    pub const DEFAULT_EPS: f32 = 1e-5;

    /// Disable trainable `weight` and `bias` by default.
    pub const DEFAULT_AFFINE: bool = false;
}

impl Module<&Array> for InstanceNorm {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let reduction_axes = (1..x.ndim() as i32 - 1).collect::<Vec<_>>();

        let x = instance_norm(x, &reduction_axes, &self.eps)?;

        if let (Some(weight), Some(bias)) = (self.weight.as_ref(), self.bias.as_ref()) {
            weight.multiply(x)?.add(bias)
        } else {
            Ok(x)
        }
    }

    fn training_mode(&mut self, _mode: bool) {}
}

/// Builder for [`LayerNorm`].
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_layer_norm,
    err = Exception,
)]
pub struct LayerNormBuilder {
    /// Number of features in the input
    pub dimensions: i32,

    /// Value added to the denominator for numerical stability. Default to
    /// [`LayerNorm::DEFAULT_EPS`].
    #[builder(optional, default = LayerNorm::DEFAULT_EPS)]
    pub eps: f32,

    /// If `true`, addes a trainable `weight` and `bias`. Default to
    /// [`LayerNorm::DEFAULT_AFFINE`].
    #[builder(optional, default = LayerNorm::DEFAULT_AFFINE)]
    pub affine: bool,
}

fn build_layer_norm(builder: LayerNormBuilder) -> Result<LayerNorm, Exception> {
    let eps = builder.eps;
    let affine = builder.affine;

    let (weight, bias) = if affine {
        (
            Some(ones::<f32>(&[builder.dimensions])?),
            Some(zeros::<f32>(&[builder.dimensions])?),
        )
    } else {
        (None, None)
    };

    Ok(LayerNorm {
        dimensions: builder.dimensions,
        eps,
        weight: Param::new(weight),
        bias: Param::new(bias),
    })
}

/// Applies layer normalization [1] on the inputs.
///
/// ### References
///
/// 1. [https://arxiv.org/abs/1607.06450](https://arxiv.org/abs/1607.06450)
#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
pub struct LayerNorm {
    /// Number of features in the input
    pub dimensions: i32,

    /// Value added to the denominator for numerical stability.
    pub eps: f32,

    /// An optional trainable weight
    #[param]
    pub weight: Param<Option<Array>>,

    /// An optional trainable bias
    #[param]
    pub bias: Param<Option<Array>>,
}

impl LayerNorm {
    /// Default value for `eps`.
    pub const DEFAULT_EPS: f32 = 1e-5;

    /// Enable trainable `weight` and `bias` by default.
    pub const DEFAULT_AFFINE: bool = true;
}

impl Module<&Array> for LayerNorm {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let weight = self.weight.as_ref();
        let bias = self.bias.as_ref();
        let eps = self.eps;
        crate::fast::layer_norm(x, weight, bias, eps)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

/// Builder for [`RmsNorm`].
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_rms_norm,
    err = Exception,
)]
pub struct RmsNormBuilder {
    /// Number of features in the input
    pub dimensions: i32,

    /// Value added to the denominator for numerical stability. Default to
    /// [`RmsNorm::DEFAULT_EPS`].
    #[builder(optional, default = RmsNorm::DEFAULT_EPS)]
    pub eps: f32,
}

fn build_rms_norm(builder: RmsNormBuilder) -> Result<RmsNorm, Exception> {
    let weight = ones::<f32>(&[builder.dimensions])?;
    let eps = builder.eps;
    Ok(RmsNorm {
        weight: Param::new(weight),
        eps,
    })
}

/// Applies Root Mean Square normalization [1] to the inputs.
///
/// Concretely:
///
/// ```swift
/// weight * x * MLX.rsqrt(x.square().mean() + eps)
/// ```
///
/// where `weight` is initialized with ones and `eps` is a small float to
/// ensure the numerical stability of inverse square root.
///
/// ### References
///
/// 1. [https://arxiv.org/abs/1910.07467](https://arxiv.org/abs/1910.07467)
#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
pub struct RmsNorm {
    /// Weight
    #[param]
    pub weight: Param<Array>,

    /// A small float to ensure the numerical stability
    pub eps: f32,
}

impl RmsNorm {
    /// Default value for `eps`.
    pub const DEFAULT_EPS: f32 = 1e-5;
}

impl Module<&Array> for RmsNorm {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let weight = self.weight.as_ref();
        let eps = self.eps;
        crate::fast::rms_norm(x, Some(weight), eps)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

/// Builder for [`GroupNorm`].
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_group_norm,
    err = Exception,
)]
pub struct GroupNormBuilder {
    /// Number of groups to separate the features into
    pub group_count: i32,

    /// Number of features in the input
    pub dimensions: i32,

    /// Value added to the denominator for numerical stability. Default to
    /// [`GroupNorm::DEFAULT_EPS`].
    #[builder(optional, default = GroupNorm::DEFAULT_EPS)]
    pub eps: f32,

    /// If `true`, add a trainable `weight` and `bias`. Default to
    /// [`GroupNorm::DEFAULT_AFFINE`].
    #[builder(optional, default = GroupNorm::DEFAULT_AFFINE)]
    pub affine: bool,

    /// If `true`, perform the group normalization in the same order/grouping as PyTorch.
    /// Default to [`GroupNorm::DEFAULT_PYTORCH_COMPATIBLE`].
    #[builder(optional, default = GroupNorm::DEFAULT_PYTORCH_COMPATIBLE)]
    pub pytorch_compatible: bool,
}

fn build_group_norm(builder: GroupNormBuilder) -> Result<GroupNorm, Exception> {
    let eps = builder.eps;
    let affine = builder.affine;
    let pytorch_compatible = builder.pytorch_compatible;

    let (weight, bias) = if affine {
        (
            Some(ones::<f32>(&[builder.dimensions])?),
            Some(zeros::<f32>(&[builder.dimensions])?),
        )
    } else {
        (None, None)
    };

    Ok(GroupNorm {
        group_count: builder.group_count,
        dimensions: builder.dimensions,
        eps: array!(eps),
        pytorch_compatible,
        weight: Param::new(weight),
        bias: Param::new(bias),
    })
}

/// Applies Group Normalization [1] on the inputs.
///
/// ### References
///
/// 1. [https://arxiv.org/abs/1803.08494](https://arxiv.org/abs/1803.08494)
#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
pub struct GroupNorm {
    /// Number of groups to separate the features into
    pub group_count: i32,

    /// Number of features in the input
    pub dimensions: i32,

    /// Value added to the denominator for numerical stability.
    pub eps: Array,

    /// If `true`, perform the group normalization in the same order/grouping as PyTorch.
    pub pytorch_compatible: bool,

    /// An optional trainable weight
    #[param]
    pub weight: Param<Option<Array>>,

    /// An optional trainable bias
    #[param]
    pub bias: Param<Option<Array>>,
}

impl GroupNorm {
    /// Default value for `eps`.
    pub const DEFAULT_EPS: f32 = 1e-5;

    /// Enable trainable `weight` and `bias` by default.
    pub const DEFAULT_AFFINE: bool = true;

    /// Default value for `pytorch_compatible`.
    pub const DEFAULT_PYTORCH_COMPATIBLE: bool = false;

    fn pytorch_group_norm(&self, x: &Array) -> Result<Array, Exception> {
        let batch = x.dim(0);
        let dims = x.dim(-1);
        let rest = &x.shape()[1..x.ndim() - 1];
        let group_size = dims / self.group_count;

        // Split into groups
        let x = x.reshape(&[batch, -1, self.group_count, group_size])?;
        let x = x
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, self.group_count, -1])?;

        // Normalize
        let x = crate::fast::layer_norm(x, None, None, self.eps.item::<f32>())?;

        let x = x.reshape(&[batch, self.group_count, -1, group_size])?;

        let new_shape: Vec<_> = [batch]
            .into_iter()
            .chain(rest.iter().copied())
            .chain([dims])
            .collect();
        x.transpose_axes(&[0, 2, 1, 3])?.reshape(&new_shape[..])
    }

    fn group_norm(&self, x: &Array) -> Result<Array, Exception> {
        let batch = x.dim(0);
        let dims = x.dim(-1);
        let rest = &x.shape()[1..x.ndim() - 1];

        // Split into groups
        let x = x.reshape(&[batch, -1, self.group_count])?;

        // Normalize
        let x = instance_norm(&x, &[1], &self.eps)?;

        let new_shape: Vec<_> = [batch]
            .into_iter()
            .chain(rest.iter().copied())
            .chain([dims])
            .collect();
        x.reshape(&new_shape[..])
    }
}

impl Module<&Array> for GroupNorm {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let x = if self.pytorch_compatible {
            self.pytorch_group_norm(x)?
        } else {
            self.group_norm(x)?
        };

        if let (Some(weight), Some(bias)) = (self.weight.as_ref(), self.bias.as_ref()) {
            weight.multiply(&x)?.add(bias)
        } else {
            Ok(x)
        }
    }

    fn training_mode(&mut self, _mode: bool) {}
}

/// Builder for [`BatchNorm`].
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_batch_norm,
    err = Exception,
)]
pub struct BatchNormBuilder {
    /// Number of features in the input
    pub feature_count: i32,

    /// Value added to the denominator for numerical stability. Default to
    /// [`BatchNorm::DEFAULT_EPS`].
    #[builder(optional, default = BatchNorm::DEFAULT_EPS)]
    pub eps: f32,

    /// Momentum for updating the running mean and variance. Default to
    /// [`BatchNorm::DEFAULT_MOMENTUM`].
    #[builder(optional, default = BatchNorm::DEFAULT_MOMENTUM)]
    pub momentum: f32,

    /// If `true`, addes a trainable `weight` and `bias`. Default to
    /// [`BatchNorm::DEFAULT_AFFINE`].
    #[builder(optional, default = BatchNorm::DEFAULT_AFFINE)]
    pub affine: bool,

    /// If `true`, track the running mean and variance. Default to
    /// [`BatchNorm::DEFAULT_TRACK_RUNNING_STATS`].
    #[builder(optional, default = BatchNorm::DEFAULT_TRACK_RUNNING_STATS)]
    pub track_running_stats: bool,
}

fn build_batch_norm(builder: BatchNormBuilder) -> Result<BatchNorm, Exception> {
    let eps = builder.eps;
    let momentum = builder.momentum;
    let affine = builder.affine;
    let track_running_stats = builder.track_running_stats;

    let (weight, bias) = if affine {
        (
            Some(ones::<f32>(&[builder.feature_count])?),
            Some(zeros::<f32>(&[builder.feature_count])?),
        )
    } else {
        (None, None)
    };

    let (running_mean, running_var) = if track_running_stats {
        (
            Some(zeros::<f32>(&[builder.feature_count])?),
            Some(ones::<f32>(&[builder.feature_count])?),
        )
    } else {
        (None, None)
    };

    Ok(BatchNorm {
        feature_count: builder.feature_count,
        eps: array!(eps),
        momentum: array!(momentum),
        weight: Param::new(weight),
        bias: Param::new(bias),
        running_mean: Param::new(running_mean),
        running_var: Param::new(running_var),
        training: BatchNorm::DEFAULT_TRAINING,
    })
}

/// Applies batch normalization [1] on the inputs.
///
/// ### References
///
/// 1. [https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)
#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
pub struct BatchNorm {
    /// Number of features in the input
    pub feature_count: i32,

    /// Value added to the denominator for numerical stability.
    pub eps: Array,

    /// Momentum for updating the running mean and variance.
    pub momentum: Array,

    /// An optional trainable weight
    #[param]
    pub weight: Param<Option<Array>>,

    /// An optional trainable bias
    #[param]
    pub bias: Param<Option<Array>>,

    /// Tracked running mean
    #[param]
    pub running_mean: Param<Option<Array>>,

    /// Tracked running variance
    #[param]
    pub running_var: Param<Option<Array>>,

    /// If `true`, the module is in training mode.
    pub training: bool,
}

impl BatchNorm {
    /// Default value for `eps`.
    pub const DEFAULT_EPS: f32 = 1e-5;

    /// Default value for `momentum`.
    pub const DEFAULT_MOMENTUM: f32 = 0.1;

    /// Enable trainable `weight` and `bias` by default.
    pub const DEFAULT_AFFINE: bool = true;

    /// Enable tracking of running mean and variance by default.
    pub const DEFAULT_TRACK_RUNNING_STATS: bool = true;

    /// Enable training mode by default.
    pub const DEFAULT_TRAINING: bool = true;

    fn stats(x: &Array) -> Result<(Array, Array), Exception> {
        let reduction_axes = (0..x.ndim() as i32 - 1).collect::<Vec<_>>();

        let mean = x.mean_axes(&reduction_axes, None)?;
        let variance = x.var_axes(&reduction_axes, None, None)?;

        Ok((mean, variance))
    }
}

impl Module<&Array> for BatchNorm {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let ndim = x.ndim();
        if !(2..=4).contains(&ndim) {
            return Err(Exception::custom(
                "Input tensor must be at least 2 dimensions and at most 4 dimensions",
            ));
        }

        let (mean, variance) = Self::stats(x)?;
        let mut mean = Cow::Owned(mean);
        let mut variance = Cow::Owned(variance);

        if let (Some(running_mean), Some(running_var)) =
            (self.running_mean.as_mut(), self.running_var.as_mut())
        {
            if self.training {
                let mu = &self.momentum;
                // SAFETY: momentum is a single element array
                let one_minus_mu = array!(1.0) - mu;

                *running_mean = one_minus_mu
                    .multiply(&running_mean)?
                    .add(mu.multiply(&mean)?)?;
                *running_var = one_minus_mu
                    .multiply(&running_var)?
                    .add(mu.multiply(&variance)?)?;
            } else {
                mean = Cow::Borrowed(&*running_mean);
                variance = Cow::Borrowed(&*running_var);
            }
        }

        let x = x
            .subtract(&mean)?
            .multiply(rsqrt(&variance.add(&self.eps)?)?)?;

        if let (Some(weight), Some(bias)) = (self.weight.as_ref(), self.bias.as_ref()) {
            weight.multiply(&x)?.add(bias)
        } else {
            Ok(x)
        }
    }

    fn training_mode(&mut self, mode: bool) {
        self.training = mode;
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ops::indexing::{Ellipsis, IndexOp},
        Dtype,
    };
    use float_eq::assert_float_eq;

    use super::*;

    #[test]
    fn test_instance_norm() {
        crate::random::seed(435).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item::<f32>(),
            0.500_064_6,
            abs <= 0.010_001_292
        );
        assert_float_eq!(
            a.sum(None).unwrap().item::<f32>(),
            128.016_54,
            abs <= 2.560_330_9
        );

        let result = InstanceNorm::new(8)
            .unwrap()
            .forward(&a)
            .unwrap()
            .index((0, 0));
        assert_eq!(result.shape(), &[16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item::<f32>(),
            0.106_454_11,
            abs <= 0.002_129_082_3
        );
        assert_float_eq!(
            result.sum(None).unwrap().item::<f32>(),
            1.703_265_8,
            abs <= 0.034_065_317
        );
    }

    #[test]
    fn test_layer_norm() {
        crate::random::seed(635).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item::<f32>(),
            0.492_690_32,
            abs <= 0.009_853_806
        );
        assert_float_eq!(
            a.sum(None).unwrap().item::<f32>(),
            126.128_72,
            abs <= 2.522_574_4
        );

        let result = LayerNorm::new(16)
            .unwrap()
            .forward(&a)
            .unwrap()
            .index((Ellipsis, 0));
        assert_eq!(result.shape(), &[2, 8]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item::<f32>(),
            0.290_990_38,
            abs <= 0.005_819_807_8
        );
        assert_float_eq!(
            result.sum(None).unwrap().item::<f32>(),
            4.655_846,
            abs <= 0.093_116_924
        );
    }

    #[test]
    fn test_rms_norm() {
        crate::random::seed(103).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item::<f32>(),
            0.505_476_36,
            abs <= 0.010_109_527
        );
        assert_float_eq!(
            a.sum(None).unwrap().item::<f32>(),
            129.401_95,
            abs <= 2.588_039
        );

        let result = RmsNorm::new(16).unwrap().forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item::<f32>(),
            0.872_938_75,
            abs <= 0.017_458_774
        );
        assert_float_eq!(
            result.sum(None).unwrap().item::<f32>(),
            223.472_32,
            abs <= 4.469_446
        );
    }

    #[test]
    fn test_group_norm() {
        crate::random::seed(855).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item::<f32>(),
            0.486_665_87,
            abs <= 0.009_733_317
        );
        assert_float_eq!(
            a.sum(None).unwrap().item::<f32>(),
            124.586_464,
            abs <= 2.491_729_3
        );

        let result = GroupNorm::new(4, 16)
            .unwrap()
            .forward(&a)
            .unwrap()
            .index((0, 0));
        assert_eq!(result.shape(), &[16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item::<f32>(),
            -0.054_606_52,
            abs <= 0.001_092_130_4
        );
        assert_float_eq!(
            result.sum(None).unwrap().item::<f32>(),
            -0.873_704_3,
            abs <= 0.017_474_087
        );
    }

    #[test]
    fn test_batch_norm() {
        crate::random::seed(266).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item::<f32>(),
            0.505_814_7,
            abs <= 0.010_116_293
        );
        assert_float_eq!(
            a.sum(None).unwrap().item::<f32>(),
            129.488_56,
            abs <= 2.589_771
        );

        let result = BatchNorm::new(16)
            .unwrap()
            .forward(&a)
            .unwrap()
            .index((0, 0));
        assert_eq!(result.shape(), &[16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item::<f32>(),
            0.439_785_24,
            abs <= 0.008_795_705
        );
        assert_float_eq!(
            result.sum(None).unwrap().item::<f32>(),
            7.036_564,
            abs <= 0.140_731_28
        );
    }
}
