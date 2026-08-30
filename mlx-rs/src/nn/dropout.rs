use crate::module::Module;
use crate::Array;
use crate::{array, error::Exception, ops::multiply, random::bernoulli};
use mlx_internal_macros::{Buildable, Builder};
use mlx_macros::ModuleParameters;

use crate::error::DropoutBuildError;

/// Builder for [`Dropout`].
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_dropout,
    default_infallible,
    err = DropoutBuildError,
)]
pub struct DropoutBuilder {
    /// The probability of zeroing an element.
    #[builder(optional, default = Dropout::DEFAULT_P)]
    p: f32,
}

fn build_dropout(builder: DropoutBuilder) -> Result<Dropout, DropoutBuildError> {
    let p = builder.p;

    if !(0.0..1.0).contains(&p) {
        return Err(DropoutBuildError::InvalidProbability);
    }

    Ok(Dropout {
        one_minus_p: 1.0 - p,
        training: Dropout::DEFAULT_TRAINING,
    })
}

/// Randomly zero a portion of the elements during training.
///
/// The remaining elements are multiplied with `1 / (1-p)` where
/// `p` is the probability of zeroing an element. This is done so the
/// expected value of a given element will remain the same.
#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
pub struct Dropout {
    /// `1-p`, where `p` is the probability of zeroing an element. `p` is default to
    /// [`Dropout::DEFAULT_P`] if not specified.
    pub one_minus_p: f32,

    /// Whether the layer is in training mode. Default to [`Dropout::DEFAULT_TRAINING`] if not
    /// specified.
    pub training: bool,
}

impl Dropout {
    /// Default value for the probability of zeroing an element.
    pub const DEFAULT_P: f32 = 0.5;

    /// Default value for the training mode.
    pub const DEFAULT_TRAINING: bool = true;
}

impl Module<&Array> for Dropout {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        if self.one_minus_p == 1.0 || !self.training {
            return Ok(x.clone());
        }

        let p1 = array!(self.one_minus_p);
        let mask = bernoulli(&p1, x.shape(), None)?;
        multiply(multiply(array!(1.0 / self.one_minus_p), mask)?, x)
    }

    fn training_mode(&mut self, mode: bool) {
        self.training = mode;
    }
}

/// Builder for [`Dropout2d`].
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_dropout2d,
    default_infallible,
    err = DropoutBuildError,
)]
pub struct Dropout2dBuilder {
    /// The probability of zeroing a channel.
    #[builder(optional, default = Dropout2d::DEFAULT_P)]
    p: f32,
}

fn build_dropout2d(builder: Dropout2dBuilder) -> Result<Dropout2d, DropoutBuildError> {
    let p = builder.p;

    if !(0.0..1.0).contains(&p) {
        return Err(DropoutBuildError::InvalidProbability);
    }

    Ok(Dropout2d {
        one_minus_p: 1.0 - p,
        training: Dropout2d::DEFAULT_TRAINING,
    })
}

/// Apply 2D channel-wise dropout during training.
///
/// Randomly zero out entire channels independently with probability `p`.
/// This layer expects the channels to be last, i.e. the input shape should be
/// `NWHC` or `WHC` where:`N` is the batch dimension,`H` is the input
/// image height,`W` is the input image width, and`C` is the number of
/// input channels
///
/// The remaining channels are scaled by `1 / (1-p)` to
/// maintain the expected value of each element. Unlike traditional dropout,
/// which zeros individual entries, this layer zeros entire channels. This is
/// beneficial for early convolution layers where adjacent pixels are
/// correlated. In such case, traditional dropout may not effectively
/// regularize activations. For more details, see [1].
///
/// [1]: Thompson, J., Goroshin, R., Jain, A., LeCun, Y. and Bregler C., 2015.
/// Efficient Object Localization Using Convolutional Networks. CVPR 2015.
#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
pub struct Dropout2d {
    /// `1-p`, where `p` is the probability of zeroing a channel. `p` is default to
    /// [`Dropout2d::DEFAULT_P`] if not specified.
    pub one_minus_p: f32,

    /// Whether the layer is in training mode. Default to [`Dropout2d::DEFAULT_TRAINING`] if not
    /// specified. Default to [`Dropout2d::DEFAULT_TRAINING`] if not specified.
    pub training: bool,
}

impl Dropout2d {
    /// Default value for the probability of zeroing a channel.
    pub const DEFAULT_P: f32 = 0.5;

    /// Default value for the training mode.
    pub const DEFAULT_TRAINING: bool = true;
}

impl Module<&Array> for Dropout2d {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let ndim = x.ndim();

        if ndim != 3 && ndim != 4 {
            return Err(Exception::custom("Expecting 3D or 4D input"));
        }

        if self.one_minus_p == 1.0 || !self.training {
            return Ok(x.clone());
        }

        // Dropout is applied on the whole channel
        // 3D input: (1, 1, C)
        // 4D input: (B, 1, 1, C)

        let mut mask_shape = x.shape().to_vec();
        let len = mask_shape.len();
        mask_shape[len - 2] = 1;
        mask_shape[len - 3] = 1;

        let p1 = array!(self.one_minus_p);
        let mask = bernoulli(&p1, &mask_shape, None)?;

        multiply(multiply(array!(1.0 / self.one_minus_p), mask)?, x)
    }

    fn training_mode(&mut self, mode: bool) {
        self.training = mode;
    }
}

/// Builder for [`Dropout3d`].
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_dropout3d,
    default_infallible,
    err = DropoutBuildError,
)]
pub struct Dropout3dBuilder {
    /// The probability of zeroing a channel.
    #[builder(optional, default = Dropout3d::DEFAULT_P)]
    p: f32,
}

fn build_dropout3d(builder: Dropout3dBuilder) -> Result<Dropout3d, DropoutBuildError> {
    let p = builder.p;

    if !(0.0..1.0).contains(&p) {
        return Err(DropoutBuildError::InvalidProbability);
    }

    Ok(Dropout3d {
        one_minus_p: 1.0 - p,
        training: Dropout3d::DEFAULT_TRAINING,
    })
}

/// Apply 3D channel-wise dropout during training.
///
/// Randomly zero out entire channels independently with probability `p`.
/// This layer expects the channels to be last, i.e., the input shape should be
/// `NDHWC` or `DHWC` where: `N` is the batch dimension, `D` is the depth,
/// `H` is the input image height, `W` is the input image width, and `C` is
/// the number of input channels.
///
/// The remaining channels are scaled by `1 / (1-p)` to
/// maintain the expected value of each element. Unlike traditional dropout,
/// which zeros individual entries, this layer zeros entire channels. This is
/// often beneficial for convolutional layers processing 3D data, like in
/// medical imaging or video processing.
#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
pub struct Dropout3d {
    /// `1-p`, where `p` is the probability of zeroing a channel. `p` is default to
    /// [`Dropout3d::DEFAULT_P`] if not specified.
    pub one_minus_p: f32,

    /// Whether the layer is in training mode. Default to [`Dropout3d::DEFAULT_TRAINING`] if not
    /// specified.
    pub training: bool,
}

impl Dropout3d {
    /// Default value for the probability of zeroing a channel.
    pub const DEFAULT_P: f32 = 0.5;

    /// Default value for the training mode.
    pub const DEFAULT_TRAINING: bool = true;
}

impl Module<&Array> for Dropout3d {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let ndim = x.ndim();

        if ndim != 4 && ndim != 5 {
            return Err(Exception::custom("Expecting 4D or 5D input"));
        }

        if self.one_minus_p == 1.0 || !self.training {
            return Ok(x.clone());
        }

        // Dropout is applied on the whole channel
        // 4D input: (1, 1, 1, C)
        // 5D input: (B, 1, 1, 1, C)

        let mut mask_shape = x.shape().to_vec();
        let len = mask_shape.len();
        mask_shape[len - 2] = 1;
        mask_shape[len - 3] = 1;
        mask_shape[len - 4] = 1;

        let p1 = array!(self.one_minus_p);
        let mask = bernoulli(&p1, &mask_shape, None)?;

        multiply(multiply(array!(1.0 / self.one_minus_p), mask)?, x)
    }

    fn training_mode(&mut self, mode: bool) {
        self.training = mode;
    }
}

// The following tests were ported from the swift binding:
// mlx-swift/Tests/MLXTests/IntegrationTests.swift
#[cfg(test)]
mod tests {
    use crate::random::uniform;
    use float_eq::assert_float_eq;

    use super::*;

    #[test]
    fn test_dropout() {
        crate::random::seed(959).unwrap();
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item_exact::<f32>(),
            0.511_429_2,
            abs <= 0.010_228_584
        );
        assert_float_eq!(
            a.sum(None).unwrap().item_exact::<f32>(),
            130.925_87,
            abs <= 2.618_517_4
        );
        let result = Dropout::new().forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            0.477_913_62,
            abs <= 0.009_558_273
        );
        assert_float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            122.345_89,
            abs <= 2.446_917_8
        );
    }

    #[test]
    fn test_dropout2d() {
        crate::random::seed(695).unwrap();
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item_exact::<f32>(),
            0.457_839_9,
            abs <= 0.009_156_798
        );
        assert_float_eq!(
            a.sum(None).unwrap().item_exact::<f32>(),
            117.207_016,
            abs <= 2.344_140_3
        );
        let result = Dropout2d::new().forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            0.368_284_34,
            abs <= 0.007_365_687
        );
        assert_float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            94.280_79,
            abs <= 1.885_615_8
        );
    }

    #[test]
    fn test_dropout3d() {
        crate::random::seed(23).unwrap();
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 8, 4], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 8, 4]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item_exact::<f32>(),
            0.500_606_2,
            abs <= 0.010_012_124
        );
        assert_float_eq!(
            a.sum(None).unwrap().item_exact::<f32>(),
            256.310_36,
            abs <= 5.126_207_4
        );
        let result = Dropout3d::new().forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 8, 4]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            0.237_284_15,
            abs <= 0.004_745_683
        );
        assert_float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            121.489_49,
            abs <= 2.429_789_8
        );
    }
}
