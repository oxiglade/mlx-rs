use std::iter::once;

use crate::{error::Exception, quantization::Quantizable, Array};
use mlx_internal_macros::{Buildable, Builder};

use crate::{
    macros::ModuleParameters,
    module::{Module, Param},
};

use super::QuantizedLinear;

/// Builder for [`Linear`] module
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_linear,
    err = Exception,
)]
pub struct LinearBuilder {
    /// The number of input dimensions.
    pub input_dims: i32,

    /// The number of output dimensions.
    pub output_dims: i32,

    /// Whether to include bias in the linear layer. Default to [`Linear::DEFAULT_BIAS`].
    #[builder(optional, default = Linear::DEFAULT_BIAS)]
    pub bias: bool,
}

/// Builds a new [`Linear`] layer.
fn build_linear(builder: LinearBuilder) -> Result<Linear, Exception> {
    let input_dims = builder.input_dims;
    let output_dims = builder.output_dims;
    let with_bias = builder.bias;

    let scale = f32::sqrt(1.0 / (input_dims as f32));
    let weight = crate::random::uniform::<_, f32>(-scale, scale, &[output_dims, input_dims], None)?;

    let bias = if with_bias {
        Some(crate::random::uniform::<_, f32>(
            -scale,
            scale,
            &[output_dims],
            None,
        )?)
    } else {
        None
    };

    Ok(Linear {
        weight: Param::new(weight),
        bias: Param::new(bias),
    })
}

/// Applies an affine transformation to the input.
#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
pub struct Linear {
    /// The weight of the linear layer.
    #[param]
    pub weight: Param<Array>,

    /// The bias of the linear layer.
    #[param]
    pub bias: Param<Option<Array>>,
}

impl Linear {
    /// Default value for `with_bias`
    pub const DEFAULT_BIAS: bool = true;

    /// Returns the shape of the linear layer.
    pub fn shape(&self) -> (i32, i32) {
        let weight_shape = self.weight.as_ref().shape();
        (weight_shape[0], weight_shape[1])
    }
}

impl Module<&Array> for Linear {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        match &self.bias.value {
            Some(bias) => crate::ops::addmm(bias, x, self.weight.value.t(), None, None),
            None => crate::ops::matmul(x, self.weight.value.t()),
        }
    }

    fn training_mode(&mut self, _: bool) {}
}

impl Quantizable for Linear {
    type Quantized = QuantizedLinear;
    type QuantizationError = Exception;

    fn try_into_quantized(
        self,
        group_size: i32,
        bits: i32,
    ) -> Result<Self::Quantized, Self::QuantizationError> {
        QuantizedLinear::try_from_linear(self, group_size, bits)
    }
}

/// Builder for [`Bilinear`] module
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_bilinear,
    err = Exception,
)]
pub struct BilinearBuilder {
    /// The number of input dimensions for the first input.
    pub input_dims_1: i32,

    /// The number of input dimensions for the second input.
    pub input_dims_2: i32,

    /// The number of output dimensions.
    pub output_dims: i32,

    /// Whether to include bias in the bilinear layer. Default to [Bilinear::DEFAULT_BIAS].
    #[builder(optional, default = Bilinear::DEFAULT_BIAS)]
    pub bias: bool,
}

fn build_bilinear(builder: BilinearBuilder) -> Result<Bilinear, Exception> {
    let input_dims_1 = builder.input_dims_1;
    let input_dims_2 = builder.input_dims_2;
    let output_dims = builder.output_dims;
    let with_bias = builder.bias;

    let scale = f32::sqrt(1.0 / (input_dims_1 as f32));
    let weights = crate::random::uniform::<_, f32>(
        -scale,
        scale,
        &[output_dims, input_dims_2, input_dims_1],
        None,
    )?;

    let bias = if with_bias {
        Some(crate::random::uniform::<_, f32>(
            -scale,
            scale,
            &[output_dims],
            None,
        )?)
    } else {
        None
    };

    Ok(Bilinear {
        weights: Param::new(weights),
        bias: Param::new(bias),
    })
}

/// Applies a bilinear transformation to the inputs.
#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
pub struct Bilinear {
    /// The weight of the bilinear layer.
    #[param]
    pub weights: Param<Array>,

    /// The bias of the bilinear layer.
    #[param]
    pub bias: Param<Option<Array>>,
}

impl Bilinear {
    /// Default value for `with_bias`
    pub const DEFAULT_BIAS: bool = true;
}

impl Module<&Array> for Bilinear {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let shape = self.weights.shape();
        let (out, in2, in1) = (shape[0], shape[1], shape[2]);
        let x_shape = &x.shape()[..x.shape().len() - 1];
        let x1 = x.reshape(&[-1, in1])?;
        let x2 = x.reshape(&[-1, 1, in2])?;

        // perform the bilinear transform
        let w = self.weights.reshape(&[out * in2, in1])?;
        let mut y = crate::ops::matmul(&x1, w.t())?;
        y = y.reshape(&[-1, out, in2])?.swap_axes(-2, -1)?;
        y = crate::ops::matmul(&x2, &y)?;
        y = y.squeeze_axes(&[1])?;

        // reset the shape
        let new_shape = x_shape.iter().cloned().chain(once(out)).collect::<Vec<_>>();
        y = y.reshape(&new_shape)?;

        if let Some(bias) = &self.bias.value {
            y = crate::ops::add(&y, bias)?;
        }

        Ok(y)
    }

    fn training_mode(&mut self, _: bool) {}
}

// The following tests are ported from the swift binding:
// mlx-swift/Tests/MLXTests/IntegrationTests.swift
#[cfg(test)]
mod tests {
    use crate::{random::uniform, Dtype};
    use float_eq::assert_float_eq;

    use super::*;

    #[test]
    fn test_linear() {
        crate::random::seed(744).unwrap();
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item_exact::<f32>(),
            0.508_688_57,
            abs <= 0.010_173_771_5
        );
        assert_float_eq!(
            a.sum(None).unwrap().item_exact::<f32>(),
            130.224_27,
            abs <= 2.604_485_5
        );
        let result = Linear::new(16, 5).unwrap().forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 5]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            0.104_193_09,
            abs <= 0.002_083_861_7
        );
        assert_float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            8.335_447,
            abs <= 0.166_708_95
        );
    }
}
