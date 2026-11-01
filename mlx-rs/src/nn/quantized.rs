use std::iter::once;

use crate::{
    array,
    error::Exception,
    module::{Module, ModuleParameters, Param},
    ops::indexing::IndexOp,
    ops::{self, dequantize, quantized_matmul, zeros},
    quantization::Quantizable,
    random::uniform,
    Array,
};
use mlx_internal_macros::{Buildable, Builder};
use mlx_macros::ModuleParameters;

use crate::nn::{Embedding, Linear};

/// Quantize a module.
///
/// # Params
///
/// - `module`: The module to quantize.
/// - `group_size`: The group size to use for the quantized weight. Default to [`Quantizable::DEFAULT_GROUP_SIZE`]
/// - `bits`: The bit width to use for the quantized weight. Default to [`Quantizable::DEFAULT_BITS`]
pub fn quantize<M>(
    module: M,
    group_size: impl Into<Option<i32>>,
    bits: impl Into<Option<i32>>,
) -> Result<M::Quantized, M::QuantizationError>
where
    M: Quantizable,
{
    let group_size = group_size.into().unwrap_or(M::DEFAULT_GROUP_SIZE);
    let bits = bits.into().unwrap_or(M::DEFAULT_BITS);
    module.try_into_quantized(group_size, bits)
}

/// Builder for [`QuantizedEmbedding`]
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_quantized_embedding,
    err = Exception,
)]
pub struct QuantizedEmbeddingBuilder {
    /// How many possible discrete tokens can we embed. Usually called the vocabulary size.
    pub embedding_count: i32,

    /// The dimensionality of the embeddings.
    pub dimensions: i32,

    /// Quantization group size. Default to [`QuantizedEmbedding::DEFAULT_GROUP_SIZE`]
    #[builder(optional, default = QuantizedEmbedding::DEFAULT_GROUP_SIZE)]
    pub group_size: i32,

    /// Bits per parameter. Default to [`QuantizedEmbedding::DEFAULT_BITS`]
    #[builder(optional, default = QuantizedEmbedding::DEFAULT_BITS)]
    pub bits: i32,
}

/// The same as ``Embedding`` but with a quantized weight matrix.
#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
pub struct QuantizedEmbedding {
    /// Quantization group size. Default to [`QuantizedEmbedding::DEFAULT_GROUP_SIZE`]
    pub group_size: i32,

    /// Bits per parameter. Default to [`QuantizedEmbedding::DEFAULT_BITS`]
    pub bits: i32,

    /// Scales
    #[param]
    pub scales: Param<Array>,

    /// Biases
    #[param]
    pub biases: Param<Array>,

    /// Inner embedding
    #[param]
    pub inner: Embedding,
}

impl QuantizedEmbeddingBuilder {
    /// Convenience method to build a new [`QuantizedEmbedding`] with an existing [`Embedding`]
    pub fn build_with_embedding(
        self,
        embedding: Embedding,
    ) -> Result<QuantizedEmbedding, Exception> {
        let weight = embedding.weight.value;
        self.build_with_weight(weight)
    }

    /// Convenience method to build a new [`QuantizedEmbedding`] with an existing weight matrix
    pub fn build_with_weight(
        self,
        weight: impl AsRef<Array>,
    ) -> Result<QuantizedEmbedding, Exception> {
        let group_size = self.group_size;
        let bits = self.bits;
        build_quantized_embedding_inner(weight.as_ref(), group_size, bits)
    }
}

fn build_quantized_embedding_inner(
    weight: &Array,
    group_size: i32,
    bits: i32,
) -> Result<QuantizedEmbedding, Exception> {
    let (quantized_weight, scales, biases) = ops::quantize(
        weight,
        ops::QuantizeOptions {
            group_size: Some(group_size),
            bits: Some(bits),
            ..Default::default()
        },
    )?;
    let biases = biases.expect("affine quantization returns biases");

    let inner = Embedding {
        weight: Param::new(quantized_weight),
    };

    let mut qe = QuantizedEmbedding {
        group_size,
        bits,
        scales: Param::new(scales),
        biases: Param::new(biases),
        inner,
    };

    // Freeze all parameters
    qe.freeze_parameters(true);

    Ok(qe)
}

fn build_quantized_embedding(
    builder: QuantizedEmbeddingBuilder,
) -> Result<QuantizedEmbedding, Exception> {
    let embedding_count = builder.embedding_count;
    let dims = builder.dimensions;

    let scale = array!(f32::sqrt(1.0 / (dims as f32)));
    // SAFETY: This is safe because the array scale is a single element array
    let weight = crate::random::normal::<f32>(&[embedding_count, dims], None, None, None)? * &scale;

    builder.build_with_weight(weight)
}

impl QuantizedEmbedding {
    /// Default group size
    pub const DEFAULT_GROUP_SIZE: i32 = 64;

    /// Default bits
    pub const DEFAULT_BITS: i32 = 4;

    /// Convert an embedding layer to a quantized embedding layer.
    ///
    /// # Params
    ///
    /// - `embedding`: The embedding layer to convert.
    /// - `group_size`: The group size to use for the quantized weight. Default to [`QuantizedEmbedding::DEFAULT_GROUP_SIZE`]
    /// - `bits`: The bit width to use for the quantized weight. Default to [`QuantizedEmbedding::DEFAULT_BITS`]
    pub fn try_from_embedding(
        embedding: Embedding,
        group_size: impl Into<Option<i32>>,
        bits: impl Into<Option<i32>>,
    ) -> Result<Self, Exception> {
        let group_size = group_size.into().unwrap_or(Self::DEFAULT_GROUP_SIZE);
        let bits = bits.into().unwrap_or(Self::DEFAULT_BITS);
        build_quantized_embedding_inner(&embedding.weight.value, group_size, bits)
    }

    /// Call the embedding layer as a linear layer.
    ///
    /// Use this for example when input embedding and output projection
    /// weights are tied.
    pub fn as_linear(&self, x: impl AsRef<Array>) -> Result<Array, Exception> {
        quantized_matmul(
            x.as_ref(),
            &self.inner.weight,
            &self.scales,
            self.biases.as_ref(),
            ops::QuantizedMatmulOptions {
                transpose: true,
                group_size: Some(self.group_size),
                bits: Some(self.bits),
                ..Default::default()
            },
        )
        .map_err(Into::into)
    }
}

impl TryFrom<Embedding> for QuantizedEmbedding {
    type Error = Exception;

    fn try_from(embedding: Embedding) -> Result<Self, Self::Error> {
        Self::try_from_embedding(embedding, None, None)
    }
}

impl Module<&Array> for QuantizedEmbedding {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let s = x.shape();
        let x = x.flatten(None, None)?;
        let w = self.inner.weight.index(&x);
        let scales = self.scales.index(&x);
        let biases = self.biases.index(&x);

        let out = dequantize(
            &w,
            &scales,
            &biases,
            ops::DequantizeOptions {
                group_size: Some(self.group_size),
                bits: Some(self.bits),
                ..Default::default()
            },
        )?;

        let ret_shape = s.iter().copied().chain(once(-1)).collect::<Vec<_>>();
        out.reshape(&ret_shape)
    }

    fn training_mode(&mut self, mode: bool) {
        self.inner.training_mode(mode);
    }
}

/// Builder for [`QuantizedLinear`]
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_quantized_linear,
    err = Exception,
)]
pub struct QuantizedLinearBuilder {
    /// The dimensionality of the input features.
    pub input_dims: i32,

    /// The dimensionality of the output features.
    pub output_dims: i32,

    /// Quantization group size. Default to [`QuantizedLinear::DEFAULT_GROUP_SIZE`]
    #[builder(optional, default = QuantizedLinear::DEFAULT_GROUP_SIZE)]
    pub group_size: i32,

    /// Bits per parameter. Default to [`QuantizedLinear::DEFAULT_BITS`]
    #[builder(optional, default = QuantizedLinear::DEFAULT_BITS)]
    pub bits: i32,

    /// Whether the linear layer has a bias. Default to [`Linear::DEFAULT_BIAS`]
    #[builder(optional, default = Linear::DEFAULT_BIAS)]
    pub bias: bool,
}

impl QuantizedLinearBuilder {
    /// Convenience method to build a new [`QuantizedLinear`] with an existing [`Linear`]
    pub fn build_with_linear(self, other: Linear) -> Result<QuantizedLinear, Exception> {
        self.build_with_weight_and_bias(other.weight.value, other.bias.value)
    }

    fn build_with_weight_and_bias(
        self,
        weight: Array,
        bias: Option<Array>,
    ) -> Result<QuantizedLinear, Exception> {
        build_quantized_linear_inner(weight, bias, self.group_size, self.bits)
    }
}

fn build_quantized_linear_inner(
    weight: Array,
    bias: Option<Array>,
    group_size: i32,
    bits: i32,
) -> Result<QuantizedLinear, Exception> {
    let (quantized_weight, scales, biases) = ops::quantize(
        &weight,
        ops::QuantizeOptions {
            group_size: Some(group_size),
            bits: Some(bits),
            ..Default::default()
        },
    )?;
    let biases = biases.expect("affine quantization returns biases");

    let inner = Linear {
        weight: Param::new(quantized_weight),
        bias: Param::new(bias),
    };

    let mut ql = QuantizedLinear {
        group_size,
        bits,
        scales: Param::new(scales),
        biases: Param::new(biases),
        inner,
    };

    // Freeze all parameters
    ql.freeze_parameters(true);

    Ok(ql)
}

/// Builds a new [`QuantizedLinear`]
pub fn build_quantized_linear(
    builder: QuantizedLinearBuilder,
) -> Result<QuantizedLinear, Exception> {
    let input_dims = builder.input_dims;
    let output_dims = builder.output_dims;
    let scale = f32::sqrt(1.0 / (input_dims as f32));
    let weight = uniform::<_, f32>(-scale, scale, &[output_dims, input_dims], None)?;

    let bias = if builder.bias {
        Some(zeros::<f32>(&[output_dims])?)
    } else {
        None
    };

    builder.build_with_weight_and_bias(weight, bias)
}

/// Applies an affine transformation to the input using a quantized weight matrix.
///
/// It is the quantized equivalent of [`Linear`].  For now its
/// parameters are frozen and will not be included in any gradient computation
/// but this will probably change in the future.
///
/// QuantizedLinear also provides several useful static to convert linear
/// layers to QuantizedLinear layers.
#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
pub struct QuantizedLinear {
    /// Quantization group size. Default to [`QuantizedLinear::DEFAULT_GROUP_SIZE`]
    pub group_size: i32,

    /// Bits per parameter. Default to [`QuantizedLinear::DEFAULT_BITS`]
    pub bits: i32,

    /// Scales
    #[param]
    pub scales: Param<Array>,

    /// Biases
    #[param]
    pub biases: Param<Array>,

    /// Inner linear layer
    #[param]
    pub inner: Linear,
}

impl QuantizedLinear {
    /// Default group size
    pub const DEFAULT_GROUP_SIZE: i32 = 64;

    /// Default bits
    pub const DEFAULT_BITS: i32 = 4;

    /// Convert a linear layer to a quantized linear layer.
    ///
    /// # Params
    ///
    /// - `linear`: The linear layer to convert.
    /// - `group_size`: The group size to use for the quantized weight. Default to [`QuantizedLinear::DEFAULT_GROUP_SIZE`]
    /// - `bits`: The bit width to use for the quantized weight. Default to [`QuantizedLinear::DEFAULT_BITS`]
    pub fn try_from_linear(
        linear: Linear,
        group_size: impl Into<Option<i32>>,
        bits: impl Into<Option<i32>>,
    ) -> Result<Self, Exception> {
        let group_size = group_size.into().unwrap_or(Self::DEFAULT_GROUP_SIZE);
        let bits = bits.into().unwrap_or(Self::DEFAULT_BITS);
        build_quantized_linear_inner(linear.weight.value, linear.bias.value, group_size, bits)
    }
}

impl TryFrom<Linear> for QuantizedLinear {
    type Error = Exception;

    fn try_from(linear: Linear) -> Result<Self, Self::Error> {
        Self::try_from_linear(linear, None, None)
    }
}

impl Module<&Array> for QuantizedLinear {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let mut x = quantized_matmul(
            x,
            &self.inner.weight,
            &self.scales,
            self.biases.as_ref(),
            ops::QuantizedMatmulOptions {
                transpose: true,
                group_size: Some(self.group_size),
                bits: Some(self.bits),
                ..Default::default()
            },
        )?;
        if let Some(bias) = &self.inner.bias.value {
            x = x.add(bias)?;
        }
        Ok(x)
    }

    fn training_mode(&mut self, mode: bool) {
        self.inner.training_mode(mode);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_with_borrowed_weight() {
        let weight = crate::ops::arange::<_, f32>(None, 256, None)
            .unwrap()
            .reshape(&[4, 64])
            .unwrap();
        let borrowed = QuantizedEmbeddingBuilder::new(4, 64)
            .build_with_weight(&weight)
            .unwrap();
        let owned = QuantizedEmbeddingBuilder::new(4, 64)
            .build_with_weight(weight.clone())
            .unwrap();
        assert_eq!(
            borrowed.inner.weight.to_vec_exact::<u32>().unwrap(),
            owned.inner.weight.to_vec_exact::<u32>().unwrap()
        );
        assert_eq!(weight.shape(), &[4, 64]);
    }
}
