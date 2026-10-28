//! Embedding layer.

use crate::error::Exception;
use crate::module::Module;
use crate::module::Param;
use crate::ops::indexing::IndexOp;
use crate::quantization::Quantizable;
use crate::Array;
use mlx_macros::ModuleParameters;

use super::QuantizedEmbedding;

/// Implements a simple lookup table that maps each input integer to a high-dimensional vector.
///
/// Typically used to embed discrete tokens for processing by neural networks.
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = crate)]
pub struct Embedding {
    /// The weight of the
    #[param]
    pub weight: Param<Array>,
}

impl Embedding {
    /// Creates a new [`Embedding`] layer.
    ///
    /// # Params
    ///
    /// - `embedding_count`: How many possible discrete tokens can we embed.  Usually called the vocabulary size.
    /// - `dimensions`: The dimensionality of the embeddings.
    pub fn new(embedding_count: i32, dimensions: i32) -> Result<Self, Exception> {
        let scale = f32::sqrt(1.0 / (dimensions as f32));
        let weight =
            crate::random::normal::<f32>(&[embedding_count, dimensions], None, None, None)? * scale;

        Ok(Self {
            weight: Param::new(weight),
        })
    }

    /// Call the embedding layer as a linear layer.
    ///
    /// Use this for example when input embedding and output projection
    /// weights are tied.
    pub fn as_linear(&self, x: &Array) -> Result<Array, Exception> {
        crate::ops::matmul(x, self.weight.value.t())
    }
}

impl Quantizable for Embedding {
    type Quantized = QuantizedEmbedding;

    type QuantizationError = Exception;

    fn try_into_quantized(
        self,
        group_size: i32,
        bits: i32,
    ) -> Result<Self::Quantized, Self::QuantizationError> {
        QuantizedEmbedding::try_from_embedding(self, group_size, bits)
    }
}

impl Module<&Array> for Embedding {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        Ok(self.weight.index(x))
    }

    fn training_mode(&mut self, _mode: bool) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_eq::float_eq;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_embedding() {
        crate::random::seed(557).unwrap();
        let a = crate::random::randint::<_, i32>(0, 10, &[2, 8, 8, 4], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 8, 4]);
        assert_eq!(a.dtype(), crate::Dtype::Int32);
        float_eq!(
            a.mean(None).unwrap().item_exact::<f32>(),
            4.605_468_8,
            abs <= 0.092_109_375
        );
        assert_eq!(a.sum(None).unwrap().item_exact::<i32>(), 2358);

        let result = Embedding::new(10, 8).unwrap().forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 8, 4, 8]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            -0.001_197_346_3,
            abs <= 2.394_692_5e-5
        );
        float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            -4.904_330_3,
            abs <= 0.098_086_6
        );
    }
}
