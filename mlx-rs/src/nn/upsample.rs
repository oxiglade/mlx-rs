use crate::{
    array,
    error::Exception,
    macros::ModuleParameters,
    module::Module,
    ops::{
        abs, broadcast_to, ceil, clip, expand_dims_axes, floor,
        indexing::{ArrayIndex, ArrayIndexOp, Ellipsis, IndexOp, NewAxis, TryIndexOp},
    },
    transforms::compile::compile,
    Array,
};

use crate::utils::SingleOrVec;

/// Upsample mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpsampleMode {
    /// Nearest neighbor upsampling
    Nearest,

    /// Linear interpolation upsampling.
    Linear {
        /// If `true`, the top and left edge of the input and output
        /// will match as will the bottom right edge
        align_corners: bool,
    },

    /// Cubic interpolation upsampling.
    Cubic {
        /// If `true`, the top and left edge of the input and output
        align_corners: bool,
    },
}

/// Upsample the input signal spatially
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = crate)]
pub struct Upsample {
    /// The multiplier for the spatial size.
    ///
    /// If a single `float` is provided, it is the multiplier for all spatial dimensions.
    /// Otherwise, the number of scale factors provided must match the
    /// number of spatial dimensions.
    pub scale_factor: SingleOrVec<f32>,

    /// The upsampling algorithm
    pub mode: UpsampleMode,
}

impl Upsample {
    /// Create a new `Upsample` module
    pub fn new(scale_factor: impl Into<SingleOrVec<f32>>, mode: UpsampleMode) -> Self {
        let scale_factor = scale_factor.into();
        Upsample { scale_factor, mode }
    }

    fn forward_inner(&self, x: &Array, scale: &[f32]) -> Result<Array, Exception> {
        match self.mode {
            UpsampleMode::Nearest => upsample_nearest(x, scale),
            UpsampleMode::Linear { align_corners } => {
                interpolate(x, scale, linear_indices, align_corners)
            }
            UpsampleMode::Cubic { align_corners } => {
                interpolate(x, scale, cubic_indices, align_corners)
            }
        }
    }
}

impl Module<&Array> for Upsample {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let dimensions = x.ndim() - 2;

        if dimensions == 0 {
            return Err(Exception::custom(format!(
                "[Upsample] The input should have at least 
                1 spatial dimension which means it should be at least 
                3D but {}D was provided",
                x.ndim()
            )));
        }

        match &self.scale_factor {
            SingleOrVec::Single(scale) => {
                let scale = vec![*scale; dimensions];
                self.forward_inner(x, &scale[..])
            }
            SingleOrVec::Vec(scales) => self.forward_inner(x, &scales[..]),
        }
    }

    fn training_mode(&mut self, _mode: bool) {}
}

#[allow(non_snake_case)]
fn upsample_nearest(x: &Array, scale: &[f32]) -> Result<Array, Exception> {
    let dimensions = x.ndim() - 2;
    if dimensions != scale.len() {
        return Err(Exception::custom(format!(
            "The number of scale factors ({}) must match the number of spatial dimensions ({})",
            scale.len(),
            dimensions
        )));
    }

    // Get a truncated version of the scales
    let int_scales = scale.iter().map(|&s| s as i32).collect::<Vec<_>>();
    let int_float_scales = int_scales.iter().map(|&s| s as f32).collect::<Vec<_>>();

    if int_float_scales == scale {
        // Int scale means we can simply expand-broadcast and reshape
        let mut shape = x.shape().to_vec();
        (0..dimensions).for_each(|d| {
            shape.insert(2 + 2 * d, 1);
        });
        let mut x = x.reshape(&shape)?;

        (0..dimensions).for_each(|d| {
            shape[2 + 2 * d] = int_scales[d];
        });
        x = broadcast_to(&x, &shape)?;

        (0..dimensions).for_each(|d| {
            shape[d + 1] *= shape[d + 2];
            shape.remove(d + 2);
        });
        x = x.reshape(&shape)?;

        Ok(x)
    } else {
        // Float scales
        let shape_len = x.shape().len();
        let N = &x.shape()[1..shape_len - 1];
        let mut indices: Vec<ArrayIndexOp> = vec![(..).index_op()];

        for (i, (n, s)) in N.iter().zip(scale.iter()).enumerate() {
            indices.push(nearest_indices(*n, *s, i, dimensions)?.index_op());
        }

        x.try_index(&indices[..])
    }
}

type IndexWeight = (Array, Array);

type IndicesFn = fn(i32, f32, bool, usize, usize) -> Result<Vec<IndexWeight>, Exception>;

#[allow(non_snake_case)]
fn interpolate(
    x: &Array,
    scale: &[f32],
    indices_fn: IndicesFn,
    align_corners: bool,
) -> Result<Array, Exception> {
    let dimensions = x.ndim() - 2;
    if dimensions != scale.len() {
        return Err(Exception::custom(format!(
            "The number of scale factors ({}) must match the number of spatial dimensions ({})",
            scale.len(),
            dimensions
        )));
    }

    let N = &x.shape()[1..x.ndim() - 1];

    // compute the sampling grid
    let mut index_weights = Vec::with_capacity(N.len());
    for (i, (n, s)) in N.iter().zip(scale.iter()).enumerate() {
        index_weights.push(indices_fn(*n, *s, align_corners, i, dimensions)?);
    }

    // sample and compute the weights
    let prod = product(&index_weights);
    let mut samples = Vec::with_capacity(prod.len());
    let mut weights = Vec::with_capacity(prod.len());
    for index_weight in prod {
        let (index, weight): (Vec<&Array>, Vec<&Array>) =
            index_weight.iter().map(|(i, w)| (i, w)).unzip();
        let mut index_ops = index.iter().map(|i| i.index_op()).collect::<Vec<_>>();

        let mut sample_indices = vec![(..).index_op()];
        sample_indices.append(&mut index_ops);
        samples.push(x.index(&sample_indices[..]));

        weights.push(weight.into_iter().product::<Array>());
    }

    // interpolate
    let acc = &weights[0] * &samples[0];
    weights[1..]
        .iter()
        .zip(samples[1..].iter())
        .try_fold(acc, |acc, (w, s)| acc.add(w.multiply(s)?))
}

fn product<T>(values: &[Vec<T>]) -> Vec<Vec<&T>> {
    if values.is_empty() {
        return vec![];
    }

    // if there are N items in values and M values per tuple there
    // will be M^N values in the result
    let per_tuple = values[0].len();
    let count = (0..values.len()).fold(1, |acc, _| acc * per_tuple);

    let mut result = Vec::with_capacity(count);
    for result_index in 0..count {
        let mut items = vec![];

        // use % and / to compute which item will be used from each value[i]
        let mut index_generator = result_index;
        for value in values {
            let index = index_generator % per_tuple;
            items.push(&value[index]);
            index_generator /= per_tuple;
        }

        result.push(items);
    }

    result
}

fn nearest_indices(
    dimension: i32,
    scale: f32,
    dim: usize,
    ndim: usize,
) -> Result<Array, Exception> {
    scaled_indices(dimension, scale, true, dim, ndim).and_then(|i| i.as_type::<i32>())
}

fn linear_indices(
    dimension: i32,
    scale: f32,
    align_corners: bool,
    dim: usize,
    ndim: usize,
) -> Result<Vec<IndexWeight>, Exception> {
    let mut indices = scaled_indices(dimension, scale, align_corners, dim, ndim)?;
    indices = clip(&indices, (0, dimension - 1))?;
    let indices_left = floor(&indices)?;
    let indices_right = ceil(&indices)?;
    let weight = expand_dims_axes(&indices.subtract(&indices_left)?, &[-1])?;

    let indices_left = indices_left.as_type::<i32>()?;
    let indices_right = indices_right.as_type::<i32>()?;

    Ok(vec![
        // SAFETY: arith ops with scalars won't panic
        (indices_left, array!(1.0) - &weight),
        (indices_right, weight),
    ])
}

fn cubic_indices(
    dimension: i32,
    scale: f32,
    align_corners: bool,
    dim: usize,
    ndim: usize,
) -> Result<Vec<IndexWeight>, Exception> {
    let indices = scaled_indices(dimension, scale, align_corners, dim, ndim)?;

    // SAFETY: arith ops with scalars won't panic
    let mut indices_l1 = floor(&indices)?;
    let mut indices_r1 = floor(&(&indices + 1))?;
    let mut indices_l2 = (&indices_l1) - 1;
    let mut indices_r2 = (&indices_r1) + 1;

    let weight_l1 = compiled_get_weight1(&indices, &indices_l1)?.index((Ellipsis, NewAxis));
    let weight_r1 = compiled_get_weight1(&indices, &indices_r1)?.index((Ellipsis, NewAxis));
    let weight_l2 = compiled_get_weight2(&indices, &indices_l2)?.index((Ellipsis, NewAxis));
    let weight_r2 = compiled_get_weight2(&indices, &indices_r2)?.index((Ellipsis, NewAxis));

    // Padding with border value
    indices_l1 = clip(&indices_l1, (0, dimension - 1))?.as_type::<i32>()?;
    indices_r1 = clip(&indices_r1, (0, dimension - 1))?.as_type::<i32>()?;
    indices_l2 = clip(&indices_l2, (0, dimension - 1))?.as_type::<i32>()?;
    indices_r2 = clip(&indices_r2, (0, dimension - 1))?.as_type::<i32>()?;

    Ok(vec![
        (indices_l1, weight_l1),
        (indices_r1, weight_r1),
        (indices_l2, weight_l2),
        (indices_r2, weight_r2),
    ])
}

fn compiled_get_weight1(ind: &Array, grid: &Array) -> Result<Array, Exception> {
    // PyTorch uses -0.5 for antialiasing=true (compatibility with PIL)
    // and uses -0.75 for antialiasing=false (compatibility with OpenCV)

    let get_weight1 = |(ind_, grid_): (&Array, &Array)| {
        let a = -0.75;
        let x = abs(ind_ - grid_)?;
        Ok((array!(a + 2.0) * &x - array!(a + 3.0)) * &x * &x + 1.0)
    };
    let mut compiled = compile(get_weight1, true);
    compiled((ind, grid))
}

fn compiled_get_weight2(ind: &Array, grid: &Array) -> Result<Array, Exception> {
    let get_weight2 = |(ind_, grid_): (&Array, &Array)| {
        let a = -0.75;
        let x = abs(ind_ - grid_)?;
        Ok((((&x - 5.0) * &x + 8.0) * &x - 4.0) * a)
    };
    let mut compiled = compile(get_weight2, true);
    compiled((ind, grid))
}

#[allow(non_snake_case)]
fn scaled_indices(
    N: i32,
    scale: f32,
    align_corners: bool,
    dim: usize,
    ndim: usize,
) -> Result<Array, Exception> {
    let M = (scale * N as f32) as i32;

    let indices = if align_corners {
        // SAFETY: arith ops on with scalars won't panic
        Array::from_iter(0..M, &[M]).as_type::<f32>()? * ((N as f32 - 1.0) / (M as f32 - 1.0))
    } else {
        let step = 1.0 / scale;
        let start = ((M as f32 - 1.0) * step - N as f32 + 1.0) / 2.0;
        // SAFETY: arith ops with scalars won't panic
        Array::from_iter(0..M, &[M]).as_type::<f32>()? * step - start
    };

    let mut shape = vec![1; ndim];
    shape[dim] = -1;

    indices.reshape(&shape)
}

#[cfg(test)]
mod tests {
    use crate::assert_array_eq;

    use super::*;

    // The unit test below is adapted from the swift binding.
    #[test]
    fn test_nearest() {
        // BHWC
        let input = array!([1, 2, 3, 4], shape = [1, 2, 2, 1]);

        let mut up = Upsample::new(2.0, UpsampleMode::Nearest);
        let result = up.forward(&input).and_then(|r| r.squeeze()).unwrap();

        assert_eq!(result.shape(), &[4, 4]);

        // array([[1, 1, 2, 2],
        //        [1, 1, 2, 2],
        //        [3, 3, 4, 4],
        //        [3, 3, 4, 4]], dtype=int32)
        let expected = array!(
            [1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4],
            shape = [4, 4]
        )
        .as_type::<i32>()
        .unwrap();
        assert_eq!(result, expected);
    }

    // The unit test below is adapted from the swift binding.
    #[test]
    fn test_linear() {
        // BHWC
        let input = array!([1, 2, 3, 4], shape = [1, 2, 2, 1]);

        let mut up = Upsample::new(
            2.0,
            UpsampleMode::Linear {
                align_corners: false,
            },
        );
        let result = up.forward(&input).and_then(|r| r.squeeze()).unwrap();

        assert_eq!(result.shape(), &[4, 4]);

        // array([[1, 1.25, 1.75, 2],
        //        [1.5, 1.75, 2.25, 2.5],
        //        [2.5, 2.75, 3.25, 3.5],
        //        [3, 3.25, 3.75, 4]], dtype=float32)
        let expected = array!(
            [
                1.0, 1.25, 1.75, 2.0, 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3.0, 3.25, 3.75,
                4.0
            ],
            shape = [4, 4]
        )
        .as_type::<f32>()
        .unwrap();
        assert_eq!(result, expected);
    }

    // The expected output for the test case below is obtained from the python binding.
    #[test]
    fn test_cubic() {
        // BHWC
        let input = array!([1, 2, 3, 4], shape = [1, 2, 2, 1]);

        let mut up = Upsample::new(
            2.0,
            UpsampleMode::Cubic {
                align_corners: false,
            },
        );
        let result = up.forward(&input).and_then(|r| r.squeeze()).unwrap();

        assert_eq!(result.shape(), &[4, 4]);

        // Expected output from the python binding version 0.17.2
        // array([[0.683594, 1.01562, 1.5625, 1.89453],
        //     [1.34766, 1.67969, 2.22656, 2.55859],
        //     [2.44141, 2.77344, 3.32031, 3.65234],
        //     [3.10547, 3.4375, 3.98438, 4.31641]], dtype=float32)
        let expected = array!(
            [
                0.683594, 1.01562, 1.5625, 1.89453, 1.34766, 1.67969, 2.22656, 2.55859, 2.44141,
                2.77344, 3.32031, 3.65234, 3.10547, 3.4375, 3.98438, 4.31641
            ],
            shape = [4, 4]
        )
        .as_type::<f32>()
        .unwrap();

        assert_array_eq!(result, expected, 1e-5);
    }
}
