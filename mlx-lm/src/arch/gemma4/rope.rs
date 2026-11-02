use crate::{ConfigError, InferenceError};
use mlx_rs::{fast, Array};

pub(super) struct LayerRope {
    dimensions: i32,
    traditional: bool,
    base: Option<f32>,
    frequencies: Option<Array>,
}

pub(super) fn proportional_frequency_table(
    dimensions: usize,
    partial_rotary_factor: f32,
    theta: f32,
) -> Vec<f32> {
    let half = dimensions / 2;
    let rotated_dims = ((dimensions as f32) * partial_rotary_factor) as usize;
    let rotated_pairs = (rotated_dims / 2).min(half);
    (0..half)
        .map(|i| {
            if i < rotated_pairs {
                theta.powf((2 * i) as f32 / dimensions as f32)
            } else {
                f32::INFINITY
            }
        })
        .collect()
}

impl LayerRope {
    pub(super) fn new(
        dimensions: usize,
        partial_rotary_factor: f32,
        theta: f32,
        traditional: bool,
    ) -> Result<Self, ConfigError> {
        let dims_i32 = super::config::dimension("rope dimensions", dimensions)?;
        let rotated_dims = ((dimensions as f32) * partial_rotary_factor) as usize;
        if rotated_dims == 0 || !rotated_dims.is_multiple_of(2) || rotated_dims > dimensions {
            return Err(super::config::invalid(
                "partial_rotary_factor",
                "rotated dimensions must be positive, even, and at most head_dim",
            ));
        }
        let frequencies = if (partial_rotary_factor - 1.0).abs() > f32::EPSILON {
            let table = proportional_frequency_table(dimensions, partial_rotary_factor, theta);
            Some(Array::from_slice(&table, &[dims_i32 / 2]))
        } else {
            None
        };
        Ok(Self {
            dimensions: dims_i32,
            traditional,
            base: frequencies.is_none().then_some(theta),
            frequencies,
        })
    }

    pub(super) fn apply(&self, input: &Array, offset: usize) -> Result<Array, InferenceError> {
        let offset = i32::try_from(offset)
            .map_err(|_| crate::CacheError::InvalidState("RoPE offset exceeds MLX range".into()))?;
        Ok(fast::rope(
            input,
            self.dimensions,
            self.traditional,
            self.base,
            1.0,
            offset,
            self.frequencies.as_ref(),
        )?)
    }
}
