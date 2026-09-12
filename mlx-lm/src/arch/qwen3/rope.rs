use crate::{InferenceError, RopeConfig, RopeScaling};
use mlx_rs::{fast, Array};

pub(super) struct Rope {
    dimensions: i32,
    traditional: bool,
    base: Option<f32>,
    scale: f32,
    frequencies: Option<Array>,
}

pub(super) fn frequency_table(config: &RopeConfig) -> Vec<f32> {
    (0..config.dimensions / 2)
        .map(|i| {
            let frequency = config.theta.powf((2 * i) as f32 / config.dimensions as f32);
            match config.scaling {
                RopeScaling::None => frequency,
                RopeScaling::Linear { factor } => frequency * factor,
                RopeScaling::Llama3 {
                    factor,
                    low_frequency_factor: low,
                    high_frequency_factor: high,
                    original_max_positions,
                } => {
                    let wavelength = 2.0 * std::f32::consts::PI * frequency;
                    let context = original_max_positions as f32;
                    if wavelength > context / low {
                        frequency * factor
                    } else if wavelength > context / high && wavelength < context / low {
                        let smooth = (context / wavelength - low) / (high - low);
                        frequency / ((1.0 - smooth) / factor + smooth)
                    } else {
                        frequency
                    }
                }
            }
        })
        .collect()
}

impl Rope {
    pub(super) fn new(config: &RopeConfig) -> Result<Self, crate::ConfigError> {
        let dimensions = super::config::dimension("rope dimensions", config.dimensions)?;
        let frequencies = if matches!(config.scaling, RopeScaling::Llama3 { .. }) {
            let table = frequency_table(config);
            Some(Array::from_slice(&table, &[dimensions / 2]))
        } else {
            None
        };
        Ok(Self {
            dimensions,
            traditional: config.traditional,
            base: frequencies.is_none().then_some(config.theta),
            scale: match config.scaling {
                RopeScaling::Linear { factor } => factor.recip(),
                _ => 1.0,
            },
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
            self.scale,
            offset,
            self.frequencies.as_ref(),
        )?)
    }
}
