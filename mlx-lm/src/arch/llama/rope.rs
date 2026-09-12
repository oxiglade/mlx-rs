use mlx_rs::{fast, Array};

use crate::{ConfigError, InferenceError, RopeConfig, RopeScaling};

pub(super) fn frequencies(config: &RopeConfig) -> Result<Option<Vec<f32>>, ConfigError> {
    let RopeScaling::Llama3 {
        factor,
        low_frequency_factor,
        high_frequency_factor,
        original_max_positions,
    } = config.scaling
    else {
        return Ok(None);
    };
    let low_wavelength = original_max_positions as f32 / low_frequency_factor;
    let high_wavelength = original_max_positions as f32 / high_frequency_factor;
    let mut frequencies = Vec::with_capacity(config.dimensions / 2);
    for index in 0..config.dimensions / 2 {
        let frequency = config
            .theta
            .powf((2 * index) as f32 / config.dimensions as f32);
        let wavelength = std::f32::consts::TAU * frequency;
        let frequency = if wavelength > low_wavelength {
            frequency * factor
        } else if wavelength > high_wavelength && wavelength < low_wavelength {
            let smooth = (original_max_positions as f32 / wavelength - low_frequency_factor)
                / (high_frequency_factor - low_frequency_factor);
            frequency / ((1.0 - smooth) / factor + smooth)
        } else {
            frequency
        };
        if !frequency.is_finite() || frequency <= 0.0 {
            return Err(ConfigError::UnsupportedRope(
                "llama3 frequency exceeds finite f32 range".into(),
            ));
        }
        frequencies.push(frequency);
    }
    Ok(Some(frequencies))
}

pub(super) struct Rope {
    config: RopeConfig,
    frequencies: Option<Array>,
}

impl Rope {
    pub(super) fn new(config: RopeConfig) -> Result<Self, ConfigError> {
        let frequencies = frequencies(&config)?
            .map(|values| {
                let length = super::config::dimension(values.len(), "rope frequency count")?;
                Ok::<_, ConfigError>(Array::from_slice(&values, &[length]))
            })
            .transpose()?;
        Ok(Self {
            config,
            frequencies,
        })
    }

    pub(super) fn apply(&self, input: &Array, offset: usize) -> Result<Array, InferenceError> {
        let offset = i32::try_from(offset)
            .map_err(|_| crate::CacheError::InvalidState("RoPE offset exceeds i32".into()))?;
        let dimensions = super::config::dimension(self.config.dimensions, "rope dimensions")
            .map_err(|error| InferenceError::UnsupportedArchitecture(error.to_string()))?;
        let scale = match self.config.scaling {
            RopeScaling::Linear { factor } => factor.recip(),
            _ => 1.0,
        };
        Ok(fast::rope(
            input,
            dimensions,
            self.config.traditional,
            self.frequencies.is_none().then_some(self.config.theta),
            scale,
            offset,
            self.frequencies.as_ref(),
        )?)
    }
}
