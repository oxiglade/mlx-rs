//! Gemma 4 rope: proportional (partial-rotation) or plain per layer.
//!
//! Proportional rope rotates only the first `partial_rotary_factor × D`
//! dimensions; the remaining freqs are `+inf` so `fast::rope` leaves those
//! positions untouched. The per-layer kind + theta come from
//! `config.rope_parameters` keyed by `"full_attention"` / `"sliding_attention"`.

use std::collections::HashMap;

use mlx_rs::builder::Builder;
use mlx_rs::module::{Module, ModuleParamMut, ModuleParamRef, ModuleParameters};
use mlx_rs::nn::{self, RopeInput};
use mlx_rs::ops::{arange, concatenate_axis, full};
use mlx_rs::{fast, Array};

use crate::error::Error;
use crate::gemma4::text::config::LayerKind;
use crate::utils::rope::FloatOrString;

const F32_INF: f32 = f32::INFINITY;

/// Partial-rotation rope: only the first `rotated_dims` dims rotate; the
/// rest have `+inf` freqs (ignored by `fast::rope`).
#[derive(Debug, Clone)]
pub struct ProportionalRope {
    pub dims: i32,
    pub rotated_dims: i32,
    pub traditional: bool,
    /// Freqs of length `dims/2`: first `rotated_dims/2` are
    /// `factor * base^exponent`, the rest `+inf`. Not a learnable param.
    pub freqs: Array,
}

impl ModuleParameters for ProportionalRope {
    fn num_parameters(&self) -> usize {
        0
    }
    fn freeze_parameters(&mut self, _r: bool) {}
    fn unfreeze_parameters(&mut self, _r: bool) {}
    fn parameters(&self) -> ModuleParamRef<'_> {
        ModuleParamRef::default()
    }
    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        ModuleParamMut::default()
    }
    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        ModuleParamRef::default()
    }
    fn all_frozen(&self) -> Option<bool> {
        None
    }
    fn any_frozen(&self) -> Option<bool> {
        None
    }
}

impl ProportionalRope {
    pub fn new(
        dims: i32,
        rotated_dims: i32,
        traditional: bool,
        base: f32,
        factor: f32,
    ) -> Result<Self, Error> {
        assert!(rotated_dims <= dims, "rotated_dims must be ≤ dims");
        assert!(rotated_dims % 2 == 0, "rotated_dims must be even");

        let exp = arange::<_, f32>(0.0f32, rotated_dims as f32, 2.0)?
            .divide(Array::from_f32(dims as f32))?;
        let rotated_freqs = exp
            .multiply(Array::from_f32(base.ln()))?
            .exp()?
            .multiply(Array::from_f32(factor))?;
        let inf_count = (dims - rotated_dims) / 2;
        let freqs = if inf_count > 0 {
            let infs = full::<f32>(&[inf_count], Array::from_f32(F32_INF))?;
            concatenate_axis(&[rotated_freqs, infs], 0)?
        } else {
            rotated_freqs
        };
        Ok(Self {
            dims,
            rotated_dims,
            traditional,
            freqs,
        })
    }
}

impl<'a> Module<RopeInput<'a>> for ProportionalRope {
    type Output = Array;
    type Error = Error;

    fn forward(&mut self, input: RopeInput<'a>) -> Result<Array, Self::Error> {
        let RopeInput { x, offset } = input;
        Ok(fast::rope(
            x,
            self.dims,
            self.traditional,
            None,
            1.0,
            offset,
            Some(&self.freqs),
        )?)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

/// Per-layer rope: plain (full rotation) or proportional (partial).
#[derive(Debug, Clone)]
pub enum LayerRope {
    Plain(nn::Rope),
    Proportional(ProportionalRope),
}

impl ModuleParameters for LayerRope {
    fn num_parameters(&self) -> usize {
        match self {
            Self::Plain(r) => r.num_parameters(),
            Self::Proportional(r) => r.num_parameters(),
        }
    }
    fn freeze_parameters(&mut self, r: bool) {
        match self {
            Self::Plain(p) => p.freeze_parameters(r),
            Self::Proportional(p) => p.freeze_parameters(r),
        }
    }
    fn unfreeze_parameters(&mut self, r: bool) {
        match self {
            Self::Plain(p) => p.unfreeze_parameters(r),
            Self::Proportional(p) => p.unfreeze_parameters(r),
        }
    }
    fn parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::Plain(p) => p.parameters(),
            Self::Proportional(p) => p.parameters(),
        }
    }
    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        match self {
            Self::Plain(p) => p.parameters_mut(),
            Self::Proportional(p) => p.parameters_mut(),
        }
    }
    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::Plain(p) => p.trainable_parameters(),
            Self::Proportional(p) => p.trainable_parameters(),
        }
    }
    fn all_frozen(&self) -> Option<bool> {
        match self {
            Self::Plain(p) => p.all_frozen(),
            Self::Proportional(p) => p.all_frozen(),
        }
    }
    fn any_frozen(&self) -> Option<bool> {
        match self {
            Self::Plain(p) => p.any_frozen(),
            Self::Proportional(p) => p.any_frozen(),
        }
    }
}

impl LayerRope {
    pub fn forward(&mut self, input: RopeInput<'_>) -> Result<Array, Error> {
        match self {
            Self::Plain(r) => Ok(r.forward(input)?),
            Self::Proportional(r) => r.forward(input),
        }
    }

    /// As [`Self::forward`] but with a 0-D `Array` offset, routing through
    /// `fast::rope_dynamic` so the per-step decode offset stays on-device
    /// and MLX reuses one compiled rope kernel across decode steps.
    pub fn forward_dynamic(&self, x: &Array, offset: &Array) -> Result<Array, Error> {
        match self {
            Self::Plain(r) => Ok(fast::rope_dynamic(
                x,
                r.dimensions,
                r.traditional,
                r.base,
                r.scale,
                offset,
                None,
            )?),
            Self::Proportional(p) => Ok(fast::rope_dynamic(
                x,
                p.dims,
                p.traditional,
                None,
                1.0_f32,
                offset,
                Some(&p.freqs),
            )?),
        }
    }
}

/// Build the rope for a layer of the given kind from `rope_parameters`.
/// Proportional when `rope_type == "proportional"` or a default rope with
/// a partial rotary factor; plain rope otherwise.
pub(crate) fn build_layer_rope(
    head_dim: i32,
    kind: LayerKind,
    rope_traditional: bool,
    rope_parameters: Option<&HashMap<String, HashMap<String, FloatOrString>>>,
) -> Result<LayerRope, Error> {
    let layer_key = match kind {
        LayerKind::FullAttention => "full_attention",
        LayerKind::SlidingAttention => "sliding_attention",
    };
    let params = rope_parameters.and_then(|m| m.get(layer_key));
    let float_param = |name: &str, default: f32| -> f32 {
        params
            .and_then(|p| p.get(name))
            .and_then(|v| match v {
                FloatOrString::Float(f) => Some(*f),
                FloatOrString::String(_) => None,
            })
            .unwrap_or(default)
    };
    let rope_theta = float_param("rope_theta", 10_000.0);
    let partial_rotary_factor = float_param("partial_rotary_factor", 1.0);
    let factor = float_param("factor", 1.0);
    let rope_type = params
        .and_then(|p| p.get("rope_type"))
        .and_then(|v| match v {
            FloatOrString::String(s) => Some(s.as_str()),
            FloatOrString::Float(_) => None,
        })
        .unwrap_or("default");

    let rotated_dims = ((head_dim as f32) * partial_rotary_factor) as i32 & !1;

    if rope_type == "proportional" || (rope_type == "default" && rotated_dims < head_dim) {
        Ok(LayerRope::Proportional(ProportionalRope::new(
            head_dim,
            rotated_dims,
            rope_traditional,
            rope_theta,
            factor,
        )?))
    } else {
        let rope = nn::RopeBuilder::new(head_dim)
            .traditional(rope_traditional)
            .base(rope_theta)
            .scale(1.0)
            .build()
            .expect("RopeBuilder with explicit base/scale is infallible");
        Ok(LayerRope::Plain(rope))
    }
}
