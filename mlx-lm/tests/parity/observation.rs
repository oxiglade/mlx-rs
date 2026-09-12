use std::collections::BTreeMap;

use anyhow::{bail, Result};
use safetensors::Dtype;
use serde::Deserialize;
use serde_json::Value;

#[derive(Clone, Debug, Default)]
pub struct Observation {
    pub config: Option<Value>,
    pub tokenizer: Option<Value>,
    pub chat: Option<Value>,
    pub tensors: BTreeMap<String, Tensor>,
    pub caches: BTreeMap<String, CacheState>,
    pub greedy_ids: Option<Vec<u32>>,
    pub sampled_ids: BTreeMap<String, Vec<u32>>,
    pub text_deltas: Option<Vec<String>>,
    pub finish: Option<(String, Option<u32>)>,
    pub errors: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheState {
    pub offset: usize,
    pub retained: std::ops::Range<usize>,
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub dtype: Dtype,
    pub bytes: Vec<u8>,
}

impl Tensor {
    pub fn floats(&self) -> Result<Vec<f64>> {
        let values = match self.dtype {
            Dtype::F32 => self
                .bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b) as f64)
                .collect(),
            Dtype::F64 => self
                .bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| f64::from_le_bytes(*b))
                .collect(),
            Dtype::F16 => self
                .bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| float16(u16::from_le_bytes(*b)))
                .collect(),
            Dtype::BF16 => self
                .bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16) as f64)
                .collect(),
            dtype => bail!("expected floating tensor, got {dtype:?}"),
        };
        Ok(values)
    }

    pub fn valid_layout(&self) -> bool {
        self.shape
            .iter()
            .try_fold(self.dtype.bitsize(), |n, d| n.checked_mul(*d))
            .and_then(|bits| bits.checked_add(7))
            .map(|bits| bits / 8)
            == Some(self.bytes.len())
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tolerance {
    pub atol: f64,
    pub rtol: f64,
}

#[derive(Clone, Copy, Debug)]
pub enum Policy {
    ExactBits,
    Float(Tolerance),
}

#[derive(Clone, Debug)]
pub struct Expectations {
    pub observation: Observation,
    pub policies: BTreeMap<String, Policy>,
}

impl Expectations {
    pub fn prototype_prefill_and_greedy(&self) -> Self {
        let tensors: BTreeMap<_, _> = self
            .observation
            .tensors
            .iter()
            .filter(|(key, _)| {
                *key == "prefill.full.logits" || key.starts_with("cache.after_prefill.")
            })
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        Self {
            policies: self
                .policies
                .iter()
                .filter(|(key, _)| tensors.contains_key(*key))
                .map(|(key, value)| (key.clone(), *value))
                .collect(),
            observation: Observation {
                tensors,
                caches: self
                    .observation
                    .caches
                    .iter()
                    .filter(|(key, _)| key.starts_with("cache.after_prefill."))
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
                greedy_ids: self.observation.greedy_ids.clone(),
                ..Observation::default()
            },
        }
    }
}

fn float16(bits: u16) -> f64 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 10) & 31;
    let fraction = f64::from(bits & 1023);
    sign * match exponent {
        0 => fraction * 2.0f64.powi(-24),
        31 if fraction == 0.0 => f64::INFINITY,
        31 => f64::NAN,
        _ => (1.0 + fraction / 1024.0) * 2.0f64.powi(i32::from(exponent) - 15),
    }
}
