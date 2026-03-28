pub mod cache;
pub mod error;
// pub mod generate;
pub mod models;
pub mod sampler;
pub mod utils;

use mlx_rs::Array;

use crate::models::qwen3;
use crate::models::qwen3_5;

pub struct ModelInputBuilder<'a, C, T> {
    pub y: &'a Array,
    pub cache: &'a mut Vec<Option<C>>,
    pub state: &'a mut T,
}

pub trait ModelInput<'a, C, T> {
    fn from_model_input_builder(builder: ModelInputBuilder<'a, C, T>) -> Self;
}

impl<'a, C> ModelInput<'a, C, Option<Array>> for qwen3::ModelInput<'a, C> {
    fn from_model_input_builder(builder: ModelInputBuilder<'a, C, Option<Array>>) -> Self {
        let ModelInputBuilder { y, cache, state } = builder;

        Self {
            inputs: y,
            mask: state.as_ref(),
            cache,
        }
    }
}

impl<'a, C> ModelInput<'a, C, Option<Array>> for qwen3_5::ModelInput<'a, C> {
    fn from_model_input_builder(builder: ModelInputBuilder<'a, C, Option<Array>>) -> Self {
        let ModelInputBuilder { y, cache, state } = builder;

        Self {
            inputs: y,
            mask: state.as_ref(),
            cache,
        }
    }
}

pub trait ModelOutput {
    fn logits(&self) -> &Array;
}

impl ModelOutput for Array {
    fn logits(&self) -> &Array {
        self
    }
}
