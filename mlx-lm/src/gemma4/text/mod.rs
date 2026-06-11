//! Gemma 4 text decoder: hybrid sliding/global attention, GeGLU MLP,
//! logit soft-capping, tied embeddings.

pub mod adapter;
pub mod cache;
pub mod config;
pub mod moe;
pub mod rope;
#[allow(
    clippy::module_inception,
    reason = "the family's core decoder module is conventionally `text`"
)]
pub mod text;
pub mod weights;

pub(crate) use adapter::load_context;
pub use config::{ModelConfig, TextConfig};
