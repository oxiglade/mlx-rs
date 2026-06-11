//! Gemma 4 Unified encoder-free audio path: config + feature framing +
//! embedder + weights. Raw 16 kHz frames → RMSNorm(no-scale) → linear.

pub mod config;
pub mod embedder;
pub mod feature;
pub mod weights;
