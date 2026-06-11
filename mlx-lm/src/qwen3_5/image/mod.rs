//! Qwen3.5 image (VLM) path: vision tower, image processor, multimodal
//! embedding stitch, the VLM adapter wrapping the dense decoder, and the
//! vision-aware weight loader. Compiled only with the `image` feature.

pub mod adapter;
pub mod multimodal;
pub mod processor;
pub mod vision;
pub mod weights;
