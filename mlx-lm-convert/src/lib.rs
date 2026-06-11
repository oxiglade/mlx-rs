//! In-tree bf16/fp16 → quantised safetensors converter for mlx-lm-supported
//! checkpoints.
//!
//! Per-model [`Rewriter`] tables decide which tensors are kept, dropped,
//! or re-mapped. Unknown shapes hard-error rather than silently dropping
//! tensors (MTP weights, exotic submodules) the way upstream converters do.
//!
//! Currently supports Qwen 3.6 MoE (with MTP). Other families are deferred
//! until they're actually needed.
//!
//! See [`convert`] for the entry point and `src/bin/mlx-lm-convert.rs`
//! for the CLI driver.

mod plan;
mod quantize;
mod runner;
mod shards;

pub mod gemma4_unified;
pub mod qwen3_5;

pub use plan::{QuantClass, RewriteOutput, Rewriter};
pub use runner::{convert, ConvertOptions, ConvertReport};

// `mlx-lm-convert` is logically a binary with helper modules — every
// caller of the lib is its own bin (`src/bin/mlx-lm-convert.rs`). The
// project's "thiserror in libs, anyhow in bins" rule applies cleanly:
// re-export `anyhow::{Result, anyhow}` so module code reads the same
// way as the bin and there is no per-crate Error enum to maintain.
pub use anyhow::{anyhow, Result};
