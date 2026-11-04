#![deny(missing_docs, missing_debug_implementations)]

//! Neural network support for MLX
//!
//! Layers with builders use `Layer::new(required_args)` for defaults and
//! `LayerBuilder::new(required_args).optional(value).build()` to override them;
//! some layers, such as [`Embedding`] and the pooling layers, have only direct constructors.
//! Import [`Builder`](crate::builder::Builder) to call `build()`, which returns a `Result`.
//! `Layer::new` returns either the layer or a `Result`, depending on the layer.
//! Defaults are documented on builder fields, often as associated `DEFAULT_*` constants
//! such as [`Linear::DEFAULT_BIAS`].
//!
//! ```rust
//! use mlx_rs::{builder::Builder, nn::{Linear, LinearBuilder}};
//!
//! let with_bias = Linear::new(8, 4)?;
//! let without_bias = LinearBuilder::new(8, 4).bias(false).build()?;
//! assert!(with_bias.bias.is_some());
//! assert!(without_bias.bias.is_none());
//! # Ok::<(), mlx_rs::error::Exception>(())
//! ```

mod activation;
mod container;
mod convolution;
mod convolution_transpose;
mod dropout;
mod embedding;
mod linear;
mod normalization;
mod pooling;
mod positional_encoding;
mod quantized;
mod recurrent;
mod transformer;
mod upsample;
mod value_and_grad;

pub use activation::*;
pub use container::*;
pub use convolution::*;
pub use convolution_transpose::*;
pub use dropout::*;
pub use embedding::*;
pub use linear::*;
pub use normalization::*;
pub use pooling::*;
pub use positional_encoding::*;
pub use quantized::*;
pub use recurrent::*;
pub use transformer::*;
pub use upsample::*;
pub use value_and_grad::*;
