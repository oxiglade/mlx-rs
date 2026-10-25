extern crate proc_macro;
use proc_macro::TokenStream;
use syn::{parse_macro_input, DeriveInput};

mod module_parameters;
mod quantizable;
mod util;

/// Derive the `ModuleParameters` trait for a struct. Mark a field with
/// `#[param]` attribute to include it in the parameters. The field type must
/// implement the `mlx_rs::module::Parameter` trait.
///
/// # Example
///
/// ```text
/// use mlx_macros::ModuleParameters;
/// use mlx_rs::module::{ModuleParameters, Param};
///
/// #[derive(ModuleParameters)]
/// struct Example {
///     #[param]
///     regular: Param<Array>,
///
///     #[param]
///     optional: Param<Option<Array>>,
///
///     #[param]
///     nested: Inner,
///
///     #[param]
///     vec_nested: Vec<Inner>,
///
///     #[param]
///     trait_object: Box<dyn Module>,
///
///     #[param]
///     trait_object_vec: Vec<Box<dyn Module>>,
/// }
///
/// #[derive(ModuleParameters)]
/// struct Inner {
///     #[param]
///     a: Param<Array>,
/// }
/// ```
#[proc_macro_derive(ModuleParameters, attributes(module, param))]
pub fn derive_module_parameters(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let module_param_impl = module_parameters::expand_module_parameters(&input).unwrap();
    TokenStream::from(module_param_impl)
}

/// Derive the `Quantizable` trait for a struct. Mark a field with
/// `#[quantizable]` attribute to include it in the quantization process.
/// Only support types `M` that `M::Quantized = Self`
///
/// See `mlx-rs/mlx-tests/tests/test_quantizable.rs` for example usage.
///
/// # Panics
///
/// This macro will panic if the struct does not have any field marked with
/// `#[quantizable]`.
#[proc_macro_derive(Quantizable, attributes(quantizable))]
pub fn derive_quantizable_module(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let quantizable_module_impl = quantizable::expand_quantizable(&input).unwrap();
    TokenStream::from(quantizable_module_impl)
}
