extern crate proc_macro;
use darling::FromMeta;
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::punctuated::Punctuated;
use syn::{parse_macro_input, parse_quote, DeriveInput, FnArg, ItemEnum, ItemFn, Pat};

mod derive_buildable;
mod derive_builder;
mod generate_builder;
mod generate_macro;
mod shared;

#[derive(Debug, FromMeta)]
enum DeviceType {
    Cpu,
    Gpu,
}

#[derive(Debug)]
struct DefaultDeviceInput {
    device: DeviceType,
}

impl FromMeta for DefaultDeviceInput {
    fn from_meta(meta: &syn::Meta) -> darling::Result<Self> {
        let syn::Meta::NameValue(meta_name_value) = meta else {
            return Err(darling::Error::unsupported_format(
                "expected a name-value attribute",
            ));
        };

        let ident = meta_name_value.path.get_ident().unwrap();
        assert_eq!(ident, "device", "expected `device`");

        let device = DeviceType::from_expr(&meta_name_value.value)?;

        Ok(DefaultDeviceInput { device })
    }
}

#[doc(hidden)]
#[proc_macro_attribute]
pub fn default_device(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = if !attr.is_empty() {
        let meta = syn::parse_macro_input!(attr as syn::Meta);
        Some(DefaultDeviceInput::from_meta(&meta).unwrap())
    } else {
        None
    };

    let mut input_fn = parse_macro_input!(item as ItemFn);
    let original_fn = input_fn.clone();

    // Ensure function name convention
    if !input_fn.sig.ident.to_string().contains("_device") {
        panic!("Function name must end with '_device'");
    }
    let new_fn_name = format_ident!("{}", &input_fn.sig.ident.to_string().replace("_device", ""));
    input_fn.sig.ident = new_fn_name;

    // Filter out the `stream` parameter and reconstruct the Punctuated collection
    let filtered_inputs = input_fn
        .sig
        .inputs
        .iter()
        .filter(|arg| match arg {
            FnArg::Typed(pat_typed) => {
                if let Pat::Ident(pat_ident) = &*pat_typed.pat {
                    pat_ident.ident != "stream"
                } else {
                    true
                }
            }
            _ => true,
        })
        .cloned()
        .collect::<Vec<_>>();

    input_fn.sig.inputs = Punctuated::from_iter(filtered_inputs);

    // Prepend default stream initialization
    let default_stream_stmt = match input.map(|input| input.device) {
        Some(DeviceType::Cpu) => parse_quote! {
            let stream = crate::Stream::thread_local_or_cpu();
        },
        Some(DeviceType::Gpu) => parse_quote! {
            let stream = crate::Stream::thread_local_or_gpu();
        },
        None => parse_quote! {
            let stream = crate::Stream::thread_local_or_default();
        },
    };
    input_fn.block.stmts.insert(0, default_stream_stmt);

    // Combine the original and modified functions into the output
    let expanded = quote! {
        #original_fn

        #input_fn
    };

    TokenStream::from(expanded)
}

#[doc(hidden)]
#[proc_macro]
pub fn generate_test_cases(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as ItemEnum);
    let name = &input.ident;

    let tests = quote! {
        /// MLX's rules for promoting two dtypes.
        #[rustfmt::skip]
        const TYPE_RULES: [[Dtype; 14]; 14] = [
            // bool             uint8               uint16              uint32              uint64              int8                int16               int32               int64               float16             float32             float64,           bfloat16            complex64
            [Dtype::Bool,       Dtype::Uint8,       Dtype::Uint16,      Dtype::Uint32,      Dtype::Uint64,      Dtype::Int8,        Dtype::Int16,       Dtype::Int32,       Dtype::Int64,       Dtype::Float16,     Dtype::Float32,     Dtype::Float64,    Dtype::Bfloat16,    Dtype::Complex64], // bool
            [Dtype::Uint8,      Dtype::Uint8,       Dtype::Uint16,      Dtype::Uint32,      Dtype::Uint64,      Dtype::Int16,       Dtype::Int16,       Dtype::Int32,       Dtype::Int64,       Dtype::Float16,     Dtype::Float32,     Dtype::Float64,    Dtype::Bfloat16,    Dtype::Complex64], // uint8
            [Dtype::Uint16,     Dtype::Uint16,      Dtype::Uint16,      Dtype::Uint32,      Dtype::Uint64,      Dtype::Int32,       Dtype::Int32,       Dtype::Int32,       Dtype::Int64,       Dtype::Float16,     Dtype::Float32,     Dtype::Float64,    Dtype::Bfloat16,    Dtype::Complex64], // uint16
            [Dtype::Uint32,     Dtype::Uint32,      Dtype::Uint32,      Dtype::Uint32,      Dtype::Uint64,      Dtype::Int64,       Dtype::Int64,       Dtype::Int64,       Dtype::Int64,       Dtype::Float16,     Dtype::Float32,     Dtype::Float64,    Dtype::Bfloat16,    Dtype::Complex64], // uint32
            [Dtype::Uint64,     Dtype::Uint64,      Dtype::Uint64,      Dtype::Uint64,      Dtype::Uint64,      Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float16,     Dtype::Float32,     Dtype::Float64,    Dtype::Bfloat16,    Dtype::Complex64], // uint64
            [Dtype::Int8,       Dtype::Int16,       Dtype::Int32,       Dtype::Int64,       Dtype::Float32,     Dtype::Int8,        Dtype::Int16,       Dtype::Int32,       Dtype::Int64,       Dtype::Float16,     Dtype::Float32,     Dtype::Float64,    Dtype::Bfloat16,    Dtype::Complex64], // int8
            [Dtype::Int16,      Dtype::Int16,       Dtype::Int32,       Dtype::Int64,       Dtype::Float32,     Dtype::Int16,       Dtype::Int16,       Dtype::Int32,       Dtype::Int64,       Dtype::Float16,     Dtype::Float32,     Dtype::Float64,    Dtype::Bfloat16,    Dtype::Complex64], // int16
            [Dtype::Int32,      Dtype::Int32,       Dtype::Int32,       Dtype::Int64,       Dtype::Float32,     Dtype::Int32,       Dtype::Int32,       Dtype::Int32,       Dtype::Int64,       Dtype::Float16,     Dtype::Float32,     Dtype::Float64,    Dtype::Bfloat16,    Dtype::Complex64], // int32
            [Dtype::Int64,      Dtype::Int64,       Dtype::Int64,       Dtype::Int64,       Dtype::Float32,     Dtype::Int64,       Dtype::Int64,       Dtype::Int64,       Dtype::Int64,       Dtype::Float16,     Dtype::Float32,     Dtype::Float64,    Dtype::Bfloat16,    Dtype::Complex64], // int64
            [Dtype::Float16,    Dtype::Float16,     Dtype::Float16,     Dtype::Float16,     Dtype::Float16,     Dtype::Float16,     Dtype::Float16,     Dtype::Float16,     Dtype::Float16,     Dtype::Float16,     Dtype::Float32,     Dtype::Float64,    Dtype::Float32,     Dtype::Complex64], // float16
            [Dtype::Float32,    Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float64,    Dtype::Float32,     Dtype::Complex64], // float32
            [Dtype::Float64,    Dtype::Float64,     Dtype::Float64,     Dtype::Float64,     Dtype::Float64,     Dtype::Float64,     Dtype::Float64,     Dtype::Float64,     Dtype::Float64,     Dtype::Float64,     Dtype::Float64,     Dtype::Float64,    Dtype::Float64,     Dtype::Complex64], // Dtype::Float64
            [Dtype::Bfloat16,   Dtype::Bfloat16,    Dtype::Bfloat16,    Dtype::Bfloat16,    Dtype::Bfloat16,    Dtype::Bfloat16,    Dtype::Bfloat16,    Dtype::Bfloat16,    Dtype::Bfloat16,    Dtype::Float32,     Dtype::Float32,     Dtype::Float64,    Dtype::Bfloat16,    Dtype::Complex64], // bfloat16
            [Dtype::Complex64,  Dtype::Complex64,   Dtype::Complex64,   Dtype::Complex64,   Dtype::Complex64,   Dtype::Complex64,   Dtype::Complex64,   Dtype::Complex64,   Dtype::Complex64,   Dtype::Complex64,   Dtype::Complex64,   Dtype::Complex64,  Dtype::Complex64,   Dtype::Complex64], // complex64
        ];

        #[cfg(test)]
        mod generated_tests {
            use super::*;
            use strum::IntoEnumIterator;
            use pretty_assertions::assert_eq;

            #[test]
            fn test_all_combinations() {
                for a in #name::iter() {
                    for b in #name::iter() {
                        let result = a.promote_with(b);
                        let expected = TYPE_RULES[a as usize][b as usize];
                        assert_eq!(result, expected, "{}", format!("Failed promotion test for {:?} and {:?}", a, b));
                    }
                }
            }
        }
    };

    TokenStream::from(quote! {
        #input
        #tests
    })
}

/// Generates a builder struct for the given struct.
///
/// This macro should be used in conjunction with the `#[derive(Buildable)]` derive macro.
/// See the [`Buildable`] macro for more information.
#[doc(hidden)]
#[proc_macro]
pub fn generate_builder(input: TokenStream) -> TokenStream {
    // let input = parse_macro_input!(input as ItemStruct);
    let input = parse_macro_input!(input as DeriveInput);
    let builder = generate_builder::expand_generate_builder(&input).unwrap();
    quote::quote! {
        #input
        #builder
    }
    .into()
}

/// Derive `mlx_rs::builder::Buildable` for a struct. When used with the `generate_builder` macro,
/// a builder struct `<Struct>Builder` will be generated.
///
/// # Attributes
///
/// ## `#[buildable]`
///
/// ### Arguments
///
/// - `builder`: Path to the builder struct. Default to `<Struct>Builder` if not provided.
/// - `root`: Path to the root module. Default to `::mlx_rs` if not provided.
///
/// ## `#[builder]`
///
/// **Note**: This attribute has no effect if NOT used with the `generate_builder` macro.
///
/// ### Arguments when applied on struct
///
/// - `build_with`: Function ident to build the struct.
/// - `root`: Path to the root module. Default to `::mlx_rs` if not provided.
/// - `err`: Type of error to return when build fails. Default to `std::convert::Infallible`
///   if not provided.
/// - `default_infallible`: Whether the default error type is infallible. Default to `err.is_none()`
///   if not provided. When `true`, the generated `<Struct>::new()` method will unwrap the build result
///   and return `<Struct>`. When `false`, the generated `<Struct>::new()` method will return `Result<<Struct>, Err>`.
///
/// ### Arguments when applied on field
///
/// - `optional`: Whether the field is optional. Default to `false` if not provided.
/// - `default`: Path to the default value for the field. This is required if the field is optional.
/// - `rename`: Rename the field in the builder struct.
/// - `ignore`: Ignore the field in the builder struct.
/// - `ty_override`: Override the type of the field in the builder struct.
/// - `skip_setter`: Skip the setter method for the field in the builder struct.
///
/// # Example
///
/// ```rust,ignore
/// use mlx_internal_macros::*;
/// use mlx_rs::builder::{Buildable, Builder};
///
/// generate_builder! {
///     /// Test struct for the builder generation.
///     #[derive(Debug, Buildable)]
///     #[builder(build_with = build_test_struct)]
///     struct TestStruct {
///         #[builder(optional, default = TestStruct::DEFAULT_OPT_FIELD_1)]
///         opt_field_1: i32,
///         #[builder(optional, default = TestStruct::DEFAULT_OPT_FIELD_2)]
///         opt_field_2: i32,
///         mandatory_field_1: i32,
///
///         #[builder(ignore)]
///         ignored_field: String,
///     }
/// }
///
/// fn build_test_struct(
///     builder: TestStructBuilder,
/// ) -> std::result::Result<TestStruct, std::convert::Infallible> {
///     Ok(TestStruct {
///         opt_field_1: builder.opt_field_1,
///         opt_field_2: builder.opt_field_2,
///         mandatory_field_1: builder.mandatory_field_1,
///         ignored_field: String::from("ignored"),
///     })
/// }
///
/// impl TestStruct {
///     pub const DEFAULT_OPT_FIELD_1: i32 = 1;
///     pub const DEFAULT_OPT_FIELD_2: i32 = 2;
/// }
///
/// #[test]
/// fn test_generated_builder() {
///     let test_struct = <TestStruct as Buildable>::Builder::new(4)
///         .opt_field_1(2)
///         .opt_field_2(3)
///         .build()
///         .unwrap();
///
///     assert_eq!(test_struct.opt_field_1, 2);
///     assert_eq!(test_struct.opt_field_2, 3);
///     assert_eq!(test_struct.mandatory_field_1, 4);
///     assert_eq!(test_struct.ignored_field, String::from("ignored"));
/// }
/// ```
#[doc(hidden)]
#[proc_macro_derive(Buildable, attributes(buildable, builder))]
pub fn derive_buildable(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let builder = derive_buildable::expand_derive_buildable(input).unwrap();
    TokenStream::from(builder)
}

/// Derive `mlx_rs::builder::Builder` trait for a struct and generate the following methods:
///
/// - `<Struct>Builder::new(mandatory_fields)`: Create a new builder with the mandatory fields.
/// - setter methods for each optinal field
/// - `<Struct>::new(mandatory_fields)`: Create the struct from the builder with the mandatory fields.
///
/// # Attributes
///
/// ## `#[builder]`
///
/// ### Arguments when applied on struct
///
/// - `build_with`: Function ident to build the struct.
/// - `root`: Path to the root module. Default to `::mlx_rs` if not provided.
/// - `err`: Type of error to return when build fails. Default to `std::convert::Infallible`
///   if not provided.
/// - `default_infallible`: Whether the default error type is infallible. Default to `err.is_none()`
///   if not provided. When `true`, the generated `<Struct>::new()` method will unwrap the build result
///   and return `<Struct>`. When `false`, the generated `<Struct>::new()` method will return `Result<<Struct>, Err>`.
///
/// ### Arguments when applied on field
///
/// - `optional`: Whether the field is optional. Default to `false` if not provided.
/// - `default`: Path to the default value for the field. This is required if the field is optional.
/// - `rename`: Rename the field in the builder struct.
/// - `ignore`: Ignore the field in the builder struct.
/// - `ty_override`: Override the type of the field in the builder struct.
/// - `skip_setter`: Skip the setter method for the field in the builder struct.
#[doc(hidden)]
#[proc_macro_derive(Builder, attributes(builder))]
pub fn derive_builder(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let builder = derive_builder::expand_derive_builder(input).unwrap();
    TokenStream::from(builder)
}

/// Generate a macro that expands to the given function for ergonomic purposes.
///
/// See `mlx-rs/mlx-tests/test_generate_macro.rs` for more usage examples.
///
/// ```rust,ignore
/// #![allow(unused_variables)]
///
/// use mlx_internal_macros::{default_device, generate_macro};
/// use mlx_rs::{Stream, StreamOrDevice};
///
/// /// Test macro generation.
/// #[generate_macro(customize(root = "$crate"))] // Default is `$crate::ops`
/// #[default_device]
/// fn foo_device(
///     a: i32, // Mandatory argument
///     b: i32, // Mandatory argument
///     #[optional] c: Option<i32>, // Optional argument
///     #[optional] d: impl Into<Option<i32>>, // Optional argument but impl Trait
///     #[optional] stream: impl AsRef<Stream>, // stream always optional and placed at the end
/// ) -> i32 {
///     a + b + c.unwrap_or(0) + d.into().unwrap_or(0)
/// }
///
/// assert_eq!(foo!(1, 2), 3);
/// assert_eq!(foo!(1, 2, c = Some(3)), 6);
/// assert_eq!(foo!(1, 2, d = Some(4)), 7);
/// assert_eq!(foo!(1, 2, c = Some(3), d = Some(4)), 10);
///
/// let stream = Stream::new();
///
/// assert_eq!(foo!(1, 2, stream = &stream), 3);
/// assert_eq!(foo!(1, 2, c = Some(3), stream = &stream), 6);
/// assert_eq!(foo!(1, 2, d = Some(4), stream = &stream), 7);
/// assert_eq!(foo!(1, 2, c = Some(3), d = Some(4), stream = &stream), 10);
/// ```
#[doc(hidden)]
#[proc_macro_attribute]
pub fn generate_macro(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr = if !attr.is_empty() {
        let meta = syn::parse_macro_input!(attr as syn::Meta);
        Some(meta)
    } else {
        None
    };
    let item = parse_macro_input!(item as ItemFn);
    generate_macro::expand_generate_macro(attr, item).unwrap()
}
