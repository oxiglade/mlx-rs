use darling::FromMeta;
use itertools::Itertools;
use proc_macro::TokenStream;
use quote::quote;
use syn::{FnArg, Ident, ItemFn, Meta};

const CUSTOM_ATTRIBUTE_OPTIONAL: &str = "optional";
const CUSTOM_ATTRIBUTE_NAMED: &str = "named";

const CUSTOM_ATTRIBUTES: &[&str] = &[CUSTOM_ATTRIBUTE_OPTIONAL, CUSTOM_ATTRIBUTE_NAMED];

#[derive(Default, Debug, FromMeta)]
#[darling(default)]
struct Customize {
    root: Option<syn::LitStr>,
    default_dtype: Option<syn::Path>,
    forwarding_shim: bool,
}

struct MacroGeneration<'a> {
    default_generics: &'a proc_macro2::TokenStream,
    dtype_generics: &'a Option<proc_macro2::TokenStream>,
    forwarding_shim: bool,
}

fn arg_type(attrs: &[syn::Attribute]) -> ArgType {
    for attr in attrs {
        if attr.path().is_ident(CUSTOM_ATTRIBUTE_OPTIONAL) {
            return ArgType::NamedOptional;
        } else if attr.path().is_ident(CUSTOM_ATTRIBUTE_NAMED) {
            return ArgType::Named;
        }
    }
    ArgType::Positional
}

fn remove_attribute(attrs: &mut Vec<syn::Attribute>, targets: &[&str]) {
    attrs.retain(|attr| !targets.iter().any(|target| !attr.path().is_ident(target)));
}

/// Remove "$" prefix from the string
fn remove_prefix_from_str(s: &str) -> String {
    s.trim_start_matches("$").to_string()
}

pub fn expand_generate_macro(
    attr: Option<Meta>,
    mut item: ItemFn, // The original function should be kept as is
) -> Result<TokenStream, syn::Error> {
    let deprecated = item
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("deprecated"))
        .cloned();
    let customize = match attr {
        Some(attr) => Customize::from_meta(&attr).map_err(|e| syn::Error::new_spanned(attr, e))?,
        None => Customize::default(),
    };

    // The mod path where the function can be accessed publicly
    let (fn_mod_path, doc_mod_path) = match customize.root {
        Some(lit_str) => {
            let tokens: proc_macro2::TokenStream = lit_str.parse()?;
            let s = remove_prefix_from_str(&lit_str.value());
            (quote! { #tokens }, s)
        }
        None => (quote! { $crate::ops }, "crate::ops".into()),
    };

    let (default_generics, dtype_generics) =
        handle_generic_args(&item.sig.generics, &customize.default_dtype);
    let generation = MacroGeneration {
        default_generics: &default_generics,
        dtype_generics: &dtype_generics,
        forwarding_shim: customize.forwarding_shim,
    };

    let args = item
        .sig
        .inputs
        .iter_mut()
        .map(|arg| match arg {
            FnArg::Receiver(_) => Err(syn::Error::new_spanned(arg, "self is not allowed")),
            FnArg::Typed(pat_type) => Ok(pat_type),
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut parsed_args = parse_args(args);

    // Check if the last optional argument is `stream`
    if let Some(arg) = parsed_args.last() {
        if arg.ident != "stream" {
            return Err(syn::Error::new_spanned(
                &item,
                "the last optional argument must be `stream`",
            ));
        }
    }
    // Remove the last optional argument `stream`
    parsed_args.pop();

    // Remove "_device" suffix from the macro name if it exists
    let fn_ident = &item.sig.ident;

    let generated = generate_macro(
        &fn_mod_path,
        &doc_mod_path,
        fn_ident,
        &parsed_args,
        &generation,
        deprecated.as_ref(),
    )?;

    let output = quote! {
        #item
        #generated
    };

    Ok(output.into())
}

/// If there are generic arguments, the last argument is assumed to be `dtype`.
///
/// Returns two `syn::Generics`:
/// 1. With the last argument set to `f32`
/// 2. With the last argument set to `$dtype`
fn handle_generic_args(
    generic_args: &syn::Generics,
    default_dtype: &Option<syn::Path>,
) -> (proc_macro2::TokenStream, Option<proc_macro2::TokenStream>) {
    // Count number of generic type arguments
    let count = generic_args
        .params
        .iter()
        .filter(|param| matches!(param, syn::GenericParam::Type(_)))
        .count();

    if count == 0 {
        return (quote! {}, None);
    }

    // All generics arguments except for the last one will be inferred
    let infer_tokens = vec![quote! { _ }; count - 1];

    let default_generics = match default_dtype {
        Some(path) => quote! { ::<#(#infer_tokens,)* #path> },
        None => quote! { ::<#(#infer_tokens,)* f32> },
    };
    let dtype_generics = quote! { ::<#(#infer_tokens,)* $dtype> };

    (default_generics, Some(dtype_generics))
}

#[derive(Debug, Clone, Copy)]
enum ArgType {
    Positional,
    Named,
    NamedOptional,
}

struct Arg {
    ident: Ident,
    arg_type: ArgType,
}

fn parse_args(args: Vec<&mut syn::PatType>) -> Vec<Arg> {
    let mut is_prev_optional = false;
    let mut parsed = Vec::new();
    for arg in args {
        match &*arg.pat {
            syn::Pat::Ident(ident) => {
                let arg_type = arg_type(&arg.attrs);

                let is_positional = matches!(arg_type, ArgType::Positional);
                if is_prev_optional && is_positional {
                    panic!("positional argument cannot follow an optional argument");
                }
                is_prev_optional = matches!(arg_type, ArgType::NamedOptional);

                parsed.push(Arg {
                    ident: ident.ident.clone(),
                    arg_type,
                });
            }
            _ => panic!("unsupported pattern"),
        }

        remove_attribute(&mut arg.attrs, CUSTOM_ATTRIBUTES);
    }
    parsed
}

fn generate_macro(
    fn_mod_path: &proc_macro2::TokenStream,
    doc_mod_path: &str,
    fn_ident: &Ident,
    args: &[Arg],
    generation: &MacroGeneration<'_>,
    deprecated: Option<&syn::Attribute>,
) -> Result<proc_macro2::TokenStream, syn::Error> {
    let mut trimmed_fn_ident_str = fn_ident.to_string();
    if trimmed_fn_ident_str.ends_with("_device") {
        trimmed_fn_ident_str = trimmed_fn_ident_str.trim_end_matches("_device").to_string();
    }
    let trimmed_fn_ident = Ident::new(&trimmed_fn_ident_str, fn_ident.span());

    let mut macro_variants = Vec::new();

    generate_macro_variants(
        fn_mod_path,
        fn_ident,
        &trimmed_fn_ident,
        args,
        generation,
        &mut macro_variants,
    );

    let macro_docs = format!(
        "Macro generated for the function [`{doc_mod_path}::{trimmed_fn_ident}`]. See the function documentation for more details."
    );

    let generated = quote! {
        #[doc = #macro_docs]
        #deprecated
        #[macro_export]
        macro_rules! #trimmed_fn_ident {
            #(
                #macro_variants
            )*
        }
    };

    Ok(generated)
}

fn generate_macro_variants(
    fn_mod_path: &proc_macro2::TokenStream,
    fn_ident: &Ident,
    trimmed_fn_ident: &Ident,
    args: &[Arg],
    generation: &MacroGeneration<'_>,
    macro_variants: &mut Vec<proc_macro2::TokenStream>,
) {
    let args_ident = args.iter().map(|arg| &arg.ident).collect::<Vec<_>>();
    let args_type = args.iter().map(|arg| arg.arg_type).collect::<Vec<_>>();
    let mut optional_indices = Vec::new();
    let mut selected = Vec::with_capacity(args.len());
    for (idx, arg) in args.iter().enumerate() {
        match arg.arg_type {
            ArgType::Positional => {
                selected.push(true);
            }
            ArgType::Named => {
                selected.push(true);
            }
            ArgType::NamedOptional => {
                selected.push(false);
                optional_indices.push(idx);
            }
        }
    }

    for perms in 0..optional_indices.len() + 1 {
        // Select `perms` number of optional arguments
        for selected_indice in optional_indices.iter().permutations(perms) {
            selected_indice.iter().for_each(|&&i| selected[i] = true);

            generate_macro_variants_for_selected_args(
                fn_mod_path,
                fn_ident,
                trimmed_fn_ident,
                &args_ident,
                &args_type,
                &selected,
                generation,
                macro_variants,
            );

            // Clear the selected flag for the next iteration
            selected_indice.iter().for_each(|&&i| selected[i] = false);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn generate_macro_variants_for_selected_args(
    fn_mod_path: &proc_macro2::TokenStream,
    fn_ident: &Ident,
    trimmed_fn_ident: &Ident,
    args_ident: &[&Ident],
    args_type: &[ArgType],
    selected: &[bool],
    generation: &MacroGeneration<'_>,
    macro_variants: &mut Vec<proc_macro2::TokenStream>,
) {
    let default_generics = generation.default_generics;
    let macro_args: Vec<proc_macro2::TokenStream> = args_ident
        .iter()
        .zip(args_type.iter())
        .zip(selected.iter())
        .filter_map(|((ident, arg_type), &selected)| match selected {
            true => {
                let token = match arg_type {
                    ArgType::Positional => quote! { $#ident:expr },
                    ArgType::Named => quote! { #ident=$#ident:expr },
                    ArgType::NamedOptional => quote! { #ident=$#ident:expr },
                };
                Some(token)
            }
            false => None,
        })
        .collect();

    let input: Vec<proc_macro2::TokenStream> = args_ident
        .iter()
        .zip(selected.iter())
        .map(|(ident, &selected)| {
            if selected {
                quote! { $#ident }
            } else {
                quote! { None }
            }
        })
        .collect();

    let default_call = if generation.forwarding_shim {
        quote! {
            #fn_mod_path::#fn_ident #default_generics(
                #(#input,)*
                $crate::Stream::thread_local_or_default(),
            )
        }
    } else {
        quote! { #fn_mod_path::#trimmed_fn_ident #default_generics(#(#input,)*) }
    };

    let variant_body = quote! {
        (
            #(#macro_args),*
        ) => {
            #default_call
        };
        (
            #(#macro_args,)*
            stream=$stream:expr
        ) => {
            #fn_mod_path::#fn_ident #default_generics(#(#input,)* $stream)
        };
    };

    macro_variants.push(variant_body);

    if let Some(dtype_generics) = generation.dtype_generics {
        let default_dtype_call = if generation.forwarding_shim {
            quote! {
                #fn_mod_path::#fn_ident #dtype_generics(
                    #(#input,)*
                    $crate::Stream::thread_local_or_default(),
                )
            }
        } else {
            quote! { #fn_mod_path::#trimmed_fn_ident #dtype_generics(#(#input,)*) }
        };
        let variant_body = quote! {
            (
                #(#macro_args,)*
                dtype=$dtype:ty
            ) => {
                #default_dtype_call
            };
            (
                #(#macro_args,)*
                dtype=$dtype:ty,
                stream=$stream:expr
            ) => {
                #fn_mod_path::#fn_ident #dtype_generics(#(#input,)* $stream)
            };
        };

        macro_variants.push(variant_body);
    }
}
