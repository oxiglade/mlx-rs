use std::fmt::Display;

use darling::{FromDeriveInput, FromField};
use quote::{quote, ToTokens};
use syn::{DeriveInput, Ident, ImplGenerics, TypeGenerics, WhereClause};

pub(crate) type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Clone, FromDeriveInput)]
#[darling(attributes(builder))]
pub(crate) struct BuilderStructProperty {
    pub ident: syn::Ident,

    pub build_with: Option<syn::Ident>,

    pub root: Option<syn::Path>,

    pub err: Option<syn::Path>,

    /// Whether building with the default parameters can fail
    pub default_infallible: Option<bool>,
}

pub(crate) struct BuilderStructAnalyzer<'a> {
    pub struct_ident: &'a Ident,
    pub builder_struct_ident: &'a PathOrIdent,
    pub root: &'a syn::Path,
    pub impl_generics: &'a ImplGenerics<'a>,
    pub type_generics: &'a TypeGenerics<'a>,
    pub where_clause: Option<&'a WhereClause>,
    pub mandatory_fields: &'a [MandatoryField],
    pub optional_fields: &'a [OptionalField],
    pub build_with: Option<&'a Ident>,
    pub err: Option<&'a syn::Path>,
}

impl BuilderStructAnalyzer<'_> {
    pub fn generate_builder_struct(&self) -> proc_macro2::TokenStream {
        let struct_ident = self.struct_ident;
        let builder_ident = self.builder_struct_ident;
        let type_generics = self.type_generics;
        let where_clause = self.where_clause;

        let mandatory_field_idents = self.mandatory_fields.iter().map(|field| &field.ident);
        let mandatory_field_tys = self.mandatory_fields.iter().map(|field| &field.ty);

        let optional_field_idents = self
            .optional_fields
            .iter()
            .map(|field| &field.ident)
            .collect::<Vec<_>>();
        let optional_field_tys = self.optional_fields.iter().map(|field| &field.ty);
        let optional_field_defaults = self.optional_fields.iter().map(|field| &field.default);

        let doc = format!("Builder for `{struct_ident}`.");

        let mandatory_field_doc = format!("See [`{struct_ident}`] for more information.");
        let optional_field_doc = optional_field_idents
            .iter()
            .zip(optional_field_defaults)
            .map(|(ident, default)| {
                format!(
                    "See [`{}::{}`] for more information. Initialized with default value [`{}`].",
                    struct_ident,
                    ident,
                    default.to_token_stream()
                )
            });

        quote! {
            #[doc = #doc]
            #[derive(Debug, Clone)]
            pub struct #builder_ident #type_generics #where_clause {
                #(
                    #[doc = #mandatory_field_doc]
                    #mandatory_field_idents: #mandatory_field_tys,
                )*
                #(
                    #[doc = #optional_field_doc]
                    #optional_field_idents: #optional_field_tys,
                )*
            }
        }
    }

    pub fn impl_builder_new(&self) -> proc_macro2::TokenStream {
        let builder_struct_ident = self.builder_struct_ident;
        let impl_generics = self.impl_generics;
        let type_generics = self.type_generics;
        let where_clause = self.where_clause;
        let mandatory_field_idents = self
            .mandatory_fields
            .iter()
            .map(|field| &field.ident)
            .collect::<Vec<_>>();
        let mandatory_field_types = self.mandatory_fields.iter().map(|field| &field.ty);

        let optional_field_idents = self.optional_fields.iter().map(|field| &field.ident);
        let optional_field_defaults = self.optional_fields.iter().map(|field| &field.default);

        let doc = format!("Creates a new [`{builder_struct_ident}`].");

        quote! {
            impl #impl_generics #builder_struct_ident #type_generics #where_clause {
                #[doc = #doc]
                pub fn new(#(#mandatory_field_idents: impl Into<#mandatory_field_types>),*) -> Self {
                    Self {
                        #(#mandatory_field_idents: #mandatory_field_idents.into(),)*
                        #(#optional_field_idents: #optional_field_defaults,)*
                    }
                }
            }
        }
    }

    pub fn impl_builder_setters(&self) -> proc_macro2::TokenStream {
        let builder_struct_ident = self.builder_struct_ident;
        let impl_generics = self.impl_generics;
        let type_generics = self.type_generics;
        let where_clause = self.where_clause;
        let setters = self.optional_fields.iter().filter_map(|field| {
            if field.skip_setter {
                return None;
            }

            let ident = &field.ident;
            let ty = &field.ty;
            let doc = format!("Sets the value of [`{ident}`].");
            Some(quote! {
                #[doc = #doc]
                pub fn #ident(mut self, #ident: impl Into<#ty>) -> Self {
                    self.#ident = #ident.into();
                    self
                }
            })
        });

        quote! {
            impl #impl_generics #builder_struct_ident #type_generics #where_clause {
                #(#setters)*
            }
        }
    }

    pub fn impl_builder_trait(&self) -> proc_macro2::TokenStream {
        let struct_ident = self.struct_ident;
        let builder_struct_ident = self.builder_struct_ident;
        let root = self.root;
        let impl_generics = self.impl_generics;
        let type_generics = self.type_generics;
        let where_clause = self.where_clause;
        let mandatory_field_idents = self.mandatory_fields.iter().map(|field| &field.ident);
        let optional_field_idents = self.optional_fields.iter().map(|field| &field.ident);

        let err_ty = match self.err {
            Some(err) => quote! { #err },
            None => quote! { std::convert::Infallible },
        };

        let build_body = match self.build_with {
            Some(f) => quote! {
                #f(self)
            },
            None => quote! {
                Ok(#struct_ident {
                    #(#mandatory_field_idents: self.#mandatory_field_idents,)*
                    #(#optional_field_idents: self.#optional_field_idents,)*
                })
            },
        };

        quote! {
            impl #impl_generics #root::builder::Builder<#struct_ident #type_generics> for #builder_struct_ident #type_generics #where_clause {
                type Error = #err_ty;

                fn build(self) -> std::result::Result<#struct_ident #type_generics, Self::Error> {
                    #build_body
                }
            }
        }
    }

    pub(crate) fn impl_builder(&self) -> proc_macro2::TokenStream {
        let builder_new = self.impl_builder_new();
        let builder_setters = self.impl_builder_setters();
        let builder_trait = self.impl_builder_trait();

        quote! {
            #builder_new
            #builder_setters
            #builder_trait
        }
    }

    pub(crate) fn impl_struct_new(&self, is_default_infallible: bool) -> proc_macro2::TokenStream {
        let struct_ident = self.struct_ident;
        let root = self.root;
        let impl_generics = self.impl_generics;
        let type_generics = self.type_generics;
        let where_clause = self.where_clause;

        let mandatory_field_idents = self
            .mandatory_fields
            .iter()
            .map(|field| &field.ident)
            .collect::<Vec<_>>();
        let mandatory_field_types = self.mandatory_fields.iter().map(|field| &field.ty);

        let doc = format!("Creates a new instance of `{struct_ident}`.");

        // TODO: do we want to generate different code for infallible and fallible cases
        let ret = if is_default_infallible {
            quote! { -> Self }
        } else {
            quote! { -> std::result::Result<Self, <<Self as #root::builder::Buildable>::Builder as #root::builder::Builder<Self>>::Error> }
        };

        let unwrap_result = if is_default_infallible {
            quote! { .expect("Build with default parameters should not fail") }
        } else {
            quote! {}
        };

        quote! {
            impl #impl_generics #struct_ident #type_generics #where_clause {
                #[doc = #doc]
                pub fn new(#(#mandatory_field_idents: impl Into<#mandatory_field_types>),*) #ret
                {
                    use #root::builder::Builder;
                    <Self as #root::builder::Buildable>::Builder::new(#(#mandatory_field_idents),*).build()
                        #unwrap_result
                }
            }
        }
    }
}

#[derive(Debug, darling::FromField, PartialEq)]
#[darling(attributes(builder))]
pub(crate) struct BuilderFieldProperty {
    pub ident: Option<syn::Ident>,

    pub ty: syn::Type,

    #[darling(default)]
    pub optional: bool,

    pub default: Option<syn::Path>,

    pub rename: Option<String>,

    #[darling(default)]
    pub ignore: bool,

    pub ty_override: Option<syn::Path>,

    #[darling(default)]
    pub skip_setter: bool,
}

pub(crate) struct MandatoryField {
    pub ident: syn::Ident,
    pub ty: syn::Type,
}

pub(crate) struct OptionalField {
    pub ident: syn::Ident,
    pub ty: syn::Type,
    pub default: syn::Path,
    pub skip_setter: bool,
}

pub(crate) fn parse_fields_from_derive_input(
    item: &DeriveInput,
) -> Result<(Vec<MandatoryField>, Vec<OptionalField>)> {
    match &item.data {
        syn::Data::Struct(data) => parse_fields_from_datastruct(data),
        _ => Err("Only structs are supported".into()),
    }
}

fn parse_fields_from_datastruct(
    item: &syn::DataStruct,
) -> Result<(Vec<MandatoryField>, Vec<OptionalField>)> {
    parse_fields(&item.fields)
}

fn parse_fields(fields: &syn::Fields) -> Result<(Vec<MandatoryField>, Vec<OptionalField>)> {
    let mut mandatory_fields = Vec::new();
    let mut optional_fields = Vec::new();

    let field_props = fields.iter().map(BuilderFieldProperty::from_field);

    for field_prop in field_props {
        let field_prop = field_prop?;
        if field_prop.ignore {
            continue;
        }

        let mut ident = match field_prop.ident {
            Some(ident) => ident,
            None => return Err("Unnamed fields are not supported".into()),
        };

        if let Some(rename) = field_prop.rename {
            ident = syn::Ident::new(&rename, ident.span());
        }

        let ty = match field_prop.ty_override {
            Some(ty_override) => syn::Type::Path(syn::TypePath {
                attrs: Vec::new(),
                qself: None,
                path: ty_override,
            }),
            None => field_prop.ty,
        };

        if field_prop.optional {
            let default = match field_prop.default {
                Some(default) => default,
                None => {
                    return Err(
                        format!("Field {ident} is optional but has no default value").into(),
                    );
                }
            };

            optional_fields.push(OptionalField {
                ident,
                ty,
                default,
                skip_setter: field_prop.skip_setter,
            });
        } else {
            mandatory_fields.push(MandatoryField { ident, ty });
        }
    }

    Ok((mandatory_fields, optional_fields))
}

#[derive(Debug, Clone)]
pub(crate) enum PathOrIdent {
    Path(syn::Path),
    Ident(syn::Ident),
}

impl ToTokens for PathOrIdent {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            PathOrIdent::Path(path) => path.to_tokens(tokens),
            PathOrIdent::Ident(ident) => ident.to_tokens(tokens),
        }
    }
}

impl Display for PathOrIdent {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            PathOrIdent::Path(path) => path.to_token_stream().fmt(f),
            PathOrIdent::Ident(ident) => Display::fmt(ident, f),
        }
    }
}
