extern crate proc_macro;

#[proc_macro_derive(ExternalDerive)]
pub fn external_derive(_: proc_macro::TokenStream) -> proc_macro::TokenStream {
    proc_macro::TokenStream::new()
}
