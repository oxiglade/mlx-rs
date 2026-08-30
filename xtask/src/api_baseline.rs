use quote::ToTokens;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use syn::{Attribute, ImplItem, Item, UseTree, Visibility};

const SCHEMA_VERSION: u32 = 1;
const LIMITATIONS: &[&str] = &[
    "cfg attributes are not evaluated or attached to entries, so the inventory is the union of source-visible configurations without per-entry availability",
    "default_device function twins and generate_macro exported macros are enumerated from their source attributes without expanding procedural macros",
    "derive macros, generate_builder output, and other procedural or declarative macro expansions are not inventoried",
    "glob reexports from external crates are recorded as unresolved wildcard entries rather than expanded",
    "non-function declarations are source-token snapshots and can include private fields or default implementation bodies",
];

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ApiBaseline {
    schema_version: u32,
    crate_name: String,
    source_root: String,
    limitations: Vec<String>,
    pub entries: Vec<ApiEntry>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub struct ApiEntry {
    pub kind: String,
    pub path: String,
    pub signature: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generated_by: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deprecated: Option<Deprecation>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub struct Deprecation {
    pub since: Option<String>,
    pub note: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct LocalEntry {
    kind: String,
    name: String,
    signature: String,
    generated_by: Option<String>,
    deprecated: Option<Deprecation>,
}

#[derive(Default)]
struct Module {
    entries: Vec<LocalEntry>,
    inherent: Vec<InherentEntry>,
    children: BTreeMap<String, Module>,
    public_children: BTreeSet<String>,
    public_uses: Vec<UseSpec>,
}

struct InherentEntry {
    owner: String,
    member: LocalEntry,
}

#[derive(Clone)]
enum UseSpec {
    Name {
        segments: Vec<String>,
        alias: String,
        signature: String,
    },
    Glob {
        segments: Vec<String>,
        signature: String,
    },
}

pub fn run(repo_root: &Path, args: &[String]) -> i32 {
    match parse_args(args).and_then(|output| {
        let baseline = generate(&repo_root.join("mlx-rs"), "mlx_rs")?;
        let mut bytes = serde_json::to_vec_pretty(&baseline)
            .map_err(|error| format!("failed to serialize API baseline: {error}"))?;
        bytes.push(b'\n');
        if let Some(path) = output {
            fs::write(&path, bytes)
                .map_err(|error| format!("failed to write {}: {error}", path.display()))
        } else {
            io::stdout()
                .write_all(&bytes)
                .map_err(|error| format!("failed to write API baseline: {error}"))
        }
    }) {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("api-baseline: {error}");
            2
        }
    }
}

fn parse_args(args: &[String]) -> Result<Option<PathBuf>, String> {
    match args {
        [] => Ok(None),
        [flag, path] if flag == "--out" => Ok(Some(PathBuf::from(path))),
        _ => Err("usage: cargo run -p xtask -- api-baseline [--out <path>]".to_owned()),
    }
}

pub(crate) fn generate(crate_root: &Path, crate_name: &str) -> Result<ApiBaseline, String> {
    let source_root = crate_root.join("src/lib.rs");
    let mut generated_macros = BTreeMap::new();
    let root = parse_module(&source_root, &mut generated_macros)?;
    let mut cache = BTreeMap::new();
    let mut visiting = BTreeSet::new();
    let mut exports = exports_for_path(&root, &[], &mut cache, &mut visiting)?;
    for (name, entry) in generated_macros {
        exports.insert(name, entry);
    }
    let public_types = exports
        .iter()
        .filter(|(_, entry)| matches!(entry.kind.as_str(), "struct" | "enum" | "union" | "type"))
        .map(|(path, _)| path.clone())
        .collect::<Vec<_>>();
    let mut inherent = Vec::new();
    collect_inherent(&root, &mut inherent);
    for method in inherent {
        let Some(owner_path) = public_types
            .iter()
            .filter(|path| path.rsplit("::").next() == Some(method.owner.as_str()))
            .min_by_key(|path| (path.matches("::").count(), path.as_str()))
        else {
            continue;
        };
        exports.insert(
            format!("{owner_path}::{}", method.member.name),
            method.member,
        );
    }
    let entries = exports
        .into_iter()
        .map(|(path, entry)| ApiEntry {
            kind: entry.kind,
            path: format!("{crate_name}::{path}"),
            signature: entry.signature,
            generated_by: entry.generated_by,
            deprecated: entry.deprecated,
        })
        .collect();
    Ok(ApiBaseline {
        schema_version: SCHEMA_VERSION,
        crate_name: crate_name.to_owned(),
        source_root: "mlx-rs/src/lib.rs".to_owned(),
        limitations: LIMITATIONS
            .iter()
            .map(|value| (*value).to_owned())
            .collect(),
        entries,
    })
}

fn parse_module(
    file_path: &Path,
    generated_macros: &mut BTreeMap<String, LocalEntry>,
) -> Result<Module, String> {
    let source = fs::read_to_string(file_path)
        .map_err(|error| format!("failed to read {}: {error}", file_path.display()))?;
    let file = syn::parse_file(&source)
        .map_err(|error| format!("failed to parse {}: {error}", file_path.display()))?;
    parse_items(&file.items, file_path, generated_macros)
}

fn parse_items(
    items: &[Item],
    file_path: &Path,
    generated_macros: &mut BTreeMap<String, LocalEntry>,
) -> Result<Module, String> {
    let mut module = Module::default();
    for item in items {
        match item {
            Item::Fn(function) if is_public(&function.vis) => {
                let name = function.sig.ident.to_string();
                module.entries.push(LocalEntry {
                    kind: "function".to_owned(),
                    name: name.clone(),
                    signature: signature(&function.sig),
                    generated_by: None,
                    deprecated: deprecation(&function.attrs),
                });
                if has_attr(&function.attrs, "default_device") {
                    if let Some(generated) = default_device_signature(&function.sig) {
                        module.entries.push(LocalEntry {
                            kind: "function".to_owned(),
                            name: generated.ident.to_string(),
                            signature: signature(&generated),
                            generated_by: Some("default_device".to_owned()),
                            deprecated: deprecation(&function.attrs),
                        });
                    }
                }
                if has_attr(&function.attrs, "generate_macro") {
                    let macro_name = name.strip_suffix("_device").unwrap_or(&name).to_owned();
                    generated_macros.insert(
                        macro_name.clone(),
                        LocalEntry {
                            kind: "macro".to_owned(),
                            name: macro_name.clone(),
                            signature: format!("macro_rules! {macro_name}"),
                            generated_by: Some("generate_macro".to_owned()),
                            deprecated: deprecation(&function.attrs),
                        },
                    );
                }
            }
            Item::Struct(item) if is_public(&item.vis) => {
                push_item(&mut module, "struct", item.ident.to_string(), item)
            }
            Item::Enum(item) if is_public(&item.vis) => {
                push_item(&mut module, "enum", item.ident.to_string(), item)
            }
            Item::Union(item) if is_public(&item.vis) => {
                push_item(&mut module, "union", item.ident.to_string(), item)
            }
            Item::Type(item) if is_public(&item.vis) => {
                push_item(&mut module, "type", item.ident.to_string(), item)
            }
            Item::Trait(item) if is_public(&item.vis) => {
                push_item(&mut module, "trait", item.ident.to_string(), item)
            }
            Item::TraitAlias(item) if is_public(&item.vis) => {
                push_item(&mut module, "trait_alias", item.ident.to_string(), item)
            }
            Item::Const(item) if is_public(&item.vis) => {
                push_item(&mut module, "constant", item.ident.to_string(), item)
            }
            Item::Static(item) if is_public(&item.vis) => {
                push_item(&mut module, "static", item.ident.to_string(), item)
            }
            Item::Impl(item) if item.trait_.is_none() => {
                let Some(owner) = type_name(&item.self_ty) else {
                    continue;
                };
                for impl_item in &item.items {
                    match impl_item {
                        ImplItem::Fn(method) if is_public(&method.vis) => {
                            let name = method.sig.ident.to_string();
                            module.inherent.push(InherentEntry {
                                owner: owner.clone(),
                                member: LocalEntry {
                                    kind: "method".to_owned(),
                                    name,
                                    signature: signature(&method.sig),
                                    generated_by: None,
                                    deprecated: deprecation(&method.attrs),
                                },
                            });
                            if has_attr(&method.attrs, "default_device") {
                                if let Some(generated) = default_device_signature(&method.sig) {
                                    module.inherent.push(InherentEntry {
                                        owner: owner.clone(),
                                        member: LocalEntry {
                                            kind: "method".to_owned(),
                                            name: generated.ident.to_string(),
                                            signature: signature(&generated),
                                            generated_by: Some("default_device".to_owned()),
                                            deprecated: deprecation(&method.attrs),
                                        },
                                    });
                                }
                            }
                        }
                        ImplItem::Const(value) if is_public(&value.vis) => {
                            module.inherent.push(InherentEntry {
                                owner: owner.clone(),
                                member: LocalEntry {
                                    kind: "associated_constant".to_owned(),
                                    name: value.ident.to_string(),
                                    signature: canonical_tokens(value),
                                    generated_by: None,
                                    deprecated: None,
                                },
                            })
                        }
                        ImplItem::Type(value) if is_public(&value.vis) => {
                            module.inherent.push(InherentEntry {
                                owner: owner.clone(),
                                member: LocalEntry {
                                    kind: "associated_type".to_owned(),
                                    name: value.ident.to_string(),
                                    signature: canonical_tokens(value),
                                    generated_by: None,
                                    deprecated: None,
                                },
                            })
                        }
                        _ => {}
                    }
                }
            }
            Item::Macro(item) if has_attr(&item.attrs, "macro_export") => {
                if let Some(ident) = &item.ident {
                    let name = ident.to_string();
                    generated_macros.insert(
                        name.clone(),
                        LocalEntry {
                            kind: "macro".to_owned(),
                            name: name.clone(),
                            signature: format!("macro_rules! {name}"),
                            generated_by: None,
                            deprecated: deprecation(&item.attrs),
                        },
                    );
                }
            }
            Item::Mod(item) => {
                let name = item.ident.to_string();
                let child = if let Some((_, items)) = &item.content {
                    parse_items(items, file_path, generated_macros)?
                } else {
                    let child_path = module_file(file_path, &name)?;
                    parse_module(&child_path, generated_macros)?
                };
                if is_public(&item.vis) {
                    module.public_children.insert(name.clone());
                }
                module.children.insert(name, child);
            }
            Item::Use(item) if is_public(&item.vis) => {
                let mut prefix = Vec::new();
                flatten_use(&item.tree, &mut prefix, &mut module.public_uses);
            }
            _ => {}
        }
    }
    Ok(module)
}

fn push_item<T: ToTokens>(module: &mut Module, kind: &str, name: String, item: &T) {
    module.entries.push(LocalEntry {
        kind: kind.to_owned(),
        name,
        signature: canonical_tokens(item),
        generated_by: None,
        deprecated: None,
    });
}

fn module_file(parent: &Path, name: &str) -> Result<PathBuf, String> {
    let parent_dir = parent
        .parent()
        .ok_or_else(|| format!("module source {} has no parent directory", parent.display()))?;
    let stem = parent
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("");
    let base = if matches!(stem, "lib" | "main" | "mod") {
        parent_dir.to_path_buf()
    } else {
        parent_dir.join(stem)
    };
    let flat = base.join(format!("{name}.rs"));
    let nested = base.join(name).join("mod.rs");
    if flat.is_file() {
        Ok(flat)
    } else if nested.is_file() {
        Ok(nested)
    } else {
        Err(format!(
            "could not resolve module {name} from {}",
            parent.display()
        ))
    }
}

fn flatten_use(tree: &UseTree, prefix: &mut Vec<String>, output: &mut Vec<UseSpec>) {
    match tree {
        UseTree::Path(path) => {
            prefix.push(path.ident.to_string());
            flatten_use(&path.tree, prefix, output);
            prefix.pop();
        }
        UseTree::Name(name) => {
            let mut segments = prefix.clone();
            segments.push(name.ident.to_string());
            output.push(UseSpec::Name {
                alias: name.ident.to_string(),
                signature: format!("pub use {}", segments.join("::")),
                segments,
            });
        }
        UseTree::Rename(rename) => {
            let mut segments = prefix.clone();
            segments.push(rename.ident.to_string());
            output.push(UseSpec::Name {
                alias: rename.rename.to_string(),
                signature: format!("pub use {} as {}", segments.join("::"), rename.rename),
                segments,
            });
        }
        UseTree::Glob(_) => output.push(UseSpec::Glob {
            segments: prefix.clone(),
            signature: format!("pub use {}::*", prefix.join("::")),
        }),
        UseTree::Group(group) => {
            for tree in &group.items {
                flatten_use(tree, prefix, output);
            }
        }
    }
}

fn exports_for_path(
    root: &Module,
    path: &[String],
    cache: &mut BTreeMap<Vec<String>, BTreeMap<String, LocalEntry>>,
    visiting: &mut BTreeSet<Vec<String>>,
) -> Result<BTreeMap<String, LocalEntry>, String> {
    if let Some(exports) = cache.get(path) {
        return Ok(exports.clone());
    }
    if !visiting.insert(path.to_vec()) {
        return Err(format!(
            "cyclic public reexport in module {}",
            path.join("::")
        ));
    }
    let module =
        get_module(root, path).ok_or_else(|| format!("unknown module {}", path.join("::")))?;
    let mut exports = module
        .entries
        .iter()
        .cloned()
        .map(|entry| (entry.name.clone(), entry))
        .collect::<BTreeMap<_, _>>();

    for child_name in &module.public_children {
        let mut child_path = path.to_vec();
        child_path.push(child_name.clone());
        exports.insert(
            child_name.clone(),
            LocalEntry {
                kind: "module".to_owned(),
                name: child_name.clone(),
                signature: format!("pub mod {child_name}"),
                generated_by: None,
                deprecated: None,
            },
        );
        for (name, entry) in exports_for_path(root, &child_path, cache, visiting)? {
            exports.insert(format!("{child_name}::{name}"), entry);
        }
    }

    for public_use in &module.public_uses {
        match public_use {
            UseSpec::Glob {
                segments,
                signature,
            } => {
                if let Some(target) = resolve_module_path(root, path, segments) {
                    for (name, entry) in exports_for_path(root, &target, cache, visiting)? {
                        exports.insert(name, entry);
                    }
                } else {
                    let name = format!("{}::*", segments.join("::"));
                    exports.insert(
                        name.clone(),
                        LocalEntry {
                            kind: "unresolved_reexport".to_owned(),
                            name,
                            signature: signature.clone(),
                            generated_by: None,
                            deprecated: None,
                        },
                    );
                }
            }
            UseSpec::Name {
                segments,
                alias,
                signature,
            } => {
                let (module_segments, item_name) = segments.split_at(segments.len() - 1);
                let resolved =
                    resolve_module_path(root, path, module_segments).and_then(|target| {
                        exports_for_path(root, &target, cache, visiting)
                            .ok()
                            .and_then(|entries| entries.get(&item_name[0]).cloned())
                    });
                exports.insert(
                    alias.clone(),
                    resolved.unwrap_or_else(|| LocalEntry {
                        kind: "reexport".to_owned(),
                        name: alias.clone(),
                        signature: signature.clone(),
                        generated_by: None,
                        deprecated: None,
                    }),
                );
            }
        }
    }

    visiting.remove(path);
    cache.insert(path.to_vec(), exports.clone());
    Ok(exports)
}

fn resolve_module_path(
    root: &Module,
    current: &[String],
    segments: &[String],
) -> Option<Vec<String>> {
    let mut candidate = current.to_vec();
    let mut index = 0;
    if segments.first().is_some_and(|segment| segment == "crate") {
        candidate.clear();
        index = 1;
    } else if segments.first().is_some_and(|segment| segment == "self") {
        index = 1;
    }
    while segments
        .get(index)
        .is_some_and(|segment| segment == "super")
    {
        candidate.pop()?;
        index += 1;
    }
    candidate.extend_from_slice(&segments[index..]);
    get_module(root, &candidate).map(|_| candidate)
}

fn get_module<'a>(root: &'a Module, path: &[String]) -> Option<&'a Module> {
    let mut module = root;
    for segment in path {
        module = module.children.get(segment)?;
    }
    Some(module)
}

fn collect_inherent(module: &Module, output: &mut Vec<InherentEntry>) {
    output.extend(module.inherent.iter().map(|entry| InherentEntry {
        owner: entry.owner.clone(),
        member: entry.member.clone(),
    }));
    for child in module.children.values() {
        collect_inherent(child, output);
    }
}

fn type_name(ty: &syn::Type) -> Option<String> {
    let syn::Type::Path(path) = ty else {
        return None;
    };
    path.path
        .segments
        .last()
        .map(|segment| segment.ident.to_string())
}

fn default_device_signature(signature: &syn::Signature) -> Option<syn::Signature> {
    let mut generated = signature.clone();
    let name = generated.ident.to_string();
    let name = name.strip_suffix("_device")?;
    generated.ident = if let Some(name) = name.strip_prefix("r#") {
        syn::Ident::new_raw(name, generated.ident.span())
    } else {
        syn::Ident::new(name, generated.ident.span())
    };
    generated.inputs = generated
        .inputs
        .into_iter()
        .filter(|input| match input {
            syn::FnArg::Typed(input) => !matches!(
                input.pat.as_ref(),
                syn::Pat::Ident(ident) if ident.ident == "stream"
            ),
            syn::FnArg::Receiver(_) => true,
        })
        .collect();
    Some(generated)
}

fn signature(signature: &syn::Signature) -> String {
    let mut signature = signature.clone();
    for input in &mut signature.inputs {
        if let syn::FnArg::Typed(input) = input {
            input
                .attrs
                .retain(|attr| !attr.path().is_ident("optional") && !attr.path().is_ident("named"));
        }
    }
    format!("pub {}", canonical_tokens(&signature))
}

fn canonical_tokens(tokens: &impl ToTokens) -> String {
    tokens.to_token_stream().to_string()
}

fn is_public(visibility: &Visibility) -> bool {
    matches!(visibility, Visibility::Public(_))
}

fn has_attr(attrs: &[Attribute], name: &str) -> bool {
    attrs.iter().any(|attr| attr.path().is_ident(name))
}

fn deprecation(attrs: &[Attribute]) -> Option<Deprecation> {
    let attr = attrs
        .iter()
        .find(|attr| attr.path().is_ident("deprecated"))?;
    let mut deprecated = Deprecation {
        since: None,
        note: None,
    };
    attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("since") {
            deprecated.since = Some(meta.value()?.parse::<syn::LitStr>()?.value());
        } else if meta.path.is_ident("note") {
            deprecated.note = Some(meta.value()?.parse::<syn::LitStr>()?.value());
        }
        Ok(())
    })
    .ok()?;
    Some(deprecated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idiom_wave_foundations_and_fft_surface_are_canonical() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let api = super::generate(&repo_root.join("mlx-rs"), "mlx_rs").unwrap();
        let entry = |path: &str| api.entries.iter().find(|entry| entry.path == path).unwrap();

        for path in [
            "mlx_rs::Axes",
            "mlx_rs::thread_local_default_stream",
            "mlx_rs::with_device",
            "mlx_rs::with_stream",
            "mlx_rs::fft::FftnOptions",
        ] {
            entry(path);
        }

        for name in [
            "fft",
            "fft2",
            "fftn",
            "fftshift",
            "ifft",
            "ifft2",
            "ifftn",
            "ifftshift",
            "irfft",
            "irfft2",
            "irfftn",
            "rfft",
            "rfft2",
            "rfftn",
        ] {
            let canonical = entry(&format!("mlx_rs::fft::{name}"));
            assert_eq!(canonical.kind, "function");
            assert_eq!(canonical.generated_by, None);
            assert_eq!(canonical.deprecated, None);

            let compatibility_function = entry(&format!("mlx_rs::fft::{name}_device"));
            let function_deprecation = compatibility_function.deprecated.as_ref().unwrap();
            assert_eq!(function_deprecation.since.as_deref(), Some("0.26.0"));
            assert!(function_deprecation
                .note
                .as_deref()
                .is_some_and(|note| note.contains(name)));

            let compatibility_macro = entry(&format!("mlx_rs::{name}"));
            assert_eq!(compatibility_macro.kind, "macro");
            assert_eq!(
                compatibility_macro.generated_by.as_deref(),
                Some("generate_macro")
            );
            assert_eq!(
                compatibility_macro.deprecated,
                compatibility_function.deprecated
            );
        }

        assert!(entry("mlx_rs::fft::fftn")
            .signature
            .contains("options : FftnOptions"));
        assert!(entry("mlx_rs::fft::rfftn")
            .signature
            .contains("options : FftnOptions"));
    }

    #[test]
    fn inventory_resolves_reexports_and_default_device_twins() {
        let root = tempfile::tempdir().unwrap();
        let src = root.path().join("src");
        fs::create_dir(&src).unwrap();
        fs::write(
            src.join("lib.rs"),
            r#"
                mod hidden;
                pub use hidden::*;
                pub mod ops;

                pub struct Array;

                #[default_device]
                pub fn r#where_device(stream: impl AsRef<Stream>) -> Result<()> {
                    Ok(())
                }

                impl Array {
                    #[default_device]
                    pub fn eval_device(&self, stream: impl AsRef<Stream>) -> Result<()> {
                        Ok(())
                    }
                }
            "#,
        )
        .unwrap();
        fs::write(
            src.join("hidden.rs"),
            r#"
                #[default_device]
                pub fn add_device(
                    lhs: &Array,
                    rhs: &Array,
                    #[optional] axis: impl Into<Option<i32>>,
                    stream: impl AsRef<Stream>,
                ) -> Result<Array> {
                    todo!()
                }

                fn private_helper() {}
            "#,
        )
        .unwrap();
        fs::create_dir(src.join("ops")).unwrap();
        fs::write(
            src.join("ops.rs"),
            r#"
                mod cumulative;
                pub use cumulative::*;
            "#,
        )
        .unwrap();
        fs::write(
            src.join("ops/cumulative.rs"),
            r#"
                impl Array {
                    #[default_device]
                    pub fn sum_device(&self, stream: impl AsRef<Stream>) -> Result<()> {
                        Ok(())
                    }
                }
            "#,
        )
        .unwrap();

        let first = super::generate(root.path(), "demo").unwrap();
        let second = super::generate(root.path(), "demo").unwrap();
        assert_eq!(first, second);

        let paths = first
            .entries
            .iter()
            .map(|entry| entry.path.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            paths,
            vec![
                "demo::Array",
                "demo::Array::eval",
                "demo::Array::eval_device",
                "demo::Array::sum",
                "demo::Array::sum_device",
                "demo::add",
                "demo::add_device",
                "demo::ops",
                "demo::r#where",
                "demo::r#where_device",
            ]
        );
        for path in ["demo::Array::eval", "demo::Array::sum", "demo::add"] {
            let entry = first
                .entries
                .iter()
                .find(|entry| entry.path == path)
                .unwrap();
            assert_eq!(entry.generated_by.as_deref(), Some("default_device"));
        }
        assert!(first
            .entries
            .iter()
            .all(|entry| !entry.signature.contains("optional")));
        assert!(first
            .entries
            .iter()
            .all(|entry| !entry.signature.is_empty()));
    }
}
