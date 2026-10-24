use public_api::tokens::Token;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use syn::spanned::Spanned;
use syn::visit::Visit;

const NIGHTLY: &str = include_str!("../api-nightly.txt");
const TARGET: &str = "aarch64-apple-darwin";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub(crate) struct ApiEntry {
    pub path: String,
    kind: String,
    signature: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    impl_header: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    macro_helpers: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    deprecated: Option<Deprecation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    alias_target: Option<String>,
    generated: bool,
    hidden: bool,
    pub feature_sets: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Ord, PartialOrd, Serialize)]
struct Deprecation {
    since: Option<String>,
    note: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct ApiBaseline {
    schema_version: u32,
    crate_name: String,
    target: String,
    nightly: String,
    rustdoc: String,
    normalizer: String,
    equivalent_feature_sets: BTreeMap<String, String>,
    feature_sets: Vec<FeatureSet>,
    pub entries: Vec<ApiEntry>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct FeatureSet {
    id: String,
    default_features: bool,
    features: Vec<String>,
}

fn feature_sets() -> Vec<FeatureSet> {
    [
        ("cpu-only", false, vec!["accelerate"]),
        (
            "cpu-only-safetensors",
            false,
            vec!["accelerate", "safetensors"],
        ),
        ("default", true, vec![]),
        ("safetensors", true, vec!["safetensors"]),
    ]
    .into_iter()
    .map(|(id, default_features, features)| FeatureSet {
        id: id.into(),
        default_features,
        features: features.into_iter().map(str::to_owned).collect(),
    })
    .collect()
}

fn nightly(tool: &str) -> Result<Command, String> {
    let mut command = Command::new("rustup");
    command.args(["run", NIGHTLY.trim(), tool]);
    let output = Command::new("rustup")
        .args(["which", "--toolchain", NIGHTLY.trim(), "rustc"])
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(format!("install {} for API extraction", NIGHTLY.trim()));
    }
    let compiler = PathBuf::from(
        String::from_utf8(output.stdout)
            .map_err(|e| e.to_string())?
            .trim(),
    );
    command
        .env("RUSTC", &compiler)
        .env("RUSTDOC", compiler.with_file_name("rustdoc"));
    command
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS");
    command
        .env_remove("RUSTDOCFLAGS")
        .env_remove("CARGO_ENCODED_RUSTDOCFLAGS")
        .env_remove("CARGO_BUILD_TARGET");
    Ok(command)
}

fn checked(command: &mut Command) -> Result<(), String> {
    let status = command.status().map_err(|e| format!("{command:?}: {e}"))?;
    if !status.success() {
        return Err(format!("{command:?} failed: {status}"));
    }
    Ok(())
}

fn target_dir(repo: &Path) -> PathBuf {
    std::env::var_os("MLX_RS_API_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo.join("target/api-inventory"))
}

pub(crate) fn generate(repo: &Path) -> Result<ApiBaseline, String> {
    let started = Instant::now();
    let target = target_dir(repo);
    fs::create_dir_all(&target).map_err(|e| e.to_string())?;
    let host = nightly("rustc")?
        .args(["--version", "--verbose"])
        .output()
        .map_err(|e| e.to_string())?;
    if !host.status.success()
        || !String::from_utf8_lossy(&host.stdout)
            .lines()
            .any(|line| line == format!("host: {TARGET}"))
    {
        return Err(format!("API extraction requires host {TARGET}"));
    }
    qualify(repo, &target)?;
    let version = nightly("rustdoc")?
        .arg("--version")
        .output()
        .map_err(|e| e.to_string())?;
    if !version.status.success() {
        return Err("cannot read pinned rustdoc version".into());
    }
    macro_dependency(repo, &target)?;
    let sets = feature_sets();
    let mut merged: BTreeMap<ApiEntry, Vec<String>> = BTreeMap::new();
    for set in &sets {
        let entries = extract_crate(repo, &target, set)?;
        eprintln!("api-baseline: {}: {} entries", set.id, entries.len());
        fs::write(
            target.join(format!("{}.json", set.id)),
            serde_json::to_vec_pretty(&entries).unwrap(),
        )
        .map_err(|e| e.to_string())?;
        for mut entry in entries {
            entry.feature_sets.clear();
            merged.entry(entry).or_default().push(set.id.clone());
        }
    }
    for (id, base, features) in [
        ("metal-jit", "default", vec!["metal-jit".into()]),
        (
            "metal-jit-safetensors",
            "safetensors",
            vec!["metal-jit".into(), "safetensors".into()],
        ),
    ] {
        let jit = FeatureSet {
            id: id.into(),
            default_features: true,
            features,
        };
        let mut entries = extract_crate(repo, &target, &jit)?;
        for entry in &mut entries {
            entry.feature_sets.clear();
        }
        let expected: Vec<_> = merged
            .iter()
            .filter(|(_, sets)| sets.iter().any(|set| set == base))
            .map(|(entry, _)| entry.clone())
            .collect();
        if entries != expected {
            return Err(format!("{id} changes the public surface; add it to the API feature sets before regenerating"));
        }
        eprintln!(
            "api-baseline: {id} matches {base} ({} entries)",
            entries.len()
        );
    }
    let entries = merged
        .into_iter()
        .map(|(mut entry, sets)| {
            entry.feature_sets = sets;
            entry
        })
        .collect();
    eprintln!(
        "api-baseline: extraction wall time {:.2}s",
        started.elapsed().as_secs_f64()
    );
    Ok(ApiBaseline {
        schema_version: 2,
        crate_name: "mlx_rs".into(),
        target: TARGET.into(),
        nightly: NIGHTLY.trim().into(),
        rustdoc: String::from_utf8(version.stdout)
            .map_err(|e| e.to_string())?
            .trim()
            .into(),
        normalizer: "public-api 0.50.0 / rustdoc JSON 55".into(),
        equivalent_feature_sets: [
            ("metal-jit".into(), "default".into()),
            ("metal-jit-safetensors".into(), "safetensors".into()),
        ]
        .into(),
        feature_sets: sets,
        entries,
    })
}

fn extract_crate(repo: &Path, target: &Path, set: &FeatureSet) -> Result<Vec<ApiEntry>, String> {
    let native_config = if set.features.iter().any(|f| f == "metal-jit") {
        "metal-jit"
    } else if set.default_features {
        "metal"
    } else {
        "cpu-only"
    };
    let build_dir = target.join(native_config);
    let json_path = build_dir.join("doc/mlx_rs.json");
    let hidden_path = build_dir.join("hidden.json");
    for hidden in [true, false] {
        let mut command = nightly("cargo")?;
        let metallib = std::env::var_os("MLX_RS_METAL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| target.join("metallib"));
        command.env(
            "MLX_RS_METAL_PATH",
            if set.features.iter().any(|f| f == "metal-jit") {
                metallib.join("metal-jit")
            } else {
                metallib
            },
        );
        command
            .current_dir(repo)
            .args([
                "rustdoc",
                "--locked",
                "-p",
                "mlx-rs",
                "--lib",
                "--target-dir",
            ])
            .arg(&build_dir);
        if !set.default_features {
            command.arg("--no-default-features");
        }
        if !set.features.is_empty() {
            command.args(["--features", &set.features.join(",")]);
        }
        command.args(["--", "-Z", "unstable-options", "--output-format", "json"]);
        if hidden {
            command.arg("--document-hidden-items");
        }
        checked(&mut command)?;
        if hidden {
            fs::copy(&json_path, &hidden_path).map_err(|e| e.to_string())?;
        }
    }
    expand_macro_reexports(&hidden_path, &target.join("macro-dependencies"), true)?;
    expand_macro_reexports(&json_path, &target.join("macro-dependencies"), false)?;
    normalize(repo, &hidden_path, &json_path, &set.id)
}

fn macro_dependency(repo: &Path, target: &Path) -> Result<(), String> {
    let output = target.join("macro-dependencies");
    for hidden in [true, false] {
        let mut command = nightly("cargo")?;
        command
            .current_dir(repo)
            .args([
                "rustdoc",
                "--locked",
                "-p",
                "mlx-macros",
                "--lib",
                "--target-dir",
            ])
            .arg(&output)
            .args(["--", "-Z", "unstable-options", "--output-format", "json"]);
        if hidden {
            command.arg("--document-hidden-items");
        }
        checked(&mut command)?;
        fs::copy(
            output.join("doc/mlx_macros.json"),
            output.join(format!("mlx_macros-{hidden}.json")),
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn expand_macro_reexports(path: &Path, dependencies: &Path, hidden: bool) -> Result<(), String> {
    let mut document = read_document(path)?;
    let index = document["index"]
        .as_object()
        .ok_or("missing rustdoc index")?;
    let unresolved: Vec<_> = index
        .values()
        .filter_map(|item| {
            let import = item["inner"].get("use")?;
            let id = import["id"].as_u64()?;
            (!index.contains_key(&id.to_string())).then_some((id, import["is_glob"] == true))
        })
        .collect();
    let mut next = index
        .keys()
        .chain(
            document["paths"]
                .as_object()
                .ok_or("missing rustdoc paths")?
                .keys(),
        )
        .map(|id| id.parse::<u64>().unwrap())
        .max()
        .unwrap_or(0)
        + 1;
    for (id, glob) in unresolved {
        let summary = &document["paths"][id.to_string()];
        let name = summary["path"][0]
            .as_str()
            .ok_or("unresolved public reexport has no crate name")?
            .to_owned();
        if !glob
            || summary["path"].as_array().is_none_or(|p| p.len() != 1)
            || summary["kind"] != "module"
        {
            return Err(format!("unsupported unresolved public reexport: {summary}"));
        }
        let external = read_document(&dependencies.join(format!("{name}-{hidden}.json")))?;
        let mut root = external["index"][external["root"].to_string()].clone();
        let crate_id = summary["crate_id"].clone();
        let mut children = Vec::new();
        for child in root["inner"]["module"]["items"]
            .as_array()
            .ok_or("dependency root is not a module")?
        {
            let mut item = external["index"][child.to_string()].clone();
            if item["inner"].get("proc_macro").is_none() {
                return Err(format!("external reexport {name} contains a non-procedural-macro item; extend the adapter"));
            }
            item["id"] = json!(next);
            item["crate_id"] = crate_id.clone();
            document["index"][next.to_string()] = item;
            children.push(json!(next));
            next += 1;
        }
        root["id"] = json!(id);
        root["crate_id"] = crate_id;
        root["inner"]["module"]["is_crate"] = json!(false);
        root["inner"]["module"]["items"] = json!(children);
        document["index"][id.to_string()] = root;
    }
    fs::write(
        path,
        serde_json::to_vec(&document).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}

fn public_api(path: &Path) -> Result<public_api::PublicApi, String> {
    let api = public_api::Builder::from_rustdoc_json(path)
        .omit_blanket_impls(true)
        .build()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    if let Some(id) = api.missing_item_ids().next() {
        return Err(format!("{}: unresolved rustdoc item {id}", path.display()));
    }
    Ok(api)
}

fn normalize(
    repo: &Path,
    hidden_path: &Path,
    visible_path: &Path,
    feature: &str,
) -> Result<Vec<ApiEntry>, String> {
    let document: Value =
        serde_json::from_slice(&fs::read(hidden_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    if document["format_version"] != 55 {
        return Err("unsupported rustdoc JSON schema (expected 55)".into());
    }
    let index = &document["index"];
    let crate_name = index[document["root"].to_string()]["name"]
        .as_str()
        .ok_or("missing crate name")?;
    let visible_document = read_document(visible_path)?;
    let visible: BTreeSet<_> = public_api(visible_path)?
        .items()
        .map(|item| item_identity(item, &visible_document, crate_name))
        .collect();
    let api = public_api(hidden_path)?;
    let mut type_paths: BTreeMap<u32, BTreeSet<String>> = BTreeMap::new();
    for item in api.items() {
        let inner = &index[item.id().0.to_string()]["inner"];
        if ["struct", "enum", "union", "type_alias"]
            .iter()
            .any(|kind| inner.get(kind).is_some())
        {
            if let Some(path) = item_path(item, crate_name) {
                type_paths.entry(item.id().0).or_default().insert(path);
            }
        }
    }
    let impl_headers: BTreeMap<_, _> = api
        .items()
        .filter(|item| {
            index[item.id().0.to_string()]["inner"]
                .get("impl")
                .is_some()
        })
        .map(|item| (item.id().0, item.to_string()))
        .collect();
    let mut sources = BTreeMap::new();
    let mut result = BTreeSet::new();
    for item in api.items() {
        let raw = &index[item.id().0.to_string()];
        let kind = raw["inner"]
            .as_object()
            .and_then(|o| o.keys().next())
            .ok_or("public item missing from rustdoc index")?;
        let signature = item.to_string();
        let impl_header = item
            .parent_id()
            .and_then(|id| impl_headers.get(&id.0))
            .cloned();
        let mut path = if kind == "impl" {
            format!("<{signature}>")
        } else {
            item_path(item, crate_name)
                .or_else(|| {
                    impl_header
                        .as_ref()
                        .map(|header| format!("<{header}>::{}", raw["name"].as_str().unwrap_or("")))
                })
                .ok_or_else(|| format!("cannot determine public path: {item}"))?
        };
        if kind == "proc_macro" && path.ends_with("::") {
            path.push_str(raw["name"].as_str().ok_or("proc macro has no name")?);
        }
        let parent = item.parent_id().map(|id| &index[id.0.to_string()]);
        let generated = generated(repo, raw, parent, &mut sources)?;
        let mut macro_helpers: Vec<String> = raw["inner"]["proc_macro"]["helpers"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect();
        macro_helpers.sort();
        result.insert(ApiEntry {
            path,
            kind: kind.clone(),
            hidden: !visible.contains(&item_identity(item, &document, crate_name))
                || has_hidden(raw),
            signature,
            impl_header,
            macro_helpers,
            deprecated: serde_json::from_value(raw["deprecation"].clone())
                .map_err(|e| e.to_string())?,
            alias_target: simple_alias_target(raw)
                .and_then(|id| type_paths.get(&id))
                .and_then(|paths| {
                    paths
                        .iter()
                        .min_by_key(|path| (path.matches("::").count(), path.as_str()))
                })
                .cloned(),
            generated,
            feature_sets: vec![feature.into()],
        });
    }
    Ok(result.into_iter().collect())
}

fn simple_alias_target(raw: &Value) -> Option<u32> {
    let alias = raw["inner"].get("type_alias")?;
    let target = &alias["type"]["resolved_path"];
    if !alias["generics"]["params"].as_array()?.is_empty()
        || !alias["generics"]["where_predicates"].as_array()?.is_empty()
        || !target["args"].is_null()
    {
        return None;
    }
    target["id"].as_u64().map(|id| id as u32)
}

fn read_document(path: &Path) -> Result<Value, String> {
    serde_json::from_slice(&fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?)
        .map_err(|e| e.to_string())
}

fn resolve_ids(value: &mut Value, document: &Value) {
    match value {
        Value::Object(object) => {
            if let Some(id) = object.remove("id") {
                let raw = &document["index"][id.to_string()];
                let identity = if raw["span"].is_null() {
                    document["paths"][id.to_string()]["path"].clone()
                } else {
                    json!([raw["name"], raw["span"]])
                };
                object.insert("declaration".into(), identity);
            }
            for value in object.values_mut() {
                resolve_ids(value, document);
            }
        }
        Value::Array(array) => {
            for value in array {
                resolve_ids(value, document);
            }
        }
        _ => {}
    }
}

fn impl_identity(item: &Value, document: &Value) -> Value {
    let mut value = item["inner"]["impl"].clone();
    if let Some(object) = value.as_object_mut() {
        object.remove("items");
        object.remove("provided_trait_methods");
    }
    resolve_ids(&mut value, document);
    value
}

fn item_identity(item: &public_api::PublicItem, document: &Value, root: &str) -> String {
    let raw = &document["index"][item.id().0.to_string()];
    let parent = item
        .parent_id()
        .map(|id| impl_identity(&document["index"][id.0.to_string()], document));
    json!([
        item_path(item, root),
        raw["name"],
        raw["span"],
        raw["inner"].as_object().unwrap().keys().next(),
        impl_identity(raw, document),
        parent
    ])
    .to_string()
}

fn item_path(item: &public_api::PublicItem, root: &str) -> Option<String> {
    let mut path = String::new();
    let mut started = false;
    let mut depth = 0usize;
    for token in item.tokens().skip_while(|token| {
        matches!(
            token,
            Token::Annotation(_) | Token::Qualifier(_) | Token::Kind(_) | Token::Whitespace
        )
    }) {
        if !started {
            if token.text() != root {
                return None;
            }
            started = true;
        }
        match token.text() {
            "<" => {
                depth += 1;
                continue;
            }
            ">" if depth > 0 => {
                depth -= 1;
                continue;
            }
            _ if depth > 0 => continue,
            "::" => path.push_str("::"),
            _ => match token {
                Token::Identifier(s) | Token::Type(s) | Token::Function(s) => {
                    if syn::parse_str::<syn::Ident>(s).is_err()
                        && !s.bytes().all(|b| b.is_ascii_digit())
                    {
                        path.push_str("r#");
                    }
                    path.push_str(s);
                }
                _ => break,
            },
        }
    }
    (!path.is_empty()).then_some(path)
}

fn has_hidden(item: &Value) -> bool {
    item["attrs"].as_array().is_some_and(|attrs| {
        attrs.iter().any(|a| {
            a["other"]
                .as_str()
                .is_some_and(|s| s.contains("doc(hidden)"))
        })
    })
}

#[derive(Default)]
struct Declarations(Vec<(String, String, proc_macro2::LineColumn)>);

impl Declarations {
    fn add(&mut self, kind: &str, name: &syn::Ident) {
        self.0.push((
            kind.into(),
            name.to_string().trim_start_matches("r#").into(),
            name.span().start(),
        ));
    }
}

impl<'ast> Visit<'ast> for Declarations {
    fn visit_item(&mut self, item: &'ast syn::Item) {
        match item {
            syn::Item::Fn(i) => self.add("function", &i.sig.ident),
            syn::Item::Struct(i) => self.add("struct", &i.ident),
            syn::Item::Enum(i) => self.add("enum", &i.ident),
            syn::Item::Union(i) => self.add("union", &i.ident),
            syn::Item::Trait(i) => self.add("trait", &i.ident),
            syn::Item::TraitAlias(i) => self.add("trait_alias", &i.ident),
            syn::Item::Type(i) => self.add("type_alias", &i.ident),
            syn::Item::Mod(i) => self.add("module", &i.ident),
            syn::Item::Const(i) => self.add("constant", &i.ident),
            syn::Item::Static(i) => self.add("static", &i.ident),
            syn::Item::Macro(i) => {
                if let Some(name) = &i.ident {
                    self.add("macro", name);
                }
            }
            syn::Item::Impl(i) => {
                self.0
                    .push(("impl".into(), String::new(), i.impl_token.span.start()))
            }
            _ => {}
        }
        syn::visit::visit_item(self, item);
    }
    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        self.add("function", &item.sig.ident);
    }
    fn visit_trait_item_fn(&mut self, item: &'ast syn::TraitItemFn) {
        self.add("function", &item.sig.ident);
    }
    fn visit_impl_item_const(&mut self, item: &'ast syn::ImplItemConst) {
        self.add("assoc_const", &item.ident);
    }
    fn visit_trait_item_const(&mut self, item: &'ast syn::TraitItemConst) {
        self.add("assoc_const", &item.ident);
    }
    fn visit_impl_item_type(&mut self, item: &'ast syn::ImplItemType) {
        self.add("assoc_type", &item.ident);
    }
    fn visit_trait_item_type(&mut self, item: &'ast syn::TraitItemType) {
        self.add("assoc_type", &item.ident);
    }
    fn visit_variant(&mut self, item: &'ast syn::Variant) {
        self.add("variant", &item.ident);
        syn::visit::visit_variant(self, item);
    }
    fn visit_field(&mut self, item: &'ast syn::Field) {
        if let Some(name) = &item.ident {
            self.add("struct_field", name);
        } else {
            self.0
                .push(("struct_field".into(), String::new(), item.span().start()));
        }
    }
}

fn generated(
    repo: &Path,
    item: &Value,
    parent: Option<&Value>,
    sources: &mut BTreeMap<PathBuf, Declarations>,
) -> Result<bool, String> {
    let derived = |i: &Value| {
        i["attrs"]
            .as_array()
            .is_some_and(|attrs| attrs.contains(&json!("automatically_derived")))
    };
    if derived(item) || parent.is_some_and(derived) || item["inner"]["impl"]["is_synthetic"] == true
    {
        return Ok(true);
    }
    if item["crate_id"] != 0 {
        return Ok(false);
    }
    let kind = item["inner"].as_object().unwrap().keys().next().unwrap();
    if kind == "use" || (kind == "module" && item["inner"]["module"]["is_crate"] == true) {
        return Ok(false);
    }
    let Some(filename) = item["span"]["filename"].as_str() else {
        return Ok(true);
    };
    let path = repo.join(filename);
    if !sources.contains_key(&path) {
        let source = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let syntax = syn::parse_file(&source).map_err(|e| e.to_string())?;
        let mut declarations = Declarations::default();
        declarations.visit_file(&syntax);
        sources.insert(path.clone(), declarations);
    }
    let begin = (
        item["span"]["begin"][0].as_u64().unwrap() as usize,
        item["span"]["begin"][1].as_u64().unwrap() as usize - 1,
    );
    let end = (
        item["span"]["end"][0].as_u64().unwrap() as usize,
        item["span"]["end"][1].as_u64().unwrap() as usize - 1,
    );
    let name = item["name"].as_str().unwrap_or("");
    Ok(!sources[&path].0.iter().any(|(k, n, pos)| {
        k == kind
            && (n == name || (kind == "struct_field" && n.is_empty()))
            && begin <= (pos.line, pos.column)
            && (pos.line, pos.column) < end
    }))
}

fn fixture_macros(repo: &Path, target: &Path) -> Result<(), String> {
    let output = target.join("qualification");
    fs::create_dir_all(&output).map_err(|e| e.to_string())?;
    let source = repo.join("xtask/tests/fixtures/api-inventory-macros.rs");
    checked(
        nightly("rustc")?
            .args([
                "--crate-name",
                "fixture_macros",
                "--crate-type",
                "proc-macro",
                "--edition",
                "2021",
            ])
            .arg(&source)
            .arg("-o")
            .arg(output.join("libfixture_macros.dylib")),
    )?;
    for hidden in [true, false] {
        let mut command = nightly("rustdoc")?;
        command
            .args([
                "--crate-name",
                "fixture_macros",
                "--crate-type",
                "proc-macro",
                "--edition",
                "2021",
                "-Z",
                "unstable-options",
                "--output-format",
                "json",
                "-o",
            ])
            .arg(&output)
            .arg(&source);
        if hidden {
            command.arg("--document-hidden-items");
        }
        checked(&mut command)?;
        fs::copy(
            output.join("fixture_macros.json"),
            output.join(format!("fixture_macros-{hidden}.json")),
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn fixture(
    repo: &Path,
    target: &Path,
    extra: bool,
    omit: Option<&str>,
) -> Result<Vec<ApiEntry>, String> {
    let output = target.join("qualification");
    fs::create_dir_all(&output).map_err(|e| e.to_string())?;
    let hidden_path = output.join("hidden.json");
    let path = output.join("fixture.json");
    for hidden in [true, false] {
        let mut command = nightly("rustdoc")?;
        command
            .current_dir(repo)
            .args([
                "--crate-name",
                "fixture",
                "--crate-type",
                "lib",
                "--edition",
                "2021",
                "-Z",
                "unstable-options",
                "--output-format",
                "json",
                "-o",
            ])
            .arg(&output)
            .arg(repo.join("xtask/tests/fixtures/api-inventory.rs"));
        command.arg("--extern").arg(format!(
            "fixture_macros={}",
            output.join("libfixture_macros.dylib").display()
        ));
        if hidden {
            command.arg("--document-hidden-items");
        }
        if extra {
            command.args(["--cfg", "feature=\"extra\""]);
        }
        if let Some(omit) = omit {
            command.args(["--cfg", omit]);
        }
        checked(&mut command)?;
        if hidden {
            fs::copy(&path, &hidden_path).map_err(|e| e.to_string())?;
        }
    }
    expand_macro_reexports(&hidden_path, &output, true)?;
    expand_macro_reexports(&path, &output, false)?;
    normalize(
        repo,
        &hidden_path,
        &path,
        if extra { "extra" } else { "base" },
    )
}

fn check_fixture(entries: &[ApiEntry], extra: bool) -> Result<(), String> {
    if entries.iter().any(|e| {
        e.signature.contains("core::borrow::Borrow")
            || e.signature.contains("core::any::Any")
            || e.signature.contains("alloc::borrow::ToOwned")
            || e.signature.contains("core::convert::From<T> for T")
    }) {
        return Err("qualification retained blanket impls".into());
    }
    for (name, trait_name) in [("SendProbe", "Send"), ("SyncProbe", "Sync")] {
        let signature = format!("impl core::marker::{trait_name} for fixture::{name}");
        if !entries
            .iter()
            .any(|e| e.signature == signature && e.generated)
        {
            return Err(format!("qualification changed auto trait: {signature}"));
        }
    }
    if !entries.iter().any(|e| {
        e.path == "fixture::WidgetAlias" && e.alias_target.as_deref() == Some("fixture::Widget")
    }) {
        return Err("qualification lost local alias target".into());
    }
    if entries
        .iter()
        .any(|e| e.path == "fixture::SpecializedAlias" && e.alias_target.is_some())
    {
        return Err("qualification expanded a specialized alias without checking bounds".into());
    }
    for (receiver, hidden) in [("visible_name", false), ("hidden_name", true)] {
        let signature = format!("impl core::marker::Send for fixture::{receiver}::Same");
        if !entries
            .iter()
            .any(|e| e.signature == signature && e.hidden == hidden)
        {
            return Err(format!("qualification misidentified {signature}"));
        }
    }
    if !entries.iter().any(|e| e.path == "fixture::r#type") {
        return Err("qualification lost raw identifier spelling".into());
    }
    if !entries.iter().any(|e| {
        e.path == "fixture::ExternalDerive" && e.kind == "proc_macro" && !e.hidden && !e.generated
    }) {
        return Err("qualification missing external derive reexport".into());
    }
    for path in ["fixture::visible_alias", "fixture::VisibleTuple"] {
        if !entries.iter().any(|e| e.path == path && !e.hidden) {
            return Err(format!("qualification incorrectly hid {path}"));
        }
    }
    for (path, kind, signature, generated, hidden) in [
        (
            "fixture::WidgetBuilder::build",
            "function",
            "pub fn fixture::WidgetBuilder::build(self) -> fixture::Widget",
            true,
            false,
        ),
        (
            "fixture::Renamed",
            "struct",
            "pub struct fixture::Renamed",
            false,
            false,
        ),
        (
            "fixture::hidden",
            "function",
            "pub fn fixture::hidden()",
            false,
            true,
        ),
        (
            "fixture::HiddenAlias",
            "struct",
            "pub struct fixture::HiddenAlias",
            false,
            true,
        ),
    ] {
        if !entries.iter().any(|entry| {
            entry.path == path
                && entry.kind == kind
                && entry.signature == signature
                && entry.generated == generated
                && entry.hidden == hidden
                && entry.feature_sets == [if extra { "extra" } else { "base" }]
        }) {
            return Err(format!("qualification missing or misreported {path}"));
        }
    }
    let gated = entries.iter().find(|e| e.path == "fixture::gated");
    if gated.is_some() != extra
        || gated.is_some_and(|entry| {
            entry.kind != "function"
                || entry.signature != "pub fn fixture::gated()"
                || entry.generated
                || entry.hidden
                || entry.feature_sets != ["extra"]
        })
    {
        return Err("qualification misreported cfg-gated item".into());
    }
    if !entries.iter().any(|e| {
        e.path == "<impl<T: fixture::Inspect> fixture::Inspect for &mut T>::inspect"
            && !e.generated
            && e.impl_header.as_deref()
                == Some("impl<T: fixture::Inspect> fixture::Inspect for &mut T")
    }) {
        return Err("qualification lost generic receiver impl context".into());
    }
    if entries
        .iter()
        .any(|e| e.path.contains("private_function") || e.path.contains("::private::"))
    {
        return Err("qualification exposed a private path".into());
    }
    if !entries
        .iter()
        .any(|e| e.path == "fixture::Widget::clone" && e.generated)
    {
        return Err("qualification missing derived Clone::clone".into());
    }
    Ok(())
}

fn stable_consumer(repo: &Path, target: &Path) -> Result<(), String> {
    let output = target.join("stable-consumer");
    fs::create_dir_all(&output).map_err(|e| e.to_string())?;
    let macro_lib = output.join("libfixture_macros.dylib");
    checked(
        Command::new("rustc")
            .args([
                "--crate-name",
                "fixture_macros",
                "--crate-type",
                "proc-macro",
                "--edition",
                "2021",
            ])
            .arg(repo.join("xtask/tests/fixtures/api-inventory-macros.rs"))
            .arg("-o")
            .arg(&macro_lib),
    )?;
    let probe = output.join("probe.rs");
    fs::write(
        &probe,
        r#"
        use fixture::{ExternalDerive, Renamed, WidgetBuilder};
        #[derive(ExternalDerive)]
        pub struct Probe;
        pub fn check() {
            let _ = WidgetBuilder.build().clone();
            let _ = Renamed;
            fixture::hidden();
            #[cfg(feature = "extra")]
            fixture::gated();
        }
    "#,
    )
    .map_err(|e| e.to_string())?;
    for extra in [false, true] {
        let mut library = Command::new("rustc");
        library
            .args([
                "--crate-name",
                "fixture",
                "--crate-type",
                "lib",
                "--edition",
                "2021",
            ])
            .arg(repo.join("xtask/tests/fixtures/api-inventory.rs"))
            .arg("--extern")
            .arg(format!("fixture_macros={}", macro_lib.display()))
            .arg("-o")
            .arg(output.join("libfixture.rlib"));
        if extra {
            library.args(["--cfg", "feature=\"extra\""]);
        }
        checked(&mut library)?;
        let mut consumer = Command::new("rustc");
        consumer
            .args([
                "--crate-name",
                "probe",
                "--crate-type",
                "lib",
                "--edition",
                "2021",
                "--emit",
                "metadata",
            ])
            .arg(&probe)
            .arg("--extern")
            .arg(format!(
                "fixture={}",
                output.join("libfixture.rlib").display()
            ))
            .arg("-L")
            .arg(format!("dependency={}", output.display()))
            .arg("-o")
            .arg(output.join("probe.rmeta"));
        if extra {
            consumer.args(["--cfg", "feature=\"extra\""]);
        }
        checked(&mut consumer)?;
    }
    eprintln!("api-baseline: stable consumers passed for base and extra");
    Ok(())
}

fn qualify(repo: &Path, target: &Path) -> Result<(), String> {
    fixture_macros(repo, target)?;
    let base = fixture(repo, target, false, None)?;
    check_fixture(&base, false)?;
    let extra = fixture(repo, target, true, None)?;
    check_fixture(&extra, true)?;
    for path in [
        "fixture::WidgetBuilder::build",
        "fixture::Renamed",
        "fixture::hidden",
        "fixture::gated",
    ] {
        for field in [
            "path",
            "kind",
            "signature",
            "generated",
            "hidden",
            "feature_sets",
        ] {
            let mut mutated = extra.clone();
            let entry = mutated.iter_mut().find(|e| e.path == path).unwrap();
            match field {
                "path" => entry.path.push_str("_wrong"),
                "kind" => entry.kind = "wrong".into(),
                "signature" => entry.signature.push_str(" -> bool"),
                "generated" => entry.generated = !entry.generated,
                "hidden" => entry.hidden = !entry.hidden,
                "feature_sets" => entry.feature_sets = vec!["base".into()],
                _ => unreachable!(),
            }
            if check_fixture(&mutated, true).is_ok() {
                return Err(format!("qualification accepted wrong {field} for {path}"));
            }
        }
    }
    eprintln!("api-baseline: qualification rejected 24 misreported fields");
    for omit in [
        "omit_builder",
        "omit_reexport",
        "omit_hidden",
        "omit_gated",
        "omit_external",
        "change_send",
        "change_sync",
    ] {
        let mutated = fixture(repo, target, true, Some(omit))?;
        if check_fixture(&mutated, true).is_ok() {
            return Err(format!("qualification accepted {omit}"));
        }
        eprintln!("api-baseline: qualification rejected {omit}");
    }
    stable_consumer(repo, target)
}

fn serialize(baseline: &ApiBaseline) -> Result<Vec<u8>, String> {
    let mut value = serde_json::to_value(baseline).map_err(|e| e.to_string())?;
    value.as_object_mut().unwrap().remove("entries");
    let mut text = serde_json::to_string_pretty(&value).map_err(|e| e.to_string())?;
    text.truncate(text.trim_end().len() - 1);
    text.truncate(text.trim_end().len());
    text.push_str(",\n  \"entries\": [\n");
    for (i, entry) in baseline.entries.iter().enumerate() {
        text.push_str("    ");
        text.push_str(&serde_json::to_string(entry).map_err(|e| e.to_string())?);
        if i + 1 < baseline.entries.len() {
            text.push(',');
        }
        text.push('\n');
    }
    text.push_str("  ]\n}\n");
    Ok(text.into_bytes())
}

pub fn run(repo: &Path, args: &[String]) -> i32 {
    let result = (|| {
        if args == ["--qualify"] {
            return qualify(repo, &target_dir(repo));
        }
        if !args.is_empty() && (args.len() != 2 || !matches!(args[0].as_str(), "--out" | "--check"))
        {
            return Err(
                "usage: cargo xtask api-baseline [--out PATH | --check PATH | --qualify]".into(),
            );
        }
        let baseline = generate(repo)?;
        let bytes = serialize(&baseline)?;
        if args.is_empty() {
            io::stdout().write_all(&bytes).map_err(|e| e.to_string())?;
        } else if args[0] == "--out" {
            fs::write(&args[1], bytes).map_err(|e| e.to_string())?;
        } else {
            let committed = fs::read(&args[1]).map_err(|e| e.to_string())?;
            if committed != bytes {
                return Err(format!("compiled API baseline differs: {}", args[1]));
            }
            println!("compiled API baseline matches: {}", args[1]);
        }
        Ok(())
    })();
    match result {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("api-baseline: {error}");
            2
        }
    }
}
