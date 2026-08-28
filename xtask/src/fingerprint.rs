use crate::bindgen_config;
use quote::ToTokens;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use syn::{Fields, FnArg, Item, ReturnType, UseTree};

const SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct ToolVersions {
    bindgen: String,
    rustc: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct FingerprintContent {
    schema_version: u32,
    tool_versions: ToolVersions,
    source_commit: String,
    headers: Vec<String>,
    entries: Vec<Entry>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct Fingerprint {
    #[serde(flatten)]
    content: FingerprintContent,
    overall_digest: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct FieldLayout {
    name: String,
    #[serde(rename = "type")]
    ty: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct EnumVariant {
    name: String,
    value: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Entry {
    Function {
        name: String,
        signature: String,
    },
    Type {
        name: String,
        definition: String,
    },
    Struct {
        name: String,
        fields: Vec<FieldLayout>,
    },
    Enum {
        name: String,
        repr: String,
        variants: Vec<EnumVariant>,
    },
    Constant {
        name: String,
        #[serde(rename = "type")]
        ty: String,
        value: String,
    },
    Handle {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        new: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        free: Option<String>,
    },
}

impl Entry {
    fn name(&self) -> &str {
        match self {
            Self::Function { name, .. }
            | Self::Type { name, .. }
            | Self::Struct { name, .. }
            | Self::Enum { name, .. }
            | Self::Constant { name, .. }
            | Self::Handle { name, .. } => name,
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Self::Function { .. } => "function",
            Self::Type { .. } => "type",
            Self::Struct { .. } => "struct",
            Self::Enum { .. } => "enum",
            Self::Constant { .. } => "constant",
            Self::Handle { .. } => "handle",
        }
    }

    fn sort_key(&self) -> (u8, &str) {
        let rank = match self {
            Self::Function { .. } => 0,
            Self::Type { .. } => 1,
            Self::Struct { .. } => 2,
            Self::Enum { .. } => 3,
            Self::Constant { .. } => 4,
            Self::Handle { .. } => 5,
        };
        (rank, self.name())
    }
}

#[derive(Debug, Serialize)]
struct FingerprintRef {
    source_commit: String,
    overall_digest: String,
}

#[derive(Debug, Serialize)]
struct ChangedEntry {
    kind: String,
    name: String,
    before: Entry,
    after: Entry,
}

#[derive(Debug, Serialize)]
struct DeltaCounts {
    added: usize,
    removed: usize,
    changed: usize,
}

#[derive(Debug, Serialize)]
struct Delta {
    schema_version: u32,
    old: FingerprintRef,
    new: FingerprintRef,
    counts: DeltaCounts,
    added: Vec<Entry>,
    removed: Vec<Entry>,
    changed: Vec<ChangedEntry>,
}

struct GitWorktree {
    repository: PathBuf,
    path: PathBuf,
    _tempdir: tempfile::TempDir,
}

impl GitWorktree {
    fn create(repository: &Path, reference: &str) -> Result<Self, String> {
        let tempdir = tempfile::Builder::new()
            .prefix("mlx-c-fingerprint-")
            .tempdir()
            .map_err(|error| format!("failed to create temporary directory: {error}"))?;
        let isolated_repository = tempdir.path().join("repository");
        run_git(
            repository,
            [
                "clone",
                "--quiet",
                "--shared",
                "--no-checkout",
                repository.to_str().ok_or("repository path is not UTF-8")?,
                isolated_repository
                    .to_str()
                    .ok_or("temporary repository path is not UTF-8")?,
            ],
        )?;
        let path = tempdir.path().join("mlx-c");
        run_git(
            &isolated_repository,
            [
                "worktree",
                "add",
                "--quiet",
                "--detach",
                path.to_str().ok_or("temporary path is not UTF-8")?,
                reference,
            ],
        )?;
        Ok(Self {
            repository: isolated_repository,
            path,
            _tempdir: tempdir,
        })
    }
}

impl Drop for GitWorktree {
    fn drop(&mut self) {
        let _ = Command::new("git")
            .args(["worktree", "remove", "--force"])
            .arg(&self.path)
            .current_dir(&self.repository)
            .status();
    }
}

pub fn run_fingerprint(repo_root: &Path, args: &[String]) -> i32 {
    match parse_fingerprint_args(args).and_then(|(reference, output)| {
        let fingerprint = generate(repo_root, &reference)?;
        write_fingerprint(&fingerprint, output.as_deref())
    }) {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("fingerprint: {error}");
            2
        }
    }
}

pub fn run_delta(args: &[String]) -> i32 {
    match parse_delta_args(args).and_then(|(old, new)| {
        let old = read_fingerprint(&old)?;
        let new = read_fingerprint(&new)?;
        let mut output = serde_json::to_vec_pretty(&build_delta(&old, &new))
            .map_err(|error| format!("failed to serialize delta: {error}"))?;
        output.push(b'\n');
        io::stdout()
            .write_all(&output)
            .map_err(|error| format!("failed to write delta: {error}"))
    }) {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("fingerprint-delta: {error}");
            2
        }
    }
}

pub(crate) fn delta_value(old: &Path, new: &Path) -> Result<serde_json::Value, String> {
    let old = read_fingerprint(old)?;
    let new = read_fingerprint(new)?;
    serde_json::to_value(build_delta(&old, &new))
        .map_err(|error| format!("failed to serialize delta: {error}"))
}

fn parse_fingerprint_args(args: &[String]) -> Result<(String, Option<PathBuf>), String> {
    let mut reference = None;
    let mut output = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--ref" if reference.is_none() => {
                index += 1;
                reference = Some(args.get(index).cloned().ok_or_else(fingerprint_usage)?);
            }
            "--out" if output.is_none() => {
                index += 1;
                output = Some(PathBuf::from(
                    args.get(index).ok_or_else(fingerprint_usage)?,
                ));
            }
            _ => return Err(fingerprint_usage()),
        }
        index += 1;
    }
    reference
        .map(|reference| (reference, output))
        .ok_or_else(fingerprint_usage)
}

fn parse_delta_args(args: &[String]) -> Result<(PathBuf, PathBuf), String> {
    let mut old = None;
    let mut new = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--old" if old.is_none() => {
                index += 1;
                old = Some(PathBuf::from(args.get(index).ok_or_else(delta_usage)?));
            }
            "--new" if new.is_none() => {
                index += 1;
                new = Some(PathBuf::from(args.get(index).ok_or_else(delta_usage)?));
            }
            _ => return Err(delta_usage()),
        }
        index += 1;
    }
    old.zip(new).ok_or_else(delta_usage)
}

fn fingerprint_usage() -> String {
    "usage: cargo run -p xtask -- fingerprint --ref <commit> [--out <path>]".to_owned()
}

fn delta_usage() -> String {
    "usage: cargo run -p xtask -- fingerprint-delta --old <file> --new <file>".to_owned()
}

fn generate(repo_root: &Path, reference: &str) -> Result<Fingerprint, String> {
    let repository = repo_root.join("mlx-sys/src/mlx-c");
    let worktree = GitWorktree::create(&repository, reference)?;
    let source_commit = run_git(&worktree.path, ["rev-parse", "HEAD"])?;
    let header_paths = bindgen_config::discover_headers(&worktree.path)
        .map_err(|error| format!("failed to discover headers: {error}"))?;
    let headers = header_paths
        .iter()
        .map(|header| bindgen_config::relative_header(&worktree.path, header))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("failed to record headers: {error}"))?;
    let bindings = bindgen_config::builder(&worktree.path, &header_paths)
        .generate()
        .map_err(|error| format!("bindgen failed: {error}"))?
        .to_string();
    let bindgen = bindings
        .lines()
        .next()
        .and_then(|line| line.split("rust-bindgen ").nth(1))
        .and_then(|version| version.strip_suffix(" */"))
        .ok_or("bindgen output did not contain its version")?
        .to_owned();
    let rustc = command_output(Command::new("rustc").arg("--version"), "rustc --version")?;
    let content = FingerprintContent {
        schema_version: SCHEMA_VERSION,
        tool_versions: ToolVersions { bindgen, rustc },
        source_commit,
        headers,
        entries: parse_entries(&bindings)?,
    };
    let overall_digest = digest_content(&content)?;
    Ok(Fingerprint {
        content,
        overall_digest,
    })
}

fn write_fingerprint(fingerprint: &Fingerprint, output: Option<&Path>) -> Result<(), String> {
    let mut bytes = serde_json::to_vec_pretty(fingerprint)
        .map_err(|error| format!("failed to serialize fingerprint: {error}"))?;
    bytes.push(b'\n');
    if let Some(path) = output {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)
                .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
        }
        fs::write(path, bytes)
            .map_err(|error| format!("failed to write {}: {error}", path.display()))
    } else {
        io::stdout()
            .write_all(&bytes)
            .map_err(|error| format!("failed to write fingerprint: {error}"))
    }
}

fn read_fingerprint(path: &Path) -> Result<Fingerprint, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let fingerprint: Fingerprint = serde_json::from_slice(&bytes)
        .map_err(|error| format!("failed to parse {}: {error}", path.display()))?;
    let actual = digest_content(&fingerprint.content)?;
    if actual != fingerprint.overall_digest {
        return Err(format!(
            "{} has digest {}, expected {}",
            path.display(),
            fingerprint.overall_digest,
            actual
        ));
    }
    Ok(fingerprint)
}

fn digest_content(content: &FingerprintContent) -> Result<String, String> {
    let bytes = serde_json::to_vec(content)
        .map_err(|error| format!("failed to serialize fingerprint content: {error}"))?;
    let mut child = Command::new("shasum")
        .args(["-a", "256"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|error| format!("failed to run shasum: {error}"))?;
    child
        .stdin
        .take()
        .ok_or("failed to open shasum stdin")?
        .write_all(&bytes)
        .map_err(|error| format!("failed to hash fingerprint: {error}"))?;
    let output = child
        .wait_with_output()
        .map_err(|error| format!("failed to wait for shasum: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "shasum failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let digest = String::from_utf8(output.stdout)
        .map_err(|error| format!("shasum returned invalid UTF-8: {error}"))?
        .split_whitespace()
        .next()
        .ok_or("shasum returned no digest")?
        .to_owned();
    Ok(format!("sha256:{digest}"))
}

fn parse_entries(bindings: &str) -> Result<Vec<Entry>, String> {
    let file =
        syn::parse_file(bindings).map_err(|error| format!("failed to parse bindings: {error}"))?;
    let mut entries = Vec::new();
    let mut functions = BTreeSet::new();
    let mut aliases = BTreeMap::new();
    let mut structs = BTreeMap::new();
    let mut constants = Vec::new();

    for item in &file.items {
        match item {
            Item::ForeignMod(foreign) => {
                for item in &foreign.items {
                    if let syn::ForeignItem::Fn(function) = item {
                        let name = function.sig.ident.to_string();
                        if is_function_name(&name) {
                            functions.insert(name.clone());
                            entries.push(Entry::Function {
                                name,
                                signature: canonical_signature(&function.sig),
                            });
                        }
                    }
                }
            }
            Item::Type(item) => {
                let name = item.ident.to_string();
                if is_type_name(&name) {
                    let definition = canonical_tokens(&item.ty);
                    aliases.insert(name.clone(), definition.clone());
                    entries.push(Entry::Type { name, definition });
                }
            }
            Item::Use(item) => {
                let mut renames = Vec::new();
                collect_use_renames(&item.tree, Vec::new(), &mut renames);
                for (name, definition) in renames {
                    if is_type_name(&name) {
                        aliases.insert(name.clone(), definition.clone());
                        entries.push(Entry::Type { name, definition });
                    }
                }
            }
            Item::Struct(item) => {
                let name = item.ident.to_string();
                if is_type_name(&name) {
                    let fields = canonical_fields(&item.fields);
                    structs.insert(name.clone(), fields.clone());
                    entries.push(Entry::Struct { name, fields });
                }
            }
            Item::Union(item) => {
                let name = item.ident.to_string();
                if is_type_name(&name) {
                    let fields: Vec<FieldLayout> = item
                        .fields
                        .named
                        .iter()
                        .map(|field| FieldLayout {
                            name: field.ident.as_ref().expect("named union field").to_string(),
                            ty: canonical_tokens(&field.ty),
                        })
                        .collect();
                    structs.insert(name.clone(), fields.clone());
                    entries.push(Entry::Struct { name, fields });
                }
            }
            Item::Enum(item) => {
                let name = item.ident.to_string();
                if is_type_name(&name) {
                    let mut next_value = 0_i128;
                    let variants = item
                        .variants
                        .iter()
                        .map(|variant| {
                            let value = variant
                                .discriminant
                                .as_ref()
                                .map(|(_, value)| canonical_tokens(value))
                                .unwrap_or_else(|| next_value.to_string());
                            if let Ok(value) = value.parse::<i128>() {
                                next_value = value + 1;
                            }
                            EnumVariant {
                                name: variant.ident.to_string(),
                                value,
                            }
                        })
                        .collect();
                    entries.push(Entry::Enum {
                        name,
                        repr: "rust-enum".to_owned(),
                        variants,
                    });
                }
            }
            Item::Const(item) => {
                let name = item.ident.to_string();
                if is_constant_name(&name) {
                    constants.push((
                        name,
                        canonical_tokens(&item.ty),
                        canonical_tokens(&item.expr),
                    ));
                }
            }
            _ => {}
        }
    }

    let mut enum_groups: BTreeMap<String, (String, Vec<EnumVariant>)> = BTreeMap::new();
    for (name, ty, value) in constants {
        let enum_parts = name
            .split_once("__")
            .filter(|(prefix, _)| ty == format!("{prefix}_"));
        if let Some((prefix, variant)) = enum_parts {
            enum_groups
                .entry(prefix.to_owned())
                .or_insert_with(|| (ty, Vec::new()))
                .1
                .push(EnumVariant {
                    name: variant.to_owned(),
                    value,
                });
        } else {
            entries.push(Entry::Constant { name, ty, value });
        }
    }
    for (name, (repr, variants)) in enum_groups {
        entries.push(Entry::Enum {
            name,
            repr,
            variants,
        });
    }

    for (name, target) in aliases {
        let opaque_name = format!("{name}_");
        if target == opaque_name
            && structs.get(&opaque_name).is_some_and(|fields| {
                fields.len() == 1 && fields[0].name == "ctx" && fields[0].ty.starts_with("* mut ")
            })
        {
            let new_name = format!("{name}_new");
            let free_name = format!("{name}_free");
            entries.push(Entry::Handle {
                name,
                new: functions.contains(&new_name).then_some(new_name),
                free: functions.contains(&free_name).then_some(free_name),
            });
        }
    }

    entries.sort_by(|left, right| left.sort_key().cmp(&right.sort_key()));
    Ok(entries)
}

fn canonical_signature(signature: &syn::Signature) -> String {
    let mut arguments = signature
        .inputs
        .iter()
        .map(|argument| match argument {
            FnArg::Typed(argument) => canonical_tokens(&argument.ty),
            FnArg::Receiver(_) => "self".to_owned(),
        })
        .collect::<Vec<_>>();
    if signature.variadic.is_some() {
        arguments.push("...".to_owned());
    }
    let output = match &signature.output {
        ReturnType::Default => "()".to_owned(),
        ReturnType::Type(_, ty) => canonical_tokens(ty),
    };
    format!("fn({})->{output}", arguments.join(","))
}

fn canonical_fields(fields: &Fields) -> Vec<FieldLayout> {
    fields
        .iter()
        .enumerate()
        .map(|(index, field)| FieldLayout {
            name: field
                .ident
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_else(|| index.to_string()),
            ty: canonical_tokens(&field.ty),
        })
        .collect()
}

fn canonical_tokens(tokens: &impl ToTokens) -> String {
    tokens
        .to_token_stream()
        .to_string()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn collect_use_renames(tree: &UseTree, prefix: Vec<String>, output: &mut Vec<(String, String)>) {
    match tree {
        UseTree::Path(path) => {
            let mut prefix = prefix;
            prefix.push(path.ident.to_string());
            collect_use_renames(&path.tree, prefix, output);
        }
        UseTree::Rename(rename) => {
            let mut target = prefix;
            target.push(rename.ident.to_string());
            let target = target
                .into_iter()
                .filter(|component| component != "self")
                .collect::<Vec<_>>()
                .join("::");
            output.push((rename.rename.to_string(), target));
        }
        UseTree::Group(group) => {
            for tree in &group.items {
                collect_use_renames(tree, prefix.clone(), output);
            }
        }
        _ => {}
    }
}

fn is_function_name(name: &str) -> bool {
    name.starts_with("mlx_") || name.starts_with("_mlx_")
}

fn is_type_name(name: &str) -> bool {
    name.starts_with("mlx_") || matches!(name, "float16_t" | "bfloat16_t")
}

fn is_constant_name(name: &str) -> bool {
    name.starts_with("mlx_")
}

fn build_delta(old: &Fingerprint, new: &Fingerprint) -> Delta {
    let old_entries = old
        .content
        .entries
        .iter()
        .cloned()
        .map(|entry| ((entry.kind(), entry.name().to_owned()), entry))
        .collect::<BTreeMap<_, _>>();
    let new_entries = new
        .content
        .entries
        .iter()
        .cloned()
        .map(|entry| ((entry.kind(), entry.name().to_owned()), entry))
        .collect::<BTreeMap<_, _>>();
    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut changed = Vec::new();

    for (key, entry) in &old_entries {
        match new_entries.get(key) {
            None => removed.push(entry.clone()),
            Some(after) if after != entry => changed.push(ChangedEntry {
                kind: key.0.to_owned(),
                name: key.1.clone(),
                before: entry.clone(),
                after: after.clone(),
            }),
            Some(_) => {}
        }
    }
    for (key, entry) in &new_entries {
        if !old_entries.contains_key(key) {
            added.push(entry.clone());
        }
    }
    Delta {
        schema_version: SCHEMA_VERSION,
        old: FingerprintRef {
            source_commit: old.content.source_commit.clone(),
            overall_digest: old.overall_digest.clone(),
        },
        new: FingerprintRef {
            source_commit: new.content.source_commit.clone(),
            overall_digest: new.overall_digest.clone(),
        },
        counts: DeltaCounts {
            added: added.len(),
            removed: removed.len(),
            changed: changed.len(),
        },
        added,
        removed,
        changed,
    }
}

fn run_git<const N: usize>(directory: &Path, args: [&str; N]) -> Result<String, String> {
    command_output(Command::new("git").args(args).current_dir(directory), "git")
}

fn command_output(command: &mut Command, description: &str) -> Result<String, String> {
    let output = command
        .output()
        .map_err(|error| format!("failed to run {description}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{description} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout)
        .map(|output| output.trim().to_owned())
        .map_err(|error| format!("{description} returned invalid UTF-8: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fingerprint(entries: Vec<Entry>) -> Fingerprint {
        let content = FingerprintContent {
            schema_version: SCHEMA_VERSION,
            tool_versions: ToolVersions {
                bindgen: "test".to_owned(),
                rustc: "test".to_owned(),
            },
            source_commit: "test".to_owned(),
            headers: vec!["mlx/c/mlx.h".to_owned()],
            entries,
        };
        Fingerprint {
            content,
            overall_digest: "test".to_owned(),
        }
    }

    #[test]
    fn signature_canonicalization_ignores_formatting_noise() {
        let compact = r#"
            extern "C" { pub fn mlx_add(x:*const mlx_array,y:mlx_array)->::std::os::raw::c_int; }
        "#;
        let expanded = r#"
            extern "C" {
                pub fn mlx_add(
                    first: *const mlx_array,
                    second: mlx_array,
                ) -> ::std::os::raw::c_int;
            }
        "#;
        assert_eq!(
            parse_entries(compact).unwrap(),
            parse_entries(expanded).unwrap()
        );
    }

    #[test]
    fn delta_reports_add_remove_and_change() {
        let old = fingerprint(vec![
            Entry::Function {
                name: "mlx_changed".to_owned(),
                signature: "fn(i32)->i32".to_owned(),
            },
            Entry::Constant {
                name: "mlx_removed".to_owned(),
                ty: "u32".to_owned(),
                value: "1".to_owned(),
            },
        ]);
        let new = fingerprint(vec![
            Entry::Function {
                name: "mlx_changed".to_owned(),
                signature: "fn(i64)->i32".to_owned(),
            },
            Entry::Type {
                name: "mlx_added".to_owned(),
                definition: "u64".to_owned(),
            },
        ]);
        let delta = build_delta(&old, &new);
        assert_eq!(delta.counts.added, 1);
        assert_eq!(delta.counts.removed, 1);
        assert_eq!(delta.counts.changed, 1);
        assert_eq!(delta.added[0].name(), "mlx_added");
        assert_eq!(delta.removed[0].name(), "mlx_removed");
        assert_eq!(delta.changed[0].name, "mlx_changed");
    }
}
