use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    process::Command,
};

pub fn run(repo_root: &Path, args: &[String]) -> i32 {
    match verify(repo_root, args) {
        Ok(count) => {
            println!(
                "{}",
                json!({"schema_version": 1, "verdict": "pass", "entries": count})
            );
            0
        }
        Err(error) => {
            println!(
                "{}",
                json!({"schema_version": 1, "verdict": "fail", "errors": [error]})
            );
            1
        }
    }
}

fn verify(repo_root: &Path, args: &[String]) -> Result<usize, String> {
    let mut ledger = repo_root.join("ledger/mlx-lm-parity.json");
    let mut package = None;
    let mut index = 0;
    while index < args.len() {
        let flag = &args[index];
        let value = args.get(index + 1).ok_or_else(usage)?;
        match flag.as_str() {
            "--ledger" => ledger = PathBuf::from(value),
            "--package" => package = Some(PathBuf::from(value)),
            _ => return Err(usage()),
        }
        index += 2;
    }
    let committed: Value = serde_json::from_slice(
        &fs::read(&ledger).map_err(|e| format!("{}: {e}", ledger.display()))?,
    )
    .map_err(|e| e.to_string())?;
    let mut extractor = Command::new("python3");
    extractor
        .arg(repo_root.join("conformance/mlx-lm/ledger_extract.py"))
        .arg("--names");
    if let Some(package) = package {
        extractor.arg("--package").arg(package);
    }
    let output = extractor
        .output()
        .map_err(|e| format!("cannot run ledger extractor: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "ledger extractor failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let discovered: Value = serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("invalid extractor output: {e}"))?;
    let api = serde_json::to_value(crate::api_baseline::generate(
        &repo_root.join("mlx-lm"),
        "mlx_lm",
    )?)
    .map_err(|e| e.to_string())?;
    let paths = api["entries"]
        .as_array()
        .ok_or("missing Rust API entries")?
        .iter()
        .filter_map(|entry| entry["path"].as_str().map(str::to_owned))
        .collect();
    validate(&committed, &discovered, &paths)
}

fn usage() -> String {
    "usage: cargo run -p xtask -- verify-lm-parity [--ledger <file>] [--package <mlx_lm source directory>]".into()
}

fn entries(value: &Value) -> Result<BTreeMap<&str, &Value>, String> {
    let mut result = BTreeMap::new();
    for entry in value["entries"]
        .as_array()
        .ok_or("ledger entries must be an array")?
    {
        let name = nonempty(entry, "name")?;
        if result.insert(name, entry).is_some() {
            return Err(format!("duplicate surface entry {name}"));
        }
    }
    if result.is_empty() {
        return Err("empty Python surface inventory".into());
    }
    Ok(result)
}

fn nonempty<'a>(entry: &'a Value, field: &str) -> Result<&'a str, String> {
    entry[field]
        .as_str()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("{} requires {field}", entry["name"]))
}

fn validate(
    committed: &Value,
    discovered: &Value,
    rust_paths: &BTreeSet<String>,
) -> Result<usize, String> {
    for value in [committed, discovered] {
        if value["schema_version"] != 1 || value["upstream_version"] != "0.31.3" {
            return Err("expected schema 1 and mlx_lm 0.31.3".into());
        }
    }
    for field in [
        "scope",
        "inventory_rule",
        "source_files",
        "upstream_commit",
        "source_kind",
    ] {
        if committed.get(field).is_none() || committed[field] != discovered[field] {
            return Err(format!(
                "ledger {field} differs from pinned source extraction"
            ));
        }
    }
    let expected = entries(discovered)?;
    let actual = entries(committed)?;
    let missing: Vec<_> = expected
        .keys()
        .filter(|name| !actual.contains_key(*name))
        .collect();
    let stale: Vec<_> = actual
        .keys()
        .filter(|name| !expected.contains_key(*name))
        .collect();
    if !missing.is_empty() || !stale.is_empty() {
        return Err(format!(
            "Python surface drift: missing={missing:?}, stale={stale:?}"
        ));
    }
    for (name, entry) in &actual {
        for field in ["kind", "signature", "source_path", "source_sha256"] {
            if entry[field] != expected[name][field] {
                return Err(format!("{name}: {field} differs from source"));
            }
        }
        let disposition = nonempty(entry, "disposition")?;
        if !matches!(
            disposition,
            "planned_wrapped"
                | "wrapped"
                | "internal"
                | "deferred"
                | "intentionally_unexposed"
                | "skipped"
                | "blocked"
        ) {
            return Err(format!("{name}: unknown disposition {disposition}"));
        }
        nonempty(entry, "trigger")?;
        if entry.get("rust_path").is_none() {
            return Err(format!("{name}: missing rust_path field"));
        }
        if !entry["rust_path"].is_null() && entry["rust_path"].as_str().is_none() {
            return Err(format!("{name}: rust_path must be a string or null"));
        }
        if !entry["semantic_differences"].is_array() || !entry["evidence"].is_array() {
            return Err(format!("{name}: missing semantic differences or evidence"));
        }
        if disposition == "planned_wrapped" {
            if !entry["rust_path"].is_null() {
                return Err(format!("{name}: use wrapped for an implemented Rust path"));
            }
            nonempty(entry, "planned_rust_path")?;
        }
        if disposition == "wrapped" {
            let path = nonempty(entry, "rust_path")?;
            if !rust_paths.contains(path) {
                return Err(format!(
                    "{name}: wrapped path {path} is absent from Rust source"
                ));
            }
        }
    }
    Ok(actual.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn documents() -> (Value, Value) {
        let discovered = json!({"schema_version": 1, "upstream_version": "0.31.3", "scope": ["__init__"], "inventory_rule": "test", "upstream_commit": "fixture", "source_kind": "installed_package_ast", "source_files": {}, "entries": [{"name": "mlx_lm.load", "kind": "function", "signature": "(path)", "source_path": "mlx_lm/utils.py", "source_sha256": "fixture"}]});
        let mut committed = discovered.clone();
        let entry = &mut committed["entries"][0];
        entry["disposition"] = json!("planned_wrapped");
        entry["rust_path"] = Value::Null;
        entry["planned_rust_path"] = json!("mlx_lm::Model::from_dir");
        entry["trigger"] = json!("Tranche 2 local loading parity.");
        entry["semantic_differences"] = json!([]);
        entry["evidence"] = json!([]);
        (committed, discovered)
    }

    #[test]
    fn ledger_rejects_missing_surfaces_and_dispositions() {
        let (committed, discovered) = documents();
        assert_eq!(
            validate(&committed, &discovered, &BTreeSet::new()).unwrap(),
            1
        );
        let mut omitted = committed.clone();
        omitted["entries"] = json!([]);
        assert!(validate(&omitted, &discovered, &BTreeSet::new()).is_err());
        let mut missing = committed.clone();
        missing["entries"][0]
            .as_object_mut()
            .unwrap()
            .remove("disposition");
        assert!(validate(&missing, &discovered, &BTreeSet::new()).is_err());
        let mut drift = committed.clone();
        drift["entries"][0]["signature"] = json!("(changed)");
        assert!(validate(&drift, &discovered, &BTreeSet::new()).is_err());
        let mut duplicate = committed.clone();
        duplicate["entries"]
            .as_array_mut()
            .unwrap()
            .push(committed["entries"][0].clone());
        assert!(validate(&duplicate, &discovered, &BTreeSet::new()).is_err());
    }

    #[test]
    fn wrapped_entries_require_existing_rust_paths() {
        let (mut committed, discovered) = documents();
        committed["entries"][0]["disposition"] = json!("wrapped");
        assert!(validate(&committed, &discovered, &BTreeSet::new()).is_err());
        committed["entries"][0]["rust_path"] = json!("mlx_lm::Model::from_dir");
        assert!(validate(&committed, &discovered, &BTreeSet::new()).is_err());
        assert!(validate(
            &committed,
            &discovered,
            &BTreeSet::from(["mlx_lm::Model::from_dir".into()])
        )
        .is_ok());
    }
}
