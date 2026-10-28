use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use syn::Item;

const SCHEMA_VERSION: u64 = 1;

pub fn run(repo_root: &Path, args: &[String]) -> i32 {
    let result = verify_value(repo_root, args);
    match result {
        Ok(report) => {
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
            0
        }
        Err(error) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "schema_version": SCHEMA_VERSION,
                    "verdict": "fail",
                    "errors": [error]
                }))
                .unwrap()
            );
            1
        }
    }
}

pub(crate) fn verify_value(repo_root: &Path, args: &[String]) -> Result<Value, String> {
    parse_args(repo_root, args).and_then(|paths| verify_files(repo_root, &paths))
}

struct Paths {
    classification: PathBuf,
    api_baseline: PathBuf,
    feature_matrix: PathBuf,
    corpus: PathBuf,
}

fn parse_args(repo_root: &Path, args: &[String]) -> Result<Paths, String> {
    let mut paths = Paths {
        classification: repo_root.join("ledger/target-delta-classification.json"),
        api_baseline: repo_root.join("ledger/rust-api-baseline.json"),
        feature_matrix: repo_root.join("ledger/supported-feature-matrix.json"),
        corpus: repo_root.join("conformance/corpus.json"),
    };
    let mut index = 0;
    while index < args.len() {
        let target = match args[index].as_str() {
            "--classification" => &mut paths.classification,
            "--api-baseline" => &mut paths.api_baseline,
            "--feature-matrix" => &mut paths.feature_matrix,
            "--corpus" => &mut paths.corpus,
            _ => return Err(usage()),
        };
        index += 1;
        *target = PathBuf::from(args.get(index).ok_or_else(usage)?);
        index += 1;
    }
    Ok(paths)
}

fn usage() -> String {
    "usage: cargo xtask verify-ledger [--classification <file>] [--api-baseline <file>] [--feature-matrix <file>] [--corpus <file>]".to_owned()
}

fn verify_files(repo_root: &Path, paths: &Paths) -> Result<Value, String> {
    let classification = read_json(&paths.classification)?;
    let (delta, abi) = crate::fingerprint::generate_delta(
        repo_root,
        required_string(&classification, "old_source_commit")?,
        required_string(&classification, "target_source_commit")?,
    )?;
    let api_baseline = read_json(&paths.api_baseline)?;
    let compiled_api_baseline = serde_json::to_value(crate::api_baseline::generate(repo_root)?)
        .map_err(|error| format!("failed to serialize compiled API baseline: {error}"))?;
    let feature_matrix = read_json(&paths.feature_matrix)?;
    let conf_ids = conformance_ids(&paths.corpus)?;
    let api_paths = api_paths(&api_baseline)?;
    let result = verify_documents(
        &delta,
        &classification,
        &api_baseline,
        &compiled_api_baseline,
        &feature_matrix,
        |id| evidence_exists(repo_root, id, &conf_ids, &api_paths, paths),
    )?;
    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "verdict": "pass",
        "delta_counts": delta["counts"],
        "abi": abi,
        "classified_entries": result.classified,
        "behavioral_renames": result.behavioral_renames,
        "dispositions": result.dispositions,
        "supported_builds": result.supported_builds
    }))
}

struct Verification {
    classified: usize,
    behavioral_renames: usize,
    dispositions: BTreeMap<String, usize>,
    supported_builds: usize,
}

fn verify_documents(
    delta: &Value,
    classification: &Value,
    api_baseline: &Value,
    compiled_api_baseline: &Value,
    feature_matrix: &Value,
    mut evidence_exists: impl FnMut(&str) -> bool,
) -> Result<Verification, String> {
    require_schema(delta, "delta")?;
    require_schema(classification, "classification")?;
    if api_baseline["schema_version"] != 2 || compiled_api_baseline["schema_version"] != 2 {
        return Err("API baseline requires compiled schema version 2".into());
    }
    require_schema(feature_matrix, "feature matrix")?;
    if api_baseline != compiled_api_baseline {
        return Err("committed API baseline does not match compiled inventory".to_owned());
    }
    validate_api_references(classification, api_baseline)?;
    validate_api_references(feature_matrix, api_baseline)?;
    let api_paths = api_paths(api_baseline)?;
    let behavioral_renames = validate_behavioral_renames(classification, &api_paths)?;
    let supported_builds = validate_feature_matrix(feature_matrix)?;
    let expected = delta_entries(delta)?;
    let entries = classification["entries"]
        .as_array()
        .ok_or("classification entries must be an array")?;
    if classification["old_source_commit"] != delta["old"]["source_commit"] {
        return Err("classification old_source_commit does not match delta".to_owned());
    }
    if classification["target_source_commit"] != delta["new"]["source_commit"] {
        return Err("classification target_source_commit does not match delta".to_owned());
    }

    let mut seen = BTreeSet::new();
    let mut dispositions = BTreeMap::new();
    for classified in entries {
        let change = required_string(classified, "change")?;
        let kind = required_string(classified, "kind")?;
        let name = required_string(classified, "name")?;
        let key = format!("{change}:{kind}:{name}");
        if !seen.insert(key.clone()) {
            return Err(format!("duplicate classification {key}"));
        }
        let Some(actual) = expected.get(&key) else {
            return Err(format!("classification {key} is not in delta"));
        };
        if !classification_matches(classified, actual, change) {
            return Err(format!("classification {key} does not match delta"));
        }

        let disposition = required_string(classified, "disposition")?;
        if !matches!(
            disposition,
            "wrapped" | "internal" | "deferred" | "intentionally_unexposed" | "removed" | "blocked"
        ) {
            return Err(format!(
                "classification {key} has invalid disposition {disposition}"
            ));
        }
        *dispositions.entry(disposition.to_owned()).or_insert(0) += 1;

        if change == "changed" {
            if classified["old_pin_affected"].as_bool() != Some(false) {
                return Err(format!(
                    "classification {key} must record old_pin_affected as false"
                ));
            }
            required_nonempty(classified, "bump_impact")?;
        }
        let handle = if change == "changed" {
            &actual["after"]
        } else {
            actual
        };
        if kind == "handle" && (handle.get("new").is_some() || handle.get("free").is_some()) {
            validate_ownership(classified, handle, disposition, &key)?;
        }
        if disposition == "wrapped" {
            let rust_path = required_nonempty(classified, "rust_path")?;
            if classified.get("unsupported_cfg").is_none() && !api_paths.contains(rust_path) {
                return Err(format!(
                    "wrapped path {rust_path} is absent from API baseline"
                ));
            }
            if rust_path.starts_with("mlx_rs::fft::") {
                let python_name = required_nonempty(classified, "python_name")?;
                if !python_name.starts_with("mlx.core.fft.") {
                    return Err(format!(
                        "FFT classification {key} has invalid python_name {python_name}"
                    ));
                }
                required_nonempty(classified, "semantic_op")?;
            }
            classified["evidence"]
                .as_array()
                .filter(|evidence| !evidence.is_empty())
                .ok_or_else(|| format!("wrapped classification {key} requires evidence"))?;
        } else if matches!(
            disposition,
            "internal" | "deferred" | "intentionally_unexposed" | "removed" | "blocked"
        ) {
            required_one_line(classified, "rationale")?;
        }
        if let Some(evidence) = classified.get("evidence").and_then(Value::as_array) {
            for evidence_id in evidence {
                let evidence_id = evidence_id
                    .as_str()
                    .ok_or_else(|| format!("classification {key} has non-string evidence"))?;
                if !evidence_exists(evidence_id) {
                    return Err(format!("unresolved evidence {evidence_id} for {key}"));
                }
            }
        }
    }

    let missing = expected
        .keys()
        .filter(|key| !seen.contains(*key))
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "unclassified delta entries: {}",
            missing.join(", ")
        ));
    }
    Ok(Verification {
        classified: seen.len(),
        behavioral_renames,
        dispositions,
        supported_builds,
    })
}

fn validate_behavioral_renames(
    classification: &Value,
    api_paths: &BTreeSet<String>,
) -> Result<usize, String> {
    let renames = classification["behavioral_renames"]
        .as_array()
        .filter(|renames| !renames.is_empty())
        .ok_or("classification behavioral_renames must be a non-empty array")?;
    let mut c_names = BTreeSet::new();
    let mut rust_paths = BTreeSet::new();
    for rename in renames {
        let c_name = required_one_line(rename, "c_name")?;
        let python_name = required_one_line(rename, "python_name")?;
        let rust_path = required_one_line(rename, "rust_path")?;
        required_one_line(rename, "semantic_op")?;
        if !c_name.starts_with("mlx_") {
            return Err(format!("behavioral rename has invalid C name {c_name}"));
        }
        if !python_name.starts_with("mlx.core.") {
            return Err(format!(
                "behavioral rename {c_name} has invalid python_name {python_name}"
            ));
        }
        if !api_paths.contains(rust_path) {
            return Err(format!(
                "behavioral rename Rust path {rust_path} is absent from API baseline"
            ));
        }
        if !c_names.insert(c_name) {
            return Err(format!("duplicate behavioral rename C name {c_name}"));
        }
        if !rust_paths.insert(rust_path) {
            return Err(format!("duplicate behavioral rename Rust path {rust_path}"));
        }
    }
    Ok(renames.len())
}

fn validate_ownership(
    classified: &Value,
    handle: &Value,
    disposition: &str,
    key: &str,
) -> Result<(), String> {
    let ownership = classified["ownership"]
        .as_object()
        .ok_or_else(|| format!("classification {key} requires ownership metadata"))?;
    if classified["risk_class"] != "ownership"
        || ownership.get("model").and_then(Value::as_str) != Some("owned_handle")
        || ownership.get("constructor") != handle.get("new")
        || ownership.get("destructor") != handle.get("free")
        || ownership.get("rust_status").and_then(Value::as_str) != Some(disposition)
    {
        return Err(format!(
            "classification {key} has invalid ownership metadata"
        ));
    }
    Ok(())
}

fn require_schema(value: &Value, description: &str) -> Result<(), String> {
    if value["schema_version"].as_u64() != Some(SCHEMA_VERSION) {
        return Err(format!(
            "{description} schema_version must be {SCHEMA_VERSION}"
        ));
    }
    Ok(())
}

fn delta_entries(delta: &Value) -> Result<BTreeMap<String, Value>, String> {
    let mut entries = BTreeMap::new();
    for change in ["added", "removed"] {
        let values = delta[change]
            .as_array()
            .ok_or_else(|| format!("delta {change} must be an array"))?;
        validate_count(delta, change, values.len())?;
        for entry in values {
            let kind = required_string(entry, "kind")?;
            let name = required_string(entry, "name")?;
            entries.insert(format!("{change}:{kind}:{name}"), entry.clone());
        }
    }
    let changed = delta["changed"]
        .as_array()
        .ok_or("delta changed must be an array")?;
    validate_count(delta, "changed", changed.len())?;
    for entry in changed {
        let kind = required_string(entry, "kind")?;
        let name = required_string(entry, "name")?;
        entries.insert(format!("changed:{kind}:{name}"), entry.clone());
    }
    Ok(entries)
}

fn validate_count(delta: &Value, change: &str, actual: usize) -> Result<(), String> {
    if delta["counts"][change].as_u64() != Some(actual as u64) {
        return Err(format!("delta count for {change} does not match entries"));
    }
    Ok(())
}

fn classification_matches(classified: &Value, actual: &Value, change: &str) -> bool {
    if change == "changed" {
        classified["before"] == actual["before"] && classified["after"] == actual["after"]
    } else {
        classified["entry"] == *actual
    }
}

fn required_string<'a>(value: &'a Value, field: &str) -> Result<&'a str, String> {
    value[field]
        .as_str()
        .ok_or_else(|| format!("classification field {field} must be a string"))
}

fn required_nonempty<'a>(value: &'a Value, field: &str) -> Result<&'a str, String> {
    let result = required_string(value, field)?;
    if result.trim().is_empty() {
        Err(format!("classification field {field} must not be empty"))
    } else {
        Ok(result)
    }
}

fn required_one_line<'a>(value: &'a Value, field: &str) -> Result<&'a str, String> {
    let result = required_nonempty(value, field)?;
    if result.contains(['\r', '\n']) {
        Err(format!("classification field {field} must be one line"))
    } else {
        Ok(result)
    }
}

fn validate_api_references(document: &Value, inventory: &Value) -> Result<(), String> {
    let supported = inventory["feature_sets"]
        .as_array()
        .filter(|sets| !sets.is_empty())
        .ok_or("compiled inventory has no feature sets")?
        .iter()
        .map(|set| required_nonempty(set, "id").map(str::to_owned))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let availability = api_availability(inventory, &supported)?;
    visit_api_references(
        document,
        &supported,
        &supported,
        &availability,
        inventory["target"].as_str().unwrap_or(""),
    )
}

fn api_availability(
    inventory: &Value,
    supported: &BTreeSet<String>,
) -> Result<BTreeMap<String, BTreeSet<String>>, String> {
    let mut availability: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for entry in inventory["entries"]
        .as_array()
        .ok_or("inventory has no entries")?
    {
        let path = required_nonempty(entry, "path")?;
        let sets = reference_sets(entry, "feature_sets", supported)?;
        availability.entry(path.into()).or_default().extend(sets);
    }
    let aliases: Vec<_> = inventory["entries"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|entry| entry["alias_target"].as_str().map(|target| (entry, target)))
        .collect();
    for _ in 0..aliases.len() {
        let mut additions: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for (entry, target) in &aliases {
            let alias = required_nonempty(entry, "path")?;
            let sets = reference_sets(entry, "feature_sets", supported)?;
            let prefix = format!("{target}::");
            for (path, target_sets) in &availability {
                if let Some(member) = path.strip_prefix(&prefix) {
                    additions
                        .entry(format!("{alias}::{member}"))
                        .or_default()
                        .extend(sets.intersection(target_sets).cloned());
                }
            }
        }
        let mut changed = false;
        for (path, sets) in additions {
            let existing = availability.entry(path).or_default();
            let before = existing.len();
            existing.extend(sets);
            changed |= before != existing.len();
        }
        if !changed {
            break;
        }
    }
    Ok(availability)
}

fn reference_sets(
    value: &Value,
    field: &str,
    supported: &BTreeSet<String>,
) -> Result<BTreeSet<String>, String> {
    let sets = value[field]
        .as_array()
        .filter(|sets| !sets.is_empty())
        .ok_or_else(|| format!("{field} must be a nonempty array"))?;
    let mut result = BTreeSet::new();
    for set in sets {
        let id = set.as_str().ok_or("API feature set must be a string")?;
        if !supported.contains(id) || !result.insert(id.to_owned()) {
            return Err(format!("unknown or duplicate API feature set {id}"));
        }
    }
    Ok(result)
}

fn visit_api_references(
    value: &Value,
    inherited: &BTreeSet<String>,
    supported: &BTreeSet<String>,
    availability: &BTreeMap<String, BTreeSet<String>>,
    target: &str,
) -> Result<(), String> {
    match value {
        Value::Object(object) => {
            if object.contains_key("unsupported_cfg") {
                validate_unsupported_cfg(value, availability, target)?;
            }
            let claims = if object.contains_key("api_feature_sets") {
                reference_sets(value, "api_feature_sets", supported)?
            } else {
                inherited.clone()
            };
            for (field, child) in object {
                if field == "rust_path" && object.contains_key("unsupported_cfg") {
                    continue;
                }
                if matches!(field.as_str(), "rust_path" | "canonical_path" | "symbols") {
                    check_path_values(child, &claims, availability)?;
                }
                visit_api_references(child, &claims, supported, availability, target)?;
            }
        }
        Value::Array(values) => {
            for child in values {
                visit_api_references(child, inherited, supported, availability, target)?;
            }
        }
        Value::String(text) => {
            if let Some(path) = text.strip_prefix("api:path:") {
                check_available(path, inherited, availability)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_unsupported_cfg(
    value: &Value,
    availability: &BTreeMap<String, BTreeSet<String>>,
    target: &str,
) -> Result<(), String> {
    let path = required_nonempty(value, "rust_path")?;
    let citation = required_nonempty(value, "unsupported_cfg")?;
    required_one_line(value, "rationale")?;
    if value["disposition"] != "wrapped"
        || !target.ends_with("-apple-darwin")
        || path != "mlx_rs::ops::qqmm"
        || citation != "api:cfg:mlx-rs/src/ops/quantization.rs#qqmm#not(target_os = \"macos\")"
        || !value["evidence"]
            .as_array()
            .is_some_and(|ids| ids.iter().any(|id| id == citation))
    {
        return Err(format!("unsupported_cfg for {path} requires the canonical qqmm cfg citation, wrapped disposition, and a supported macOS target"));
    }
    if availability.get(path).is_some_and(|sets| !sets.is_empty()) {
        return Err(format!(
            "unsupported_cfg path {path} appears in supported configurations"
        ));
    }
    Ok(())
}

fn check_path_values(
    value: &Value,
    claims: &BTreeSet<String>,
    availability: &BTreeMap<String, BTreeSet<String>>,
) -> Result<(), String> {
    match value {
        Value::String(text) => {
            for path in text.split(" and ") {
                check_available(path, claims, availability)?;
            }
        }
        Value::Object(values) => {
            for child in values.values() {
                check_path_values(child, claims, availability)?;
            }
        }
        Value::Array(values) => {
            for child in values {
                check_path_values(child, claims, availability)?;
            }
        }
        _ => return Err("Rust path reference must be text or a collection of paths".into()),
    }
    Ok(())
}

fn check_available(
    path: &str,
    claims: &BTreeSet<String>,
    availability: &BTreeMap<String, BTreeSet<String>>,
) -> Result<(), String> {
    for set in claims {
        if !availability
            .get(path)
            .is_some_and(|sets| sets.contains(set))
        {
            return Err(format!(
                "Rust path {path} is absent from compiled inventory for {set}"
            ));
        }
    }
    Ok(())
}

fn api_paths(inventory: &Value) -> Result<BTreeSet<String>, String> {
    let supported = inventory["feature_sets"]
        .as_array()
        .ok_or("inventory has no feature sets")?
        .iter()
        .map(|set| required_nonempty(set, "id").map(str::to_owned))
        .collect::<Result<BTreeSet<_>, _>>()?;
    Ok(api_availability(inventory, &supported)?
        .into_iter()
        .filter(|(_, sets)| !sets.is_empty())
        .map(|(path, _)| path)
        .collect())
}

fn validate_feature_matrix(matrix: &Value) -> Result<usize, String> {
    let builds = matrix["builds"]
        .as_array()
        .filter(|builds| !builds.is_empty())
        .ok_or("feature matrix builds must be a non-empty array")?;
    let mut ids = BTreeSet::new();
    for build in builds {
        let id = build["id"]
            .as_str()
            .filter(|value| !value.is_empty())
            .ok_or("feature matrix build id must be non-empty")?;
        if !ids.insert(id) {
            return Err(format!("duplicate feature matrix build {id}"));
        }
        for field in ["target_triple", "purpose"] {
            build[field]
                .as_str()
                .filter(|value| !value.is_empty())
                .ok_or_else(|| format!("feature matrix build {id} requires {field}"))?;
        }
        if build["default_features"].as_bool().is_none() {
            return Err(format!(
                "feature matrix build {id} requires default_features"
            ));
        }
        for field in ["features", "verification"] {
            let values = build[field]
                .as_array()
                .filter(|values| !values.is_empty())
                .ok_or_else(|| format!("feature matrix build {id} requires {field}"))?;
            let strings = values
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .ok_or_else(|| format!("{field} must contain strings"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if !strings.windows(2).all(|pair| pair[0] < pair[1]) {
                return Err(format!(
                    "feature matrix build {id} {field} must be sorted and unique"
                ));
            }
        }
    }
    crate::config_report::configurations(matrix)?;
    Ok(builds.len())
}

fn read_json(path: &Path) -> Result<Value, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("failed to parse {}: {error}", path.display()))
}

fn conformance_ids(corpus_path: &Path) -> Result<BTreeSet<String>, String> {
    let corpus = read_json(corpus_path)?;
    let root = corpus_path.parent().ok_or("corpus path has no parent")?;
    let suites = corpus["suites"]
        .as_array()
        .ok_or("corpus suites must be an array")?;
    let mut ids = BTreeSet::new();
    for suite in suites {
        let suite = suite.as_str().ok_or("corpus suite path must be a string")?;
        let manifest = read_json(&root.join(suite))?;
        for case in manifest["cases"]
            .as_array()
            .ok_or_else(|| format!("corpus suite {suite} cases must be an array"))?
        {
            ids.insert(
                case["id"]
                    .as_str()
                    .ok_or_else(|| format!("corpus suite {suite} case id must be a string"))?
                    .to_owned(),
            );
        }
    }
    Ok(ids)
}

fn evidence_exists(
    repo_root: &Path,
    id: &str,
    conf_ids: &BTreeSet<String>,
    api_paths: &BTreeSet<String>,
    paths: &Paths,
) -> bool {
    if let Some(case_id) = id.strip_prefix("conf:") {
        return conf_ids.contains(case_id);
    }
    if let Some(reference) = id.strip_prefix("api:cfg:") {
        return cfg_evidence_exists(repo_root, reference);
    }
    if let Some(path) = id.strip_prefix("api:path:") {
        return api_paths.contains(path);
    }
    if let Some(report) = id.strip_prefix("api:report:") {
        return match report {
            "rust-api-baseline" => paths.api_baseline.is_file(),
            "supported-feature-matrix" => paths.feature_matrix.is_file(),
            _ => report_exists(repo_root, "api", report),
        };
    }
    for class in ["ffi", "state", "thread", "api"] {
        if let Some(reference) = id.strip_prefix(&format!("{class}:test:")) {
            return test_exists(repo_root, reference);
        }
        if let Some(reference) = id.strip_prefix(&format!("{class}:report:")) {
            return report_exists(repo_root, class, reference);
        }
    }
    false
}

fn cfg_evidence_exists(repo_root: &Path, reference: &str) -> bool {
    let mut parts = reference.split('#');
    let (Some(path), Some(name), Some(predicate), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return false;
    };
    if predicate != "not(target_os = \"macos\")" {
        return false;
    }
    let Some(path) = safe_repo_path(repo_root, path) else {
        return false;
    };
    let Ok(source) = fs::read_to_string(path) else {
        return false;
    };
    let Ok(file) = syn::parse_file(&source) else {
        return false;
    };
    let expected: syn::Meta = syn::parse_quote!(cfg(not(target_os = "macos")));
    file.items.iter().any(|item| match item {
        Item::Fn(function) => {
            function.sig.ident == name
                && matches!(function.vis, syn::Visibility::Public(_))
                && function.attrs.iter().any(|attr| {
                    quote::ToTokens::to_token_stream(&attr.meta).to_string()
                        == quote::ToTokens::to_token_stream(&expected).to_string()
                })
        }
        _ => false,
    })
}

fn test_exists(repo_root: &Path, reference: &str) -> bool {
    let Some((path, name)) = reference.split_once('#') else {
        return false;
    };
    let Some(path) = safe_repo_path(repo_root, path) else {
        return false;
    };
    let Ok(source) = fs::read_to_string(path) else {
        return false;
    };
    let Ok(file) = syn::parse_file(&source) else {
        return false;
    };
    contains_test(&file.items, name)
}

fn contains_test(items: &[Item], name: &str) -> bool {
    items.iter().any(|item| match item {
        Item::Fn(function) => {
            function.sig.ident == name
                && function
                    .attrs
                    .iter()
                    .any(|attr| attr.path().is_ident("test"))
        }
        Item::Mod(module) => module
            .content
            .as_ref()
            .is_some_and(|(_, items)| contains_test(items, name)),
        _ => false,
    })
}

fn report_exists(repo_root: &Path, class: &str, reference: &str) -> bool {
    let Some((path, id)) = reference.split_once('#') else {
        return false;
    };
    let Some(path) = safe_repo_path(repo_root, path) else {
        return false;
    };
    read_json(&path).is_ok_and(|report| {
        report["schema_version"].as_u64() == Some(SCHEMA_VERSION)
            && report["report_id"].as_str() == Some(id)
            && report["evidence_class"].as_str() == Some(class)
            && report["source_commit"]
                .as_str()
                .is_some_and(|commit| !commit.is_empty())
            && report["verdict"].as_str() == Some("pass")
    })
}

fn safe_repo_path(repo_root: &Path, relative: &str) -> Option<PathBuf> {
    let relative = Path::new(relative);
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return None;
    }
    Some(repo_root.join(relative))
}

#[cfg(test)]
fn verify_fixture(fixture: &Value, valid_evidence: &BTreeSet<String>) -> Result<(), String> {
    verify_fixture_with_compiled(fixture, &fixture["api_baseline"], valid_evidence)
}

#[cfg(test)]
fn verify_fixture_with_compiled(
    fixture: &Value,
    compiled_api_baseline: &Value,
    valid_evidence: &BTreeSet<String>,
) -> Result<(), String> {
    verify_documents(
        &fixture["delta"],
        &fixture["classification"],
        &fixture["api_baseline"],
        compiled_api_baseline,
        &fixture["feature_matrix"],
        |id| valid_evidence.contains(id),
    )
    .map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::{api_paths, validate_api_references};
    use serde_json::Value;
    use std::collections::BTreeSet;

    fn fixture() -> Value {
        serde_json::from_str(include_str!("../tests/fixtures/ledger-qualification.json")).unwrap()
    }

    fn valid_evidence() -> BTreeSet<String> {
        ["api:path:demo::added", "conf:add.basic"]
            .into_iter()
            .map(str::to_owned)
            .collect()
    }

    fn verify_fixture(fixture: &Value) -> Result<(), String> {
        super::verify_fixture(fixture, &valid_evidence())
    }

    #[test]
    fn unsupported_wrapper_requires_cfg_evidence_and_absence_from_every_set() {
        let mut fixture = fixture();
        let entry = &mut fixture["classification"]["entries"][3];
        entry["disposition"] = "wrapped".into();
        entry["rust_path"] = "mlx_rs::ops::qqmm".into();
        entry["rationale"] = "only available outside supported macOS targets".into();
        let citation = "api:cfg:mlx-rs/src/ops/quantization.rs#qqmm#not(target_os = \"macos\")";
        entry["unsupported_cfg"] = citation.into();
        entry["evidence"] = serde_json::json!([citation]);
        fixture["api_baseline"]["target"] = "aarch64-apple-darwin".into();
        let mut evidence = valid_evidence();
        evidence.insert(citation.into());
        super::verify_fixture(&fixture, &evidence).unwrap();

        let mut missing = fixture.clone();
        missing["classification"]["entries"][3]["evidence"] = serde_json::json!([]);
        assert!(super::verify_fixture(&missing, &evidence).is_err());
        assert!(super::verify_fixture(&fixture, &valid_evidence()).is_err());
        let mut plain = fixture.clone();
        plain["classification"]["entries"][3]
            .as_object_mut()
            .unwrap()
            .remove("unsupported_cfg");
        assert!(super::verify_fixture(&plain, &evidence)
            .unwrap_err()
            .contains("mlx_rs::ops::qqmm"));
        for set in ["default", "cpu-only"] {
            let mut present = fixture.clone();
            present["api_baseline"]["entries"]
                .as_array_mut()
                .unwrap()
                .push(serde_json::json!({
                    "path": "mlx_rs::ops::qqmm", "feature_sets": [set]
                }));
            assert!(super::verify_fixture(&present, &evidence)
                .unwrap_err()
                .contains("appears in supported"));
        }
        for (field, value) in [
            ("rust_path", "mlx_rs::wrong::qqmm"),
            (
                "unsupported_cfg",
                "api:cfg:demo/src/lib.rs#another#not(target_os = \"macos\")",
            ),
            (
                "unsupported_cfg",
                "api:cfg:demo/src/lib.rs#changed#feature = \"other\"",
            ),
            ("disposition", "deferred"),
            ("rationale", ""),
        ] {
            let mut invalid = fixture.clone();
            invalid["classification"]["entries"][3][field] = value.into();
            assert!(super::verify_fixture(&invalid, &evidence).is_err());
        }
        fixture["api_baseline"]["target"] = "x86_64-unknown-linux-gnu".into();
        assert!(super::verify_fixture(&fixture, &evidence).is_err());
    }

    #[test]
    fn cfg_evidence_checks_the_named_public_function_attribute() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("lib.rs");
        let reference = "lib.rs#qqmm#not(target_os = \"macos\")";
        for (source, expected) in [
            ("#[cfg(not(target_os = \"macos\"))] pub fn qqmm() {}", true),
            ("#[cfg(target_os = \"macos\")] pub fn qqmm() {}", false),
            (
                "#[cfg(not(target_os = \"macos\"))] pub fn other() {}",
                false,
            ),
            ("#[cfg(not(target_os = \"macos\"))] fn qqmm() {}", false),
            ("pub fn qqmm() {}", false),
        ] {
            std::fs::write(&path, source).unwrap();
            assert_eq!(
                super::cfg_evidence_exists(root.path(), reference),
                expected,
                "{source}"
            );
        }
        assert!(!super::cfg_evidence_exists(
            root.path(),
            "../lib.rs#qqmm#not(target_os = \"macos\")"
        ));
    }

    #[test]
    fn local_alias_members_require_alias_and_target_availability() {
        let inventory = serde_json::json!({
            "feature_sets": [{"id":"default"}, {"id":"cpu-only"}],
            "entries": [
                {"path":"crate::Alias", "alias_target":"crate::Original", "feature_sets":["default"]},
                {"path":"crate::Original::method", "feature_sets":["default","cpu-only"]}
            ]
        });
        let reference =
            serde_json::json!({"rust_path":"crate::Alias::method", "api_feature_sets":["default"]});
        assert!(validate_api_references(&reference, &inventory).is_ok());
        assert!(api_paths(&inventory)
            .unwrap()
            .contains("crate::Alias::method"));
        let reference = serde_json::json!({"rust_path":"crate::Alias::method"});
        assert!(validate_api_references(&reference, &inventory)
            .unwrap_err()
            .contains("cpu-only"));
        let reference = serde_json::json!({"rust_path":"crate::Alias::missing", "api_feature_sets":["default"]});
        assert!(validate_api_references(&reference, &inventory).is_err());
    }

    #[test]
    fn qualification_rejects_a_stale_api_signature() {
        let mut fixture = fixture();
        let source_api = fixture["api_baseline"].clone();
        fixture["api_baseline"]["entries"][0]["signature"] = "pub fn added()->i64".into();
        assert!(
            super::verify_fixture_with_compiled(&fixture, &source_api, &valid_evidence())
                .unwrap_err()
                .contains("does not match compiled inventory")
        );
    }

    #[test]
    fn qualification_rejects_a_path_missing_from_a_claimed_feature_set() {
        let mut fixture = fixture();
        fixture["api_baseline"]["feature_sets"] = serde_json::json!([
            {"id": "default"}, {"id": "cpu-only"}
        ]);
        fixture["api_baseline"]["entries"][0]["feature_sets"] = serde_json::json!(["default"]);
        assert!(verify_fixture(&fixture).unwrap_err().contains("cpu-only"));
    }

    #[test]
    fn qualification_accepts_explicit_availability_and_rejects_unknown_sets() {
        let mut fixture = fixture();
        fixture["api_baseline"]["entries"][0]["feature_sets"] = serde_json::json!(["default"]);
        fixture["classification"]["api_feature_sets"] = serde_json::json!(["default"]);
        verify_fixture(&fixture).unwrap();
        fixture["classification"]["api_feature_sets"] = serde_json::json!(["typo"]);
        assert!(verify_fixture(&fixture).unwrap_err().contains("unknown"));
    }

    #[test]
    fn qualification_checks_admission_symbols_and_path_evidence() {
        let mut fixture = fixture();
        fixture["feature_matrix"]["admissions"] = serde_json::json!([{
            "canonical_path": "demo::added", "symbols": {"mlx_added": "demo::missing"}
        }]);
        assert!(verify_fixture(&fixture)
            .unwrap_err()
            .contains("demo::missing"));
        fixture["feature_matrix"]["admissions"][0]["symbols"]["mlx_added"] = "demo::added".into();
        verify_fixture(&fixture).unwrap();
        fixture["feature_matrix"]["admissions"][0]["symbols"]["mlx_added"] = "missing".into();
        assert!(verify_fixture(&fixture).unwrap_err().contains("missing"));
        fixture["feature_matrix"]["admissions"][0]["symbols"]["mlx_added"] = "demo::added".into();
        fixture["classification"]["entries"][0]["evidence"][0] = "api:path:demo::missing".into();
        assert!(verify_fixture(&fixture)
            .unwrap_err()
            .contains("demo::missing"));
    }

    #[test]
    fn qualification_rejects_a_fabricated_api_path() {
        let mut fixture = fixture();
        let source_api = fixture["api_baseline"].clone();
        fixture["api_baseline"]["entries"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({
                "kind": "function",
                "path": "demo::missing",
                "signature": "pub fn missing()"
            }));
        fixture["classification"]["entries"][0]["rust_path"] = "demo::missing".into();
        fixture["classification"]["entries"][0]["evidence"][0] = "api:path:demo::missing".into();
        assert!(
            super::verify_fixture_with_compiled(&fixture, &source_api, &valid_evidence())
                .unwrap_err()
                .contains("does not match compiled inventory")
        );
    }

    #[test]
    fn qualification_requires_semantic_metadata_for_fft_pilot_entries() {
        let mut fixture = fixture();
        fixture["classification"]["entries"][0]["rust_path"] = "mlx_rs::fft::fft".into();
        fixture["classification"]["entries"][0]["evidence"][0] = "api:path:mlx_rs::fft::fft".into();
        fixture["classification"]["behavioral_renames"][0]["rust_path"] = "mlx_rs::fft::fft".into();
        fixture["api_baseline"]["entries"][0]["path"] = "mlx_rs::fft::fft".into();
        let evidence = ["api:path:mlx_rs::fft::fft", "conf:add.basic"]
            .into_iter()
            .map(str::to_owned)
            .collect();

        let error = super::verify_fixture(&fixture, &evidence).unwrap_err();
        assert!(error.contains("python_name") || error.contains("semantic_op"));
    }

    #[test]
    fn qualification_rejects_missing_configuration_metadata() {
        let mut fixture = fixture();
        fixture["feature_matrix"]["builds"][0]
            .as_object_mut()
            .unwrap()
            .remove("profile");
        assert!(verify_fixture(&fixture).unwrap_err().contains("profile"));
    }

    #[test]
    fn qualification_rejects_an_altered_supported_build() {
        let mut fixture = fixture();
        fixture["feature_matrix"]["builds"][1]["target_triple"] = "x86_64-unknown-linux-gnu".into();
        assert!(verify_fixture(&fixture)
            .unwrap_err()
            .contains("wrong configuration"));
    }

    #[test]
    fn qualification_accepts_the_captured_fixture() {
        verify_fixture(&fixture()).unwrap();
    }

    #[test]
    fn qualification_requires_behavioral_rename_metadata() {
        let mut fixture = fixture();
        fixture["classification"]
            .as_object_mut()
            .unwrap()
            .remove("behavioral_renames");
        assert!(verify_fixture(&fixture)
            .unwrap_err()
            .contains("behavioral_renames"));
    }

    #[test]
    fn qualification_rejects_a_behavioral_rename_without_a_public_rust_path() {
        let mut fixture = fixture();
        fixture["classification"]["behavioral_renames"][0]["rust_path"] = "demo::missing".into();
        assert!(verify_fixture(&fixture)
            .unwrap_err()
            .contains("absent from compiled inventory"));
    }

    #[test]
    fn qualification_rejects_an_added_delta_entry() {
        let mut fixture = fixture();
        fixture["delta"]["added"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({
                "kind": "function",
                "name": "mlx_unclassified",
                "signature": "fn()->i32"
            }));
        fixture["delta"]["counts"]["added"] = 3.into();
        assert!(verify_fixture(&fixture)
            .unwrap_err()
            .contains("unclassified"));
    }

    #[test]
    fn qualification_rejects_a_removed_delta_entry() {
        let mut fixture = fixture();
        fixture["delta"]["removed"].as_array_mut().unwrap().clear();
        fixture["delta"]["counts"]["removed"] = 0.into();
        assert!(verify_fixture(&fixture)
            .unwrap_err()
            .contains("not in delta"));
    }

    #[test]
    fn qualification_rejects_a_changed_signature() {
        let mut fixture = fixture();
        fixture["delta"]["changed"][0]["after"]["signature"] = "fn(i128)->i32".into();
        assert!(verify_fixture(&fixture)
            .unwrap_err()
            .contains("does not match delta"));
    }

    #[test]
    fn qualification_rejects_a_broken_evidence_id() {
        let mut fixture = fixture();
        fixture["classification"]["entries"][0]["evidence"][0] = "conf:missing".into();
        assert!(verify_fixture(&fixture)
            .unwrap_err()
            .contains("unresolved evidence"));
    }

    #[test]
    fn qualification_rejects_an_unclassified_entry() {
        let mut fixture = fixture();
        fixture["classification"]["entries"][0]
            .as_object_mut()
            .unwrap()
            .remove("disposition");
        assert!(verify_fixture(&fixture)
            .unwrap_err()
            .contains("disposition"));
    }

    #[test]
    fn qualification_rejects_a_handle_without_ownership_metadata() {
        let mut fixture = fixture();
        let handle = fixture["classification"]["entries"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|entry| entry["kind"] == "handle")
            .unwrap();
        handle.as_object_mut().unwrap().remove("ownership");
        assert!(verify_fixture(&fixture)
            .unwrap_err()
            .contains("ownership metadata"));
    }

    #[test]
    fn qualification_rejects_a_multiline_rationale() {
        let mut fixture = fixture();
        let deferred = fixture["classification"]["entries"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|entry| entry["disposition"] == "deferred")
            .unwrap();
        deferred["rationale"] = "first line\nsecond line".into();
        assert!(verify_fixture(&fixture).unwrap_err().contains("one line"));
    }

    #[test]
    fn report_evidence_requires_a_typed_passing_report() {
        let root = tempfile::tempdir().unwrap();
        let report = root.path().join("report.json");
        std::fs::write(
            &report,
            serde_json::to_vec(&serde_json::json!({
                "schema_version": 1,
                "report_id": "leaks",
                "evidence_class": "ffi",
                "source_commit": "abc123",
                "verdict": "fail"
            }))
            .unwrap(),
        )
        .unwrap();
        assert!(!super::report_exists(
            root.path(),
            "ffi",
            "report.json#leaks"
        ));
        std::fs::write(
            &report,
            serde_json::to_vec(&serde_json::json!({
                "schema_version": 1,
                "report_id": "leaks",
                "evidence_class": "ffi",
                "source_commit": "abc123",
                "verdict": "pass"
            }))
            .unwrap(),
        )
        .unwrap();
        assert!(super::report_exists(
            root.path(),
            "ffi",
            "report.json#leaks"
        ));
    }

    #[test]
    fn api_report_evidence_routes_to_typed_report_validation() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("report.json"),
            serde_json::to_vec(&serde_json::json!({
                "schema_version": 1,
                "report_id": "surface",
                "evidence_class": "api",
                "source_commit": "abc123",
                "verdict": "pass"
            }))
            .unwrap(),
        )
        .unwrap();
        let paths = super::Paths {
            classification: root.path().join("classification.json"),
            api_baseline: root.path().join("api.json"),
            feature_matrix: root.path().join("features.json"),
            corpus: root.path().join("corpus.json"),
        };
        assert!(super::evidence_exists(
            root.path(),
            "api:report:report.json#surface",
            &BTreeSet::new(),
            &BTreeSet::new(),
            &paths
        ));
    }
}
