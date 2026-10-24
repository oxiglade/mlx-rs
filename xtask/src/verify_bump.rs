use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};

const OLD_COMMIT: &str = "c74db5307cc8ce122f48d97ef951b30578674e7f";
const TARGET_COMMIT: &str = "a341b4925024b88b2c593468f16e12f5e17315da";
const TARGET_MLX: &str = "0.32.2";
const ALLOWED_REPLAY_VERDICTS: &[&str] = &[
    "identical",
    "value_changed",
    "dtype_or_shape_changed",
    "error_behavior_changed",
    "recipe_failed",
];
pub(crate) fn run(repo_root: &Path, args: &[String]) -> i32 {
    let result = (|| {
        if !args.is_empty() {
            return Err("usage: cargo xtask verify-bump".to_owned());
        }
        verify_tuple(repo_root)?;
        verify_sentinel(repo_root)?;
        verify_race_regression(repo_root)?;
        let path = repo_root.join("conformance/target/replay-report.json");
        require_tracked_unchanged(repo_root, &path)?;
        let source = fs::read_to_string(&path).map_err(|e| e.to_string())?;
        let replay = validate_replay_document(path.parent().unwrap(), &source, true)?;
        println!("tuple, sentinel inputs and target replay passed: {replay}");
        Ok(())
    })();
    match result {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("verify-bump: {error}");
            1
        }
    }
}

fn verify_tuple(repo_root: &Path) -> Result<(), String> {
    let mlx_c = repo_root.join("mlx-sys/src/mlx-c");
    let current = git(&mlx_c, &["rev-parse", "HEAD"])?;
    let pin =
        nested_pin(&fs::read_to_string(mlx_c.join("CMakeLists.txt")).map_err(|e| e.to_string())?)?;
    validate_tuple_values(&current, &pin)?;
    let classification = read_json(&repo_root.join("ledger/target-delta-classification.json"))?;
    if classification["old_source_commit"] != OLD_COMMIT
        || classification["target_source_commit"] != TARGET_COMMIT
    {
        return Err("ABI classification does not match the bump tuple".into());
    }
    Ok(())
}

fn validate_tuple_values(current: &str, pin: &str) -> Result<(), String> {
    if current != TARGET_COMMIT || pin != TARGET_MLX {
        return Err(format!(
            "expected mlx-c {TARGET_COMMIT} / MLX {TARGET_MLX}, got {current} / {pin}"
        ));
    }
    Ok(())
}

fn verify_race_regression(repo_root: &Path) -> Result<Value, String> {
    let path = repo_root.join("mlx-tests/tests/ffi_safety.rs");
    let source = fs::read_to_string(&path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let syntax = syn::parse_file(&source)
        .map_err(|error| format!("failed to parse {}: {error}", path.display()))?;
    let name = "concurrent_invoke_errors_stay_on_the_calling_thread";
    let found = syntax.items.iter().any(|item| {
        matches!(item, syn::Item::Fn(function) if function.sig.ident == name && function.attrs.iter().any(|attr| attr.path().is_ident("test")))
    });
    if !found {
        return Err(format!("missing concurrent race regression {name}"));
    }
    Ok(json!({"test": format!("mlx-tests/tests/ffi_safety.rs#{name}")}))
}

fn verify_sentinel(repo_root: &Path) -> Result<(), String> {
    let root = repo_root.join("conformance/sentinel/fixtures/tiny-llama");
    let expectations = read_json(&root.join("expectations.json"))?;
    require_schema(&expectations, "sentinel expectations")?;
    for name in [
        "config.json",
        "tokenizer.json",
        "model.safetensors",
        "model.perturbed.safetensors",
        "expectations.safetensors",
    ] {
        if !root.join(name).is_file() {
            return Err(format!("sentinel is missing {name}"));
        }
    }
    for (name, digest) in expectations["provenance"]["artifact_sha256"]
        .as_object()
        .ok_or("sentinel artifact_sha256 must be an object")?
    {
        if hash_file(&root.join(name))? != format!("sha256:{}", digest.as_str().unwrap_or("")) {
            return Err(format!("sentinel artifact {name} digest mismatch"));
        }
    }
    if expectations["qualification"]["comparator_rejects"] != true {
        return Err("sentinel comparator qualification is missing".to_owned());
    }
    Ok(())
}

fn nested_pin(cmake: &str) -> Result<String, String> {
    cmake
        .lines()
        .find_map(|line| {
            line.split_whitespace()
                .collect::<Vec<_>>()
                .windows(2)
                .find(|pair| pair[0] == "GIT_TAG")
                .map(|pair| {
                    pair[1]
                        .trim_end_matches(')')
                        .trim_start_matches('v')
                        .to_owned()
                })
        })
        .ok_or_else(|| "mlx-c CMakeLists.txt has no MLX GIT_TAG".to_owned())
}

fn validate_replay_document(base: &Path, source: &str, strict: bool) -> Result<Value, String> {
    let report: Value = serde_json::from_str(source).map_err(|e| e.to_string())?;
    let fields = JsonText::new(source)?.object()?;
    let payload = fields
        .get("payload")
        .ok_or("target replay report has no payload")?;
    require_schema(&report, "target replay report")?;
    require_value(&report, "command", "target-replay", "target replay report")?;
    if report["payload_sha256"] != hash_bytes(payload.as_bytes())? {
        return Err("target replay payload_sha256 does not match payload".to_owned());
    }
    let double = &report["double_run"];
    if double["identical"] != true || double["first_run_sha256"] != double["second_run_sha256"] {
        return Err("target replay deterministic double-run did not match".to_owned());
    }
    let handshake = &report["payload"]["handshake"];
    for (field, expected) in [
        ("python", "3.12.14"),
        ("architecture", "arm64"),
        ("mlx", TARGET_MLX),
        ("mlx_metal", TARGET_MLX),
        ("numpy", "2.2.6"),
        ("mlx_runtime", TARGET_MLX),
        ("device", "cpu"),
    ] {
        if handshake[field] != expected {
            return Err(format!(
                "target replay handshake {field} must be {expected}"
            ));
        }
    }
    required_case_field(handshake, "handshake", "venv")?;
    let suites = report["payload"]["suites"]
        .as_array()
        .ok_or("target replay suites must be an array")?;
    let mut seen = BTreeSet::new();
    let mut replay_shards = BTreeMap::new();
    for suite in suites {
        let id = suite["id"]
            .as_str()
            .ok_or("target replay suite id must be a string")?;
        if !seen.insert(id) {
            return Err(format!("duplicate target replay suite {id}"));
        }
        let relative = suite["expectation_shard"]
            .as_str()
            .ok_or_else(|| format!("target replay suite {id} has no expectation_shard"))?;
        let path = safe_join(base, relative)
            .ok_or_else(|| format!("invalid target expectation shard path {relative}"))?;
        if !path.is_file() {
            return Err(format!("missing target expectation shard {relative}"));
        }
        if strict {
            let repo_root = base
                .parent()
                .and_then(Path::parent)
                .ok_or("target replay report is not under conformance/target")?;
            require_tracked_unchanged(repo_root, &path)?;
        }
        if suite["sha256"] != hash_file(&path)? {
            return Err(format!(
                "target expectation shard {relative} digest mismatch"
            ));
        }
        let source = fs::read_to_string(&path).map_err(|e| e.to_string())?;
        let shard: Value = serde_json::from_str(&source).map_err(|e| e.to_string())?;
        replay_shards.insert(id.to_owned(), JsonText::new(&source)?.value()?);
        if shard["schema_version"] != 1 || shard["suite"] != id {
            return Err(format!(
                "target expectation shard {relative} identity mismatch"
            ));
        }
        if shard["target_environment"] != report["payload"]["handshake"] {
            return Err(format!(
                "target expectation shard {relative} handshake mismatch"
            ));
        }
        let cases = shard["cases"]
            .as_array()
            .ok_or_else(|| format!("target expectation shard {relative} cases must be an array"))?;
        if suite["case_count"].as_u64() != Some(cases.len() as u64) {
            return Err(format!(
                "target expectation shard {relative} case_count mismatch"
            ));
        }
        let mut counts = BTreeMap::<&str, u64>::new();
        let mut case_ids = BTreeSet::new();
        for case in cases {
            let case_id = case["id"]
                .as_str()
                .ok_or_else(|| format!("target replay suite {id} has a missing case id"))?;
            if !case_ids.insert(case_id) {
                return Err(format!(
                    "target replay suite {id} has duplicate case {case_id}"
                ));
            }
            let verdict = case["verdict"]
                .as_str()
                .ok_or_else(|| format!("target replay suite {id} has a missing verdict"))?;
            if !ALLOWED_REPLAY_VERDICTS.contains(&verdict) {
                return Err(format!(
                    "target replay suite {id} has invalid verdict {verdict}"
                ));
            }
            *counts.entry(verdict).or_default() += 1;
        }
        if counts.contains_key("recipe_failed") {
            return Err(format!("target replay suite {id} contains recipe_failed"));
        }
        if suite["verdict"] != "pass" || suite["verdict_counts"] != json!(counts) {
            return Err(format!("target replay suite {id} summary mismatch"));
        }
    }
    if report["verdict"] != "pass" {
        return Err("target replay report verdict is not pass".to_owned());
    }
    let reconstructed = hash_bytes(
        encode_object(&BTreeMap::from([
            ("payload".to_owned(), payload.clone()),
            ("shards".to_owned(), encode_object(&replay_shards)),
        ]))
        .as_bytes(),
    )?;
    if double["first_run_sha256"] != reconstructed || double["second_run_sha256"] != reconstructed {
        return Err("target replay double-run hash does not match payload and shards".to_owned());
    }
    let committed_provenance = strict
        .then(|| validate_strict_replay(base, &report, &seen))
        .transpose()?;
    Ok(json!({
        "payload_sha256": report["payload_sha256"],
        "suites": suites.len(),
        "double_run_sha256": double["first_run_sha256"],
        "committed_provenance": committed_provenance
    }))
}

fn validate_strict_replay(
    base: &Path,
    report: &Value,
    seen: &BTreeSet<&str>,
) -> Result<Value, String> {
    if report["payload"]["isolation"]
        != json!({
            "process_scope": "fresh_subprocess_per_suite",
            "state_reset": "new_model_and_optimizer_per_trajectory"
        })
    {
        return Err("target replay isolation declaration is missing or invalid".to_owned());
    }
    let repo_root = base
        .parent()
        .and_then(Path::parent)
        .ok_or("target replay report is not under conformance/target")?;
    let corpus = read_json(&repo_root.join("conformance/corpus.json"))?;
    let manifest = read_json(&repo_root.join("conformance/state/manifest.json"))?;
    let mut expected_suites = BTreeSet::from([
        "state".to_owned(),
        "transforms".to_owned(),
        "probe_oob_take".to_owned(),
        "probe_singular_inv".to_owned(),
    ]);
    let mut expected_cases = BTreeMap::<String, BTreeSet<String>>::new();
    for suite in corpus["suites"]
        .as_array()
        .ok_or("corpus suites must be an array")?
    {
        let path = suite.as_str().ok_or("corpus suite path must be a string")?;
        let id = Path::new(path)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or("corpus suite path has no UTF-8 stem")?
            .to_owned();
        expected_suites.insert(id.clone());
        let document = read_json(&repo_root.join("conformance").join(path))?;
        expected_cases.insert(
            id,
            document["cases"]
                .as_array()
                .ok_or("corpus suite cases must be an array")?
                .iter()
                .map(|case| {
                    case["id"]
                        .as_str()
                        .map(str::to_owned)
                        .ok_or("corpus case id must be a string".to_owned())
                })
                .collect::<Result<_, _>>()?,
        );
    }
    if expected_suites != seen.iter().map(|id| (*id).to_owned()).collect() {
        return Err("target replay report does not contain every required suite".to_owned());
    }
    expected_cases.insert(
        "state".to_owned(),
        manifest["trajectories"]
            .as_array()
            .ok_or("state manifest trajectories must be an array")?
            .iter()
            .map(|case| {
                case["id"]
                    .as_str()
                    .map(|id| format!("state.{id}"))
                    .ok_or("state trajectory id must be a string".to_owned())
            })
            .collect::<Result<_, _>>()?,
    );
    expected_cases.insert(
        "transforms".to_owned(),
        [
            "transforms.nonlinear_value_and_grad",
            "transforms.argnums_selection",
            "transforms.jvp",
            "transforms.vjp",
            "transforms.module_value_and_grad",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect(),
    );
    expected_cases.insert(
        "probe_oob_take".to_owned(),
        BTreeSet::from(["probe.oob_take".to_owned()]),
    );
    expected_cases.insert(
        "probe_singular_inv".to_owned(),
        BTreeSet::from(["probe.singular_inv".to_owned()]),
    );
    let mut provenance_matches = BTreeMap::from([
        ("pre_migration", Vec::<String>::new()),
        ("post_migration", Vec::<String>::new()),
    ]);
    for suite in report["payload"]["suites"]
        .as_array()
        .ok_or("target replay suites must be an array")?
    {
        let id = suite["id"]
            .as_str()
            .ok_or("target replay suite id must be a string")?;
        let relative = suite["expectation_shard"]
            .as_str()
            .ok_or("target replay expectation_shard must be a string")?;
        let shard = read_json(
            &safe_join(base, relative)
                .ok_or_else(|| format!("invalid target expectation shard path {relative}"))?,
        )?;
        let actual = shard["cases"]
            .as_array()
            .ok_or("target replay shard cases must be an array")?
            .iter()
            .map(|case| {
                case["id"]
                    .as_str()
                    .map(str::to_owned)
                    .ok_or("target replay case id must be a string".to_owned())
            })
            .collect::<Result<BTreeSet<_>, _>>()?;
        let committed_environment = if matches!(id, "state" | "transforms") {
            &manifest["provenance"]["environment"]
        } else {
            &corpus["environment"]
        };
        let provenance_match = classify_replay_provenance(
            committed_environment,
            &shard["old_environment"],
            &shard["target_environment"],
        )
        .map_err(|error| format!("target replay suite {id} {error}"))?;
        provenance_matches
            .get_mut(provenance_match)
            .expect("known provenance match")
            .push(id.to_owned());
        for case in shard["cases"]
            .as_array()
            .ok_or("target replay shard cases must be an array")?
        {
            if id == "gguf" && case["recipe"].is_object() {
                required_case_field(&case["recipe"], id, "kind")?;
            } else {
                required_case_field(case, id, "recipe")?;
            }
            required_case_field(case, id, "input_sha256")?;
            case["input_refs"]
                .as_array()
                .filter(|refs| refs.iter().all(|reference| reference.as_str().is_some()))
                .ok_or_else(|| format!("target replay suite {id} has invalid input_refs"))?;
            if case["verdict"] == "value_changed" && !case.get("max_error").is_some() {
                return Err(format!(
                    "target replay suite {id} value_changed case has no max_error"
                ));
            }
        }
        if expected_cases.get(id) != Some(&actual) {
            return Err(format!(
                "target replay suite {id} does not contain every named case"
            ));
        }
    }
    let sources = report["payload"]["source_artifacts"]
        .as_object()
        .ok_or("target replay source_artifacts must be an object")?;
    let mut expected_paths = BTreeSet::from([
        "conformance/generate.py".to_owned(),
        "conformance/corpus.json".to_owned(),
        "conformance/state/generate_state.py".to_owned(),
        "conformance/state/manifest.json".to_owned(),
        "conformance/state/state.safetensors".to_owned(),
        "conformance/target/replay_target.py".to_owned(),
        "conformance/target/requirements-target.lock".to_owned(),
    ]);
    for field in ["suites", "fixture_shards"] {
        let entries = if field == "suites" {
            corpus[field]
                .as_array()
                .ok_or("corpus suites must be an array")?
                .iter()
                .map(|value| value.as_str().map(str::to_owned))
                .collect::<Option<Vec<_>>>()
                .ok_or("corpus suites must contain strings")?
        } else {
            corpus[field]
                .as_object()
                .ok_or("corpus fixture_shards must be an object")?
                .keys()
                .cloned()
                .collect()
        };
        for entry in entries {
            expected_paths.insert(format!("conformance/{entry}"));
        }
    }
    if let Some(fixtures) = corpus.get("gguf_fixtures") {
        for path in fixtures
            .as_object()
            .ok_or("corpus gguf_fixtures must be an object")?
            .keys()
        {
            expected_paths.insert(format!("conformance/{path}"));
        }
    }
    if expected_paths != sources.keys().cloned().collect() {
        return Err(
            "target replay source_artifacts set does not match committed inputs".to_owned(),
        );
    }
    let historical_source_commit = if provenance_matches["post_migration"]
        .iter()
        .all(|suite| matches!(suite.as_str(), "probe_oob_take" | "probe_singular_inv"))
    {
        None
    } else {
        let replay_report = base.join("replay-report.json");
        let replay_report = replay_report
            .strip_prefix(repo_root)
            .map_err(|_| "target replay report is outside the repository")?
            .to_str()
            .ok_or("target replay report path is not UTF-8")?;
        Some(replay_introduction_commit(repo_root, replay_report)?)
    };
    for (relative, digest) in sources {
        let path = safe_join(repo_root, relative)
            .ok_or_else(|| format!("invalid target replay source path {relative}"))?;
        require_tracked_unchanged(repo_root, &path)?;
        let actual = if let Some(commit) = historical_source_commit.as_deref() {
            hash_git_file(repo_root, commit, relative)?
        } else {
            hash_file(&path)?
        };
        if digest != &json!(actual) {
            return Err(format!(
                "target replay source artifact {relative} digest mismatch"
            ));
        }
    }
    if manifest["trajectories"]
        .as_array()
        .is_none_or(Vec::is_empty)
    {
        return Err("state manifest has no trajectories".to_owned());
    }
    Ok(json!(provenance_matches))
}

fn classify_replay_provenance(
    committed: &Value,
    old: &Value,
    target: &Value,
) -> Result<&'static str, String> {
    if committed == old {
        return Ok("pre_migration");
    }
    if committed == &project_target_environment(old, target)? {
        return Ok("post_migration");
    }
    Err("committed provenance matches neither replay old nor target environment".to_owned())
}

fn project_target_environment(old: &Value, target: &Value) -> Result<Value, String> {
    let old = old
        .as_object()
        .ok_or("replay old environment must be an object")?;
    let target = target
        .as_object()
        .ok_or("replay target environment must be an object")?;
    let mut projected = old.clone();
    for (provenance_field, target_field) in [
        ("python", "python"),
        ("architecture", "architecture"),
        ("mlx_package", "mlx"),
        ("mlx_metal_package", "mlx_metal"),
        ("mlx", "mlx"),
        ("mlx_metal", "mlx_metal"),
        ("mlx_runtime", "mlx_runtime"),
        ("numpy", "numpy"),
    ] {
        if old.contains_key(provenance_field) {
            let value = target
                .get(target_field)
                .ok_or_else(|| format!("replay target environment is missing {target_field}"))?;
            projected.insert(provenance_field.to_owned(), value.clone());
        }
    }
    Ok(Value::Object(projected))
}

fn required_case_field<'a>(case: &'a Value, suite: &str, field: &str) -> Result<&'a str, String> {
    case[field]
        .as_str()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("target replay suite {suite} has invalid {field}"))
}

fn require_schema(value: &Value, name: &str) -> Result<(), String> {
    if value["schema_version"].as_u64() == Some(1) {
        Ok(())
    } else {
        Err(format!("{name} has unsupported schema_version"))
    }
}

fn require_value(value: &Value, field: &str, expected: &str, name: &str) -> Result<(), String> {
    if value[field].as_str() == Some(expected) {
        Ok(())
    } else {
        Err(format!("{name} {field} must be {expected}"))
    }
}

fn read_json(path: &Path) -> Result<Value, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("failed to parse {}: {error}", path.display()))
}

fn require_tracked_unchanged(repo_root: &Path, path: &Path) -> Result<(), String> {
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        repo_root.join(path)
    };
    let relative = absolute
        .strip_prefix(repo_root)
        .map_err(|_| format!("evidence {} is outside the repository", path.display()))?;
    let relative = relative
        .to_str()
        .ok_or_else(|| format!("evidence path {} is not UTF-8", relative.display()))?;
    git(repo_root, &["ls-files", "--error-unmatch", "--", relative])
        .map_err(|_| format!("evidence {relative} is not tracked"))?;
    let status = Command::new("git")
        .args(["diff", "--quiet", "HEAD", "--", relative])
        .current_dir(repo_root)
        .status()
        .map_err(|error| format!("failed to inspect evidence {relative}: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("evidence {relative} differs from HEAD"))
    }
}

fn safe_join(base: &Path, relative: &str) -> Option<PathBuf> {
    let relative = Path::new(relative);
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return None;
    }
    Some(base.join(relative))
}

fn hash_file(path: &Path) -> Result<String, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    hash_bytes(&bytes)
}

fn hash_git_file(repo_root: &Path, commit: &str, path: &str) -> Result<String, String> {
    let output = Command::new("git")
        .args(["show", &format!("{commit}:{path}")])
        .current_dir(repo_root)
        .output()
        .map_err(|error| format!("failed to read historical replay source {path}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "failed to read historical replay source {path}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    hash_bytes(&output.stdout)
}

fn replay_introduction_commit(repo_root: &Path, path: &str) -> Result<String, String> {
    let history = git(
        repo_root,
        &[
            "log",
            "--reverse",
            "--diff-filter=A",
            "--format=%H",
            "--",
            path,
        ],
    )?;
    history
        .lines()
        .next()
        .map(str::to_owned)
        .ok_or_else(|| format!("target replay report {path} has no introduction commit"))
}

#[cfg(test)]
fn hash_json(value: &Value) -> Result<String, String> {
    hash_bytes(&canonical_json(&value.to_string())?)
}

#[cfg(test)]
fn canonical_json(source: &str) -> Result<Vec<u8>, String> {
    let mut parser = JsonText::new(source)?;
    Ok(parser.value()?.into_bytes())
}

struct JsonText<'a>(&'a str);

impl<'a> JsonText<'a> {
    fn new(source: &'a str) -> Result<Self, String> {
        // Validate syntax without rounding numbers through serde_json::Value.
        serde_json::from_str::<serde::de::IgnoredAny>(source).map_err(|e| e.to_string())?;
        Ok(Self(source.trim_start()))
    }

    fn consume(&mut self) -> u8 {
        let byte = self.0.as_bytes()[0];
        self.0 = self.0[1..].trim_start();
        byte
    }

    fn string(&mut self) -> Result<String, String> {
        let mut stream = serde_json::Deserializer::from_str(self.0).into_iter::<String>();
        let value = stream.next().unwrap().map_err(|e| e.to_string())?;
        self.0 = self.0[stream.byte_offset()..].trim_start();
        Ok(value)
    }

    fn object(&mut self) -> Result<BTreeMap<String, String>, String> {
        if self.consume() != b'{' {
            return Err("expected JSON object".to_owned());
        }
        let mut fields = BTreeMap::new();
        while !self.0.starts_with('}') {
            let key = self.string()?;
            self.consume();
            fields.insert(key, self.value()?);
            if !self.0.starts_with(',') {
                break;
            }
            self.consume();
        }
        self.consume();
        Ok(fields)
    }

    fn value(&mut self) -> Result<String, String> {
        match self.0.as_bytes()[0] {
            b'"' => Ok(canonical_string(&self.string()?)),
            b'{' => Ok(encode_object(&self.object()?)),
            b'[' => {
                self.consume();
                let mut values = Vec::new();
                while !self.0.starts_with(']') {
                    values.push(self.value()?);
                    if !self.0.starts_with(',') {
                        break;
                    }
                    self.consume();
                }
                self.consume();
                Ok(format!("[{}]", values.join(",")))
            }
            _ => {
                let end = self
                    .0
                    .find(|c: char| c.is_ascii_whitespace() || ",]}".contains(c))
                    .unwrap_or(self.0.len());
                let token = &self.0[..end];
                self.0 = self.0[end..].trim_start();
                if token.contains(['.', 'e', 'E']) && !matches!(token, "true" | "false") {
                    let value = token.parse::<f64>().map_err(|e| e.to_string())?;
                    if !value.is_finite() {
                        return Err("non-finite canonical JSON number".to_owned());
                    }
                    Ok(python_float(value))
                } else {
                    Ok(if token == "-0" { "0" } else { token }.to_owned())
                }
            }
        }
    }
}

fn canonical_string(value: &str) -> String {
    use std::fmt::Write;

    let mut output = String::new();
    for character in serde_json::to_string(value).unwrap().chars() {
        if character >= '\u{7f}' {
            for unit in character.encode_utf16(&mut [0; 2]) {
                write!(output, "\\u{unit:04x}").unwrap();
            }
        } else {
            output.push(character);
        }
    }
    output
}

fn encode_object(fields: &BTreeMap<String, String>) -> String {
    let fields = fields
        .iter()
        .map(|(key, value)| format!("{}:{value}", canonical_string(key)))
        .collect::<Vec<_>>();
    format!("{{{}}}", fields.join(","))
}

fn python_float(value: f64) -> String {
    // Python uses shortest ties-to-even digits, but different exponent formatting.
    let shortest = serde_json::Number::from_f64(value).unwrap().to_string();
    if value == 0.0 {
        return shortest;
    }
    let (mantissa, exponent) = shortest.split_once('e').unwrap_or((&shortest, "0"));
    let sign = if value.is_sign_negative() { "-" } else { "" };
    let mantissa = mantissa.trim_start_matches('-');
    let point = mantissa.find('.').unwrap_or(mantissa.len());
    let digits = mantissa.replace('.', "");
    let leading_zeros = digits.len() - digits.trim_start_matches('0').len();
    let exponent = exponent.parse::<i32>().unwrap() + point as i32 - leading_zeros as i32 - 1;
    let digits = digits.trim_start_matches('0').trim_end_matches('0');
    if !(-4..16).contains(&exponent) {
        let (first, rest) = digits.split_at(1);
        let dot = if rest.is_empty() { "" } else { "." };
        return format!("{sign}{first}{dot}{rest}e{exponent:+03}");
    }
    let point = exponent + 1;
    if point <= 0 {
        format!("{sign}0.{}{digits}", "0".repeat((-point) as usize))
    } else if point as usize >= digits.len() {
        format!(
            "{sign}{digits}{}.0",
            "0".repeat(point as usize - digits.len())
        )
    } else {
        let (integer, fraction) = digits.split_at(point as usize);
        format!("{sign}{integer}.{fraction}")
    }
}

fn hash_bytes(bytes: &[u8]) -> Result<String, String> {
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
        .write_all(bytes)
        .map_err(|error| format!("failed to hash bytes: {error}"))?;
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

fn git(directory: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(directory)
        .output()
        .map_err(|error| format!("failed to run git: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout)
        .map(|output| output.trim().to_owned())
        .map_err(|error| format!("git returned invalid UTF-8: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn replay_tree() -> (tempfile::TempDir, Value) {
        let root = tempfile::tempdir().unwrap();
        let handshake = json!({
            "python": "3.12.14", "architecture": "arm64", "venv": "/scratch/replay/venv",
            "mlx": TARGET_MLX, "mlx_metal": TARGET_MLX, "numpy": "2.2.6",
            "mlx_runtime": TARGET_MLX, "device": "cpu"
        });
        let shard = json!({
            "schema_version": 1,
            "suite": "arithmetic",
            "target_environment": handshake,
            "cases": [{"id": "arithmetic.001", "verdict": "identical"}]
        });
        let shard_path = root.path().join("arithmetic.json");
        std::fs::write(&shard_path, serde_json::to_vec_pretty(&shard).unwrap()).unwrap();
        let payload = json!({
            "handshake": handshake,
            "source_artifacts": {},
            "suites": [{
                "id": "arithmetic", "expectation_shard": "arithmetic.json",
                "sha256": hash_file(&shard_path).unwrap(), "case_count": 1,
                "verdict_counts": {"identical": 1}, "verdict": "pass"
            }]
        });
        let digest = hash_json(&payload).unwrap();
        let double_digest = hash_json(&json!({
            "payload": payload,
            "shards": {"arithmetic": shard}
        }))
        .unwrap();
        let report = json!({
            "schema_version": 1,
            "command": "target-replay",
            "verdict": "pass",
            "payload": payload,
            "payload_sha256": digest,
            "double_run": {
                "first_run_sha256": double_digest,
                "second_run_sha256": double_digest,
                "identical": true
            }
        });
        (root, report)
    }

    fn replay_provenance_qualification() -> Value {
        serde_json::from_str(include_str!(
            "../tests/fixtures/replay-provenance-qualification.json"
        ))
        .unwrap()
    }

    fn write_replay_fixture_file(root: &Path, relative: &str, contents: &[u8]) {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, contents).unwrap();
    }

    fn write_replay_fixture_json(root: &Path, relative: &str, value: &Value) {
        write_replay_fixture_file(root, relative, &serde_json::to_vec_pretty(value).unwrap());
    }

    fn run_replay_fixture_git(root: &Path, args: &[&str]) {
        let status = Command::new("git")
            .args(args)
            .current_dir(root)
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_COUNT", "0")
            .status()
            .unwrap();
        assert!(status.success());
    }

    fn replay_provenance_documents(post_migration: bool) -> (Value, Value) {
        if post_migration {
            return (
                json!({
                    "environment": {
                        "python": "3.12.14", "architecture": "arm64",
                        "mlx_package": TARGET_MLX, "mlx_metal_package": TARGET_MLX,
                        "mlx_runtime": TARGET_MLX, "numpy": "2.2.6"
                    },
                    "suites": [],
                    "fixture_shards": {}
                }),
                json!({
                    "provenance": {"environment": {
                        "python": "3.12.14", "architecture": "arm64",
                        "pinned_venv": "conformance/.venv", "mlx": TARGET_MLX,
                        "mlx_metal": TARGET_MLX, "numpy": "2.2.6"
                    }},
                    "trajectories": [{"id": "adam"}]
                }),
            );
        }

        (
            json!({
                "environment": {
                    "python": "3.12.11", "architecture": "arm64",
                    "mlx_package": "0.31.0", "mlx_metal_package": "0.31.0",
                    "mlx_runtime": "0.31.0", "numpy": "2.1.3"
                },
                "suites": [],
                "fixture_shards": {}
            }),
            json!({
                "provenance": {"environment": {
                    "python": "3.12.11", "architecture": "arm64",
                    "pinned_venv": "conformance/.venv", "mlx": "0.31.0",
                    "mlx_metal": "0.31.0", "numpy": "2.1.3"
                }},
                "trajectories": [{"id": "adam"}]
            }),
        )
    }

    fn replay_provenance_tree() -> (tempfile::TempDir, Value) {
        let root = tempfile::tempdir().unwrap();
        let (corpus, manifest) = replay_provenance_documents(false);
        write_replay_fixture_json(root.path(), "conformance/corpus.json", &corpus);
        write_replay_fixture_json(root.path(), "conformance/state/manifest.json", &manifest);
        for relative in [
            "conformance/generate.py",
            "conformance/state/generate_state.py",
            "conformance/state/state.safetensors",
            "conformance/target/replay_target.py",
            "conformance/target/requirements-target.lock",
        ] {
            write_replay_fixture_file(root.path(), relative, relative.as_bytes());
        }

        let target_environment = json!({
            "python": "3.12.14", "architecture": "arm64",
            "venv": "conformance/.venv-target", "mlx": TARGET_MLX,
            "mlx_metal": TARGET_MLX, "numpy": "2.2.6",
            "mlx_runtime": TARGET_MLX, "device": "cpu"
        });
        let suites = [
            ("state", vec!["state.adam"]),
            (
                "transforms",
                vec![
                    "transforms.nonlinear_value_and_grad",
                    "transforms.argnums_selection",
                    "transforms.jvp",
                    "transforms.vjp",
                    "transforms.module_value_and_grad",
                ],
            ),
            ("probe_oob_take", vec!["probe.oob_take"]),
            ("probe_singular_inv", vec!["probe.singular_inv"]),
        ];
        let mut replay_shards = BTreeMap::new();
        let mut replay_suites = Vec::new();
        for (id, case_ids) in suites {
            let old_environment = if matches!(id, "state" | "transforms") {
                &manifest["provenance"]["environment"]
            } else {
                &corpus["environment"]
            };
            let cases = case_ids
                .into_iter()
                .map(|case_id| {
                    json!({
                        "id": case_id,
                        "verdict": "identical",
                        "recipe": "fixture",
                        "input_sha256": "sha256:fixture",
                        "input_refs": ["fixture"]
                    })
                })
                .collect::<Vec<_>>();
            let shard = json!({
                "schema_version": 1,
                "suite": id,
                "old_environment": old_environment,
                "target_environment": target_environment,
                "cases": cases
            });
            let relative = format!("target-expectations/{id}.json");
            let path = format!("conformance/target/{relative}");
            write_replay_fixture_json(root.path(), &path, &shard);
            replay_suites.push(json!({
                "id": id,
                "expectation_shard": relative,
                "sha256": hash_file(&root.path().join(path)).unwrap(),
                "case_count": shard["cases"].as_array().unwrap().len(),
                "verdict_counts": {"identical": shard["cases"].as_array().unwrap().len()},
                "verdict": "pass"
            }));
            replay_shards.insert(id.to_owned(), shard);
        }

        let source_artifacts = [
            "conformance/generate.py",
            "conformance/corpus.json",
            "conformance/state/generate_state.py",
            "conformance/state/manifest.json",
            "conformance/state/state.safetensors",
            "conformance/target/replay_target.py",
            "conformance/target/requirements-target.lock",
        ]
        .into_iter()
        .map(|relative| {
            (
                relative.to_owned(),
                json!(hash_file(&root.path().join(relative)).unwrap()),
            )
        })
        .collect::<serde_json::Map<_, _>>();
        let payload = json!({
            "handshake": target_environment,
            "isolation": {
                "process_scope": "fresh_subprocess_per_suite",
                "state_reset": "new_model_and_optimizer_per_trajectory"
            },
            "source_artifacts": source_artifacts,
            "suites": replay_suites
        });
        let double_run_sha256 = hash_json(&json!({
            "payload": payload,
            "shards": replay_shards
        }))
        .unwrap();
        let report = json!({
            "schema_version": 1,
            "command": "target-replay",
            "verdict": "pass",
            "payload_sha256": hash_json(&payload).unwrap(),
            "payload": payload,
            "double_run": {
                "first_run_sha256": double_run_sha256,
                "second_run_sha256": double_run_sha256,
                "identical": true
            }
        });
        write_replay_fixture_json(
            root.path(),
            "conformance/target/replay-report.json",
            &report,
        );

        run_replay_fixture_git(root.path(), &["init", "--quiet"]);
        run_replay_fixture_git(
            root.path(),
            &["config", "user.email", "qualification@example.invalid"],
        );
        run_replay_fixture_git(root.path(), &["config", "user.name", "Qualification"]);
        run_replay_fixture_git(root.path(), &["add", "conformance"]);
        run_replay_fixture_git(
            root.path(),
            &["commit", "--quiet", "-m", "test: add replay report"],
        );

        (root, report)
    }

    #[test]
    fn sha256_matches_known_vector() {
        assert_eq!(
            hash_bytes(b"abc").unwrap(),
            "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn json_numbers_deserialize_through_untagged_values() {
        #[derive(serde::Deserialize)]
        #[serde(untagged)]
        enum Scalar {
            Float(f64),
            Text(String),
        }
        let values: Vec<Scalar> = serde_json::from_str(r#"[1.25, "nan"]"#).unwrap();
        assert!(matches!(values[0], Scalar::Float(value) if value == 1.25));
        assert!(matches!(&values[1], Scalar::Text(value) if value == "nan"));
    }

    #[test]
    fn json_hash_matches_python_canonical_encoding() {
        let value = json!({"z": [1, true, null], "a": {"mlx": "0.32.2", "count": 2}});

        assert_eq!(
            hash_json(&value).unwrap(),
            "sha256:9c374238cded86c892911a96b5afb7dc5dedca833e58969d0a6a55be4326c10c"
        );
    }

    fn python_canonical(expression: &str) -> Vec<u8> {
        let output = Command::new("python3")
            .args([
                "-c",
                &format!(
                    "import json, math, random, struct, sys\nvalue = {expression}\nsys.stdout.buffer.write(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())"
                ),
            ])
            .output()
            .unwrap();
        assert!(output.status.success(), "{:?}", output);
        output.stdout
    }

    fn assert_python_canonical(expression: &str) {
        let expected = python_canonical(expression);
        assert_eq!(
            canonical_json(std::str::from_utf8(&expected).unwrap()).unwrap(),
            expected
        );
    }

    #[test]
    fn canonical_json_matches_python_ascii() {
        assert_python_canonical(
            r#"{'z': [1, True, None], 'a': {'controls': ''.join(map(chr, range(128))), 'slash': '/\\\"'}}"#,
        );
    }

    #[test]
    fn canonical_json_matches_python_bmp() {
        assert_python_canonical(
            r#"{'é': 'árvíztűrő tükörfúrógép', 'a': '\u0080\u2028\u2029\uffff', '中': '日本語'}"#,
        );
    }

    #[test]
    fn canonical_json_matches_python_astral() {
        assert_python_canonical(r#"{'\U0001f600': '🎵𐀀\U0010ffff', '\ue000': ['雪', '🚀']}"#);
    }

    #[test]
    fn canonical_json_matches_python_numbers() {
        assert_python_canonical("[0, -1, 2**63, 2**64, -(2**100), 0.0, -0.0, 1.0, 1e-4, 1e-5, 1e15, 1e16, 1e20, 1e23, 1.2345678901234567, 5e-324, sys.float_info.max, 1.0000000000000002, 781366138262913.25]");
    }

    #[test]
    fn canonical_json_matches_python_float_roundtrips() {
        assert_python_canonical("[v for r in [random.Random(341)] for _ in range(4096) for v in [struct.unpack('!d', r.getrandbits(64).to_bytes(8, 'big'))[0]] if math.isfinite(v)]");
    }

    #[test]
    fn canonical_json_matches_python_float_boundaries() {
        assert_python_canonical("[v for exponent in range(-1074, 1024) for value in [2.0**exponent] for v in [value, math.nextafter(value, -math.inf), math.nextafter(value, math.inf)] if math.isfinite(v)]");
    }

    #[test]
    fn canonical_json_normalizes_number_spellings() {
        let source = "[-0, -0.0, 1e0, 100e-2, 1.2300, 1e-400, -1e-400, 0e0]";
        let expected = python_canonical(&format!("json.loads('{source}')"));
        assert_eq!(canonical_json(source).unwrap(), expected);
    }

    #[test]
    fn committed_replay_passes_admission() {
        let base = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../conformance/target")
            .canonicalize()
            .unwrap();
        let report = fs::read_to_string(base.join("replay-report.json")).unwrap();
        validate_replay_document(&base, &report, true).unwrap();
    }

    #[test]
    fn replay_accepts_pinned_environment_in_scratch_venv() {
        let (root, report) = replay_tree();
        validate_replay_document(root.path(), &report.to_string(), false).unwrap();
    }

    #[test]
    fn replay_rejects_payload_tampering() {
        let (root, mut report) = replay_tree();
        report["payload"]["handshake"]["mlx_runtime"] = "0.32.1".into();

        let error = validate_replay_document(root.path(), &report.to_string(), false).unwrap_err();

        assert!(error.contains("payload_sha256"));
    }

    #[test]
    fn replay_rejects_missing_shard() {
        let (root, report) = replay_tree();
        std::fs::remove_file(root.path().join("arithmetic.json")).unwrap();

        let error = validate_replay_document(root.path(), &report.to_string(), false).unwrap_err();

        assert!(error.contains("missing target expectation shard"));
    }

    #[test]
    fn replay_rejects_recipe_failure() {
        let (root, mut report) = replay_tree();
        let shard_path = root.path().join("arithmetic.json");
        let shard = json!({
            "schema_version": 1, "suite": "arithmetic",
            "target_environment": report["payload"]["handshake"],
            "cases": [{"id": "arithmetic.001", "verdict": "recipe_failed"}]
        });
        std::fs::write(&shard_path, serde_json::to_vec_pretty(&shard).unwrap()).unwrap();
        report["payload"]["suites"][0]["sha256"] = hash_file(&shard_path).unwrap().into();
        report["payload"]["suites"][0]["verdict_counts"] = json!({"recipe_failed": 1});
        report["payload"]["suites"][0]["verdict"] = "fail".into();
        report["payload_sha256"] = hash_json(&report["payload"]).unwrap().into();
        let double_digest = hash_json(&json!({
            "payload": report["payload"],
            "shards": {"arithmetic": shard}
        }))
        .unwrap();
        report["double_run"]["first_run_sha256"] = double_digest.clone().into();
        report["double_run"]["second_run_sha256"] = double_digest.into();

        let error = validate_replay_document(root.path(), &report.to_string(), false).unwrap_err();

        assert!(error.contains("recipe_failed"));
    }

    #[test]
    fn replay_rejects_forged_equal_double_run_hashes() {
        let (root, mut report) = replay_tree();
        report["double_run"]["first_run_sha256"] = "sha256:forged".into();
        report["double_run"]["second_run_sha256"] = "sha256:forged".into();

        let error = validate_replay_document(root.path(), &report.to_string(), false).unwrap_err();

        assert!(error.contains("double-run hash"));
    }

    #[test]
    fn replay_provenance_qualification_accepts_pre_and_post_migration_states() {
        let fixture = replay_provenance_qualification();
        let target = &fixture["target_environment"];

        for family in fixture["families"].as_array().unwrap() {
            for state in family["states"].as_array().unwrap() {
                assert_eq!(
                    classify_replay_provenance(
                        &state["committed_environment"],
                        &family["old_environment"],
                        target,
                    )
                    .unwrap(),
                    state["expected_match"].as_str().unwrap(),
                    "{}",
                    family["id"].as_str().unwrap(),
                );
            }
        }
    }

    #[test]
    fn replay_provenance_rejects_an_environment_outside_the_transition() {
        let fixture = replay_provenance_qualification();
        let family = &fixture["families"][0];
        let mut unrelated = family["states"][1]["committed_environment"].clone();
        unrelated["numpy"] = "3.0.0".into();

        let error = classify_replay_provenance(
            &unrelated,
            &family["old_environment"],
            &fixture["target_environment"],
        )
        .unwrap_err();

        assert!(error.contains("neither replay old nor target environment"));
    }

    #[test]
    fn strict_replay_accepts_pre_and_post_migration_provenance() {
        let (root, report) = replay_provenance_tree();
        let replay_root = root.path().join("conformance/target");

        let evidence = validate_replay_document(&replay_root, &report.to_string(), true).unwrap();
        assert_eq!(
            evidence["committed_provenance"],
            json!({
                "pre_migration": [
                    "state", "transforms", "probe_oob_take", "probe_singular_inv"
                ],
                "post_migration": []
            })
        );

        let (corpus, manifest) = replay_provenance_documents(true);
        write_replay_fixture_json(root.path(), "conformance/corpus.json", &corpus);
        write_replay_fixture_json(root.path(), "conformance/state/manifest.json", &manifest);
        run_replay_fixture_git(root.path(), &["add", "conformance"]);
        run_replay_fixture_git(
            root.path(),
            &["commit", "--quiet", "-m", "test: migrate provenance"],
        );

        let evidence = validate_replay_document(&replay_root, &report.to_string(), true).unwrap();

        assert_eq!(
            evidence["committed_provenance"],
            json!({
                "pre_migration": [],
                "post_migration": [
                    "state", "transforms", "probe_oob_take", "probe_singular_inv"
                ]
            })
        );
    }

    #[test]
    fn fresh_replay_keeps_current_sources_with_historical_probes() {
        let (root, mut report) = replay_provenance_tree();
        let (corpus, manifest) = replay_provenance_documents(true);
        write_replay_fixture_json(root.path(), "conformance/corpus.json", &corpus);
        write_replay_fixture_json(root.path(), "conformance/state/manifest.json", &manifest);
        for path in ["conformance/corpus.json", "conformance/state/manifest.json"] {
            report["payload"]["source_artifacts"][path] =
                json!(hash_file(&root.path().join(path)).unwrap());
        }
        for suite in ["state", "transforms"] {
            let path = format!("conformance/target/target-expectations/{suite}.json");
            let mut shard = read_json(&root.path().join(&path)).unwrap();
            shard["old_environment"] = manifest["provenance"]["environment"].clone();
            write_replay_fixture_json(root.path(), &path, &shard);
        }
        run_replay_fixture_git(root.path(), &["add", "conformance"]);
        run_replay_fixture_git(
            root.path(),
            &["commit", "--quiet", "-m", "test: replay current corpus"],
        );
        let seen = BTreeSet::from([
            "state",
            "transforms",
            "probe_oob_take",
            "probe_singular_inv",
        ]);

        validate_strict_replay(&root.path().join("conformance/target"), &report, &seen).unwrap();
    }

    #[test]
    fn strict_replay_accepts_cases_without_tensor_inputs() {
        let (root, report) = replay_provenance_tree();
        let path = "conformance/target/target-expectations/state.json";
        let mut shard = read_json(&root.path().join(path)).unwrap();
        shard["cases"][0]["input_refs"] = json!([]);
        write_replay_fixture_json(root.path(), path, &shard);
        let seen = BTreeSet::from([
            "state",
            "transforms",
            "probe_oob_take",
            "probe_singular_inv",
        ]);

        validate_strict_replay(&root.path().join("conformance/target"), &report, &seen).unwrap();
    }

    #[test]
    fn strict_replay_binds_gguf_sources() {
        let (root, mut report) = replay_provenance_tree();
        let path = "conformance/fixtures/gguf/basic.gguf";
        write_replay_fixture_file(root.path(), path, b"GGUF fixture");
        let mut corpus = read_json(&root.path().join("conformance/corpus.json")).unwrap();
        corpus["gguf_fixtures"] = json!({"fixtures/gguf/basic.gguf": {
            "sha256": hash_file(&root.path().join(path)).unwrap()
        }});
        write_replay_fixture_json(root.path(), "conformance/corpus.json", &corpus);
        for path in [path, "conformance/corpus.json"] {
            report["payload"]["source_artifacts"][path] =
                json!(hash_file(&root.path().join(path)).unwrap());
        }
        run_replay_fixture_git(root.path(), &["add", "conformance"]);
        run_replay_fixture_git(
            root.path(),
            &["commit", "--quiet", "-m", "test: add GGUF fixture"],
        );
        let seen = BTreeSet::from([
            "state",
            "transforms",
            "probe_oob_take",
            "probe_singular_inv",
        ]);
        let replay_root = root.path().join("conformance/target");

        validate_strict_replay(&replay_root, &report, &seen).unwrap();
        fs::write(root.path().join(path), b"tampered GGUF fixture").unwrap();
        run_replay_fixture_git(root.path(), &["add", path]);
        run_replay_fixture_git(
            root.path(),
            &["commit", "--quiet", "-m", "test: tamper with GGUF fixture"],
        );
        let error = validate_strict_replay(&replay_root, &report, &seen).unwrap_err();
        assert!(error.contains("digest mismatch"), "{error}");
    }

    #[test]
    fn replay_sources_stay_anchored_to_the_report_introduction() {
        let root = tempfile::tempdir().unwrap();
        let run_git = |args: &[&str]| {
            let status = Command::new("git")
                .args(args)
                .current_dir(root.path())
                .status()
                .unwrap();
            assert!(status.success());
        };
        run_git(&["init", "--quiet"]);
        run_git(&["config", "user.email", "qualification@example.invalid"]);
        run_git(&["config", "user.name", "Qualification"]);
        fs::write(root.path().join("replay-report.json"), "historical\n").unwrap();
        run_git(&["add", "replay-report.json"]);
        run_git(&["commit", "--quiet", "-m", "test: add replay report"]);
        let introduction = git(root.path(), &["rev-parse", "HEAD"]).unwrap();

        fs::write(root.path().join("replay-report.json"), "metadata update\n").unwrap();
        run_git(&["commit", "--quiet", "-am", "test: update replay report"]);

        assert_eq!(
            replay_introduction_commit(root.path(), "replay-report.json").unwrap(),
            introduction
        );
    }

    #[test]
    fn committed_evidence_rejects_missing_and_tampered_files() {
        let root = tempfile::tempdir().unwrap();
        let run_git = |args: &[&str]| {
            let status = Command::new("git")
                .args(args)
                .current_dir(root.path())
                .status()
                .unwrap();
            assert!(status.success());
        };
        run_git(&["init", "--quiet"]);
        run_git(&["config", "user.email", "qualification@example.invalid"]);
        run_git(&["config", "user.name", "Qualification"]);
        let evidence = root.path().join("evidence.json");
        fs::write(&evidence, "{}\n").unwrap();
        run_git(&["add", "evidence.json"]);
        run_git(&["commit", "--quiet", "-m", "test: capture evidence"]);

        require_tracked_unchanged(root.path(), &evidence).unwrap();
        fs::write(&evidence, "{\"tampered\":true}\n").unwrap();
        assert!(require_tracked_unchanged(root.path(), &evidence)
            .unwrap_err()
            .contains("differs from HEAD"));
        assert!(
            require_tracked_unchanged(root.path(), &root.path().join("missing.json"))
                .unwrap_err()
                .contains("not tracked")
        );
    }

    #[test]
    fn tuple_requires_target_commit_and_nested_version() {
        validate_tuple_values(TARGET_COMMIT, TARGET_MLX).unwrap();
        assert!(validate_tuple_values(OLD_COMMIT, TARGET_MLX).is_err());
        assert!(validate_tuple_values(TARGET_COMMIT, "0.32.1").is_err());
    }
}
