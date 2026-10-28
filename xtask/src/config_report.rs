use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Configuration {
    pub id: String,
    pub required: bool,
    pub checkpoints_required: bool,
    target_triple: String,
    default_features: bool,
    features: Vec<String>,
    profile: String,
    toolchain: String,
    packages: Vec<String>,
    native: BTreeMap<String, String>,
    resolved_features: BTreeMap<String, Vec<String>>,
    allow_empty: Vec<String>,
}

pub(crate) fn configurations(matrix: &Value) -> Result<Vec<Configuration>, String> {
    let configs: Vec<Configuration> = serde_json::from_value(matrix["builds"].clone())
        .map_err(|e| format!("invalid configuration matrix: {e}"))?;
    let mut ids = BTreeSet::new();
    if !configs.iter().any(|c| {
        c.id == "macos-arm64-default"
            && c.required
            && c.checkpoints_required
            && c.toolchain == "stable"
    }) {
        return Err(
            "configuration matrix needs the required stable default checkpoint configuration"
                .into(),
        );
    }
    for c in &configs {
        if c.id.is_empty()
            || !c.id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
            || !ids.insert(&c.id)
        {
            return Err(format!("invalid or duplicate configuration id {}", c.id));
        }
        if c.target_triple != "aarch64-apple-darwin"
            || !["debug", "release"].contains(&c.profile.as_str())
            || c.toolchain.is_empty()
        {
            return Err(format!(
                "{} has wrong configuration target, profile or toolchain",
                c.id
            ));
        }
        for values in [&c.features, &c.packages, &c.allow_empty] {
            if values.iter().any(|v| v.is_empty()) || !values.windows(2).all(|p| p[0] < p[1]) {
                return Err(format!("{} lists must be sorted and unique", c.id));
            }
        }
        for (key, expected) in [
            (
                "MLX_BUILD_METAL",
                if c.features.iter().any(|f| f == "metal") {
                    "ON"
                } else {
                    "OFF"
                },
            ),
            (
                "MLX_BUILD_ACCELERATE",
                if c.features.iter().any(|f| f == "accelerate") {
                    "ON"
                } else {
                    "OFF"
                },
            ),
            (
                "MLX_METAL_JIT",
                if c.features.iter().any(|f| f == "metal-jit") {
                    "ON"
                } else {
                    "OFF"
                },
            ),
            (
                "CMAKE_BUILD_TYPE",
                if c.profile == "debug" {
                    "Debug"
                } else {
                    "Release"
                },
            ),
        ] {
            if c.native.get(key).map(String::as_str) != Some(expected) {
                return Err(format!(
                    "{} native {key} disagrees with configuration",
                    c.id
                ));
            }
        }
        for name in ["mlx-rs", "mlx-sys", "mlx-tests"] {
            let Some(features) = c.resolved_features.get(name) else {
                return Err(format!("{} lacks resolved features for {name}", c.id));
            };
            if !features.windows(2).all(|p| p[0] < p[1]) || features.iter().any(|f| f.is_empty()) {
                return Err(format!(
                    "{} resolved features must be sorted and unique for {name}",
                    c.id
                ));
            }
            for backend in ["metal", "accelerate"] {
                if features.iter().any(|f| f == backend) != c.features.iter().any(|f| f == backend)
                {
                    return Err(format!("{} resolved features disagree for {name}", c.id));
                }
            }
        }
    }
    Ok(configs)
}

fn read_json(path: &Path) -> Result<Value, String> {
    serde_json::from_slice(&fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?)
        .map_err(|e| e.to_string())
}

fn output(root: &Path, command: &[&str]) -> Result<String, String> {
    let out = Command::new(command[0])
        .args(&command[1..])
        .current_dir(root)
        .output()
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!(
            "{}: {}",
            command.join(" "),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

pub(crate) fn identity(root: &Path) -> Result<Value, String> {
    if !root.join("Cargo.lock").exists() {
        output(root, &["cargo", "generate-lockfile"])?;
    }
    let native = root.join("mlx-sys/src/mlx-c");
    let cmake = fs::read_to_string(native.join("CMakeLists.txt")).map_err(|e| e.to_string())?;
    let version = cmake
        .lines()
        .find_map(|line| {
            line.trim()
                .strip_prefix("GIT_TAG v")
                .map(|v| v.trim_end_matches(')'))
        })
        .ok_or("missing nested MLX version")?;
    let hash = output(root, &["shasum", "-a", "256", "Cargo.lock"])?;
    Ok(json!({
        "source_commit": output(root, &["git", "rev-parse", "HEAD"])?,
        "mlx_c_commit": output(&native, &["git", "rev-parse", "HEAD"])?,
        "nested_mlx_version": version,
        "cargo_lock_sha256": hash.split_whitespace().next().ok_or("missing lock hash")?
    }))
}

fn test_command(c: &Configuration) -> Vec<String> {
    let mut cmd = vec![
        "cargo".into(),
        "test".into(),
        "--locked".into(),
        "--color=never".into(),
        "--message-format=json".into(),
        "--target".into(),
        c.target_triple.clone(),
    ];
    if c.packages.is_empty() {
        cmd.push("--workspace".into());
    }
    for p in &c.packages {
        cmd.extend(["-p".into(), p.clone()]);
    }
    if !c.default_features {
        cmd.push("--no-default-features".into());
    }
    if !c.features.is_empty() {
        cmd.extend(["--features".into(), c.features.join(",")]);
    }
    if c.profile != "debug" {
        cmd.extend(["--profile".into(), c.profile.clone()]);
    }
    cmd.extend(["--".into(), "--test-threads=1".into(), "--nocapture".into()]);
    cmd
}

fn metadata(root: &Path) -> Result<Value, String> {
    serde_json::from_str(&output(
        root,
        &[
            "cargo",
            "metadata",
            "--locked",
            "--no-deps",
            "--format-version=1",
        ],
    )?)
    .map_err(|e| e.to_string())
}

fn inventory(
    metadata: &Value,
    c: &Configuration,
    features: &Value,
) -> Result<BTreeSet<String>, String> {
    let mut ids = BTreeSet::new();
    for p in metadata["packages"].as_array().ok_or("missing packages")? {
        let name = p["name"].as_str().ok_or("missing package name")?;
        if !c.packages.is_empty() && !c.packages.iter().any(|n| n == name) {
            continue;
        }
        for t in p["targets"].as_array().ok_or("missing targets")? {
            if t["required-features"].as_array().is_some_and(|required| {
                required.iter().any(|f| {
                    !features[name]
                        .as_array()
                        .is_some_and(|enabled| enabled.contains(f))
                })
            }) {
                continue;
            }
            let kind = t["kind"][0].as_str().ok_or("missing target kind")?;
            if !["lib", "proc-macro", "bin", "test"].contains(&kind) {
                continue;
            }
            let kind = if kind == "proc-macro" { "lib" } else { kind };
            let target = t["name"].as_str().ok_or("missing target name")?;
            if t["test"] == true {
                ids.insert(format!("{name}::{kind}::{target}"));
            }
            if t["doctest"] == true {
                ids.insert(format!("{name}::doctest::{target}"));
            }
        }
    }
    Ok(ids)
}

fn parse_runs(log: &str) -> Result<Vec<Value>, String> {
    let mut runs: Vec<Value> = Vec::new();
    let mut groups = Vec::new();
    for line in log.lines().map(str::trim) {
        if serde_json::from_str::<Value>(line)
            .ok()
            .is_some_and(|message| {
                matches!(
                    message["reason"].as_str(),
                    Some(
                        "compiler-artifact"
                            | "compiler-message"
                            | "build-script-executed"
                            | "build-finished"
                    )
                )
            })
        {
            continue;
        }
        let heading = if let Some(doc) = line.strip_prefix("Doc-tests ") {
            Some(format!("doc:{doc}"))
        } else if line.starts_with("Running ") {
            line.rsplit_once(" (")
                .and_then(|(_, p)| p.strip_suffix(')'))
                .map(str::to_owned)
        } else {
            None
        };
        if let Some(id) = heading {
            runs.push(json!({"id": id, "passed": 0, "failed": 0, "ignored": 0, "filtered_out": 0, "not_run": false, "not_run_reasons": [], "excluded_by_configuration": false, "summaries": 0, "outcome": "incomplete", "nested_summaries": []}));
            groups.clear();
            continue;
        }
        if let Some(count) = line
            .strip_prefix("running ")
            .and_then(|s| s.strip_suffix(" tests").or_else(|| s.strip_suffix(" test")))
        {
            let count: u64 = count.parse().map_err(|_| "invalid announced test count")?;
            if groups.is_empty() {
                let run = runs.last_mut().ok_or("test group outside a run")?;
                if run["outcome"] != "fail" {
                    run["outcome"] = "incomplete".into();
                }
            }
            groups.push(count);
        }
        if line.contains("NOT RUN") {
            let run = runs.last_mut().ok_or("NOT RUN outside a test run")?;
            run["not_run"] = true.into();
            run["not_run_reasons"]
                .as_array_mut()
                .unwrap()
                .push(line[line.find("NOT RUN").unwrap()..].into());
        }
        if let Some(summary) = line.strip_prefix("test result: ") {
            if !summary.contains("; finished in ") {
                continue;
            }
            let run = runs.last_mut().ok_or("test summary outside a run")?;
            let (status, counts) = summary
                .split_once(". ")
                .ok_or("unrecognized test summary")?;
            if !["ok", "FAILED"].contains(&status) {
                return Err("unrecognized test summary status".into());
            }
            let mut parsed = json!({"outcome": if status == "ok" { "pass" } else { "fail" }});
            for (field, suffix) in [
                ("passed", " passed"),
                ("failed", " failed"),
                ("ignored", " ignored"),
                ("filtered_out", " filtered out"),
            ] {
                let n: u64 = counts
                    .split(';')
                    .find_map(|s| s.trim().strip_suffix(suffix))
                    .ok_or("missing test count")?
                    .parse()
                    .map_err(|_| "invalid test count")?;
                parsed[field] = n.into();
            }
            if let Some(announced) = groups.pop() {
                parsed["announced_tests"] = announced.into();
                if announced
                    != parsed["passed"].as_u64().unwrap()
                        + parsed["failed"].as_u64().unwrap()
                        + parsed["ignored"].as_u64().unwrap()
                {
                    return Err("test counts disagree with announced group size".into());
                }
            }
            if !groups.is_empty() {
                run["nested_summaries"].as_array_mut().unwrap().push(parsed);
                continue;
            }
            for field in ["passed", "failed", "ignored", "filtered_out"] {
                run[field] =
                    (run[field].as_u64().unwrap() + parsed[field].as_u64().unwrap()).into();
            }
            run["summaries"] = (run["summaries"].as_u64().unwrap() + 1).into();
            if run["outcome"] != "fail" {
                run["outcome"] = if status == "ok" { "pass" } else { "fail" }.into();
            }
        }
    }
    Ok(runs)
}

struct BuildEvidence {
    features: Value,
    rust_profiles: Value,
    native: Value,
    binaries: BTreeMap<String, String>,
}

fn build_evidence(log: &str, metadata: &Value) -> Result<BuildEvidence, String> {
    let packages: BTreeMap<_, _> = metadata["packages"]
        .as_array()
        .ok_or("missing packages")?
        .iter()
        .map(|p| (p["id"].as_str().unwrap(), p["name"].as_str().unwrap()))
        .collect();
    let mut features = serde_json::Map::new();
    let mut rust_profiles = serde_json::Map::new();
    let mut native = serde_json::Map::new();
    let mut binaries = BTreeMap::new();
    for line in log.lines() {
        let Ok(message) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let Some(name) = message["package_id"]
            .as_str()
            .and_then(|id| packages.get(id))
        else {
            continue;
        };
        if message["reason"] == "compiler-artifact"
            && message["target"]["kind"][0] != "custom-build"
        {
            let profile = json!({
                "opt_level": message["profile"]["opt_level"],
                "debug_assertions": message["profile"]["debug_assertions"],
            });
            if let Some(previous) = rust_profiles.insert((*name).into(), profile.clone()) {
                if previous != profile {
                    return Err(format!("multiple Rust profiles for {name}"));
                }
            }
            if let Some(previous) = features.insert((*name).into(), message["features"].clone()) {
                if previous != message["features"] {
                    return Err(format!("multiple feature resolutions for {name}"));
                }
            }
            if message["profile"]["test"] == true {
                if let Some(executable) = message["executable"].as_str() {
                    let kind = message["target"]["kind"][0]
                        .as_str()
                        .ok_or("missing artifact kind")?;
                    let kind = if kind == "proc-macro" { "lib" } else { kind };
                    binaries.insert(
                        executable.into(),
                        format!(
                            "{name}::{kind}::{}",
                            message["target"]["name"]
                                .as_str()
                                .ok_or("missing artifact name")?
                        ),
                    );
                }
            }
        }
        if message["reason"] == "build-script-executed" && *name == "mlx-sys" {
            let path = Path::new(
                message["out_dir"]
                    .as_str()
                    .ok_or("missing native out_dir")?,
            )
            .join("build/CMakeCache.txt");
            let cache =
                fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            for line in cache.lines() {
                if let Some((key, value)) = line.split_once('=') {
                    if let Some((key, _)) = key.split_once(':') {
                        native.insert(key.into(), value.into());
                    }
                }
            }
            let source = native
                .get("mlx_SOURCE_DIR")
                .and_then(Value::as_str)
                .map(PathBuf::from)
                .unwrap_or_else(|| path.parent().unwrap().join("_deps/mlx-src"));
            native.insert(
                "mlx_version".into(),
                output(&source, &["git", "describe", "--tags", "--exact-match"])?
                    .trim_start_matches('v')
                    .into(),
            );
            native.insert(
                "mlx_commit".into(),
                output(&source, &["git", "rev-parse", "HEAD"])?.into(),
            );
            native.insert(
                "cache_path".into(),
                path.to_string_lossy().into_owned().into(),
            );
        }
    }
    Ok(BuildEvidence {
        features: features.into(),
        rust_profiles: rust_profiles.into(),
        native: native.into(),
        binaries,
    })
}

fn execution_configuration(c: &Configuration, bump_run: bool) -> Configuration {
    let mut c = c.clone();
    c.checkpoints_required &= bump_run;
    c
}

fn report_exit_code(code: i32, report: &Value) -> i32 {
    if code == 0 && report["verdict"] != "pass" {
        1
    } else {
        code
    }
}

fn checkpoint_excluded(c: &Configuration, run: &Value) -> bool {
    !c.checkpoints_required
        && run["id"] == "mlx-lm::test::real_checkpoints"
        && run["not_run"] == true
        && run["not_run_reasons"] == json!(["NOT RUN: local real-checkpoint manifest not supplied"])
}

fn validate_report(
    c: &Configuration,
    report: &Value,
    identity: &Value,
    expected: &BTreeSet<String>,
) -> Result<(), String> {
    let fail = |why: &str| format!("{}: {why}", c.id);
    if report["schema_version"] != 1 || report["configuration_id"] != c.id {
        return Err(fail("wrong report schema or configuration"));
    }
    for key in [
        "source_commit",
        "mlx_c_commit",
        "nested_mlx_version",
        "cargo_lock_sha256",
    ] {
        if report[key] != identity[key] || report[key].as_str().is_none_or(str::is_empty) {
            return Err(fail(&format!("stale or missing {key}")));
        }
    }
    if report["dirty"] != false {
        return Err(fail("dirty checkout"));
    }
    if report["exit_code"] != 0 || report["verdict"] != "pass" {
        return Err(fail("failed or skipped execution"));
    }
    if report["checkpoints_required"] != c.checkpoints_required
        || report["profile"] != c.profile
        || report["toolchain"] != c.toolchain
        || report["target_triple"] != c.target_triple
        || report["command"] != json!(test_command(c))
    {
        return Err(fail("command, profile, target or toolchain mismatch"));
    }
    for field in ["rustc", "macos", "sdk"] {
        if report[field].as_str().is_none_or(str::is_empty) {
            return Err(fail(&format!("missing {field}")));
        }
    }
    let rustc = report["rustc"].as_str().unwrap();
    if c.toolchain == "stable" {
        if rustc.contains("nightly") || rustc.contains("beta") {
            return Err(fail("toolchain is not stable"));
        }
    } else if rustc.split_whitespace().nth(1) != Some(c.toolchain.as_str()) {
        return Err(fail("rustc version does not match toolchain"));
    }
    for (name, features) in &c.resolved_features {
        if report["resolved_features"][name] != json!(features) {
            return Err(fail(&format!("resolved features mismatch for {name}")));
        }
        let profile = &report["rust_profiles"][name];
        if profile["opt_level"] != if c.profile == "release" { "3" } else { "0" }
            || profile["debug_assertions"] != (c.profile == "debug")
        {
            return Err(fail(&format!("compiled Rust profile mismatch for {name}")));
        }
    }
    if report["native"]["mlx_version"] != identity["nested_mlx_version"]
        || report["native"]["mlx_commit"]
            .as_str()
            .is_none_or(str::is_empty)
    {
        return Err(fail("built nested MLX identity mismatch"));
    }
    for (key, value) in &c.native {
        if report["native"][key] != *value {
            return Err(fail(&format!("native setting mismatch: {key}")));
        }
    }
    let runs = report["tests"]
        .as_array()
        .ok_or_else(|| fail("missing test inventory"))?;
    let mut found = BTreeSet::new();
    for run in runs {
        let id = run["id"].as_str().ok_or_else(|| fail("missing test id"))?;
        if !found.insert(id.to_owned()) {
            return Err(fail("duplicate test run"));
        }
        if run["outcome"] != "pass"
            || run["excluded_by_configuration"] != checkpoint_excluded(c, run)
            || (run["not_run"] != false && !checkpoint_excluded(c, run))
            || run["failed"] != 0
            || run["filtered_out"] != 0
            || run["summaries"].as_u64().unwrap_or(0) == 0
        {
            return Err(fail(&format!(
                "failed, skipped or incomplete test run {id}"
            )));
        }
        let passed = run["passed"]
            .as_u64()
            .ok_or_else(|| fail("invalid passed count"))?;
        if passed == 0 && !c.allow_empty.iter().any(|s| s == id) {
            return Err(fail(&format!("zero tests in {id}")));
        }
    }
    if c.id == "macos-arm64-default" {
        for id in [
            "mlx-lm::test::parity",
            "mlx-lm::test::sentinel",
            "mlx-tests::test::conformance",
            "mlx-tests::test::state_optimizers",
            "mlx-tests::test::state_compile",
            "mlx-tests::test::state_transforms",
        ] {
            let run = runs
                .iter()
                .find(|run| run["id"] == id)
                .ok_or_else(|| fail(&format!("missing required test {id}")))?;
            if run["passed"].as_u64().unwrap_or(0) == 0
                || run["ignored"] != 0
                || run["summaries"] != 1
            {
                return Err(fail(&format!("incomplete required test {id}")));
            }
        }
    }
    if &found != expected || expected.is_empty() {
        return Err(fail("test inventory differs from workspace targets"));
    }
    Ok(())
}

pub(crate) fn run(root: &Path, args: &[String]) -> i32 {
    match execute(root, args) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("configuration report: {error}");
            1
        }
    }
}

fn execute(root: &Path, args: &[String]) -> Result<i32, String> {
    let configs = configurations(&read_json(
        &root.join("ledger/supported-feature-matrix.json"),
    )?)?;
    if args.len() != 3 || args[1] != "--report" {
        return Err("usage: config-report <configuration-id> --report <path>".into());
    }
    let c = configs
        .iter()
        .find(|c| c.id == args[0])
        .ok_or("unknown configuration id")?;
    let c = &execution_configuration(c, std::env::var("MLX_RS_BUMP_RUN").as_deref() == Ok("1"));
    let path = PathBuf::from(&args[2]);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let identity = identity(root)?;
    let mut report = identity.clone();
    report["schema_version"] = 1.into();
    report["configuration_id"] = c.id.clone().into();
    report["github_sha"] = std::env::var("GITHUB_SHA")
        .ok()
        .filter(|v| !v.is_empty())
        .into();
    report["github_pr_head_sha"] = std::env::var("GITHUB_PR_HEAD_SHA")
        .ok()
        .filter(|v| !v.is_empty())
        .into();
    report["dirty"] = (!output(
        root,
        &["git", "status", "--porcelain", "--untracked-files=normal"],
    )?
    .is_empty())
    .into();
    report["rustc"] = output(root, &["rustc", "--version", "--verbose"])?.into();
    report["macos"] = output(root, &["sw_vers", "-productVersion"])?.into();
    report["sdk"] = output(root, &["xcrun", "--sdk", "macosx", "--show-sdk-version"])?.into();
    report["checkpoints_required"] = c.checkpoints_required.into();
    report["profile"] = c.profile.clone().into();
    report["toolchain"] = c.toolchain.clone().into();
    report["target_triple"] = c.target_triple.clone().into();
    report["cargo_target_dir"] = std::env::var("CARGO_TARGET_DIR")
        .map_err(|_| "CARGO_TARGET_DIR must isolate this configuration")?
        .into();
    let command = test_command(c);
    report["command"] = json!(command);
    report["verdict"] = "fail".into();
    report["exit_code"] = Value::Null;
    fs::write(&path, serde_json::to_vec_pretty(&report).unwrap()).map_err(|e| e.to_string())?;
    let meta = metadata(root)?;
    let log_path = path.with_extension("log");
    let log = fs::File::create(&log_path).map_err(|e| e.to_string())?;
    eprintln!("{}; output: {}", command.join(" "), log_path.display());
    let started = std::time::Instant::now();
    let mut process = Command::new(&command[0]);
    if !c.checkpoints_required {
        process.env_remove("MLX_LM_REAL_CHECKPOINTS");
    }
    let status = process
        .args(&command[1..])
        .current_dir(root)
        .stdout(Stdio::from(log.try_clone().map_err(|e| e.to_string())?))
        .stderr(Stdio::from(log))
        .status()
        .map_err(|e| e.to_string())?;
    report["elapsed_seconds"] = started.elapsed().as_secs_f64().into();
    use std::os::unix::process::ExitStatusExt;
    let code = status
        .code()
        .unwrap_or_else(|| 128 + status.signal().unwrap_or(1));
    report["exit_code"] = code.into();
    let collected = (|| {
        let log = fs::read_to_string(&log_path).map_err(|e| e.to_string())?;
        if c.checkpoints_required {
            report["checkpoints"] = crate::checkpoints::checkpoint_evidence(root, &log)?;
        }
        let evidence = build_evidence(&log, &meta)?;
        report["resolved_features"] = evidence.features;
        report["rust_profiles"] = evidence.rust_profiles;
        report["native"] = evidence.native;
        let mut runs = parse_runs(&log)?;
        let expected = inventory(&meta, c, &report["resolved_features"])?;
        for run in &mut runs {
            let id = run["id"].as_str().ok_or("missing run id")?;
            let id = if let Some(doc) = id.strip_prefix("doc:") {
                expected
                    .iter()
                    .find(|id| id.ends_with(&format!("::doctest::{doc}")))
                    .cloned()
                    .ok_or_else(|| format!("unknown doctest {doc}"))?
            } else {
                evidence
                    .binaries
                    .get(id)
                    .cloned()
                    .ok_or_else(|| format!("unknown test executable {id}"))?
            };
            run["id"] = id.into();
            run["excluded_by_configuration"] = checkpoint_excluded(c, run).into();
        }
        report["tests"] = json!(runs);
        if identity != self::identity(root)?
            || !output(
                root,
                &["git", "status", "--porcelain", "--untracked-files=normal"],
            )?
            .is_empty()
        {
            report["dirty"] = true.into();
        }
        report["verdict"] = "pass".into();
        validate_report(c, &report, &identity, &expected)
    })();
    if let Err(error) = collected {
        report["verdict"] = "fail".into();
        report["error"] = error.into();
    }
    fs::write(&path, serde_json::to_vec_pretty(&report).unwrap())
        .map_err(|error| format!("failed to write {}: {error}", path.display()))?;
    eprintln!(
        "configuration {}: {}; cargo exit {code}",
        c.id, report["verdict"]
    );
    Ok(report_exit_code(code, &report))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captures_binary_counts_and_doctest_groups() {
        let runs = parse_runs("Running unittests src/lib.rs (/tmp/lib-abcd)\nrunning 2 tests\ntest result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.1s\nDoc-tests demo\nrunning 1 test\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.1s\nrunning 1 test\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.1s").unwrap();
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0]["passed"], 2);
        assert_eq!(runs[1]["passed"], 2);
    }

    #[test]
    fn detects_not_run_even_with_successful_summary() {
        let runs = parse_runs("Doc-tests demo\nNOT RUN: fixture absent\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0s").unwrap();
        assert_eq!(runs[0]["not_run"], true);
    }

    #[test]
    fn unrecognized_summary_cannot_complete_run() {
        let runs = parse_runs("Doc-tests demo\ntest result: unexpected").unwrap();
        assert_eq!(runs[0]["outcome"], "incomplete");
    }
}

#[cfg(test)]
#[path = "config_report_tests.rs"]
mod admission_tests;
