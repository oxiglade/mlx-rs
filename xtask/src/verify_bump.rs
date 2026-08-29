use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};

const OLD_COMMIT: &str = "a1290d221f92bd020af805b7d14207eee4ec973b";
const TARGET_COMMIT: &str = "c74db5307cc8ce122f48d97ef951b30578674e7f";
const OLD_MLX: &str = "0.30.6";
const TARGET_MLX: &str = "0.32.2";
const LOCAL_FFI_PROCEDURE: &str = "ROADMAP.md#tranche-2-leak-and-use-after-free-gate-done";
const ALLOWED_REPLAY_VERDICTS: &[&str] = &[
    "identical",
    "value_changed",
    "dtype_or_shape_changed",
    "error_behavior_changed",
    "recipe_failed",
];
const CHECK_IDS: &[&str] = &[
    "tuple",
    "boundary",
    "race",
    "ffi",
    "semantic",
    "ledger",
    "fingerprints",
    "feature_matrix",
    "state",
    "replay",
    "sentinel",
];

struct RequiredSuite {
    id: &'static str,
    command: &'static [&'static str],
}

const REQUIRED_SUITES: &[RequiredSuite] = &[
    RequiredSuite {
        id: "conformance",
        command: &[
            "cargo",
            "test",
            "-p",
            "mlx-tests",
            "--test",
            "conformance",
            "--",
            "--test-threads=1",
        ],
    },
    RequiredSuite {
        id: "state_optimizers",
        command: &[
            "cargo",
            "test",
            "-p",
            "mlx-tests",
            "--test",
            "state_optimizers",
            "--",
            "--test-threads=1",
        ],
    },
    RequiredSuite {
        id: "state_compile",
        command: &[
            "cargo",
            "test",
            "-p",
            "mlx-tests",
            "--test",
            "state_compile",
            "--",
            "--test-threads=1",
        ],
    },
    RequiredSuite {
        id: "state_transforms",
        command: &[
            "cargo",
            "test",
            "-p",
            "mlx-tests",
            "--test",
            "state_transforms",
            "--",
            "--test-threads=1",
        ],
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Pre,
    Admit,
}

#[derive(Debug)]
enum SuiteSource {
    File(PathBuf),
    Rerun,
}

#[derive(Debug)]
struct Options {
    mode: Mode,
    boundary_base: Option<String>,
    ffi_report: Option<PathBuf>,
    suite_source: SuiteSource,
    replay_report: PathBuf,
    waivers: PathBuf,
    allow_local_ffi: bool,
}

impl Options {
    fn parse(repo_root: &Path, args: &[String]) -> Result<Self, String> {
        let mut mode = None;
        let mut ffi_report = None;
        let mut boundary_base = None;
        let mut suite_source = None;
        let mut replay_report = repo_root.join("conformance/target/replay-report.json");
        let mut waivers = repo_root.join("admission/waivers.json");
        let mut allow_local_ffi = false;
        let mut index = 0;
        while index < args.len() {
            match args[index].as_str() {
                "--pre" if mode.is_none() => mode = Some(Mode::Pre),
                "--admit" if mode.is_none() => mode = Some(Mode::Admit),
                "--ffi-report" if ffi_report.is_none() => {
                    index += 1;
                    ffi_report = Some(PathBuf::from(args.get(index).ok_or_else(usage)?));
                }
                "--boundary-base" if boundary_base.is_none() => {
                    index += 1;
                    boundary_base = Some(args.get(index).ok_or_else(usage)?.clone());
                }
                "--suite-results" if suite_source.is_none() => {
                    index += 1;
                    suite_source = Some(SuiteSource::File(PathBuf::from(
                        args.get(index).ok_or_else(usage)?,
                    )));
                }
                "--rerun-suites" if suite_source.is_none() => {
                    suite_source = Some(SuiteSource::Rerun)
                }
                "--replay-report" => {
                    index += 1;
                    replay_report = PathBuf::from(args.get(index).ok_or_else(usage)?);
                }
                "--waivers" => {
                    index += 1;
                    waivers = PathBuf::from(args.get(index).ok_or_else(usage)?);
                }
                "--allow-local-ffi" if !allow_local_ffi => allow_local_ffi = true,
                _ => return Err(usage()),
            }
            index += 1;
        }
        Ok(Self {
            mode: mode.ok_or_else(usage)?,
            boundary_base,
            ffi_report,
            suite_source: suite_source.ok_or_else(usage)?,
            replay_report,
            waivers,
            allow_local_ffi,
        })
    }
}

fn usage() -> String {
    "usage: cargo run -p xtask -- verify-bump (--pre|--admit) [--boundary-base <ref>] [--ffi-report <file>] (--suite-results <file>|--rerun-suites) [--replay-report <file>] [--waivers <file>] [--allow-local-ffi]".to_owned()
}

pub(crate) fn run(repo_root: &Path, args: &[String]) -> i32 {
    let report = match Options::parse(repo_root, args) {
        Ok(options) => aggregate(repo_root, &options),
        Err(error) => json!({
            "schema_version": 1,
            "command": "verify-bump",
            "verdict": "fail",
            "errors": [error]
        }),
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("serialize verify-bump report")
    );
    i32::from(report["verdict"] != "pass")
}

fn aggregate(repo_root: &Path, options: &Options) -> Value {
    let workspace_commit = git(repo_root, &["rev-parse", "HEAD"]);
    let tuple = verify_tuple(repo_root, options.mode);
    let expected_mlx_c = match options.mode {
        Mode::Pre => OLD_COMMIT,
        Mode::Admit => TARGET_COMMIT,
    };
    let suites = match (&options.suite_source, workspace_commit.as_deref()) {
        (SuiteSource::File(path), Ok(commit)) => read_json(&resolve_path(repo_root, path))
            .and_then(|report| {
                validate_suite_results(&report, commit, expected_mlx_c)?;
                Ok(report)
            })
            .map_err(|error| format!("suite results: {error}")),
        (SuiteSource::Rerun, Ok(commit)) => {
            let report = rerun_suites(repo_root, commit, expected_mlx_c);
            validate_suite_results(&report, commit, expected_mlx_c).map(|_| report)
        }
        (_, Err(error)) => Err(error.clone()),
    };
    let ledger = crate::verify_ledger::verify_value(repo_root, &[]);
    let boundary_args = options
        .boundary_base
        .as_ref()
        .map(|base| vec!["--base".to_owned(), base.clone()])
        .unwrap_or_default();
    let boundary = match (options.mode, options.boundary_base.as_deref()) {
        (Mode::Admit, None) => Err(
            "--admit requires --boundary-base <old-tuple-ref> for diff-based oracle separation"
                .to_owned(),
        ),
        (Mode::Admit, Some(base)) => validate_boundary_base(repo_root, base)
            .and_then(|_| crate::verify_oracle_boundary::verify_value(repo_root, &boundary_args)),
        (Mode::Pre, _) => crate::verify_oracle_boundary::verify_value(repo_root, &boundary_args),
    };
    let fingerprint = match &tuple {
        Ok(_) => [OLD_COMMIT, TARGET_COMMIT]
            .into_iter()
            .map(|reference| {
                let path = repo_root.join(format!("ledger/mlx-c-{reference}.json"));
                crate::fingerprint::verify_committed(repo_root, reference, &path)
                    .map(|report| (reference, report))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()
            .map(|reports| json!(reports)),
        Err(error) => Err(error.clone()),
    };
    let ffi = match (&options.ffi_report, workspace_commit.as_deref()) {
        (Some(path), Ok(commit)) => crate::verify_ffi::discovered_target_ids(repo_root)
            .and_then(|inventory| {
                read_json(&resolve_path(repo_root, path)).and_then(|report| {
                    validate_ffi_report(
                        &report,
                        commit,
                        expected_mlx_c,
                        options.allow_local_ffi,
                        &inventory.into_iter().collect(),
                    )
                })
            }),
        (None, _) => Err(format!(
            "missing FFI report; run verify-ffi --guard-malloc and pass --ffi-report <file>{}; lower-trust procedure: {LOCAL_FFI_PROCEDURE}",
            if options.allow_local_ffi {
                ""
            } else {
                "; local evidence also requires --allow-local-ffi"
            }
        )),
        (_, Err(error)) => Err(error.clone()),
    };
    let replay_report = resolve_path(repo_root, &options.replay_report);
    let replay_path = if options.mode == Mode::Admit
        && replay_report != repo_root.join("conformance/target/replay-report.json")
    {
        Err(
            "--admit requires the canonical conformance/target/replay-report.json artifact"
                .to_owned(),
        )
    } else {
        require_tracked_unchanged(repo_root, &replay_report)
    };
    let replay = replay_path
        .and_then(|_| read_json(&replay_report))
        .and_then(|report| {
            let parent = replay_report
                .parent()
                .ok_or("replay report path has no parent")?;
            validate_replay_document(parent, &report, true)
        });
    let waiver_manifest = resolve_path(repo_root, &options.waivers);
    let waiver_path = if options.mode == Mode::Admit
        && waiver_manifest != repo_root.join("admission/waivers.json")
    {
        Err("--admit requires the canonical admission/waivers.json manifest".to_owned())
    } else {
        require_tracked_unchanged(repo_root, &waiver_manifest)
    };
    let waiver_result = waiver_path
        .and_then(|_| read_json(&waiver_manifest))
        .and_then(|document| {
            let ledger_entries = ledger_waiver_scopes(repo_root)?;
            validate_waivers(&document, &current_date(), &ledger_entries)
        });
    let waiver_error = waiver_result.as_ref().err().cloned();
    let waivers = waiver_result.unwrap_or_default();

    let checks = vec![
        check(1, "tuple", "immutable dependency tuple", tuple.map(|_| json!({
            "current_mlx_c_commit": expected_mlx_c,
            "required_nested_mlx": if options.mode == Mode::Pre { OLD_MLX } else { TARGET_MLX }
        }))),
        check(2, "boundary", "oracle boundary and digest integrity", boundary),
        check(
            3,
            "race",
            "error registration race regression",
            verify_race_regression(repo_root),
        ),
        check(4, "ffi", "qualified FFI safety gate", ffi),
        check(
            5,
            "semantic",
            "strict semantic conformance",
            suites.as_ref().map_err(Clone::clone).and_then(|report| {
                require_suite_pass(report, "conformance")
            }),
        ),
        check(6, "ledger", "classified target delta ledger", ledger.clone()),
        check(6, "fingerprints", "canonical ABI fingerprints", fingerprint),
        check(
            6,
            "feature_matrix",
            "supported feature matrix",
            ledger.and_then(|report| {
                if report["supported_builds"].as_u64() == Some(2) {
                    Ok(json!({"supported_builds": 2}))
                } else {
                    Err("verify-ledger did not validate two supported builds".to_owned())
                }
            }),
        ),
        check(
            7,
            "state",
            "state and transform packs",
            suites.as_ref().map_err(Clone::clone).and_then(|report| {
                for id in ["state_optimizers", "state_compile", "state_transforms"] {
                    require_suite_pass(report, id)?;
                }
                Ok(json!({"suites": ["state_optimizers", "state_compile", "state_transforms"]}))
            }),
        ),
        check(8, "replay", "deterministic target-version replay", replay),
        check(
            0,
            "sentinel",
            "mlx-lm sentinel presence",
            validate_sentinel(repo_root),
        ),
        check(
            0,
            "waivers",
            "waiver manifest",
            waiver_error.map_or_else(
                || Ok(json!({"active": waivers.len()})),
                Err,
            ),
        ),
    ];
    build_verdict(options.mode, checks, waivers)
}

fn check(item: u8, id: &str, name: &str, result: Result<Value, String>) -> Value {
    match result {
        Ok(evidence) => json!({
            "id": id,
            "admission_item": if item == 0 { Value::Null } else { json!(item) },
            "name": name,
            "status": "pass",
            "evidence": evidence,
            "reason": Value::Null
        }),
        Err(reason) => json!({
            "id": id,
            "admission_item": if item == 0 { Value::Null } else { json!(item) },
            "name": name,
            "status": "fail",
            "evidence": Value::Null,
            "reason": reason
        }),
    }
}

fn build_verdict(mode: Mode, mut checks: Vec<Value>, waivers: Vec<Value>) -> Value {
    for waiver in &waivers {
        let Some(scope) = waiver.get("check").and_then(Value::as_str) else {
            continue;
        };
        if matches!(scope, "tuple" | "ledger" | "fingerprints") {
            continue;
        }
        if let Some(check) = checks
            .iter_mut()
            .find(|check| check["id"] == scope && check["status"] == "fail")
        {
            check["status"] = "waived".into();
            check["waiver"] = waiver.clone();
        }
    }
    let failed = checks.iter().any(|check| check["status"] == "fail");
    let mut admit_requirements = checks
        .iter()
        .filter(|check| check["status"] == "fail")
        .map(|check| {
            format!(
                "{}: {}",
                check["id"].as_str().unwrap_or("unknown"),
                check["reason"].as_str().unwrap_or("unmet")
            )
        })
        .collect::<Vec<_>>();
    if mode == Mode::Pre {
        admit_requirements.push(format!(
            "update mlx-c submodule to {TARGET_COMMIT}, retain nested MLX v{TARGET_MLX}, then rerun with --admit"
        ));
        admit_requirements.push(
            "pass --boundary-base <old-tuple-ref> so --admit verifies the complete bump diff"
                .to_owned(),
        );
    }
    json!({
        "schema_version": 1,
        "command": "verify-bump",
        "mode": mode,
        "recorded_tuple": {
            "old": {"mlx_c_commit": OLD_COMMIT, "mlx": OLD_MLX},
            "target": {"mlx_c_commit": TARGET_COMMIT, "mlx": TARGET_MLX}
        },
        "checks": checks,
        "active_waivers": waivers,
        "admit_requirements": admit_requirements,
        "verdict": if failed { "fail" } else { "pass" }
    })
}

fn verify_tuple(repo_root: &Path, mode: Mode) -> Result<Value, String> {
    let mlx_c = repo_root.join("mlx-sys/src/mlx-c");
    let current_commit = git(&mlx_c, &["rev-parse", "HEAD"])?;
    let current_pin = nested_pin(
        &fs::read_to_string(mlx_c.join("CMakeLists.txt"))
            .map_err(|error| format!("failed to read current mlx-c CMakeLists.txt: {error}"))?,
    )?;
    let target_cmake = git(
        &mlx_c,
        &["show", &format!("{TARGET_COMMIT}:CMakeLists.txt")],
    )?;
    let target_pin = nested_pin(&target_cmake)?;
    validate_tuple_values(
        mode,
        &current_commit,
        &current_pin,
        TARGET_COMMIT,
        &target_pin,
    )?;
    if mode == Mode::Admit {
        require_clean_tree(repo_root, "workspace")?;
        require_clean_tree(&mlx_c, "mlx-c submodule")?;
    }
    Ok(json!({
        "current_mlx_c_commit": current_commit,
        "current_nested_mlx": current_pin,
        "target_mlx_c_commit": TARGET_COMMIT,
        "target_nested_mlx": target_pin
    }))
}

fn validate_boundary_base(repo_root: &Path, base: &str) -> Result<(), String> {
    let base_commit = git(repo_root, &["rev-parse", &format!("{base}^{{commit}}")])?;
    let head_commit = git(repo_root, &["rev-parse", "HEAD"])?;
    let ancestor = Command::new("git")
        .args(["merge-base", "--is-ancestor", &base_commit, &head_commit])
        .current_dir(repo_root)
        .status()
        .map_err(|error| format!("failed to validate boundary base ancestry: {error}"))?
        .success();
    let gitlink = git(
        repo_root,
        &["rev-parse", &format!("{base_commit}:mlx-sys/src/mlx-c")],
    )?;
    validate_boundary_base_values(&base_commit, &head_commit, ancestor, &gitlink)
}

fn validate_boundary_base_values(
    base_commit: &str,
    head_commit: &str,
    ancestor: bool,
    mlx_c_gitlink: &str,
) -> Result<(), String> {
    if base_commit == head_commit {
        return Err("boundary base must be distinct from HEAD".to_owned());
    }
    if !ancestor {
        return Err("boundary base must be an ancestor of HEAD".to_owned());
    }
    if mlx_c_gitlink != OLD_COMMIT {
        return Err(format!(
            "boundary base must record old mlx-c {OLD_COMMIT}, got {mlx_c_gitlink}"
        ));
    }
    Ok(())
}

fn validate_tuple_values(
    mode: Mode,
    current_commit: &str,
    current_pin: &str,
    target_commit: &str,
    target_pin: &str,
) -> Result<(), String> {
    let (expected_commit, expected_pin) = match mode {
        Mode::Pre => (OLD_COMMIT, OLD_MLX),
        Mode::Admit => (TARGET_COMMIT, TARGET_MLX),
    };
    if current_commit != expected_commit {
        return Err(format!(
            "{mode:?} mode requires mlx-c {expected_commit}, got {current_commit}"
        ));
    }
    if current_pin != expected_pin {
        return Err(format!(
            "{mode:?} mode requires current nested MLX v{expected_pin}, got v{current_pin}"
        ));
    }
    if target_commit != TARGET_COMMIT {
        return Err(format!(
            "recorded target commit must be {TARGET_COMMIT}, got {target_commit}"
        ));
    }
    if target_pin != TARGET_MLX {
        return Err(format!(
            "target nested MLX pin must be v{TARGET_MLX}, got v{target_pin}"
        ));
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

fn validate_suite_results(
    report: &Value,
    workspace_commit: &str,
    mlx_c_commit: &str,
) -> Result<(), String> {
    require_schema(report, "suite results")?;
    require_value(report, "source_commit", workspace_commit, "suite results")?;
    require_value(report, "mlx_c_commit", mlx_c_commit, "suite results")?;
    if report["source_clean"] != true || report["mlx_c_clean"] != true {
        return Err("suite results were not captured from clean source trees".to_owned());
    }
    if report["verdict"] != "pass" {
        return Err("suite results verdict is not pass".to_owned());
    }
    let suites = report["suites"]
        .as_array()
        .ok_or("suite results suites must be an array")?;
    for required in REQUIRED_SUITES {
        let suite = suites
            .iter()
            .find(|suite| suite["id"] == required.id)
            .ok_or_else(|| format!("missing suite {}", required.id))?;
        if suite["command"] != json!(required.command) {
            return Err(format!("suite {} command does not match", required.id));
        }
        if suite["verdict"] != "pass" {
            return Err(format!("suite {} verdict is not pass", required.id));
        }
    }
    if suites.len() != REQUIRED_SUITES.len() {
        return Err("suite results contain unexpected suites".to_owned());
    }
    Ok(())
}

fn require_suite_pass(report: &Value, id: &str) -> Result<Value, String> {
    let suite = report["suites"]
        .as_array()
        .and_then(|suites| suites.iter().find(|suite| suite["id"] == id))
        .ok_or_else(|| format!("missing suite {id}"))?;
    if suite["verdict"] != "pass" {
        return Err(format!("suite {id} verdict is not pass"));
    }
    Ok(suite.clone())
}

fn rerun_suites(repo_root: &Path, workspace_commit: &str, mlx_c_commit: &str) -> Value {
    let suites = REQUIRED_SUITES
        .iter()
        .map(|suite| {
            eprintln!("running bump suite {}", suite.id);
            let output = Command::new(suite.command[0])
                .args(&suite.command[1..])
                .current_dir(repo_root)
                .output();
            let (verdict, error) = match output {
                Ok(output) if output.status.success() => ("pass", None),
                Ok(output) => (
                    "fail",
                    Some(format!(
                        "exited {}; stderr: {}",
                        output.status,
                        String::from_utf8_lossy(&output.stderr).trim()
                    )),
                ),
                Err(error) => ("fail", Some(format!("failed to launch: {error}"))),
            };
            json!({
                "id": suite.id,
                "command": suite.command,
                "verdict": verdict,
                "error": error
            })
        })
        .collect::<Vec<_>>();
    let passed = suites.iter().all(|suite| suite["verdict"] == "pass");
    json!({
        "schema_version": 1,
        "source_commit": workspace_commit,
        "mlx_c_commit": mlx_c_commit,
        "source_clean": tree_is_clean(repo_root),
        "mlx_c_clean": tree_is_clean(&repo_root.join("mlx-sys/src/mlx-c")),
        "suites": suites,
        "verdict": if passed { "pass" } else { "fail" }
    })
}

fn validate_ffi_report(
    report: &Value,
    workspace_commit: &str,
    mlx_c_commit: &str,
    allow_local: bool,
    expected_inventory: &BTreeSet<String>,
) -> Result<Value, String> {
    require_schema(report, "FFI report")?;
    require_value(report, "command", "verify-ffi", "FFI report")?;
    require_value(report, "source_commit", workspace_commit, "FFI report")?;
    require_value(report, "mlx_c_commit", mlx_c_commit, "FFI report")?;
    if report["source_clean"] != true || report["mlx_c_clean"] != true {
        return Err("FFI report was not captured from clean source trees".to_owned());
    }
    if report["environment"]["architecture"] != "aarch64"
        || report["environment"]["os"] != "macos"
        || report["environment"]["rustc"]
            .as_str()
            .is_none_or(str::is_empty)
    {
        return Err("FFI report environment is not qualified macOS aarch64".to_owned());
    }
    if report["guard_malloc_requested"] != true {
        return Err("FFI report must include the --guard-malloc lane".to_owned());
    }
    if report["discovery"]["status"] != "pass" || report["verdict"] != "pass" {
        return Err("FFI report verdict is not pass".to_owned());
    }
    let binaries = report["binaries"]
        .as_array()
        .filter(|binaries| !binaries.is_empty())
        .ok_or("FFI report contains no binaries")?;
    let mut paths = BTreeSet::new();
    let mut leak_coverage = BTreeSet::new();
    let mut guard_coverage = BTreeSet::new();
    let mut has_ffi_safety = false;
    let mut inventory = BTreeSet::new();
    for binary in binaries {
        let path = binary["binary"]["path"]
            .as_str()
            .ok_or("FFI report binary path must be a string")?;
        if !paths.insert(path) {
            return Err(format!("FFI report contains duplicate binary {path}"));
        }
        has_ffi_safety |= binary["binary"]["target"] == "ffi_safety";
        let package_id = binary["binary"]["package_id"]
            .as_str()
            .ok_or("FFI report binary package_id must be a string")?;
        let target = binary["binary"]["target"]
            .as_str()
            .ok_or("FFI report binary target must be a string")?;
        let target_kind = binary["binary"]["target_kind"]
            .as_array()
            .ok_or("FFI report binary target_kind must be an array")?
            .iter()
            .map(|kind| {
                kind.as_str()
                    .ok_or("FFI report target_kind must contain strings")
            })
            .collect::<Result<Vec<_>, _>>()?;
        let identity = format!(
            "{}|{target}|{}",
            crate::verify_ffi::stable_package_id(package_id),
            target_kind.join(",")
        );
        if !inventory.insert(identity.clone()) {
            return Err(format!("FFI report contains duplicate target {identity}"));
        }
        let leak = binary["leaks"]["status"].as_str();
        let guard_passes = match leak {
            Some("pass") => binary["guard_malloc"]["status"] == "pass",
            Some("not_applicable") => {
                binary["guard_malloc"].is_null() || binary["guard_malloc"]["status"] == "pass"
            }
            _ => false,
        };
        if !guard_passes || binary["test"] != "passed" {
            return Err("FFI report contains a failed binary".to_owned());
        }
        if leak != Some("not_applicable") {
            leak_coverage.insert(path);
        }
        guard_coverage.insert(path);
    }
    if !has_ffi_safety {
        return Err("FFI report does not cover the ffi_safety target".to_owned());
    }
    if &inventory != expected_inventory {
        return Err("FFI report binary inventory does not match fresh cargo discovery".to_owned());
    }
    let recorded_leaks = string_set(&report["covered_binaries"]["leaks"], "FFI leak coverage")?;
    let recorded_guard = string_set(
        &report["covered_binaries"]["guard_malloc"],
        "FFI Guard Malloc coverage",
    )?;
    if recorded_leaks != leak_coverage || recorded_guard != guard_coverage {
        return Err("FFI report covered_binaries does not match binary results".to_owned());
    }
    let trust = report["execution_context"]["trust"]
        .as_str()
        .ok_or("FFI report is missing execution_context.trust")?;
    match trust {
        "ci" if report["execution_context"]["procedure"]
            .as_str()
            .is_some_and(|procedure| procedure.starts_with("github-actions:")) => {}
        "ci" => return Err("CI FFI report has no GitHub Actions run identity".to_owned()),
        "local" if !allow_local => {
            return Err(
                "lower-trust local FFI report requires explicit --allow-local-ffi".to_owned(),
            )
        }
        "local" if report["execution_context"]["procedure"] != LOCAL_FFI_PROCEDURE => {
            return Err("local FFI report does not name the recorded procedure".to_owned())
        }
        "local" => {}
        _ => return Err(format!("unsupported FFI report trust {trust}")),
    }
    Ok(json!({"trust": trust, "binaries": binaries.len()}))
}

fn validate_replay_document(base: &Path, report: &Value, strict: bool) -> Result<Value, String> {
    require_schema(report, "target replay report")?;
    require_value(report, "command", "target-replay", "target replay report")?;
    if report["payload_sha256"] != hash_json(&report["payload"])? {
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
        ("venv", "conformance/.venv-target"),
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
        let shard = read_json(&path)?;
        replay_shards.insert(id.to_owned(), shard.clone());
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
    let reconstructed = hash_json(&json!({
        "payload": report["payload"],
        "shards": replay_shards
    }))?;
    if double["first_run_sha256"] != reconstructed || double["second_run_sha256"] != reconstructed {
        return Err("target replay double-run hash does not match payload and shards".to_owned());
    }
    if strict {
        validate_strict_replay(base, report, &seen)?;
    }
    Ok(json!({
        "payload_sha256": report["payload_sha256"],
        "suites": suites.len(),
        "double_run_sha256": double["first_run_sha256"]
    }))
}

fn validate_strict_replay(
    base: &Path,
    report: &Value,
    seen: &BTreeSet<&str>,
) -> Result<(), String> {
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
        let expected_old_environment = if matches!(id, "state" | "transforms") {
            &manifest["provenance"]["environment"]
        } else {
            &corpus["environment"]
        };
        if shard["old_environment"] != *expected_old_environment {
            return Err(format!(
                "target replay suite {id} old environment does not match committed provenance"
            ));
        }
        for case in shard["cases"]
            .as_array()
            .ok_or("target replay shard cases must be an array")?
        {
            required_case_field(case, id, "recipe")?;
            required_case_field(case, id, "input_sha256")?;
            case["input_refs"]
                .as_array()
                .filter(|refs| {
                    !refs.is_empty() && refs.iter().all(|reference| reference.as_str().is_some())
                })
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
    if expected_paths != sources.keys().cloned().collect() {
        return Err(
            "target replay source_artifacts set does not match committed inputs".to_owned(),
        );
    }
    for (relative, digest) in sources {
        let path = safe_join(repo_root, relative)
            .ok_or_else(|| format!("invalid target replay source path {relative}"))?;
        require_tracked_unchanged(repo_root, &path)?;
        if digest != &json!(hash_file(&path)?) {
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
    Ok(())
}

fn required_case_field<'a>(case: &'a Value, suite: &str, field: &str) -> Result<&'a str, String> {
    case[field]
        .as_str()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("target replay suite {suite} has invalid {field}"))
}

fn validate_waivers(
    document: &Value,
    today: &str,
    ledger_entries: &BTreeSet<String>,
) -> Result<Vec<Value>, String> {
    require_schema(document, "waiver manifest")?;
    let waivers = document["waivers"]
        .as_array()
        .ok_or("waiver manifest waivers must be an array")?;
    let mut seen = BTreeSet::new();
    for waiver in waivers {
        let check = waiver
            .get("check")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty());
        let ledger_entry = waiver.get("ledger_entry").and_then(Value::as_object);
        let waiver_id = match (check, ledger_entry) {
            (Some(check), None) if CHECK_IDS.contains(&check) => format!("check:{check}"),
            (Some(check), None) => return Err(format!("waiver has invalid check {check}")),
            (None, Some(entry)) => {
                let kind = entry
                    .get("kind")
                    .and_then(Value::as_str)
                    .filter(|value| !value.trim().is_empty())
                    .ok_or("waiver ledger_entry.kind must be non-empty")?;
                let name = entry
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|value| !value.trim().is_empty())
                    .ok_or("waiver ledger_entry.name must be non-empty")?;
                let id = format!("ledger_entry:{kind}:{name}");
                if !ledger_entries.contains(&id) {
                    return Err(format!("waiver names unknown {id}"));
                }
                id
            }
            _ => return Err("waiver requires exactly one of check or ledger_entry".to_owned()),
        };
        if !seen.insert(waiver_id.clone()) {
            return Err(format!("waiver has duplicate scope {waiver_id}"));
        }
        for field in [
            "scope",
            "failure_or_missing_capability",
            "risk",
            "owner",
            "reviewer_approval",
        ] {
            required_nonempty(waiver, field)?;
        }
        waiver["compensating_evidence"]
            .as_array()
            .filter(|evidence| {
                !evidence.is_empty()
                    && evidence
                        .iter()
                        .all(|item| item.as_str().is_some_and(|item| !item.trim().is_empty()))
            })
            .ok_or_else(|| format!("waiver {waiver_id} requires compensating_evidence"))?;
        match (
            waiver.get("expires_on").and_then(Value::as_str),
            waiver.get("expires_milestone").and_then(Value::as_str),
        ) {
            (Some(date), None) if valid_date(date) && date >= today => {}
            (Some(date), None) if valid_date(date) => {
                return Err(format!("waiver {waiver_id} expired on {date}"))
            }
            (None, Some("next_dependency_bump")) => {}
            _ => {
                return Err(format!(
                "waiver {waiver_id} requires expires_on or expires_milestone next_dependency_bump"
            ))
            }
        }
    }
    Ok(waivers.clone())
}

fn ledger_waiver_scopes(repo_root: &Path) -> Result<BTreeSet<String>, String> {
    let document = read_json(&repo_root.join("ledger/target-delta-classification.json"))?;
    document["entries"]
        .as_array()
        .ok_or("target delta classification entries must be an array")?
        .iter()
        .map(|entry| {
            let kind = entry["kind"]
                .as_str()
                .ok_or("target delta entry kind must be a string")?;
            let name = entry["name"]
                .as_str()
                .ok_or("target delta entry name must be a string")?;
            Ok(format!("ledger_entry:{kind}:{name}"))
        })
        .collect()
}

fn validate_sentinel(repo_root: &Path) -> Result<Value, String> {
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
    if expectations["qualification"]["comparator_rejects"] != true
        || !repo_root.join("mlx-lm/tests/sentinel.rs").is_file()
    {
        return Err("sentinel qualification or executing test is missing".to_owned());
    }
    Ok(json!({"fixture": "tiny-llama", "qualification": "present"}))
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

fn required_nonempty<'a>(value: &'a Value, field: &str) -> Result<&'a str, String> {
    value[field]
        .as_str()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("waiver field {field} must be non-empty"))
}

fn valid_date(date: &str) -> bool {
    if !(date.len() == 10
        && date.as_bytes()[4] == b'-'
        && date.as_bytes()[7] == b'-'
        && date
            .bytes()
            .enumerate()
            .all(|(index, byte)| matches!(index, 4 | 7) || byte.is_ascii_digit()))
    {
        return false;
    }
    let year = date[0..4].parse::<u32>().unwrap_or_default();
    let month = date[5..7].parse::<u32>().unwrap_or_default();
    let day = date[8..10].parse::<u32>().unwrap_or_default();
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    (1..=days).contains(&day)
}

fn current_date() -> String {
    Command::new("date")
        .arg("+%F")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|date| date.trim().to_owned())
        .filter(|date| valid_date(date))
        .unwrap_or_else(|| "9999-12-31".to_owned())
}

fn read_json(path: &Path) -> Result<Value, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("failed to parse {}: {error}", path.display()))
}

fn resolve_path(repo_root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        repo_root.join(path)
    }
}

fn string_set<'a>(value: &'a Value, name: &str) -> Result<BTreeSet<&'a str>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("{name} must be an array"))?
        .iter()
        .map(|item| {
            item.as_str()
                .ok_or_else(|| format!("{name} must contain strings"))
        })
        .collect()
}

fn tree_is_clean(directory: &Path) -> bool {
    git(
        directory,
        &["status", "--porcelain", "--untracked-files=normal"],
    )
    .is_ok_and(|status| status.is_empty())
}

fn require_clean_tree(directory: &Path, name: &str) -> Result<(), String> {
    if tree_is_clean(directory) {
        Ok(())
    } else {
        Err(format!("{name} must be clean for --admit"))
    }
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

fn hash_json(value: &Value) -> Result<String, String> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| format!("failed to serialize JSON for hashing: {error}"))?;
    hash_bytes(&bytes)
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
fn synthetic_checks(ffi: Option<Value>, suites: Option<Value>, replay_passes: bool) -> Vec<Value> {
    let pass = || Ok(json!({"captured": true}));
    vec![
        check(1, "tuple", "tuple", pass()),
        check(2, "boundary", "boundary", pass()),
        check(3, "race", "race", pass()),
        check(
            4,
            "ffi",
            "ffi",
            ffi.map(|value| json!({"report": value}))
                .ok_or_else(|| "missing lower-trust local FFI procedure report".to_owned()),
        ),
        check(
            5,
            "semantic",
            "semantic",
            suites
                .as_ref()
                .map(|_| json!({"captured": true}))
                .ok_or_else(|| "missing suite results".to_owned()),
        ),
        check(6, "ledger", "ledger", pass()),
        check(6, "fingerprints", "fingerprints", pass()),
        check(6, "feature_matrix", "feature matrix", pass()),
        check(
            7,
            "state",
            "state",
            suites
                .map(|_| json!({"captured": true}))
                .ok_or_else(|| "missing suite results".to_owned()),
        ),
        check(
            8,
            "replay",
            "replay",
            replay_passes
                .then(|| json!({"captured": true}))
                .ok_or_else(|| "missing replay".to_owned()),
        ),
        check(0, "sentinel", "sentinel", pass()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn valid_ffi() -> Value {
        json!({
            "schema_version": 1,
            "command": "verify-ffi",
            "source_commit": "workspace",
            "mlx_c_commit": OLD_COMMIT,
            "source_clean": true,
            "mlx_c_clean": true,
            "environment": {"architecture": "aarch64", "os": "macos", "rustc": "rustc 1.98.0"},
            "execution_context": {"trust": "local", "procedure": LOCAL_FFI_PROCEDURE},
            "guard_malloc_requested": true,
            "discovery": {"status": "pass", "error": null},
            "binaries": [{
                "binary": {"package_id": "mlx-tests", "target": "ffi_safety", "target_kind": ["test"], "path": "binary"},
                "leaks": {"status": "pass", "result": null, "tool_status": null, "error": null},
                "test": "passed",
                "guard_malloc": {"status": "pass", "process_status": {"exit_code": 0, "signal": null}, "error": null}
            }],
            "covered_binaries": {"leaks": ["binary"], "guard_malloc": ["binary"]},
            "verdict": "pass"
        })
    }

    fn valid_ffi_inventory() -> BTreeSet<String> {
        BTreeSet::from(["mlx-tests|ffi_safety|test".to_owned()])
    }

    fn valid_suite_results() -> Value {
        json!({
            "schema_version": 1,
            "source_commit": "workspace",
            "mlx_c_commit": OLD_COMMIT,
            "source_clean": true,
            "mlx_c_clean": true,
            "verdict": "pass",
            "suites": REQUIRED_SUITES.iter().map(|suite| json!({
                "id": suite.id,
                "command": suite.command,
                "verdict": "pass"
            })).collect::<Vec<_>>()
        })
    }

    fn replay_tree() -> (tempfile::TempDir, Value) {
        let root = tempfile::tempdir().unwrap();
        let handshake = json!({
            "python": "3.12.14", "architecture": "arm64", "venv": "conformance/.venv-target",
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

    #[test]
    fn sha256_matches_known_vector() {
        assert_eq!(
            hash_bytes(b"abc").unwrap(),
            "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn json_hash_matches_python_canonical_encoding() {
        let value = json!({"z": [1, true, null], "a": {"mlx": "0.32.2", "count": 2}});

        assert_eq!(
            hash_json(&value).unwrap(),
            "sha256:9c374238cded86c892911a96b5afb7dc5dedca833e58969d0a6a55be4326c10c"
        );
    }

    #[test]
    fn tuple_rejects_target_commit_with_wrong_nested_pin() {
        let error = validate_tuple_values(
            Mode::Admit,
            TARGET_COMMIT,
            TARGET_MLX,
            TARGET_COMMIT,
            "0.32.1",
        )
        .unwrap_err();

        assert!(error.contains("target nested MLX pin"));
    }

    #[test]
    fn boundary_base_rejects_head_non_ancestor_and_wrong_old_tuple() {
        assert!(
            validate_boundary_base_values("head", "head", true, OLD_COMMIT)
                .unwrap_err()
                .contains("distinct")
        );
        assert!(
            validate_boundary_base_values("base", "head", false, OLD_COMMIT)
                .unwrap_err()
                .contains("ancestor")
        );
        assert!(
            validate_boundary_base_values("base", "head", true, TARGET_COMMIT)
                .unwrap_err()
                .contains("old mlx-c")
        );
        validate_boundary_base_values("base", "head", true, OLD_COMMIT).unwrap();
    }

    #[test]
    fn replay_rejects_payload_tampering() {
        let (root, mut report) = replay_tree();
        report["payload"]["handshake"]["mlx_runtime"] = "0.32.1".into();

        let error = validate_replay_document(root.path(), &report, false).unwrap_err();

        assert!(error.contains("payload_sha256"));
    }

    #[test]
    fn replay_rejects_missing_shard() {
        let (root, report) = replay_tree();
        std::fs::remove_file(root.path().join("arithmetic.json")).unwrap();

        let error = validate_replay_document(root.path(), &report, false).unwrap_err();

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

        let error = validate_replay_document(root.path(), &report, false).unwrap_err();

        assert!(error.contains("recipe_failed"));
    }

    #[test]
    fn replay_rejects_forged_equal_double_run_hashes() {
        let (root, mut report) = replay_tree();
        report["double_run"]["first_run_sha256"] = "sha256:forged".into();
        report["double_run"]["second_run_sha256"] = "sha256:forged".into();

        let error = validate_replay_document(root.path(), &report, false).unwrap_err();

        assert!(error.contains("double-run hash"));
    }

    #[test]
    fn suite_results_reject_a_missing_named_suite() {
        let mut report = valid_suite_results();
        report["suites"].as_array_mut().unwrap().pop();

        let error = validate_suite_results(&report, "workspace", OLD_COMMIT).unwrap_err();

        assert!(error.contains("missing suite state_transforms"));
    }

    #[test]
    fn ffi_requires_explicit_local_procedure_admission() {
        let error = validate_ffi_report(
            &valid_ffi(),
            "workspace",
            OLD_COMMIT,
            false,
            &valid_ffi_inventory(),
        )
        .unwrap_err();

        assert!(error.contains("--allow-local-ffi"));
        validate_ffi_report(
            &valid_ffi(),
            "workspace",
            OLD_COMMIT,
            true,
            &valid_ffi_inventory(),
        )
        .unwrap();
    }

    #[test]
    fn ffi_accepts_guard_malloc_for_a_leak_ineligible_binary() {
        let mut report = valid_ffi();
        report["binaries"][0]["leaks"]["status"] = "not_applicable".into();
        report["covered_binaries"]["leaks"] = json!([]);

        validate_ffi_report(
            &report,
            "workspace",
            OLD_COMMIT,
            true,
            &valid_ffi_inventory(),
        )
        .unwrap();
    }

    #[test]
    fn ffi_rejects_missing_critical_target_and_inconsistent_coverage() {
        let mut missing = valid_ffi();
        missing["binaries"][0]["binary"]["target"] = "unrelated".into();
        assert!(validate_ffi_report(
            &missing,
            "workspace",
            OLD_COMMIT,
            true,
            &valid_ffi_inventory()
        )
        .unwrap_err()
        .contains("ffi_safety"));

        let mut coverage = valid_ffi();
        coverage["covered_binaries"]["guard_malloc"] = json!([]);
        assert!(validate_ffi_report(
            &coverage,
            "workspace",
            OLD_COMMIT,
            true,
            &valid_ffi_inventory(),
        )
        .unwrap_err()
        .contains("covered_binaries"));
    }

    #[test]
    fn waivers_reject_expired_and_unreviewed_entries() {
        let expired = json!({
            "schema_version": 1,
            "waivers": [{
                "check": "ffi", "scope": "one binary", "failure_or_missing_capability": "no runner",
                "risk": "leak regression", "compensating_evidence": ["local report"], "owner": "owner",
                "reviewer_approval": "reviewer", "expires_on": "2026-08-28"
            }]
        });
        assert!(validate_waivers(&expired, "2026-08-29", &BTreeSet::new())
            .unwrap_err()
            .contains("expired"));
        let mut unreviewed = expired;
        unreviewed["waivers"][0]["expires_on"] = "2026-08-30".into();
        unreviewed["waivers"][0]["reviewer_approval"] = "".into();
        assert!(
            validate_waivers(&unreviewed, "2026-08-29", &BTreeSet::new())
                .unwrap_err()
                .contains("reviewer_approval")
        );
    }

    #[test]
    fn waiver_dates_must_be_real_calendar_dates() {
        assert!(valid_date("2028-02-29"));
        assert!(!valid_date("2026-02-29"));
        assert!(!valid_date("2026-13-01"));
        assert!(!valid_date("2026-04-31"));
    }

    #[test]
    fn waiver_manifest_accepts_a_named_ledger_entry_scope() {
        let document = json!({
            "schema_version": 1,
            "waivers": [{
                "ledger_entry": {"kind": "function", "name": "mlx_example"},
                "scope": "target-only symbol",
                "failure_or_missing_capability": "runtime probe unavailable",
                "risk": "behavior is not independently exercised",
                "compensating_evidence": ["canonical ABI fingerprint"],
                "owner": "owner",
                "reviewer_approval": "reviewer",
                "expires_milestone": "next_dependency_bump"
            }]
        });

        let scopes = BTreeSet::from(["ledger_entry:function:mlx_example".to_owned()]);
        assert_eq!(
            validate_waivers(&document, "2026-08-29", &scopes)
                .unwrap()
                .len(),
            1
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
    fn pre_verdict_names_only_ffi_as_unmet_and_lists_admit_transition() {
        let checks = synthetic_checks(None, Some(valid_suite_results()), true);
        let report = build_verdict(Mode::Pre, checks, Vec::new());

        assert_eq!(report["verdict"], "fail");
        let failed = report["checks"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|check| check["status"] == "fail")
            .map(|check| check["id"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(failed, ["ffi"]);
        assert!(report["admit_requirements"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item.as_str().unwrap().contains(TARGET_COMMIT)));
    }
}
