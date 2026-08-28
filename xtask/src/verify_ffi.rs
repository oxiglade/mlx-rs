use serde::Serialize;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

#[derive(Debug, PartialEq, Eq, Serialize)]
struct LeakResult {
    count: u64,
    bytes: u64,
    baseline_subtracted: bool,
    regression_count: u64,
    regression_bytes: u64,
    named_sites: Vec<NamedSite>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct NamedSite {
    site: String,
    count: u64,
    bytes: u64,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct TestBinary {
    package_id: String,
    target: String,
    target_kind: Vec<String>,
    path: PathBuf,
}

impl TestBinary {
    fn leaks_applicable(&self) -> bool {
        !self.target_kind.iter().any(|kind| kind == "proc-macro")
    }
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum TestStatus {
    Passed,
    Failed,
    Abnormal,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Verdict {
    Pass,
    Fail,
    Error,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum LeakStatus {
    Pass,
    Fail,
    Error,
    NotApplicable,
}

#[derive(Serialize)]
struct ProcessExit {
    exit_code: Option<i32>,
    signal: Option<i32>,
}

#[derive(Serialize)]
struct LeakCheck {
    status: LeakStatus,
    result: Option<LeakResult>,
    tool_status: Option<ProcessExit>,
    error: Option<String>,
}

#[derive(Serialize)]
struct GuardMallocCheck {
    status: Verdict,
    process_status: Option<ProcessExit>,
    error: Option<String>,
}

#[derive(Serialize)]
struct BinaryReport {
    binary: TestBinary,
    leaks: LeakCheck,
    test: TestStatus,
    guard_malloc: Option<GuardMallocCheck>,
}

#[derive(Serialize)]
struct CoveredBinaries {
    leaks: Vec<String>,
    guard_malloc: Vec<String>,
}

#[derive(Serialize)]
struct Discovery {
    status: Verdict,
    error: Option<String>,
}

#[derive(Serialize)]
struct VerifyFfiReport {
    command: &'static str,
    guard_malloc_requested: bool,
    discovery: Discovery,
    binaries: Vec<BinaryReport>,
    covered_binaries: CoveredBinaries,
    verdict: Verdict,
}

#[derive(Debug, PartialEq, Eq)]
struct Options {
    guard_malloc: bool,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, String> {
        match args {
            [] => Ok(Self {
                guard_malloc: false,
            }),
            [flag] if flag == "--guard-malloc" => Ok(Self { guard_malloc: true }),
            _ => Err("usage: cargo run -p xtask -- verify-ffi [--guard-malloc]".to_owned()),
        }
    }
}

pub(crate) fn run(root_dir: &Path, args: &[String]) -> i32 {
    let report = match Options::parse(args) {
        Ok(options) => verify(root_dir, options),
        Err(error) => {
            eprintln!("{error}");
            failed_report(false, error)
        }
    };
    let success = report.verdict == Verdict::Pass;
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, &report).expect("failed to serialize verify-ffi report");
    writeln!(stdout).expect("failed to write verify-ffi report");
    if success {
        0
    } else {
        1
    }
}

fn verify(root_dir: &Path, options: Options) -> VerifyFfiReport {
    let binaries = match discover_test_binaries(root_dir) {
        Ok(binaries) => binaries,
        Err(error) => return failed_report(options.guard_malloc, error),
    };

    let mut reports = Vec::with_capacity(binaries.len());
    let mut covered_leaks = Vec::new();
    let mut covered_guard_malloc = Vec::new();
    for binary in binaries {
        let test = run_tests(&binary);
        let (leaks, covered_path) = run_leaks(&binary);
        covered_leaks.extend(covered_path);
        let guard_malloc = options.guard_malloc.then(|| {
            let (check, covered_path) = run_guard_malloc(&binary);
            covered_guard_malloc.extend(covered_path);
            check
        });

        reports.push(BinaryReport {
            binary,
            leaks,
            test,
            guard_malloc,
        });
    }

    let passed = reports.iter().all(binary_passes);
    VerifyFfiReport {
        command: "verify-ffi",
        guard_malloc_requested: options.guard_malloc,
        discovery: Discovery {
            status: Verdict::Pass,
            error: None,
        },
        binaries: reports,
        covered_binaries: CoveredBinaries {
            leaks: covered_leaks,
            guard_malloc: covered_guard_malloc,
        },
        verdict: if passed { Verdict::Pass } else { Verdict::Fail },
    }
}

fn discover_test_binaries(root_dir: &Path) -> Result<Vec<TestBinary>, String> {
    eprintln!("discovering workspace test executables");
    let discovery = Command::new("cargo")
        .args(["test", "--workspace", "--no-run", "--message-format=json"])
        .current_dir(root_dir)
        .output()
        .map_err(|error| format!("failed to run cargo test discovery: {error}"))?;
    if !discovery.stderr.is_empty() {
        eprint!("{}", String::from_utf8_lossy(&discovery.stderr));
    }
    if !discovery.status.success() {
        return Err(format!(
            "cargo test discovery exited with {}",
            discovery.status
        ));
    }
    let messages = String::from_utf8(discovery.stdout)
        .map_err(|error| format!("cargo test discovery emitted non-UTF-8 JSON: {error}"))?;
    let binaries = parse_test_binaries(&messages)?;
    if binaries.is_empty() {
        return Err("cargo test discovery found no test executables".to_owned());
    }
    Ok(binaries)
}

fn run_tests(binary: &TestBinary) -> TestStatus {
    let path = binary.path.display();
    eprintln!("running tests for {} ({path})", binary.target);
    let output = match Command::new(&binary.path).arg("--test-threads=1").output() {
        Ok(output) => output,
        Err(error) => {
            eprintln!("failed to run tests for {path}: {error}");
            return TestStatus::Abnormal;
        }
    };
    let text = combined_output(&output.stdout, &output.stderr);
    let status = classify_test_status(&text, &output.status);
    if status != TestStatus::Passed {
        eprintln!("tests failed for {path}:\n{text}");
    }
    status
}

fn run_leaks(binary: &TestBinary) -> (LeakCheck, Option<String>) {
    if !binary.leaks_applicable() {
        return (
            LeakCheck {
                status: LeakStatus::NotApplicable,
                result: None,
                tool_status: None,
                error: None,
            },
            None,
        );
    }

    let path = binary.path.display().to_string();
    eprintln!("running leaks for {} ({path})", binary.target);
    let output = match Command::new("leaks")
        .args(["--atExit", "--"])
        .arg(&binary.path)
        .arg("--test-threads=1")
        .output()
    {
        Ok(output) => output,
        Err(error) => {
            return (
                LeakCheck {
                    status: LeakStatus::Error,
                    result: None,
                    tool_status: None,
                    error: Some(format!("failed to run leaks: {error}")),
                },
                None,
            );
        }
    };

    let text = combined_output(&output.stdout, &output.stderr);
    let tool_status = Some(process_exit(&output.status));
    let result = match parse_leaks_report(&text) {
        Ok(result) => result,
        Err(error) => {
            eprintln!("leaks report for {path} could not be parsed: {error}");
            return (
                LeakCheck {
                    status: LeakStatus::Error,
                    result: None,
                    tool_status,
                    error: Some(error),
                },
                Some(path),
            );
        }
    };
    let status = if result.regression_count == 0 && result.regression_bytes == 0 {
        LeakStatus::Pass
    } else {
        LeakStatus::Fail
    };
    (
        LeakCheck {
            status,
            result: Some(result),
            tool_status,
            error: None,
        },
        Some(path),
    )
}

fn run_guard_malloc(binary: &TestBinary) -> (GuardMallocCheck, Option<String>) {
    let path = binary.path.display().to_string();
    eprintln!("running guard malloc for {} ({path})", binary.target);
    let output = match Command::new(&binary.path)
        .arg("--test-threads=1")
        .env("DYLD_INSERT_LIBRARIES", "/usr/lib/libgmalloc.dylib")
        .output()
    {
        Ok(output) => output,
        Err(error) => {
            return (
                GuardMallocCheck {
                    status: Verdict::Error,
                    process_status: None,
                    error: Some(format!("failed to run guard malloc pass: {error}")),
                },
                None,
            );
        }
    };
    let success = output.status.success();
    if !success {
        eprintln!(
            "guard malloc failed for {path}:\n{}",
            combined_output(&output.stdout, &output.stderr)
        );
    }
    (
        GuardMallocCheck {
            status: if success {
                Verdict::Pass
            } else {
                Verdict::Fail
            },
            process_status: Some(process_exit(&output.status)),
            error: None,
        },
        Some(path),
    )
}

fn binary_passes(report: &BinaryReport) -> bool {
    matches!(
        report.leaks.status,
        LeakStatus::Pass | LeakStatus::NotApplicable
    ) && report.test == TestStatus::Passed
        && report
            .guard_malloc
            .as_ref()
            .is_none_or(|guard| guard.status == Verdict::Pass)
}

fn failed_report(guard_malloc_requested: bool, error: String) -> VerifyFfiReport {
    VerifyFfiReport {
        command: "verify-ffi",
        guard_malloc_requested,
        discovery: Discovery {
            status: Verdict::Error,
            error: Some(error),
        },
        binaries: Vec::new(),
        covered_binaries: CoveredBinaries {
            leaks: Vec::new(),
            guard_malloc: Vec::new(),
        },
        verdict: Verdict::Fail,
    }
}

fn combined_output(stdout: &[u8], stderr: &[u8]) -> String {
    let mut output = String::from_utf8_lossy(stdout).into_owned();
    output.push_str(&String::from_utf8_lossy(stderr));
    output
}

fn process_exit(status: &ExitStatus) -> ProcessExit {
    ProcessExit {
        exit_code: status.code(),
        signal: exit_signal(status),
    }
}

#[cfg(unix)]
fn exit_signal(status: &ExitStatus) -> Option<i32> {
    use std::os::unix::process::ExitStatusExt;
    status.signal()
}

#[cfg(not(unix))]
fn exit_signal(_status: &ExitStatus) -> Option<i32> {
    None
}

fn classify_test_status(output: &str, status: &ExitStatus) -> TestStatus {
    if exit_signal(status).is_some() {
        return TestStatus::Abnormal;
    }
    match (status.success(), classify_test_output(output)) {
        (true, status) | (false, status @ TestStatus::Failed) => status,
        (false, _) => TestStatus::Abnormal,
    }
}

fn classify_test_output(output: &str) -> TestStatus {
    if output
        .lines()
        .any(|line| line.contains("test result: FAILED."))
    {
        TestStatus::Failed
    } else if output.lines().any(|line| line.contains("test result: ok.")) {
        TestStatus::Passed
    } else {
        TestStatus::Abnormal
    }
}

fn parse_test_binaries(messages: &str) -> Result<Vec<TestBinary>, String> {
    let mut binaries = BTreeMap::new();
    for (index, line) in messages.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let message: serde_json::Value = serde_json::from_str(line)
            .map_err(|error| format!("invalid cargo JSON on line {}: {error}", index + 1))?;
        if message["reason"] != "compiler-artifact" || message["profile"]["test"] != true {
            continue;
        }

        let field = |pointer: &str| {
            message
                .pointer(pointer)
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
                .ok_or_else(|| {
                    format!(
                        "test compiler artifact on line {} is missing {pointer}",
                        index + 1
                    )
                })
        };
        let path = PathBuf::from(field("/executable")?);
        let target_kind = message
            .pointer("/target/kind")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                format!(
                    "test compiler artifact on line {} is missing /target/kind",
                    index + 1
                )
            })?
            .iter()
            .map(|kind| {
                kind.as_str().map(str::to_owned).ok_or_else(|| {
                    format!(
                        "test compiler artifact on line {} has a non-string /target/kind entry",
                        index + 1
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        binaries.entry(path.clone()).or_insert(TestBinary {
            package_id: field("/package_id")?,
            target: field("/target/name")?,
            target_kind,
            path,
        });
    }
    Ok(binaries.into_values().collect())
}

#[derive(Debug)]
struct RootLeak {
    site: String,
    bytes: u64,
}

fn parse_leaks_report(report: &str) -> Result<LeakResult, String> {
    let summary = report
        .lines()
        .find(|line| line.contains(" total leaked bytes"))
        .ok_or_else(|| "leaks report did not contain a total leaked bytes summary".to_owned())?;
    let count = number_before(summary, " leak")?;
    let bytes = number_before(summary, " total leaked bytes")?;
    let mut roots = report
        .lines()
        .filter_map(parse_root_leak)
        .collect::<Vec<_>>();

    let baseline = roots
        .iter()
        .position(|root| root.site == "<NSArray>" && root.bytes == 32)
        .filter(|_| count >= 1 && bytes >= 32);
    if let Some(index) = baseline {
        roots.remove(index);
    }

    let mut named_sites = BTreeMap::<String, (u64, u64)>::new();
    for root in roots {
        let (count, bytes) = named_sites.entry(root.site).or_default();
        *count += 1;
        *bytes += root.bytes;
    }

    Ok(LeakResult {
        count,
        bytes,
        baseline_subtracted: baseline.is_some(),
        regression_count: count - u64::from(baseline.is_some()),
        regression_bytes: bytes - u64::from(baseline.is_some()) * 32,
        named_sites: named_sites
            .into_iter()
            .map(|(site, (count, bytes))| NamedSite { site, count, bytes })
            .collect(),
    })
}

fn number_before(line: &str, marker: &str) -> Result<u64, String> {
    let prefix = line
        .split_once(marker)
        .map(|(prefix, _)| prefix)
        .ok_or_else(|| format!("leaks summary is missing {marker:?}"))?;
    let number = prefix
        .split_whitespace()
        .next_back()
        .ok_or_else(|| format!("leaks summary has no number before {marker:?}"))?
        .replace(',', "");
    number
        .parse()
        .map_err(|_| format!("invalid number {number:?} before {marker:?}"))
}

fn parse_root_leak(line: &str) -> Option<RootLeak> {
    let (_, root) = line.split_once("ROOT LEAK:")?;
    let site_start = root.find('<')?;
    let site_end = root[site_start..].find('>')? + site_start;
    let site = normalize_site(&root[site_start + 1..site_end]);
    let remainder = &root[site_end + 1..];
    let bytes = remainder
        .split_once('[')
        .and_then(|(_, value)| value.split_once(']'))
        .and_then(|(value, _)| value.trim().replace(',', "").parse().ok())
        .unwrap_or(0);
    Some(RootLeak { site, bytes })
}

fn normalize_site(site: &str) -> String {
    let mut parts = site.split_whitespace().collect::<Vec<_>>();
    if parts.last().is_some_and(|part| part.starts_with("0x")) {
        parts.pop();
    }
    let site = parts.join(" ").trim_end_matches(':').to_owned();
    format!("<{site}>")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binary_report(leak_status: LeakStatus, test: TestStatus) -> BinaryReport {
        BinaryReport {
            binary: TestBinary {
                package_id: "package".to_owned(),
                target: "target".to_owned(),
                target_kind: vec!["test".to_owned()],
                path: PathBuf::from("binary"),
            },
            leaks: LeakCheck {
                status: leak_status,
                result: None,
                tool_status: None,
                error: None,
            },
            test,
            guard_malloc: None,
        }
    }

    #[test]
    fn exact_nsarray_baseline_is_subtracted() {
        let report = r#"
Process 123: 1 leak for 32 total leaked bytes.
ROOT LEAK: <NSArray: 0x600000008000> [32]
"#;

        assert_eq!(
            parse_leaks_report(report).unwrap(),
            LeakResult {
                count: 1,
                bytes: 32,
                baseline_subtracted: true,
                regression_count: 0,
                regression_bytes: 0,
                named_sites: Vec::new(),
            }
        );
    }

    #[test]
    fn repeated_named_sites_are_aggregated() {
        let report = r#"
Process 456: 4 leaks for 80 total leaked bytes.
ROOT LEAK: <NSArray: 0x600000008000> [32]
ROOT LEAK: <malloc in mlx_map_string_to_string_iterator_new> [16]
ROOT LEAK: <malloc in mlx_map_string_to_string_iterator_new> [16]
ROOT LEAK: <malloc in mlx_map_string_to_string_iterator_new> [16]
"#;

        assert_eq!(
            parse_leaks_report(report).unwrap(),
            LeakResult {
                count: 4,
                bytes: 80,
                baseline_subtracted: true,
                regression_count: 3,
                regression_bytes: 48,
                named_sites: vec![NamedSite {
                    site: "<malloc in mlx_map_string_to_string_iterator_new>".to_owned(),
                    count: 3,
                    bytes: 48,
                }],
            }
        );
    }

    #[test]
    fn wrong_site_is_not_treated_as_baseline() {
        let report = r#"
Process 789: 1 leak for 32 total leaked bytes.
ROOT LEAK: <malloc in unexpected_allocator> [32]
"#;

        assert_eq!(
            parse_leaks_report(report).unwrap(),
            LeakResult {
                count: 1,
                bytes: 32,
                baseline_subtracted: false,
                regression_count: 1,
                regression_bytes: 32,
                named_sites: vec![NamedSite {
                    site: "<malloc in unexpected_allocator>".to_owned(),
                    count: 1,
                    bytes: 32,
                }],
            }
        );
    }

    #[test]
    fn zero_leaks_passes_without_a_baseline() {
        let report = "Process 42: 0 leaks for 0 total leaked bytes.";

        assert_eq!(
            parse_leaks_report(report).unwrap(),
            LeakResult {
                count: 0,
                bytes: 0,
                baseline_subtracted: false,
                regression_count: 0,
                regression_bytes: 0,
                named_sites: Vec::new(),
            }
        );
    }

    #[test]
    fn launched_leaks_run_without_summary_is_an_error() {
        assert!(parse_leaks_report("ROOT LEAK: <NSArray: 0x1> [32]").is_err());
        let report = r#"
running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
"#;

        assert!(parse_leaks_report(report).is_err());
    }

    #[test]
    fn leak_not_applicable_passes_and_is_reported() {
        let binary = binary_report(LeakStatus::NotApplicable, TestStatus::Passed);

        assert!(binary_passes(&binary));
        assert!(!binary_passes(&binary_report(
            LeakStatus::NotApplicable,
            TestStatus::Failed
        )));
        assert_eq!(
            serde_json::to_value(&binary).unwrap()["leaks"]["status"],
            "not_applicable"
        );
    }

    #[test]
    fn compiler_artifacts_select_current_test_executables() {
        let messages = r#"
{"reason":"compiler-artifact","package_id":"path+file:///repo/mlx-rs#0.25.3","target":{"name":"mlx_rs","kind":["lib"]},"profile":{"test":false},"executable":null}
{"reason":"compiler-artifact","package_id":"path+file:///repo/mlx-rs#0.25.3","target":{"name":"mlx_rs","kind":["lib"]},"profile":{"test":true},"executable":"/repo/target/debug/deps/mlx_rs-current"}
{"reason":"compiler-message","package_id":"path+file:///repo/mlx-rs#0.25.3","message":{}}
{"reason":"compiler-artifact","package_id":"path+file:///repo/mlx-tests#0.25.3","target":{"name":"ffi_safety","kind":["test"]},"profile":{"test":true},"executable":"/repo/target/debug/deps/ffi_safety-current"}
{"reason":"compiler-artifact","package_id":"path+file:///repo/mlx-rs#0.25.3","target":{"name":"mlx_rs","kind":["lib"]},"profile":{"test":true},"executable":"/repo/target/debug/deps/mlx_rs-current"}
{"reason":"build-finished","success":true}
"#;

        assert_eq!(
            parse_test_binaries(messages).unwrap(),
            vec![
                TestBinary {
                    package_id: "path+file:///repo/mlx-tests#0.25.3".to_owned(),
                    target: "ffi_safety".to_owned(),
                    target_kind: vec!["test".to_owned()],
                    path: PathBuf::from("/repo/target/debug/deps/ffi_safety-current"),
                },
                TestBinary {
                    package_id: "path+file:///repo/mlx-rs#0.25.3".to_owned(),
                    target: "mlx_rs".to_owned(),
                    target_kind: vec!["lib".to_owned()],
                    path: PathBuf::from("/repo/target/debug/deps/mlx_rs-current"),
                },
            ]
        );
    }

    #[test]
    fn captured_proc_macro_artifact_is_not_leak_eligible() {
        let binaries =
            parse_test_binaries(include_str!("../tests/fixtures/cargo-test-artifacts.json"))
                .unwrap();

        assert_eq!(binaries[0].target_kind, ["test"]);
        assert_eq!(binaries[1].target_kind, ["proc-macro"]);
        assert_eq!(
            serde_json::to_value(&binaries[1]).unwrap()["target_kind"],
            serde_json::json!(["proc-macro"])
        );
        assert_eq!(
            binaries
                .iter()
                .filter(|binary| binary.leaks_applicable())
                .map(|binary| binary.target.as_str())
                .collect::<Vec<_>>(),
            ["ffi_safety"]
        );
    }

    #[test]
    fn malformed_compiler_artifact_is_rejected() {
        let messages = r#"{"reason":"compiler-artifact","profile":{"test":true}}"#;

        assert!(parse_test_binaries(messages).is_err());
    }

    #[test]
    fn failed_captured_test_report_fails_binary() {
        let failed_report = r#"
running 2 tests
test passes ... ok
test fails ... FAILED

test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s

Process 456: 1 leak for 32 total leaked bytes.
ROOT LEAK: <NSArray: 0x600000008000> [32]
"#;

        assert_eq!(classify_test_output(failed_report), TestStatus::Failed);
        let binary = binary_report(LeakStatus::Pass, classify_test_output(failed_report));

        assert!(!binary_passes(&binary));
        assert_eq!(serde_json::to_value(&binary).unwrap()["test"], "failed");
    }

    #[cfg(unix)]
    #[test]
    fn signal_after_success_summary_is_abnormal() {
        let status = Command::new("sh")
            .args(["-c", "kill -ABRT $$"])
            .status()
            .unwrap();

        assert_eq!(
            classify_test_status("test result: ok. 1 passed; 0 failed", &status),
            TestStatus::Abnormal
        );
    }

    #[test]
    fn verify_ffi_accepts_only_guard_malloc_flag() {
        assert_eq!(
            Options::parse(&[]).unwrap(),
            Options {
                guard_malloc: false
            }
        );
        assert_eq!(
            Options::parse(&["--guard-malloc".to_owned()]).unwrap(),
            Options { guard_malloc: true }
        );
        assert!(Options::parse(&["--unknown".to_owned()]).is_err());
        assert!(
            Options::parse(&["--guard-malloc".to_owned(), "--guard-malloc".to_owned()]).is_err()
        );
    }
}
