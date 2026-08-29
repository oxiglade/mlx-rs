use serde::Serialize;
use std::collections::BTreeMap;
use std::env;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::time::{Duration, Instant};

const CALIBRATION_BINARY: &str = "ci_leak_calibration";
const CALIBRATION_TEST: &str = "deliberate_iterator_handle_leak";
const CALIBRATION_LEAK_SITE: &str = "<malloc in mlx_map_string_to_string_iterator_new>";
const DEFAULT_CALIBRATION_BUDGET_SECONDS: u64 = 20 * 60;

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

    fn identity(&self) -> String {
        format!(
            "{}|{}|{}",
            stable_package_id(&self.package_id),
            self.target,
            self.target_kind.join(",")
        )
    }
}

pub(crate) fn stable_package_id(package_id: &str) -> &str {
    package_id
        .rsplit_once('#')
        .map_or(package_id, |(_, identity)| identity)
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

#[derive(Debug, Serialize)]
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
struct ReportEnvironment {
    architecture: &'static str,
    os: &'static str,
    rustc: String,
}

#[derive(Serialize)]
struct ExecutionContext {
    trust: &'static str,
    procedure: String,
}

#[derive(Serialize)]
struct VerifyFfiReport {
    schema_version: u32,
    command: &'static str,
    source_commit: String,
    mlx_c_commit: String,
    source_clean: bool,
    mlx_c_clean: bool,
    environment: ReportEnvironment,
    execution_context: ExecutionContext,
    guard_malloc_requested: bool,
    discovery: Discovery,
    binaries: Vec<BinaryReport>,
    covered_binaries: CoveredBinaries,
    verdict: Verdict,
}

#[derive(Debug)]
struct CapturedCommand {
    stdout: String,
    stderr: String,
    status: CapturedExit,
}

#[derive(Debug, Serialize)]
struct CapturedExit {
    exit_code: Option<i32>,
    signal: Option<i32>,
}

#[derive(Debug)]
struct SpawnFailure {
    kind: std::io::ErrorKind,
    message: String,
}

#[derive(Serialize)]
struct CalibrationLeakPhase {
    status: Verdict,
    duration_ms: u64,
    target: &'static str,
    expected_site: Option<&'static str>,
    result: Option<LeakResult>,
    tool_status: Option<CapturedExit>,
    failure: Option<String>,
    error: Option<String>,
}

#[derive(Serialize)]
struct CleanGatePhase {
    status: Verdict,
    duration_ms: u64,
    report: VerifyFfiReport,
}

#[derive(Serialize)]
struct BudgetPhase {
    status: Verdict,
    limit_ms: u64,
    observed_ms: u64,
    failure: Option<String>,
    error: Option<String>,
}

#[derive(Serialize)]
struct CalibrationPhases {
    environment_probe: CalibrationLeakPhase,
    deliberate_leak_detection: CalibrationLeakPhase,
    clean_gate: CleanGatePhase,
    budget: BudgetPhase,
}

#[derive(Serialize)]
struct CalibrationReport {
    schema_version: u32,
    command: &'static str,
    mode: &'static str,
    source_commit: String,
    mlx_c_commit: String,
    budget_seconds: u64,
    phases: CalibrationPhases,
    total_duration_ms: u64,
    verdict: Verdict,
}

#[derive(Debug, PartialEq, Eq)]
enum Options {
    Verify { guard_malloc: bool },
    Calibrate { budget_seconds: u64 },
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, String> {
        match args {
            [] => Ok(Self::Verify {
                guard_malloc: false,
            }),
            [flag] if flag == "--guard-malloc" => Ok(Self::Verify { guard_malloc: true }),
            [flag] if flag == "--calibrate" => Ok(Self::Calibrate {
                budget_seconds: DEFAULT_CALIBRATION_BUDGET_SECONDS,
            }),
            [calibrate, budget, seconds]
                if calibrate == "--calibrate" && budget == "--budget-seconds" =>
            {
                let budget_seconds = seconds
                    .parse::<u64>()
                    .ok()
                    .filter(|seconds| *seconds > 0)
                    .ok_or_else(usage)?;
                Ok(Self::Calibrate { budget_seconds })
            }
            _ => Err(usage()),
        }
    }
}

fn usage() -> String {
    "usage: cargo run -p xtask -- verify-ffi [--guard-malloc | --calibrate [--budget-seconds N]]"
        .to_owned()
}

pub(crate) fn run(root_dir: &Path, args: &[String]) -> i32 {
    match Options::parse(args) {
        Ok(Options::Verify { guard_malloc }) => {
            let report = verify(root_dir, guard_malloc);
            write_report(&report, report.verdict == Verdict::Pass)
        }
        Ok(Options::Calibrate { budget_seconds }) => {
            let report = calibrate(root_dir, budget_seconds);
            write_report(&report, report.verdict == Verdict::Pass)
        }
        Err(error) => {
            eprintln!("{error}");
            let report = failed_report(root_dir, false, error);
            write_report(&report, false)
        }
    }
}

fn write_report(report: &impl Serialize, success: bool) -> i32 {
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, &report).expect("failed to serialize verify-ffi report");
    writeln!(stdout).expect("failed to write verify-ffi report");
    if success {
        0
    } else {
        1
    }
}

fn calibrate(root_dir: &Path, budget_seconds: u64) -> CalibrationReport {
    let total_started = Instant::now();

    // The probe child must allocate: a no-malloc process (e.g. /usr/bin/true)
    // yields no leaks summary and misreports a healthy runner as unparseable.
    // The calibration binary with its only test skipped is clean but still
    // initializes the runtime, so it always produces a parseable report.
    eprintln!("probing leaks child inspection with the skipped calibration binary");
    let probe_started = Instant::now();
    let environment_probe = match discover_calibration_binary(root_dir) {
        Ok(binary) => {
            let mut probe_command = Command::new("leaks");
            probe_command.args(["--atExit", "--"]);
            probe_command.arg(&binary.path);
            probe_command.arg("--test-threads=1");
            assess_environment_probe(capture_command(&mut probe_command), probe_started.elapsed())
        }
        Err(error) => CalibrationLeakPhase {
            status: Verdict::Error,
            duration_ms: duration_ms(probe_started.elapsed()),
            target: CALIBRATION_BINARY,
            expected_site: None,
            result: None,
            tool_status: None,
            failure: Some("calibration_binary_discovery_failed".to_owned()),
            error: Some(error),
        },
    };

    let deliberate_leak_detection = run_deliberate_leak_calibration(root_dir);

    eprintln!("running clean full verify-ffi gate");
    let clean_started = Instant::now();
    let clean_report = verify(root_dir, false);
    let clean_duration = clean_started.elapsed();
    let clean_gate = CleanGatePhase {
        status: if clean_report.verdict == Verdict::Pass {
            Verdict::Pass
        } else {
            Verdict::Fail
        },
        duration_ms: duration_ms(clean_duration),
        report: clean_report,
    };
    let budget = assess_budget(clean_duration, Duration::from_secs(budget_seconds));
    let phases = CalibrationPhases {
        environment_probe,
        deliberate_leak_detection,
        clean_gate,
        budget,
    };
    let verdict = calibration_verdict(&phases);

    CalibrationReport {
        schema_version: 1,
        command: "verify-ffi",
        mode: "calibration",
        source_commit: git_head(root_dir),
        mlx_c_commit: git_head(&root_dir.join("mlx-sys/src/mlx-c")),
        budget_seconds,
        phases,
        total_duration_ms: duration_ms(total_started.elapsed()),
        verdict,
    }
}

fn run_deliberate_leak_calibration(root_dir: &Path) -> CalibrationLeakPhase {
    let started = Instant::now();
    let binary = match discover_calibration_binary(root_dir) {
        Ok(binary) => binary,
        Err(error) => {
            return CalibrationLeakPhase {
                status: Verdict::Error,
                duration_ms: duration_ms(started.elapsed()),
                target: CALIBRATION_BINARY,
                expected_site: Some(CALIBRATION_LEAK_SITE),
                result: None,
                tool_status: None,
                failure: Some("calibration_binary_discovery_failed".to_owned()),
                error: Some(error),
            };
        }
    };

    eprintln!(
        "running deliberate leak calibration for {} ({})",
        binary.target,
        binary.path.display()
    );
    let mut command = Command::new("leaks");
    command.args(["--atExit", "--"]).arg(&binary.path).args([
        "--ignored",
        "--exact",
        CALIBRATION_TEST,
        "--test-threads=1",
    ]);
    assess_deliberate_leak(capture_command(&mut command), started.elapsed())
}

fn discover_calibration_binary(root_dir: &Path) -> Result<TestBinary, String> {
    eprintln!("discovering deliberate leak calibration executable");
    let discovery = Command::new("cargo")
        .args([
            "test",
            "-p",
            "mlx-tests",
            "--test",
            CALIBRATION_BINARY,
            "--no-run",
            "--message-format=json",
        ])
        .current_dir(root_dir)
        .output()
        .map_err(|error| format!("failed to discover calibration binary: {error}"))?;
    if !discovery.stderr.is_empty() {
        eprint!("{}", String::from_utf8_lossy(&discovery.stderr));
    }
    if !discovery.status.success() {
        return Err(format!(
            "calibration binary discovery exited with {}",
            discovery.status
        ));
    }
    let messages = String::from_utf8(discovery.stdout)
        .map_err(|error| format!("calibration discovery emitted non-UTF-8 JSON: {error}"))?;
    let matches = parse_test_binaries(&messages)?
        .into_iter()
        .filter(|binary| binary.target == CALIBRATION_BINARY)
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [binary] => Ok(TestBinary {
            package_id: binary.package_id.clone(),
            target: binary.target.clone(),
            target_kind: binary.target_kind.clone(),
            path: binary.path.clone(),
        }),
        [] => Err("cargo did not emit the calibration test executable".to_owned()),
        _ => Err("cargo emitted multiple calibration test executables".to_owned()),
    }
}

fn capture_command(command: &mut Command) -> Result<CapturedCommand, SpawnFailure> {
    command
        .output()
        .map(|output| CapturedCommand {
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            status: CapturedExit {
                exit_code: output.status.code(),
                signal: exit_signal(&output.status),
            },
        })
        .map_err(|error| SpawnFailure {
            kind: error.kind(),
            message: error.to_string(),
        })
}

fn assess_environment_probe(
    attempt: Result<CapturedCommand, SpawnFailure>,
    duration: Duration,
) -> CalibrationLeakPhase {
    let target = CALIBRATION_BINARY;
    let captured = match attempt {
        Ok(captured) => captured,
        Err(error) => {
            let failure = if error.kind == std::io::ErrorKind::NotFound {
                "leaks_missing"
            } else {
                "leaks_probe_spawn_failed"
            };
            return calibration_leak_failure(
                Verdict::Error,
                duration,
                target,
                None,
                None,
                failure,
                error.message,
            );
        }
    };
    let text = captured_output(&captured);
    if child_task_port_denied(&text) {
        return calibration_leak_failure(
            Verdict::Error,
            duration,
            target,
            None,
            Some(captured.status),
            "child_task_port_denied",
            text.trim().to_owned(),
        );
    }
    let process_succeeded =
        captured.status.exit_code == Some(0) && captured.status.signal.is_none();
    match parse_leaks_report(&text) {
        Ok(result) if result.regression_count != 0 || result.regression_bytes != 0 => {
            calibration_leak_failure(
                Verdict::Fail,
                duration,
                target,
                Some(result),
                Some(captured.status),
                "probe_not_clean",
                "the trivial environment probe reported a leak regression".to_owned(),
            )
        }
        Ok(result) if !process_succeeded => calibration_leak_failure(
            Verdict::Error,
            duration,
            target,
            Some(result),
            Some(captured.status),
            "probe_process_failed",
            "leaks did not exit normally after inspecting the trivial probe".to_owned(),
        ),
        Ok(result) => CalibrationLeakPhase {
            status: Verdict::Pass,
            duration_ms: duration_ms(duration),
            target,
            expected_site: None,
            result: Some(result),
            tool_status: Some(captured.status),
            failure: None,
            error: None,
        },
        Err(error) => calibration_leak_failure(
            Verdict::Error,
            duration,
            target,
            None,
            Some(captured.status),
            "leaks_report_unparsed",
            error,
        ),
    }
}

fn assess_deliberate_leak(
    attempt: Result<CapturedCommand, SpawnFailure>,
    duration: Duration,
) -> CalibrationLeakPhase {
    let captured = match attempt {
        Ok(captured) => captured,
        Err(error) => {
            let failure = if error.kind == std::io::ErrorKind::NotFound {
                "leaks_missing"
            } else {
                "deliberate_leak_spawn_failed"
            };
            return calibration_leak_failure(
                Verdict::Error,
                duration,
                CALIBRATION_BINARY,
                None,
                None,
                failure,
                error.message,
            );
        }
    };
    let text = captured_output(&captured);
    if child_task_port_denied(&text) {
        return calibration_leak_failure(
            Verdict::Error,
            duration,
            CALIBRATION_BINARY,
            None,
            Some(captured.status),
            "child_task_port_denied",
            text.trim().to_owned(),
        );
    }
    let result = match parse_leaks_report(&text) {
        Ok(result) => result,
        Err(error) => {
            return calibration_leak_failure(
                Verdict::Error,
                duration,
                CALIBRATION_BINARY,
                None,
                Some(captured.status),
                "leaks_report_unparsed",
                error,
            );
        }
    };

    if captured.status.signal.is_some() || classify_test_output(&text) != TestStatus::Passed {
        return calibration_leak_failure(
            Verdict::Fail,
            duration,
            CALIBRATION_BINARY,
            Some(result),
            Some(captured.status),
            "deliberate_test_failed",
            "the ignored calibration test did not complete successfully".to_owned(),
        );
    }

    let site_detected = result
        .named_sites
        .iter()
        .any(|site| site.site == CALIBRATION_LEAK_SITE && site.count > 0);
    if result.regression_count == 0 || !site_detected {
        return calibration_leak_failure(
            Verdict::Fail,
            duration,
            CALIBRATION_BINARY,
            Some(result),
            Some(captured.status),
            "expected_site_missing",
            format!("leaks did not report a nonzero regression at {CALIBRATION_LEAK_SITE}"),
        );
    }

    CalibrationLeakPhase {
        status: Verdict::Pass,
        duration_ms: duration_ms(duration),
        target: CALIBRATION_BINARY,
        expected_site: Some(CALIBRATION_LEAK_SITE),
        result: Some(result),
        tool_status: Some(captured.status),
        failure: None,
        error: None,
    }
}

fn calibration_leak_failure(
    status: Verdict,
    duration: Duration,
    target: &'static str,
    result: Option<LeakResult>,
    tool_status: Option<CapturedExit>,
    failure: &'static str,
    error: String,
) -> CalibrationLeakPhase {
    CalibrationLeakPhase {
        status,
        duration_ms: duration_ms(duration),
        target,
        expected_site: (target == CALIBRATION_BINARY).then_some(CALIBRATION_LEAK_SITE),
        result,
        tool_status,
        failure: Some(failure.to_owned()),
        error: Some(error),
    }
}

fn captured_output(captured: &CapturedCommand) -> String {
    let mut text = captured.stdout.clone();
    text.push_str(&captured.stderr);
    text
}

fn child_task_port_denied(output: &str) -> bool {
    let output = output.to_ascii_lowercase();
    output.contains("task_for_pid")
        || output.contains("cannot acquire child task port")
        || output.contains("couldn't get task port")
        || output.contains("could not get task port")
        || (output.contains("task port")
            && (output.contains("denied")
                || output.contains("failed")
                || output.contains("not permitted")))
}

fn assess_budget(observed: Duration, limit: Duration) -> BudgetPhase {
    let exceeded = observed > limit;
    BudgetPhase {
        status: if exceeded {
            Verdict::Fail
        } else {
            Verdict::Pass
        },
        limit_ms: duration_ms(limit),
        observed_ms: duration_ms(observed),
        failure: exceeded.then(|| "budget_exceeded".to_owned()),
        error: exceeded.then(|| {
            format!(
                "clean full gate took {} ms, exceeding the {} ms budget",
                duration_ms(observed),
                duration_ms(limit)
            )
        }),
    }
}

fn calibration_verdict(phases: &CalibrationPhases) -> Verdict {
    calibration_verdict_from_statuses([
        &phases.environment_probe.status,
        &phases.deliberate_leak_detection.status,
        &phases.clean_gate.status,
        &phases.budget.status,
    ])
}

fn calibration_verdict_from_statuses(statuses: [&Verdict; 4]) -> Verdict {
    if statuses.iter().any(|status| **status == Verdict::Error) {
        Verdict::Error
    } else if statuses.iter().all(|status| **status == Verdict::Pass) {
        Verdict::Pass
    } else {
        Verdict::Fail
    }
}

fn duration_ms(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

fn verify(root_dir: &Path, guard_malloc: bool) -> VerifyFfiReport {
    let binaries = match discover_test_binaries(root_dir) {
        Ok(binaries) => binaries,
        Err(error) => return failed_report(root_dir, guard_malloc, error),
    };

    let mut reports = Vec::with_capacity(binaries.len());
    let mut covered_leaks = Vec::new();
    let mut covered_guard_malloc = Vec::new();
    for binary in binaries {
        let test = run_tests(&binary);
        let (leaks, covered_path) = run_leaks(&binary);
        covered_leaks.extend(covered_path);
        let guard_malloc = guard_malloc.then(|| {
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
    report_header(
        root_dir,
        guard_malloc,
        Discovery {
            status: Verdict::Pass,
            error: None,
        },
        reports,
        CoveredBinaries {
            leaks: covered_leaks,
            guard_malloc: covered_guard_malloc,
        },
        if passed { Verdict::Pass } else { Verdict::Fail },
    )
}

fn report_header(
    root_dir: &Path,
    guard_malloc_requested: bool,
    discovery: Discovery,
    binaries: Vec<BinaryReport>,
    covered_binaries: CoveredBinaries,
    verdict: Verdict,
) -> VerifyFfiReport {
    let github_run = env::var("GITHUB_RUN_ID")
        .ok()
        .filter(|value| !value.is_empty());
    let ci = env::var("GITHUB_ACTIONS").as_deref() == Ok("true") && github_run.is_some();
    VerifyFfiReport {
        schema_version: 1,
        command: "verify-ffi",
        source_commit: git_head(root_dir),
        mlx_c_commit: git_head(&root_dir.join("mlx-sys/src/mlx-c")),
        source_clean: git_clean(root_dir),
        mlx_c_clean: git_clean(&root_dir.join("mlx-sys/src/mlx-c")),
        environment: ReportEnvironment {
            architecture: env::consts::ARCH,
            os: env::consts::OS,
            rustc: command_text(Command::new("rustc").arg("--version")),
        },
        execution_context: ExecutionContext {
            trust: if ci { "ci" } else { "local" },
            procedure: if ci {
                format!(
                    "github-actions:{}:{}",
                    env::var("GITHUB_WORKFLOW").unwrap_or_else(|_| "unnamed".to_owned()),
                    github_run.expect("GitHub Actions context checked above")
                )
            } else {
                "ROADMAP.md#tranche-2-leak-and-use-after-free-gate-done".to_owned()
            },
        },
        guard_malloc_requested,
        discovery,
        binaries,
        covered_binaries,
        verdict,
    }
}

fn git_head(directory: &Path) -> String {
    command_text(
        Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(directory),
    )
}

fn git_clean(directory: &Path) -> bool {
    Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=normal"])
        .current_dir(directory)
        .output()
        .is_ok_and(|output| output.status.success() && output.stdout.is_empty())
}

fn command_text(command: &mut Command) -> String {
    command
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|output| output.trim().to_owned())
        .unwrap_or_else(|| "unknown".to_owned())
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

pub(crate) fn discovered_target_ids(root_dir: &Path) -> Result<Vec<String>, String> {
    discover_test_binaries(root_dir).map(|binaries| {
        binaries
            .into_iter()
            .map(|binary| binary.identity())
            .collect()
    })
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

fn failed_report(root_dir: &Path, guard_malloc_requested: bool, error: String) -> VerifyFfiReport {
    report_header(
        root_dir,
        guard_malloc_requested,
        Discovery {
            status: Verdict::Error,
            error: Some(error),
        },
        Vec::new(),
        CoveredBinaries {
            leaks: Vec::new(),
            guard_malloc: Vec::new(),
        },
        Verdict::Fail,
    )
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
    if line.trim_start().starts_with("STACK OF ") {
        return None;
    }
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
STACK OF 3 INSTANCES OF 'ROOT LEAK: <malloc in mlx_map_string_to_string_iterator_new>':
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
    fn qualification_report_counts_only_root_leak_entries() {
        let report = include_str!("../tests/fixtures/leaks-deliberate-leak.txt");

        assert_eq!(
            parse_leaks_report(report).unwrap(),
            LeakResult {
                count: 201,
                bytes: 3232,
                baseline_subtracted: true,
                regression_count: 200,
                regression_bytes: 3200,
                named_sites: vec![NamedSite {
                    site: "<malloc in mlx_map_string_to_string_iterator_new>".to_owned(),
                    count: 200,
                    bytes: 3200,
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
    fn verify_ffi_accepts_only_supported_modes() {
        assert_eq!(
            Options::parse(&[]).unwrap(),
            Options::Verify {
                guard_malloc: false
            }
        );
        assert_eq!(
            Options::parse(&["--guard-malloc".to_owned()]).unwrap(),
            Options::Verify { guard_malloc: true }
        );
        assert_eq!(
            Options::parse(&["--calibrate".to_owned()]).unwrap(),
            Options::Calibrate {
                budget_seconds: DEFAULT_CALIBRATION_BUDGET_SECONDS
            }
        );
        assert_eq!(
            Options::parse(&[
                "--calibrate".to_owned(),
                "--budget-seconds".to_owned(),
                "900".to_owned(),
            ])
            .unwrap(),
            Options::Calibrate {
                budget_seconds: 900
            }
        );
        assert!(Options::parse(&["--unknown".to_owned()]).is_err());
        assert!(Options::parse(&[
            "--calibrate".to_owned(),
            "--budget-seconds".to_owned(),
            "0".to_owned(),
        ])
        .is_err());
        assert!(
            Options::parse(&["--guard-malloc".to_owned(), "--guard-malloc".to_owned()]).is_err()
        );
    }

    #[test]
    fn package_identity_does_not_depend_on_checkout_path() {
        assert_eq!(
            stable_package_id("path+file:///checkout/one/mlx-tests#mlx-tests@0.25.3"),
            "mlx-tests@0.25.3"
        );
        assert_eq!(
            stable_package_id("path+file:///checkout/two/mlx-tests#mlx-tests@0.25.3"),
            "mlx-tests@0.25.3"
        );
    }

    fn captured_command(stdout: &str, stderr: &str, exit_code: i32) -> CapturedCommand {
        CapturedCommand {
            stdout: stdout.to_owned(),
            stderr: stderr.to_owned(),
            status: CapturedExit {
                exit_code: Some(exit_code),
                signal: None,
            },
        }
    }

    #[test]
    fn calibration_accepts_parsed_probe_and_expected_named_site() {
        let probe = assess_environment_probe(
            Ok(captured_command(
                "Process 1: 0 leaks for 0 total leaked bytes.",
                "",
                0,
            )),
            Duration::from_millis(5),
        );
        let deliberate = assess_deliberate_leak(
            Ok(captured_command(
                r#"
running 1 test
test deliberate_iterator_handle_leak ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
Process 2: 3 leaks for 64 total leaked bytes.
ROOT LEAK: <NSArray: 0x600000008000> [32]
ROOT LEAK: <malloc in mlx_map_string_to_string_iterator_new> [16]
ROOT LEAK: <malloc in mlx_map_string_to_string_iterator_new> [16]
"#,
                "",
                1,
            )),
            Duration::from_millis(8),
        );

        assert_eq!(probe.status, Verdict::Pass);
        assert_eq!(probe.failure, None);
        assert_eq!(deliberate.status, Verdict::Pass);
        assert_eq!(deliberate.failure, None);
        assert_eq!(
            deliberate.result.unwrap().named_sites,
            vec![NamedSite {
                site: CALIBRATION_LEAK_SITE.to_owned(),
                count: 2,
                bytes: 32,
            }]
        );
    }

    #[test]
    fn calibration_reports_missing_leaks_from_spawn_failure() {
        let phase = assess_environment_probe(
            Err(SpawnFailure {
                kind: std::io::ErrorKind::NotFound,
                message: "No such file or directory".to_owned(),
            }),
            Duration::from_millis(1),
        );

        assert_eq!(phase.status, Verdict::Error);
        assert_eq!(phase.failure.as_deref(), Some("leaks_missing"));
        assert!(phase.error.unwrap().contains("No such file or directory"));
    }

    #[test]
    fn calibration_reports_child_task_port_denial_from_captured_output() {
        let phase = assess_environment_probe(
            Ok(captured_command(
                "",
                "leaks[42]: [fatal] Couldn't get task port for pid 43 immediately after launch",
                1,
            )),
            Duration::from_millis(2),
        );

        assert_eq!(phase.status, Verdict::Error);
        assert_eq!(phase.failure.as_deref(), Some("child_task_port_denied"));
        assert!(phase.error.unwrap().contains("Couldn't get task port"));
    }

    #[test]
    fn calibration_rejects_a_leaking_environment_probe() {
        let phase = assess_environment_probe(
            Ok(captured_command(
                r#"
Process 4: 1 leak for 16 total leaked bytes.
ROOT LEAK: <malloc in unexpected_probe_allocator> [16]
"#,
                "",
                1,
            )),
            Duration::from_millis(2),
        );

        assert_eq!(phase.status, Verdict::Fail);
        assert_eq!(phase.failure.as_deref(), Some("probe_not_clean"));
        assert_eq!(phase.result.unwrap().regression_count, 1);
    }

    #[test]
    fn calibration_rejects_zero_summary_from_failed_probe_process() {
        let phase = assess_environment_probe(
            Ok(captured_command(
                "Process 5: 0 leaks for 0 total leaked bytes.",
                "probe terminated abnormally",
                1,
            )),
            Duration::from_millis(2),
        );

        assert_eq!(phase.status, Verdict::Error);
        assert_eq!(phase.failure.as_deref(), Some("probe_process_failed"));
        assert_eq!(phase.tool_status.unwrap().exit_code, Some(1));
    }

    #[test]
    fn calibration_rejects_captured_leak_report_without_expected_site() {
        let phase = assess_deliberate_leak(
            Ok(captured_command(
                r#"
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
Process 3: 2 leaks for 48 total leaked bytes.
ROOT LEAK: <NSArray: 0x600000008000> [32]
ROOT LEAK: <malloc in another_allocator> [16]
"#,
                "",
                1,
            )),
            Duration::from_millis(3),
        );

        assert_eq!(phase.status, Verdict::Fail);
        assert_eq!(phase.failure.as_deref(), Some("expected_site_missing"));
        assert_eq!(phase.result.unwrap().regression_count, 1);
    }

    #[test]
    fn calibration_rejects_full_gate_over_budget() {
        let phase = assess_budget(
            Duration::from_secs(1_201),
            Duration::from_secs(DEFAULT_CALIBRATION_BUDGET_SECONDS),
        );

        assert_eq!(phase.status, Verdict::Fail);
        assert_eq!(phase.failure.as_deref(), Some("budget_exceeded"));
        assert_eq!(phase.observed_ms, 1_201_000);
        assert_eq!(phase.limit_ms, 1_200_000);
    }

    #[test]
    fn calibration_exit_verdict_requires_every_phase_to_pass() {
        assert_eq!(
            calibration_verdict_from_statuses([
                &Verdict::Pass,
                &Verdict::Pass,
                &Verdict::Pass,
                &Verdict::Pass,
            ]),
            Verdict::Pass
        );
        assert_eq!(
            calibration_verdict_from_statuses([
                &Verdict::Pass,
                &Verdict::Fail,
                &Verdict::Pass,
                &Verdict::Pass,
            ]),
            Verdict::Fail
        );
        assert_eq!(
            calibration_verdict_from_statuses([
                &Verdict::Error,
                &Verdict::Pass,
                &Verdict::Pass,
                &Verdict::Pass,
            ]),
            Verdict::Error
        );
    }
}
