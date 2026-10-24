use serde::Serialize;
use std::collections::BTreeMap;
use std::env;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const CALIBRATION_BINARY: &str = "ci_leak_calibration";
const CALIBRATION_TEST: &str = "deliberate_iterator_handle_leak";
const CALIBRATION_LEAK_SITE: &str = "<malloc in mlx_map_string_to_string_iterator_new>";
const DEFAULT_CALIBRATION_BUDGET_SECONDS: u64 = 20 * 60;
// CI task-port acquisition can block forever instead of returning a leaks error.
const DEFAULT_LEAKS_TIMEOUT_SECONDS: u64 = 180;
const DEFAULT_GUARD_MALLOC_TIMEOUT_SECONDS: u64 = 600;
const PROCESS_WAIT_POLL_INTERVAL: Duration = Duration::from_millis(10);

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

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
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

#[derive(Clone, Debug, Serialize)]
struct ProcessExit {
    exit_code: Option<i32>,
    signal: Option<i32>,
}

#[derive(Serialize)]
struct LeakCheck {
    status: LeakStatus,
    duration_ms: u64,
    result: Option<LeakResult>,
    tool_status: Option<ProcessExit>,
    failure: Option<String>,
    error: Option<String>,
}

#[derive(Clone, Serialize)]
struct GuardMallocCheck {
    status: Verdict,
    duration_ms: u64,
    process_status: Option<ProcessExit>,
    failure: Option<String>,
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
struct VerifyFfiReport {
    schema_version: u32,
    command: &'static str,
    leaks_timeout_seconds: u64,
    source_commit: String,
    github_sha: Option<String>,
    github_pr_head_sha: Option<String>,
    mlx_c_commit: String,
    source_clean: bool,
    mlx_c_clean: bool,
    environment: ReportEnvironment,
    guard_malloc_requested: bool,
    guard_malloc_timeout_seconds: u64,
    qualification: Option<Qualification>,
    discovery: Discovery,
    binaries: Vec<BinaryReport>,
    covered_binaries: CoveredBinaries,
    verdict: Verdict,
}

#[derive(Clone, Serialize)]
struct GuardMallocQualification {
    clean: GuardMallocCheck,
    use_after_free: GuardMallocCheck,
    double_free: GuardMallocCheck,
}

impl GuardMallocQualification {
    fn verdict(&self) -> Verdict {
        calibration_verdict_from_statuses([
            &self.clean.status,
            &self.use_after_free.status,
            &self.double_free.status,
        ])
    }
}

#[derive(Serialize)]
struct Qualification {
    source_commit: String,
    environment_probe: Verdict,
    deliberate_leak_detection: Verdict,
    guard_malloc: Option<GuardMallocQualification>,
    budget: Verdict,
    verdict: Verdict,
}

#[derive(Clone, Copy)]
struct Timeouts {
    leaks: u64,
    guard_malloc: u64,
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

#[derive(Debug)]
enum CaptureFailure {
    Io(SpawnFailure),
    Timeout { elapsed: Duration },
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
    guard_malloc: Option<GuardMallocQualification>,
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
    leaks_timeout_seconds: u64,
    phases: CalibrationPhases,
    total_duration_ms: u64,
    verdict: Verdict,
}

#[derive(Debug, PartialEq, Eq)]
enum Options {
    Verify {
        guard_malloc: bool,
        leaks_timeout_seconds: u64,
        guard_malloc_timeout_seconds: u64,
    },
    Calibrate {
        guard_malloc: bool,
        budget_seconds: u64,
        leaks_timeout_seconds: u64,
        guard_malloc_timeout_seconds: u64,
        report_path: Option<PathBuf>,
    },
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut guard_malloc = false;
        let mut calibrate = false;
        let mut budget_seconds = DEFAULT_CALIBRATION_BUDGET_SECONDS;
        let mut leaks_timeout_seconds = DEFAULT_LEAKS_TIMEOUT_SECONDS;
        let mut budget_seen = false;
        let mut timeout_seen = false;
        let mut guard_timeout_seen = false;
        let mut guard_malloc_timeout_seconds = DEFAULT_GUARD_MALLOC_TIMEOUT_SECONDS;
        let mut report_path = None;
        let mut index = 0;
        while index < args.len() {
            match args[index].as_str() {
                "--guard-malloc" if !guard_malloc => guard_malloc = true,
                "--calibrate" if !calibrate => calibrate = true,
                "--budget-seconds" if !budget_seen => {
                    index += 1;
                    budget_seconds = parse_positive_seconds(args.get(index))?;
                    budget_seen = true;
                }
                "--leaks-timeout-seconds" if !timeout_seen => {
                    index += 1;
                    leaks_timeout_seconds = parse_positive_seconds(args.get(index))?;
                    timeout_seen = true;
                }
                "--guard-malloc-timeout-seconds" if !guard_timeout_seen => {
                    index += 1;
                    guard_malloc_timeout_seconds = parse_positive_seconds(args.get(index))?;
                    guard_timeout_seen = true;
                }
                "--report" if report_path.is_none() => {
                    index += 1;
                    report_path = Some(PathBuf::from(args.get(index).ok_or_else(usage)?));
                }
                _ => return Err(usage()),
            }
            index += 1;
        }

        if guard_timeout_seen && !guard_malloc {
            return Err(usage());
        }
        if calibrate {
            Ok(Self::Calibrate {
                guard_malloc,
                budget_seconds,
                leaks_timeout_seconds,
                guard_malloc_timeout_seconds,
                report_path,
            })
        } else if budget_seen || report_path.is_some() {
            Err(usage())
        } else {
            Ok(Self::Verify {
                guard_malloc,
                leaks_timeout_seconds,
                guard_malloc_timeout_seconds,
            })
        }
    }
}

fn parse_positive_seconds(value: Option<&String>) -> Result<u64, String> {
    value
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .ok_or_else(usage)
}

fn usage() -> String {
    "usage: cargo xtask verify-ffi [--guard-malloc] [--leaks-timeout-seconds N] [--guard-malloc-timeout-seconds N] [--calibrate [--budget-seconds N] [--report PATH]]".to_owned()
}

pub(crate) fn run(root_dir: &Path, args: &[String]) -> i32 {
    match Options::parse(args) {
        Ok(Options::Verify {
            guard_malloc,
            leaks_timeout_seconds,
            guard_malloc_timeout_seconds,
        }) => {
            let report = verify(
                root_dir,
                guard_malloc,
                Timeouts {
                    leaks: leaks_timeout_seconds,
                    guard_malloc: guard_malloc_timeout_seconds,
                },
            );
            write_report(&report, report.verdict == Verdict::Pass)
        }
        Ok(Options::Calibrate {
            guard_malloc,
            budget_seconds,
            leaks_timeout_seconds,
            guard_malloc_timeout_seconds,
            report_path,
        }) => {
            let report = calibrate(
                root_dir,
                guard_malloc,
                budget_seconds,
                Timeouts {
                    leaks: leaks_timeout_seconds,
                    guard_malloc: guard_malloc_timeout_seconds,
                },
            );
            if let Some(path) = report_path {
                if let Err(error) = save_report(&path, &report.phases.clean_gate.report) {
                    eprintln!(
                        "failed to write admission report {}: {error}",
                        path.display()
                    );
                    return write_report(&report, false);
                }
            }
            write_report(&report, report.verdict == Verdict::Pass)
        }
        Err(error) => {
            eprintln!("{error}");
            let report = failed_report(
                root_dir,
                false,
                Timeouts {
                    leaks: DEFAULT_LEAKS_TIMEOUT_SECONDS,
                    guard_malloc: DEFAULT_GUARD_MALLOC_TIMEOUT_SECONDS,
                },
                error,
            );
            write_report(&report, false)
        }
    }
}

fn save_report(path: &Path, report: &VerifyFfiReport) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|error| error.to_string())?;
    serde_json::to_writer_pretty(file, report).map_err(|error| error.to_string())
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

fn calibrate(
    root_dir: &Path,
    guard_malloc: bool,
    budget_seconds: u64,
    timeouts: Timeouts,
) -> CalibrationReport {
    let total_started = Instant::now();
    let source_commit = git_head(root_dir);
    let source_clean = git_clean(root_dir);
    let mlx_c_dir = root_dir.join("mlx-sys/src/mlx-c");
    let mlx_c_commit = git_head(&mlx_c_dir);
    let mlx_c_clean = git_clean(&mlx_c_dir);
    let leaks_timeout = Duration::from_secs(timeouts.leaks);

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
            assess_environment_probe(
                capture_command(&mut probe_command, leaks_timeout),
                probe_started.elapsed(),
            )
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

    let deliberate_leak_detection = run_deliberate_leak_calibration(root_dir, leaks_timeout);

    let guard_qualification =
        guard_malloc.then(|| qualify_guard_malloc(Duration::from_secs(timeouts.guard_malloc)));

    eprintln!("running clean full verify-ffi gate");
    let clean_started = Instant::now();
    let mut clean_report = verify(root_dir, guard_malloc, timeouts);
    clean_report.source_clean &= source_clean;
    clean_report.mlx_c_clean &= mlx_c_clean && clean_report.mlx_c_commit == mlx_c_commit;
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
    let mut phases = CalibrationPhases {
        environment_probe,
        deliberate_leak_detection,
        guard_malloc: guard_qualification,
        clean_gate,
        budget,
    };
    let verdict = bind_qualification(&mut phases, &source_commit);

    CalibrationReport {
        schema_version: 1,
        command: "verify-ffi",
        mode: "calibration",
        source_commit,
        mlx_c_commit: git_head(&root_dir.join("mlx-sys/src/mlx-c")),
        budget_seconds,
        leaks_timeout_seconds: timeouts.leaks,
        phases,
        total_duration_ms: duration_ms(total_started.elapsed()),
        verdict,
    }
}

fn bind_qualification(phases: &mut CalibrationPhases, source_commit: &str) -> Verdict {
    if phases.clean_gate.report.source_commit != source_commit
        || !phases.clean_gate.report.source_clean
        || !phases.clean_gate.report.mlx_c_clean
    {
        phases.clean_gate.status = Verdict::Fail;
    }
    let verdict = calibration_verdict(phases);
    phases.clean_gate.report.verdict = verdict.clone();
    phases.clean_gate.report.qualification = Some(Qualification {
        source_commit: source_commit.to_owned(),
        environment_probe: phases.environment_probe.status.clone(),
        deliberate_leak_detection: phases.deliberate_leak_detection.status.clone(),
        guard_malloc: phases.guard_malloc.clone(),
        budget: phases.budget.status.clone(),
        verdict: verdict.clone(),
    });
    verdict
}

fn run_deliberate_leak_calibration(
    root_dir: &Path,
    leaks_timeout: Duration,
) -> CalibrationLeakPhase {
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
    assess_deliberate_leak(
        capture_command(&mut command, leaks_timeout),
        started.elapsed(),
    )
}

fn discover_calibration_binary(root_dir: &Path) -> Result<TestBinary, String> {
    eprintln!("discovering deliberate leak calibration executable");
    let discovery = Command::new("cargo")
        .args([
            "test",
            "--locked",
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

fn capture_command(
    command: &mut Command,
    timeout: Duration,
) -> Result<CapturedCommand, CaptureFailure> {
    configure_process_group(command);
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let started = Instant::now();
    let mut child = command.spawn().map_err(capture_io_failure)?;
    let stdout = child.stdout.take().expect("stdout configured as piped");
    let stderr = child.stderr.take().expect("stderr configured as piped");
    let stdout_reader = thread::spawn(move || read_pipe(stdout));
    let stderr_reader = thread::spawn(move || read_pipe(stderr));

    let mut status = None;
    loop {
        if status.is_none() {
            match child.try_wait() {
                Ok(completed) => status = completed,
                Err(error) => {
                    terminate_capture(&mut child, stdout_reader, stderr_reader);
                    return Err(capture_io_failure(error));
                }
            }
        }
        if status.is_some() && stdout_reader.is_finished() && stderr_reader.is_finished() {
            break;
        }
        if started.elapsed() >= timeout {
            terminate_capture(&mut child, stdout_reader, stderr_reader);
            return Err(CaptureFailure::Timeout {
                elapsed: started.elapsed(),
            });
        }
        thread::sleep(PROCESS_WAIT_POLL_INTERVAL.min(timeout.saturating_sub(started.elapsed())));
    }
    let stdout = join_pipe(stdout_reader).map_err(CaptureFailure::Io)?;
    let stderr = join_pipe(stderr_reader).map_err(CaptureFailure::Io)?;
    let status = status.expect("completed process checked before leaving wait loop");
    Ok(CapturedCommand {
        stdout: String::from_utf8_lossy(&stdout).into_owned(),
        stderr: String::from_utf8_lossy(&stderr).into_owned(),
        status: CapturedExit {
            exit_code: status.code(),
            signal: exit_signal(&status),
        },
    })
}

fn terminate_capture(
    child: &mut Child,
    stdout_reader: thread::JoinHandle<std::io::Result<Vec<u8>>>,
    stderr_reader: thread::JoinHandle<std::io::Result<Vec<u8>>>,
) {
    kill_process_group(child);
    let _ = child.wait();
    let _ = join_pipe(stdout_reader);
    let _ = join_pipe(stderr_reader);
}

fn capture_io_failure(error: std::io::Error) -> CaptureFailure {
    CaptureFailure::Io(SpawnFailure {
        kind: error.kind(),
        message: error.to_string(),
    })
}

fn read_pipe(mut pipe: impl Read) -> std::io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    pipe.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn join_pipe(
    reader: thread::JoinHandle<std::io::Result<Vec<u8>>>,
) -> Result<Vec<u8>, SpawnFailure> {
    reader
        .join()
        .map_err(|_| SpawnFailure {
            kind: std::io::ErrorKind::Other,
            message: "command output reader panicked".to_owned(),
        })?
        .map_err(|error| SpawnFailure {
            kind: error.kind(),
            message: error.to_string(),
        })
}

#[cfg(unix)]
fn configure_process_group(command: &mut Command) {
    use std::os::unix::process::CommandExt;

    command.process_group(0);
}

#[cfg(not(unix))]
fn configure_process_group(_command: &mut Command) {}

#[cfg(unix)]
fn kill_process_group(child: &mut Child) {
    unsafe extern "C" {
        fn kill(pid: i32, signal: i32) -> i32;
    }

    if let Ok(process_group) = i32::try_from(child.id()) {
        unsafe {
            kill(-process_group, 9);
        }
    }
    let _ = child.kill();
}

#[cfg(not(unix))]
fn kill_process_group(child: &mut Child) {
    let _ = child.kill();
}

fn assess_environment_probe(
    attempt: Result<CapturedCommand, CaptureFailure>,
    duration: Duration,
) -> CalibrationLeakPhase {
    let target = CALIBRATION_BINARY;
    let captured = match attempt {
        Ok(captured) => captured,
        Err(CaptureFailure::Timeout { elapsed }) => {
            return calibration_leak_failure(
                Verdict::Error,
                duration.max(elapsed),
                target,
                None,
                None,
                "leaks_timeout",
                format!("leaks timed out after {} ms", duration_ms(elapsed)),
            );
        }
        Err(CaptureFailure::Io(error)) => {
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
    attempt: Result<CapturedCommand, CaptureFailure>,
    duration: Duration,
) -> CalibrationLeakPhase {
    let captured = match attempt {
        Ok(captured) => captured,
        Err(CaptureFailure::Timeout { elapsed }) => {
            return calibration_leak_failure(
                Verdict::Error,
                duration.max(elapsed),
                CALIBRATION_BINARY,
                None,
                None,
                "leaks_timeout",
                format!("leaks timed out after {} ms", duration_ms(elapsed)),
            );
        }
        Err(CaptureFailure::Io(error)) => {
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
    let guard = phases
        .guard_malloc
        .as_ref()
        .map_or(Verdict::Pass, GuardMallocQualification::verdict);
    calibration_verdict_from_statuses([
        &guard,
        &phases.environment_probe.status,
        &phases.deliberate_leak_detection.status,
        &phases.clean_gate.status,
        &phases.budget.status,
    ])
}

fn calibration_verdict_from_statuses<const N: usize>(statuses: [&Verdict; N]) -> Verdict {
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

fn verify(root_dir: &Path, guard_malloc: bool, timeouts: Timeouts) -> VerifyFfiReport {
    let binaries = match discover_test_binaries(root_dir) {
        Ok(binaries) => binaries,
        Err(error) => {
            return failed_report(root_dir, guard_malloc, timeouts, error);
        }
    };
    let leaks_timeout = Duration::from_secs(timeouts.leaks);
    let guard_malloc_timeout = Duration::from_secs(timeouts.guard_malloc);

    let mut reports = Vec::with_capacity(binaries.len());
    let mut covered_leaks = Vec::new();
    let mut covered_guard_malloc = Vec::new();
    for binary in binaries {
        let test = run_tests(&binary);
        let (leaks, covered_path) = run_leaks(&binary, leaks_timeout);
        covered_leaks.extend(covered_path);
        let guard_malloc = guard_malloc.then(|| {
            let (check, covered_path) = run_guard_malloc(&binary, guard_malloc_timeout);
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
        timeouts,
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
    timeouts: Timeouts,
    discovery: Discovery,
    binaries: Vec<BinaryReport>,
    covered_binaries: CoveredBinaries,
    verdict: Verdict,
) -> VerifyFfiReport {
    VerifyFfiReport {
        schema_version: 1,
        command: "verify-ffi",
        leaks_timeout_seconds: timeouts.leaks,
        guard_malloc_timeout_seconds: timeouts.guard_malloc,
        qualification: None,
        source_commit: git_head(root_dir),
        github_sha: nonempty_env("GITHUB_SHA"),
        github_pr_head_sha: nonempty_env("GITHUB_PR_HEAD_SHA"),
        mlx_c_commit: git_head(&root_dir.join("mlx-sys/src/mlx-c")),
        source_clean: git_clean(root_dir),
        mlx_c_clean: git_clean(&root_dir.join("mlx-sys/src/mlx-c")),
        environment: ReportEnvironment {
            architecture: env::consts::ARCH,
            os: env::consts::OS,
            rustc: command_text(Command::new("rustc").arg("--version")),
        },
        guard_malloc_requested,
        discovery,
        binaries,
        covered_binaries,
        verdict,
    }
}

fn nonempty_env(name: &str) -> Option<String> {
    env::var(name).ok().filter(|value| !value.is_empty())
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
        .args([
            "test",
            "--locked",
            "--workspace",
            "--no-run",
            "--message-format=json",
        ])
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
    let mut actual = binaries
        .iter()
        .map(|binary| {
            format!(
                "{}|{}|{}",
                stable_package_id(&binary.package_id),
                binary.target,
                binary.target_kind.join(",")
            )
        })
        .collect::<Vec<_>>();
    actual.sort();
    if actual != workspace_target_ids(root_dir)? {
        return Err("discovered test binaries differ from workspace inventory".into());
    }
    Ok(binaries)
}

fn workspace_target_ids(root_dir: &Path) -> Result<Vec<String>, String> {
    let output = Command::new("cargo")
        .args(["metadata", "--locked", "--format-version=1"])
        .current_dir(root_dir)
        .output()
        .map_err(|error| error.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }
    let metadata = serde_json::from_slice(&output.stdout).map_err(|error| error.to_string())?;
    metadata_target_ids(&metadata)
}

fn metadata_target_ids(metadata: &serde_json::Value) -> Result<Vec<String>, String> {
    let members = metadata["workspace_members"]
        .as_array()
        .ok_or("missing workspace members")?;
    let nodes = metadata["resolve"]["nodes"]
        .as_array()
        .ok_or("missing resolved features")?;
    let mut ids = Vec::new();
    for package in metadata["packages"].as_array().ok_or("missing packages")? {
        if !members.contains(&package["id"]) {
            continue;
        }
        let features = &nodes
            .iter()
            .find(|node| node["id"] == package["id"])
            .ok_or("missing workspace feature resolution")?["features"];
        for target in package["targets"].as_array().ok_or("missing targets")? {
            let kinds = target["kind"].as_array().ok_or("missing target kind")?;
            if target["test"] != true
                || !kinds.iter().any(|kind| {
                    ["lib", "proc-macro", "bin", "test"]
                        .iter()
                        .any(|expected| kind == expected)
                })
            {
                continue;
            }
            if target["required-features"]
                .as_array()
                .is_some_and(|required| {
                    required.iter().any(|feature| {
                        !features
                            .as_array()
                            .is_some_and(|enabled| enabled.contains(feature))
                    })
                })
            {
                continue;
            }
            ids.push(format!(
                "{}|{}|{}",
                stable_package_id(package["id"].as_str().ok_or("missing package id")?),
                target["name"].as_str().ok_or("missing target name")?,
                kinds
                    .iter()
                    .map(|kind| kind.as_str().ok_or("invalid target kind"))
                    .collect::<Result<Vec<_>, _>>()?
                    .join(",")
            ));
        }
    }
    ids.sort();
    if ids.is_empty() {
        return Err("empty workspace test inventory".into());
    }
    Ok(ids)
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

fn run_leaks(binary: &TestBinary, timeout: Duration) -> (LeakCheck, Option<String>) {
    if !binary.leaks_applicable() {
        return (
            LeakCheck {
                status: LeakStatus::NotApplicable,
                duration_ms: 0,
                result: None,
                tool_status: None,
                failure: None,
                error: None,
            },
            None,
        );
    }

    let path = binary.path.display().to_string();
    eprintln!("running leaks for {} ({path})", binary.target);
    let started = Instant::now();
    let mut command = Command::new("leaks");
    command
        .args(["--atExit", "--"])
        .arg(&binary.path)
        .arg("--test-threads=1");
    let output = match capture_command(&mut command, timeout) {
        Ok(output) => output,
        Err(CaptureFailure::Timeout { elapsed }) => {
            return (
                LeakCheck {
                    status: LeakStatus::Error,
                    duration_ms: duration_ms(elapsed),
                    result: None,
                    tool_status: None,
                    failure: Some("leaks_timeout".to_owned()),
                    error: Some(format!("leaks timed out after {} ms", duration_ms(elapsed))),
                },
                Some(path),
            );
        }
        Err(CaptureFailure::Io(error)) => {
            return (
                LeakCheck {
                    status: LeakStatus::Error,
                    duration_ms: duration_ms(started.elapsed()),
                    result: None,
                    tool_status: None,
                    failure: None,
                    error: Some(format!("failed to run leaks: {}", error.message)),
                },
                None,
            );
        }
    };
    let elapsed = started.elapsed();

    let text = captured_output(&output);
    let tool_status = Some(process_exit_from_captured(&output.status));
    let result = match parse_leaks_report(&text) {
        Ok(result) => result,
        Err(error) => {
            eprintln!("leaks report for {path} could not be parsed: {error}");
            return (
                LeakCheck {
                    status: LeakStatus::Error,
                    duration_ms: duration_ms(elapsed),
                    result: None,
                    tool_status,
                    failure: None,
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
            duration_ms: duration_ms(elapsed),
            result: Some(result),
            tool_status,
            failure: None,
            error: None,
        },
        Some(path),
    )
}

fn run_guard_malloc(binary: &TestBinary, timeout: Duration) -> (GuardMallocCheck, Option<String>) {
    let path = binary.path.display().to_string();
    eprintln!("running guard malloc for {} ({path})", binary.target);
    let started = Instant::now();
    let mut command = Command::new(&binary.path);
    command
        .arg("--test-threads=1")
        .env("DYLD_INSERT_LIBRARIES", "/usr/lib/libgmalloc.dylib");
    let check = assess_guard_malloc(
        capture_command(&mut command, timeout),
        started.elapsed(),
        None,
    );
    (check, Some(path))
}

fn assess_guard_malloc(
    attempt: Result<CapturedCommand, CaptureFailure>,
    duration: Duration,
    control: Option<&str>,
) -> GuardMallocCheck {
    let output = match attempt {
        Ok(output) => output,
        Err(error) => {
            let (failure, message) = match error {
                CaptureFailure::Timeout { elapsed } => (
                    "guard_malloc_timeout",
                    format!("Guard Malloc timed out after {} ms", duration_ms(elapsed)),
                ),
                CaptureFailure::Io(error) => ("guard_malloc_spawn_failed", error.message),
            };
            eprintln!("{failure}: {message}");
            return GuardMallocCheck {
                status: Verdict::Error,
                duration_ms: duration_ms(duration),
                process_status: None,
                failure: Some(failure.into()),
                error: Some(message),
            };
        }
    };
    let text = captured_output(&output);
    let loaded = text
        .lines()
        .any(|line| line.starts_with("GuardMalloc[") && line.contains(": version "));
    let reached = control.is_none_or(|name| {
        text.lines()
            .any(|line| line == format!("guard-malloc-control:{name}"))
    });
    let detected = match control {
        Some("use_after_free") => matches!(output.status.signal, Some(10 | 11)),
        Some("double_free") => output.status.signal == Some(6),
        _ => output.status.exit_code == Some(0) && output.status.signal.is_none(),
    };
    let success = loaded && reached && detected;
    let failure = if !loaded {
        Some("guard_malloc_unavailable")
    } else if !reached {
        Some("guard_malloc_control_not_reached")
    } else if !detected {
        Some("guard_malloc_unexpected_exit")
    } else {
        None
    };
    if !success {
        eprintln!("Guard Malloc check failed ({control:?}): {failure:?}\n{text}");
    }
    GuardMallocCheck {
        status: if !loaded {
            Verdict::Error
        } else if success {
            Verdict::Pass
        } else {
            Verdict::Fail
        },
        duration_ms: duration_ms(duration),
        process_status: Some(process_exit_from_captured(&output.status)),
        failure: failure.map(str::to_owned),
        error: (!success).then_some(text),
    }
}

fn qualify_guard_malloc(timeout: Duration) -> GuardMallocQualification {
    let prepared = (|| -> Result<_, String> {
        let parent = env::var_os("RUNNER_TEMP")
            .map(PathBuf::from)
            .unwrap_or_else(env::temp_dir);
        let directory = tempfile::Builder::new()
            .prefix("guard-malloc-")
            .tempdir_in(parent)
            .map_err(|error| error.to_string())?;
        let source = directory.path().join("control.c");
        let binary = directory.path().join("control");
        std::fs::write(
            &source,
            include_str!("../tests/fixtures/guard-malloc-control.c"),
        )
        .map_err(|error| error.to_string())?;
        let output = capture_command(
            Command::new("cc")
                .arg("-O0")
                .arg(&source)
                .arg("-o")
                .arg(&binary),
            timeout,
        )
        .map_err(|error| format!("Guard Malloc control build failed: {error:?}"))?;
        if output.status.exit_code != Some(0) || output.status.signal.is_some() {
            return Err(format!(
                "Guard Malloc control build failed: {}",
                captured_output(&output)
            ));
        }
        Ok((directory, binary))
    })();
    let run = |name: &str| {
        eprintln!("qualifying Guard Malloc: {name}");
        let started = Instant::now();
        let attempt = match &prepared {
            Ok((_, binary)) => capture_command(
                Command::new(binary)
                    .arg(name)
                    .env("DYLD_INSERT_LIBRARIES", "/usr/lib/libgmalloc.dylib"),
                timeout,
            ),
            Err(error) => Err(capture_io_failure(std::io::Error::other(error.clone()))),
        };
        assess_guard_malloc(attempt, started.elapsed(), Some(name))
    };
    GuardMallocQualification {
        clean: run("clean"),
        use_after_free: run("use_after_free"),
        double_free: run("double_free"),
    }
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

fn failed_report(
    root_dir: &Path,
    guard_malloc_requested: bool,
    timeouts: Timeouts,
    error: String,
) -> VerifyFfiReport {
    report_header(
        root_dir,
        guard_malloc_requested,
        timeouts,
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

fn process_exit_from_captured(status: &CapturedExit) -> ProcessExit {
    ProcessExit {
        exit_code: status.exit_code,
        signal: status.signal,
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
                duration_ms: 0,
                result: None,
                tool_status: None,
                failure: None,
                error: None,
            },
            test,
            guard_malloc: None,
        }
    }

    #[test]
    fn metadata_inventory_respects_members_features_and_target_kinds() {
        let metadata = serde_json::json!({
            "workspace_members": ["path+file:///repo/demo#demo@1.0"],
            "resolve": {"nodes": [{"id": "path+file:///repo/demo#demo@1.0", "features": ["enabled"]}]},
            "packages": [
                {"id": "dependency", "targets": [{"name": "dependency", "kind": ["lib"], "test": true}]},
                {"id": "path+file:///repo/demo#demo@1.0", "targets": [
                    {"name": "demo", "kind": ["proc-macro"], "test": true},
                    {"name": "included", "kind": ["test"], "test": true, "required-features": ["enabled"]},
                    {"name": "excluded", "kind": ["test"], "test": true, "required-features": ["disabled"]},
                    {"name": "example", "kind": ["example"], "test": false},
                    {"name": "build", "kind": ["custom-build"], "test": false}
                ]}
            ]
        });
        assert_eq!(
            metadata_target_ids(&metadata).unwrap(),
            ["demo@1.0|demo|proc-macro", "demo@1.0|included|test"]
        );
        let mut missing = metadata;
        missing["resolve"] = serde_json::Value::Null;
        assert!(metadata_target_ids(&missing).is_err());
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
                guard_malloc_timeout_seconds: DEFAULT_GUARD_MALLOC_TIMEOUT_SECONDS,
                guard_malloc: false,
                leaks_timeout_seconds: DEFAULT_LEAKS_TIMEOUT_SECONDS,
            }
        );
        assert_eq!(
            Options::parse(&["--guard-malloc".to_owned()]).unwrap(),
            Options::Verify {
                guard_malloc_timeout_seconds: DEFAULT_GUARD_MALLOC_TIMEOUT_SECONDS,
                guard_malloc: true,
                leaks_timeout_seconds: DEFAULT_LEAKS_TIMEOUT_SECONDS,
            }
        );
        assert_eq!(
            Options::parse(&["--calibrate".to_owned()]).unwrap(),
            Options::Calibrate {
                guard_malloc: false,
                guard_malloc_timeout_seconds: DEFAULT_GUARD_MALLOC_TIMEOUT_SECONDS,
                report_path: None,
                budget_seconds: DEFAULT_CALIBRATION_BUDGET_SECONDS,
                leaks_timeout_seconds: DEFAULT_LEAKS_TIMEOUT_SECONDS,
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
                guard_malloc: false,
                guard_malloc_timeout_seconds: DEFAULT_GUARD_MALLOC_TIMEOUT_SECONDS,
                report_path: None,
                budget_seconds: 900,
                leaks_timeout_seconds: DEFAULT_LEAKS_TIMEOUT_SECONDS,
            }
        );
        assert_eq!(
            Options::parse(&[
                "--guard-malloc".to_owned(),
                "--leaks-timeout-seconds".to_owned(),
                "7".to_owned(),
            ])
            .unwrap(),
            Options::Verify {
                guard_malloc_timeout_seconds: DEFAULT_GUARD_MALLOC_TIMEOUT_SECONDS,
                guard_malloc: true,
                leaks_timeout_seconds: 7,
            }
        );
        assert_eq!(
            Options::parse(&[
                "--calibrate".to_owned(),
                "--leaks-timeout-seconds".to_owned(),
                "9".to_owned(),
                "--budget-seconds".to_owned(),
                "900".to_owned(),
            ])
            .unwrap(),
            Options::Calibrate {
                guard_malloc: false,
                guard_malloc_timeout_seconds: DEFAULT_GUARD_MALLOC_TIMEOUT_SECONDS,
                report_path: None,
                budget_seconds: 900,
                leaks_timeout_seconds: 9,
            }
        );
        assert!(Options::parse(&["--unknown".to_owned()]).is_err());
        assert!(Options::parse(&[
            "--calibrate".to_owned(),
            "--budget-seconds".to_owned(),
            "0".to_owned(),
        ])
        .is_err());
        assert!(Options::parse(&["--leaks-timeout-seconds".to_owned(), "0".to_owned(),]).is_err());
        assert!(
            Options::parse(&["--guard-malloc".to_owned(), "--guard-malloc".to_owned()]).is_err()
        );
    }

    #[test]
    fn guard_malloc_qualification_requires_detected_faults_and_clean_control() {
        for (control, code, signal, banner, marker, expected) in [
            ("clean", Some(0), None, true, true, Verdict::Pass),
            ("use_after_free", None, Some(11), true, true, Verdict::Pass),
            ("double_free", None, Some(6), true, true, Verdict::Pass),
            ("use_after_free", Some(0), None, true, true, Verdict::Fail),
            ("double_free", Some(0), None, true, true, Verdict::Fail),
            ("clean", None, Some(11), true, true, Verdict::Fail),
            (
                "use_after_free",
                None,
                Some(11),
                false,
                true,
                Verdict::Error,
            ),
            ("use_after_free", None, Some(11), true, false, Verdict::Fail),
            ("double_free", Some(1), None, true, true, Verdict::Fail),
            ("use_after_free", None, Some(9), true, true, Verdict::Fail),
        ] {
            let output = CapturedCommand {
                stdout: String::new(),
                stderr: format!(
                    "{}{}",
                    if banner {
                        "GuardMalloc[probe-1]: version 1\n"
                    } else {
                        ""
                    },
                    if marker {
                        format!("guard-malloc-control:{control}\n")
                    } else {
                        String::new()
                    }
                ),
                status: CapturedExit {
                    exit_code: code,
                    signal,
                },
            };
            assert_eq!(
                assess_guard_malloc(Ok(output), Duration::ZERO, Some(control)).status,
                expected,
                "{control} {code:?} {signal:?} {banner} {marker}"
            );
        }
    }

    #[test]
    fn guard_malloc_timeout_and_missing_tool_are_errors() {
        for failure in [
            CaptureFailure::Timeout {
                elapsed: Duration::from_secs(1),
            },
            capture_io_failure(std::io::Error::from(std::io::ErrorKind::NotFound)),
        ] {
            assert_eq!(
                assess_guard_malloc(Err(failure), Duration::ZERO, None).status,
                Verdict::Error
            );
        }
    }

    #[test]
    fn exported_report_cannot_pass_failed_qualification() {
        let directory = tempfile::tempdir().unwrap();
        let make_probe = || {
            assess_environment_probe(
                Ok(captured_command(
                    "Process 1: 0 leaks for 0 total leaked bytes.",
                    "",
                    0,
                )),
                Duration::ZERO,
            )
        };
        let good_control = || GuardMallocCheck {
            status: Verdict::Pass,
            duration_ms: 1,
            process_status: None,
            failure: None,
            error: None,
        };
        let mut phases = CalibrationPhases {
            environment_probe: make_probe(),
            deliberate_leak_detection: make_probe(),
            guard_malloc: Some(GuardMallocQualification {
                clean: good_control(),
                use_after_free: good_control(),
                double_free: good_control(),
            }),
            clean_gate: CleanGatePhase {
                status: Verdict::Pass,
                duration_ms: 1,
                report: failed_report(
                    directory.path(),
                    true,
                    Timeouts {
                        leaks: 1,
                        guard_malloc: 1,
                    },
                    "fixture".into(),
                ),
            },
            budget: assess_budget(Duration::from_secs(1), Duration::from_secs(2)),
        };
        phases.clean_gate.report.source_commit = "workspace".into();
        phases.clean_gate.report.source_clean = true;
        phases.clean_gate.report.mlx_c_clean = true;
        assert_eq!(bind_qualification(&mut phases, "workspace"), Verdict::Pass);
        for (source_clean, mlx_c_clean) in [(false, true), (true, false)] {
            phases.clean_gate.report.source_clean = source_clean;
            phases.clean_gate.report.mlx_c_clean = mlx_c_clean;
            assert_eq!(bind_qualification(&mut phases, "workspace"), Verdict::Fail);
            phases.clean_gate.status = Verdict::Pass;
        }
        phases.clean_gate.report.source_clean = true;
        phases.clean_gate.report.mlx_c_clean = true;
        phases.guard_malloc.as_mut().unwrap().use_after_free.status = Verdict::Fail;
        assert_eq!(bind_qualification(&mut phases, "workspace"), Verdict::Fail);
        let path = directory.path().join("admission.json");
        save_report(&path, &phases.clean_gate.report).unwrap();
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(report["verdict"], "fail");
        assert_eq!(report["qualification"]["source_commit"], "workspace");
        assert_eq!(
            report["qualification"]["guard_malloc"]["use_after_free"]["status"],
            "fail"
        );
        phases.guard_malloc.as_mut().unwrap().use_after_free.status = Verdict::Pass;
        assert_eq!(
            bind_qualification(&mut phases, "changed-checkout"),
            Verdict::Fail
        );
    }

    #[test]
    fn calibration_accepts_guard_malloc() {
        assert_eq!(
            Options::parse(&[
                "--calibrate".into(),
                "--guard-malloc".into(),
                "--budget-seconds".into(),
                "3600".into(),
                "--leaks-timeout-seconds".into(),
                "600".into(),
                "--guard-malloc-timeout-seconds".into(),
                "1200".into(),
                "--report".into(),
                "/scratch/verify-ffi.json".into(),
            ])
            .unwrap(),
            Options::Calibrate {
                guard_malloc: true,
                budget_seconds: 3600,
                leaks_timeout_seconds: 600,
                guard_malloc_timeout_seconds: 1200,
                report_path: Some("/scratch/verify-ffi.json".into()),
            }
        );
        for args in [
            vec!["--guard-malloc-timeout-seconds", "1"],
            vec!["--guard-malloc", "--guard-malloc-timeout-seconds", "0"],
            vec!["--calibrate", "--report"],
            vec!["--report", "/scratch/report.json"],
        ] {
            assert!(
                Options::parse(&args.into_iter().map(str::to_owned).collect::<Vec<_>>()).is_err()
            );
        }
    }

    #[test]
    fn guard_malloc_rejects_uninstrumented_success() {
        let output = captured_command("test result: ok. 1 passed; 0 failed", "", 0);
        let check = assess_guard_malloc(Ok(output), Duration::ZERO, None);
        assert_eq!(check.status, Verdict::Error);
        assert_eq!(check.failure.as_deref(), Some("guard_malloc_unavailable"));
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
            Err(CaptureFailure::Io(SpawnFailure {
                kind: std::io::ErrorKind::NotFound,
                message: "No such file or directory".to_owned(),
            })),
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

    #[cfg(unix)]
    #[test]
    fn hanging_leaks_child_is_killed_and_classified_as_tool_error() {
        let started = Instant::now();
        let mut command = Command::new("sh");
        command.args(["-c", "sleep 1 &"]);

        let attempt = capture_command(&mut command, Duration::from_millis(50));
        let phase = assess_environment_probe(attempt, started.elapsed());

        assert_eq!(phase.status, Verdict::Error);
        assert_eq!(phase.failure.as_deref(), Some("leaks_timeout"));
        assert!(phase.duration_ms >= 50);
        // Guard Malloc pages every allocation, so spawning and killing the child costs orders of
        // magnitude more than the timeout itself; the bound only rules out waiting for the child.
        let spawn_allowance = if std::env::var("DYLD_INSERT_LIBRARIES")
            .is_ok_and(|value| value.contains("libgmalloc"))
        {
            300_000
        } else {
            500
        };
        assert!(phase.duration_ms < spawn_allowance);
        assert!(phase.error.unwrap().contains("timed out"));
    }

    #[test]
    fn maximum_timeout_does_not_overflow_the_deadline() {
        let mut command = Command::new("true");

        let captured = capture_command(&mut command, Duration::from_secs(u64::MAX)).unwrap();

        assert_eq!(captured.status.exit_code, Some(0));
        assert_eq!(captured.status.signal, None);
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
