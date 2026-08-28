use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::Write;
use std::fs;
use std::path::Path;
use std::process::Command;

const REQUIRED_PROTECTED_PATHS: &[&str] = &[
    "conformance/**",
    "mlx-tests/tests/conformance.rs",
    "mlx-tests/tests/conformance/oracle.rs",
    "xtask/src/main.rs",
    "xtask/src/verify_oracle_boundary.rs",
];
const REQUIRED_IMPLEMENTATION_PATHS: &[&str] =
    &["mlx-rs/src/**", "mlx-tests/tests/conformance/adapters.rs"];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BoundaryConfig {
    schema_version: u32,
    protected_paths: Vec<String>,
    implementation_paths: Vec<String>,
    staged_case_marker: String,
}

#[derive(Deserialize)]
struct CorpusIntegrity {
    generator_digest: String,
    fixture_shards: Option<BTreeMap<String, String>>,
    suites: Vec<String>,
}

#[derive(Serialize)]
struct ChangeSetReport {
    id: String,
    protected_paths: Vec<String>,
    implementation_paths: Vec<String>,
    other_paths: Vec<String>,
    mixed: bool,
    marker_present: bool,
    staged_case_admitted: bool,
    verdict: &'static str,
}

#[derive(Serialize)]
struct DigestReport {
    expected: Option<String>,
    actual: Option<String>,
    verdict: &'static str,
    error: Option<String>,
}

#[derive(Serialize)]
struct ShardReport {
    checked: usize,
    verdict: &'static str,
    errors: Vec<String>,
}

#[derive(Serialize)]
struct IntegrityReport {
    generator: DigestReport,
    fixture_shards: ShardReport,
}

#[derive(Serialize)]
struct BoundaryReport {
    command: &'static str,
    mode: &'static str,
    base: Option<String>,
    staged_case_marker: Option<String>,
    change_sets: Vec<ChangeSetReport>,
    integrity: IntegrityReport,
    errors: Vec<String>,
    verdict: &'static str,
}

struct RawChangeSet {
    id: String,
    paths: Vec<String>,
    message: Option<String>,
}

pub fn run(repo_root: &Path, args: &[String]) -> i32 {
    let (base, mut errors) = parse_args(args);
    let config_path = repo_root.join("conformance/protected-paths.json");
    let config = read_json::<BoundaryConfig>(&config_path).map_err(|error| {
        errors.push(error);
    });

    let mode = if base.is_some() {
        "commit_range"
    } else {
        "working_tree"
    };
    let raw_change_sets = if errors.is_empty() {
        collect_change_sets(repo_root, base.as_deref()).unwrap_or_else(|error| {
            errors.push(error);
            Vec::new()
        })
    } else {
        Vec::new()
    };
    let (marker, change_sets) = match config {
        Ok(config) => {
            if config.schema_version != 1 {
                errors.push(format!(
                    "{} has unsupported schema_version {}",
                    config_path.display(),
                    config.schema_version
                ));
            }
            if let Err(error) = validate_config(&config) {
                errors.push(error);
            }
            let marker = config.staged_case_marker.clone();
            let reports = raw_change_sets
                .into_iter()
                .map(|change_set| evaluate_change_set(change_set, &config))
                .collect();
            (Some(marker), reports)
        }
        Err(()) => (None, Vec::new()),
    };
    let integrity = verify_integrity(repo_root);
    let passed = errors.is_empty()
        && change_sets.iter().all(|report| report.verdict == "pass")
        && integrity.generator.verdict == "pass"
        && integrity.fixture_shards.verdict == "pass";
    let report = BoundaryReport {
        command: "verify-oracle-boundary",
        mode,
        base,
        staged_case_marker: marker,
        change_sets,
        integrity,
        errors,
        verdict: if passed { "pass" } else { "fail" },
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("serialize oracle boundary report")
    );
    if report
        .change_sets
        .iter()
        .any(|change_set| change_set.staged_case_admitted)
    {
        eprintln!("ORACLE BOUNDARY OVERRIDE: staged mixed oracle/implementation case admitted");
    }
    i32::from(!passed)
}

fn parse_args(args: &[String]) -> (Option<String>, Vec<String>) {
    match args {
        [] => (None, Vec::new()),
        [flag, base] if flag == "--base" && !base.is_empty() => (Some(base.clone()), Vec::new()),
        _ => (
            None,
            vec!["usage: cargo run -p xtask -- verify-oracle-boundary [--base <ref>]".into()],
        ),
    }
}

fn validate_config(config: &BoundaryConfig) -> Result<(), String> {
    if config.protected_paths.is_empty()
        || config.implementation_paths.is_empty()
        || config.staged_case_marker.is_empty()
    {
        return Err("protected-paths manifest contains an empty required field".into());
    }
    for pattern in config
        .protected_paths
        .iter()
        .chain(&config.implementation_paths)
    {
        if pattern.starts_with('/')
            || pattern.contains("..")
            || (pattern.contains('*') && !pattern.ends_with("/**"))
        {
            return Err(format!("unsupported protected-paths pattern {pattern}"));
        }
    }
    for required in REQUIRED_PROTECTED_PATHS {
        if !config.protected_paths.iter().any(|path| path == required) {
            return Err(format!("protected-paths manifest is missing {required}"));
        }
    }
    for required in REQUIRED_IMPLEMENTATION_PATHS {
        if !config
            .implementation_paths
            .iter()
            .any(|path| path == required)
        {
            return Err(format!("protected-paths manifest is missing {required}"));
        }
    }
    if matches_any(
        "mlx-tests/tests/conformance/adapters.rs",
        &config.protected_paths,
    ) {
        return Err("adapter dispatch must not be a protected path".into());
    }
    Ok(())
}

fn collect_change_sets(repo_root: &Path, base: Option<&str>) -> Result<Vec<RawChangeSet>, String> {
    match base {
        Some(base) => collect_commit_range(repo_root, base),
        None => collect_working_tree(repo_root),
    }
}

fn collect_working_tree(repo_root: &Path) -> Result<Vec<RawChangeSet>, String> {
    let mut paths = git_paths(
        repo_root,
        &["diff", "--no-renames", "--name-only", "-z", "HEAD", "--"],
    )?;
    paths.extend(git_paths(
        repo_root,
        &["ls-files", "--others", "--exclude-standard", "-z", "--"],
    )?);
    paths.sort();
    paths.dedup();
    Ok(vec![RawChangeSet {
        id: "working-tree".into(),
        paths,
        message: None,
    }])
}

fn collect_commit_range(repo_root: &Path, base: &str) -> Result<Vec<RawChangeSet>, String> {
    git(
        repo_root,
        &["rev-parse", "--verify", &format!("{base}^{{commit}}")],
    )?;
    let output = git(
        repo_root,
        &["rev-list", "--reverse", &format!("{base}..HEAD")],
    )?;
    output
        .lines()
        .filter(|commit| !commit.is_empty())
        .map(|commit| {
            let mut paths = git_paths(
                repo_root,
                &[
                    "diff-tree",
                    "--root",
                    "-m",
                    "--no-commit-id",
                    "--no-renames",
                    "--name-only",
                    "-r",
                    "-z",
                    commit,
                ],
            )?;
            paths.sort();
            paths.dedup();
            let message = git(repo_root, &["log", "-1", "--format=%B", commit])?;
            Ok(RawChangeSet {
                id: commit.to_string(),
                paths,
                message: Some(message),
            })
        })
        .collect()
}

fn git_paths(repo_root: &Path, args: &[&str]) -> Result<Vec<String>, String> {
    let output = git_bytes(repo_root, args)?;
    output
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .map(|path| {
            String::from_utf8(path.to_vec())
                .map_err(|_| format!("git returned a non-UTF-8 path for {}", args.join(" ")))
        })
        .collect()
}

fn git(repo_root: &Path, args: &[&str]) -> Result<String, String> {
    String::from_utf8(git_bytes(repo_root, args)?)
        .map_err(|_| format!("git returned non-UTF-8 output for {}", args.join(" ")))
}

fn git_bytes(repo_root: &Path, args: &[&str]) -> Result<Vec<u8>, String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo_root)
        .output()
        .map_err(|error| format!("failed to run git {}: {error}", args.join(" ")))?;
    if output.status.success() {
        Ok(output.stdout)
    } else {
        Err(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

fn evaluate_change_set(change_set: RawChangeSet, config: &BoundaryConfig) -> ChangeSetReport {
    let mut protected_paths = Vec::new();
    let mut implementation_paths = Vec::new();
    let mut other_paths = Vec::new();
    for path in change_set.paths {
        if matches_any(&path, &config.protected_paths) {
            protected_paths.push(path);
        } else if matches_any(&path, &config.implementation_paths) {
            implementation_paths.push(path);
        } else {
            other_paths.push(path);
        }
    }
    let mixed = !protected_paths.is_empty() && !implementation_paths.is_empty();
    let marker_present = change_set.message.as_deref().is_some_and(|message| {
        message
            .lines()
            .any(|line| line.trim_start().starts_with(&config.staged_case_marker))
    });
    let staged_case_admitted = mixed && marker_present;
    ChangeSetReport {
        id: change_set.id,
        protected_paths,
        implementation_paths,
        other_paths,
        mixed,
        marker_present,
        staged_case_admitted,
        verdict: if mixed && !marker_present {
            "fail"
        } else {
            "pass"
        },
    }
}

fn matches_any(path: &str, patterns: &[String]) -> bool {
    patterns
        .iter()
        .any(|pattern| match pattern.strip_suffix("/**") {
            Some(prefix) => path == prefix || path.starts_with(&format!("{prefix}/")),
            None => path == pattern,
        })
}

fn verify_integrity(repo_root: &Path) -> IntegrityReport {
    let corpus_path = repo_root.join("conformance/corpus.json");
    let corpus = read_json::<CorpusIntegrity>(&corpus_path);
    let generator_path = repo_root.join("conformance/generate.py");
    let actual_generator = hash_file(&generator_path);
    let generator = match (&corpus, actual_generator) {
        (Ok(corpus), Ok(actual)) if corpus.generator_digest == actual => DigestReport {
            expected: Some(corpus.generator_digest.clone()),
            actual: Some(actual),
            verdict: "pass",
            error: None,
        },
        (Ok(corpus), Ok(actual)) => DigestReport {
            expected: Some(corpus.generator_digest.clone()),
            actual: Some(actual),
            verdict: "fail",
            error: Some("generator digest does not match conformance/generate.py".into()),
        },
        (Ok(corpus), Err(error)) => DigestReport {
            expected: Some(corpus.generator_digest.clone()),
            actual: None,
            verdict: "fail",
            error: Some(error),
        },
        (Err(error), _) => DigestReport {
            expected: None,
            actual: None,
            verdict: "fail",
            error: Some(error.clone()),
        },
    };
    let fixture_shards = match corpus {
        Ok(corpus) => verify_shards(repo_root, &corpus),
        Err(error) => ShardReport {
            checked: 0,
            verdict: "fail",
            errors: vec![error],
        },
    };
    IntegrityReport {
        generator,
        fixture_shards,
    }
}

fn verify_shards(repo_root: &Path, corpus: &CorpusIntegrity) -> ShardReport {
    let Some(recorded) = &corpus.fixture_shards else {
        return ShardReport {
            checked: 0,
            verdict: "fail",
            errors: vec![
                "corpus.json has no fixture_shards; regenerate with conformance/generate.py".into(),
            ],
        };
    };
    let mut errors = Vec::new();
    let expected_paths = corpus
        .suites
        .iter()
        .filter_map(|suite| {
            suite
                .strip_prefix("suites/")
                .and_then(|path| path.strip_suffix(".json"))
                .map(|name| format!("fixtures/{name}.safetensors"))
        })
        .collect::<Vec<_>>();
    if expected_paths.len() != corpus.suites.len() {
        errors.push("corpus.json contains an invalid suite path".into());
    }
    for path in recorded.keys() {
        if !expected_paths.contains(path) {
            errors.push(format!("fixture_shards contains unexpected path {path}"));
        }
    }
    let mut checked = 0;
    for path in expected_paths {
        let Some(expected) = recorded.get(&path) else {
            errors.push(format!(
                "fixture_shards is missing {path}; regenerate corpus"
            ));
            continue;
        };
        match hash_file(&repo_root.join("conformance").join(&path)) {
            Ok(actual) => {
                checked += 1;
                if *expected != actual {
                    errors.push(format!("fixture shard digest mismatch for {path}"));
                }
            }
            Err(error) => errors.push(error),
        }
    }
    ShardReport {
        checked,
        verdict: if errors.is_empty() { "pass" } else { "fail" },
        errors,
    }
}

fn hash_file(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    Ok(format!("sha256:{}", hex_sha256(&bytes)))
}

fn hex_sha256(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(64);
    for byte in sha256(bytes) {
        write!(&mut encoded, "{byte:02x}").expect("write SHA-256 hex");
    }
    encoded
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    const INITIAL: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const ROUND: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    let mut padded = bytes.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    let mut hash = INITIAL;
    for chunk in padded.chunks(64) {
        let mut words = [0u32; 64];
        for (index, word) in words.iter_mut().take(16).enumerate() {
            *word = u32::from_be_bytes(chunk[index * 4..index * 4 + 4].try_into().unwrap());
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = hash;
        for index in 0..64 {
            let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ (!e & g);
            let temp1 = h
                .wrapping_add(sum1)
                .wrapping_add(choose)
                .wrapping_add(ROUND[index])
                .wrapping_add(words[index]);
            let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = sum0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        for (value, compressed) in hash.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *value = value.wrapping_add(compressed);
        }
    }

    let mut digest = [0u8; 32];
    for (bytes, value) in digest.chunks_mut(4).zip(hash) {
        bytes.copy_from_slice(&value.to_be_bytes());
    }
    digest
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("{}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn config() -> BoundaryConfig {
        BoundaryConfig {
            schema_version: 1,
            protected_paths: vec!["conformance/**".into(), "oracle.rs".into()],
            implementation_paths: vec!["mlx-rs/src/**".into(), "adapters.rs".into()],
            staged_case_marker: "oracle-change:".into(),
        }
    }

    fn change_set(paths: &[&str], message: Option<&str>) -> ChangeSetReport {
        evaluate_change_set(
            RawChangeSet {
                id: "synthetic".into(),
                paths: paths.iter().map(|path| (*path).into()).collect(),
                message: message.map(str::to_string),
            },
            &config(),
        )
    }

    fn integrity_tree() -> TempDir {
        let directory = tempfile::tempdir().unwrap();
        let conformance = directory.path().join("conformance");
        fs::create_dir(&conformance).unwrap();
        fs::create_dir(conformance.join("fixtures")).unwrap();
        fs::write(conformance.join("generate.py"), b"generator").unwrap();
        fs::write(conformance.join("fixtures/a.safetensors"), b"fixture").unwrap();
        let generator_digest = hash_file(&conformance.join("generate.py")).unwrap();
        let shard_digest = hash_file(&conformance.join("fixtures/a.safetensors")).unwrap();
        let corpus = serde_json::json!({
            "generator_digest": generator_digest,
            "fixture_shards": {"fixtures/a.safetensors": shard_digest},
            "suites": ["suites/a.json"]
        });
        fs::write(
            conformance.join("corpus.json"),
            serde_json::to_vec(&corpus).unwrap(),
        )
        .unwrap();
        directory
    }

    #[test]
    fn mixed_change_fails() {
        assert_eq!(
            change_set(&["conformance/corpus.json", "mlx-rs/src/lib.rs"], None).verdict,
            "fail"
        );
    }

    #[test]
    fn oracle_only_passes() {
        assert_eq!(change_set(&["oracle.rs"], None).verdict, "pass");
    }

    #[test]
    fn implementation_only_passes() {
        assert_eq!(change_set(&["adapters.rs"], None).verdict, "pass");
    }

    #[test]
    fn staged_marker_admits_mixed_change() {
        let report = change_set(
            &["conformance/corpus.json", "mlx-rs/src/lib.rs"],
            Some("fix: staged case\n\noracle-change: update expectation"),
        );
        assert_eq!(report.verdict, "pass");
        assert!(report.staged_case_admitted);
    }

    #[test]
    fn generator_digest_tamper_fails() {
        let directory = integrity_tree();
        fs::write(
            directory.path().join("conformance/generate.py"),
            b"tampered",
        )
        .unwrap();
        assert_eq!(verify_integrity(directory.path()).generator.verdict, "fail");
    }

    #[test]
    fn fixture_shard_tamper_fails() {
        let directory = integrity_tree();
        fs::write(
            directory.path().join("conformance/fixtures/a.safetensors"),
            b"tampered",
        )
        .unwrap();
        assert_eq!(
            verify_integrity(directory.path()).fixture_shards.verdict,
            "fail"
        );
    }

    #[test]
    fn sha256_matches_known_vector() {
        assert_eq!(
            hex_sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
