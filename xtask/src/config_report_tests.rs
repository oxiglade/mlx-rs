use super::*;

fn fixture() -> (Vec<Configuration>, Vec<Value>, Value) {
    let configs = configurations(
        &serde_json::from_str(include_str!("../../ledger/supported-feature-matrix.json")).unwrap(),
    )
    .unwrap();
    let identity = json!({"source_commit": "checkout", "mlx_c_commit": "native", "nested_mlx_version": "0.32.2", "cargo_lock_sha256": "lock"});
    let reports =
        configs
            .iter()
            .map(|c| {
                let mut r = identity.clone();
                r.as_object_mut().unwrap().extend(json!({
            "schema_version": 1, "configuration_id": c.id, "dirty": false,
            "checkpoints_required": c.checkpoints_required,
            "profile": c.profile, "toolchain": c.toolchain,
            "target_triple": c.target_triple, "command": test_command(c),
            "rustc": if c.toolchain == "stable" { "rustc 1.98.0" } else { "rustc 1.88.0" },
            "macos": "26.0", "sdk": "26.0", "resolved_features": c.resolved_features,
            "native": c.native, "verdict": "pass", "exit_code": 0,
            "tests": [{"id": "demo::lib::demo", "passed": 2, "failed": 0, "ignored": 0,
                "filtered_out": 0, "summaries": 1, "not_run": false, "not_run_reasons": [], "excluded_by_configuration": false, "outcome": "pass"}]
        }).as_object().unwrap().clone());
                r["native"]["mlx_version"] = "0.32.2".into();
                r["native"]["mlx_commit"] = "nested-commit".into();
                r["rust_profiles"] = c
                    .resolved_features
                    .keys()
                    .map(|name| {
                        (
                            name.clone(),
                            json!({
                                "opt_level": if c.profile == "release" { "3" } else { "0" },
                                "debug_assertions": c.profile == "debug",
                            }),
                        )
                    })
                    .collect::<serde_json::Map<_, _>>()
                    .into();
                r
            })
            .collect();
    (configs, reports, identity)
}

fn validate_set(
    configs: &[Configuration],
    reports: &[Value],
    identity: &Value,
    expected: impl Fn(&Configuration, &Value) -> Result<BTreeSet<String>, String>,
) -> Result<Value, String> {
    for (config, report) in configs.iter().zip(reports) {
        let mut report = report.clone();
        let mut expected = expected(config, &report)?;
        if config.id == "macos-arm64-default" {
            for id in [
                "mlx-lm::test::parity",
                "mlx-lm::test::sentinel",
                "mlx-tests::test::conformance",
                "mlx-tests::test::state_optimizers",
                "mlx-tests::test::state_compile",
                "mlx-tests::test::state_transforms",
            ] {
                let mut run = report["tests"][0].clone();
                run["id"] = id.into();
                report["tests"].as_array_mut().unwrap().push(run);
                expected.insert(id.to_owned());
            }
        }
        validate_report(config, &report, identity, &expected)?;
    }
    Ok(json!({"verdict": "pass"}))
}

fn check(configs: &[Configuration], reports: &[Value], identity: &Value) -> Result<Value, String> {
    validate_set(configs, reports, identity, |_, _| {
        Ok(BTreeSet::from(["demo::lib::demo".into()]))
    })
}

fn rejects(change: impl FnOnce(&mut Value), expected: &str) {
    let (configs, mut reports, identity) = fixture();
    change(&mut reports[0]);
    assert!(check(&configs, &reports, &identity)
        .unwrap_err()
        .contains(expected));
}

#[test]
fn qualifies_declared_configurations() {
    let (configs, reports, identity) = fixture();
    assert_eq!(
        check(&configs, &reports, &identity).unwrap()["verdict"],
        "pass"
    );
}

#[test]
fn rejects_non_jit_cache_for_jit_report() {
    let (configs, mut reports, identity) = fixture();
    let report = reports
        .iter_mut()
        .find(|r| r["configuration_id"] == "macos-arm64-metal-jit")
        .unwrap();
    report["native"]["MLX_METAL_JIT"] = "OFF".into();
    assert!(check(&configs, &reports, &identity)
        .unwrap_err()
        .contains("native setting mismatch: MLX_METAL_JIT"));
}

#[test]
fn rejects_release_report_built_in_debug() {
    for field in ["profile", "rust_profiles", "native"] {
        let (configs, mut reports, identity) = fixture();
        let debug = reports
            .iter()
            .find(|r| r["configuration_id"] == "macos-arm64-cpu-only")
            .unwrap()[field]
            .clone();
        let report = reports
            .iter_mut()
            .find(|r| r["configuration_id"] == "macos-arm64-cpu-only-release")
            .unwrap();
        report[field] = debug;
        let expected = match field {
            "profile" => "command, profile, target or toolchain mismatch",
            "rust_profiles" => "compiled Rust profile mismatch",
            _ => "native setting mismatch: CMAKE_BUILD_TYPE",
        };
        assert!(check(&configs, &reports, &identity)
            .unwrap_err()
            .contains(expected));
    }
}

#[test]
fn captures_compiled_rust_profile() {
    let metadata = json!({"packages": [{"id": "mlx-rs-id", "name": "mlx-rs"}]});
    let artifact = json!({
        "reason": "compiler-artifact", "package_id": "mlx-rs-id",
        "target": {"kind": ["lib"], "name": "mlx_rs"},
        "features": ["accelerate"],
        "profile": {"opt_level": "3", "debug_assertions": false},
    });
    let evidence = build_evidence(&artifact.to_string(), &metadata).unwrap();
    assert_eq!(evidence.rust_profiles["mlx-rs"], artifact["profile"]);
    let mut debug = artifact.clone();
    debug["profile"]["opt_level"] = "0".into();
    debug["profile"]["debug_assertions"] = true.into();
    assert!(build_evidence(&format!("{artifact}\n{debug}"), &metadata).is_err());
}

#[test]
fn rejects_failed_report() {
    rejects(|r| r["verdict"] = "fail".into(), "failed or skipped");
}

#[test]
fn rejects_skipped_report() {
    rejects(|r| r["verdict"] = "skipped".into(), "failed or skipped");
}

#[test]
fn rejects_nonzero_command_exit() {
    rejects(|r| r["exit_code"] = 101.into(), "failed or skipped");
}

#[test]
fn rejects_wrong_resolved_features() {
    rejects(
        |r| r["resolved_features"]["mlx-sys"] = json!(["accelerate", "metal"]),
        "features mismatch",
    );
}

#[test]
fn rejects_wrong_native_setting() {
    rejects(
        |r| r["native"]["MLX_BUILD_METAL"] = "ON".into(),
        "native setting",
    );
}

#[test]
fn rejects_not_run_binary() {
    rejects(
        |r| r["tests"][0]["not_run"] = true.into(),
        "skipped or incomplete",
    );
}

#[test]
fn rejects_zero_test_binary() {
    rejects(|r| r["tests"][0]["passed"] = 0.into(), "zero tests");
}

#[test]
fn rejects_missing_binary() {
    rejects(|r| r["tests"] = json!([]), "inventory differs");
}

#[test]
fn rejects_duplicate_binary() {
    rejects(
        |r| {
            let run = r["tests"][0].clone();
            r["tests"].as_array_mut().unwrap().push(run);
        },
        "duplicate test",
    );
}

#[test]
fn rejects_failed_binary_even_if_report_claims_pass() {
    rejects(|r| r["tests"][0]["failed"] = 1.into(), "failed, skipped");
}

#[test]
fn rejects_incomplete_binary() {
    rejects(|r| r["tests"][0]["summaries"] = 0.into(), "incomplete");
}

#[test]
fn rejects_filtered_tests() {
    rejects(|r| r["tests"][0]["filtered_out"] = 1.into(), "skipped");
}

#[test]
fn rejects_wrong_environment_and_command() {
    for field in [
        "mlx_c_commit",
        "nested_mlx_version",
        "cargo_lock_sha256",
        "profile",
        "toolchain",
        "target_triple",
        "command",
        "rustc",
        "macos",
        "sdk",
    ] {
        let (configs, mut reports, identity) = fixture();
        reports[0][field] = Value::Null;
        assert!(check(&configs, &reports, &identity).is_err(), "{field}");
    }
    rejects(|r| r["dirty"] = true.into(), "dirty");
}

#[test]
fn rejects_wrong_msrv_and_unstable_toolchains() {
    let (configs, mut reports, identity) = fixture();
    reports[2]["rustc"] = "rustc 1.98.0".into();
    assert!(check(&configs, &reports, &identity)
        .unwrap_err()
        .contains("rustc version"));
    rejects(|r| r["rustc"] = "rustc 1.99.0-nightly".into(), "not stable");
}

#[test]
fn rejects_not_run_and_zero_doctests() {
    let (configs, mut reports, identity) = fixture();
    for report in &mut reports {
        report["tests"][0]["id"] = "demo::doctest::demo".into();
    }
    let expected =
        |_: &Configuration, _: &Value| Ok(BTreeSet::from(["demo::doctest::demo".into()]));
    validate_set(&configs, &reports, &identity, expected).unwrap();
    reports[0]["tests"][0]["not_run"] = true.into();
    assert!(validate_set(&configs, &reports, &identity, expected).is_err());
    reports[0]["tests"][0]["not_run"] = false.into();
    reports[0]["tests"][0]["passed"] = 0.into();
    assert!(validate_set(&configs, &reports, &identity, expected).is_err());
}

#[test]
fn matrix_rejects_invalid_shapes() {
    let matrix: Value =
        serde_json::from_str(include_str!("../../ledger/supported-feature-matrix.json")).unwrap();
    for field in [
        "id",
        "required",
        "profile",
        "toolchain",
        "native",
        "resolved_features",
        "packages",
        "allow_empty",
    ] {
        let mut broken = matrix.clone();
        broken["builds"][0].as_object_mut().unwrap().remove(field);
        assert!(configurations(&broken).is_err(), "{field}");
    }
    let mut broken = matrix.clone();
    broken["builds"][1]["id"] = broken["builds"][0]["id"].clone();
    assert!(configurations(&broken).is_err());
    let mut broken = matrix;
    for b in broken["builds"].as_array_mut().unwrap() {
        b["required"] = false.into();
    }
    assert!(configurations(&broken).is_err());
}

#[test]
fn rejects_stale_built_nested_mlx() {
    rejects(
        |r| r["native"]["mlx_version"] = "0.30.6".into(),
        "nested MLX identity",
    );
}

#[test]
fn command_pins_compilation_target() {
    let (configs, _, _) = fixture();
    let command = test_command(&configs[0]);
    assert!(command
        .windows(2)
        .any(|p| p == ["--target", "aarch64-apple-darwin"]));
}

#[test]
fn rejects_jit_cache_for_non_jit_configuration() {
    rejects(
        |r| r["native"]["MLX_METAL_JIT"] = "ON".into(),
        "native setting",
    );
}

#[test]
fn detects_json_not_run_diagnostic() {
    let runs = parse_runs("Doc-tests demo\n{\"status\":\"NOT RUN\"}\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0s").unwrap();
    assert_eq!(runs[0]["not_run"], true);
}

#[test]
fn nested_harness_summary_does_not_inflate_parent_counts() {
    let log = "Running tests/state_compile.rs (/tmp/state)\nrunning 15 tests\ntest nested ...\nrunning 1 test\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 14 filtered out; finished in 0s\nok\ntest result: ok. 15 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0s";
    let runs = parse_runs(log).unwrap();
    assert_eq!(runs[0]["passed"], 15);
    assert_eq!(runs[0]["filtered_out"], 0);
    assert_eq!(runs[0]["nested_summaries"][0]["passed"], 1);
}

#[test]
fn incomplete_second_doctest_group_cannot_pass() {
    let log = "Doc-tests demo\nrunning 1 test\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0s\nrunning 1 test";
    let runs = parse_runs(log).unwrap();
    assert_eq!(runs[0]["outcome"], "incomplete");
}

#[test]
fn captured_workspace_output_preserves_target_counts() {
    let runs = parse_runs(include_str!("../tests/fixtures/configuration-output.txt")).unwrap();
    assert_eq!(runs.len(), 40);
    let xtask = runs
        .iter()
        .find(|r| r["id"].as_str().unwrap().contains("/xtask-"))
        .unwrap();
    assert_eq!(xtask["passed"], 117);
    assert_eq!(xtask["summaries"], 1);
    assert_eq!(runs.iter().filter(|r| r["not_run"] == true).count(), 1);
    let state = runs
        .iter()
        .find(|r| r["id"].as_str().unwrap().contains("/state_compile-"))
        .unwrap();
    assert_eq!(state["passed"], 15);
    assert_eq!(state["filtered_out"], 0);
    assert!(runs.iter().all(|r| r["outcome"] == "pass"));
}

#[test]
fn empty_allowlist_matches_captured_workspace_execution() {
    let runs = parse_runs(include_str!("../tests/fixtures/configuration-output.txt")).unwrap();
    let empty: BTreeSet<String> = runs
        .iter()
        .filter(|r| r["passed"] == 0)
        .map(|r| {
            let id = r["id"].as_str().unwrap();
            if let Some(name) = id.strip_prefix("doc:") {
                format!("{}::doctest::{name}", name.replace('_', "-"))
            } else {
                let file = Path::new(id).file_name().unwrap().to_str().unwrap();
                let name = file.rsplit_once('-').unwrap().0;
                if name.starts_with("mlx_") {
                    format!("{}::lib::{name}", name.replace('_', "-"))
                } else {
                    format!("mlx-tests::test::{name}")
                }
            }
        })
        .collect();
    let (configs, _, _) = fixture();
    let config = configs
        .iter()
        .find(|c| c.id == "macos-arm64-default")
        .unwrap();
    assert_eq!(
        config.allow_empty.iter().cloned().collect::<BTreeSet<_>>(),
        empty
    );
}

#[test]
fn only_manifest_absent_checkpoints_can_be_excluded_by_configuration() {
    let (configs, reports, identity) = fixture();
    for (c, mut report) in configs.iter().zip(reports) {
        let id = "mlx-lm::test::real_checkpoints";
        let expected = BTreeSet::from([id.to_owned()]);
        let run = &mut report["tests"][0];
        run["id"] = id.into();
        run["not_run"] = true.into();
        run["not_run_reasons"] = json!(["NOT RUN: local real-checkpoint manifest not supplied"]);
        run["excluded_by_configuration"] = (!c.checkpoints_required).into();
        assert_eq!(
            validate_report(c, &report, &identity, &expected).is_ok(),
            !c.checkpoints_required
        );
        for (field, value) in [
            ("not_run_reasons", json!(["NOT RUN: other fixture absent"])),
            (
                "not_run_reasons",
                json!([
                    "NOT RUN: local real-checkpoint manifest not supplied",
                    "NOT RUN: another test"
                ]),
            ),
            ("id", json!("mlx-lm::test::parity")),
            ("outcome", json!("incomplete")),
            ("filtered_out", json!(1)),
            ("failed", json!(1)),
            ("passed", json!(0)),
        ] {
            let mut bad = report.clone();
            bad["tests"][0][field] = value;
            assert!(
                validate_report(c, &bad, &identity, &expected).is_err(),
                "{}: {field}",
                c.id
            );
        }
    }
}

#[test]
fn ordinary_default_excludes_only_absent_checkpoints_but_admission_requires_them() {
    let (configs, reports, identity) = fixture();
    let index = configs
        .iter()
        .position(|c| c.id == "macos-arm64-default")
        .unwrap();
    let c = &configs[index];
    let mut report = reports[index].clone();
    let ordinary = execution_configuration(c, false);
    assert!(!ordinary.checkpoints_required);
    assert!(execution_configuration(c, true).checkpoints_required);
    report["checkpoints_required"] = false.into();
    let id = "mlx-lm::test::real_checkpoints";
    report["tests"][0]["id"] = id.into();
    report["tests"][0]["not_run"] = true.into();
    report["tests"][0]["not_run_reasons"] =
        json!(["NOT RUN: local real-checkpoint manifest not supplied"]);
    report["tests"][0]["excluded_by_configuration"] = true.into();
    let mut expected = BTreeSet::from([id.to_owned()]);
    for id in [
        "mlx-lm::test::parity",
        "mlx-lm::test::sentinel",
        "mlx-tests::test::conformance",
        "mlx-tests::test::state_optimizers",
        "mlx-tests::test::state_compile",
        "mlx-tests::test::state_transforms",
    ] {
        let mut run = reports[index]["tests"][0].clone();
        run["id"] = id.into();
        report["tests"].as_array_mut().unwrap().push(run);
        expected.insert(id.to_owned());
    }
    validate_report(&ordinary, &report, &identity, &expected).unwrap();
    assert!(validate_report(c, &report, &identity, &expected).is_err());
    report["tests"][0]["not_run_reasons"] = json!(["NOT RUN: another reason"]);
    assert!(validate_report(&ordinary, &report, &identity, &expected).is_err());
}

#[test]
fn qualification_failure_overrides_successful_cargo_exit() {
    assert_eq!(report_exit_code(0, &json!({"verdict": "fail"})), 1);
    assert_eq!(report_exit_code(0, &json!({"verdict": "pass"})), 0);
    assert_eq!(report_exit_code(101, &json!({"verdict": "fail"})), 101);
}

#[test]
fn identity_creates_product_lockfile_once() {
    let repo = tempfile::tempdir().unwrap();
    let root = repo.path();
    fs::write(
        root.join("Cargo.toml"),
        "[package]\nname = \"product\"\nversion = \"0.1.0\"\nedition = \"2021\"\n[lib]\npath = \"lib.rs\"\n[workspace]\n",
    )
    .unwrap();
    fs::write(root.join("lib.rs"), "").unwrap();
    let native = root.join("mlx-sys/src/mlx-c");
    fs::create_dir_all(&native).unwrap();
    fs::write(native.join("CMakeLists.txt"), "GIT_TAG v0.32.2)\n").unwrap();
    output(root, &["git", "init", "-q"]).unwrap();
    output(
        root,
        &[
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
    )
    .unwrap();
    let lock_path = root.join("Cargo.lock");
    assert!(!lock_path.exists());
    let first = identity(root).unwrap();
    let lock = fs::read_to_string(&lock_path).unwrap();
    assert!(lock.contains("name = \"product\""));
    assert!(!lock.contains("name = \"xtask\""));
    let retained = format!("{lock}\n# retained resolution\n");
    fs::write(&lock_path, &retained).unwrap();
    let second = identity(root).unwrap();
    assert_ne!(first["cargo_lock_sha256"], second["cargo_lock_sha256"]);
    assert_eq!(fs::read_to_string(lock_path).unwrap(), retained);
}

#[test]
fn default_requires_complete_parity_and_sentinel_execution() {
    let (configs, reports, identity) = fixture();
    let c = configs
        .iter()
        .find(|c| c.id == "macos-arm64-default")
        .unwrap();
    let report = reports
        .iter()
        .find(|r| r["configuration_id"] == c.id)
        .unwrap();
    let mut report = report.clone();
    for id in [
        "mlx-lm::test::parity",
        "mlx-lm::test::sentinel",
        "mlx-tests::test::conformance",
        "mlx-tests::test::state_optimizers",
        "mlx-tests::test::state_compile",
        "mlx-tests::test::state_transforms",
    ] {
        let mut run = report["tests"][0].clone();
        run["id"] = id.into();
        report["tests"].as_array_mut().unwrap().push(run);
    }
    let expected = report["tests"]
        .as_array()
        .unwrap()
        .iter()
        .map(|run| run["id"].as_str().unwrap().to_owned())
        .collect();
    validate_report(c, &report, &identity, &expected).unwrap();
    for index in [1, 2] {
        let mut bad = report.clone();
        bad["tests"][index]["ignored"] = 1.into();
        assert!(validate_report(c, &bad, &identity, &expected).is_err());
        let mut bad = report.clone();
        bad["tests"][index]["summaries"] = 2.into();
        assert!(validate_report(c, &bad, &identity, &expected).is_err());
    }
}

#[test]
fn default_checkpoint_configuration_cannot_be_removed_or_disabled() {
    let matrix: Value =
        serde_json::from_str(include_str!("../../ledger/supported-feature-matrix.json")).unwrap();
    for field in ["required", "checkpoints_required", "toolchain"] {
        let mut bad = matrix.clone();
        let default = bad["builds"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|build| build["id"] == "macos-arm64-default")
            .unwrap();
        default[field] = if field == "toolchain" {
            json!("1.88.0")
        } else {
            json!(false)
        };
        assert!(configurations(&bad).is_err(), "{field}");
    }
    let mut missing = matrix;
    missing["builds"]
        .as_array_mut()
        .unwrap()
        .retain(|build| build["id"] != "macos-arm64-default");
    assert!(configurations(&missing).is_err());
}
