use super::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{cell::Cell, os::unix::fs::symlink};

fn frozen_cases() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../conformance/mlx-lm/hub_cases.json");
    let cases: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    assert_eq!(cases["schema_version"], 1);
    assert_eq!(cases["cohort"], "hub_contract_v1");
    cases
}

fn string<'a>(value: &'a Value, key: &str) -> &'a str {
    value[key]
        .as_str()
        .unwrap_or_else(|| panic!("missing string {key}: {value}"))
}

fn fixture(name: &str, file: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../conformance/mlx-lm/fixtures")
        .join(name)
        .join(file)
}

// The receipt is private; the golden uses the logical name for deliberate absences.
fn stored_receipt(mut value: Value) -> Value {
    let absent = value
        .as_object_mut()
        .unwrap()
        .remove("recorded_absences")
        .unwrap();
    value["absent_files"] = absent;
    value
}

fn logical_receipt(mut value: Value) -> Value {
    let absent = value
        .as_object_mut()
        .unwrap()
        .remove("absent_files")
        .unwrap();
    value["recorded_absences"] = absent;
    value
}

struct CaseCache {
    root: PathBuf,
    snapshot: PathBuf,
    blobs: PathBuf,
    case: Value,
    applied: RefCell<BTreeSet<usize>>,
    calls: RefCell<Vec<Value>>,
    clients: Cell<usize>,
}

impl CaseCache {
    fn new(directory: &Path, symbols: &Value, case: &Value) -> Self {
        let directory = fs::canonicalize(directory).unwrap();
        let snapshot = directory.join(string(symbols, "snapshot"));
        let blobs = directory.join(string(symbols, "blobs"));
        let root = blobs.parent().unwrap().parent().unwrap().to_owned();
        fs::create_dir_all(&snapshot).unwrap();
        fs::create_dir_all(&blobs).unwrap();
        assert_eq!(case["setup"]["file_layout"], "same_repository_blob_links");
        let cache = Self {
            root,
            snapshot,
            blobs,
            case: case.clone(),
            applied: RefCell::default(),
            calls: RefCell::default(),
            clients: Cell::new(0),
        };
        for name in case["selected_files"].as_array().unwrap() {
            let name = name.as_str().unwrap();
            let bytes = fs::read(fixture(string(case, "fixture"), name)).unwrap();
            let blob = cache.blobs.join(format!("{:x}", Sha256::digest(&bytes)));
            fs::write(&blob, bytes).unwrap();
            let path = cache.snapshot.join(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            symlink(blob, path).unwrap();
        }
        for (revision, commit) in case["setup"]["refs"].as_object().unwrap() {
            cache.write_ref(revision, commit.as_str().unwrap());
        }
        cache.mutate("before_receipt", None).unwrap();
        if !case["setup"]["receipt"].is_null() {
            for identity in case["setup"]["receipt"]["selected_files"]
                .as_array()
                .unwrap()
            {
                let bytes = fs::read(cache.snapshot.join(string(identity, "path"))).unwrap();
                assert_eq!(
                    json!(bytes.len()),
                    identity["size"],
                    "setup size: {}",
                    identity["path"]
                );
                assert_eq!(
                    json!(format!("{:x}", Sha256::digest(&bytes))),
                    identity["sha256"],
                    "setup hash: {}",
                    identity["path"]
                );
            }
            let receipt = cache.receipt_path();
            fs::create_dir_all(receipt.parent().unwrap()).unwrap();
            fs::write(
                receipt,
                serde_json::to_vec(&stored_receipt(case["setup"]["receipt"].clone())).unwrap(),
            )
            .unwrap();
        }
        cache.mutate("before_validation", None).unwrap();
        cache
    }

    fn receipt_path(&self) -> PathBuf {
        self.blobs
            .parent()
            .unwrap()
            .join(".mlx-lm/snapshots")
            .join(format!(
                "{}.json",
                self.snapshot.file_name().unwrap().to_str().unwrap()
            ))
    }

    fn write_ref(&self, revision: &str, commit: &str) {
        let path = self.blobs.parent().unwrap().join("refs").join(revision);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, commit).unwrap();
    }

    fn mutate(&self, stage: &str, mut download: Option<&mut PathBuf>) -> Result<(), HubError> {
        for (index, mutation) in self.case["setup"]["mutations"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            if string(mutation, "at") != stage {
                continue;
            }
            assert!(
                self.applied.borrow_mut().insert(index),
                "mutation applied twice"
            );
            let path = mutation["path"]
                .as_str()
                .map(|name| self.snapshot.join(name));
            match string(mutation, "operation") {
                "move_ref" => self.write_ref(
                    string(&self.case["request"], "revision"),
                    string(mutation, "commit"),
                ),
                "return_commit" => {
                    let path = download.as_deref_mut().unwrap();
                    let other = self
                        .snapshot
                        .parent()
                        .unwrap()
                        .join(string(mutation, "commit"))
                        .join(path.file_name().unwrap());
                    fs::create_dir_all(other.parent().unwrap()).unwrap();
                    fs::copy(&*path, &other).unwrap();
                    *path = other;
                }
                "transport_error" => {
                    return Err(hf_hub::api::sync::ApiError::IoError(std::io::Error::other(
                        "scripted transport failure",
                    ))
                    .into())
                }
                "replace_bytes" => {
                    fs::write(download.as_deref().unwrap(), string(mutation, "utf8")).unwrap()
                }
                "remove_file" => fs::remove_file(path.unwrap()).unwrap(),
                "break_blob_link" => {
                    fs::remove_file(fs::canonicalize(path.unwrap()).unwrap()).unwrap()
                }
                "copy_fixture_file" => {
                    fs::copy(
                        fixture(string(mutation, "fixture"), string(mutation, "path")),
                        path.unwrap(),
                    )
                    .unwrap();
                }
                "flip_byte" => {
                    let path = path.unwrap();
                    let mut bytes = fs::read(&path).unwrap();
                    bytes[mutation["offset"].as_u64().unwrap() as usize] ^= 1;
                    fs::write(path, bytes).unwrap();
                }
                "truncate" => {
                    fs::OpenOptions::new()
                        .write(true)
                        .open(path.unwrap())
                        .unwrap()
                        .set_len(mutation["size"].as_u64().unwrap())
                        .unwrap();
                }
                "set_json" => {
                    let path = path.unwrap();
                    let source = fs::read_to_string(&path).unwrap();
                    let mut value: Value = serde_json::from_str(&source).unwrap();
                    let key = string(mutation, "key");
                    let field = serde_json::to_string(key).unwrap();
                    let before =
                        format!("{field}: {}", serde_json::to_string(&value[key]).unwrap());
                    value[key] = mutation["value"].clone();
                    let after = format!("{field}: {}", serde_json::to_string(&value[key]).unwrap());
                    // Preserve the frozen fixture's numeric spelling (Python emits 1e-05).
                    let bytes = source.replace(&before, &after);
                    assert_eq!(serde_json::from_str::<Value>(&bytes).unwrap(), value);
                    fs::write(path, bytes).unwrap();
                }
                operation @ ("link_other_repository_blob"
                | "link_other_snapshot_file"
                | "link_snapshot_directory_outside_cache") => {
                    let path = path.unwrap();
                    let target = match operation {
                        "link_other_repository_blob" => self
                            .root
                            .join("models--other--model/blobs")
                            .join(path.file_name().unwrap()),
                        "link_other_snapshot_file" => self
                            .snapshot
                            .parent()
                            .unwrap()
                            .join(OTHER)
                            .join(path.file_name().unwrap()),
                        _ => self
                            .root
                            .parent()
                            .unwrap()
                            .join("outside")
                            .join(path.file_name().unwrap()),
                    };
                    fs::create_dir_all(target.parent().unwrap()).unwrap();
                    fs::copy(&path, &target).unwrap();
                    fs::remove_file(&path).unwrap();
                    if operation == "link_snapshot_directory_outside_cache" {
                        let escape = self.snapshot.join("escape");
                        symlink(target.parent().unwrap(), &escape).unwrap();
                        symlink(escape.join(path.file_name().unwrap()), path).unwrap();
                    } else {
                        symlink(target, path).unwrap();
                    }
                }
                other => panic!("unconsumed mutation operation: {other}"),
            }
        }
        Ok(())
    }

    fn path_symbol(&self, path: &Path) -> String {
        for (symbol, base) in [
            ("snapshot", &self.snapshot),
            ("blobs", &self.blobs),
            ("cache", &self.root),
        ] {
            if let Ok(relative) = path.strip_prefix(base) {
                return format!("{symbol}/{}", relative.display());
            }
        }
        path.to_string_lossy().into_owned()
    }
}

impl HubTransport for &CaseCache {
    fn info(&self, repo: &str, revision: &str) -> Result<SnapshotInfo, HubError> {
        self.calls
            .borrow_mut()
            .push(json!({"operation":"info", "repo":repo, "revision":revision}));
        self.mutate("info", None)?;
        let info = SnapshotInfo {
            commit: string(&self.case, "returned_commit").into(),
            siblings: serde_json::from_value(self.case["siblings"].clone()).unwrap(),
        };
        self.mutate("after_info", None)?;
        Ok(info)
    }

    fn download(&self, repo: &str, commit: &str, filename: &str) -> Result<PathBuf, HubError> {
        self.calls.borrow_mut().push(
            json!({"operation":"download", "repo":repo, "commit":commit, "filename":filename}),
        );
        let mut path = self
            .root
            .join(format!("models--{}", repo.replace('/', "--")))
            .join("snapshots")
            .join(commit)
            .join(filename);
        self.mutate(&format!("download:{filename}"), Some(&mut path))?;
        Ok(path)
    }
}

fn error_value(cache: &CaseCache, error: &HubError) -> Value {
    let (variant, fields) = match error {
        HubError::RevisionMismatch { expected, actual } => (
            "RevisionMismatch",
            json!({"expected":expected,"actual":actual}),
        ),
        HubError::OfflineCacheMiss { repo, revision } => {
            ("OfflineCacheMiss", json!({"repo":repo,"revision":revision}))
        }
        HubError::MissingFile {
            repo,
            revision,
            filename,
        } => (
            "MissingFile",
            json!({"repo":repo,"revision":revision,"filename":filename}),
        ),
        HubError::UnsafePath { path } => ("UnsafePath", json!({"path":cache.path_symbol(path)})),
        HubError::SnapshotIntegrity { path, .. } => {
            ("SnapshotIntegrity", json!({"path":cache.path_symbol(path)}))
        }
        HubError::Api(_) => ("Api", json!({})),
        HubError::Load(LoadError::Weights(crate::WeightError::ConflictingIndex { key, shard })) => {
            (
                "Load::Weights::ConflictingIndex",
                json!({"key":key,"shard":shard}),
            )
        }
        HubError::Load(LoadError::Config(crate::ConfigError::UnsupportedArchitecture(value))) => (
            "Load::Config::UnsupportedArchitecture",
            json!({"value":value}),
        ),
        other => return json!({"unexpected_error":format!("{other:?}")}),
    };
    json!({"variant":format!("HubError::{variant}"), "fields":fields})
}

fn compare(failures: &mut Vec<String>, label: &str, actual: Value, expected: &Value) {
    if actual != *expected {
        failures.push(format!("{label}: expected {expected}, got {actual}"));
    }
}

fn run_case(symbols: &Value, case: &Value) -> Vec<String> {
    let root = tempfile::tempdir().unwrap();
    let cache = CaseCache::new(root.path(), symbols, case);
    let request = &case["request"];
    let make_client = |_: &Path| {
        cache.clients.set(cache.clients.get() + 1);
        Ok(&cache)
    };
    let options = options(
        &cache.root,
        string(request, "revision"),
        request["offline"].as_bool().unwrap(),
    );
    let mut failures = Vec::new();
    let expected = &case["expected"];
    let result = if string(case, "name").starts_with("public_offline_") {
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            load_with(string(request, "repo"), options, make_client)
                .map(|model| model.hub_provenance().unwrap().clone())
        })
    } else {
        resolve_with(string(request, "repo"), options, make_client).and_then(|resolved| {
            resolved.validate()?;
            let receipt = logical_receipt(serde_json::to_value(&resolved.receipt).unwrap());
            compare(
                &mut failures,
                "resolved receipt",
                receipt,
                &expected["receipt"],
            );
            Ok(resolved.provenance)
        })
    };
    let actual_provenance = match &result {
        Ok(p) => {
            json!({"repo":p.repo,"requested_revision":p.requested_revision,"resolved_revision":p.resolved_revision})
        }
        Err(_) => Value::Null,
    };
    compare(
        &mut failures,
        "error",
        result
            .err()
            .map(|error| error_value(&cache, &error))
            .unwrap_or(Value::Null),
        &expected["error"],
    );
    compare(
        &mut failures,
        "provenance",
        actual_provenance,
        &expected["provenance"],
    );
    let receipt = match fs::read(cache.receipt_path()) {
        Ok(bytes) => logical_receipt(serde_json::from_slice(&bytes).unwrap()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Value::Null,
        Err(error) => panic!("read receipt: {error}"),
    };
    if !receipt.is_null() {
        compare(
            &mut failures,
            "selected files",
            json!(receipt["selected_files"]
                .as_array()
                .unwrap()
                .iter()
                .map(|file| &file["path"])
                .collect::<Vec<_>>()),
            &expected["selected_files"],
        );
        compare(
            &mut failures,
            "recorded absences",
            receipt["recorded_absences"].clone(),
            &expected["recorded_absences"],
        );
    }
    compare(
        &mut failures,
        "recorded receipt",
        receipt,
        &expected["receipt"],
    );
    compare(
        &mut failures,
        "client constructions",
        json!(cache.clients.get()),
        &expected["client_constructions"],
    );
    compare(
        &mut failures,
        "transport calls",
        json!(*cache.calls.borrow()),
        &expected["transport_calls"],
    );
    assert_eq!(
        cache.applied.borrow().len(),
        case["setup"]["mutations"].as_array().unwrap().len(),
        "unconsumed mutations"
    );
    failures
}

#[test]
fn hub_cases() {
    let document = frozen_cases();
    let cases = document["cases"].as_array().expect("cases array");
    assert_eq!(cases.len(), 24);
    let mut names = BTreeSet::new();
    let mut failures = Vec::new();
    for case in cases {
        let name = string(case, "name");
        assert!(names.insert(name), "duplicate case {name}");
        let result = std::panic::catch_unwind(|| run_case(&document["path_symbols"], case));
        let errors = match result {
            Ok(errors) => errors,
            Err(panic) => vec![panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "case panicked".into())],
        };
        if errors.is_empty() {
            eprintln!("PASS {name}");
        }
        failures.extend(errors.into_iter().map(|error| format!("{name}: {error}")));
    }
    assert!(
        failures.is_empty(),
        "{} cases executed; contract failures:\n{}",
        cases.len(),
        failures.join("\n")
    );
}

#[test]
fn client_contract() {
    let document = frozen_cases();
    let contract = &document["client_contract"];
    assert_eq!(
        contract,
        &json!({"anonymous":true,"authorization_header":null,"offline_client_constructions":0,"token_file_discovery_claim":false})
    );
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("token"), "hf_contract_test_token").unwrap();
    let api = anonymous_builder(&root.path().join("hub")).build().unwrap();
    let request = api
        .repo(Repo::with_revision(
            REPO.into(),
            RepoType::Model,
            "main".into(),
        ))
        .info_request();
    assert_eq!(
        json!(request.header("Authorization")),
        contract["authorization_header"]
    );
    assert_eq!(
        request.header("Authorization").is_none(),
        contract["anonymous"]
    );
    for case in document["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|case| case["request"]["offline"] == true)
    {
        let directory = tempfile::tempdir().unwrap();
        let cache = CaseCache::new(directory.path(), &document["path_symbols"], case);
        let request = &case["request"];
        let _ = resolve_with(
            string(request, "repo"),
            options(&cache.root, string(request, "revision"), true),
            |_| {
                cache.clients.set(cache.clients.get() + 1);
                Ok(&cache)
            },
        );
        assert_eq!(
            json!(cache.clients.get()),
            contract["offline_client_constructions"],
            "{}",
            case["name"]
        );
        assert!(cache.calls.borrow().is_empty(), "{}", case["name"]);
    }
}
