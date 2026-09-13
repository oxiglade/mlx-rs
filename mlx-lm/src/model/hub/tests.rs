use super::*;
use std::{cell::RefCell, collections::BTreeMap, fs, rc::Rc};

const REPO: &str = "org/model";
const SHA: &str = "0123456789abcdef0123456789abcdef01234567";
const OTHER: &str = "abcdef0123456789abcdef0123456789abcdef01";

#[derive(Clone)]
struct Fake {
    root: PathBuf,
    commit: String,
    files: BTreeMap<String, Vec<u8>>,
    log: Rc<RefCell<Vec<String>>>,
    fail: Option<String>,
    returned_commit: Option<String>,
    move_ref: bool,
}

impl Fake {
    fn single(root: &Path) -> Self {
        Self {
            root: fs::canonicalize(root).unwrap(),
            commit: SHA.into(),
            files: [
                ("config.json", b"{}".to_vec()),
                ("tokenizer.json", b"{}".to_vec()),
                (SINGLE, vec![1, 2, 3]),
            ]
            .into_iter()
            .map(|(name, bytes)| (name.into(), bytes))
            .collect(),
            log: Rc::default(),
            fail: None,
            returned_commit: None,
            move_ref: false,
        }
    }

    fn run(&self, revision: &str, downloads: &[&str]) -> Result<ResolvedSnapshot, HubError> {
        let result = resolve_with(REPO, options(&self.root, revision, false), |_| {
            Ok(self.clone())
        });
        let expected = std::iter::once(format!("info:{REPO}@{revision}"))
            .chain(
                downloads
                    .iter()
                    .map(|name| format!("download:{REPO}@{}:{name}", self.commit)),
            )
            .collect::<Vec<_>>();
        assert_eq!(*self.log.borrow(), expected);
        self.log.borrow_mut().clear();
        result
    }
}

impl HubTransport for Fake {
    fn info(&self, repo: &str, revision: &str) -> Result<SnapshotInfo, HubError> {
        self.log
            .borrow_mut()
            .push(format!("info:{repo}@{revision}"));
        if self.fail.as_deref() == Some("info") {
            return Err(std::io::Error::other("info failed").into());
        }
        if self.move_ref {
            let refs = self.root.join("models--org--model/refs");
            fs::create_dir_all(&refs)?;
            fs::write(refs.join("main"), OTHER)?;
        }
        Ok(SnapshotInfo {
            commit: self.commit.clone(),
            siblings: self.files.keys().cloned().collect(),
        })
    }

    fn download(&self, repo: &str, commit: &str, filename: &str) -> Result<PathBuf, HubError> {
        self.log
            .borrow_mut()
            .push(format!("download:{repo}@{commit}:{filename}"));
        if self.fail.as_deref() == Some(filename) {
            return Err(std::io::Error::other("download failed").into());
        }
        let commit = self.returned_commit.as_deref().unwrap_or(commit);
        let path = self
            .root
            .join("models--org--model/snapshots")
            .join(commit)
            .join(filename);
        fs::create_dir_all(path.parent().unwrap())?;
        fs::write(&path, self.files.get(filename).expect("unlisted download"))?;
        Ok(path)
    }
}

fn options(root: &Path, revision: &str, offline: bool) -> HubOptions {
    HubOptions {
        cache_dir: Some(root.to_owned()),
        revision: Some(revision.into()),
        offline,
    }
}

struct ForbiddenTransport;
impl HubTransport for ForbiddenTransport {
    fn info(&self, _: &str, _: &str) -> Result<SnapshotInfo, HubError> {
        panic!("offline info call");
    }
    fn download(&self, _: &str, _: &str, _: &str) -> Result<PathBuf, HubError> {
        panic!("offline download call");
    }
}

fn offline(root: &Path, revision: &str) -> Result<ResolvedSnapshot, HubError> {
    resolve_with::<ForbiddenTransport>(REPO, options(root, revision, true), |_| {
        panic!("offline client construction")
    })
}

const SINGLE_CALLS: &[&str] = &["config.json", SINGLE, "tokenizer.json"];

#[test]
fn single_plan_pins_moving_ref_and_records_deliberate_absences() {
    let root = tempfile::tempdir().unwrap();
    let mut fake = Fake::single(root.path());
    fake.move_ref = true;
    fake.files.insert("remote.py".into(), vec![]);
    fake.files.insert("model.gguf".into(), vec![]);
    fake.files.insert("tokenizer.model".into(), vec![]);
    let resolved = fake.run("main", SINGLE_CALLS).unwrap();
    assert_eq!(resolved.provenance, provenance(REPO, "main", SHA));
    assert_eq!(
        fs::read_to_string(root.path().join("models--org--model/refs/main")).unwrap(),
        SHA
    );
    let receipt = serde_json::to_value(&resolved.receipt).unwrap();
    assert_eq!(receipt["weight_mode"], "single");
    assert_eq!(
        receipt["absent_files"],
        serde_json::json!(["generation_config.json", INDEX, "tokenizer_config.json"])
    );
    assert_eq!(receipt["selected_files"].as_array().unwrap().len(), 3);
    offline(root.path(), "main").unwrap().validate().unwrap();
    fs::remove_dir_all(root.path().join("models--org--model/refs")).unwrap();
    offline(root.path(), SHA).unwrap().validate().unwrap();
    assert!(fake.log.borrow().is_empty());
}

#[test]
fn indexed_plan_downloads_exact_distinct_shards_and_sidecars() {
    let root = tempfile::tempdir().unwrap();
    let mut fake = Fake::single(root.path());
    fake.files.insert(INDEX.into(), br#"{"weight_map":{"a":"parts/a.safetensors","b":"parts/z.safetensors","c":"parts/a.safetensors"}}"#.to_vec());
    for name in [
        "parts/a.safetensors",
        "parts/z.safetensors",
        "tokenizer_config.json",
        "generation_config.json",
    ] {
        fake.files.insert(name.into(), b"{}".to_vec());
    }
    let resolved = fake
        .run(
            "release/v1",
            &[
                INDEX,
                "config.json",
                "generation_config.json",
                "parts/a.safetensors",
                "parts/z.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
        )
        .unwrap();
    assert!(!resolved.layout.snapshot().join(SINGLE).exists());
    let receipt = serde_json::to_value(&resolved.receipt).unwrap();
    assert_eq!(receipt["weight_mode"], "indexed");
    assert_eq!(receipt["absent_files"], serde_json::json!([]));
    offline(root.path(), "release/v1").unwrap();
}

#[test]
fn invalid_names_never_construct_transport() {
    let root = tempfile::tempdir().unwrap();
    for repo in [
        "",
        "/abs",
        "a/b/c",
        "a//b",
        "https://host/a",
        "a\\b",
        "a b",
        "../a",
        "a/%2f",
        "a\n",
    ] {
        assert!(
            matches!(
                resolve_with::<ForbiddenTransport>(
                    repo,
                    options(root.path(), "main", false),
                    |_| panic!("invalid repository constructed client")
                ),
                Err(HubError::InvalidRepository(_))
            ),
            "{repo:?}"
        );
    }
    for revision in [
        "", "/main", "a//b", "a/../b", "a..b", "v1.", "a.lock", "a.lock/b", "a%20", "a?b", "a#b",
        "a\\b", "a b",
    ] {
        assert!(
            matches!(
                resolve_with::<ForbiddenTransport>(
                    REPO,
                    options(root.path(), revision, false),
                    |_| panic!("invalid revision constructed client")
                ),
                Err(HubError::InvalidRevision(_))
            ),
            "{revision:?}"
        );
    }
    for revision in [SHA, "123abc", "refs/pr/12", "_release/v1.2-rc"] {
        validate_revision(revision).unwrap();
    }
}

#[test]
fn invalid_or_different_server_commit_is_rejected_before_download() {
    let root = tempfile::tempdir().unwrap();
    let mut fake = Fake::single(root.path());
    fake.commit = "123abc".into();
    assert!(matches!(
        fake.run("main", &[]),
        Err(HubError::InvalidRevision(_))
    ));
    fake.commit = OTHER.into();
    assert!(
        matches!(fake.run(SHA, &[]), Err(HubError::RevisionMismatch { expected, actual }) if expected == SHA && actual == OTHER)
    );
    fake.commit = SHA.into();
    fake.returned_commit = Some(OTHER.into());
    assert!(
        matches!(fake.run("main", &["config.json"]), Err(HubError::RevisionMismatch { expected, actual }) if expected == SHA && actual == OTHER)
    );
    assert!(matches!(
        offline(root.path(), SHA),
        Err(HubError::OfflineCacheMiss { .. })
    ));
}

#[test]
fn missing_selected_assets_and_bad_indexes_stop_the_exact_plan() {
    let root = tempfile::tempdir().unwrap();
    for name in ["config.json", "tokenizer.json", SINGLE] {
        let mut fake = Fake::single(root.path());
        fake.files.remove(name);
        assert!(
            matches!(fake.run("main", &[]), Err(HubError::MissingFile { filename, .. }) if filename == name)
        );
    }
    for index in [
        r#"{"weight_map":{"a":"x.safetensors","a":"y.safetensors"}}"#,
        r#"{"weight_map":{"a":"../x.safetensors"}}"#,
        r#"{"weight_map":{"a":"/x.safetensors"}}"#,
        r#"{"weight_map":{"a":"x%20.safetensors"}}"#,
        r#"{"weight_map":{"a":"x\\y.safetensors"}}"#,
        r#"{"weight_map":{"a":"a//b.safetensors"}}"#,
        r#"{"weight_map":{"a":"a/./b.safetensors"}}"#,
        r#"{"weight_map":{"a":"model.gguf"}}"#,
    ] {
        let mut fake = Fake::single(root.path());
        fake.files.insert(INDEX.into(), index.as_bytes().to_vec());
        assert!(
            matches!(
                fake.run("main", &[INDEX]),
                Err(HubError::Load(_)) | Err(HubError::UnsafePath { .. })
            ),
            "{index}"
        );
    }
    let mut fake = Fake::single(root.path());
    fake.files.insert(
        INDEX.into(),
        br#"{"weight_map":{"a":"missing.safetensors"}}"#.to_vec(),
    );
    assert!(
        matches!(fake.run("main", &[INDEX]), Err(HubError::MissingFile { filename, .. }) if filename == "missing.safetensors")
    );
}

#[test]
fn failed_download_invalidates_completion_and_retry_recovers_without_fallback() {
    let root = tempfile::tempdir().unwrap();
    let mut fake = Fake::single(root.path());
    fake.run("main", SINGLE_CALLS).unwrap();
    fake.fail = Some(SINGLE.into());
    assert!(matches!(
        fake.run("main", &["config.json", SINGLE]),
        Err(HubError::Io(_))
    ));
    assert!(matches!(
        offline(root.path(), "main"),
        Err(HubError::OfflineCacheMiss { .. })
    ));
    fake.fail = None;
    fake.run("main", SINGLE_CALLS).unwrap();
    offline(root.path(), "main").unwrap();
    fake.fail = Some("info".into());
    assert!(matches!(fake.run("main", &[]), Err(HubError::Io(_))));
}

#[test]
fn offline_missing_ref_or_receipt_forbids_client_and_transport() {
    let root = tempfile::tempdir().unwrap();
    for revision in ["main", SHA] {
        assert!(
            matches!(offline(root.path(), revision), Err(HubError::OfflineCacheMiss { repo, revision: got }) if repo == REPO && got == revision)
        );
    }
    let snapshot = root.path().join("models--org--model/snapshots").join(SHA);
    fs::create_dir_all(&snapshot).unwrap();
    for name in ["config.json", "tokenizer.json", SINGLE] {
        fs::write(snapshot.join(name), b"{}").unwrap();
    }
    assert!(matches!(
        offline(root.path(), SHA),
        Err(HubError::OfflineCacheMiss { .. })
    ));
}

#[test]
fn receipt_rejects_changed_hash_missing_file_and_new_absence() {
    for mutation in [
        "hash",
        "missing",
        "generation_config.json",
        "tokenizer_config.json",
        INDEX,
    ] {
        let root = tempfile::tempdir().unwrap();
        let fake = Fake::single(root.path());
        let resolved = fake.run("main", SINGLE_CALLS).unwrap();
        match mutation {
            "hash" => fs::write(resolved.layout.snapshot().join(SINGLE), [3, 2, 1]).unwrap(),
            "missing" => fs::remove_file(resolved.layout.snapshot().join(SINGLE)).unwrap(),
            name => fs::write(resolved.layout.snapshot().join(name), b"{}").unwrap(),
        }
        let error = offline(root.path(), SHA).err().unwrap();
        if mutation == "missing" {
            assert!(matches!(error, HubError::MissingFile { .. }));
        } else {
            assert!(matches!(error, HubError::SnapshotIntegrity { .. }));
        }
        assert!(resolved.validate().is_err());
    }
}

#[test]
fn receipt_rejects_corruption_in_identity_schema_selection_and_absences() {
    for mutation in [
        "json",
        "schema",
        "repo",
        "commit",
        "duplicate",
        "order",
        "selection",
        "absence",
        "mode",
        "unknown",
    ] {
        let root = tempfile::tempdir().unwrap();
        let fake = Fake::single(root.path());
        let resolved = fake.run("main", SINGLE_CALLS).unwrap();
        let mut receipt = serde_json::to_value(&resolved.receipt).unwrap();
        match mutation {
            "schema" => receipt["schema_version"] = 2.into(),
            "repo" => receipt["repo"] = "other/model".into(),
            "commit" => receipt["commit"] = OTHER.into(),
            "duplicate" => {
                let item = receipt["selected_files"][0].clone();
                receipt["selected_files"]
                    .as_array_mut()
                    .unwrap()
                    .insert(0, item);
            }
            "order" => receipt["selected_files"].as_array_mut().unwrap().reverse(),
            "selection" => {
                receipt["selected_files"].as_array_mut().unwrap().remove(0);
            }
            "absence" => receipt["absent_files"] = serde_json::json!([]),
            "mode" => receipt["weight_mode"] = "indexed".into(),
            "unknown" => receipt["extra"] = true.into(),
            "json" => (),
            _ => unreachable!(),
        }
        let bytes = if mutation == "json" {
            b"{".to_vec()
        } else {
            serde_json::to_vec(&receipt).unwrap()
        };
        fs::write(
            root.path()
                .join(format!("models--org--model/.mlx-lm/snapshots/{SHA}.json")),
            bytes,
        )
        .unwrap();
        assert!(offline(root.path(), SHA).is_err(), "{mutation}");
        assert!(resolved.validate().is_err(), "{mutation}");
    }
}

#[cfg(unix)]
#[test]
fn standard_blob_links_work_but_repository_snapshot_and_ref_escapes_fail() {
    use std::os::unix::fs::symlink;
    for target in [
        "blob",
        "other_repo",
        "other_snapshot",
        "refs",
        "outside",
        "broken",
        "blob_chain_escape",
    ] {
        let root = tempfile::tempdir().unwrap();
        let fake = Fake::single(root.path());
        let resolved = fake.run("main", SINGLE_CALLS).unwrap();
        let repository = root.path().join("models--org--model");
        let path = resolved.layout.snapshot().join(SINGLE);
        let destination = match target {
            "blob" | "broken" | "blob_chain_escape" => repository.join("blobs/abc"),
            "other_repo" => root.path().join("models--other--model/blobs/abc"),
            "other_snapshot" => repository.join("snapshots").join(OTHER).join(SINGLE),
            "refs" => repository.join("refs/asset"),
            "outside" => root.path().join("outside"),
            _ => unreachable!(),
        };
        fs::create_dir_all(destination.parent().unwrap()).unwrap();
        if target != "broken" {
            fs::copy(&path, &destination).unwrap();
        }
        if target == "blob_chain_escape" {
            let outside = root.path().join("outside");
            fs::rename(&destination, &outside).unwrap();
            symlink(outside, &destination).unwrap();
        }
        fs::remove_file(&path).unwrap();
        let link = if target == "blob" {
            PathBuf::from("../../blobs/abc")
        } else {
            destination
        };
        symlink(link, &path).unwrap();
        if target == "blob" {
            offline(root.path(), SHA).unwrap();
        } else {
            assert!(
                matches!(
                    offline(root.path(), SHA),
                    Err(HubError::UnsafePath { .. }) | Err(HubError::MissingFile { .. })
                ),
                "{target}"
            );
        }
    }
}

#[cfg(unix)]
#[test]
fn directory_and_control_symlinks_cannot_escape() {
    use std::os::unix::fs::symlink;
    for relative in [
        format!("models--org--model/snapshots/{SHA}"),
        "models--org--model".into(),
        "models--org--model/.mlx-lm".into(),
        "models--org--model/refs".into(),
    ] {
        let root = tempfile::tempdir().unwrap();
        let fake = Fake::single(root.path());
        fake.run("main", SINGLE_CALLS).unwrap();
        let path = root.path().join(relative);
        let outside = tempfile::tempdir().unwrap();
        let moved = outside.path().join("moved");
        fs::rename(&path, &moved).unwrap();
        symlink(moved, &path).unwrap();
        assert!(matches!(
            offline(root.path(), "main"),
            Err(HubError::UnsafePath { .. })
        ));
    }
}

#[cfg(unix)]
#[test]
fn symlink_parent_traversal_cannot_validate_different_bytes_than_the_loader() {
    let root = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let fake = Fake::single(root.path());
    let resolved = fake.run("main", SINGLE_CALLS).unwrap();
    let blobs = fake.root.join("models--org--model/blobs");
    fs::create_dir_all(&blobs).unwrap();
    fs::create_dir(outside.path().join("child")).unwrap();
    fs::write(outside.path().join("weight"), b"external bytes").unwrap();
    let asset = resolved.layout.snapshot().join(SINGLE);
    fs::rename(&asset, blobs.join("weight")).unwrap();
    std::os::unix::fs::symlink(outside.path().join("child"), blobs.join("jump")).unwrap();
    std::os::unix::fs::symlink("../../blobs/jump/../weight", &asset).unwrap();
    assert_eq!(fs::read(&asset).unwrap(), b"external bytes");
    assert!(matches!(
        offline(root.path(), SHA),
        Err(HubError::UnsafePath { .. })
    ));
}

#[cfg(unix)]
#[test]
fn symlink_targets_must_be_openable_as_files_without_normalization() {
    for target in ["../../blobs/weight/", "../../blobs/weight/."] {
        let root = tempfile::tempdir().unwrap();
        let fake = Fake::single(root.path());
        let resolved = fake.run("main", SINGLE_CALLS).unwrap();
        let blobs = fake.root.join("models--org--model/blobs");
        fs::create_dir_all(&blobs).unwrap();
        let asset = resolved.layout.snapshot().join(SINGLE);
        fs::rename(&asset, blobs.join("weight")).unwrap();
        std::os::unix::fs::symlink(target, &asset).unwrap();
        assert!(fs::read(&asset).is_err());
        assert!(offline(root.path(), SHA).is_err(), "{target}");
    }
}

#[cfg(unix)]
#[test]
fn blob_links_cannot_visit_refs_or_another_snapshot_before_returning() {
    for target in [
        "../../refs/../blobs/weight".to_owned(),
        format!("../{OTHER}/../../blobs/weight"),
    ] {
        let root = tempfile::tempdir().unwrap();
        let fake = Fake::single(root.path());
        let resolved = fake.run("main", SINGLE_CALLS).unwrap();
        let repository = fake.root.join("models--org--model");
        fs::create_dir_all(repository.join("blobs")).unwrap();
        fs::create_dir_all(repository.join("snapshots").join(OTHER)).unwrap();
        let asset = resolved.layout.snapshot().join(SINGLE);
        fs::rename(&asset, repository.join("blobs/weight")).unwrap();
        std::os::unix::fs::symlink(target, &asset).unwrap();
        assert_eq!(fs::read(&asset).unwrap(), [1, 2, 3]);
        assert!(matches!(
            offline(root.path(), SHA),
            Err(HubError::UnsafePath { .. })
        ));
    }
}

#[test]
fn anonymous_client_has_no_authorization_header() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("token"), "hf_test_token").unwrap();
    let api = anonymous_builder(&root.path().join("hub")).build().unwrap();
    let request = api
        .repo(Repo::with_revision(
            REPO.into(),
            RepoType::Model,
            "main".into(),
        ))
        .info_request();
    assert_eq!(request.header("Authorization"), None);
}

#[cfg(unix)]
#[test]
fn nested_shard_directory_escape_is_rejected_before_transport_writes() {
    let root = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let mut fake = Fake::single(root.path());
    fake.files.insert(
        INDEX.into(),
        br#"{"weight_map":{"a":"parts/a.safetensors"}}"#.to_vec(),
    );
    fake.files.insert("parts/a.safetensors".into(), vec![1]);
    let snapshot = root.path().join("models--org--model/snapshots").join(SHA);
    fs::create_dir_all(&snapshot).unwrap();
    std::os::unix::fs::symlink(outside.path(), snapshot.join("parts")).unwrap();
    assert!(matches!(
        fake.run("main", &[INDEX, "config.json"]),
        Err(HubError::UnsafePath { .. })
    ));
    assert!(fs::read_dir(outside.path()).unwrap().next().is_none());
}

#[cfg(unix)]
#[test]
fn nested_directory_symlink_within_the_snapshot_is_allowed() {
    let root = tempfile::tempdir().unwrap();
    let mut fake = Fake::single(root.path());
    fake.files.insert(
        INDEX.into(),
        br#"{"weight_map":{"a":"parts/a.safetensors"}}"#.to_vec(),
    );
    fake.files.insert("parts/a.safetensors".into(), vec![1]);
    let snapshot = fake.root.join("models--org--model/snapshots").join(SHA);
    fs::create_dir_all(snapshot.join("actual")).unwrap();
    std::os::unix::fs::symlink("actual", snapshot.join("parts")).unwrap();
    fake.run(
        "main",
        &[
            INDEX,
            "config.json",
            "parts/a.safetensors",
            "tokenizer.json",
        ],
    )
    .unwrap();
    offline(root.path(), SHA).unwrap();
}

#[test]
fn downloaded_path_must_identify_the_expected_repository_and_filename() {
    let root = tempfile::tempdir().unwrap();
    let fake = Fake::single(root.path());
    let resolved = fake.run("main", SINGLE_CALLS).unwrap();
    for path in [
        root.path()
            .join("models--other--model/snapshots")
            .join(SHA)
            .join(SINGLE),
        resolved.layout.snapshot().join("tokenizer.json"),
        root.path().join("models--org--model/blobs/abc"),
    ] {
        assert!(matches!(
            resolved.layout.validate_download(&path, SINGLE),
            Err(HubError::UnsafePath { .. })
        ));
    }
}

#[test]
fn cache_environment_precedence_and_empty_values() {
    struct Restore(Vec<(&'static str, Option<std::ffi::OsString>)>);
    impl Drop for Restore {
        fn drop(&mut self) {
            for (name, value) in self.0.drain(..) {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }
    let root = tempfile::tempdir().unwrap();
    let _restore = Restore(
        ["HF_HUB_CACHE", "HF_HOME"]
            .into_iter()
            .map(|name| (name, std::env::var_os(name)))
            .collect(),
    );
    std::env::set_var("HF_HUB_CACHE", root.path().join("cache"));
    std::env::set_var("HF_HOME", root.path().join("home"));
    assert_eq!(cache_root(Some(root.path())).unwrap(), root.path());
    assert_eq!(cache_root(None).unwrap(), root.path().join("cache"));
    std::env::set_var("HF_HUB_CACHE", "");
    assert_eq!(cache_root(None).unwrap(), root.path().join("home/hub"));
    std::env::set_var("HF_HOME", "");
    assert_eq!(
        cache_root(None).unwrap(),
        std::env::home_dir().unwrap().join(".cache/huggingface/hub")
    );
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct LogicalPlan {
    name: String,
    requested_ref: String,
    returned_commit: String,
    siblings: Vec<String>,
    selected_files: Vec<String>,
    recorded_absences: Vec<String>,
    expected_transport_calls: Vec<String>,
    expected_provenance: Option<serde_json::Value>,
    expected_error: Option<String>,
    #[serde(default)]
    weight_map: BTreeMap<String, String>,
}

fn run_logical_plan(plan: LogicalPlan) {
    let root = tempfile::tempdir().unwrap();
    let mut fake = Fake::single(root.path());
    fake.commit = plan.returned_commit;
    fake.files = plan
        .siblings
        .into_iter()
        .map(|name| (name, b"{}".to_vec()))
        .collect();
    if fake.files.contains_key(INDEX) {
        fake.files.insert(
            INDEX.into(),
            serde_json::to_vec(&serde_json::json!({"weight_map":plan.weight_map})).unwrap(),
        );
    }
    let result = resolve_with(
        REPO,
        options(root.path(), &plan.requested_ref, false),
        |_| Ok(fake.clone()),
    );
    assert_eq!(
        *fake.log.borrow(),
        plan.expected_transport_calls,
        "{}",
        plan.name
    );
    match (result, plan.expected_error) {
        (Ok(resolved), None) => {
            let receipt = serde_json::to_value(resolved.receipt).unwrap();
            let selected: Vec<_> = receipt["selected_files"]
                .as_array()
                .unwrap()
                .iter()
                .map(|file| file["path"].as_str().unwrap())
                .collect();
            assert_eq!(selected, plan.selected_files, "{}", plan.name);
            assert_eq!(
                receipt["absent_files"],
                serde_json::to_value(plan.recorded_absences).unwrap(),
                "{}",
                plan.name
            );
            assert_eq!(
                plan.expected_provenance.unwrap(),
                serde_json::json!({"repo":resolved.provenance.repo,"requested_revision":resolved.provenance.requested_revision,"resolved_revision":resolved.provenance.resolved_revision}),
                "{}",
                plan.name
            );
        }
        (Err(error), Some(expected)) => assert_eq!(error_class(&error), expected, "{}", plan.name),
        _ => panic!("unexpected outcome for {}", plan.name),
    }
}

fn error_class(error: &HubError) -> &'static str {
    match error {
        HubError::InvalidRepository(_) => "InvalidRepository",
        HubError::InvalidRevision(_) => "InvalidRevision",
        HubError::MissingFile { .. } => "MissingFile",
        HubError::UnsafePath { .. } => "UnsafePath",
        HubError::RevisionMismatch { .. } => "RevisionMismatch",
        HubError::SnapshotIntegrity { .. } => "SnapshotIntegrity",
        HubError::OfflineCacheMiss { .. } => "OfflineCacheMiss",
        HubError::Load(_) => "Load",
        HubError::Io(_) => "Io",
        HubError::Api(_) => "Api",
        HubError::CacheDirectoryUnavailable => "CacheDirectoryUnavailable",
    }
}

#[test]
fn documented_logical_plan_schema() {
    run_logical_plan(serde_json::from_value(serde_json::json!({
        "name":"single", "requested_ref":"main", "returned_commit":SHA,
        "siblings":["config.json",SINGLE,"tokenizer.json","remote.py"],
        "selected_files":["config.json",SINGLE,"tokenizer.json"],
        "recorded_absences":["generation_config.json",INDEX,"tokenizer_config.json"],
        "expected_transport_calls":[format!("info:{REPO}@main"),format!("download:{REPO}@{SHA}:config.json"),format!("download:{REPO}@{SHA}:{SINGLE}"),format!("download:{REPO}@{SHA}:tokenizer.json")],
        "expected_provenance":{"repo":REPO,"requested_revision":"main","resolved_revision":SHA},
        "expected_error":null
    })).unwrap());
}

#[test]
#[ignore = "NOT RUN: conformance/mlx-lm/hub_cases.json is absent from this worktree"]
fn hub_cases_snapshot_plans() {
    let cases = frozen_cases();
    let plans = cases["plans"].as_array().expect("plans array");
    assert!(!plans.is_empty());
    for plan in plans {
        run_logical_plan(serde_json::from_value(plan.clone()).unwrap());
    }
}

#[test]
#[ignore = "NOT RUN: conformance/mlx-lm/hub_cases.json is absent from this worktree"]
fn hub_cases_mutations() {
    let cases = frozen_cases();
    let mutations = cases["mutations"].as_array().expect("mutations array");
    assert!(!mutations.is_empty());
    for mutation in mutations {
        let name = mutation["name"].as_str().expect("mutation name");
        match name {
            "ref_changes" => single_plan_pins_moving_ref_and_records_deliberate_absences(),
            "server_commit_changes" => {
                invalid_or_different_server_commit_is_rejected_before_download()
            }
            "duplicate_or_traversing_index" => {
                missing_selected_assets_and_bad_indexes_stop_the_exact_plan()
            }
            "absent_sidecar_appears" | "hash_changes" | "missing_shard" => {
                receipt_rejects_changed_hash_missing_file_and_new_absence()
            }
            "unreceipted_cache" | "offline_transport" => {
                offline_missing_ref_or_receipt_forbids_client_and_transport()
            }
            "sha_without_ref" => single_plan_pins_moving_ref_and_records_deliberate_absences(),
            "online_failure_fallback" => {
                failed_download_invalidates_completion_and_retry_recovers_without_fallback()
            }
            #[cfg(unix)]
            "wrong_repository_blob" | "broken_shard_link" => {
                standard_blob_links_work_but_repository_snapshot_and_ref_escapes_fail()
            }
            #[cfg(unix)]
            "snapshot_directory_escape" => directory_and_control_symlinks_cannot_escape(),
            _ => panic!("unconsumed hub mutation: {name}"),
        }
    }
}

fn frozen_cases() -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../conformance/mlx-lm/hub_cases.json");
    let cases: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    assert_eq!(cases["schema_version"], 1);
    cases
}
