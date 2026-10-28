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
fn nested_index_shards_are_deduplicated_for_a_slash_ref() {
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
fn invalid_server_commit_is_rejected_before_download() {
    let root = tempfile::tempdir().unwrap();
    let mut fake = Fake::single(root.path());
    fake.commit = "123abc".into();
    assert!(matches!(
        fake.run("main", &[]),
        Err(HubError::InvalidRevision(_))
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
fn failed_download_preserves_completion_and_retry_recovers() {
    let root = tempfile::tempdir().unwrap();
    let mut fake = Fake::single(root.path());
    let resolved = fake.run("main", SINGLE_CALLS).unwrap();
    let receipt_path = root
        .path()
        .join(format!("models--org--model/.mlx-lm/snapshots/{SHA}.json"));
    let receipt = fs::read(&receipt_path).unwrap();
    for (failure, downloads) in [
        ("info", &SINGLE_CALLS[..0]),
        ("config.json", &SINGLE_CALLS[..1]),
        (SINGLE, &SINGLE_CALLS[..2]),
        ("tokenizer.json", SINGLE_CALLS),
    ] {
        fake.fail = Some(failure.into());
        assert!(matches!(fake.run("main", downloads), Err(HubError::Io(_))));
        assert_eq!(fs::read(&receipt_path).unwrap(), receipt);
        for (name, bytes) in &fake.files {
            assert_eq!(
                fs::read(resolved.layout.snapshot().join(name)).unwrap(),
                *bytes
            );
        }
        assert_eq!(
            offline(root.path(), "main").unwrap().receipt,
            resolved.receipt
        );
        resolved.validate().unwrap();
    }
    fake.fail = None;
    fake.run("main", SINGLE_CALLS).unwrap();
    offline(root.path(), "main").unwrap();
}

#[test]
fn receipt_rejects_changed_size_on_offline_and_reuse_paths() {
    let root = tempfile::tempdir().unwrap();
    let fake = Fake::single(root.path());
    let resolved = fake.run("main", SINGLE_CALLS).unwrap();
    fs::write(resolved.layout.snapshot().join(SINGLE), [3, 2]).unwrap();
    assert!(matches!(
        offline(root.path(), SHA),
        Err(HubError::SnapshotIntegrity { .. })
    ));
    assert!(resolved.validate().is_err());
}

#[test]
fn receipt_records_hash_once_and_rejects_changed_recorded_identity() {
    let root = tempfile::tempdir().unwrap();
    let fake = Fake::single(root.path());
    let resolved = fake.run("main", SINGLE_CALLS).unwrap();
    let mut receipt = serde_json::to_value(&resolved.receipt).unwrap();
    assert_eq!(
        receipt["selected_files"][1]["sha256"],
        "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81"
    );
    fs::write(resolved.layout.snapshot().join(SINGLE), [3, 2, 1]).unwrap();
    resolved.validate().unwrap();
    let reused = offline(root.path(), SHA).unwrap();
    assert_eq!(reused.receipt, resolved.receipt);
    reused.validate().unwrap();

    receipt["selected_files"][1]["sha256"] = "0".repeat(64).into();
    resolved
        .layout
        .write_receipt(&serde_json::from_value(receipt).unwrap())
        .unwrap();
    assert!(matches!(
        resolved.validate(),
        Err(HubError::SnapshotIntegrity { .. })
    ));
    assert!(matches!(
        reused.validate(),
        Err(HubError::SnapshotIntegrity { .. })
    ));
}

#[cfg(unix)]
#[test]
fn offline_and_reuse_validation_do_not_read_selected_file_contents() {
    use std::os::unix::fs::PermissionsExt;

    for indexed in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let mut fake = Fake::single(root.path());
        if indexed {
            fake.files.insert(
                INDEX.into(),
                br#"{"weight_map":{"a":"model.safetensors"}}"#.to_vec(),
            );
        }
        let calls = if indexed {
            &[INDEX, "config.json", SINGLE, "tokenizer.json"][..]
        } else {
            SINGLE_CALLS
        };
        let resolved = fake.run("main", calls).unwrap();
        for name in fake.files.keys() {
            let path = resolved.layout.snapshot().join(name);
            fs::set_permissions(&path, fs::Permissions::from_mode(0o000)).unwrap();
            assert_eq!(
                fs::read(path).unwrap_err().kind(),
                std::io::ErrorKind::PermissionDenied
            );
        }
        resolved.validate().unwrap();
        offline(root.path(), "main").unwrap().validate().unwrap();
        offline(root.path(), SHA).unwrap().validate().unwrap();
    }
}

#[test]
fn receipt_rejects_corruption_in_identity_schema_selection_and_absences() {
    for mutation in [
        "json",
        "schema",
        "repo",
        "commit",
        "hash_format",
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
            "hash_format" => receipt["selected_files"][0]["sha256"] = "invalid".into(),
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
fn blob_links_cannot_escape_via_refs_or_chained_links() {
    use std::os::unix::fs::symlink;
    for target in ["refs", "outside", "blob_chain_escape"] {
        let root = tempfile::tempdir().unwrap();
        let fake = Fake::single(root.path());
        let resolved = fake.run("main", SINGLE_CALLS).unwrap();
        let repository = root.path().join("models--org--model");
        fs::create_dir_all(repository.join("blobs")).unwrap();
        let path = resolved.layout.snapshot().join(SINGLE);
        let destination = match target {
            "blob_chain_escape" => repository.join("blobs/abc"),
            "refs" => repository.join("refs/asset"),
            "outside" => root.path().join("outside"),
            _ => unreachable!(),
        };
        fs::create_dir_all(destination.parent().unwrap()).unwrap();
        fs::copy(&path, &destination).unwrap();
        if target == "blob_chain_escape" {
            let outside = root.path().join("outside");
            fs::rename(&destination, &outside).unwrap();
            symlink(outside, &destination).unwrap();
        }
        fs::remove_file(&path).unwrap();
        symlink(destination, &path).unwrap();
        assert!(
            matches!(offline(root.path(), SHA), Err(HubError::UnsafePath { .. })),
            "{target}"
        );
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

mod contract;
