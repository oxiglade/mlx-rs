use mlx_lm::{HubError, HubOptions, LoadError, Model};
use sha2::{Digest, Sha256};
use std::{
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    sync::{Mutex, MutexGuard},
};

const REPO: &str = "org/model";
const SHA: &str = "0123456789abcdef0123456789abcdef01234567";
const INDEX: &str = "model.safetensors.index.json";
static ENVIRONMENT: Mutex<()> = Mutex::new(());

struct OfflineEnvironment {
    _lock: MutexGuard<'static, ()>,
    cache: tempfile::TempDir,
    previous: Vec<(&'static str, Option<OsString>)>,
}

impl OfflineEnvironment {
    fn enter() -> Self {
        let lock = ENVIRONMENT
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let cache = tempfile::tempdir().unwrap();
        let variables = [
            ("HF_HOME", cache.path().join("home").into_os_string()),
            ("HF_HUB_CACHE", cache.path().join("hub").into_os_string()),
            (
                "HUGGINGFACE_HUB_CACHE",
                cache.path().join("legacy").into_os_string(),
            ),
            (
                "TRANSFORMERS_CACHE",
                cache.path().join("transformers").into_os_string(),
            ),
            ("HF_HUB_OFFLINE", "1".into()),
            ("TRANSFORMERS_OFFLINE", "1".into()),
            ("HF_ENDPOINT", "http://127.0.0.1:9".into()),
        ];
        let previous = variables
            .into_iter()
            .map(|(name, value)| {
                let old = env::var_os(name);
                env::set_var(name, value);
                (name, old)
            })
            .collect();
        Self {
            _lock: lock,
            cache,
            previous,
        }
    }

    fn root(&self) -> PathBuf {
        self.cache.path().join("hub")
    }

    fn options(&self, revision: Option<&str>) -> HubOptions {
        let mut options = HubOptions::default();
        options.offline = true;
        options.revision = revision.map(String::from);
        options
    }

    fn snapshot(&self) -> PathBuf {
        self.root().join("models--org--model/snapshots").join(SHA)
    }

    fn receipt(&self, indexed: bool) {
        let snapshot = self.snapshot();
        let mut names: Vec<&str> = vec!["config.json", "tokenizer.json"];
        let mut absent = Vec::new();
        for name in ["generation_config.json", "tokenizer_config.json"] {
            if snapshot.join(name).exists() {
                names.push(name);
            } else {
                absent.push(name);
            }
        }
        if indexed {
            names.extend([INDEX, "weights.safetensors"]);
        } else {
            names.push("model.safetensors");
            absent.push(INDEX);
        }
        names.sort();
        absent.sort();
        let files: Vec<_> = names.into_iter().map(|name| {
            let bytes = fs::read(snapshot.join(name)).unwrap();
            serde_json::json!({"path":name,"size":bytes.len(),"sha256":format!("{:x}", Sha256::digest(bytes))})
        }).collect();
        let receipt = serde_json::json!({
            "schema_version":1,"repo":REPO,"commit":SHA,
            "weight_mode":if indexed { "indexed" } else { "single" },
            "selected_files":files,"absent_files":absent
        });
        let receipts = self.root().join("models--org--model/.mlx-lm/snapshots");
        fs::create_dir_all(&receipts).unwrap();
        fs::write(
            receipts.join(format!("{SHA}.json")),
            serde_json::to_vec(&receipt).unwrap(),
        )
        .unwrap();
    }

    fn populate(&self, indexed: bool) {
        let snapshot = self.snapshot();
        fs::create_dir_all(&snapshot).unwrap();
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../conformance/mlx-lm/fixtures/llama-base");
        for name in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
            "model.safetensors",
        ] {
            if fixture.join(name).exists() {
                fs::copy(fixture.join(name), snapshot.join(name)).unwrap();
            }
        }
        if indexed {
            let bytes = fs::read(snapshot.join("model.safetensors")).unwrap();
            let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
            let map: std::collections::BTreeMap<_, _> = tensors
                .names()
                .into_iter()
                .map(|name| (name, "weights.safetensors"))
                .collect();
            fs::write(
                snapshot.join(INDEX),
                serde_json::to_vec(&serde_json::json!({"weight_map":map})).unwrap(),
            )
            .unwrap();
            fs::rename(
                snapshot.join("model.safetensors"),
                snapshot.join("weights.safetensors"),
            )
            .unwrap();
        }
        self.receipt(indexed);
    }
}

impl Drop for OfflineEnvironment {
    fn drop(&mut self) {
        for (name, value) in self.previous.drain(..).rev() {
            match value {
                Some(value) => env::set_var(name, value),
                None => env::remove_var(name),
            }
        }
    }
}

#[test]
fn public_offline_unreceipted_cache_is_a_miss() {
    let environment = OfflineEnvironment::enter();
    let options = environment.options(None);
    assert!(matches!(
        Model::from_hub("org/model", options),
        Err(HubError::OfflineCacheMiss { repo, revision })
            if repo == "org/model" && revision == "main"
    ));
    environment.populate(false);
    fs::remove_dir_all(environment.root().join("models--org--model/.mlx-lm")).unwrap();
    assert!(matches!(
        Model::from_hub(REPO, environment.options(Some(SHA))),
        Err(HubError::OfflineCacheMiss { .. })
    ));
}

#[test]
fn public_offline_invalid_revision_and_repository_are_typed() {
    let environment = OfflineEnvironment::enter();
    assert!(
        matches!(Model::from_hub(REPO, environment.options(Some("../main"))), Err(HubError::InvalidRevision(revision)) if revision == "../main")
    );
    assert!(
        matches!(Model::from_hub("../model", environment.options(None)), Err(HubError::InvalidRepository(repo)) if repo == "../model")
    );
    assert!(!environment.root().exists());
}

#[test]
fn public_offline_preserves_local_load_error() {
    let environment = OfflineEnvironment::enter();
    let snapshot = environment.snapshot();
    fs::create_dir_all(&snapshot).unwrap();
    for name in ["config.json", "tokenizer.json", "model.safetensors"] {
        fs::write(snapshot.join(name), b"{").unwrap();
    }
    environment.receipt(false);
    let local = Model::from_dir(&snapshot).err().unwrap();
    let hub = Model::from_hub(REPO, environment.options(Some(SHA)))
        .err()
        .unwrap();
    assert!(matches!(&hub, HubError::Load(LoadError::Config(_))));
    match hub {
        HubError::Load(source) => assert_eq!(source.to_string(), local.to_string()),
        other => panic!("{other}"),
    }
}

#[test]
fn public_offline_rejects_changed_file_and_new_absent_sidecar() {
    let environment = OfflineEnvironment::enter();
    environment.populate(false);
    fs::write(environment.snapshot().join("model.safetensors"), b"changed").unwrap();
    assert!(matches!(
        Model::from_hub(REPO, environment.options(Some(SHA))),
        Err(HubError::SnapshotIntegrity { .. })
    ));
    environment.receipt(false);
    fs::write(environment.snapshot().join(INDEX), b"{}").unwrap();
    assert!(matches!(
        Model::from_hub(REPO, environment.options(Some(SHA))),
        Err(HubError::SnapshotIntegrity { .. })
    ));
}

#[test]
fn public_offline_single_snapshot_provenance() {
    let environment = OfflineEnvironment::enter();
    environment.populate(false);
    let model = Model::from_hub(REPO, environment.options(Some(SHA))).unwrap();
    let provenance = model.hub_provenance().unwrap();
    assert_eq!(provenance.repo, REPO);
    assert_eq!(provenance.requested_revision, SHA);
    assert_eq!(provenance.resolved_revision, SHA);
    assert!(Model::from_dir(environment.snapshot())
        .unwrap()
        .hub_provenance()
        .is_none());
}

#[test]
fn public_offline_indexed_snapshot_provenance() {
    let environment = OfflineEnvironment::enter();
    environment.populate(true);
    let refs = environment.root().join("models--org--model/refs");
    fs::create_dir_all(&refs).unwrap();
    fs::write(refs.join("main"), SHA).unwrap();
    let model = Model::from_hub(REPO, environment.options(None)).unwrap();
    let provenance = model.hub_provenance().unwrap();
    assert_eq!(provenance.requested_revision, "main");
    assert_eq!(provenance.resolved_revision, SHA);
}

#[cfg(unix)]
#[test]
fn public_offline_standard_blob_snapshot_provenance() {
    let environment = OfflineEnvironment::enter();
    environment.populate(false);
    let blobs = environment.root().join("models--org--model/blobs");
    fs::create_dir_all(&blobs).unwrap();
    for entry in fs::read_dir(environment.snapshot()).unwrap() {
        let entry = entry.unwrap();
        fs::rename(entry.path(), blobs.join(entry.file_name())).unwrap();
        std::os::unix::fs::symlink(
            Path::new("../../blobs").join(entry.file_name()),
            entry.path(),
        )
        .unwrap();
    }
    let model = Model::from_hub(REPO, environment.options(Some(SHA))).unwrap();
    assert_eq!(model.hub_provenance().unwrap().resolved_revision, SHA);
}

#[test]
fn public_gguf_provenance_is_none() {
    let _environment = OfflineEnvironment::enter();
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../conformance/mlx-lm/fixtures");
    let file = mlx_rs::io::GgufFile::load(fixtures.join("gguf-llama-f32/model.gguf")).unwrap();
    let tokenizer = mlx_lm::Tokenizer::from_dir(fixtures.join("llama-base")).unwrap();
    assert!(Model::from_gguf(file, tokenizer)
        .unwrap()
        .hub_provenance()
        .is_none());
}
