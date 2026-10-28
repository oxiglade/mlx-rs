use super::{is_commit, safe_filename, shard_names, HubError, INDEX, OPTIONAL, REQUIRED, SINGLE};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum WeightMode {
    Single,
    Indexed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Receipt {
    schema_version: u32,
    repo: String,
    commit: String,
    weight_mode: WeightMode,
    selected_files: Vec<FileIdentity>,
    absent_files: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileIdentity {
    path: String,
    size: u64,
    sha256: String,
}

pub(super) struct CacheLayout {
    root: PathBuf,
    repository: PathBuf,
    snapshot: PathBuf,
    repo: String,
    commit: String,
}

impl CacheLayout {
    fn new(root: PathBuf, repo: &str, commit: &str) -> Self {
        let repository = root.join(format!("models--{}", repo.replace('/', "--")));
        Self {
            snapshot: repository.join("snapshots").join(commit),
            root,
            repository,
            repo: repo.into(),
            commit: commit.into(),
        }
    }

    pub(super) fn online(root: &Path, repo: &str, commit: &str) -> Result<Self, HubError> {
        fs::create_dir_all(root)?;
        let layout = Self::new(fs::canonicalize(root)?, repo, commit);
        create_directories(&layout.root, &layout.snapshot)?;
        create_directories(&layout.root, &layout.repository.join(".mlx-lm/snapshots"))?;
        Ok(layout)
    }

    pub(super) fn offline(root: &Path, repo: &str, requested: &str) -> Result<Self, HubError> {
        let miss = || HubError::OfflineCacheMiss {
            repo: repo.into(),
            revision: requested.into(),
        };
        let root = fs::canonicalize(root).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                miss()
            } else {
                error.into()
            }
        })?;
        let mut layout = Self::new(root, repo, requested);
        if !is_commit(requested) {
            let path = layout.repository.join("refs").join(requested);
            let commit = (|| {
                regular_control(&layout.root, &path)?;
                Ok::<_, HubError>(fs::read_to_string(&path)?)
            })()
            .map_err(|error| match error {
                HubError::Io(source) if source.kind() == std::io::ErrorKind::NotFound => miss(),
                other => other,
            })?;
            if !is_commit(&commit) {
                return Err(HubError::InvalidRevision(commit));
            }
            layout = Self::new(layout.root, repo, &commit);
        }
        Ok(layout)
    }

    pub(super) fn repo(&self) -> &str {
        &self.repo
    }
    pub(super) fn commit(&self) -> &str {
        &self.commit
    }
    pub(super) fn snapshot(&self) -> &Path {
        &self.snapshot
    }

    fn receipt_path(&self) -> PathBuf {
        self.repository
            .join(".mlx-lm/snapshots")
            .join(format!("{}.json", self.commit))
    }

    pub(super) fn missing(&self, name: &str) -> HubError {
        HubError::MissingFile {
            repo: self.repo.clone(),
            revision: self.commit.clone(),
            filename: name.into(),
        }
    }

    pub(super) fn integrity(&self, reason: impl Into<String>) -> HubError {
        HubError::SnapshotIntegrity {
            path: self.receipt_path(),
            reason: reason.into(),
        }
    }

    pub(super) fn read_receipt_offline(&self, requested: &str) -> Result<Receipt, HubError> {
        self.read_receipt().map_err(|error| match error {
            HubError::Io(source) if source.kind() == std::io::ErrorKind::NotFound => {
                HubError::OfflineCacheMiss {
                    repo: self.repo.clone(),
                    revision: requested.into(),
                }
            }
            other => other,
        })
    }

    pub(super) fn read_receipt(&self) -> Result<Receipt, HubError> {
        let path = self.receipt_path();
        regular_control(&self.root, &path)?;
        serde_json::from_slice(&fs::read(path)?).map_err(|error| self.integrity(error.to_string()))
    }

    pub(super) fn write_receipt(&self, receipt: &Receipt) -> Result<(), HubError> {
        let bytes =
            serde_json::to_vec(receipt).map_err(|error| self.integrity(error.to_string()))?;
        atomic_write(&self.root, &self.receipt_path(), &bytes)
    }

    pub(super) fn write_ref(&self, revision: &str) -> Result<(), HubError> {
        atomic_write(
            &self.root,
            &self.repository.join("refs").join(revision),
            self.commit.as_bytes(),
        )
    }

    fn asset(&self, name: &str) -> Result<PathBuf, HubError> {
        if !safe_filename(name) {
            return Err(unsafe_path(Path::new(name)));
        }
        control_directories(&self.root, &self.snapshot)?;
        let requested = self.snapshot.join(name);
        let parent = resolve_inside(&self.snapshot, requested.parent().unwrap(), 0)?;
        let path = parent.join(requested.file_name().unwrap());
        let metadata = fs::symlink_metadata(&path)?;
        let resolved = if metadata.file_type().is_symlink() {
            let blobs = self.repository.join("blobs");
            control_directories(&self.root, &blobs)?;
            let target = normalize(
                &parent.join(fs::read_link(&path)?),
                &self.repository,
                &[&self.snapshot, &blobs],
            )?;
            resolve_inside(&blobs, &target, 0)?
        } else {
            path
        };
        if !fs::metadata(&resolved)?.is_file() {
            return Err(unsafe_path(&resolved));
        }
        if fs::canonicalize(&requested)? != resolved {
            return Err(unsafe_path(&requested));
        }
        if !fs::metadata(&requested)?.is_file() {
            return Err(unsafe_path(&requested));
        }
        Ok(resolved)
    }

    fn checked_asset(&self, name: &str) -> Result<PathBuf, HubError> {
        let path = self.snapshot.join(name);
        self.asset(name).map_err(|error| match error {
            HubError::Io(source) if source.kind() == std::io::ErrorKind::NotFound => {
                if fs::symlink_metadata(&path).is_ok_and(|metadata| metadata.is_symlink()) {
                    HubError::SnapshotIntegrity {
                        path,
                        reason: "broken asset link".into(),
                    }
                } else {
                    self.missing(name)
                }
            }
            HubError::UnsafePath { .. } => HubError::UnsafePath { path },
            other => other,
        })
    }

    pub(super) fn read_asset(&self, name: &str) -> Result<Vec<u8>, HubError> {
        Ok(fs::read(self.checked_asset(name)?)?)
    }

    pub(super) fn prepare_download(&self, name: &str) -> Result<(), HubError> {
        if !safe_filename(name) {
            return Err(unsafe_path(Path::new(name)));
        }
        let path = self.snapshot.join(name);
        control_directories(&self.root, &self.snapshot)?;
        let mut parent = self.snapshot.clone();
        for component in path
            .parent()
            .unwrap()
            .strip_prefix(&self.snapshot)
            .unwrap()
            .components()
        {
            parent.push(component);
            match fs::create_dir(&parent) {
                Ok(()) => (),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => (),
                Err(error) => return Err(error.into()),
            }
            parent = resolve_inside(&self.snapshot, &parent, 0)?;
            if !fs::metadata(&parent)?.is_dir() {
                return Err(unsafe_path(&parent));
            }
        }
        match fs::symlink_metadata(&path) {
            Ok(_) => {
                self.checked_asset(name)?;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
            Err(error) => return Err(error.into()),
        }
        Ok(())
    }

    pub(super) fn validate_download(&self, path: &Path, name: &str) -> Result<(), HubError> {
        let expected = self.snapshot.join(name);
        if path != expected {
            let snapshots = self.repository.join("snapshots");
            if let Ok(relative) = path.strip_prefix(&snapshots) {
                if let Some(Component::Normal(commit)) = relative.components().next() {
                    let commit = commit.to_string_lossy();
                    if is_commit(&commit) && commit != self.commit {
                        return Err(HubError::RevisionMismatch {
                            expected: self.commit.clone(),
                            actual: commit.into(),
                        });
                    }
                }
            }
            return Err(unsafe_path(path));
        }
        self.checked_asset(name)?;
        Ok(())
    }

    fn identity(&self, name: &str) -> Result<FileIdentity, HubError> {
        let mut file = File::open(self.checked_asset(name)?)?;
        let mut digest = Sha256::new();
        let mut size = 0u64;
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
            size += count as u64;
        }
        Ok(FileIdentity {
            path: name.into(),
            size,
            sha256: format!("{:x}", digest.finalize()),
        })
    }

    pub(super) fn complete(
        &self,
        selected: BTreeSet<String>,
        absent: BTreeSet<String>,
        weight_mode: WeightMode,
    ) -> Result<Receipt, HubError> {
        if weight_mode == WeightMode::Indexed {
            let mut expected: BTreeSet<_> = selected
                .iter()
                .filter(|name| !name.ends_with(".safetensors"))
                .cloned()
                .collect();
            expected.extend(shard_names(&self.read_asset(INDEX)?)?);
            if selected != expected {
                return Err(self.integrity("selected shards differ from the downloaded index"));
            }
        }
        let receipt = Receipt {
            schema_version: 1,
            repo: self.repo.clone(),
            commit: self.commit.clone(),
            weight_mode,
            selected_files: selected
                .iter()
                .map(|name| self.identity(name))
                .collect::<Result<_, _>>()?,
            absent_files: absent.into_iter().collect(),
        };
        self.validate(&receipt)?;
        Ok(receipt)
    }

    pub(super) fn validate(&self, receipt: &Receipt) -> Result<(), HubError> {
        if receipt.schema_version != 1 || receipt.repo != self.repo || receipt.commit != self.commit
        {
            return Err(self.integrity("receipt schema or snapshot identity differs"));
        }
        if !receipt
            .selected_files
            .windows(2)
            .all(|pair| pair[0].path < pair[1].path)
            || !receipt
                .absent_files
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        {
            return Err(self.integrity("receipt entries must be unique and sorted"));
        }
        let selected: BTreeSet<_> = receipt
            .selected_files
            .iter()
            .map(|file| file.path.clone())
            .collect();
        let mut expected: BTreeSet<String> = REQUIRED.into_iter().map(String::from).collect();
        let mut absent: BTreeSet<String> = BTreeSet::new();
        for name in OPTIONAL {
            if selected.contains(name) {
                expected.insert(name.into());
            } else {
                absent.insert(name.into());
            }
        }
        for file in &receipt.selected_files {
            let path = self.checked_asset(&file.path)?;
            if fs::metadata(path)?.len() != file.size
                || file.sha256.len() != 64
                || !file
                    .sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Err(HubError::SnapshotIntegrity {
                    path: self.snapshot.join(&file.path),
                    reason: "asset identity changed".into(),
                });
            }
        }
        match receipt.weight_mode {
            WeightMode::Single => {
                expected.insert(SINGLE.into());
                absent.insert(INDEX.into());
            }
            WeightMode::Indexed => {
                expected.insert(INDEX.into());
                let shards: BTreeSet<_> = selected
                    .iter()
                    .filter(|name| name.ends_with(".safetensors"))
                    .cloned()
                    .collect();
                if shards.is_empty() {
                    return Err(self.integrity("receipt contains no indexed shards"));
                }
                expected.extend(shards);
            }
        }
        if selected != expected || receipt.absent_files != absent.into_iter().collect::<Vec<_>>() {
            return Err(
                self.integrity("receipt does not describe the local loader's exact selection")
            );
        }
        for name in &receipt.absent_files {
            match fs::symlink_metadata(self.snapshot.join(name)) {
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
                Err(error) => return Err(error.into()),
                Ok(_) => {
                    return Err(HubError::SnapshotIntegrity {
                        path: self.snapshot.join(name),
                        reason: "declared absent asset appeared".into(),
                    })
                }
            }
        }
        Ok(())
    }
}

fn unsafe_path(path: &Path) -> HubError {
    HubError::UnsafePath {
        path: path.to_owned(),
    }
}

fn normalize(path: &Path, floor: &Path, allowed: &[&Path]) -> Result<PathBuf, HubError> {
    let mut result = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                // Collapsing a symlink/.. changes filesystem traversal semantics.
                if !fs::symlink_metadata(&result)?.file_type().is_dir() {
                    return Err(unsafe_path(path));
                }
                if !result.pop() {
                    return Err(unsafe_path(path));
                }
                if !result.starts_with(floor) {
                    return Err(unsafe_path(path));
                }
            }
            Component::CurDir => (),
            other => result.push(other.as_os_str()),
        }
        if !allowed
            .iter()
            .any(|root| root.starts_with(&result) || result.starts_with(root))
        {
            return Err(unsafe_path(path));
        }
    }
    Ok(result)
}

fn resolve_inside(root: &Path, path: &Path, depth: usize) -> Result<PathBuf, HubError> {
    if depth > 40 {
        return Err(unsafe_path(path));
    }
    let relative = path.strip_prefix(root).map_err(|_| unsafe_path(path))?;
    let mut current = root.to_owned();
    for component in relative.components() {
        if !matches!(component, Component::Normal(_)) {
            return Err(unsafe_path(path));
        }
        current.push(component);
        if fs::symlink_metadata(&current)?.file_type().is_symlink() {
            let target = normalize(
                &current.parent().unwrap().join(fs::read_link(&current)?),
                root,
                &[root],
            )?;
            current = resolve_inside(root, &target, depth + 1)?;
        }
    }
    Ok(current)
}

fn control_directories(root: &Path, path: &Path) -> Result<(), HubError> {
    let relative = path.strip_prefix(root).map_err(|_| unsafe_path(path))?;
    let mut current = root.to_owned();
    for component in relative.components() {
        if !matches!(component, Component::Normal(_)) {
            return Err(unsafe_path(path));
        }
        current.push(component);
        if !fs::symlink_metadata(&current)?.file_type().is_dir() {
            return Err(unsafe_path(&current));
        }
    }
    Ok(())
}

fn regular_control(root: &Path, path: &Path) -> Result<(), HubError> {
    control_directories(root, path.parent().ok_or_else(|| unsafe_path(path))?)?;
    if !fs::symlink_metadata(path)?.file_type().is_file() {
        return Err(unsafe_path(path));
    }
    Ok(())
}

fn create_directories(root: &Path, path: &Path) -> Result<(), HubError> {
    let relative = path.strip_prefix(root).map_err(|_| unsafe_path(path))?;
    let mut current = root.to_owned();
    for component in relative.components() {
        if !matches!(component, Component::Normal(_)) {
            return Err(unsafe_path(path));
        }
        current.push(component);
        match fs::create_dir(&current) {
            Ok(()) => (),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => (),
            Err(error) => return Err(error.into()),
        }
        if !fs::symlink_metadata(&current)?.file_type().is_dir() {
            return Err(unsafe_path(&current));
        }
    }
    Ok(())
}

fn atomic_write(root: &Path, path: &Path, bytes: &[u8]) -> Result<(), HubError> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let parent = path.parent().ok_or_else(|| unsafe_path(path))?;
    create_directories(root, parent)?;
    match fs::symlink_metadata(path) {
        Ok(metadata) if !metadata.file_type().is_file() => return Err(unsafe_path(path)),
        Err(error) if error.kind() != std::io::ErrorKind::NotFound => return Err(error.into()),
        _ => (),
    }
    let (temporary, mut file) = loop {
        let temporary = parent.join(format!(
            ".mlx-lm-{}-{}.tmp",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => break (temporary, file),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    };
    let result = (|| {
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
    result
}
