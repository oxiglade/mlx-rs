//! Completion receipts prove snapshot completeness and recorded commit identity,
//! not tamper resistance: a writer who can modify the cache can modify the receipt.
//! Content hashes are recorded once at download completion as provenance. Reuse
//! checks receipt identity, paths, presence, and sizes without rereading assets.
//! Cache contents must remain unchanged while a model is loading.

use super::{HubOptions, HubProvenance, Model};
use crate::{weights::ShardIndex, HubError, LoadError};
use hf_hub::{api::sync::Api, Cache, Repo, RepoType};
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

mod snapshot;
use snapshot::{CacheLayout, Receipt, WeightMode};

const INDEX: &str = "model.safetensors.index.json";
const SINGLE: &str = "model.safetensors";
const REQUIRED: [&str; 2] = ["config.json", "tokenizer.json"];
const OPTIONAL: [&str; 2] = ["tokenizer_config.json", "generation_config.json"];

trait HubTransport {
    fn info(&self, repo: &str, revision: &str) -> Result<SnapshotInfo, HubError>;
    fn download(&self, repo: &str, commit: &str, filename: &str) -> Result<PathBuf, HubError>;
}

struct SnapshotInfo {
    commit: String,
    siblings: BTreeSet<String>,
}

struct OnlineTransport(Api);

impl HubTransport for OnlineTransport {
    fn info(&self, repo: &str, revision: &str) -> Result<SnapshotInfo, HubError> {
        let info = self
            .0
            .repo(Repo::with_revision(
                repo.into(),
                RepoType::Model,
                revision.into(),
            ))
            .info()?;
        Ok(SnapshotInfo {
            commit: info.sha,
            siblings: info
                .siblings
                .into_iter()
                .map(|entry| entry.rfilename)
                .collect(),
        })
    }

    fn download(&self, repo: &str, commit: &str, filename: &str) -> Result<PathBuf, HubError> {
        Ok(self
            .0
            .repo(Repo::with_revision(
                repo.into(),
                RepoType::Model,
                commit.into(),
            ))
            .download(filename)?)
    }
}

fn anonymous_builder(cache: &Path) -> hf_hub::api::sync::ApiBuilder {
    hf_hub::api::sync::ApiBuilder::from_cache(Cache::new(cache.to_owned()))
        .with_token(None)
        .with_progress(false)
}

pub(super) fn load(repo: &str, options: HubOptions) -> Result<Model, HubError> {
    load_with(repo, options, |cache| {
        Ok(OnlineTransport(anonymous_builder(cache).build()?))
    })
}

fn load_with<T: HubTransport>(
    repo: &str,
    options: HubOptions,
    online: impl FnOnce(&Path) -> Result<T, HubError>,
) -> Result<Model, HubError> {
    let resolved = resolve_with(repo, options, online)?;
    let mut model = Model::from_dir(resolved.layout.snapshot())?;
    resolved.validate()?;
    model.hub_provenance = Some(resolved.provenance);
    Ok(model)
}

struct ResolvedSnapshot {
    layout: CacheLayout,
    receipt: Receipt,
    provenance: HubProvenance,
}

impl ResolvedSnapshot {
    fn validate(&self) -> Result<(), HubError> {
        let current = self.layout.read_receipt()?;
        if current != self.receipt {
            return Err(self
                .layout
                .integrity("completion receipt changed during loading"));
        }
        self.layout.validate(&self.receipt)
    }
}

fn resolve_with<T: HubTransport>(
    repo: &str,
    options: HubOptions,
    online: impl FnOnce(&Path) -> Result<T, HubError>,
) -> Result<ResolvedSnapshot, HubError> {
    validate_repository(repo)?;
    let requested = options.revision.as_deref().unwrap_or("main");
    validate_revision(requested)?;
    let root = cache_root(options.cache_dir.as_deref())?;
    if options.offline {
        let layout = CacheLayout::offline(&root, repo, requested)?;
        let receipt = layout.read_receipt_offline(requested)?;
        layout.validate(&receipt)?;
        return Ok(ResolvedSnapshot {
            provenance: provenance(repo, requested, layout.commit()),
            layout,
            receipt,
        });
    }

    std::fs::create_dir_all(&root)?;
    let root = std::fs::canonicalize(root)?;
    let transport = online(&root)?;
    let info = transport.info(repo, requested)?;
    if !is_commit(&info.commit) {
        return Err(HubError::InvalidRevision(info.commit));
    }
    if is_commit(requested) && requested != info.commit {
        return Err(HubError::RevisionMismatch {
            expected: requested.into(),
            actual: info.commit,
        });
    }
    let layout = CacheLayout::online(&root, repo, &info.commit)?;
    let (selected, absent, mode) = select(&transport, &layout, &info.siblings)?;
    let receipt = layout.complete(selected, absent, mode)?;
    layout.write_receipt(&receipt)?;
    if !is_commit(requested) {
        layout.write_ref(requested)?;
    }
    Ok(ResolvedSnapshot {
        provenance: provenance(repo, requested, &info.commit),
        layout,
        receipt,
    })
}

fn provenance(repo: &str, requested: &str, commit: &str) -> HubProvenance {
    HubProvenance {
        repo: repo.into(),
        requested_revision: requested.into(),
        resolved_revision: commit.into(),
    }
}

fn select(
    transport: &impl HubTransport,
    layout: &CacheLayout,
    siblings: &BTreeSet<String>,
) -> Result<(BTreeSet<String>, BTreeSet<String>, WeightMode), HubError> {
    for name in REQUIRED {
        require(siblings, layout, name)?;
    }
    let mode = if siblings.contains(INDEX) {
        WeightMode::Indexed
    } else {
        WeightMode::Single
    };
    if mode == WeightMode::Single {
        require(siblings, layout, SINGLE)?;
    }
    let mut selected: BTreeSet<String> = REQUIRED.into_iter().map(String::from).collect();
    let mut absent = BTreeSet::new();
    for name in OPTIONAL {
        if siblings.contains(name) {
            selected.insert(name.into());
        } else {
            absent.insert(name.into());
        }
    }
    if mode == WeightMode::Indexed {
        download(transport, layout, INDEX)?;
        let index = layout.read_asset(INDEX)?;
        let shards = shard_names(&index)?;
        for name in &shards {
            require(siblings, layout, name)?;
        }
        selected.extend(shards);
        selected.insert(INDEX.into());
    } else {
        selected.insert(SINGLE.into());
        absent.insert(INDEX.into());
    }
    for name in &selected {
        if name != INDEX {
            download(transport, layout, name)?;
        }
    }
    Ok((selected, absent, mode))
}

fn download(
    transport: &impl HubTransport,
    layout: &CacheLayout,
    name: &str,
) -> Result<(), HubError> {
    layout.prepare_download(name)?;
    let path = transport.download(layout.repo(), layout.commit(), name)?;
    layout.validate_download(&path, name)
}

fn require(siblings: &BTreeSet<String>, layout: &CacheLayout, name: &str) -> Result<(), HubError> {
    if siblings.contains(name) {
        Ok(())
    } else {
        Err(layout.missing(name))
    }
}

fn shard_names(bytes: &[u8]) -> Result<BTreeSet<String>, HubError> {
    let index = ShardIndex::from_bytes(bytes).map_err(LoadError::from)?;
    index
        .shard_paths()
        .into_iter()
        .map(|path| {
            let name = path
                .to_str()
                .ok_or_else(|| HubError::UnsafePath { path: path.clone() })?;
            if !safe_filename(name) || !name.ends_with(".safetensors") {
                return Err(HubError::UnsafePath { path });
            }
            Ok(name.to_owned())
        })
        .collect()
}

fn safe_filename(name: &str) -> bool {
    name.split('/').all(|part| {
        !part.is_empty()
            && part != "."
            && part != ".."
            && part
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"_.-".contains(&byte))
    })
}

fn name_component(part: &str) -> bool {
    part.as_bytes()
        .first()
        .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
        && part
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_.-".contains(&byte))
}

fn validate_repository(repo: &str) -> Result<(), HubError> {
    if repo.split('/').count() <= 2 && repo.split('/').all(name_component) {
        Ok(())
    } else {
        Err(HubError::InvalidRepository(repo.into()))
    }
}

fn validate_revision(revision: &str) -> Result<(), HubError> {
    if !revision.contains("..")
        && revision
            .split('/')
            .all(|part| name_component(part) && !part.ends_with('.') && !part.ends_with(".lock"))
    {
        Ok(())
    } else {
        Err(HubError::InvalidRevision(revision.into()))
    }
}

fn is_commit(revision: &str) -> bool {
    revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn cache_root(explicit: Option<&Path>) -> Result<PathBuf, HubError> {
    fn variable(name: &str) -> Option<std::ffi::OsString> {
        std::env::var_os(name).filter(|value| !value.is_empty())
    }
    if let Some(path) = explicit {
        return Ok(path.to_owned());
    }
    if let Some(path) = variable("HF_HUB_CACHE") {
        return Ok(path.into());
    }
    if let Some(path) = variable("HF_HOME") {
        return Ok(PathBuf::from(path).join("hub"));
    }
    std::env::home_dir()
        .filter(|path| !path.as_os_str().is_empty())
        .map(|path| path.join(".cache/huggingface/hub"))
        .ok_or(HubError::CacheDirectoryUnavailable)
}

#[cfg(test)]
mod tests;
