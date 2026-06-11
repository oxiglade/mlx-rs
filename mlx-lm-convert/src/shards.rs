//! Sharded safetensors writer + `model.safetensors.index.json` builder.
//!
//! Matches the convention used by `mlx-community/*` checkpoints: pack
//! tensors into shards up to a configurable target size, emit an index
//! file mapping every key to its shard filename.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;
use serde::Serialize;

use crate::{quantize::OutTensor, Result};

/// Target bytes per shard. The 35B-A3B q8 reference uses ~5 GB. We use
/// the same — any larger and clients with constrained inode caches start
/// paging the index.
pub const SHARD_TARGET_BYTES: u64 = 5_000_000_000;

/// Pack `tensors` into shards up to `SHARD_TARGET_BYTES` each, write the
/// safetensors files to `dst`, and return the `weight_map` for the index.
/// Tensors are written in insertion order — caller can pre-sort for
/// deterministic shard contents.
pub fn write_shards(dst: &Path, tensors: Vec<OutTensor>) -> Result<HashMap<String, String>> {
    let mut weight_map: HashMap<String, String> = HashMap::new();
    let mut shard_idx: usize = 0;
    let mut shard_bytes: u64 = 0;
    let mut buffer: Vec<OutTensor> = Vec::new();
    let total_tensors = tensors.len();

    // First pass: estimate total shard count so we know the
    // `model-NNNNN-of-MMMMM.safetensors` denominator. Without the total
    // up front we'd have to rename files after the fact.
    let mut planned_shards: usize = 1;
    let mut running: u64 = 0;
    for t in &tensors {
        let n = t.array.nbytes() as u64;
        if running > 0 && running + n > SHARD_TARGET_BYTES {
            planned_shards += 1;
            running = 0;
        }
        running += n;
    }

    let shard_name = |i: usize| format!("model-{:05}-of-{:05}.safetensors", i + 1, planned_shards);

    for t in tensors {
        let n = t.array.nbytes() as u64;
        if !buffer.is_empty() && shard_bytes + n > SHARD_TARGET_BYTES {
            flush_shard(dst, &shard_name(shard_idx), &mut buffer, &mut weight_map)?;
            shard_idx += 1;
            shard_bytes = 0;
        }
        shard_bytes += n;
        buffer.push(t);
    }
    if !buffer.is_empty() {
        flush_shard(dst, &shard_name(shard_idx), &mut buffer, &mut weight_map)?;
    }
    log::info!("wrote {total_tensors} tensors across {planned_shards} shard(s)");
    Ok(weight_map)
}

fn flush_shard(
    dst: &Path,
    name: &str,
    buf: &mut Vec<OutTensor>,
    weight_map: &mut HashMap<String, String>,
) -> Result<()> {
    let path = dst.join(name);
    let mut total = 0_u64;
    let entries: Vec<(String, Array)> = buf
        .drain(..)
        .map(|t| {
            total += t.array.nbytes() as u64;
            weight_map.insert(t.key.clone(), name.to_owned());
            (t.key, t.array)
        })
        .collect();
    Array::save_safetensors(entries.iter().map(|(k, v)| (k.as_str(), v)), None, &path)?;
    log::info!(
        "  {name}: {:.2} GB ({} tensors)",
        total as f64 / 1e9,
        entries.len()
    );
    Ok(())
}

/// `model.safetensors.index.json` envelope. Mirrors the HF / mlx-community
/// format consumed by `mlx_lm::load`.
#[derive(Serialize)]
pub struct Index {
    pub metadata: IndexMetadata,
    pub weight_map: HashMap<String, String>,
}

#[derive(Serialize)]
pub struct IndexMetadata {
    pub total_size: u64,
}

/// Write the index JSON for a set of shards. `total_size` is the sum
/// of every tensor's on-disk byte count (not the file sizes — those
/// include the safetensors header overhead, which the index excludes).
pub fn write_index(dst: &Path, weight_map: HashMap<String, String>, total_size: u64) -> Result<()> {
    let idx = Index {
        metadata: IndexMetadata { total_size },
        weight_map,
    };
    let path = dst.join("model.safetensors.index.json");
    let text = serde_json::to_string_pretty(&idx)?;
    std::fs::write(&path, text)?;
    log::info!("wrote {}", path.display());
    Ok(())
}
