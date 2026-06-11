//! Top-level conversion driver.
//!
//! Reads source shards one at a time (memory-cap), applies the
//! [`Rewriter`] rules, quantises, accumulates the destination tensor
//! list in memory, then writes sharded output + index + an updated
//! `config.json` carrying the `quantization` field that `mlx_lm::load`
//! expects.

use std::path::{Path, PathBuf};

use mlx_rs::transforms::eval;
use mlx_rs::Array;
use serde_json::Value;

use crate::plan::Rewriter;
use crate::quantize::{classify_and_quantize, OutTensor};
use crate::shards::{write_index, write_shards};
use crate::{anyhow, Result};

/// User-visible conversion knobs.
pub struct ConvertOptions {
    pub src: PathBuf,
    pub dst: PathBuf,
    pub body_bits: i32,
    pub body_group_size: i32,
}

/// Summary returned to the CLI.
pub struct ConvertReport {
    pub tensors_in: usize,
    pub tensors_out: usize,
    pub shards_in: usize,
    pub bytes_out: u64,
}

/// Drive the full convert.
pub fn convert(opts: &ConvertOptions, rewriter: &dyn Rewriter) -> Result<ConvertReport> {
    log::info!(
        "convert: {} → {} ({}-bit, gs={}, rewriter={})",
        opts.src.display(),
        opts.dst.display(),
        opts.body_bits,
        opts.body_group_size,
        rewriter.name()
    );

    std::fs::create_dir_all(&opts.dst)?;

    let shards = list_source_shards(&opts.src)?;
    log::info!("found {} source shard(s)", shards.len());

    let mut out: Vec<OutTensor> = Vec::new();
    let mut tensors_in: usize = 0;
    for (i, shard) in shards.iter().enumerate() {
        log::info!(
            "loading shard {}/{}: {}",
            i + 1,
            shards.len(),
            shard.display()
        );
        let loaded = Array::load_safetensors(shard)?;
        tensors_in += loaded.len();
        for (k, v) in loaded {
            if rewriter.skip_source(&k) {
                continue;
            }
            for (dst_key, tensor, class) in rewriter.rewrite(&k, v)? {
                let entries = classify_and_quantize(
                    dst_key,
                    tensor,
                    class,
                    opts.body_bits,
                    opts.body_group_size,
                )?;
                out.extend(entries);
            }
        }
        // Force eval so the previous shard's input arrays can be freed
        // before we load the next — otherwise the converter peaks at
        // ~2× model size.
        eval(out.iter().map(|t| &t.array))?;
    }
    log::info!(
        "rewrite + quantise complete: {} → {} tensors",
        tensors_in,
        out.len()
    );

    // Pre-sort for deterministic shard contents.
    out.sort_by(|a, b| a.key.cmp(&b.key));

    let total_size: u64 = out.iter().map(|t| t.array.nbytes() as u64).sum();
    let tensors_out = out.len();
    let weight_map = write_shards(&opts.dst, out)?;
    write_index(&opts.dst, weight_map, total_size)?;

    write_converted_config(&opts.src, &opts.dst, opts.body_bits, opts.body_group_size)?;
    copy_tokenizer_assets(&opts.src, &opts.dst)?;

    Ok(ConvertReport {
        tensors_in,
        tensors_out,
        shards_in: shards.len(),
        bytes_out: total_size,
    })
}

/// Enumerate source `model-*.safetensors` shards. Returns just the file
/// paths (not the index JSON or any extras), sorted lexicographically so
/// shard order matches the original.
fn list_source_shards(src: &Path) -> Result<Vec<PathBuf>> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(src)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|s| s.to_str()) == Some("safetensors")
                && p.file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| s.starts_with("model-") || s == "model.safetensors")
        })
        .collect();
    out.sort();
    if out.is_empty() {
        return Err(anyhow!("no model-*.safetensors found in {}", src.display()));
    }
    Ok(out)
}

/// Copy + edit `config.json` to record the quantisation. `mlx_lm::load`
/// reads the top-level `bits`/`group_size`. Per-tensor overrides are
/// not yet wired in our loader (the kept set doesn't need them), so we
/// emit only the global block.
fn write_converted_config(src: &Path, dst: &Path, bits: i32, group_size: i32) -> Result<()> {
    let src_path = src.join("config.json");
    let text = std::fs::read_to_string(&src_path)?;
    let mut value: Value = serde_json::from_str(&text)?;
    let obj = value
        .as_object_mut()
        .ok_or_else(|| anyhow!("config.json at {} is not a JSON object", src_path.display()))?;

    let mut quant = serde_json::Map::new();
    quant.insert("group_size".to_owned(), Value::from(group_size));
    quant.insert("bits".to_owned(), Value::from(bits));
    quant.insert("mode".to_owned(), Value::from("affine"));
    obj.insert("quantization".to_owned(), Value::Object(quant.clone()));
    obj.insert("quantization_config".to_owned(), Value::Object(quant));

    let dst_path = dst.join("config.json");
    let pretty = serde_json::to_string_pretty(&value)?;
    std::fs::write(&dst_path, pretty)?;
    log::info!("wrote {}", dst_path.display());
    Ok(())
}

/// Copy tokenizer + chat template files alongside the shards so the
/// converted dir is a self-contained checkpoint.
fn copy_tokenizer_assets(src: &Path, dst: &Path) -> Result<()> {
    const FILES: &[&str] = &[
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "merges.txt",
        "vocab.json",
        "generation_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "special_tokens_map.json",
    ];
    let mut copied = 0;
    for name in FILES {
        let s = src.join(name);
        if s.is_file() {
            std::fs::copy(&s, dst.join(name))?;
            copied += 1;
        }
    }
    log::info!("copied {copied} tokenizer asset(s)");
    Ok(())
}
