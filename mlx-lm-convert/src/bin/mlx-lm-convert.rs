//! Convert a bf16/fp16 safetensors checkpoint to a quantised one that
//! `mlx_lm::load` (Rust) can read.
//!
//! Drives [`mlx_lm_convert::convert`] with the right per-family
//! [`mlx_lm_convert::Rewriter`]. Supports Qwen 3.5 / 3.6 (dense + MoE,
//! including MTP) and Gemma 4 Unified (gemma-4-12B, encoder-free MM).
//!
//! Usage:
//!     mlx-lm-convert --src /Volumes/backup/full-models/Qwen/Qwen3.6-35B-A3B \
//!                    --dst ~/.cache/mlx-rs-bench/mlx-community/Qwen3.6-35B-A3B-q8-mtp \
//!                    [--bits 8] [--group-size 64] [--verify]

#![allow(clippy::print_stderr, reason = "CLI binary logs to stderr")]
#![allow(clippy::print_stdout, reason = "CLI binary prints to stdout")]

use std::path::PathBuf;
use std::time::Instant;

use argh::FromArgs;
use mlx_lm::{generate, load, GenerateParams, UserInput};
use mlx_lm_convert::{
    convert, gemma4_unified::Gemma4UnifiedRewriter, qwen3_5::Qwen35Rewriter, ConvertOptions,
};

/// Quantise a bf16 safetensors checkpoint into mlx-rs-loadable form.
#[derive(FromArgs)]
struct Args {
    /// source checkpoint directory (bf16 safetensors + config.json)
    #[argh(option)]
    src: PathBuf,

    /// destination directory (created if missing)
    #[argh(option)]
    dst: PathBuf,

    /// body quantisation bits (default 8)
    #[argh(option, default = "8")]
    bits: i32,

    /// quantisation group size (default 64)
    #[argh(option, default = "64")]
    group_size: i32,

    /// model family: `qwen3_5` (qwen 3.5/3.6 dense + MoE, incl. MTP) or
    /// `gemma4_unified` (gemma-4-12B, encoder-free MM).
    #[argh(option, default = "String::from(\"qwen3_5\")")]
    family: String,

    /// after convert, load the destination via `mlx_lm::load` and run a
    /// 4-token greedy generation — catches unbound keys / shape errors
    /// before the user hits them in real use.
    #[argh(switch)]
    verify: bool,
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args: Args = argh::from_env();
    let opts = ConvertOptions {
        src: args.src,
        dst: args.dst,
        body_bits: args.bits,
        body_group_size: args.group_size,
    };

    let t = Instant::now();
    let report = match args.family.as_str() {
        "qwen3_5" | "qwen3_6" | "qwen3_5_moe" | "qwen3_6_moe" => {
            convert(&opts, &Qwen35Rewriter::default())?
        }
        "gemma4_unified" => convert(&opts, &Gemma4UnifiedRewriter::default())?,
        other => anyhow::bail!("unsupported family {other:?}; supported: qwen3_5, gemma4_unified"),
    };
    let dt = t.elapsed();

    log::info!(
        "done in {:.1}s: {} → {} tensors across {} input shard(s), {:.2} GB out",
        dt.as_secs_f64(),
        report.tensors_in,
        report.tensors_out,
        report.shards_in,
        report.bytes_out as f64 / 1e9
    );

    if args.verify {
        verify_load_and_generate(&opts.dst)?;
    }
    Ok(())
}

/// Round-trip the destination dir through `mlx_lm::load` and run a
/// 4-token greedy generation. Errors propagate — a verify failure
/// non-zero-exits the process so CI catches a bad conversion.
fn verify_load_and_generate(dst: &std::path::Path) -> anyhow::Result<()> {
    log::info!("verify: loading {} via mlx_lm::load", dst.display());
    let t = Instant::now();
    let mut ctx = load(dst)?;
    log::info!("verify: load OK in {:.1}s", t.elapsed().as_secs_f64());

    let input = UserInput::text("Hello");
    let params = GenerateParams {
        max_new_tokens: 4,
        ..GenerateParams::default()
    };
    let t = Instant::now();
    let result = generate(&mut ctx, input, params, &mut |_, _| {
        std::ops::ControlFlow::Continue(())
    })?;
    if result.completion_tokens <= 0 {
        anyhow::bail!("verify: generate produced zero tokens");
    }
    log::info!(
        "verify: generated {} token(s) in {:.1}s, text={:?}",
        result.completion_tokens,
        t.elapsed().as_secs_f64(),
        result.text
    );
    Ok(())
}
