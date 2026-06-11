//! End-to-end smoke test for the quantized KV cache on a real model.
//!
//! Loads a checkpoint, generates once with the dense cache and once with a
//! `k8/v4` quantized cache, and asserts the quantized run (1) still produces
//! coherent text and (2) peaks at less GPU memory than the dense run.
//!
//! Ignored by default — needs a local checkpoint. Run with:
//!   MODEL=/path/to/model cargo test -p mlx-tests --test quant_cache_e2e -- --ignored --nocapture

use std::ops::ControlFlow;

use mlx_lm::cache::{CacheKind, CacheOptions};
use mlx_lm::{generate, load, GenerateParams, ModelContext, Sampler, UserInput};
use mlx_rs::memory::{peak_memory, reset_peak_memory};

const PROMPT: &str = "Explain in detail how a hydraulic press works, step by step.";
const MAX_TOKENS: i32 = 128;

/// Generate `MAX_TOKENS` greedily under `kind`, returning `(text, peak_bytes)`.
/// `generate` resets the cache per call, so the freshly-set options take
/// effect on the next run.
fn run(ctx: &mut ModelContext, kind: CacheKind) -> (String, usize) {
    let opts = CacheOptions {
        kind,
        ..Default::default()
    };
    ctx.model
        .set_cache_options(opts)
        .expect("set cache options");

    let params = GenerateParams {
        max_new_tokens: MAX_TOKENS,
        sampling: Sampler::Greedy,
        ..Default::default()
    };
    reset_peak_memory();
    let res = generate(ctx, UserInput::text(PROMPT), params, &mut |_id, _delta| {
        ControlFlow::Continue(())
    })
    .expect("generation succeeds");
    (res.text, peak_memory())
}

#[test]
#[ignore = "requires a local model checkpoint via MODEL env var"]
fn quantized_cache_generates_and_saves_memory() {
    let model_dir = std::env::var("MODEL").expect("set MODEL=/path/to/checkpoint");
    let mut ctx = load(&model_dir).expect("load model");

    let (dense_text, dense_peak) = run(&mut ctx, CacheKind::Dense);
    let (quant_text, quant_peak) = run(&mut ctx, CacheKind::quantized_k8_v4());

    println!("dense peak = {dense_peak} bytes");
    println!("k8v4  peak = {quant_peak} bytes");
    println!("dense text = {dense_text:?}");
    println!("k8v4  text = {quant_text:?}");

    assert!(
        !quant_text.trim().is_empty(),
        "quantized run produced empty text"
    );
    assert!(
        quant_peak < dense_peak,
        "quantized KV cache should peak below dense: quant={quant_peak} dense={dense_peak}"
    );
}
