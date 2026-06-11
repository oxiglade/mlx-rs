//! Decode-throughput bench for llama + qwen3 + qwen3.5 (GDN-hybrid dense
//! + MoE, with an MTP-on/off A/B cell). See `BENCHMARK.md`.

use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mlx_lm::lm_input::{LMInput, Text};
use mlx_lm::sampler::{Sampler, SamplerState};
use mlx_lm::{decode_step, generate, load, GenerateParams, ModelContext, UserInput};
use mlx_rs::{
    ops::indexing::{IndexOp, NewAxis},
    transforms::eval,
    Array,
};

const DECODE_TOKENS: i32 = 100;
const LONG_PROMPT_LEN: usize = 1024;
const SHORT_PROMPT_LEN: usize = 13;
const WARMUP_TOKENS: i32 = 4;
const SAMPLE_SIZE: usize = 10;
const MEASUREMENT_SECS: u64 = 20;
/// Realistic sampling temperature — exercises the categorical + cached
/// `inv_temp` decode path, not the greedy argmax shortcut.
const DECODE_TEMP: f32 = 0.7;

/// Resolve `<cache>/<repo_id>`; download via `hf` on first miss.
fn ensure_model(repo_id: &str) -> Option<PathBuf> {
    let cache = bench_cache_root().join(repo_id);
    match checkpoint_status(&cache) {
        CheckpointStatus::Complete => return Some(cache),
        CheckpointStatus::Partial { missing } => {
            eprintln!(
                "skipping {repo_id}: partial checkpoint at {} (missing {}: {}).",
                cache.display(),
                missing.len(),
                missing.join(", "),
            );
            return None;
        }
        CheckpointStatus::Missing => {}
    }
    if std::env::var_os("MLX_LM_BENCH_NO_DOWNLOAD").is_some() {
        return None;
    }
    if std::fs::create_dir_all(&cache).is_err() {
        eprintln!("skipping {repo_id}: could not create {}", cache.display());
        return None;
    }
    let status = Command::new("hf")
        .args([
            "download",
            repo_id,
            "--local-dir",
            cache.to_str().unwrap_or_default(),
        ])
        .status();
    match status {
        Ok(s) if s.success() => Some(cache),
        Ok(s) => {
            eprintln!("skipping {repo_id}: `hf download` exited {s}");
            None
        }
        Err(e) => {
            eprintln!("skipping {repo_id}: `hf` not available ({e})");
            None
        }
    }
}

enum CheckpointStatus {
    Missing,
    Complete,
    Partial { missing: Vec<String> },
}

fn checkpoint_status(dir: &Path) -> CheckpointStatus {
    if !dir.join("config.json").exists() {
        return CheckpointStatus::Missing;
    }
    if dir.join("model.safetensors").exists() {
        return CheckpointStatus::Complete;
    }
    let index_path = dir.join("model.safetensors.index.json");
    let Ok(json) = std::fs::read_to_string(&index_path) else {
        return CheckpointStatus::Missing;
    };
    let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&json) else {
        return CheckpointStatus::Missing;
    };
    let Some(weight_map) = parsed.get("weight_map").and_then(|v| v.as_object()) else {
        return CheckpointStatus::Missing;
    };
    let mut shards: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for v in weight_map.values() {
        if let Some(s) = v.as_str() {
            shards.insert(s);
        }
    }
    let missing: Vec<String> = shards
        .iter()
        .filter(|s| !dir.join(s).exists())
        .map(|s| (*s).to_string())
        .collect();
    if missing.is_empty() {
        CheckpointStatus::Complete
    } else {
        CheckpointStatus::Partial { missing }
    }
}

/// Checkpoint cache root: `$MLX_LM_BENCH_CACHE` >
/// `$XDG_CACHE_HOME/mlx-rs-bench` > `$HOME/.cache/mlx-rs-bench`.
fn bench_cache_root() -> PathBuf {
    if let Ok(override_dir) = std::env::var("MLX_LM_BENCH_CACHE") {
        return PathBuf::from(override_dir);
    }
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        return PathBuf::from(xdg).join("mlx-rs-bench");
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".cache").join("mlx-rs-bench");
    }
    PathBuf::from(".mlx-rs-bench-cache")
}

fn synthetic_prompt(len: usize, base_id: i32) -> Array {
    let ids: Vec<i32> = (0..len as i32).map(|i| base_id + (i % 100)).collect();
    Array::from_slice(&ids, &[ids.len() as i32]).index(NewAxis)
}

/// `MLX_LM_BENCH_ONLY=<substr>` filters cells by group-prefix substring;
/// non-matching cells skip even the model load.
fn bench_only_skip(group_prefix: &str) -> bool {
    match std::env::var("MLX_LM_BENCH_ONLY") {
        Ok(v) if !v.is_empty() => !group_prefix.contains(&v),
        _ => false,
    }
}

/// `[1, len]` int32 prompt as an `LMInput` (text-only).
fn lm_input(prompt: &Array) -> LMInput {
    LMInput {
        text: Text {
            tokens: prompt.clone(),
            mask: None,
        },
        #[cfg(feature = "image")]
        image: None,
        #[cfg(feature = "audio")]
        audio: None,
    }
}

/// Prime the cache with `prompt`, chunking at `prefill_chunk_size` (e.g.
/// gemma4 sliding-window models) exactly as production does, and return the
/// last chunk's `prepare` logits. Calling `prepare` with a prompt longer
/// than the sliding window would build a full-length mask against a
/// window-truncated KV and panic — chunking is the production path.
fn prime(ctx: &mut ModelContext, prompt: &Array) -> Array {
    let prompt_len = prompt.shape()[1];
    let logits_of = |res: mlx_lm::PrepareResult, ctx: &mut ModelContext| match res {
        mlx_lm::PrepareResult::Logits(l) => l,
        mlx_lm::PrepareResult::Primed => {
            let seed = Array::from_slice::<i32>(&[0], &[1]);
            ctx.model.step(&seed).unwrap().logits
        }
    };
    if let Some(window) = ctx.model.prefill_chunk_size() {
        if prompt_len > window {
            let mut start = 0;
            while prompt_len - start > window {
                let chunk = prompt.index((.., start..start + window));
                ctx.model.prefill_chunk(&chunk).unwrap();
                start += window;
            }
            let tail = prompt.index((.., start..prompt_len));
            let res = ctx.model.prepare(lm_input(&tail)).unwrap();
            return logits_of(res, ctx);
        }
    }
    let res = ctx.model.prepare(lm_input(prompt)).unwrap();
    logits_of(res, ctx)
}

/// Prefill timing: chunked prime, eval'd; logits discarded.
fn time_prefill(ctx: &mut ModelContext, prompt: &Array) -> Duration {
    ctx.model.reset();
    let t_start = Instant::now();
    let logits = prime(ctx, prompt);
    eval([&logits]).unwrap();
    Instant::now() - t_start
}

/// Decode timing through the exact production decode step
/// ([`decode_step`], shared with `mlx_lm::generate`): N+1 is submitted
/// before N is fenced, so the pipelining can't drift from production.
/// `eval`-fence (not `.item()`, whose host readback hides decode cost).
fn time_decode(ctx: &mut ModelContext, prompt: &Array, steps: i32) -> Duration {
    ctx.model.reset();
    let mut sampler = SamplerState::new(Sampler::Temperature(DECODE_TEMP));
    let initial = prime(ctx, prompt);
    let mut pending = sampler.sample(&initial).unwrap();
    // Fence prefill + first token before timing — otherwise the prompt
    // forward (large for long prompts) folds into the first decode step.
    eval([&pending]).unwrap();
    let t_start = Instant::now();
    for _ in 0..steps as usize {
        let next = decode_step(ctx.model.as_mut(), &mut sampler, &pending).unwrap();
        eval([&pending]).unwrap();
        pending = next;
    }
    eval([&pending]).unwrap();
    Instant::now() - t_start
}

/// MTP self-speculative A/B: time a fixed-length `generate` with MTP on
/// vs off on the same checkpoint. Throughput is keyed on actually-emitted
/// tokens (MTP commits a variable count per call), so the two cells are
/// directly comparable tok/s. Sampling is top-p (temp > 0) so the MTP
/// path exercises the Leviathan rejection branch, not the greedy
/// shortcut; on/off outputs differ (stochastic) — this measures speed,
/// not parity (parity is the e2e greedy test's job).
fn maybe_bench_mtp(c: &mut Criterion, label: &str, repo_id: &str) {
    let group_name = format!("qwen3_5_moe_mtp_{label}");
    if bench_only_skip(&group_name) {
        return;
    }
    let Some(dir) = ensure_model(repo_id) else {
        return;
    };
    let mut ctx = match load(&dir) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping {group_name}: load failed: {e:?}");
            return;
        }
    };
    if !ctx.model.has_mtp() {
        eprintln!("skipping {group_name}: checkpoint has no MTP head");
        return;
    }

    const PROMPT: &str = "Summarize the history of the city of Paris in detail.";
    let params = |disable_mtp: bool| GenerateParams {
        max_new_tokens: DECODE_TOKENS,
        sampling: Sampler::TopP {
            temperature: DECODE_TEMP,
            p: 0.95,
        },
        disable_mtp,
        ..Default::default()
    };
    let run = |ctx: &mut ModelContext, disable_mtp: bool| -> Duration {
        let t = Instant::now();
        let out = generate(
            ctx,
            UserInput::text(PROMPT),
            params(disable_mtp),
            &mut |_, _| ControlFlow::Continue(()),
        )
        .unwrap();
        // Guard against a zero-token result skewing throughput.
        debug_assert!(out.completion_tokens > 0);
        Instant::now() - t
    };

    // Warm kernel/compile caches outside the timing window.
    let _ = run(&mut ctx, false);

    let mut group = c.benchmark_group(&group_name);
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));
    group.throughput(Throughput::Elements(DECODE_TOKENS as u64));
    for (id, disable_mtp) in [
        (BenchmarkId::new("mtp_on", DECODE_TOKENS), false),
        (BenchmarkId::new("mtp_off", DECODE_TOKENS), true),
    ] {
        group.bench_function(id, |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    total += run(&mut ctx, disable_mtp);
                }
                total
            });
        });
    }
    group.finish();
}

fn maybe_bench(c: &mut Criterion, family: &str, label: &str, repo_id: &str) {
    let group_name = format!("{family}_decode_{label}");
    if bench_only_skip(&group_name) {
        return;
    }
    let Some(dir) = ensure_model(repo_id) else {
        return;
    };
    let mut ctx = match load(&dir) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping {group_name}: load failed: {e:?}");
            return;
        }
    };

    let short = synthetic_prompt(SHORT_PROMPT_LEN, 1000);
    let long = synthetic_prompt(LONG_PROMPT_LEN, 1000);

    // Warm the compile/kernel cache outside the timing window.
    for _ in 0..WARMUP_TOKENS {
        let _ = time_decode(&mut ctx, &short, 1);
    }

    let decode_steps = DECODE_TOKENS - 1;
    let mut group = c.benchmark_group(&group_name);
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));

    for (id, prompt) in [
        (
            BenchmarkId::new("prefill_short", SHORT_PROMPT_LEN as i32),
            &short,
        ),
        (
            BenchmarkId::new("prefill_long", LONG_PROMPT_LEN as i32),
            &long,
        ),
    ] {
        let prompt_len = prompt.shape().last().copied().unwrap_or(0) as u64;
        group.throughput(Throughput::Elements(prompt_len));
        group.bench_function(id, |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    total += time_prefill(&mut ctx, prompt);
                }
                total
            });
        });
    }

    group.throughput(Throughput::Elements(decode_steps as u64));
    for (id, prompt) in [
        (BenchmarkId::new("decode_short", decode_steps), &short),
        (BenchmarkId::new("decode_long", decode_steps), &long),
    ] {
        group.bench_function(id, |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    total += time_decode(&mut ctx, prompt, decode_steps);
                }
                total
            });
        });
    }
    group.finish();
}

/// `MLX_LM_BENCH_SET=full` adds llama 3B + qwen3 0.6B cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchSet {
    Trimmed,
    Full,
}

fn bench_set() -> BenchSet {
    match std::env::var("MLX_LM_BENCH_SET").as_deref() {
        Ok("full") | Ok("all") => BenchSet::Full,
        _ => BenchSet::Trimmed,
    }
}

fn bench_decode(c: &mut Criterion) {
    eprintln!("lm_decode cache root: {}", bench_cache_root().display());
    let set = bench_set();
    eprintln!("lm_decode bench set: {set:?} (override with MLX_LM_BENCH_SET={{trimmed,full}})");

    maybe_bench(c, "qwen3", "large_bf16", "mlx-community/Qwen3-1.7B-bf16");
    maybe_bench(c, "qwen3", "large_q8", "mlx-community/Qwen3-1.7B-8bit");
    maybe_bench(c, "qwen3", "large_q4", "mlx-community/Qwen3-1.7B-4bit");
    maybe_bench(
        c,
        "llama",
        "small_bf16",
        "mlx-community/Llama-3.2-1B-Instruct-bf16",
    );
    maybe_bench(
        c,
        "llama",
        "small_q8",
        "mlx-community/Llama-3.2-1B-Instruct-8bit",
    );
    maybe_bench(
        c,
        "llama",
        "small_q4",
        "mlx-community/Llama-3.2-1B-Instruct-4bit",
    );

    if set == BenchSet::Full {
        maybe_bench(c, "qwen3", "small_bf16", "mlx-community/Qwen3-0.6B-bf16");
        maybe_bench(c, "qwen3", "small_q8", "mlx-community/Qwen3-0.6B-8bit");
        maybe_bench(c, "qwen3", "small_q4", "mlx-community/Qwen3-0.6B-4bit");
        maybe_bench(
            c,
            "llama",
            "large_bf16",
            "mlx-community/Llama-3.2-3B-Instruct-bf16",
        );
        maybe_bench(
            c,
            "llama",
            "large_q8",
            "mlx-community/Llama-3.2-3B-Instruct-8bit",
        );
        maybe_bench(
            c,
            "llama",
            "large_q4",
            "mlx-community/Llama-3.2-3B-Instruct-4bit",
        );

        // qwen3.5 GDN-hybrid dense + MoE — the new perf-sensitive paths
        // (GDN scan kernel, gather_qmm experts, MTP). Heavy models, so
        // Full-set only; each self-skips when its checkpoint is absent.
        maybe_bench(
            c,
            "qwen3_5",
            "dense_q8",
            "mlx-community/Qwen3.5-4B-MLX-8bit",
        );
        maybe_bench(
            c,
            "qwen3_5",
            "dense_q4",
            "mlx-community/Qwen3.5-4B-MLX-4bit",
        );
        maybe_bench(
            c,
            "qwen3_5_moe",
            "a3b_q8",
            "mlx-community/Qwen3.6-35B-A3B-q8-mtp",
        );
        maybe_bench_mtp(c, "a3b_q8", "mlx-community/Qwen3.6-35B-A3B-q8-mtp");

        // gemma4 dense 31B — hybrid sliding/global attention + proportional
        // rope. Memory-bandwidth-bound; q4 vs q8 isolates the weight-bits
        // effect on decode. Full-set only; each self-skips when absent.
        maybe_bench(
            c,
            "gemma4",
            "31b_it_q4",
            "mlx-community/gemma-4-31b-it-4bit",
        );
        maybe_bench(
            c,
            "gemma4",
            "31b_it_q8",
            "mlx-community/gemma-4-31b-it-8bit",
        );

        // gemma4 MoE 26B-A4B — dual-branch (dense MLP + 128-expert top-8)
        // over the same attention spine. Exercises the gather_qmm expert
        // path + the router top-k kernel.
        maybe_bench(
            c,
            "gemma4",
            "26b_a4b_it_q8",
            "mlx-community/gemma-4-26b-a4b-it-8bit",
        );
        maybe_bench(
            c,
            "gemma4",
            "26b_a4b_it_q4",
            "mlx-community/gemma-4-26b-a4b-it-4bit",
        );

        // gemma4 E2B/E4B — per-layer-input embeddings + KV-sharing.
        maybe_bench(
            c,
            "gemma4",
            "e2b_it_q8",
            "mlx-community/gemma-4-e2b-it-8bit",
        );
        maybe_bench(
            c,
            "gemma4",
            "e4b_it_q8",
            "mlx-community/gemma-4-e4b-it-8bit",
        );
    }
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);
