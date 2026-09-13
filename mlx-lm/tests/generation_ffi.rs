use std::{num::NonZeroUsize, path::PathBuf};

use anyhow::{ensure, Context, Result};
use mlx_lm::{
    Cache, CacheKind, CacheOptions, CachePolicy, FinishReason, Generation, GenerationEvent,
    GenerationOptions, Model, Prompt, StopTokenPolicy, TokenId,
};
use mlx_rs::{memory, with_device, Device};
use safetensors::{Dtype, SafeTensors};

const SAMPLES: usize = 4096;
const CHECKPOINTS: [usize; 4] = [512, 1024, 2048, 4096];
const MIB: usize = 1024 * 1024;

#[derive(Clone, Copy)]
struct Workload {
    samples: usize,
    measure_memory: bool,
}

fn nz(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).unwrap()
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../conformance/mlx-lm/fixtures")
        .join(name)
}

fn prompt() -> Vec<TokenId> {
    [12, 14, 15, 16, 17, 18, 19, 20]
        .into_iter()
        .map(TokenId::from)
        .collect()
}

fn options(samples: usize, policy: CachePolicy) -> GenerationOptions {
    let mut options = GenerationOptions::default();
    options.max_tokens = nz(samples);
    options.prefill_chunk_size = nz(64);
    options.stop.tokens = StopTokenPolicy::Exact(vec![]);
    options.cache.policy = policy;
    options
}

struct Footprint {
    model_bytes: usize,
    kv_bytes_per_position: usize,
}

impl Footprint {
    fn read(name: &str, model: &Model) -> Result<Self> {
        let bytes = std::fs::read(fixture(name).join("model.safetensors"))?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        let embedding = tensors.tensor(if model.config().quantization.is_some() {
            "model.embed_tokens.scales"
        } else {
            "model.embed_tokens.weight"
        })?;
        ensure!(embedding.dtype() == Dtype::F32, "fixture K/V must be f32");
        let dimensions = &model.config().dimensions;
        Ok(Self {
            // Payload bytes exclude the safetensors header and allocator overhead.
            model_bytes: tensors.tensors().iter().map(|(_, t)| t.data().len()).sum(),
            kv_bytes_per_position: 2 * dimensions.kv_heads * dimensions.head_dim * 4,
        })
    }

    fn kv_bytes(&self, cache: &Cache) -> usize {
        cache.info().map(|info| info.capacity).sum::<usize>() * self.kv_bytes_per_position
    }
}

struct MemoryGuard {
    name: String,
    warm_active: usize,
    warm_kv: usize,
    first: Option<(usize, usize)>,
}

impl MemoryGuard {
    fn warm(name: &str, footprint: &Footprint, cache: &Cache) -> Result<Self> {
        let warm_active = report(name, "warm", 64, footprint, cache, 0)?;
        Ok(Self {
            name: name.to_owned(),
            warm_active,
            warm_kv: footprint.kv_bytes(cache),
            first: None,
        })
    }

    fn checkpoint(&mut self, count: usize, footprint: &Footprint, cache: &Cache) -> Result<()> {
        let active = report(&self.name, "checkpoint", count, footprint, cache, 0)?;
        let kv = footprint.kv_bytes(cache);
        let expected_growth = kv as i128 - self.warm_kv as i128;
        let slack = 2 * kv + MIB;
        let envelope = self.warm_active as i128 + expected_growth + slack as i128;
        let first = *self.first.get_or_insert((active, kv));
        let adjusted_growth = active as i128 - first.0 as i128 - (kv as i128 - first.1 as i128);
        eprintln!(
            "memory_guard scenario={} count={count} warm_active_bytes={} expected_kv_growth_bytes={expected_growth} slack_bytes={slack} envelope_bytes={envelope} adjusted_growth_bytes={adjusted_growth} within_envelope={} within_growth={} qualification=proposed_F",
            self.name, self.warm_active, active as i128 <= envelope, adjusted_growth <= slack as i128,
        );
        ensure!(
            active as i128 <= envelope,
            "{} exceeds proposed F envelope; review on host, do not ratchet",
            self.name
        );
        ensure!(
            adjusted_growth <= slack as i128,
            "{} exceeds proposed F growth guard",
            self.name
        );
        Ok(())
    }
}

fn report(
    name: &str,
    phase: &str,
    count: usize,
    footprint: &Footprint,
    cache: &Cache,
    snapshots: usize,
) -> Result<usize> {
    let active = memory::active_memory()?;
    let allocator = memory::cache_memory()?;
    eprintln!(
        "memory scenario={name} phase={phase} count={count} model_payload_bytes={} configured_kv_bytes={} active_bytes={active} allocator_cache_bytes={allocator} held_snapshots={snapshots} layers={:?}",
        footprint.model_bytes, footprint.kv_bytes(cache), cache.info().collect::<Vec<_>>(),
    );
    Ok(active)
}

fn consume(generation: &mut Generation<'_>) -> Result<(usize, FinishReason)> {
    let mut count = 0;
    let mut terminal = None;
    for event in generation.by_ref() {
        if let GenerationEvent::Token { finish_reason, .. } = event? {
            ensure!(terminal.is_none(), "token after terminal event");
            count += 1;
            terminal = finish_reason;
        }
    }
    ensure!(generation.next().is_none(), "iterator must remain fused");
    Ok((count, terminal.context("missing terminal event")?))
}

fn long_generation(
    name: &str,
    policy: CachePolicy,
    rotating: bool,
    workload: Workload,
) -> Result<()> {
    let mut model = Model::from_dir(fixture(name))?;
    let footprint = Footprint::read(name, &model)?;
    let samples = workload.samples + usize::from(rotating);
    let prompt = prompt();
    let scenario = format!("{name}/{}", if rotating { "rotating" } else { "full" });
    let mut generation = model.generate(Prompt::Tokens(&prompt), options(samples, policy))?;
    let mut guard = None;
    let mut count = 0;
    while let Some(event) = generation.next() {
        let (is_token, finish) = match event? {
            GenerationEvent::Token { finish_reason, .. } => (true, finish_reason),
            _ => (false, None),
        };
        if !is_token {
            continue;
        }
        count += 1;
        ensure!(finish == (count == samples).then_some(FinishReason::Length));
        let updates = count - usize::from(rotating);
        if workload.measure_memory && updates == 64 {
            guard = Some(MemoryGuard::warm(
                &scenario,
                &footprint,
                generation.cache(),
            )?);
        }
        if workload.measure_memory && CHECKPOINTS.contains(&updates) {
            guard
                .as_mut()
                .unwrap()
                .checkpoint(updates, &footprint, generation.cache())?;
        }
        for info in generation.cache().info() {
            ensure!(info.processed_tokens == prompt.len() + count - 1);
            if rotating {
                ensure!(info.kind == CacheKind::Rotating && info.capacity == 32);
                ensure!(info.retained_prefix == (0..4));
                ensure!(info.retained_prefix.len() + info.retained_positions.len() <= 32);
            } else {
                ensure!(info.kind == CacheKind::Full && info.capacity == prompt.len() + samples);
            }
        }
    }
    ensure!(count == samples);
    ensure!(generation.next().is_none());
    Ok(())
}

fn reuse_growth(name: &str, workload: Workload) -> Result<()> {
    let mut model = Model::from_dir(fixture(name))?;
    let footprint = Footprint::read(name, &model)?;
    let options = options(1, CachePolicy::Full);
    let mut cache = model.new_cache(options.cache.clone())?;
    ensure!(cache.info().all(|info| info.capacity == 16));
    let mut capacities = vec![cache.info().next().unwrap().capacity];
    let mut full_prompt = Vec::new();
    let mut guard = None;
    for count in 1..=workload.samples {
        full_prompt.push(TokenId::from(12));
        let mut generation =
            model.generate_with_cache(Prompt::Tokens(&full_prompt), options.clone(), &mut cache)?;
        ensure!(consume(&mut generation)? == (1, FinishReason::Length));
        drop(generation);
        ensure!(cache.tokens() == full_prompt);
        let capacity = cache.info().next().unwrap().capacity;
        if capacities.last() != Some(&capacity) {
            capacities.push(capacity);
        }
        ensure!(cache
            .info()
            .all(|info| info.capacity == capacity && info.processed_tokens == count));
        if workload.measure_memory && count == 64 {
            guard = Some(MemoryGuard::warm(
                &format!("{name}/reuse"),
                &footprint,
                &cache,
            )?);
        }
        if workload.measure_memory && CHECKPOINTS.contains(&count) {
            guard
                .as_mut()
                .unwrap()
                .checkpoint(count, &footprint, &cache)?;
        }
    }
    let expected_capacities: &[usize] = if workload.measure_memory {
        &[16, 32, 64, 128, 256, 512]
    } else {
        &[16, 32, 64]
    };
    ensure!(capacities.starts_with(expected_capacities));
    eprintln!(
        "reuse scenario={name} calls={} capacities={capacities:?}",
        workload.samples
    );
    Ok(())
}

fn sliding_prefill_and_snapshot(workload: Workload) -> Result<()> {
    let name = "llama-sliding";
    let mut model = Model::from_dir(fixture(name))?;
    let footprint = Footprint::read(name, &model)?;
    let (prompt_len, samples, chunk_size) = if workload.measure_memory {
        (193, 130, 64)
    } else {
        (25, 24, 8)
    };
    let full_prompt = vec![TokenId::from(12); prompt_len];
    let mut generation_options = options(samples, CachePolicy::ModelDefault);
    generation_options.prefill_chunk_size = nz(chunk_size);
    let mut cache = model.new_cache(CacheOptions::default())?;
    let mut generation =
        model.generate_with_cache(Prompt::Tokens(&full_prompt), generation_options, &mut cache)?;
    let mut oversized = false;
    let mut snapshot = None;
    let mut saved_tokens = Vec::new();
    let mut count = 0;
    while let Some(event) = generation.next() {
        match event? {
            GenerationEvent::Prefill {
                processed, total, ..
            } if processed > 0 && processed < total => {
                oversized |= generation.cache().info().any(|info| {
                    info.kind == CacheKind::Rotating
                        && info.retained_positions.len() > info.capacity
                });
            }
            GenerationEvent::Token { finish_reason, .. } => {
                count += 1;
                ensure!(finish_reason == (count == samples).then_some(FinishReason::Length));
                if count == 16 {
                    saved_tokens = generation.cache().tokens().to_vec();
                    snapshot = Some(generation.snapshot()?);
                    if workload.measure_memory {
                        report(
                            name,
                            "snapshot-retained",
                            count,
                            &footprint,
                            generation.cache(),
                            1,
                        )?;
                    }
                }
                if count > 1 {
                    for info in generation.cache().info() {
                        if info.kind == CacheKind::Rotating {
                            ensure!(info.capacity == 5 && info.retained_prefix.is_empty());
                            ensure!(info.retained_positions.len() <= 5);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    ensure!(oversized && count == samples);
    drop(generation);
    cache.restore(snapshot.context("snapshot not captured")?)?;
    ensure!(cache.tokens() == saved_tokens);
    ensure!(cache
        .info()
        .all(|info| info.processed_tokens == saved_tokens.len()));
    saved_tokens.push(TokenId::from(12));
    let mut generation = model.generate_with_cache(
        Prompt::Tokens(&saved_tokens),
        options(2, CachePolicy::ModelDefault),
        &mut cache,
    )?;
    ensure!(consume(&mut generation)? == (2, FinishReason::Length));
    drop(generation);
    if workload.measure_memory {
        report(name, "snapshot-released", count, &footprint, &cache, 0)?;
    }
    Ok(())
}

fn completions_and_cancellation(model: &mut Model) -> Result<()> {
    let prompt = prompt();
    let base = options(8, CachePolicy::Full);
    let mut generation = model.generate(Prompt::Tokens(&prompt), base.clone())?;
    let mut first = None;
    let mut text = String::new();
    for event in generation.by_ref() {
        if let GenerationEvent::Token {
            token_id,
            text: delta,
            ..
        } = event?
        {
            first.get_or_insert(token_id);
            text.push_str(&delta);
        }
    }
    drop(generation);
    let first = first.context("no sampled token")?;
    let stop_text = text
        .split_whitespace()
        .next()
        .context("no text to stop on")?
        .to_owned();
    for text_stop in [false, true] {
        let mut options = base.clone();
        if text_stop {
            options.stop.stop_strings.push(stop_text.clone());
        } else {
            options.stop.tokens = StopTokenPolicy::Exact(vec![first]);
        }
        let mut generation = model.generate(Prompt::Tokens(&prompt), options)?;
        let mut visible = String::new();
        let mut terminal = None;
        for event in generation.by_ref() {
            if let GenerationEvent::Token {
                text,
                finish_reason,
                ..
            } = event?
            {
                ensure!(terminal.is_none());
                visible.push_str(&text);
                terminal = finish_reason;
            }
        }
        ensure!(terminal == Some(FinishReason::Stop) && visible.is_empty());
        ensure!(generation.next().is_none());
    }
    for boundary in [0, 1, 2] {
        let mut options = base.clone();
        options.prefill_chunk_size = nz(3);
        let mut cache = model.new_cache(options.cache.clone())?;
        let mut generation =
            model.generate_with_cache(Prompt::Tokens(&prompt), options, &mut cache)?;
        let mut expected = 0;
        if boundary != 0 {
            loop {
                match generation
                    .next()
                    .context("missing cancellation boundary")??
                {
                    GenerationEvent::Prefill {
                        processed, total, ..
                    } if boundary == 1 && processed > 0 && processed < total => {
                        expected = processed;
                        break;
                    }
                    GenerationEvent::Token { .. } if boundary == 2 => {
                        expected = prompt.len();
                        break;
                    }
                    _ => {}
                }
            }
        }
        drop(generation);
        ensure!(cache.tokens() == &prompt[..expected]);
        ensure!(cache.info().all(|info| info.processed_tokens == expected));
    }
    Ok(())
}

fn load_drop_cycles(workload: Workload) -> Result<()> {
    let mut warm = None;
    let mut slack = 0;
    let (cycles, names): (usize, &[&str]) = if workload.measure_memory {
        (9, &["llama-base", "qwen3-base", "llama-quant4"])
    } else {
        (1, &["llama-quant4"])
    };
    for cycle in 0..cycles {
        for &name in names {
            let mut model = Model::from_dir(fixture(name))?;
            let footprint = Footprint::read(name, &model)?;
            if name == "llama-quant4" {
                ensure!(model.config().quantization.is_some());
            }
            let mut generation =
                model.generate(Prompt::Tokens(&prompt()), options(8, CachePolicy::Full))?;
            ensure!(consume(&mut generation)? == (8, FinishReason::Length));
            if workload.measure_memory {
                slack = slack.max(2 * footprint.kv_bytes(generation.cache()) + MIB);
                report(
                    name,
                    "load-generate-drop",
                    cycle,
                    &footprint,
                    generation.cache(),
                    0,
                )?;
            }
            drop(generation);
            completions_and_cancellation(&mut model)?;
            drop(model);
        }
        if workload.measure_memory {
            let active = memory::active_memory()?;
            let baseline = *warm.get_or_insert(active);
            eprintln!("load_drop cycle={cycle} active_bytes={active} allocator_cache_bytes={} warm_active_bytes={baseline} slack_bytes={slack} model_payload_bytes=0 configured_kv_bytes=0 held_snapshots=0", memory::cache_memory()?);
            ensure!(
                active <= baseline + slack,
                "load/drop exceeds proposed F envelope"
            );
        }
    }
    Ok(())
}

fn tokenizer_eos_completion() -> Result<()> {
    let source = fixture("llama-base");
    let mut model = Model::from_dir(&source)?;
    let mut generation =
        model.generate(Prompt::Tokens(&prompt()), options(1, CachePolicy::Full))?;
    let mut eos = None;
    for event in generation.by_ref() {
        if let GenerationEvent::Token { token_id, .. } = event? {
            eos = Some(token_id);
        }
    }
    drop(generation);
    drop(model);
    let eos = eos.context("no EOS candidate")?;
    let directory = tempfile::tempdir()?;
    for file in [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ] {
        std::fs::copy(source.join(file), directory.path().join(file))?;
    }
    std::fs::write(
        directory.path().join("generation_config.json"),
        serde_json::to_vec(&serde_json::json!({"eos_token_id": u32::from(eos)}))?,
    )?;
    let mut model = Model::from_dir(directory.path())?;
    ensure!(model.tokenizer().eos_tokens() == [eos]);
    let mut options = options(8, CachePolicy::Full);
    options.stop.tokens = StopTokenPolicy::Tokenizer;
    let mut generation = model.generate(Prompt::Tokens(&prompt()), options)?;
    let mut count = 0;
    for event in generation.by_ref() {
        if let GenerationEvent::Token {
            token_id,
            text,
            finish_reason,
            ..
        } = event?
        {
            count += 1;
            ensure!(
                token_id == eos && text.is_empty() && finish_reason == Some(FinishReason::Stop)
            );
        }
    }
    ensure!(count == 1 && generation.next().is_none());
    Ok(())
}

#[test]
fn metal_generation_memory_and_ffi_workload() -> Result<()> {
    let guard_malloc = std::env::var("DYLD_INSERT_LIBRARIES")
        .is_ok_and(|libraries| libraries.contains("libgmalloc"));
    let workload = Workload {
        samples: if guard_malloc { 40 } else { SAMPLES },
        measure_memory: !guard_malloc,
    };
    // One test serializes process-global allocator observations, including default test runs.
    with_device(Device::gpu(), || {
        let probe = mlx_rs::ops::zeros::<f32>(&[1])?;
        probe.eval().context(
            "Metal unavailable: qualification failed/not run; CPU substitution is forbidden",
        )?;
        drop(probe);
        eprintln!(
            "qualification device=Metal os={} arch={} fixture_tuple={}",
            std::env::consts::OS,
            std::env::consts::ARCH,
            serde_json::from_str::<serde_json::Value>(include_str!(
                "../../conformance/mlx-lm/manifest.json"
            ))?
        );
        if guard_malloc {
            long_generation("qwen3-base", CachePolicy::Full, false, workload)?;
            reuse_growth("llama-base", workload)?;
        } else {
            for name in ["llama-base", "qwen3-base"] {
                long_generation(name, CachePolicy::Full, false, workload)?;
                reuse_growth(name, workload)?;
            }
        }
        long_generation(
            "llama-base",
            CachePolicy::Rotating {
                capacity: nz(32),
                keep_prefix: 4,
            },
            true,
            workload,
        )?;
        sliding_prefill_and_snapshot(workload)?;
        tokenizer_eos_completion()?;
        load_drop_cycles(workload)?;
        if guard_malloc {
            eprintln!(
                "qualification mode=guard-malloc-reduced memory_checks=disabled scenarios_ran=qwen3-base/full(40),llama-base/reuse(40),llama-base/rotating(40-updates,keep-prefix),llama-sliding/oversized-prefill(25,chunk=8)+snapshot-across-wrap(24),llama-base/tokenizer-eos,llama-quant4/load-generate-drop(1)+length+exact-token-stop+text-stop+drop-before-next+drop-at-prefill+drop-after-token"
            );
        }
        Ok(())
    })
}
