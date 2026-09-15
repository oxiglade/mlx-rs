//! Local ruling-C worker. The Python coordinator owns pairing and report interpretation.

#[allow(dead_code)]
#[path = "../tests/real_checkpoints.rs"]
mod checkpoints;

use anyhow::{ensure, Context, Result};
use mlx_lm::{FinishReason, GenerationEvent, Model, Prompt, TokenId};
use mlx_rs::{with_device, Device};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    time::Instant,
};

fn request(model: &mut Model, prompt: &[TokenId], chunk: usize) -> Result<Value> {
    let options = checkpoints::options(256, chunk)?;
    let mut ids = Vec::with_capacity(256);
    let mut timestamps = Vec::with_capacity(256);
    let mut sink = String::new();
    let mut forwards = 0;
    let mut completed = false;
    let start = Instant::now();
    for event in model.generate(Prompt::Tokens(prompt), options)? {
        match event? {
            GenerationEvent::Prefill { processed, .. } if processed > 0 => forwards += 1,
            GenerationEvent::Token {
                token_id,
                text,
                finish_reason,
                ..
            } => {
                ensure!(!completed, "token after completion");
                if !ids.is_empty() {
                    forwards += 1;
                }
                sink.push_str(&text);
                ids.push(u32::from(token_id));
                timestamps.push(start.elapsed().as_nanos() as u64);
                completed = finish_reason == Some(FinishReason::Length);
                ensure!(finish_reason.is_none() || completed, "unexpected stop");
            }
            _ => {}
        }
    }
    // Public Generation evaluates every output and cache transaction before its event.
    let completion = start.elapsed().as_nanos() as u64;
    ensure!(
        completed && ids.len() == 256,
        "incomplete fixed-count request"
    );
    ensure!(
        forwards == (prompt.len() - 1).div_ceil(chunk) + 256,
        "forward count differs"
    );
    std::hint::black_box(&sink);
    Ok(
        json!({"prompt_ids": checkpoints::raw_ids(prompt), "ids": ids, "text": sink, "start_ns": 0,
        "token_ns": timestamps, "completion_ns": completion, "forward_count": forwards,
        "forward_count_method": "one per completed prefill boundary plus each decode token after first",
        "drain": "synchronous evaluated public Generation exhausted", "completed": true}),
    )
}

fn main() -> Result<()> {
    checkpoints::refuse_ci()?;
    let mut args = BTreeMap::new();
    let mut iter = std::env::args().skip(1);
    while let Some(key) = iter.next() {
        if key == "--help" {
            println!("benchmark --manifest PATH --case qwen3-06b-4bit --device cpu|metal --output NEW_PATH");
            return Ok(());
        }
        ensure!(
            ["--manifest", "--case", "--device", "--output"].contains(&key.as_str()),
            "unknown argument {key}"
        );
        let value = iter.next().context("argument value missing")?;
        ensure!(args.insert(key, value).is_none(), "duplicate argument");
    }
    let get = |key| {
        args.get(key)
            .map(String::as_str)
            .context("required argument missing")
    };
    let output = Path::new(get("--output")?);
    let mut report_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let name = get("--case")?;
    ensure!(name == "qwen3-06b-4bit", "ruling C requires qwen3-06b-4bit");
    let manifest: Value = serde_json::from_slice(&fs::read(get("--manifest")?)?)?;
    ensure!(
        manifest["schema_version"] == 1 && manifest["devices"] == json!(["cpu", "metal"]),
        "invalid manifest/devices"
    );
    checkpoints::verify_identity(name, &manifest["entries"][name])?;
    checkpoints::verify_entry(&manifest["entries"][name])?;
    let benchmark = &manifest["benchmark"];
    ensure!(
        benchmark["case"] == name
            && benchmark["max_tokens"] == 256
            && benchmark["stop_token_ids"] == json!([]),
        "invalid benchmark protocol"
    );
    ensure!(
        benchmark["tokenizer_sha256"] == manifest["entries"][name]["files"]["tokenizer.json"]
            && !benchmark["tokenizer_sha256"].is_null(),
        "tokenizer identity differs"
    );
    ensure!(
        benchmark["prompt_rule"] == "first_128_without_special_tokens",
        "unknown prompt rule"
    );
    let prompt: Vec<_> = checkpoints::ids(&benchmark["prompt_ids"])?
        .into_iter()
        .map(TokenId::from)
        .collect();
    ensure!(
        prompt.len() == 128,
        "ruling C requires 128 explicit prompt IDs"
    );
    let chunk = benchmark["prefill_chunk_size"]
        .as_u64()
        .context("missing prefill chunk")? as usize;
    ensure!(chunk > 0, "zero prefill chunk");
    let device_name = get("--device")?;
    let device = match device_name {
        "cpu" => Device::cpu(),
        "metal" => Device::gpu(),
        _ => anyhow::bail!("explicit cpu|metal device required"),
    };
    let report = with_device(device, || -> Result<Value> {
        let mut model = checkpoints::load_case(&manifest, name)?;
        let source_ids = model
            .tokenizer()
            .encode(checkpoints::string(&benchmark["source_text"])?)?;
        ensure!(
            source_ids.get(..128) == Some(prompt.as_slice()),
            "source-text prompt prefix differs"
        );
        let config = checkpoints::config_value(model.config());
        let mut warmups = Vec::new();
        for _ in 0..3 {
            warmups.push(request(&mut model, &prompt, chunk)?);
        }
        let sample = request(&mut model, &prompt, chunk)?;
        Ok(
            json!({"schema_version": 1, "language": "rust", "device": device_name,
            "stream": format!("explicit {device_name} default device stream via with_device"),
            "case": name, "protocol": benchmark, "resolved_config": config,
            "allocator_policy": "unchanged defaults; no wired-limit adjustment",
            "warmups": warmups, "sample": sample}),
        )
    })?;
    serde_json::to_writer_pretty(&mut report_file, &report)?;
    writeln!(report_file)?;
    Ok(())
}
