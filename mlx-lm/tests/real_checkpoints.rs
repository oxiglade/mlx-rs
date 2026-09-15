use anyhow::{bail, ensure, Context, Result};
use mlx_lm::{
    AttentionKind, CachePolicy, Config, FinishReason, GenerationEvent, GenerationOptions,
    LayerQuantization, Model, Prompt, RopeScaling, StopTokenPolicy, TokenId, Tokenizer,
};
use mlx_rs::{with_device, Device};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    num::NonZeroUsize,
    path::{Component, Path, PathBuf},
    process::Command,
};

pub const CASES: [&str; 6] = [
    "llama-1b-bf16",
    "llama-1b-4bit",
    "qwen3-06b-bf16",
    "qwen3-06b-4bit",
    "gguf-tinyllama-q4_0",
    "gguf-qwen3-q8_0",
];
pub const CI_KEYS: [&str; 12] = [
    "CI",
    "CONTINUOUS_INTEGRATION",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "BUILDKITE",
    "JENKINS_URL",
    "TEAMCITY_VERSION",
    "TF_BUILD",
    "CIRCLECI",
    "TRAVIS",
    "BLAZE_JOB_ID",
    "BUILD_BUILDID",
];

pub fn refuse_ci() -> Result<()> {
    ensure!(
        !CI_KEYS.iter().any(|key| std::env::var_os(key).is_some()),
        "local harness refuses CI environments"
    );
    Ok(())
}

pub fn string(value: &Value) -> Result<&str> {
    value.as_str().context("required string missing")
}

pub fn sha256(path: &Path) -> Result<String> {
    let output = Command::new("shasum")
        .args(["-a", "256", "--"])
        .arg(path)
        .output()?;
    ensure!(
        output.status.success(),
        "hash command failed for {}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout)?;
    let digest = text
        .split_whitespace()
        .next()
        .context("missing hash output")?;
    ensure!(hex(digest, 64), "invalid hash output");
    Ok(digest.to_owned())
}

fn hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn relative(value: &str) -> Result<&Path> {
    ensure!(
        !value.is_empty()
            && !value.contains('\\')
            && value.split('/').all(|p| !matches!(p, "" | "." | "..")),
        "invalid relative path {value}"
    );
    let path = Path::new(value);
    ensure!(
        path.components()
            .all(|part| matches!(part, Component::Normal(_))),
        "invalid relative path {value}"
    );
    Ok(path)
}

pub fn verify_hash(path: &Path, expected: &Value) -> Result<()> {
    let expected = string(expected)?;
    ensure!(
        hex(expected, 64),
        "missing/invalid SHA-256 for {}",
        path.display()
    );
    ensure!(
        sha256(path)? == expected,
        "hash mismatch: {}",
        path.display()
    );
    Ok(())
}

fn discover(root: &Path, dir: &Path, files: &serde_json::Map<String, Value>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            ensure!(!path.is_symlink(), "directory symlink is not admitted");
            discover(root, &path, files)?;
        } else if matches!(
            path.extension().and_then(|s| s.to_str()),
            Some(
                "json"
                    | "safetensors"
                    | "gguf"
                    | "model"
                    | "txt"
                    | "jinja"
                    | "tiktoken"
                    | "py"
                    | "jsonl"
            )
        ) {
            let name = path
                .strip_prefix(root)?
                .to_str()
                .context("non-UTF8 checkpoint path")?;
            ensure!(
                files.contains_key(name),
                "unlisted consumed/discoverable file: {}",
                path.display()
            );
        }
    }
    Ok(())
}

pub fn verify_entry(entry: &Value) -> Result<PathBuf> {
    ensure!(
        !string(&entry["repository"])?.is_empty() && hex(string(&entry["revision"])?, 40),
        "full immutable repository revision required"
    );
    let root = PathBuf::from(string(&entry["local_directory"])?);
    ensure!(
        root.is_absolute() && root.is_dir(),
        "missing absolute local directory: {}",
        root.display()
    );
    let files = entry["files"].as_object().context("missing files")?;
    ensure!(!files.is_empty(), "empty files");
    let mut total = 0u64;
    for (name, digest) in files {
        let path = root.join(relative(name)?);
        verify_hash(&path, digest)?;
        total = total
            .checked_add(fs::metadata(path)?.len())
            .context("byte total overflow")?;
    }
    ensure!(
        Some(total) == entry["bytes"].as_u64(),
        "byte total mismatch"
    );
    discover(&root, &root, files)?;
    Ok(root)
}

pub fn tokenizer_entry(name: &str) -> &str {
    match name {
        "gguf-tinyllama-q4_0" => "gguf-tinyllama-tokenizer",
        "gguf-qwen3-q8_0" => "gguf-qwen3-tokenizer",
        _ => name,
    }
}

pub fn verify_identity(name: &str, entry: &Value) -> Result<()> {
    let repository = match name {
        "llama-1b-bf16" => "mlx-community/Llama-3.2-1B-Instruct-bf16",
        "llama-1b-4bit" => "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "qwen3-06b-bf16" => "mlx-community/Qwen3-0.6B-bf16",
        "qwen3-06b-4bit" => "mlx-community/Qwen3-0.6B-4bit",
        "gguf-tinyllama-q4_0" => "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "gguf-qwen3-q8_0" => "ggml-org/Qwen3-0.6B-GGUF",
        "gguf-tinyllama-tokenizer" => "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "gguf-qwen3-tokenizer" => "Qwen/Qwen3-0.6B",
        _ => bail!("unknown checkpoint case {name}"),
    };
    ensure!(
        entry["repository"] == repository,
        "wrong repository for {name}"
    );
    let file = match name {
        "gguf-tinyllama-q4_0" => Some("tinyllama-1.1b-chat-v1.0.Q4_0.gguf"),
        "gguf-qwen3-q8_0" => Some("Qwen3-0.6B-Q8_0.gguf"),
        _ => None,
    };
    if let Some(file) = file {
        ensure!(
            !entry["files"][file].is_null(),
            "missing required GGUF file"
        );
    }
    Ok(())
}

pub fn read_manifest(path: &Path) -> Result<Value> {
    let manifest: Value = serde_json::from_slice(&fs::read(path)?)?;
    ensure!(manifest["schema_version"] == 1, "requires schema_version 1");
    ensure!(
        manifest["devices"] == json!(["cpu", "metal"]),
        "both devices must be explicit: cpu metal"
    );
    for name in CASES
        .into_iter()
        .chain(["gguf-tinyllama-tokenizer", "gguf-qwen3-tokenizer"])
    {
        verify_identity(name, &manifest["entries"][name])?;
        verify_entry(&manifest["entries"][name]).with_context(|| name.to_owned())?;
    }
    for name in CASES {
        let case = &manifest["cases"][name];
        ensure!(
            case["max_tokens"] == 32 && case["stop_token_ids"] == json!([]),
            "invalid fixed-count case {name}"
        );
        ensure!(
            case["tokenizer_entry"] == tokenizer_entry(name),
            "wrong tokenizer entry {name}"
        );
        ensure!(!ids(&case["prompt_ids"])?.is_empty(), "empty prompt {name}");
        ensure!(
            case["prefill_chunk_size"].as_u64().is_some_and(|n| n > 0),
            "missing chunk size"
        );
    }
    Ok(manifest)
}

pub fn ids(value: &Value) -> Result<Vec<u32>> {
    Ok(serde_json::from_value(value.clone())?)
}

pub fn contract(manifest: &Value) -> Value {
    let mut result = serde_json::Map::new();
    for key in [
        "schema_version",
        "entries",
        "cases",
        "devices",
        "python_environment",
    ] {
        result.insert(key.into(), manifest[key].clone());
    }
    Value::Object(result)
}

pub fn load_case(manifest: &Value, name: &str) -> Result<Model> {
    let entry = &manifest["entries"][name];
    let root = Path::new(string(&entry["local_directory"])?);
    if name.starts_with("gguf-") {
        let files = entry["files"].as_object().context("missing files")?;
        let names: Vec<_> = files
            .keys()
            .filter(|name| name.ends_with(".gguf"))
            .collect();
        ensure!(names.len() == 1, "requires exactly one GGUF file");
        let tokenizer = Tokenizer::from_dir(string(
            &manifest["entries"][tokenizer_entry(name)]["local_directory"],
        )?)?;
        let file = with_device(Device::cpu(), || {
            mlx_rs::io::GgufFile::load(root.join(names[0]))
        })?;
        Ok(Model::from_gguf(file, tokenizer)?)
    } else {
        Ok(Model::from_dir(root)?)
    }
}

pub fn tokenization(tokenizer: &Tokenizer, text: &str) -> Result<Value> {
    let encoded = tokenizer.encode(text)?;
    let special = tokenizer
        .bos_token()
        .is_none_or(|bos| !text.starts_with(bos));
    Ok(
        json!({"encoded_ids": raw_ids(&encoded), "decoded_text": tokenizer.decode(&encoded)?,
              "prompt_ids": raw_ids(&tokenizer.encode_with_special_tokens(text, special)?)}),
    )
}

pub fn raw_ids(ids: &[TokenId]) -> Vec<u32> {
    ids.iter().copied().map(u32::from).collect()
}

pub fn config_value(config: &Config) -> Value {
    let d = &config.dimensions;
    let scaling = match config.rope.scaling {
        RopeScaling::None => json!({"type": "none"}),
        RopeScaling::Linear { factor } => json!({"type": "linear", "factor": factor as f64}),
        RopeScaling::Llama3 {
            factor,
            low_frequency_factor,
            high_frequency_factor,
            original_max_positions,
        } => json!({
            "type": "llama3", "factor": factor as f64, "low_freq_factor": low_frequency_factor as f64,
            "high_freq_factor": high_frequency_factor as f64, "original_max_position_embeddings": original_max_positions}),
    };
    let attention: Vec<_> = config
        .attention
        .iter()
        .map(|a| match a {
            AttentionKind::Full => json!({"type": "full"}),
            AttentionKind::Sliding { window } => json!({"type": "sliding", "window": window.get()}),
        })
        .collect();
    let quantization = config.quantization.as_ref().map(|q| {
        let layers: BTreeMap<_, _> = q.layers.iter().map(|(path, setting)| {
            let path = path.as_str();
            let path = if config.model_type.as_str() == "qwen3" && path != "lm_head" && !path.starts_with("model.") { format!("model.{path}") } else { path.to_owned() };
            let value = match setting {
                LayerQuantization::Unquantized => json!(false),
                LayerQuantization::Affine(q) => json!({"group_size": q.group_size.get(), "bits": q.bits}),
            };
            (path, value)
        }).collect();
        json!({"default": {"group_size": q.default.group_size.get(), "bits": q.default.bits}, "layers": layers})
    });
    json!({"model_type": config.model_type.as_str(), "hidden_size": d.hidden_size,
        "layer_count": d.layer_count, "intermediate_size": d.intermediate_size,
        "attention_heads": d.attention_heads, "kv_heads": d.kv_heads, "head_dim": d.head_dim,
        "vocabulary_size": d.vocabulary_size, "max_positions": d.max_positions,
        "rms_norm_epsilon": d.rms_norm_epsilon as f64,
        "rope": {"dimensions": config.rope.dimensions, "theta": config.rope.theta as f64,
                 "traditional": config.rope.traditional, "scaling": scaling},
        "attention": attention, "tie_word_embeddings": config.tie_word_embeddings,
        "attention_bias": config.attention_bias, "mlp_bias": config.mlp_bias, "quantization": quantization})
}

pub fn options(count: usize, chunk: usize) -> Result<GenerationOptions> {
    let mut options = GenerationOptions::default();
    options.max_tokens = NonZeroUsize::new(count).context("zero token count")?;
    options.prefill_chunk_size = NonZeroUsize::new(chunk).context("zero chunk size")?;
    options.stop.tokens = StopTokenPolicy::Exact(Vec::new());
    options.cache.policy = CachePolicy::Full;
    Ok(options)
}

#[derive(serde::Deserialize)]
struct GreedyStep {
    top1_id: u32,
    top2_id: u32,
    top1_logit: f64,
    gap: f64,
    dtype: String,
}

fn dtype_ulp(dtype: &str, value: f64) -> Result<f64> {
    ensure!(value.is_finite(), "non-finite top-1 logit");
    let (fraction_bits, min_exponent, max_value) = match dtype {
        "bfloat16" => (7, -126, f64::from(f32::from_bits(0x7f7f0000))),
        "float16" => (10, -14, 65504.0),
        "float32" => (23, -126, f64::from(f32::MAX)),
        _ => bail!("unsupported model logits dtype: {dtype}"),
    };
    ensure!(
        value.abs() <= max_value,
        "top-1 logit outside {dtype} range"
    );
    let exponent = ((value.abs().to_bits() >> 52) & 0x7ff) as i32 - 1023;
    Ok(2.0_f64.powi(exponent.max(min_exponent) - fraction_bits))
}

fn compare_greedy(generated: &[u32], expected: &Value, device: &str) -> Result<String> {
    ensure!(matches!(device, "cpu" | "metal"), "unknown device {device}");
    let python = ids(&expected["greedy_ids"])?;
    let steps: Vec<GreedyStep> = serde_json::from_value(expected["greedy_steps"].clone())?;
    ensure!(
        generated.len() == 32 && python.len() == 32 && steps.len() == 32,
        "requires 32 greedy IDs and step records"
    );
    let mut ulps = Vec::with_capacity(32);
    for (index, step) in steps.iter().enumerate() {
        ensure!(
            step.top1_id == python[index] && step.top1_id != step.top2_id,
            "invalid top-1/top-2 IDs at step {index}"
        );
        ensure!(
            step.gap.is_finite() && step.gap >= 0.0,
            "invalid gap at step {index}"
        );
        ulps.push(dtype_ulp(&step.dtype, step.top1_logit)?);
    }
    for (index, ((rust, step), ulp)) in generated.iter().zip(&steps).zip(ulps).enumerate() {
        if device == "metal" && step.gap <= ulp {
            ensure!(*rust == step.top1_id || *rust == step.top2_id,
                "greedy choice outside recorded top-2 at step {index}: Rust={rust} top1={} top2={}, gap {}",
                step.top1_id, step.top2_id, step.gap);
            return Ok(format!("tie-limited at step {index}, gap {}", step.gap));
        }
        ensure!(
            *rust == step.top1_id,
            "greedy IDs differ at step {index}: Rust={rust} Python={}, gap {}, {}/ulp {ulp}",
            step.top1_id,
            step.gap,
            step.dtype
        );
    }
    Ok("exact equality for 32 greedy IDs".into())
}

fn run_every_case(
    mut run: impl FnMut(&str, &str) -> Result<String>,
    mut report: impl FnMut(&str, &str, &Result<String>),
) -> Result<()> {
    let mut failures = Vec::new();
    for name in CASES {
        for device in ["cpu", "metal"] {
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(name, device)))
                    .unwrap_or_else(|panic| {
                        let message = panic
                            .downcast_ref::<String>()
                            .map(String::as_str)
                            .or_else(|| panic.downcast_ref::<&str>().copied())
                            .unwrap_or("non-string panic");
                        Err(anyhow::anyhow!("case panicked: {message}"))
                    });
            report(name, device, &result);
            if let Err(error) = result {
                failures.push(format!("{name}/{device}: {error:#}"));
            }
        }
    }
    ensure!(
        failures.is_empty(),
        "{} case/device failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
    Ok(())
}

pub fn run_matrix(manifest: &Value) -> Result<()> {
    let report_path = Path::new(string(&manifest["expected_report"]["path"])?);
    ensure!(report_path.is_absolute(), "report path must be absolute");
    verify_hash(report_path, &manifest["expected_report"]["sha256"])?;
    let report: Value = serde_json::from_slice(&fs::read(report_path)?)?;
    ensure!(
        report["schema_version"] == 1 && report["contract"] == contract(manifest),
        "Python report contract mismatch"
    );
    ensure!(
        report["environment"] == manifest["python_environment"]
            && report["environment"]["mlx_lm"] == "0.31.3"
            && report["environment"]["mlx"] == "0.32.2",
        "Python oracle pin mismatch"
    );
    run_every_case(
        |name, device| {
            with_device(
                if device == "cpu" {
                    Device::cpu()
                } else {
                    Device::gpu()
                },
                || -> Result<String> {
                    let expected = &report["results"][name][device];
                    ensure!(
                        expected["completed"] == true,
                        "Python case incomplete: {}",
                        expected["error"]
                    );
                    let provenance = &expected["provenance"];
                    let kind = if name.starts_with("gguf-") {
                        "actual_gguf_reviewed_bridge"
                    } else {
                        "native_safetensors"
                    };
                    ensure!(provenance["kind"] == kind, "wrong Python provenance");
                    if name.starts_with("gguf-") {
                        let file = string(&provenance["gguf_file"])?;
                        ensure!(
                            provenance["gguf_sha256"] == manifest["entries"][name]["files"][file]
                                && !provenance["gguf_sha256"].is_null(),
                            "GGUF source hash mismatch"
                        );
                        verify_hash(
                            &Path::new(env!("CARGO_MANIFEST_DIR"))
                                .join("../conformance/mlx-lm/gguf_bridge.py"),
                            &provenance["bridge_sha256"],
                        )?;
                        ensure!(
                            provenance["tokenizer_entry"] == tokenizer_entry(name),
                            "GGUF tokenizer provenance mismatch"
                        );
                    }
                    let case = &manifest["cases"][name];
                    let mut model = load_case(manifest, name)?;
                    let tokens = tokenization(model.tokenizer(), string(&case["canonical_text"])?)?;
                    ensure!(
                        tokens["decoded_text"] == case["canonical_text"],
                        "canonical round trip differs"
                    );
                    ensure!(
                        tokens == expected["tokenization"]
                            && tokens["prompt_ids"] == case["prompt_ids"],
                        "exact tokenizer/prompt IDs differ"
                    );
                    ensure!(
                        config_value(model.config()) == expected["resolved_config"],
                        "resolved config differs: Rust={} Python={}",
                        config_value(model.config()),
                        expected["resolved_config"]
                    );
                    let prompt: Vec<_> = ids(&case["prompt_ids"])?
                        .into_iter()
                        .map(TokenId::from)
                        .collect();
                    let mut generated = Vec::new();
                    let mut completed = false;
                    let mut generation = model.generate(
                        Prompt::Text(string(&case["canonical_text"])?),
                        options(
                            32,
                            case["prefill_chunk_size"].as_u64().context("chunk")? as usize,
                        )?,
                    )?;
                    for event in generation.by_ref() {
                        if let GenerationEvent::Token {
                            token_id,
                            finish_reason,
                            ..
                        } = event?
                        {
                            ensure!(!completed, "token after completion");
                            generated.push(u32::from(token_id));
                            completed = finish_reason == Some(FinishReason::Length);
                            ensure!(finish_reason.is_none() || completed, "unexpected stop");
                        }
                    }
                    ensure!(
                        generation.cache().tokens().get(..prompt.len()) == Some(prompt.as_slice()),
                        "generation's original-string prompt IDs differ"
                    );
                    ensure!(
                        completed && generated.len() == 32 && expected["completed"] == true,
                        "incomplete generation"
                    );
                    compare_greedy(&generated, expected, device)
                },
            )
        },
        |name, device, result| match result {
            Ok(detail) => println!("PASS: {name}/{device}: {detail}"),
            Err(error) => println!("FAIL: {name}/{device}: {error:#}"),
        },
    )
}

#[test]
fn real_checkpoints() -> Result<()> {
    let Some(path) = std::env::var_os("MLX_LM_REAL_CHECKPOINTS") else {
        println!("NOT RUN: local real-checkpoint manifest not supplied");
        return Ok(());
    };
    refuse_ci()?;
    run_matrix(&read_manifest(Path::new(&path))?)
}

#[test]
fn pure_manifest_rejects_missing_hash_changed_bytes_and_unlisted_files() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("config.json");
    fs::write(&path, "{}")?;
    let mut entry = json!({"repository": "owner/model", "revision": "a".repeat(40),
        "local_directory": dir.path(), "files": {"config.json": sha256(&path)?}, "bytes": 2});
    verify_entry(&entry)?;
    entry["files"]["config.json"] = Value::Null;
    ensure!(verify_entry(&entry).is_err());
    entry["files"]["config.json"] = json!(sha256(&path)?);
    fs::write(&path, "[]")?;
    ensure!(verify_entry(&entry).is_err());
    entry["files"]["config.json"] = json!(sha256(&path)?);
    fs::write(dir.path().join("tokenizer.json"), "{}")?;
    ensure!(verify_entry(&entry).is_err());
    Ok(())
}

#[test]
fn pure_manifest_rejects_remote_refs_and_unsafe_paths() -> Result<()> {
    ensure!(!hex("main", 40));
    ensure!(verify_identity(
        "qwen3-06b-4bit",
        &json!({"repository": "owner/replacement"})
    )
    .is_err());
    for name in ["../config.json", "/config.json", "a/../config.json", "a\\b"] {
        if relative(name).is_ok() {
            bail!("admitted unsafe path {name}");
        }
    }
    Ok(())
}

#[test]
fn pure_report_missing_or_unhashed_is_failure() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("python-report.json");
    let mut manifest = json!({"expected_report": {"path": path, "sha256": "0".repeat(64)}});
    ensure!(run_matrix(&manifest).is_err());
    fs::write(&path, "{}")?;
    manifest["expected_report"]["sha256"] = Value::Null;
    ensure!(run_matrix(&manifest).is_err());
    manifest["expected_report"]["sha256"] = json!("0".repeat(64));
    ensure!(run_matrix(&manifest).is_err());
    Ok(())
}

#[test]
fn pure_greedy_tie_contract() -> Result<()> {
    let mut expected = json!({
        "greedy_ids": vec![832; 32],
        "greedy_steps": vec![json!({"top1_id": 832, "top2_id": 264,
            "top1_logit": 20.875, "gap": 0.25, "dtype": "bfloat16"}); 32]
    });
    let mut generated = vec![832; 32];
    ensure!(compare_greedy(&generated, &expected, "cpu")?.contains("exact"));
    ensure!(compare_greedy(&generated, &expected, "metal")?.contains("exact"));
    generated[18] = 264;
    ensure!(compare_greedy(&generated, &expected, "metal").is_err());
    expected["greedy_steps"][18]["gap"] = json!(0.125);
    generated[19] = 999;
    ensure!(compare_greedy(&generated, &expected, "metal")? == "tie-limited at step 18, gap 0.125");
    ensure!(compare_greedy(&generated, &expected, "cpu").is_err());
    generated[18] = 999;
    ensure!(compare_greedy(&generated, &expected, "metal").is_err());
    generated[18] = 832;
    ensure!(compare_greedy(&generated, &expected, "cpu").is_err());
    ensure!(compare_greedy(&[832; 32], &expected, "cpu")?.contains("exact"));
    ensure!(compare_greedy(&generated, &expected, "metal")?.contains("step 18"));
    expected["greedy_steps"][18]["gap"] = json!(0.0);
    ensure!(compare_greedy(&generated, &expected, "metal")?.contains("step 18"));
    expected["greedy_steps"][18]["gap"] = json!(0.125);
    for dtype in ["float16", "float32"] {
        expected["greedy_steps"][18]["dtype"] = json!(dtype);
        generated[18] = 264;
        ensure!(compare_greedy(&generated, &expected, "metal").is_err());
    }
    Ok(())
}

#[test]
fn pure_matrix_reports_every_case_after_errors_and_panics() {
    let mut reports = Vec::new();
    let result = run_every_case(
        |name, device| {
            if name == CASES[0] && device == "cpu" {
                bail!("injected failure");
            }
            if name == CASES[2] && device == "metal" {
                panic!("injected panic");
            }
            Ok("exact".into())
        },
        |name, device, result| reports.push((name.to_owned(), device.to_owned(), result.is_ok())),
    );
    assert!(result.is_err());
    assert_eq!(reports.len(), 12);
    assert_eq!(reports.iter().filter(|r| !r.2).count(), 2);
    assert_eq!(
        reports.last(),
        Some(&(CASES[5].into(), "metal".into(), true))
    );
    assert!(run_every_case(|_, _| Ok("exact".into()), |_, _, _| {}).is_ok());
}

#[test]
fn pure_ulp_uses_dtype_magnitude_and_subnormal_spacing() -> Result<()> {
    for (dtype, value, expected) in [
        ("bfloat16", 20.875, 0.125),
        ("float16", 20.875, 0.015625),
        ("float32", 20.875, 0.0000019073486328125),
        ("bfloat16", -20.875, 0.125),
        ("bfloat16", 16.0, 0.125),
        ("bfloat16", 15.9375, 0.0625),
        ("float16", 0.0, (1.0 / 16_777_216.0)),
        ("float16", -(1.0 / 16_777_216.0), (1.0 / 16_777_216.0)),
    ] {
        ensure!(
            dtype_ulp(dtype, value)? == expected,
            "wrong ULP for {dtype}/{value}"
        );
    }
    ensure!(dtype_ulp("uint32", 20.0).is_err());
    ensure!(dtype_ulp("bfloat16", f64::NAN).is_err());
    ensure!(dtype_ulp("float16", 65536.0).is_err());
    Ok(())
}
