use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
};

use mlx_lm::{FinishReason, GenerationEvent, GenerationOptions, Model, Prompt};
use serde_json::{json, Value};
use tempfile::TempDir;

const PROMPT: &str = "hello the small fox runs over green hill";

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find(|path| path.join("conformance/mlx-lm/fixtures").is_dir())
        .unwrap()
        .to_owned()
}
fn fixture(name: &str) -> PathBuf {
    repository().join("conformance/mlx-lm/fixtures").join(name)
}

struct Process {
    command: Command,
    _cache: TempDir,
}
impl Process {
    fn new() -> Self {
        let cache = tempfile::tempdir().unwrap();
        let mut command = Command::new(env!("CARGO_BIN_EXE_mlx-lm"));
        command
            .stdin(Stdio::null())
            .env("HF_HOME", cache.path())
            .env("HF_HUB_CACHE", cache.path().join("hub"))
            .env("HUGGINGFACE_HUB_CACHE", cache.path().join("hub"))
            .env("TRANSFORMERS_CACHE", cache.path().join("transformers"))
            .env("HF_HUB_OFFLINE", "1")
            .env("TRANSFORMERS_OFFLINE", "1")
            .env("HF_ENDPOINT", "http://127.0.0.1:9")
            .env_remove("HF_TOKEN")
            .env_remove("HUGGING_FACE_HUB_TOKEN");
        Self {
            command,
            _cache: cache,
        }
    }
    fn generate(&mut self, name: &str, format: &str) -> &mut Command {
        self.command
            .args(["generate", "--model"])
            .arg(fixture(name))
            .args(["--prompt", PROMPT, "--max-tokens", "8", "--format", format])
    }
}
fn success(output: &Output) {
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
fn failure(output: &Output, code: i32, diagnostic: &str) {
    assert_eq!(output.status.code(), Some(code), "{output:?}");
    assert!(output.stdout.is_empty(), "stdout: {:?}", output.stdout);
    assert!(
        String::from_utf8_lossy(&output.stderr).contains(diagnostic),
        "{output:?}"
    );
}

fn public_records(name: &str) -> Vec<Value> {
    let mut model = Model::from_dir(fixture(name)).unwrap();
    let mut options = GenerationOptions::default();
    options.max_tokens = 8.try_into().unwrap();
    model.generate(Prompt::Text(PROMPT), options).unwrap().map(|event| match event.unwrap() {
        GenerationEvent::Prefill { processed, total, .. } => json!({"version":1,"event":"prefill","processed":processed,"total":total}),
        GenerationEvent::Token { token_id, text, finish_reason, .. } => {
            let finish = match finish_reason {
                None => None,
                Some(FinishReason::Stop) => Some("stop"),
                Some(FinishReason::Length) => Some("length"),
                Some(other) => panic!("unknown finish: {other:?}"),
            };
            json!({"version":1,"event":"token","token_id":u32::from(token_id),"text":text,"finish_reason":finish})
        }
        other => panic!("unknown event: {other:?}"),
    }).collect()
}

#[test]
fn local_llama_text_generation() {
    let expected: String = public_records("llama-base")
        .iter()
        .filter_map(|record| record["text"].as_str())
        .collect();
    assert!(!expected.is_empty());
    let output = Process::new()
        .generate("llama-base", "text")
        .output()
        .unwrap();
    success(&output);
    assert_eq!(output.stdout, expected.as_bytes());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Prefill:"));
}

#[test]
fn local_llama_jsonl_generation() {
    let output = Process::new()
        .generate("llama-base", "jsonl")
        .output()
        .unwrap();
    success(&output);
    assert!(output.stderr.is_empty(), "{output:?}");
    assert!(output.stdout.ends_with(b"\n"));
    let records: Vec<Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| {
            assert!(line.starts_with("{\"version\":1,\"event\":"));
            serde_json::from_str(line).unwrap()
        })
        .collect();
    assert_eq!(records, public_records("llama-base"));
    assert_eq!(
        records[0],
        json!({"version":1,"event":"prefill","processed":0,"total":9})
    );
    assert_eq!(
        records
            .iter()
            .filter(|record| !record["finish_reason"].is_null())
            .count(),
        1
    );
}

#[test]
fn local_qwen3_quantized_info() {
    let output = Process::new()
        .command
        .args(["info", "--model"])
        .arg(fixture("qwen3-quant4"))
        .args(["--format", "json"])
        .output()
        .unwrap();
    success(&output);
    assert!(output.stderr.is_empty());
    let actual: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(actual.as_object().unwrap().len(), 4);
    assert_eq!(actual["version"], 1);
    assert!(actual["hub_provenance"].is_null());
    assert_eq!(
        actual["tokenizer"],
        json!({"bos_token":null,"eos_tokens":[8,9]})
    );
    let config = &actual["config"];
    assert_eq!(config.as_object().unwrap().len(), 8);
    assert_eq!(config["model_type"], "qwen3");
    assert_eq!(
        config["dimensions"],
        json!({
            "hidden_size":64,"layer_count":2,"intermediate_size":64,"attention_heads":4,
            "kv_heads":2,"head_dim":16,"vocabulary_size":64,"max_positions":128,"rms_norm_epsilon":1e-5_f32,
        })
    );
    assert_eq!(
        config["rope"],
        json!({"dimensions":16,"theta":10000.0,"traditional":false,"scaling":{"kind":"none"}})
    );
    assert_eq!(
        config["attention"],
        json!([{"kind":"full"},{"kind":"full"}])
    );
    assert_eq!(config["tie_word_embeddings"], true);
    assert_eq!(config["attention_bias"], false);
    assert_eq!(config["mlp_bias"], false);
    assert_eq!(
        config["quantization"],
        json!({"default":{"bits":4,"group_size":32},"layers":{}})
    );
}

#[test]
fn sharded_load() {
    let output = Process::new()
        .generate("llama-sharded", "jsonl")
        .output()
        .unwrap();
    success(&output);
    assert!(output.stderr.is_empty());
    let records: Vec<Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(records, public_records("llama-sharded"));
}

#[test]
fn invalid_options_rejected_before_load() {
    for flags in [
        vec!["--max-tokens", "0"],
        vec!["--temperature", "NaN"],
        vec!["--top-p", "0"],
        vec!["--top-k", "0"],
        vec!["--min-p", "2"],
        vec!["--min-tokens-to-keep", "1"],
        vec!["--seed", "-1"],
        vec!["--stop", ""],
    ] {
        let mut process = Process::new();
        let missing = process._cache.path().join("does-not-exist");
        let output = process
            .command
            .args(["generate", "--model"])
            .arg(missing)
            .args(["--prompt", "x"])
            .args(&flags)
            .output()
            .unwrap();
        failure(&output, 2, "error:");
        assert!(!String::from_utf8_lossy(&output.stderr).contains("model load failed"));
    }
}

#[test]
fn missing_model_files() {
    for command in ["info", "generate"] {
        let mut process = Process::new();
        process
            .command
            .args([command, "--model"])
            .arg(process._cache.path());
        if command == "generate" {
            process.command.args(["--prompt", "x"]);
        }
        let output = process.command.output().unwrap();
        failure(&output, 1, "model load failed");
        assert!(String::from_utf8_lossy(&output.stderr).contains("caused by:"));
    }
}

#[test]
fn missing_weights_after_valid_sidecars() {
    let directory = tempfile::tempdir().unwrap();
    for name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
    ] {
        fs::copy(
            fixture("llama-base").join(name),
            directory.path().join(name),
        )
        .unwrap();
    }
    let output = Process::new()
        .command
        .args(["info", "--model"])
        .arg(directory.path())
        .output()
        .unwrap();
    failure(&output, 1, "model load failed");
}

#[test]
fn help_and_syntax_keep_stdout_clean() {
    for args in [
        vec!["--help"],
        vec!["generate", "--help"],
        vec!["info", "--help"],
        vec!["--version"],
    ] {
        let output = Process::new().command.args(args).output().unwrap();
        success(&output);
        assert!(output.stdout.is_empty());
        assert!(!output.stderr.is_empty());
    }
    for args in [
        vec![],
        vec!["serve"],
        vec!["info"],
        vec!["generate", "--model", "x"],
    ] {
        let output = Process::new().command.args(args).output().unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(output.stdout.is_empty());
        assert!(!output.stderr.is_empty());
    }
}

#[cfg(not(feature = "hf-hub"))]
#[test]
fn disabled_hub_grammar() {
    for args in [
        vec!["info", "--repo", "org/model"],
        vec!["info", "--model", "x", "--offline"],
        vec!["info", "--model", "x", "--revision", "main"],
        vec!["info", "--model", "x", "--cache-dir", "cache"],
    ] {
        let output = Process::new().command.args(args).output().unwrap();
        failure(&output, 2, "unexpected argument");
    }
}

#[cfg(feature = "hf-hub")]
#[test]
fn enabled_offline_hub_miss() {
    let mut process = Process::new();
    let output = process
        .command
        .args(["info", "--repo", "org/model", "--offline", "--cache-dir"])
        .arg(process._cache.path())
        .args(["--format", "json"])
        .output()
        .unwrap();
    failure(&output, 1, "offline cache miss for org/model@main");
}

#[cfg(unix)]
#[test]
fn early_broken_pipe() {
    use std::{
        os::{fd::OwnedFd, unix::net::UnixStream},
        time::{Duration, Instant},
    };
    for format in ["text", "jsonl"] {
        let (reader, writer) = UnixStream::pair().unwrap();
        drop(reader);
        let mut process = Process::new();
        let command = process.generate("llama-base", format);
        let mut child = command
            .stdout(Stdio::from(OwnedFd::from(writer)))
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if child.try_wait().unwrap().is_some() {
                break;
            }
            if Instant::now() >= deadline {
                child.kill().unwrap();
                let output = child.wait_with_output().unwrap();
                panic!("BrokenPipe did not cancel generation: {output:?}");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let output = child.wait_with_output().unwrap();
        success(&output);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!stderr.contains("output failed"));
        assert!(!stderr.contains("Broken pipe"));
        if format == "jsonl" {
            assert!(stderr.is_empty());
        }
    }
}
