use serde_json::{json, Value};
use std::{fs, path::Path, process::Command};

// Reviewed feature-qualified inventory v1. These are external Rust consumers, not
// source enumeration: rustc checks every declaration even though main is empty.
const COMMON: &str = r#"
#![allow(dead_code, unused_imports, unreachable_patterns)]
use mlx_lm::{
    Cache, CacheInfo, CacheKind, CacheOptions, CachePolicy, CacheSnapshot,
    AffineQuantization, AttentionKind, Config, LayerQuantization, ModelType,
    ParameterPath, QuantizationConfig, RopeConfig, RopeScaling, TransformerDimensions,
    CacheError, ChatTemplateError, ConfigError, GenerationError, HubError,
    InferenceError, LoadError, SamplingError, TokenizerError, WeightError,
    FinishReason, Generation, GenerationEvent, GenerationOptions, Model, Prompt,
    RepetitionPenaltyOptions, StopPolicy, StopTokenPolicy, AdditivePenaltyOptions,
    MinPOptions, SamplerOptions, ChatContinuation, ChatTemplateOptions, Message,
    Role, TokenId, Tokenizer,
};
use std::{collections::BTreeMap, num::NonZeroUsize, ops::Range, path::{Path, PathBuf}};
fn config(c: Config) {
    let _: ModelType = c.model_type;
    let _: Vec<AttentionKind> = c.attention;
    let _: (bool, bool, bool) = (c.tie_word_embeddings, c.attention_bias, c.mlp_bias);
    let _: Option<QuantizationConfig> = c.quantization;
    let d: TransformerDimensions = c.dimensions;
    let _: (usize, usize, usize, usize, usize, usize, usize) =
        (d.hidden_size, d.layer_count, d.intermediate_size, d.attention_heads,
         d.kv_heads, d.head_dim, d.vocabulary_size);
    let _: Option<usize> = d.max_positions;
    let _: f32 = d.rms_norm_epsilon;
    let r: RopeConfig = c.rope;
    let _: usize = r.dimensions;
    let _: f32 = r.theta;
    let _: bool = r.traditional;
    match r.scaling {
        RopeScaling::None => (),
        RopeScaling::Linear { factor } => { let _: f32 = factor; },
        RopeScaling::Llama3 { factor, low_frequency_factor, high_frequency_factor, original_max_positions } => {
            let _: (f32, f32, f32, usize) = (factor, low_frequency_factor, high_frequency_factor, original_max_positions);
        }
    }
}
fn quantization(q: QuantizationConfig, a: AttentionKind) {
    let _: BTreeMap<ParameterPath, LayerQuantization> = q.layers;
    let _: NonZeroUsize = q.default.group_size;
    let _: u8 = q.default.bits;
    match LayerQuantization::Affine(q.default) {
        LayerQuantization::Unquantized => (),
        LayerQuantization::Affine(v) => { let _: AffineQuantization = v; }
    }
    match a { AttentionKind::Full => (), AttentionKind::Sliding { window } => { let _: NonZeroUsize = window; } }
}
fn generation_options(o: GenerationOptions) {
    let _: NonZeroUsize = o.max_tokens;
    let _: NonZeroUsize = o.prefill_chunk_size;
    let _: Option<RepetitionPenaltyOptions> = o.repetition_penalty;
    let _: Option<AdditivePenaltyOptions> = o.presence_penalty;
    let _: Option<AdditivePenaltyOptions> = o.frequency_penalty;
    let _: CacheOptions = o.cache;
    let _: Vec<String> = o.stop.stop_strings;
    match o.stop.tokens {
        StopTokenPolicy::Tokenizer => (),
        StopTokenPolicy::TokenizerPlus(ids) | StopTokenPolicy::Exact(ids) => { let _: Vec<TokenId> = ids; }
    }
    let s: SamplerOptions = o.sampling;
    let _: f32 = s.temperature;
    let _: Option<f32> = s.top_p;
    let _: Option<NonZeroUsize> = s.top_k;
    let _: Option<u64> = s.seed;
    if let Some(p) = s.min_p {
        let _: f32 = p.probability;
        let _: NonZeroUsize = p.min_tokens_to_keep;
    }
}
fn penalties(r: RepetitionPenaltyOptions, a: AdditivePenaltyOptions) {
    let _: (f32, NonZeroUsize) = (r.penalty, r.context_size);
    let _: (f32, NonZeroUsize) = (a.penalty, a.context_size);
    let _: Result<(), SamplingError> = r.validate();
    let _: Result<(), SamplingError> = a.validate();
}
fn events(e: GenerationEvent) {
    match e {
        GenerationEvent::Prefill { processed, total, .. } => { let _: (usize, usize) = (processed, total); }
        GenerationEvent::Token { token_id, text, finish_reason, .. } => {
            let _: (TokenId, String) = (token_id, text);
            let _: Option<FinishReason> = finish_reason;
            if let Some(reason) = finish_reason { match reason { FinishReason::Stop | FinishReason::Length => (), _ => () } }
        }
        _ => ()
    }
}
fn cache(c: &mut Cache, s: CacheSnapshot, o: CacheOptions, i: CacheInfo) {
    let _: &[TokenId] = c.tokens();
    let _: Result<CacheSnapshot, CacheError> = c.snapshot();
    let _: Result<(), CacheError> = c.restore(s);
    let _: usize = c.info().len();
    let _: (usize, usize, usize) = (i.layer, i.processed_tokens, i.capacity);
    let _: (Range<usize>, Range<usize>) = (i.retained_prefix, i.retained_positions);
    match i.kind { CacheKind::Full | CacheKind::Rotating => () }
    match o.policy {
        CachePolicy::ModelDefault | CachePolicy::Full => (),
        CachePolicy::Rotating { capacity, keep_prefix } => { let _: (NonZeroUsize, usize) = (capacity, keep_prefix); }
    }
}
fn tokenizer(t: &Tokenizer, m: Message, o: ChatTemplateOptions) {
    let _: String = m.content;
    match m.role { Role::System | Role::User | Role::Assistant | Role::Tool => (), Role::Other(s) => { let _: String = s; }, _ => () }
    match o.continuation { ChatContinuation::Closed | ChatContinuation::StartAssistant | ChatContinuation::ContinueLast => () }
    let _: Option<bool> = o.enable_thinking;
    let _: Option<&str> = t.bos_token();
    let _: &[TokenId] = t.eos_tokens();
    let _: Result<Vec<TokenId>, TokenizerError> = t.encode("");
    let _: Result<Vec<TokenId>, TokenizerError> = t.encode_with_special_tokens("", true);
    let _: Result<String, TokenizerError> = t.decode(&[]);
    let _: Result<String, ChatTemplateError> = t.render_chat(&[], o);
    let _: Result<Tokenizer, TokenizerError> = Tokenizer::from_dir(Path::new(""));
    let _: Result<Tokenizer, TokenizerError> = Tokenizer::from_file(Path::new(""));
    let _: Result<Tokenizer, TokenizerError> = Tokenizer::from_bytes(&[]);
    let id: TokenId = 1u32.into();
    let _: u32 = id.into();
}
fn model(m: &mut Model, c: &mut Cache) {
    let _: Result<Model, LoadError> = Model::from_dir(Path::new(""));
    let _: fn(mlx_rs::io::GgufFile, Tokenizer) -> Result<Model, LoadError> = Model::from_gguf;
    let _: &Config = m.config();
    let _: &Tokenizer = m.tokenizer();
    let _: Result<Cache, CacheError> = m.new_cache(CacheOptions::default());
    let _: Result<Generation<'_>, GenerationError> = m.generate(Prompt::Text(""), GenerationOptions::default());
    let _: Result<Generation<'_>, GenerationError> = m.generate_with_cache(Prompt::Tokens(&[]), GenerationOptions::default(), c);
}
fn iterator(g: &mut Generation<'_>) {
    let _: Option<Result<GenerationEvent, GenerationError>> = g.next();
    let _: &Cache = g.cache();
    let _: Result<CacheSnapshot, CacheError> = g.snapshot();
}
fn paths() {
    let p = ParameterPath::new("model.layers.0");
    let _: &str = p.as_str();
    let m = ModelType::new("llama");
    let _: &str = m.as_str();
}
fn gguf_errors(c: ConfigError, l: LoadError) {
    match c {
        ConfigError::InvalidGgufMetadata { key, expected, actual } => { let _: (String, &'static str, String) = (key, expected, actual); }
        ConfigError::UnsupportedGgufMetadata { key, value } => { let _: (String, String) = (key, value); }
        _ => ()
    }
    match l {
        LoadError::TokenizerVocabularyOutOfRange { token_id, vocabulary_size } => { let _: (TokenId, usize) = (token_id, vocabulary_size); }
        LoadError::TokenizerMetadataMismatch { key, expected, actual } => { let _: (String, String, String) = (key, expected, actual); }
        _ => ()
    }
}
fn common_hub_errors(e: HubError) {
    match e {
        HubError::OfflineCacheMiss { repo, revision } => { let _: (String, String) = (repo, revision); }
        HubError::InvalidRevision(s) => { let _: String = s; }
        HubError::Io(e) => { let _: std::io::Error = e; }
        HubError::Load(e) => { let _: LoadError = e; }
        _ => ()
    }
}
fn main() {}
"#;

const HUB: &str = r#"
use mlx_lm::{HubOptions, HubProvenance};
fn hub(m: &Model, e: HubError) {
    let mut o = HubOptions::default();
    o.revision = Some(String::from("main"));
    o.offline = true;
    o.cache_dir = Some(PathBuf::from("cache"));
    let _: HubOptions = o.clone();
    let _: Result<Model, HubError> = Model::from_hub("contract/tiny-llama", o);
    let p: Option<&HubProvenance> = m.hub_provenance();
    if let Some(p) = p {
        let _: (&String, &String, &String) = (&p.repo, &p.requested_revision, &p.resolved_revision);
        let _: HubProvenance = p.clone();
        fn traits<T: std::fmt::Debug + Clone + PartialEq + Eq>() {}
        traits::<HubProvenance>();
    }
    match e {
        HubError::Api(source) => { let _: &dyn std::error::Error = &source; }
        _ => ()
    }
}
"#;

// HubError itself is an unconditional root export. Only Api depends on hf-hub.
const HUB_ERRORS: &str = r#"
fn hub_contract_errors(e: HubError) {
    match e {
        HubError::InvalidRepository(s) => { let _: String = s; }
        HubError::MissingFile { repo, revision, filename } => { let _: (String, String, String) = (repo, revision, filename); }
        HubError::UnsafePath { path } => { let _: PathBuf = path; }
        HubError::RevisionMismatch { expected, actual } => { let _: (String, String) = (expected, actual); }
        HubError::SnapshotIntegrity { path, reason } => { let _: (PathBuf, String) = (path, reason); }
        HubError::CacheDirectoryUnavailable => (),
        _ => ()
    }
}
"#;

const HOOKS: &str = r#"
use mlx_lm::oracle_hooks::{self, OracleSession, CacheSnapshotView, LayerView};
fn hooks<'a>(m: &'a mut Model, g: &Generation<'_>, s: &mut OracleSession<'_>, v: CacheSnapshotView, l: LayerView) {
    let _ = oracle_hooks::prefill_logits(m, &[], None);
    let _ = oracle_hooks::first_filtered_logprobs(g);
    let _ = oracle_hooks::generate_with_logprobs;
    let _ = s.decode_step(0u32.into());
    let _: Result<CacheSnapshotView, CacheError> = s.cache_view();
    let _: Vec<LayerView> = v.layers;
    let _: usize = l.layer;
    let _: Range<usize> = l.positions;
    let _: (mlx_rs::Array, mlx_rs::Array) = (l.keys, l.values);
}
"#;

struct Absent {
    name: &'static str,
    source: &'static str,
    code: &'static str,
    feature: &'static str,
}

const ABSENT: &[Absent] = &[
    Absent {
        name: "HubOptions",
        source: "use mlx_lm::HubOptions; fn main() {}",
        code: "E0432",
        feature: "hf-hub",
    },
    Absent {
        name: "HubProvenance",
        source: "use mlx_lm::HubProvenance; fn main() {}",
        code: "E0432",
        feature: "hf-hub",
    },
    Absent {
        name: "from_hub",
        source: "fn main() { let _ = mlx_lm::Model::from_hub; }",
        code: "E0599",
        feature: "hf-hub",
    },
    Absent {
        name: "hub_provenance",
        source: "fn main() { let _ = mlx_lm::Model::hub_provenance; }",
        code: "E0599",
        feature: "hf-hub",
    },
    Absent {
        name: "Api",
        source: "fn main() { let _ = mlx_lm::HubError::Api; }",
        code: "E0599",
        feature: "hf-hub",
    },
    Absent {
        name: "oracle_hooks",
        source: "use mlx_lm::oracle_hooks; fn main() {}",
        code: "E0432",
        feature: "oracle-hooks",
    },
];

fn expected_rejection(output: &std::process::Output, probe: &Absent) -> bool {
    if output.status.success() {
        return false;
    }
    let errors: Vec<Value> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|v| v["reason"] == "compiler-message" && v["message"]["level"] == "error")
        .collect();
    !errors.is_empty()
        && errors.iter().all(|v| {
            let m = &v["message"];
            m["code"]["code"] == probe.code
                && m["message"]
                    .as_str()
                    .is_some_and(|s| s.contains(probe.name))
                && v["target"]["name"] == "lm_feature_consumer"
                && m["spans"].as_array().is_some_and(|spans| {
                    spans.iter().any(|s| {
                        s["is_primary"] == true
                            && s["file_name"]
                                .as_str()
                                .is_some_and(|p| p.ends_with("src/main.rs"))
                    })
                })
        })
}

fn compile(dir: &Path, source: &str) -> Result<std::process::Output, String> {
    fs::write(dir.join("src/main.rs"), source).map_err(|e| e.to_string())?;
    Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
        .args([
            "check",
            "--offline",
            "--message-format=json",
            "--manifest-path",
        ])
        .arg(dir.join("Cargo.toml"))
        .current_dir(dir)
        .output()
        .map_err(|e| e.to_string())
}

fn diagnostics(output: &std::process::Output) -> Vec<String> {
    let mut messages: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter(|v| v["reason"] == "compiler-message" && v["message"]["level"] == "error")
        .filter_map(|v| v["message"]["message"].as_str().map(str::to_owned))
        .collect();
    if messages.is_empty() && !output.status.success() {
        messages.push(String::from_utf8_lossy(&output.stderr).into_owned());
    }
    messages
}

pub fn run(root: &Path, args: &[String]) -> i32 {
    if !args.is_empty() {
        eprintln!("usage: cargo run -p xtask -- verify-lm-features");
        return 2;
    }
    match verify(root) {
        Ok(report) => {
            let passed = report["verdict"] == "pass";
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
            i32::from(!passed)
        }
        Err(error) => {
            println!(
                "{}",
                json!({"schema_version": 1, "verdict": "fail", "error": error})
            );
            1
        }
    }
}

fn verify(root: &Path) -> Result<Value, String> {
    let directory = tempfile::tempdir().map_err(|e| e.to_string())?;
    let dir = directory.path();
    fs::create_dir(dir.join("src")).map_err(|e| e.to_string())?;
    let mut reports = Vec::new();
    for (name, defaults, features) in [
        ("default", true, vec![]),
        ("no-default-features", false, vec![]),
        ("hf-hub", false, vec!["hf-hub"]),
        ("oracle-hooks", false, vec!["oracle-hooks"]),
        ("hf-hub+oracle-hooks", false, vec!["hf-hub", "oracle-hooks"]),
    ] {
        // An isolated workspace prevents feature unification with this xtask's workspace.
        let manifest = format!(
            "[package]\nname = \"lm_feature_consumer\"\nversion = \"0.0.0\"\nedition = \"2021\"\n[workspace]\n[dependencies]\nmlx-lm = {{ path = {}, default-features = {defaults}, features = {} }}\nmlx-rs = {{ path = {} }}\n",
            serde_json::to_string(&root.join("mlx-lm")).map_err(|e| e.to_string())?,
            serde_json::to_string(&features).map_err(|e| e.to_string())?,
            serde_json::to_string(&root.join("mlx-rs")).map_err(|e| e.to_string())?,
        );
        fs::write(dir.join("Cargo.toml"), manifest).map_err(|e| e.to_string())?;
        // First prove the dependency can compile; infrastructure failure cannot qualify absence.
        let control = compile(
            dir,
            "fn main() { let _ = mlx_lm::GenerationOptions::default; }",
        )?;
        let ready = control.status.success();
        reports.push(
            json!({"features": name, "probe": "dependency_control", "passed": ready,
                            "diagnostics": diagnostics(&control)}),
        );
        if !ready {
            continue;
        }
        for (probe, source, enabled) in [
            ("root_exports_fields_variants", COMMON.to_owned(), true),
            (
                "hub_error_additions",
                format!("{COMMON}\n{HUB_ERRORS}"),
                true,
            ),
            (
                "hub_exports_fields_methods",
                format!("{COMMON}\n{HUB}"),
                features.contains(&"hf-hub"),
            ),
            (
                "oracle_hooks_exports",
                format!("{COMMON}\n{HOOKS}"),
                features.contains(&"oracle-hooks"),
            ),
        ] {
            if enabled {
                let output = compile(dir, &source)?;
                reports.push(
                    json!({"features": name, "probe": probe, "passed": output.status.success(),
                                    "diagnostics": diagnostics(&output)}),
                );
            }
        }
        for probe in ABSENT.iter().filter(|p| !features.contains(&p.feature)) {
            let output = compile(dir, probe.source)?;
            reports.push(
                json!({"features": name, "probe": format!("absent::{}", probe.name),
                                "passed": expected_rejection(&output, probe),
                                "diagnostics": diagnostics(&output)}),
            );
        }
    }
    let passed = reports.iter().all(|r| r["passed"] == true);
    Ok(
        json!({"schema_version": 1, "inventory": "reviewed-consumer-v1", "evidence": "cargo-check-external-consumer",
              "verdict": if passed { "pass" } else { "fail" }, "probes": reports,
              "scope": "All reviewed root exports; public data fields and named variants in the static consumer inventory. Supplements api-baseline, which does not evaluate cfg. Not enumeration of every possible public path."}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::process::ExitStatusExt;

    fn output(success: bool, messages: Vec<Value>) -> std::process::Output {
        std::process::Output {
            status: std::process::ExitStatus::from_raw(if success { 0 } else { 256 }),
            stdout: messages
                .iter()
                .map(|m| format!("{m}\n"))
                .collect::<String>()
                .into_bytes(),
            stderr: Vec::new(),
        }
    }

    fn missing() -> Value {
        json!({"reason": "compiler-message", "target": {"name": "lm_feature_consumer"},
               "message": {"level": "error", "code": {"code": "E0432"},
               "message": "unresolved import mlx_lm::HubOptions", "spans": [{"is_primary": true, "file_name": "src/main.rs"}]}})
    }

    #[test]
    fn absence_requires_the_named_consumer_diagnostic() {
        assert!(expected_rejection(
            &output(false, vec![missing()]),
            &ABSENT[0]
        ));
        assert!(!expected_rejection(
            &output(true, vec![missing()]),
            &ABSENT[0]
        ));
        assert!(!expected_rejection(&output(false, vec![]), &ABSENT[0]));
        let mut error = missing();
        error["message"]["code"]["code"] = json!("E0308");
        assert!(!expected_rejection(&output(false, vec![error]), &ABSENT[0]));
        let mut error = missing();
        error["target"]["name"] = json!("mlx_lm");
        assert!(!expected_rejection(&output(false, vec![error]), &ABSENT[0]));
        let mut error = missing();
        error["message"]["message"] = json!("unresolved import unrelated");
        assert!(!expected_rejection(
            &output(false, vec![missing(), error]),
            &ABSENT[0]
        ));
    }
}
