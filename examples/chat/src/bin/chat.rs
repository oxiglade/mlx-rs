//! Interactive REPL against any `mlx_lm` checkpoint. KV cache resets
//! between turns; the full chat history is re-rendered each request.

#![allow(clippy::print_stderr, reason = "CLI binary logs to stderr")]
#![allow(clippy::print_stdout, reason = "CLI binary prints to stdout")]

use std::io::Write;
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use argh::FromArgs;
use chat::think_stream::ThinkStream;
use mlx_lm::cache::{CacheKind, CacheOptions, DEFAULT_KV_GROUP_SIZE};
use mlx_lm::chat_template::ChatMessage;
#[cfg(feature = "audio")]
use mlx_lm::Audio;
use mlx_lm::{
    generate, load_with_drafter, GenerateParams, Image, ModelContext, Sampler, UserInput,
};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

const DEFAULT_MAX_TOKENS: i32 = 1024;

/// Bold cyan readline prompt so the user's input reads distinctly from
/// the model's answer (bold green) and reasoning (dim) — those two are
/// coloured by [`ThinkStream`]. Stats lines use dim.
const PROMPT: &str = "\x1b[1;36m> \x1b[0m";
const C_DIM: &str = "\x1b[2m";
const C_RESET: &str = "\x1b[0m";

/// Interactive REPL against any `mlx_lm` checkpoint.
#[derive(FromArgs)]
struct Args {
    /// path to a loadable model directory (config.json + safetensors)
    #[argh(option)]
    model: PathBuf,

    /// sampling temperature; 0.0 = greedy (default 0.0)
    #[argh(option, default = "0.0")]
    temperature: f32,

    /// nucleus top-p threshold; omit for pure temperature sampling
    #[argh(option, long = "top-p")]
    top_p: Option<f32>,

    /// maximum new tokens per assistant turn (default 1024)
    #[argh(option, default = "DEFAULT_MAX_TOKENS")]
    max_tokens: i32,

    /// thinking mode: on | off | default (template's `enable_thinking`)
    #[argh(option, default = "ThinkMode::Default", from_str_fn(parse_think_mode))]
    think: ThinkMode,

    /// KV cache backing: standard | q8 | q4 (default standard). Any of
    /// --k-bits/--v-bits/--kv-group below override this preset.
    #[argh(
        option,
        long = "kv-cache",
        default = "KvCacheArg::Dense",
        from_str_fn(parse_kv_cache)
    )]
    kv_cache: KvCacheArg,

    /// k cache bits (2/3/4/6/8). Overrides --kv-cache; forces a quantised
    /// cache. K is clamped up to 8 (softmax-sensitive).
    #[argh(option, long = "k-bits")]
    k_bits: Option<i32>,

    /// v cache bits (2/3/4/6/8). Overrides --kv-cache; forces a quantised
    /// cache. Defaults to --k-bits when only that is given.
    #[argh(option, long = "v-bits")]
    v_bits: Option<i32>,

    /// quantisation group size (default 64). Only meaningful with a
    /// quantised cache.
    #[argh(option, long = "kv-group")]
    kv_group: Option<i32>,

    /// max tokens per prefill chunk (default 2048). 0 disables chunking.
    #[argh(option, long = "prefill-chunk-size")]
    prefill_chunk_size: Option<i32>,

    /// path to an image (VLM models); runs one-shot with --prompt, then exits
    #[argh(option)]
    image: Option<PathBuf>,

    /// path to a 16 kHz mono WAV (audio models); one-shot with --prompt (needs
    /// the `audio` feature)
    #[cfg(feature = "audio")]
    #[argh(option)]
    audio: Option<PathBuf>,

    /// instruction sent alongside --image/--audio (default empty)
    #[argh(option, default = "String::new()")]
    prompt: String,

    /// path to a gemma4 MTP assistant (drafter) checkpoint; enables
    /// speculative decode (gemma4 text only)
    #[argh(option, long = "draft-model")]
    draft_model: Option<PathBuf>,

    /// drafter depth γ (tokens drafted per step); default 1
    #[argh(option, long = "mtp-depth")]
    mtp_depth: Option<u32>,

    /// load the drafter but run plain decode (A/B baseline; same prompt path)
    #[argh(switch, long = "no-mtp")]
    no_mtp: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ThinkMode {
    On,
    Off,
    Default,
}

fn parse_think_mode(s: &str) -> std::result::Result<ThinkMode, String> {
    match s {
        "on" | "true" | "1" => Ok(ThinkMode::On),
        "off" | "false" | "0" => Ok(ThinkMode::Off),
        "default" => Ok(ThinkMode::Default),
        other => Err(format!("--think: expected on|off|default, got {other}")),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum KvCacheArg {
    Dense,
    Q8,
    Q4,
}

fn parse_kv_cache(s: &str) -> std::result::Result<KvCacheArg, String> {
    match s {
        "standard" | "std" | "dense" => Ok(KvCacheArg::Dense),
        "q8" | "quantized-q8" => Ok(KvCacheArg::Q8),
        "q4" | "quantized-q4" => Ok(KvCacheArg::Q4),
        other => Err(format!("--kv-cache: expected standard|q8|q4, got {other}")),
    }
}

impl KvCacheArg {
    /// The preset's `(group_size, k_bits, v_bits)`, or `None` for dense.
    fn quant_base(self) -> Option<(i32, i32, i32)> {
        match self {
            Self::Dense => None,
            Self::Q8 => Some((DEFAULT_KV_GROUP_SIZE, 8, 8)),
            Self::Q4 => Some((DEFAULT_KV_GROUP_SIZE, 4, 4)),
        }
    }
}

/// Build the `CacheKind` from the `--kv-cache` preset plus any explicit
/// `--k-bits`/`--v-bits`/`--kv-group` overrides, and report the resolved
/// configuration.
///
/// Resolution rules:
/// - No preset and no overrides → dense (fp16).
/// - Any of `--k-bits`/`--v-bits`/`--kv-group` forces a quantised cache.
/// - An explicit bit arg overrides only its own tensor. The other tensor
///   takes the preset's value when a preset is set, otherwise mirrors the
///   given bit (so a bare `--k-bits 8` yields k8/v8).
/// - `--kv-group` overrides the group size; default 64.
///
/// The cache itself clamps unsafe bit-widths (k up to 8, v down to k) and
/// warns; this echoes the *requested* values before that clamp.
fn resolve_cache_kind(
    preset: KvCacheArg,
    k_bits: Option<i32>,
    v_bits: Option<i32>,
    kv_group: Option<i32>,
) -> CacheKind {
    let overridden = k_bits.is_some() || v_bits.is_some() || kv_group.is_some();
    let base = preset.quant_base();

    if base.is_none() && !overridden {
        eprintln!("[kv-cache: dense (fp16)]");
        return CacheKind::Dense;
    }

    // With no preset, each unset bit mirrors the other given bit (falling
    // back to 8 when neither is set); with a preset, unset bits keep the
    // preset's value.
    let (base_group, base_k, base_v) = base.unwrap_or((
        DEFAULT_KV_GROUP_SIZE,
        k_bits.or(v_bits).unwrap_or(8),
        v_bits.or(k_bits).unwrap_or(8),
    ));
    let k = k_bits.unwrap_or(base_k);
    let v = v_bits.unwrap_or(base_v);
    let group = kv_group.unwrap_or(base_group);

    eprintln!(
        "[kv-cache: quantised k_bits={k} v_bits={v} group_size={group} \
         (requested; cache clamps k<8 and v>k)]"
    );
    CacheKind::Quantized {
        group_size: group,
        k_bits: k,
        v_bits: v,
    }
}

fn main() -> Result<()> {
    let args: Args = argh::from_env();

    eprintln!("[loading {}]", args.model.display());
    if let Some(d) = &args.draft_model {
        eprintln!("[drafter {}]", d.display());
    }
    let mut ctx =
        load_with_drafter(&args.model, args.draft_model.as_deref()).context("load model")?;
    if let Some(n) = args.mtp_depth {
        ctx.model.set_mtp_depth(n);
    }
    // 0 → disable chunking. Otherwise the user-provided value (or
    // `CacheOptions::default()`'s 2048 fallback) is used.
    let max_prefill_chunk = match args.prefill_chunk_size {
        Some(0) => None,
        Some(n) => Some(n),
        None => CacheOptions::default().max_prefill_chunk,
    };
    let kind = resolve_cache_kind(args.kv_cache, args.k_bits, args.v_bits, args.kv_group);
    ctx.model
        .set_cache_options(CacheOptions {
            kind,
            max_prefill_chunk,
        })
        .context("set kv-cache options")?;

    // One-shot VLM path: --image runs a single image+prompt turn, then exits.
    if let Some(image_path) = &args.image {
        let img = image::open(image_path).context("open image")?;
        let msg = ChatMessage::user_with_image(&args.prompt);
        let input = apply_think(
            UserInput::chat(vec![msg]).with_images(vec![Image::Decoded(img)]),
            args.think,
        );
        run_turn(&mut ctx, input, build_params(&args))?;
        println!();
        return Ok(());
    }

    // One-shot audio path: --audio runs a single audio+prompt turn, then exits.
    #[cfg(feature = "audio")]
    if let Some(audio_path) = &args.audio {
        let samples = load_wav_16k_mono(audio_path)?;
        let msg = ChatMessage::user_with_audio(&args.prompt);
        let input = apply_think(
            UserInput::chat(vec![msg]).with_audio(vec![Audio { samples }]),
            args.think,
        );
        run_turn(&mut ctx, input, build_params(&args))?;
        println!();
        return Ok(());
    }

    let mut history: Vec<ChatMessage> = Vec::new();
    let mut editor = DefaultEditor::new().context("rustyline init")?;
    eprintln!("[ready. /exit to quit. /reset to clear history.]");

    loop {
        let input = match editor.readline(PROMPT) {
            Ok(s) => s,
            Err(ReadlineError::Interrupted | ReadlineError::Eof) => break,
            Err(e) => return Err(anyhow::anyhow!("readline: {e}")),
        };
        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed == "/exit" || trimmed == "/quit" {
            break;
        }
        if trimmed == "/reset" {
            history.clear();
            eprintln!("[history cleared]");
            continue;
        }
        editor.add_history_entry(trimmed).ok();

        history.push(ChatMessage::user(trimmed));
        let user_input = apply_think(UserInput::chat(history.clone()), args.think);

        match run_turn(&mut ctx, user_input, build_params(&args)) {
            Ok(text) => history.push(ChatMessage::assistant(text)),
            Err(e) => {
                // Pop the unanswered user turn so the next prompt
                // isn't a duplicate of the failed one.
                history.pop();
                eprintln!("[error: {e:#}]");
            }
        }
        println!();
    }
    Ok(())
}

/// Decode a 16 kHz mono WAV to `f32` samples in `[-1, 1]`. Errors on any other
/// sample rate or channel count — the caller pre-converts (e.g.
/// `ffmpeg -ar 16000 -ac 1`).
#[cfg(feature = "audio")]
fn load_wav_16k_mono(path: &std::path::Path) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path).context("open wav")?;
    let spec = reader.spec();
    if spec.sample_rate != 16_000 || spec.channels != 1 {
        anyhow::bail!(
            "wav must be 16 kHz mono (got {} Hz, {} ch); convert with `ffmpeg -ar 16000 -ac 1`",
            spec.sample_rate,
            spec.channels
        );
    }
    let mut reader = reader;
    match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("read wav samples"),
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max))
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("read wav samples")
        }
    }
}

/// Apply the `--think` mode as the template's `enable_thinking` kwarg.
fn apply_think(input: UserInput, think: ThinkMode) -> UserInput {
    match think {
        ThinkMode::On => {
            input.with_template_kwarg("enable_thinking", serde_json::Value::Bool(true))
        }
        ThinkMode::Off => {
            input.with_template_kwarg("enable_thinking", serde_json::Value::Bool(false))
        }
        ThinkMode::Default => input,
    }
}

/// Resolve `--temperature`/`--top-p`/`--max-tokens` into [`GenerateParams`].
fn build_params(args: &Args) -> GenerateParams {
    let sampling = match (args.temperature, args.top_p) {
        (0.0, _) => Sampler::Greedy,
        (t, None) => Sampler::Temperature(t),
        (t, Some(p)) => Sampler::TopP { temperature: t, p },
    };
    GenerateParams {
        max_new_tokens: args.max_tokens,
        sampling,
        disable_mtp: args.no_mtp,
        ..GenerateParams::default()
    }
}

fn run_turn(ctx: &mut ModelContext, input: UserInput, params: GenerateParams) -> Result<String> {
    // `ThinkStream` owns its colours (answer = bold green, reasoning =
    // dim) and resets on `finish`.
    let mut md = ThinkStream::new(std::io::stdout().lock());

    let t_start = Instant::now();
    let mut t_first: Option<Instant> = None;
    let mut push_err: Option<std::io::Error> = None;
    let result = generate(ctx, input, params, &mut |_, delta| {
        if t_first.is_none() {
            t_first = Some(Instant::now());
        }
        if let Err(e) = md.push(delta) {
            push_err = Some(e);
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    })?;
    let t_end = Instant::now();

    md.finish()?;
    if let Some(e) = push_err {
        return Err(e.into());
    }
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    drop(stdout);

    let t_first = t_first.unwrap_or(t_end);
    let prefill_s = (t_first - t_start).as_secs_f64();
    let decode_s = (t_end - t_first).as_secs_f64();
    let prefill_tps = safe_rate(result.prompt_tokens as f64, prefill_s);
    let decode_steps = result.completion_tokens.saturating_sub(1);
    let decode_tps = safe_rate(decode_steps as f64, decode_s);
    eprintln!(
        "{C_DIM}[prefill: {n_prompt} tok in {prefill_s:.2}s ({prefill_tps:.1} tok/s) | \
         decode: {decode_steps} tok in {decode_s:.2}s ({decode_tps:.1} tok/s)]{C_RESET}",
        n_prompt = result.prompt_tokens,
    );
    Ok(result.text)
}

/// Token-rate `n / seconds`, returning 0.0 for the degenerate
/// zero-duration case (single-token prompt + zero-token decode).
fn safe_rate(n: f64, seconds: f64) -> f64 {
    if seconds > 0.0 {
        n / seconds
    } else {
        0.0
    }
}
