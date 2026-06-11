//! Weight loader for Qwen3.5 / Chandra OCR-2 checkpoints.
//!
//! Key sanitiser:
//!
//! - `model.language_model.X` → `language_model.model.X`
//! - `model.visual.X`         → `vision_tower.X`
//! - `lm_head.X`              → `language_model.lm_head.X`
//! - `conv1d.weight` whose last axis != 1 is normalised with
//!   `moveaxis(2, 1)`.
//! - Norm weights (`*.input_layernorm.weight`,
//!   `*.post_attention_layernorm.weight`, `model.norm.weight`,
//!   `*.q_norm.weight`, `*.k_norm.weight`) receive `+1.0` when their
//!   dtype is a floating-point 1-D tensor — recovers the standard
//!   RMSNorm parameterisation from the centred form stored in the
//!   checkpoint.
//! - `mtp.*` keys are routed to the `MtpHead` parameter walk (when
//!   the model was constructed with `mtp_num_hidden_layers > 0`).

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{
    module::ModuleParameters, ops::move_axis, quantization::Quantizable as _,
    transforms::eval_params, Array, Dtype,
};

pub use super::config::ModelConfig;
pub use super::layer::Qwen35Model;
use crate::config::ModelConfig as Config;
use crate::error::Error;
use crate::loader::{apply_post_load_memory_policy, list_shards, rewrite_quantised_keys};
use crate::quantization::QuantizationConfig;

const NORM_SUFFIXES: &[&str] = &[
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
    "model.norm.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.norm.weight",
];

/// Returns `true` if the safetensors file's header metadata advertises
/// `format == "mlx"`. mlx-format checkpoints already have the
/// norm-weight `+1.0` shift baked into the stored tensors, so the
/// sanitiser must skip the shift on these checkpoints.
fn safetensors_is_mlx_format(path: &Path) -> Result<bool, Error> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut len_bytes = [0_u8; 8];
    if f.read_exact(&mut len_bytes).is_err() {
        return Ok(false);
    }
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    if header_len == 0 || header_len > 64 * 1024 * 1024 {
        return Ok(false);
    }
    let mut buf = vec![0_u8; header_len];
    if f.read_exact(&mut buf).is_err() {
        return Ok(false);
    }
    let header = std::str::from_utf8(&buf)
        .map_err(|e| Error::Other(format!("safetensors header is not utf-8: {e}").into()))?;
    // The metadata lives under `"__metadata__"`. Just scan for the
    // `"format":"mlx"` literal — full JSON parsing of the header is
    // overkill for one boolean.
    Ok(header.contains("\"__metadata__\"") && header.contains("\"format\":\"mlx\""))
}

/// Rewrite checkpoint key prefixes to match the Rust module tree.
fn sanitize_key(key: &str) -> String {
    // Prefix alignment first (mutually exclusive), then the GDN
    // param-walk rewrite applies regardless of which prefix matched.
    let mut k = if key.contains("model.language_model") {
        key.replace("model.language_model", "language_model.model")
    } else if key.contains("model.visual") {
        key.replace("model.visual", "vision_tower")
    } else if let Some(rest) = key.strip_prefix("lm_head") {
        format!("language_model.lm_head{rest}")
    } else if let Some(rest) = key.strip_prefix("mtp.") {
        format!("language_model.mtp.{rest}")
    } else {
        key.to_owned()
    };
    // GDN: `norm.weight`→`norm_weight` (collapsed Param), `A_log`→`a_log`.
    if k.contains(".linear_attn.") {
        k = k
            .replace(".linear_attn.norm.weight", ".linear_attn.norm_weight")
            .replace(".linear_attn.A_log", ".linear_attn.a_log");
    }
    k
}

/// Strip the `language_model.` prefix to match the Rust Qwen35Model's
/// parameter paths.
#[cfg(test)]
fn strip_language_model_prefix(key: &str) -> &str {
    key.strip_prefix("language_model.").unwrap_or(key)
}

/// Bucket a sanitised key into the language-model or vision-tower
/// param-path namespace (prefix stripped), or neither.
#[derive(Debug)]
pub(crate) enum Bucketed {
    /// Routes to [`Qwen35Model`] under the returned path.
    Language(String),
    /// Routes to the vision tower under the returned path. Consumed by the
    /// VLM loader (`crate::qwen3_5::image::weights`); the text-only loaders
    /// drop these.
    Vision(String),
    /// Neither bucket — typically a `mtp.*` or unknown key that should be
    /// dropped or surfaced in the loader's `leftover` list.
    Other(String),
}

pub(crate) fn bucket_key(key: String) -> Bucketed {
    if let Some(rest) = key.strip_prefix("language_model.") {
        return Bucketed::Language(rest.to_owned());
    }
    if let Some(rest) = key.strip_prefix("vision_tower.") {
        return Bucketed::Vision(rest.to_owned());
    }
    Bucketed::Other(key)
}

/// Apply the `+1.0` centring fix to a norm weight tensor.
fn add_one_to_norm(value: &Array) -> Result<Array, Error> {
    let dt = value.dtype();
    let one = Array::from_f32(1.0)
        .as_dtype(dt)
        .map_err(Error::Exception)?;
    value.add(&one).map_err(Error::Exception)
}

/// Apply the conv1d moveaxis sanitisation: `[out, in, k]` -> `[out, k, in]`
/// when the last axis is not already `1`.
fn sanitize_conv1d_weight(value: Array) -> Result<Array, Error> {
    let shape = value.shape();
    if shape.len() != 3 || shape[2] == 1 {
        return Ok(value);
    }
    let moved = move_axis(&value, 2, 1).map_err(Error::Exception)?;
    Ok(moved)
}

/// Returns the `Array` after applying any per-key sanitisation rules.
///
/// `is_mlx_format` tracks whether the source safetensors carry the
/// `format == "mlx"` metadata flag — when true, the conv1d moveaxis and the
/// norm `+1.0` shift are *already* baked into the stored weights, and
/// re-applying them here doubles the bias / mis-orients the kernel.
fn sanitize_value(key: &str, value: Array, is_mlx_format: bool) -> Result<Array, Error> {
    if !is_mlx_format && key.contains("conv1d.weight") {
        return sanitize_conv1d_weight(value);
    }
    if is_mlx_format {
        return Ok(value);
    }
    let needs_plus_one = NORM_SUFFIXES.iter().any(|sfx| key.ends_with(sfx));
    if needs_plus_one && value.ndim() == 1 && is_float(value.dtype()) {
        return add_one_to_norm(&value);
    }
    Ok(value)
}

fn is_float(dtype: Dtype) -> bool {
    matches!(
        dtype,
        Dtype::Float16 | Dtype::Float32 | Dtype::Float64 | Dtype::Bfloat16
    )
}

/// Load and sanitise every shard listed in `model.safetensors.index.json`.
///
/// Returns a flat map keyed by the **fully-qualified** sanitised path
/// (`language_model.model.layers.0.self_attn.q_proj.weight`,
/// `language_model.mtp.*`, ...).
/// Caller buckets the result into the LM parameter walk.
pub(crate) fn load_sanitized_weights(
    model_dir: impl AsRef<Path>,
) -> Result<HashMap<String, Array>, Error> {
    let model_dir = model_dir.as_ref();
    let shards = list_shards(model_dir)?;
    let mut raw: HashMap<String, Array> = HashMap::new();
    for path in shards {
        let is_mlx_format = safetensors_is_mlx_format(&path)?;
        let loaded = Array::load_safetensors(&path).map_err(Error::LoadWeights)?;
        for (k, v) in loaded {
            let san_k = sanitize_key(&k);
            let san_v = sanitize_value(&san_k, v, is_mlx_format)?;
            raw.insert(san_k, san_v);
        }
    }

    // GDN naming alignment (`norm.weight`→`norm_weight`, `A_log`→`a_log`)
    // is handled per-key in `sanitize_key`.
    //
    // Rewrite `<prefix>.weight` → `<prefix>.inner.weight` for keys with
    // a `.scales` sibling so they align with the
    // `MaybeQuantized::Quantized(QuantizedLinear { inner })` param path.
    Ok(rewrite_quantised_keys(raw))
}

/// Load weights into a Rust [`Qwen35Model`] only. Vision-tower keys are
/// returned in the `leftover` list so callers can decide whether to ignore
/// them or feed them through [`load_full_model`] instead.
///
/// The model is rebuilt as quantised first when the checkpoint declares
/// `quantization_config`. Returns the loaded model and the list of
/// fully-qualified sanitised paths that did not bind to a model parameter.
pub(crate) fn load_language_model(
    cfg: &Config,
    env: &ModelConfig,
    model_dir: &Path,
) -> Result<(Qwen35Model, Vec<String>), Error> {
    env.text_config.validate()?;
    let mut model = Qwen35Model::with_mlp(env.text_config.clone())?;
    if let Some(q) = cfg.quantization() {
        quantize_language_model(&mut model, q)?;
    }
    let weights = load_sanitized_weights(model_dir)?;

    let mut leftover = Vec::new();
    {
        let mut params = model.parameters_mut().flatten();
        for (k, v) in weights {
            match bucket_key(k) {
                Bucketed::Language(p) => {
                    if let Some(slot) = params.get_mut(&*p) {
                        **slot = v;
                    } else {
                        leftover.push(format!("language_model.{p}"));
                    }
                }
                // Text-only loader: a VL checkpoint's vision keys are
                // surfaced as leftover (the VLM loader consumes them).
                Bucketed::Vision(p) => leftover.push(format!("vision_tower.{p}")),
                Bucketed::Other(p) => leftover.push(p),
            }
        }
    }

    eval_params(model.parameters()).map_err(Error::Exception)?;
    apply_post_load_memory_policy();
    leftover.sort();
    Ok((model, leftover))
}

pub(crate) fn quantize_language_model(
    model: &mut Qwen35Model,
    q: &QuantizationConfig,
) -> Result<(), Error> {
    let original = std::mem::replace(model, Qwen35Model::with_mlp(model.cfg.clone())?);
    let quantized = original
        .try_into_quantized(q.group_size, q.bits)
        .map_err(Error::Exception)?;
    *model = quantized;
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    #![allow(clippy::print_stdout, reason = "test code")]
    #![allow(clippy::print_stderr, reason = "test code")]
    use super::*;

    #[test]
    fn sanitize_key_rewrites_language_model_prefix() {
        assert_eq!(
            sanitize_key("model.language_model.embed_tokens.weight"),
            "language_model.model.embed_tokens.weight"
        );
        assert_eq!(
            sanitize_key("lm_head.weight"),
            "language_model.lm_head.weight"
        );
        assert_eq!(
            sanitize_key("model.embed_tokens.weight"),
            "model.embed_tokens.weight"
        );
    }

    #[test]
    fn sanitize_key_gdn_rewrite_runs_after_prefix_swap() {
        // GDN suffix rewrite applies even when a prefix branch matched
        // (the `model.language_model.…linear_attn.…` VLM-raw layout).
        assert_eq!(
            sanitize_key("model.language_model.layers.0.linear_attn.norm.weight"),
            "language_model.model.layers.0.linear_attn.norm_weight"
        );
        assert_eq!(
            sanitize_key("language_model.model.layers.0.linear_attn.A_log"),
            "language_model.model.layers.0.linear_attn.a_log"
        );
    }

    #[test]
    fn strip_prefix_drops_language_model_segment() {
        assert_eq!(
            strip_language_model_prefix("language_model.model.embed_tokens.weight"),
            "model.embed_tokens.weight"
        );
    }

    #[test]
    #[ignore = "requires local model files at ~/MLXModels/chandra2/chandra-ocr-2-mlx-q8"]
    fn text_only_prefill_runs_on_loaded_chandra_q8() {
        use crate::cache::CacheOptions;
        use crate::qwen3_5::text::cache::make_caches;
        use mlx_rs::transforms::eval;
        use tokenizers::Tokenizer;

        let home = std::env::var("HOME").unwrap();
        let dir = std::path::PathBuf::from(home).join("MLXModels/chandra2/chandra-ocr-2-mlx-q8");
        let cfg = Config::from_dir(&dir).expect("parse config");
        let env = cfg.family.as_qwen35().expect("expected qwen3_5 family");
        let (mut model, _leftover) = load_language_model(&cfg, env, &dir).expect("load weights");

        let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("load tokenizer");
        let enc = tok.encode("Hello, world!", true).expect("encode");
        let ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
        let s = ids.len() as i32;
        let inputs = Array::from_slice(&ids, &[1, s]);

        let mut caches = make_caches(env, CacheOptions::default());
        let logits = model
            .forward(Some(&inputs), &mut caches, None)
            .expect("forward");
        eval([&logits]).expect("eval");
        assert_eq!(
            logits.shape(),
            &[1, s, env.text_config.vocab_size],
            "logits shape mismatch"
        );
        // No NaNs anywhere.
        let any_nan: Array = logits
            .as_dtype(Dtype::Float32)
            .unwrap()
            .ne(logits.as_dtype(Dtype::Float32).unwrap())
            .unwrap()
            .any(None)
            .unwrap();
        eval([&any_nan]).unwrap();
        assert!(!any_nan.item::<bool>(), "logits contain NaN");
    }

    #[test]
    #[ignore = "requires local model files at ~/MLXModels/chandra2/chandra-ocr-2-mlx-q8"]
    fn loads_chandra_q8_weights_into_language_model() {
        let home = std::env::var("HOME").unwrap();
        let dir = std::path::PathBuf::from(home).join("MLXModels/chandra2/chandra-ocr-2-mlx-q8");
        let cfg = Config::from_dir(&dir).expect("parse config");
        let env = cfg.family.as_qwen35().expect("expected qwen3_5 family");
        let (model, leftover) = load_language_model(&cfg, env, &dir).expect("load weights");

        // We should at least have layer 0 weights populated. The exact param
        // ergonomics get exercised in subsequent commits; for now we sanity-
        // check that the model survived the load and there are no
        // *unexpected* leftover keys: only vision-tower keys, which are not
        // wired up yet, may remain.
        let mut unexpected: Vec<&String> = leftover
            .iter()
            .filter(|k| !k.starts_with("vision_tower"))
            .collect();
        unexpected.sort();
        if !unexpected.is_empty() {
            eprintln!("first 30 unexpected leftover keys:");
            for k in unexpected.iter().take(30) {
                eprintln!("  {k}");
            }
            panic!(
                "{} unexpected leftover safetensors keys (see stderr)",
                unexpected.len()
            );
        }
        // Smoke-eval all parameters to catch loader errors.
        use mlx_rs::module::ModuleParametersExt;
        model.eval().expect("eval params");
    }
}
