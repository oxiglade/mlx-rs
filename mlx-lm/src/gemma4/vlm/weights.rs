//! VLM weight loader: split a `gemma4` multimodal checkpoint into the text
//! `Model`, the bf16 `VisionModel` tower + quantized `EmbedVision` projector,
//! and (behind the `audio` feature) the bf16 `AudioEncoder` + quantized
//! `EmbedAudio`. Tower weights are bf16; text + projectors are quantized per
//! `cfg.quantization()`. With `audio` off, audio keys are dropped so e2b/e4b
//! still load text+vision.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::module::ModuleParameters;
use mlx_rs::quantization::Quantizable;
use mlx_rs::transforms::eval_params;
use mlx_rs::Array;

use crate::config::ModelConfig as Config;
use crate::error::Error;
use crate::gemma4::image::config::VisionConfig;
use crate::gemma4::image::vision::{EmbedVision, VisionModel};
use crate::gemma4::text::config::ModelConfig;
use crate::gemma4::text::text::Model;
use crate::gemma4::text::weights::{is_shared_kv_layer_key, rewrite_outer_key};
use crate::loader::{apply_post_load_memory_policy, list_shards, rewrite_quantised_keys};

#[cfg(feature = "audio")]
use crate::gemma4::audio::encoder::{AudioEncoder, EmbedAudio};

/// Clip-buffer key substrings dropped from the VISION tower only: vision sets
/// `use_clipped_linears: false`, so its clip buffers are inert. The AUDIO tower
/// sets it `true` — those clips are LIVE and must NOT be dropped (do not reuse
/// this list for audio).
const VISION_CLIP_DROP_SUBSTRINGS: &[&str] =
    &["input_max", "input_min", "output_max", "output_min"];

/// One bucket a checkpoint key routes to after prefix rewriting.
enum Bucket {
    /// `vision_tower.…` (key with the prefix stripped). bf16, never quantized.
    Vision(String),
    /// `embed_vision.…` (key with the prefix stripped). Quantized.
    EmbedVision(String),
    /// `audio_tower.…` (key rewritten onto the `AudioEncoder` param walk). bf16.
    #[cfg(feature = "audio")]
    Audio(String),
    /// `embed_audio.…` (key with the prefix stripped). Quantized.
    #[cfg(feature = "audio")]
    EmbedAudio(String),
    /// Audio keys when the `audio` feature is off — dropped at load.
    #[cfg(not(feature = "audio"))]
    Drop,
    /// Everything else → the text `Model` (post `rewrite_outer_key`).
    Text(String),
}

/// Map a `vision_tower.…`-stripped checkpoint key onto the `VisionModel`
/// param walk: drop the `ClippableLinear` `.linear.` wrapper segment, and
/// collapse the `encoder.layers.N` nesting to `encoder.N` (our `encoder` is a
/// bare `Vec`, not a wrapped sub-module).
fn rewrite_vision_key(key: &str) -> String {
    key.replace(".linear.", ".")
        .replace("encoder.layers.", "encoder.")
}

/// Map an `audio_tower.…`-stripped key onto the `AudioEncoder` param walk:
/// the SSCP sub-blocks are flat fields (`layer0.conv` → `layer0_conv`), and the
/// depthwise conv is a bare `Param` (`depthwise_conv1d.weight` → `…`). The
/// `ClippableLinear` `.linear.` wrapper is KEPT (our `ClippableLinear` nests a
/// `linear` field) — only the `.weight`-suffixed depthwise conv collapses.
#[cfg(feature = "audio")]
fn rewrite_audio_key(key: &str) -> String {
    key.replace(
        "subsample_conv_projection.layer0.conv",
        "subsample_conv_projection.layer0_conv",
    )
    .replace(
        "subsample_conv_projection.layer0.norm",
        "subsample_conv_projection.layer0_norm",
    )
    .replace(
        "subsample_conv_projection.layer1.conv",
        "subsample_conv_projection.layer1_conv",
    )
    .replace(
        "subsample_conv_projection.layer1.norm",
        "subsample_conv_projection.layer1_norm",
    )
    .replace("depthwise_conv1d.weight", "depthwise_conv1d")
}

fn bucket_key(key: &str) -> Bucket {
    if let Some(rest) = key.strip_prefix("vision_tower.") {
        return Bucket::Vision(rewrite_vision_key(rest));
    }
    if let Some(rest) = key.strip_prefix("embed_vision.") {
        return Bucket::EmbedVision(rest.to_owned());
    }
    if key.starts_with("audio_tower.") || key.starts_with("embed_audio.") {
        #[cfg(feature = "audio")]
        {
            if let Some(rest) = key.strip_prefix("audio_tower.") {
                return Bucket::Audio(rewrite_audio_key(rest));
            }
            let rest = key.strip_prefix("embed_audio.").expect("checked above");
            return Bucket::EmbedAudio(rest.to_owned());
        }
        #[cfg(not(feature = "audio"))]
        return Bucket::Drop;
    }
    Bucket::Text(rewrite_outer_key(key))
}

/// Loaded multimodal towers. Vision is always present (the VLM dispatch
/// requires `vision_config`); audio is present only on e2b/e4b with the `audio`
/// feature on.
pub(crate) struct LoadedTowers {
    pub text: Model,
    pub vision: VisionModel,
    pub embed_vision: EmbedVision,
    #[cfg(feature = "audio")]
    pub audio: Option<(AudioEncoder, EmbedAudio)>,
}

/// Load the text model, vision tower + projector, and (feature-gated) the audio
/// tower + projector from one checkpoint.
pub(crate) fn load_full_model(
    cfg: &Config,
    env: &ModelConfig,
    vision_cfg: &VisionConfig,
    model_dir: &Path,
) -> Result<LoadedTowers, Error> {
    let mut text = Model::new(env.text_config.clone())?;
    let mut vision = VisionModel::new(vision_cfg)?;
    let mut embed_vision = EmbedVision::new(vision_cfg, env.text_config.hidden_size)?;
    if let Some(q) = cfg.quantization() {
        text = text.try_into_quantized(q.group_size, q.bits)?;
        embed_vision = embed_vision.try_into_quantized(q.group_size, q.bits)?;
    }

    // Audio tower (bf16) + quantized projector, built only when the `audio`
    // feature is on AND the checkpoint declares an audio config.
    #[cfg(feature = "audio")]
    let mut audio: Option<(AudioEncoder, EmbedAudio)> = match env.audio_config.as_ref() {
        Some(ac) => {
            let mut embed = EmbedAudio::new(ac, env.text_config.hidden_size)?;
            if let Some(q) = cfg.quantization() {
                embed = embed.try_into_quantized(q.group_size, q.bits)?;
            }
            Some((AudioEncoder::new(ac)?, embed))
        }
        None => None,
    };

    let num_layers = env.text_config.num_hidden_layers;
    let num_kv_shared = env.text_config.num_kv_shared_layers;

    let shards = list_shards(model_dir)?;
    let mut text_raw: HashMap<String, Array> = HashMap::new();
    let mut vision_quant_raw: HashMap<String, Array> = HashMap::new();
    let mut vision_raw: HashMap<String, Array> = HashMap::new();
    #[cfg(feature = "audio")]
    let mut audio_raw: HashMap<String, Array> = HashMap::new();
    #[cfg(feature = "audio")]
    let mut audio_quant_raw: HashMap<String, Array> = HashMap::new();
    for path in shards {
        let loaded = Array::load_safetensors(&path).map_err(Error::LoadWeights)?;
        for (k, v) in loaded {
            match bucket_key(&k) {
                Bucket::Vision(p) => {
                    if VISION_CLIP_DROP_SUBSTRINGS.iter().any(|s| p.contains(s)) {
                        continue;
                    }
                    vision_raw.insert(p, v);
                }
                Bucket::EmbedVision(p) => {
                    vision_quant_raw.insert(format!("embed_vision.{p}"), v);
                }
                #[cfg(feature = "audio")]
                Bucket::Audio(p) => {
                    audio_raw.insert(p, v);
                }
                #[cfg(feature = "audio")]
                Bucket::EmbedAudio(p) => {
                    audio_quant_raw.insert(format!("embed_audio.{p}"), v);
                }
                // Audio off: drop the audio keys so e2b/e4b still load.
                #[cfg(not(feature = "audio"))]
                Bucket::Drop => {}
                Bucket::Text(key) => {
                    if is_shared_kv_layer_key(&key, num_layers, num_kv_shared) {
                        continue;
                    }
                    text_raw.insert(key, v);
                }
            }
        }
    }
    let text_weights = rewrite_quantised_keys(text_raw);
    let embed_weights =
        strip_projector_prefix(rewrite_quantised_keys(vision_quant_raw), "embed_vision.");

    let mut leftover: Vec<String> = Vec::new();
    bind(&mut text, text_weights, "text", &mut leftover);
    bind(&mut vision, vision_raw, "vision_tower", &mut leftover);
    bind(
        &mut embed_vision,
        embed_weights,
        "embed_vision",
        &mut leftover,
    );

    #[cfg(feature = "audio")]
    if let Some((enc, embed)) = audio.as_mut() {
        // Audio uses `use_clipped_linears: true` — the input/output clip
        // buffers are LIVE, so (unlike vision) they are NOT dropped here.
        let embed_audio_weights =
            strip_projector_prefix(rewrite_quantised_keys(audio_quant_raw), "embed_audio.");
        bind(enc, audio_raw, "audio_tower", &mut leftover);
        bind(embed, embed_audio_weights, "embed_audio", &mut leftover);
    }

    if !leftover.is_empty() {
        leftover.sort();
        return Err(Error::Other(
            format!(
                "gemma4 VLM loader: {} unbound key(s); first 8: {:?}",
                leftover.len(),
                &leftover.iter().take(8).collect::<Vec<_>>()
            )
            .into(),
        ));
    }

    eval_params(text.parameters()).map_err(Error::Exception)?;
    eval_params(vision.parameters()).map_err(Error::Exception)?;
    eval_params(embed_vision.parameters()).map_err(Error::Exception)?;
    #[cfg(feature = "audio")]
    if let Some((enc, embed)) = audio.as_ref() {
        eval_params(enc.parameters()).map_err(Error::Exception)?;
        eval_params(embed.parameters()).map_err(Error::Exception)?;
    }
    apply_post_load_memory_policy();
    Ok(LoadedTowers {
        text,
        vision,
        embed_vision,
        #[cfg(feature = "audio")]
        audio,
    })
}

/// Quantised projector keys come back as `<prefix>embedding_projection.inner.*`;
/// strip the bucket prefix back off for the projector's own param walk.
fn strip_projector_prefix(weights: HashMap<String, Array>, prefix: &str) -> HashMap<String, Array> {
    weights
        .into_iter()
        .map(|(k, v)| (k.strip_prefix(prefix).unwrap_or(&k).to_owned(), v))
        .collect()
}

/// Bind a bucket's weights into a module's parameter walk; record unbound keys.
fn bind<M: ModuleParameters>(
    module: &mut M,
    weights: HashMap<String, Array>,
    prefix: &str,
    leftover: &mut Vec<String>,
) {
    let mut params = module.parameters_mut().flatten();
    for (k, v) in weights {
        if let Some(slot) = params.get_mut(&*k) {
            **slot = v;
        } else {
            leftover.push(format!("{prefix}.{k}"));
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::missing_assert_message, reason = "test code")]
    use super::*;

    #[test]
    fn buckets_route_by_prefix() {
        assert!(matches!(
            bucket_key("vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight"),
            Bucket::Vision(p) if p == "encoder.0.self_attn.q_proj.weight"
        ));
        assert!(matches!(
            bucket_key("embed_vision.embedding_projection.weight"),
            Bucket::EmbedVision(p) if p == "embedding_projection.weight"
        ));
        assert!(matches!(
            bucket_key("model.layers.0.self_attn.q_proj.weight"),
            Bucket::Text(p) if p == "model.layers.0.self_attn.q_proj.weight"
        ));
    }

    #[cfg(feature = "audio")]
    #[test]
    fn audio_keys_rewrite_onto_encoder_walk() {
        assert!(matches!(
            bucket_key("audio_tower.layers.0.self_attn.q_proj.linear.weight"),
            Bucket::Audio(p) if p == "layers.0.self_attn.q_proj.linear.weight"
        ));
        assert!(matches!(
            bucket_key("audio_tower.subsample_conv_projection.layer0.conv.weight"),
            Bucket::Audio(p) if p == "subsample_conv_projection.layer0_conv.weight"
        ));
        assert!(matches!(
            bucket_key("audio_tower.layers.0.lconv1d.depthwise_conv1d.weight"),
            Bucket::Audio(p) if p == "layers.0.lconv1d.depthwise_conv1d"
        ));
        assert!(matches!(
            bucket_key("embed_audio.embedding_projection.weight"),
            Bucket::EmbedAudio(p) if p == "embedding_projection.weight"
        ));
    }

    /// Real-weight regression: load an e2b/e4b checkpoint, run the loaded audio
    /// tower over a deterministic 440 Hz sine, and check the output statistics.
    /// Locks the load path (clip buffers, key binding, dtype) — a dropped clip
    /// or unbound weight shifts the magnitude by ~5× and fails. Run:
    /// `MODEL=<e4b dir> cargo test -p mlx-lm --features audio --lib -- \
    ///   --ignored --test-threads=1 audio_encoder_real_weights_stats`
    #[cfg(feature = "audio")]
    #[test]
    #[ignore = "requires a local gemma-4 e2b/e4b checkpoint via MODEL"]
    fn audio_encoder_real_weights_stats() {
        use crate::gemma4::audio::feature::log_mel;
        use mlx_rs::ops::abs;
        use std::path::PathBuf;

        let dir = PathBuf::from(std::env::var("MODEL").expect("set MODEL=<e4b checkpoint dir>"));
        let cfg = Config::from_dir(&dir).expect("parse config");
        let env = cfg.family.as_gemma4().expect("gemma4 config");
        let vision_cfg = env.vision_config.as_ref().expect("vision_config");
        let towers = load_full_model(&cfg, env, vision_cfg, &dir).expect("load");
        let (enc, _) = towers.audio.expect("checkpoint has no audio tower");

        // Deterministic 0.3 s 440 Hz sine at 16 kHz.
        let sr = 16_000usize;
        let n = sr * 3 / 10;
        let wav: Vec<f32> = (0..n)
            .map(|i| 0.3 * (std::f32::consts::TAU * 440.0 * i as f32 / sr as f32).sin())
            .collect();
        let mel = log_mel(&wav).expect("log_mel");
        let mut enc = enc;
        let out = enc.forward(&mel).expect("encode");
        let out = out.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        let mean = out.mean(None).unwrap().item::<f32>();
        let absmax = abs(&out).unwrap().max(None).unwrap().item::<f32>();
        eprintln!(
            "audio encoder: shape {:?} mean {mean:.4} absmax {absmax:.4}",
            out.shape()
        );

        // Reference (gemma-4-e4b-it-8bit): mean ≈ 0, absmax ≈ 47. The clip-drop
        // bug pushed absmax to ~240; an unbound projection collapses it toward 0.
        assert!(
            mean.abs() < 1.0,
            "mean {mean} out of band — load path regressed"
        );
        assert!(
            (20.0..80.0).contains(&absmax),
            "absmax {absmax} out of band — clip buffers likely dropped/unbound"
        );
    }
}
