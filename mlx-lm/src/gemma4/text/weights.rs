//! Gemma 4 weight loader + safetensors sanitiser (dense base).
//!
//! - Drop `vision_tower.*`, `multi_modal_projector.*`, `audio_tower.*`,
//!   `embed_audio.*`, `embed_vision.*`, quantiser stats keys, and
//!   `self_attn.rotary_emb` (rope freqs are computed in code).
//! - `language_model.model.X` → `model.X`; `model.language_model.X` →
//!   `model.X`; bare `language_model.X` (lm_head) → `X`.
//! - Quantised `<prefix>.weight` (with a `<prefix>.scales` sibling) is
//!   remapped to `<prefix>.inner.weight` for the `MaybeQuantized::Quantized`
//!   param path (via [`rewrite_quantised_keys`]).
//!
//! MoE expert-key rewrites and KV-shared-layer key drops are deferred —
//! they land with those extensions (the dense base never has experts or
//! shared layers).

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::module::ModuleParameters;
use mlx_rs::quantization::Quantizable;
use mlx_rs::transforms::eval_params;
use mlx_rs::Array;

use crate::config::ModelConfig as Config;
use crate::error::Error;
use crate::gemma4::text::config::TextConfig;
use crate::gemma4::text::text::Model;
use crate::loader::{apply_post_load_memory_policy, list_shards, rewrite_quantised_keys};
use crate::quantization::{requantise_linear, QuantizationConfig};

/// Substrings that mark a checkpoint key for unconditional removal.
const DROP_SUBSTRINGS: &[&str] = &[
    "vision_tower",
    "multi_modal_projector",
    "audio_tower",
    "embed_audio",
    "embed_vision",
    // gemma4_unified encoder-free vision front-end; dropped on the text path.
    "vision_embedder",
    "self_attn.rotary_emb",
    "input_max",
    "input_min",
    "output_max",
    "output_min",
];

fn should_drop(key: &str) -> bool {
    DROP_SUBSTRINGS.iter().any(|s| key.contains(s))
}

/// Strip the multimodal-wrapper prefix(es) so a text-only `Model` can
/// consume the keys.
pub(crate) fn rewrite_outer_key(key: &str) -> String {
    if let Some(rest) = key.strip_prefix("language_model.model.") {
        return format!("model.{rest}");
    }
    if let Some(rest) = key.strip_prefix("model.language_model.") {
        return format!("model.{rest}");
    }
    if let Some(rest) = key.strip_prefix("language_model.") {
        return rest.to_owned();
    }
    key.to_owned()
}

/// `k_*`/`v_*` keys for KV-shared layers (`>= num_layers - num_shared`)
/// must be dropped: those layers own no K/V projection (they reuse a prior
/// layer's), so the model has no slot for them. Key is post-`rewrite_outer_key`
/// (`model.layers.N.…`).
pub(crate) fn is_shared_kv_layer_key(key: &str, num_layers: i32, num_shared: i32) -> bool {
    if num_shared <= 0 {
        return false;
    }
    let first_shared = num_layers - num_shared;
    let Some(rest) = key.strip_prefix("model.layers.") else {
        return false;
    };
    let Some(dot) = rest.find('.') else {
        return false;
    };
    let Ok(layer_idx) = rest[..dot].parse::<i32>() else {
        return false;
    };
    if layer_idx < first_shared {
        return false;
    }
    let tail = &rest[dot + 1..];
    tail.starts_with("self_attn.k_") || tail.starts_with("self_attn.v_")
}

/// Load every shard, drop/rewrite per-key, then map quantised
/// `<prefix>.weight` → `<prefix>.inner.weight`. `num_kv_shared` drops the
/// K/V keys of KV-shared layers (E2B/E4B); 0 for dense/MoE.
pub(crate) fn load_sanitized_weights(
    model_dir: impl AsRef<Path>,
    num_layers: i32,
    num_kv_shared: i32,
) -> Result<HashMap<String, Array>, Error> {
    let shards = list_shards(model_dir.as_ref())?;
    let mut raw: HashMap<String, Array> = HashMap::new();
    for path in shards {
        let loaded = Array::load_safetensors(&path).map_err(Error::LoadWeights)?;
        for (k, v) in loaded {
            if should_drop(&k) {
                continue;
            }
            let key = rewrite_outer_key(&k);
            if is_shared_kv_layer_key(&key, num_layers, num_kv_shared) {
                continue;
            }
            raw.insert(key, v);
        }
    }
    Ok(rewrite_quantised_keys(raw))
}

/// After body quantisation, re-quantise per-tensor override slots at their
/// declared bit width. mlx-community 4-bit gemma4 ships the dense MLP
/// projections (`mlp.{gate,down,up}_proj`) at 8-bit while the body is 4-bit;
/// a uniform body quant builds those slots at the wrong width and the
/// quantized matmul then fails on the scale shape.
fn quantize_mlp_overrides(model: &mut Model, q: &QuantizationConfig) -> Result<(), Error> {
    if q.overrides.is_empty() {
        return Ok(());
    }
    for (layer_idx, layer) in model.model.layers.iter_mut().enumerate() {
        for (proj, slot) in [
            ("gate_proj", &mut layer.mlp.gate_proj),
            ("down_proj", &mut layer.mlp.down_proj),
            ("up_proj", &mut layer.mlp.up_proj),
        ] {
            let path = format!("language_model.model.layers.{layer_idx}.mlp.{proj}");
            let (gs, bits) = q.for_path(&path);
            if (gs, bits) != (q.group_size, q.bits) {
                requantise_linear(slot, gs, bits)?;
            }
        }
    }
    Ok(())
}

/// Build `Model::new`, apply quantisation, load sanitised weights into the
/// parameter walk, then `eval_params`. Takes the bare [`TextConfig`] so both
/// the gemma4 and gemma4_unified families (which carry the same text config in
/// different envelopes) share one loader.
pub(crate) fn load_model(
    cfg: &Config,
    text_config: &TextConfig,
    model_dir: &Path,
) -> Result<Model, Error> {
    let mut model = Model::new(text_config.clone())?;
    if let Some(q) = cfg.quantization() {
        model = model.try_into_quantized(q.group_size, q.bits)?;
        quantize_mlp_overrides(&mut model, q)?;
    }

    let weights = load_sanitized_weights(
        model_dir,
        text_config.num_hidden_layers,
        text_config.num_kv_shared_layers,
    )?;

    let mut leftover: Vec<String> = Vec::new();
    {
        let mut params = model.parameters_mut().flatten();
        for (k, v) in weights {
            if let Some(slot) = params.get_mut(&*k) {
                **slot = v;
            } else {
                leftover.push(k);
            }
        }
    }

    if !leftover.is_empty() {
        leftover.sort();
        return Err(Error::Other(
            format!(
                "gemma4 loader: {} unbound key(s); first 8: {:?}",
                leftover.len(),
                &leftover.iter().take(8).collect::<Vec<_>>()
            )
            .into(),
        ));
    }
    eval_params(model.parameters()).map_err(Error::Exception)?;
    apply_post_load_memory_policy();
    Ok(model)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::missing_assert_message, reason = "test code")]
    use mlx_rs::nn;
    use mlx_rs::quantization::{MaybeQuantized, Quantizable};

    use super::*;

    #[test]
    fn drops_vision_and_quant_stats() {
        assert!(should_drop("vision_tower.encoder.layer.0.attn.q.weight"));
        assert!(should_drop("multi_modal_projector.proj.weight"));
        assert!(should_drop("audio_tower.layer.0.proj.weight"));
        assert!(should_drop("embed_vision.weight"));
        assert!(should_drop("layers.0.self_attn.rotary_emb.inv_freq"));
        assert!(should_drop("layers.0.self_attn.q_proj.input_max"));
        assert!(!should_drop("model.layers.0.self_attn.q_proj.weight"));
    }

    #[test]
    fn rewrites_language_model_prefix() {
        assert_eq!(
            rewrite_outer_key("language_model.model.layers.0.self_attn.q_proj.weight"),
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            rewrite_outer_key("model.language_model.layers.0.self_attn.q_proj.weight"),
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            rewrite_outer_key("language_model.lm_head.weight"),
            "lm_head.weight"
        );
        assert_eq!(
            rewrite_outer_key("model.layers.0.self_attn.q_proj.weight"),
            "model.layers.0.self_attn.q_proj.weight"
        );
    }

    /// mlx-community 4-bit gemma4 ships the dense MLP projections at 8-bit
    /// while the body is 4-bit. The override pass must build those slots at
    /// 8-bit; a uniform 4-bit body quant otherwise crashes the matmul on the
    /// scale shape.
    #[test]
    fn mlp_overrides_quantize_at_8bit_over_4bit_body() {
        #![allow(clippy::unwrap_used, reason = "test code")]
        let cfg: TextConfig = serde_json::from_value(serde_json::json!({
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "head_dim": 16,
            "global_head_dim": 16,
            "num_key_value_heads": 2,
            "vocab_size": 128,
            "layer_types": ["sliding_attention", "full_attention"],
        }))
        .unwrap();
        let q: QuantizationConfig = serde_json::from_value(serde_json::json!({
            "group_size": 64,
            "bits": 4,
            "language_model.model.layers.0.mlp.gate_proj": { "group_size": 64, "bits": 8 },
        }))
        .unwrap();

        let mut model = Model::new(cfg).unwrap();
        model = model.try_into_quantized(q.group_size, q.bits).unwrap();
        quantize_mlp_overrides(&mut model, &q).unwrap();

        let bits = |slot: &MaybeQuantized<nn::Linear>| match slot {
            MaybeQuantized::Quantized(l) => l.bits,
            MaybeQuantized::Original(_) => panic!("slot not quantized"),
        };
        // Layer 0 gate_proj has an 8-bit override; everything else stays 4-bit.
        assert_eq!(bits(&model.model.layers[0].mlp.gate_proj), 8);
        assert_eq!(bits(&model.model.layers[0].mlp.down_proj), 4);
        assert_eq!(bits(&model.model.layers[1].mlp.gate_proj), 4);
    }
}
