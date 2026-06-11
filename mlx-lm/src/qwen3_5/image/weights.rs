//! Vision-tower-aware weight loader. Splits a checkpoint into the
//! shared language-model parameters (loaded via the text-side
//! [`crate::qwen3_5::text::weights::load_sanitized_weights`] +
//! [`crate::qwen3_5::text::weights::bucket_key`] pipeline) and the
//! vision-tower parameters consumed by [`VisionModel`].

use std::path::Path;

use mlx_rs::{module::ModuleParameters, transforms::eval_params};

use crate::config::ModelConfig as Config;
use crate::error::Error;
use crate::loader::apply_post_load_memory_policy;
use crate::qwen3_5::image::vision::VisionModel;
use crate::qwen3_5::text::config::ModelConfig;
use crate::qwen3_5::text::weights::{
    bucket_key, load_sanitized_weights, quantize_language_model, Bucketed, Qwen35Model,
};

/// Load both the language model and the vision tower from the same
/// checkpoint. Vision weights are bf16 (not quantised in chandra-ocr-2),
/// so the vision module is not run through `quantize_language_model`.
pub(crate) fn load_full_model(
    cfg: &Config,
    env: &ModelConfig,
    model_dir: &Path,
) -> Result<(Qwen35Model, VisionModel, Vec<String>), Error> {
    let mut lm = Qwen35Model::with_mlp(env.text_config.clone())?;
    if let Some(q) = cfg.quantization() {
        quantize_language_model(&mut lm, q)?;
    }
    let vision_cfg = env
        .vision_config
        .as_ref()
        .ok_or_else(|| Error::config("qwen3_5 load_full_model: config has no vision_config"))?;
    let mut vision = VisionModel::new(vision_cfg)?;
    let weights = load_sanitized_weights(model_dir)?;

    let mut leftover = Vec::new();
    {
        let mut lm_params = lm.parameters_mut().flatten();
        let mut vision_params = vision.parameters_mut().flatten();
        for (k, v) in weights {
            match bucket_key(k) {
                Bucketed::Language(p) => {
                    if let Some(slot) = lm_params.get_mut(&*p) {
                        **slot = v;
                    } else {
                        leftover.push(format!("language_model.{p}"));
                    }
                }
                Bucketed::Vision(p) => {
                    if let Some(slot) = vision_params.get_mut(&*p) {
                        **slot = v;
                    } else {
                        leftover.push(format!("vision_tower.{p}"));
                    }
                }
                Bucketed::Other(p) => leftover.push(p),
            }
        }
    }

    eval_params(lm.parameters()).map_err(Error::Exception)?;
    eval_params(vision.parameters()).map_err(Error::Exception)?;
    apply_post_load_memory_policy();
    leftover.sort();
    Ok((lm, vision, leftover))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::print_stderr, reason = "test code")]
    use mlx_rs::module::ModuleParametersExt;

    use super::*;

    #[test]
    #[ignore = "requires local model files at ~/MLXModels/chandra2/chandra-ocr-2-mlx-q8"]
    fn loads_chandra_q8_full_model_with_vision() {
        let home = std::env::var("HOME").unwrap();
        let dir = std::path::PathBuf::from(home).join("MLXModels/chandra2/chandra-ocr-2-mlx-q8");
        let cfg = Config::from_dir(&dir).expect("parse config");
        let env = cfg.family.as_qwen35().expect("expected qwen3_5 family");
        let (lm, vision, leftover) = load_full_model(&cfg, env, &dir).expect("load full model");
        if !leftover.is_empty() {
            eprintln!("unexpected leftover keys ({}):", leftover.len());
            for k in &leftover[..leftover.len().min(20)] {
                eprintln!("  {k}");
            }
            panic!("unexpected leftover keys");
        }
        lm.eval().expect("eval LM");
        vision.eval().expect("eval vision");
    }
}
