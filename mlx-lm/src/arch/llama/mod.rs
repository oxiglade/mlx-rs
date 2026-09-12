use super::{ArchitectureFactory, DecoderModel, ParsedArchitecture};
use crate::{
    config::RawConfig,
    weights::{WeightDisposition, WeightManifest},
    ConfigError, LoadError, ParameterPath,
};

mod config;
mod decoder;
mod rope;
#[cfg(test)]
mod tests;

pub(crate) use config::ParsedConfig;
pub(crate) struct Factory;

impl ArchitectureFactory for Factory {
    fn parse_config(&self, raw: &RawConfig) -> Result<ParsedArchitecture, ConfigError> {
        Ok(ParsedArchitecture::Llama(config::parse(raw)?))
    }

    fn build(
        &self,
        parsed: ParsedArchitecture,
        weights: &WeightManifest,
    ) -> Result<Box<dyn DecoderModel>, LoadError> {
        Ok(Box::new(build_decoder(parsed, weights)?))
    }

    #[cfg(test)]
    fn map_safetensors_key(&self, external: &str) -> WeightDisposition {
        map_key(external, false)
    }

    fn map_gguf_key(&self, _external: &str) -> WeightDisposition {
        WeightDisposition::Reject
    }
}

fn layer_suffix(key: &str) -> Option<&str> {
    let (index, suffix) = key.strip_prefix("model.layers.")?.split_once('.')?;
    let number = index.parse::<usize>().ok()?;
    if number.to_string() != index {
        return None;
    }
    Some(suffix)
}

fn is_matrix_key(key: &str) -> bool {
    if matches!(key, "model.embed_tokens.weight" | "lm_head.weight") {
        return true;
    }
    matches!(
        layer_suffix(key),
        Some(
            "self_attn.q_proj.weight"
                | "self_attn.k_proj.weight"
                | "self_attn.v_proj.weight"
                | "self_attn.o_proj.weight"
                | "mlp.gate_proj.weight"
                | "mlp.up_proj.weight"
                | "mlp.down_proj.weight"
        )
    )
}

fn map_key(key: &str, tied: bool) -> WeightDisposition {
    if tied && key == "lm_head.weight" {
        return WeightDisposition::Ignore {
            reason: "llama.redundant_tied_lm_head",
        };
    }
    if key == "model.rotary_emb.inv_freq"
        || layer_suffix(key) == Some("self_attn.rotary_emb.inv_freq")
    {
        return WeightDisposition::Ignore {
            reason: "llama.recomputed_rotary_frequencies",
        };
    }
    let norm = key == "model.norm.weight"
        || matches!(
            layer_suffix(key),
            Some("input_layernorm.weight" | "post_attention_layernorm.weight")
        );
    let matrix = key.rsplit_once('.').is_some_and(|(group, leaf)| {
        matches!(leaf, "weight" | "scales" | "biases" | "bias")
            && is_matrix_key(&format!("{group}.weight"))
            && (leaf != "bias" || (group != "model.embed_tokens" && group != "lm_head"))
    });
    if norm || matrix {
        WeightDisposition::Parameter(ParameterPath::new(key))
    } else {
        WeightDisposition::Reject
    }
}

fn build_decoder(
    parsed: ParsedArchitecture,
    weights: &WeightManifest,
) -> Result<decoder::Decoder, LoadError> {
    let ParsedArchitecture::Llama(parsed) = parsed else {
        return Err(
            ConfigError::UnsupportedArchitecture("expected llama configuration".into()).into(),
        );
    };
    let tied = parsed.config.tie_word_embeddings;
    let mut decoder = decoder::Decoder::new(parsed, weights)?;
    weights.load_with(&mut decoder.weight_projection()?, |key| map_key(key, tied))?;
    for parameter in decoder.weight_projection()?.values() {
        parameter.eval()?;
    }
    Ok(decoder)
}
