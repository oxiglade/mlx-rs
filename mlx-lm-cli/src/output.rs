use std::{
    collections::BTreeMap,
    io::{self, Write},
};

use mlx_lm::{
    AttentionKind, Config, FinishReason, GenerationEvent, LayerQuantization, Model, RopeScaling,
};
use serde::Serialize;
use serde_json::{json, Value};

use crate::{
    args::{GenerationFormat, InfoFormat},
    error::CliError,
};

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OutputFinish {
    Stop,
    Length,
}

#[derive(Debug, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub(crate) enum OutputEvent {
    Prefill {
        processed: usize,
        total: usize,
    },
    Token {
        token_id: u32,
        text: String,
        finish_reason: Option<OutputFinish>,
    },
}

pub(crate) fn encode_event(event: GenerationEvent) -> Result<OutputEvent, CliError> {
    match event {
        GenerationEvent::Prefill {
            processed, total, ..
        } => Ok(OutputEvent::Prefill { processed, total }),
        GenerationEvent::Token {
            token_id,
            text,
            finish_reason,
            ..
        } => {
            let finish_reason = match finish_reason {
                None => None,
                Some(FinishReason::Stop) => Some(OutputFinish::Stop),
                Some(FinishReason::Length) => Some(OutputFinish::Length),
                Some(_) => return Err(CliError::UnsupportedEvent),
            };
            Ok(OutputEvent::Token {
                token_id: token_id.into(),
                text,
                finish_reason,
            })
        }
        _ => Err(CliError::UnsupportedEvent),
    }
}

pub(crate) fn write_event(
    output: &mut impl Write,
    event: &OutputEvent,
    format: GenerationFormat,
) -> Result<(), CliError> {
    match format {
        GenerationFormat::Text => {
            if let OutputEvent::Token { text, .. } = event {
                write_record(output, text.as_bytes())?;
            }
        }
        GenerationFormat::Jsonl => {
            #[derive(Serialize)]
            struct Record<'a> {
                version: u8,
                #[serde(flatten)]
                event: &'a OutputEvent,
            }
            let mut bytes = serde_json::to_vec(&Record { version: 1, event })?;
            bytes.push(b'\n');
            write_record(output, &bytes)?;
        }
    }
    Ok(())
}

fn write_record(output: &mut impl Write, bytes: &[u8]) -> Result<(), CliError> {
    output.write_all(bytes)?;
    loop {
        match output.flush() {
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            result => return result.map_err(CliError::Output),
        }
    }
}

#[derive(Serialize)]
pub(crate) struct InfoRecord<'a> {
    version: u8,
    config: Value,
    tokenizer: TokenizerRecord<'a>,
    hub_provenance: Option<ProvenanceRecord<'a>>,
}
#[derive(Serialize)]
struct TokenizerRecord<'a> {
    bos_token: Option<&'a str>,
    eos_tokens: Vec<u32>,
}
#[derive(Serialize)]
struct ProvenanceRecord<'a> {
    repo: &'a str,
    requested_revision: &'a str,
    resolved_revision: &'a str,
}

pub(crate) fn model_info(model: &Model) -> InfoRecord<'_> {
    #[cfg(feature = "hf-hub")]
    let hub_provenance = model.hub_provenance().map(|provenance| ProvenanceRecord {
        repo: &provenance.repo,
        requested_revision: &provenance.requested_revision,
        resolved_revision: &provenance.resolved_revision,
    });
    #[cfg(not(feature = "hf-hub"))]
    let hub_provenance = None;
    InfoRecord {
        version: 1,
        config: config_value(model.config()),
        tokenizer: TokenizerRecord {
            bos_token: model.tokenizer().bos_token(),
            eos_tokens: model
                .tokenizer()
                .eos_tokens()
                .iter()
                .copied()
                .map(u32::from)
                .collect(),
        },
        hub_provenance,
    }
}

fn config_value(config: &Config) -> Value {
    let d = &config.dimensions;
    let scaling = match config.rope.scaling {
        RopeScaling::None => json!({"kind":"none"}),
        RopeScaling::Linear { factor } => json!({"kind":"linear", "factor":factor}),
        RopeScaling::Llama3 {
            factor,
            low_frequency_factor,
            high_frequency_factor,
            original_max_positions,
        } => json!({
            "kind":"llama3", "factor":factor, "low_frequency_factor":low_frequency_factor,
            "high_frequency_factor":high_frequency_factor, "original_max_positions":original_max_positions,
        }),
    };
    let attention: Vec<_> = config
        .attention
        .iter()
        .map(|kind| match kind {
            AttentionKind::Full => json!({"kind":"full"}),
            AttentionKind::Sliding { window } => json!({"kind":"sliding", "window":window.get()}),
        })
        .collect();
    let quantization = config.quantization.as_ref().map(|quantization| {
        let layers: BTreeMap<_, _> = quantization.layers.iter().map(|(path, layer)| {
            let value = match layer {
                LayerQuantization::Unquantized => json!({"kind":"unquantized"}),
                LayerQuantization::Affine(affine) => json!({"kind":"affine", "group_size":affine.group_size.get(), "bits":affine.bits}),
            };
            (path.as_str(), value)
        }).collect();
        json!({"default":{"group_size":quantization.default.group_size.get(), "bits":quantization.default.bits}, "layers":layers})
    });
    json!({
        "model_type":config.model_type.as_str(),
        "dimensions": {
            "hidden_size":d.hidden_size, "layer_count":d.layer_count, "intermediate_size":d.intermediate_size,
            "attention_heads":d.attention_heads, "kv_heads":d.kv_heads, "head_dim":d.head_dim,
            "vocabulary_size":d.vocabulary_size, "max_positions":d.max_positions, "rms_norm_epsilon":d.rms_norm_epsilon,
        },
        "rope":{"dimensions":config.rope.dimensions, "theta":config.rope.theta, "traditional":config.rope.traditional, "scaling":scaling},
        "attention":attention, "tie_word_embeddings":config.tie_word_embeddings,
        "attention_bias":config.attention_bias, "mlp_bias":config.mlp_bias, "quantization":quantization,
    })
}

pub(crate) fn write_info(
    output: &mut impl Write,
    info: &InfoRecord<'_>,
    format: InfoFormat,
) -> Result<(), CliError> {
    let mut bytes = match format {
        InfoFormat::Json => serde_json::to_vec(info)?,
        InfoFormat::Text => format!(
            "Config: {}\nTokenizer: {}\nHub provenance: {}",
            serde_json::to_string_pretty(&info.config)?,
            serde_json::to_string(&info.tokenizer)?,
            serde_json::to_string(&info.hub_provenance)?,
        )
        .into_bytes(),
    };
    bytes.push(b'\n');
    write_record(output, &bytes)
}

#[cfg(test)]
mod tests;
