use crate::{weights::WeightDisposition, ParameterPath};

const GLOBAL: &[(&str, &str, bool)] = &[
    ("token_embd", "model.embed_tokens", true),
    ("output", "lm_head", true),
    ("output_norm", "model.norm", false),
];
const LAYER: &[(&str, &str, bool)] = &[
    ("attn_norm", "input_layernorm", false),
    ("ffn_norm", "post_attention_layernorm", false),
    ("attn_q", "self_attn.q_proj", true),
    ("attn_k", "self_attn.k_proj", true),
    ("attn_v", "self_attn.v_proj", true),
    ("attn_output", "self_attn.o_proj", true),
    ("ffn_gate", "mlp.gate_proj", true),
    ("ffn_up", "mlp.up_proj", true),
    ("ffn_down", "mlp.down_proj", true),
    ("attn_q_norm", "self_attn.q_norm", false),
    ("attn_k_norm", "self_attn.k_norm", false),
];

pub(super) fn map(key: &str, qwen3: bool) -> WeightDisposition {
    let Some((group, suffix)) = key.rsplit_once('.') else {
        return WeightDisposition::Reject;
    };
    let found = if let Some(layer) = group.strip_prefix("blk.") {
        let Some((index, name)) = layer.split_once('.') else {
            return WeightDisposition::Reject;
        };
        if index
            .parse::<usize>()
            .ok()
            .is_none_or(|n| n.to_string() != index)
        {
            return WeightDisposition::Reject;
        }
        LAYER
            .iter()
            .find(|(external, _, _)| {
                *external == name && (qwen3 || !matches!(name, "attn_q_norm" | "attn_k_norm"))
            })
            .map(|(_, canonical, matrix)| (format!("model.layers.{index}.{canonical}"), *matrix))
    } else {
        GLOBAL
            .iter()
            .find(|(external, _, _)| *external == group)
            .map(|(_, canonical, matrix)| ((*canonical).to_owned(), *matrix))
    };
    match found {
        Some((canonical, matrix))
            if suffix == "weight" || (matrix && matches!(suffix, "scales" | "biases")) =>
        {
            WeightDisposition::Parameter(ParameterPath::new(format!("{canonical}.{suffix}")))
        }
        _ => WeightDisposition::Reject,
    }
}

pub(crate) fn external(key: &str) -> String {
    let Some((group, suffix)) = key.rsplit_once('.') else {
        return key.to_owned();
    };
    if let Some(layer) = group.strip_prefix("model.layers.") {
        if let Some((index, name)) = layer.split_once('.') {
            if let Some((external, _, _)) =
                LAYER.iter().find(|(_, canonical, _)| *canonical == name)
            {
                return format!("blk.{index}.{external}.{suffix}");
            }
        }
    }
    GLOBAL
        .iter()
        .find(|(_, canonical, _)| *canonical == group)
        .map_or_else(
            || key.to_owned(),
            |(external, _, _)| format!("{external}.{suffix}"),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhaustive_table_and_suffixes() {
        for qwen3 in [false, true] {
            for (external, canonical, matrix) in GLOBAL.iter().chain(LAYER) {
                let layer = LAYER.iter().any(|row| row.0 == *external);
                for suffix in ["weight", "scales", "biases", "bias", "unknown"] {
                    let key = if layer {
                        format!("blk.12.{external}.{suffix}")
                    } else {
                        format!("{external}.{suffix}")
                    };
                    let admitted = (suffix == "weight"
                        || (*matrix && matches!(suffix, "scales" | "biases")))
                        && (qwen3 || !matches!(*external, "attn_q_norm" | "attn_k_norm"));
                    match map(&key, qwen3) {
                        WeightDisposition::Parameter(path) => {
                            assert!(admitted, "{key}");
                            let expected = if layer {
                                format!("model.layers.12.{canonical}.{suffix}")
                            } else {
                                format!("{canonical}.{suffix}")
                            };
                            assert_eq!(path.as_str(), expected);
                            assert_eq!(super::external(path.as_str()), key);
                        }
                        WeightDisposition::Reject => assert!(!admitted, "{key}"),
                        WeightDisposition::Ignore { .. } => panic!("GGUF has no ignore rule"),
                    }
                }
            }
            for key in [
                "blk.00.attn_q.weight",
                "blk.-1.attn_k.weight",
                "blk.+1.attn_k.weight",
                "rope_freqs.weight",
                "model.embed_tokens.weight",
                "blk.0.attn_q_norm.scales",
            ] {
                assert!(
                    matches!(map(key, qwen3), WeightDisposition::Reject),
                    "{key}"
                );
            }
        }
    }
}
