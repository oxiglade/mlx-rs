use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroUsize,
};

use serde::Deserialize;
use serde_json::Value;

use crate::{
    config::RawConfig, AffineQuantization, AttentionKind, Config, ConfigError, LayerQuantization,
    ModelType, ParameterPath, QuantizationConfig, RopeConfig, RopeScaling, TransformerDimensions,
};

#[derive(Deserialize)]
struct WireConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    intermediate_size: usize,
    #[serde(default = "default_attention_heads")]
    num_attention_heads: usize,
    vocab_size: usize,
    #[serde(default = "default_kv_heads")]
    num_key_value_heads: usize,
    #[serde(default)]
    num_global_key_value_heads: Option<usize>,
    #[serde(default = "default_head_dim")]
    head_dim: usize,
    #[serde(default)]
    global_head_dim: Option<usize>,
    #[serde(default)]
    num_kv_shared_layers: usize,
    #[serde(default)]
    hidden_size_per_layer_input: usize,
    #[serde(default)]
    vocab_size_per_layer_input: Option<usize>,
    #[serde(default = "default_sliding_window")]
    sliding_window: usize,
    #[serde(default = "default_sliding_window_pattern")]
    sliding_window_pattern: usize,
    #[serde(default)]
    max_position_embeddings: Option<usize>,
    #[serde(default = "default_epsilon")]
    rms_norm_eps: f32,
    #[serde(default)]
    rope_traditional: bool,
    #[serde(default)]
    rope_parameters: Option<Value>,
    #[serde(default)]
    attention_k_eq_v: bool,
    #[serde(default = "default_logit_softcapping")]
    final_logit_softcapping: Option<f32>,
    #[serde(default = "default_true")]
    use_double_wide_mlp: bool,
    #[serde(default)]
    enable_moe_block: bool,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default)]
    per_layer_config: Option<BTreeMap<String, Value>>,
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    #[serde(default)]
    quantization: Option<Value>,
    #[serde(default)]
    quantization_config: Option<Value>,
}

fn default_attention_heads() -> usize {
    8
}
fn default_kv_heads() -> usize {
    1
}
fn default_head_dim() -> usize {
    256
}
fn default_sliding_window() -> usize {
    512
}
fn default_sliding_window_pattern() -> usize {
    5
}
fn default_epsilon() -> f32 {
    1e-6
}
fn default_logit_softcapping() -> Option<f32> {
    Some(30.0)
}
fn default_true() -> bool {
    true
}

#[derive(Debug, Clone)]
pub(crate) struct ParsedConfig {
    pub(crate) config: Config,
    pub(crate) num_kv_shared_layers: usize,
    pub(crate) first_kv_shared_layer_idx: usize,
    pub(crate) previous_kvs: Vec<usize>,
    pub(crate) layer_head_dims: Vec<usize>,
    pub(crate) layer_kv_heads: Vec<usize>,
    pub(crate) layer_intermediate_sizes: Vec<usize>,
    pub(crate) attention_k_eq_v: bool,
    pub(crate) final_logit_softcapping: Option<f32>,
    pub(crate) hidden_size_per_layer_input: usize,
    pub(crate) vocab_size_per_layer_input: usize,
    pub(crate) full_rope_theta: f32,
    pub(crate) full_partial_rotary_factor: f32,
    pub(crate) sliding_rope_theta: f32,
    pub(crate) sliding_partial_rotary_factor: f32,
}

pub(super) fn invalid(field: &str, reason: &str) -> ConfigError {
    ConfigError::InvalidNumericField {
        field: field.into(),
        reason: reason.into(),
    }
}

pub(super) fn dimension(field: &str, value: usize) -> Result<i32, ConfigError> {
    if value == 0 {
        return Err(invalid(field, "must be positive"));
    }
    i32::try_from(value).map_err(|_| invalid(field, "exceeds MLX dimension range"))
}

fn positive(field: &str, value: f32) -> Result<f32, ConfigError> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(invalid(field, "must be finite and positive"))
    }
}

fn effective_fields(raw: &RawConfig) -> Result<BTreeMap<String, Value>, ConfigError> {
    let mut merged = raw.fields.clone();
    if let Some(nested) = raw
        .fields
        .get("text_config")
        .or_else(|| raw.fields.get("language_config"))
        .filter(|v| !v.is_null())
    {
        let obj = nested
            .as_object()
            .ok_or_else(|| invalid("text_config", "must be a JSON object"))?;
        for (k, v) in obj {
            merged.insert(k.clone(), v.clone());
        }
        if let Some(top_vocab) = raw.fields.get("vocab_size").filter(|v| !v.is_null()) {
            merged
                .entry("vocab_size".into())
                .or_insert_with(|| top_vocab.clone());
        }
        if let Some(top_quant) = raw.fields.get("quantization").filter(|v| !v.is_null()) {
            merged.insert("quantization".into(), top_quant.clone());
        }
    }
    Ok(merged)
}

pub(crate) fn parse(raw: &RawConfig) -> Result<ParsedConfig, ConfigError> {
    if !matches!(
        raw.model_type.as_str(),
        "gemma4" | "gemma4_text" | "gemma4_unified"
    ) {
        return Err(ConfigError::UnsupportedArchitecture(raw.model_type.clone()));
    }
    let fields = effective_fields(raw)?;
    for field in [
        "hidden_size",
        "num_hidden_layers",
        "intermediate_size",
        "vocab_size",
    ] {
        if !fields.contains_key(field) {
            return Err(ConfigError::MissingField {
                field: field.into(),
            });
        }
    }
    let wire: WireConfig = serde_json::from_value(serde_json::to_value(&fields)?)?;
    if wire.enable_moe_block {
        return Err(ConfigError::UnsupportedArchitecture(
            "gemma4 MoE block is not yet supported in mlx-lm".into(),
        ));
    }

    for (field, value) in [
        ("hidden_size", wire.hidden_size),
        ("num_hidden_layers", wire.num_hidden_layers),
        ("intermediate_size", wire.intermediate_size),
        ("num_attention_heads", wire.num_attention_heads),
        ("num_key_value_heads", wire.num_key_value_heads),
        ("head_dim", wire.head_dim),
        ("vocab_size", wire.vocab_size),
    ] {
        dimension(field, value)?;
    }
    if !wire.hidden_size.is_multiple_of(wire.num_attention_heads) {
        return Err(invalid(
            "hidden_size",
            "must be divisible by num_attention_heads",
        ));
    }
    if !wire
        .num_attention_heads
        .is_multiple_of(wire.num_key_value_heads)
    {
        return Err(invalid(
            "num_key_value_heads",
            "must divide num_attention_heads",
        ));
    }
    if !wire.head_dim.is_multiple_of(2) {
        return Err(invalid("head_dim", "RoPE requires an even dimension"));
    }
    if wire.num_kv_shared_layers >= wire.num_hidden_layers {
        return Err(invalid(
            "num_kv_shared_layers",
            "must be strictly less than num_hidden_layers",
        ));
    }
    positive("rms_norm_eps", wire.rms_norm_eps)?;
    if let Some(softcap) = wire.final_logit_softcapping {
        positive("final_logit_softcapping", softcap)?;
    }
    if let Some(max_pos) = wire.max_position_embeddings {
        dimension("max_position_embeddings", max_pos)?;
    }

    let kinds = match wire.layer_types {
        Some(types) => {
            if types.len() != wire.num_hidden_layers {
                return Err(ConfigError::InvalidLayerPattern {
                    expected: wire.num_hidden_layers,
                    actual: types.len(),
                });
            }
            types
        }
        None => {
            dimension("sliding_window_pattern", wire.sliding_window_pattern)?;
            let mut pattern = vec![
                "sliding_attention".to_string();
                wire.sliding_window_pattern.saturating_sub(1)
            ];
            pattern.push("full_attention".to_string());
            pattern
                .into_iter()
                .cycle()
                .take(wire.num_hidden_layers)
                .collect()
        }
    };

    let mut derived_global_head_dim = wire.global_head_dim;
    let mut derived_global_kv_heads = wire.num_global_key_value_heads;
    if let Some(per_layer) = &wire.per_layer_config {
        let mut head_dims = BTreeSet::new();
        let mut kv_heads = BTreeSet::new();
        for (idx, kind) in kinds.iter().enumerate() {
            if kind == "full_attention" {
                if let Some(entry) = per_layer
                    .get(&format!("{idx:02}"))
                    .or_else(|| per_layer.get(&idx.to_string()))
                {
                    if let Some(hd) = entry.get("head_dim").and_then(Value::as_u64) {
                        if let Ok(hd) = usize::try_from(hd) {
                            head_dims.insert(hd);
                        }
                    }
                    if let Some(kvh) = entry.get("num_key_value_heads").and_then(Value::as_u64) {
                        if let Ok(kvh) = usize::try_from(kvh) {
                            kv_heads.insert(kvh);
                        }
                    }
                }
            }
        }
        if derived_global_head_dim.is_none() && head_dims.len() == 1 {
            derived_global_head_dim = head_dims.into_iter().next();
        }
        if derived_global_kv_heads.is_none() && kv_heads.len() == 1 {
            derived_global_kv_heads = kv_heads.into_iter().next();
        }
    }

    let global_head_dim = derived_global_head_dim.unwrap_or(wire.head_dim);
    dimension("global_head_dim", global_head_dim)?;
    if !global_head_dim.is_multiple_of(2) {
        return Err(invalid(
            "global_head_dim",
            "RoPE requires an even dimension",
        ));
    }
    if let Some(g_kv) = derived_global_kv_heads {
        dimension("num_global_key_value_heads", g_kv)?;
        if !wire.num_attention_heads.is_multiple_of(g_kv) {
            return Err(invalid(
                "num_global_key_value_heads",
                "must divide num_attention_heads",
            ));
        }
    }

    let first_kv_shared_layer_idx = wire.num_hidden_layers - wire.num_kv_shared_layers;
    let mut previous_kvs: Vec<usize> = (0..wire.num_hidden_layers).collect();
    if wire.num_kv_shared_layers > 0 {
        let mut latest_by_type = BTreeMap::new();
        for (idx, kind) in kinds.iter().enumerate().take(first_kv_shared_layer_idx) {
            latest_by_type.insert(kind.as_str(), idx);
        }
        for idx in first_kv_shared_layer_idx..wire.num_hidden_layers {
            let kind = kinds[idx].as_str();
            let src = latest_by_type.get(kind).copied().ok_or_else(|| {
                invalid(
                    "num_kv_shared_layers",
                    "shared layer has no preceding non-shared layer of the same attention type",
                )
            })?;
            previous_kvs[idx] = src;
        }
    }

    let mut attention = Vec::with_capacity(wire.num_hidden_layers);
    let mut layer_head_dims = Vec::with_capacity(wire.num_hidden_layers);
    let mut layer_kv_heads = Vec::with_capacity(wire.num_hidden_layers);
    let mut layer_intermediate_sizes = Vec::with_capacity(wire.num_hidden_layers);

    for (idx, kind) in kinds.iter().enumerate() {
        let per_layer_entry = wire.per_layer_config.as_ref().and_then(|map| {
            map.get(&format!("{idx:02}"))
                .or_else(|| map.get(&idx.to_string()))
        });
        let override_hd = per_layer_entry
            .and_then(|e| e.get("head_dim"))
            .and_then(Value::as_u64)
            .and_then(|v| usize::try_from(v).ok());
        let override_kvh = per_layer_entry
            .and_then(|e| e.get("num_key_value_heads"))
            .and_then(Value::as_u64)
            .and_then(|v| usize::try_from(v).ok());

        match kind.as_str() {
            "full_attention" => {
                attention.push(AttentionKind::Full);
                let hd = override_hd.unwrap_or(global_head_dim);
                let kvh = override_kvh
                    .or(derived_global_kv_heads)
                    .unwrap_or(wire.num_key_value_heads);
                dimension("layer head_dim", hd)?;
                dimension("layer num_key_value_heads", kvh)?;
                if !hd.is_multiple_of(2) || !wire.num_attention_heads.is_multiple_of(kvh) {
                    return Err(invalid(
                        "per_layer_config",
                        "invalid full_attention head_dim or num_key_value_heads",
                    ));
                }
                layer_head_dims.push(hd);
                layer_kv_heads.push(kvh);
            }
            "sliding_attention" => {
                dimension("sliding_window", wire.sliding_window)?;
                let window = NonZeroUsize::new(wire.sliding_window)
                    .ok_or_else(|| invalid("sliding_window", "must be positive"))?;
                attention.push(AttentionKind::Sliding { window });
                let hd = override_hd.unwrap_or(wire.head_dim);
                let kvh = override_kvh.unwrap_or(wire.num_key_value_heads);
                dimension("layer head_dim", hd)?;
                dimension("layer num_key_value_heads", kvh)?;
                if !hd.is_multiple_of(2) || !wire.num_attention_heads.is_multiple_of(kvh) {
                    return Err(invalid(
                        "per_layer_config",
                        "invalid sliding_attention head_dim or num_key_value_heads",
                    ));
                }
                layer_head_dims.push(hd);
                layer_kv_heads.push(kvh);
            }
            other => {
                return Err(ConfigError::UnsupportedArchitecture(format!(
                    "gemma4 attention {other}"
                )));
            }
        }

        let is_kv_shared = idx >= first_kv_shared_layer_idx && first_kv_shared_layer_idx > 0;
        let mult = if wire.use_double_wide_mlp && is_kv_shared {
            2
        } else {
            1
        };
        let inter = wire
            .intermediate_size
            .checked_mul(mult)
            .ok_or_else(|| invalid("intermediate_size", "double-wide MLP size overflow"))?;
        dimension("intermediate_size", inter)?;
        layer_intermediate_sizes.push(inter);
    }

    let (
        full_rope_theta,
        full_partial_rotary_factor,
        sliding_rope_theta,
        sliding_partial_rotary_factor,
    ) = parse_gemma4_rope_parameters(wire.rope_parameters.as_ref())?;

    let vocab_size_per_layer_input = wire.vocab_size_per_layer_input.unwrap_or(wire.vocab_size);
    if wire.hidden_size_per_layer_input > 0 {
        dimension(
            "hidden_size_per_layer_input",
            wire.hidden_size_per_layer_input,
        )?;
        dimension("vocab_size_per_layer_input", vocab_size_per_layer_input)?;
        let total_pli = wire
            .num_hidden_layers
            .checked_mul(wire.hidden_size_per_layer_input)
            .ok_or_else(|| {
                invalid(
                    "hidden_size_per_layer_input",
                    "per-layer input width overflow",
                )
            })?;
        dimension("per-layer projection width", total_pli)?;
    }

    if wire.quantization.is_none() {
        if let Some(legacy) = &wire.quantization_config {
            let method = legacy
                .get("quant_method")
                .and_then(Value::as_str)
                .unwrap_or("missing quant_method");
            return Err(ConfigError::UnsupportedQuantization(format!(
                "legacy quantization_config {method}"
            )));
        }
    }
    let quantization = wire
        .quantization
        .as_ref()
        .map(parse_quantization)
        .transpose()?;

    let config = Config {
        model_type: ModelType::new(raw.model_type.as_str()),
        dimensions: TransformerDimensions {
            hidden_size: wire.hidden_size,
            layer_count: wire.num_hidden_layers,
            intermediate_size: wire.intermediate_size,
            attention_heads: wire.num_attention_heads,
            kv_heads: wire.num_key_value_heads,
            head_dim: wire.head_dim,
            vocabulary_size: wire.vocab_size,
            max_positions: wire.max_position_embeddings,
            rms_norm_epsilon: wire.rms_norm_eps,
        },
        rope: RopeConfig {
            dimensions: wire.head_dim,
            theta: sliding_rope_theta,
            traditional: wire.rope_traditional,
            scaling: RopeScaling::None,
        },
        attention,
        tie_word_embeddings: wire.tie_word_embeddings,
        attention_bias: wire.attention_bias,
        mlp_bias: wire.mlp_bias,
        quantization,
    };

    Ok(ParsedConfig {
        config,
        num_kv_shared_layers: wire.num_kv_shared_layers,
        first_kv_shared_layer_idx,
        previous_kvs,
        layer_head_dims,
        layer_kv_heads,
        layer_intermediate_sizes,
        attention_k_eq_v: wire.attention_k_eq_v,
        final_logit_softcapping: wire.final_logit_softcapping,
        hidden_size_per_layer_input: wire.hidden_size_per_layer_input,
        vocab_size_per_layer_input,
        full_rope_theta,
        full_partial_rotary_factor,
        sliding_rope_theta,
        sliding_partial_rotary_factor,
    })
}

fn parse_gemma4_rope_parameters(
    value: Option<&Value>,
) -> Result<(f32, f32, f32, f32), ConfigError> {
    let mut full_theta = 1_000_000.0_f32;
    let mut full_factor = 0.25_f32;
    let mut sliding_theta = 10_000.0_f32;
    let mut sliding_factor = 1.0_f32;

    let Some(value) = value.filter(|v| !v.is_null()) else {
        return Ok((full_theta, full_factor, sliding_theta, sliding_factor));
    };
    let obj = value
        .as_object()
        .ok_or_else(|| ConfigError::UnsupportedRope(value.to_string()))?;

    if let Some(full) = obj.get("full_attention").and_then(Value::as_object) {
        if let Some(kind) = full
            .get("rope_type")
            .or_else(|| full.get("type"))
            .and_then(Value::as_str)
        {
            if !matches!(kind, "proportional" | "default") {
                return Err(ConfigError::UnsupportedRope(kind.to_string()));
            }
        }
        if let Some(theta) = full.get("rope_theta").and_then(Value::as_f64) {
            full_theta = positive("rope_parameters.full_attention.rope_theta", theta as f32)?;
        }
        if let Some(factor) = full.get("partial_rotary_factor").and_then(Value::as_f64) {
            let f = factor as f32;
            if !(f.is_finite() && f > 0.0 && f <= 1.0) {
                return Err(invalid(
                    "rope_parameters.full_attention.partial_rotary_factor",
                    "must be in (0, 1]",
                ));
            }
            full_factor = f;
        }
    }

    if let Some(sliding) = obj.get("sliding_attention").and_then(Value::as_object) {
        if let Some(kind) = sliding
            .get("rope_type")
            .or_else(|| sliding.get("type"))
            .and_then(Value::as_str)
        {
            if !matches!(kind, "default" | "proportional") {
                return Err(ConfigError::UnsupportedRope(kind.to_string()));
            }
        }
        if let Some(theta) = sliding.get("rope_theta").and_then(Value::as_f64) {
            sliding_theta = positive("rope_parameters.sliding_attention.rope_theta", theta as f32)?;
        }
        if let Some(factor) = sliding.get("partial_rotary_factor").and_then(Value::as_f64) {
            let f = factor as f32;
            if !(f.is_finite() && f > 0.0 && f <= 1.0) {
                return Err(invalid(
                    "rope_parameters.sliding_attention.partial_rotary_factor",
                    "must be in (0, 1]",
                ));
            }
            sliding_factor = f;
        }
    }

    Ok((full_theta, full_factor, sliding_theta, sliding_factor))
}

fn parse_affine(value: &Value) -> Result<AffineQuantization, ConfigError> {
    if let Some(mode) = value.get("mode") {
        if mode.as_str() != Some("affine") {
            return Err(ConfigError::UnsupportedQuantization(mode.to_string()));
        }
    }
    let group_size = value
        .get("group_size")
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .and_then(NonZeroUsize::new)
        .ok_or_else(|| {
            ConfigError::UnsupportedQuantization("invalid quantization.group_size".into())
        })?;
    if !matches!(group_size.get(), 32 | 64 | 128) {
        return Err(ConfigError::UnsupportedQuantization(format!(
            "group_size {}",
            group_size.get()
        )));
    }
    let bits = value
        .get("bits")
        .and_then(Value::as_u64)
        .and_then(|n| u8::try_from(n).ok())
        .ok_or_else(|| ConfigError::UnsupportedQuantization("invalid quantization.bits".into()))?;
    if !matches!(bits, 2 | 3 | 4 | 6 | 8) {
        return Err(ConfigError::UnsupportedQuantization(format!("bits {bits}")));
    }
    Ok(AffineQuantization { group_size, bits })
}

fn normalize_quant_path(path: &str) -> Option<&str> {
    for prefix in [
        "model.language_model.model.",
        "language_model.model.",
        "model.language_model.",
        "language_model.",
        "model.",
    ] {
        if let Some(stripped) = path.strip_prefix(prefix) {
            return (!stripped.is_empty()).then_some(stripped);
        }
    }
    (path == "lm_head").then_some("lm_head")
}

fn parse_quantization(value: &Value) -> Result<QuantizationConfig, ConfigError> {
    let fields = value
        .as_object()
        .ok_or_else(|| ConfigError::UnsupportedQuantization(value.to_string()))?;
    let default = parse_affine(value)?;
    let mut layers = BTreeMap::new();
    for (key, entry) in fields {
        if matches!(key.as_str(), "group_size" | "bits" | "mode") {
            continue;
        }
        let Some(canonical) = normalize_quant_path(key) else {
            return Err(ConfigError::UnsupportedQuantization(format!(
                "invalid quantization path {key}"
            )));
        };
        let layer = match entry {
            Value::Bool(false) => LayerQuantization::Unquantized,
            Value::Bool(true) => LayerQuantization::Affine(default.clone()),
            Value::Object(obj) => {
                if obj
                    .keys()
                    .any(|f| !matches!(f.as_str(), "group_size" | "bits" | "mode"))
                {
                    return Err(ConfigError::UnsupportedQuantization(format!(
                        "unsupported affine fields for {key}"
                    )));
                }
                LayerQuantization::Affine(parse_affine(entry)?)
            }
            _ => {
                return Err(ConfigError::UnsupportedQuantization(format!(
                    "invalid override for {key}"
                )));
            }
        };
        layers.insert(ParameterPath::new(canonical), layer);
    }
    Ok(QuantizationConfig { default, layers })
}
