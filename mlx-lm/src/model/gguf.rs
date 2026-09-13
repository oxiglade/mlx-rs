//! The admitted GGUF profile uses explicit standard metadata keys. Unknown vendor
//! metadata cannot be enumerated; converted layouts do not identify original GGML types.
use crate::{
    arch::ParsedArchitecture,
    config::RawConfig,
    weights::{external_error, WeightManifest},
    ConfigError, LayerQuantization, LoadError, Model, ParameterPath, TokenId, Tokenizer,
};
use mlx_rs::{
    io::{GgufFile, GgufMetadataValue},
    Array, Dtype,
};
use serde_json::json;
use std::rc::Rc;

pub(crate) fn load(file: GgufFile, tokenizer: Tokenizer) -> Result<Model, LoadError> {
    // Source preparation must not inherit a caller's GPU Load primitive.
    mlx_rs::with_stream(&mlx_rs::Stream::cpu(), || {
        let architecture = architecture(&file)?;
        let factory = crate::arch::factory(&architecture)?;
        let mut weights =
            WeightManifest::from_gguf(&file)?.normalize_gguf(|key| factory.map_gguf_key(key))?;
        let parsed = factory
            .parse_gguf_config(&file, &weights)
            .map_err(external_load_error)?;
        let config = match &parsed {
            ParsedArchitecture::Llama(parsed) => &parsed.config,
            ParsedArchitecture::Qwen3(parsed) => &parsed.config,
        };
        weights
            .validate_gguf_shapes(config)
            .map_err(external_error)?;
        tokenizer.validate_gguf(&file, config.dimensions.vocabulary_size)?;
        if architecture == "llama" {
            let d = &config.dimensions;
            for layer in 0..d.layer_count {
                for (projection, heads) in [("q_proj", d.attention_heads), ("k_proj", d.kv_heads)] {
                    weights.permute_gguf_rows(
                        &ParameterPath::new(format!("model.layers.{layer}.self_attn.{projection}")),
                        heads,
                        d.head_dim,
                    )?;
                }
            }
        }
        let decoder = factory
            .build(parsed, &weights)
            .map_err(external_load_error)?;
        let config = decoder.config().clone();
        Ok(Model {
            #[cfg(feature = "hf-hub")]
            hub_provenance: None,
            decoder,
            tokenizer,
            config,
            identity: Rc::new(()),
        })
    })
}

fn external_load_error(error: LoadError) -> LoadError {
    match error {
        LoadError::Weights(error) => LoadError::Weights(external_error(error)),
        other => other,
    }
}

fn missing(key: &str) -> LoadError {
    ConfigError::MissingField { field: key.into() }.into()
}
fn invalid(key: &str, reason: &str) -> LoadError {
    ConfigError::InvalidNumericField {
        field: key.into(),
        reason: reason.into(),
    }
    .into()
}
fn mistyped(key: &str, expected: &'static str, value: &GgufMetadataValue) -> LoadError {
    let actual = match value {
        GgufMetadataValue::Array(array) => {
            format!("array shape {:?}, dtype {:?}", array.shape(), array.dtype())
        }
        GgufMetadataValue::String(_) => "string".into(),
        GgufMetadataValue::Strings(_) => "string array".into(),
    };
    ConfigError::InvalidGgufMetadata {
        key: key.into(),
        expected,
        actual,
    }
    .into()
}
fn unsupported(key: &str, value: impl ToString) -> LoadError {
    ConfigError::UnsupportedGgufMetadata {
        key: key.into(),
        value: value.to_string(),
    }
    .into()
}

fn item<T: mlx_rs::ArrayElement>(array: &Array, key: &str) -> Result<T, LoadError> {
    array.try_item_exact::<T>().map_err(|error| match error {
        mlx_rs::error::ConversionError::Exception(source) => {
            LoadError::Gguf(mlx_rs::io::GgufError::Exception(source))
        }
        other => ConfigError::InvalidGgufMetadata {
            key: key.into(),
            expected: "validated scalar",
            actual: other.to_string(),
        }
        .into(),
    })
}

struct Metadata<'a>(&'a GgufFile);
impl Metadata<'_> {
    fn string(&self, key: &str) -> Result<Option<String>, LoadError> {
        self.0
            .get_metadata(key)?
            .map(|value| match value {
                GgufMetadataValue::String(text) => Ok(text),
                other => Err(mistyped(key, "string", &other)),
            })
            .transpose()
    }
    fn scalar(
        &self,
        key: &str,
        expected: &'static str,
        dtypes: &[Dtype],
    ) -> Result<Option<Array>, LoadError> {
        self.0
            .get_metadata(key)?
            .map(|value| match &value {
                GgufMetadataValue::Array(array)
                    if array.shape().is_empty() && dtypes.contains(&array.dtype()) =>
                {
                    Ok(array.clone())
                }
                _ => Err(mistyped(key, expected, &value)),
            })
            .transpose()
    }
    fn integer(&self, key: &str) -> Result<Option<i128>, LoadError> {
        self.scalar(
            key,
            "integral scalar excluding bool",
            &[
                Dtype::Uint8,
                Dtype::Uint16,
                Dtype::Uint32,
                Dtype::Uint64,
                Dtype::Int8,
                Dtype::Int16,
                Dtype::Int32,
                Dtype::Int64,
            ],
        )?
        .map(|a| {
            Ok(match a.dtype() {
                Dtype::Uint8 => i128::from(item::<u8>(&a, key)?),
                Dtype::Uint16 => i128::from(item::<u16>(&a, key)?),
                Dtype::Uint32 => i128::from(item::<u32>(&a, key)?),
                Dtype::Uint64 => i128::from(item::<u64>(&a, key)?),
                Dtype::Int8 => i128::from(item::<i8>(&a, key)?),
                Dtype::Int16 => i128::from(item::<i16>(&a, key)?),
                Dtype::Int32 => i128::from(item::<i32>(&a, key)?),
                Dtype::Int64 => i128::from(item::<i64>(&a, key)?),
                _ => unreachable!("validated integral scalar"),
            })
        })
        .transpose()
    }
    fn size(&self, key: &str, required: bool) -> Result<Option<usize>, LoadError> {
        match self.integer(key)? {
            Some(n) if n > 0 && n <= i128::from(i32::MAX) => Ok(Some(n as usize)),
            Some(_) => Err(invalid(
                key,
                "must be positive and within MLX dimension range",
            )),
            None if required => Err(missing(key)),
            None => Ok(None),
        }
    }
    fn required_size(&self, key: &str) -> Result<usize, LoadError> {
        self.size(key, true)?.ok_or_else(|| missing(key))
    }
    fn real(&self, key: &str, required: bool) -> Result<Option<f32>, LoadError> {
        let value = self
            .scalar(key, "F32 scalar", &[Dtype::Float32])?
            .map(|a| item::<f32>(&a, key))
            .transpose()?;
        match value {
            Some(n) if n.is_finite() && n > 0.0 => Ok(Some(n)),
            Some(_) => Err(invalid(key, "must be finite and positive")),
            None if required => Err(missing(key)),
            None => Ok(None),
        }
    }
    fn boolean(&self, key: &str) -> Result<Option<bool>, LoadError> {
        self.scalar(key, "bool scalar", &[Dtype::Bool])?
            .map(|a| item::<bool>(&a, key))
            .transpose()
    }
    fn token_id(&self, key: &str) -> Result<Option<TokenId>, LoadError> {
        self.integer(key)?
            .map(|n| {
                u32::try_from(n)
                    .map(TokenId::from)
                    .map_err(|_| invalid(key, "must be a nonnegative u32 token ID"))
            })
            .transpose()
    }
}

fn architecture(file: &GgufFile) -> Result<String, LoadError> {
    let value = Metadata(file)
        .string("general.architecture")?
        .ok_or_else(|| missing("general.architecture"))?;
    match value.as_str() {
        "llama" | "qwen3" => Ok(value),
        _ => Err(ConfigError::UnsupportedArchitecture(value).into()),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GgufTokenizerMetadata {
    pub(crate) tokens: Option<Vec<String>>,
    pub(crate) bos_token_id: Option<TokenId>,
    pub(crate) eos_token_id: Option<TokenId>,
}

pub(crate) fn tokenizer_metadata(file: &GgufFile) -> Result<GgufTokenizerMetadata, LoadError> {
    let tokens = file
        .get_metadata("tokenizer.ggml.tokens")?
        .map(|value| match value {
            GgufMetadataValue::Strings(tokens) => Ok(tokens),
            other => Err(mistyped("tokenizer.ggml.tokens", "string array", &other)),
        })
        .transpose()?;
    let bos_token_id = Metadata(file).token_id("tokenizer.ggml.bos_token_id")?;
    let eos_token_id = Metadata(file).token_id("tokenizer.ggml.eos_token_id")?;
    Ok(GgufTokenizerMetadata {
        tokens,
        bos_token_id,
        eos_token_id,
    })
}

pub(crate) fn raw_config(
    file: &GgufFile,
    weights: &WeightManifest,
) -> Result<RawConfig, LoadError> {
    let a = architecture(file)?;
    let m = Metadata(file);
    let key = |suffix: &str| format!("{a}.{suffix}");
    let layers = m.required_size(&key("block_count"))?;
    let hidden = m.required_size(&key("embedding_length"))?;
    let intermediate = m.required_size(&key("feed_forward_length"))?;
    let heads = m.required_size(&key("attention.head_count"))?;
    let kv_heads = m.required_size(&key("attention.head_count_kv"))?;
    if !heads.is_multiple_of(kv_heads) {
        return Err(invalid(
            &key("attention.head_count_kv"),
            "must divide query heads",
        ));
    }
    let epsilon = m
        .real(&key("attention.layer_norm_rms_epsilon"), true)?
        .ok_or_else(|| missing(&key("attention.layer_norm_rms_epsilon")))?;
    let theta = m
        .real(&key("rope.freq_base"), true)?
        .ok_or_else(|| missing(&key("rope.freq_base")))?;
    let key_length = m.size(&key("attention.key_length"), false)?;
    let value_length = m.size(&key("attention.value_length"), false)?;
    let rope_dimensions = m.size(&key("rope.dimension_count"), false)?;
    let context = m.size(&key("context_length"), false)?;
    let vocabulary = m.size(&key("vocab_size"), false)?;
    let scaling_type = m.string(&key("rope.scaling.type"))?;
    if !matches!(scaling_type.as_deref(), None | Some("none" | "linear")) {
        return Err(ConfigError::UnsupportedRope(scaling_type.unwrap()).into());
    }
    let linear = scaling_type.as_deref() == Some("linear");
    let factor = m.real(&key("rope.scaling.factor"), linear)?;
    if !linear && factor.is_some() {
        return Err(unsupported(
            &key("rope.scaling.factor"),
            "orphan scaling factor",
        ));
    }
    if factor.is_some_and(|n| !n.recip().is_finite()) {
        return Err(invalid(
            &key("rope.scaling.factor"),
            "inverse factor must be finite",
        ));
    }
    m.size(&key("rope.scaling.original_context_length"), false)?;
    m.boolean(&key("rope.scaling.finetuned"))?;
    let embedded = tokenizer_metadata(file)?;
    m.integer("general.file_type")?;
    m.integer("general.quantization_version")?;
    for (suffix, required) in [("attention.causal", true), ("use_parallel_residual", false)] {
        if let Some(value) = m.boolean(&key(suffix))? {
            if value != required {
                return Err(unsupported(&key(suffix), value));
            }
        }
    }
    if let Some(value) = m.integer(&key("attention.sliding_window"))? {
        if value != 0 {
            return Err(unsupported(&key("attention.sliding_window"), value));
        }
    }
    if let Some(value) = m.string(&key("tensor_data_layout"))? {
        return Err(unsupported(&key("tensor_data_layout"), value));
    }
    if let Some(value) = m.scalar(&key("attention.scale"), "F32 scalar", &[Dtype::Float32])? {
        return Err(unsupported(
            &key("attention.scale"),
            item::<f32>(&value, &key("attention.scale"))?,
        ));
    }

    // Complete triples are checked before shape derivation, including orphan outputs.
    weights.validate_gguf_keys(layers, a == "qwen3")?;
    let groups = weights.gguf_quantization_groups()?;
    let rows = |name: &str| -> Result<usize, LoadError> {
        let entry = weights
            .tensors
            .get(name)
            .ok_or_else(|| crate::WeightError::MissingKey(name.into()))?;
        if entry.shape.len() != 2 || entry.shape.contains(&0) {
            return Err(crate::WeightError::UnsupportedFormat(format!(
                "{} must be a nonempty matrix",
                crate::arch::gguf::external(name)
            ))
            .into());
        }
        Ok(entry.shape[0])
    };
    let vocab = rows("model.embed_tokens.weight")?;
    if vocabulary.is_some_and(|v| v != vocab) {
        return Err(invalid(&key("vocab_size"), "does not match embedding rows"));
    }
    if embedded
        .tokens
        .as_ref()
        .is_some_and(|tokens| tokens.len() > vocab)
    {
        return Err(invalid(
            "tokenizer.ggml.tokens",
            "token count exceeds embedding rows",
        ));
    }
    let mut head_dim = None;
    for layer in 0..layers {
        for (projection, count) in [("q", heads), ("k", kv_heads), ("v", kv_heads)] {
            let name = format!("model.layers.{layer}.self_attn.{projection}_proj.weight");
            let width = rows(&name)?;
            if !width.is_multiple_of(count) {
                return Err(invalid(
                    &key(if projection == "q" {
                        "attention.head_count"
                    } else {
                        "attention.head_count_kv"
                    }),
                    "must divide projection rows",
                ));
            }
            let d = width / count;
            if let Some(expected) = head_dim {
                if d != expected {
                    return Err(crate::WeightError::ShapeMismatch {
                        key: name.clone(),
                        expected: vec![count * expected, weights.tensors[&name].shape[1]],
                        actual: weights.tensors
                            [&format!("model.layers.{layer}.self_attn.{projection}_proj.weight")]
                            .shape
                            .clone(),
                    }
                    .into());
                }
            } else {
                head_dim = Some(d);
            }
        }
    }
    let d = head_dim.ok_or_else(|| invalid(&key("block_count"), "no attention layers"))?;
    if !d.is_multiple_of(2) {
        return Err(
            ConfigError::UnsupportedRope("GGUF requires an even full head width".into()).into(),
        );
    }
    for (suffix, value) in [
        ("attention.key_length", key_length),
        ("attention.value_length", value_length),
    ] {
        if value.is_some_and(|n| n != d) {
            return Err(invalid(
                &key(suffix),
                "does not match projection head width",
            ));
        }
    }
    if rope_dimensions.is_some_and(|n| n != d) {
        return Err(ConfigError::UnsupportedRope("GGUF requires full-head RoPE".into()).into());
    }
    let tied = !weights
        .tensors
        .keys()
        .any(|name| name.starts_with("lm_head."));
    let mut value = json!({
        "model_type": a, "num_hidden_layers": layers, "hidden_size": hidden,
        "intermediate_size": intermediate, "num_attention_heads": heads, "num_key_value_heads": kv_heads,
        "rms_norm_eps": epsilon, "rope_theta": theta, "head_dim": d, "vocab_size": vocab,
        "max_position_embeddings": context, "tie_word_embeddings": tied, "attention_bias": false,
        "mlp_bias": false, "rope_traditional": false,
        "layer_types": vec!["full_attention"; layers],
    });
    if let Some(factor) = factor {
        value["rope_scaling"] = json!({"type": "linear", "factor": factor});
    }
    if let Some(default) = groups.iter().find_map(|g| match &g.quantization {
        LayerQuantization::Affine(q) => Some(q),
        _ => None,
    }) {
        let mut quantization = json!({"group_size": 32, "bits": default.bits, "mode": "affine"});
        for group in groups {
            let path = group.path.as_str();
            let path = if a == "qwen3" {
                path.strip_prefix("model.").unwrap_or(path)
            } else {
                path
            };
            quantization[path] = match group.quantization {
                LayerQuantization::Unquantized => json!(false),
                LayerQuantization::Affine(q) => {
                    json!({"group_size":32, "bits":q.bits, "mode":"affine"})
                }
            };
        }
        value["quantization"] = quantization;
    }
    Ok(RawConfig::from_bytes(
        &serde_json::to_vec(&value).map_err(ConfigError::from)?,
    )?)
}

#[cfg(test)]
mod tests;
