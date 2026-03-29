use std::path::Path;

use mlx_rs::{error::Exception, module::ModuleParametersExt};
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::{error::Error, models::llama, utils::rope::FloatOrString};

pub type Attention = llama::Attention;
pub type Mlp = llama::Mlp;
pub type TransformerBlock = llama::TransformerBlock;
pub type Qwen2Model = llama::LlamaModel;
pub type Model = llama::Model;
pub type ModelInput<'a, C> = llama::ModelInput<'a, C>;
pub type Generate<'a, C> = llama::Generate<'a, C>;
pub type GenerateState<'a> = llama::GenerateState<'a>;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub num_key_value_heads: i32,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_traditional: bool,
    #[serde(default)]
    pub rope_scaling: Option<std::collections::HashMap<String, FloatOrString>>,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
}

fn default_true() -> bool {
    true
}

fn default_max_position_embeddings() -> i32 {
    32768
}

fn default_rope_theta() -> f32 {
    1_000_000.0
}

impl From<ModelArgs> for llama::ModelArgs {
    fn from(value: ModelArgs) -> Self {
        let head_dim = value.hidden_size / value.num_attention_heads;
        Self {
            model_type: value.model_type,
            hidden_size: value.hidden_size,
            num_hidden_layers: value.num_hidden_layers,
            intermediate_size: value.intermediate_size,
            num_attention_heads: value.num_attention_heads,
            rms_norm_eps: value.rms_norm_eps,
            vocab_size: value.vocab_size,
            num_key_value_heads: value.num_key_value_heads,
            max_position_embeddings: value.max_position_embeddings,
            rope_theta: value.rope_theta,
            head_dim,
            tie_word_embeddings: value.tie_word_embeddings,
            attention_bias: true,
            mlp_bias: false,
            rope_scaling: value.rope_scaling,
        }
    }
}

pub fn load_qwen2_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    llama::load_llama_tokenizer(model_dir)
}

pub fn get_qwen2_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let model_args_filename = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(model_args_filename)?;
    let model_args: ModelArgs = serde_json::from_reader(file)?;

    Ok(model_args)
}

pub fn load_qwen2_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let model_args = get_qwen2_model_args(model_dir)?;
    let mut model = Model::new(model_args.into())?;

    let weights_index = model_dir.join("model.safetensors.index.json");
    if weights_index.exists() {
        let json = std::fs::read_to_string(weights_index)?;
        let weight_map: llama::WeightMap = serde_json::from_str(&json)?;

        let weight_files: std::collections::HashSet<&String> =
            weight_map.weight_map.values().collect();
        for weight_file in weight_files {
            let weights_filename = model_dir.join(weight_file);
            model.load_safetensors(weights_filename)?;
        }
    } else {
        let weights_filename = model_dir.join("model.safetensors");
        model.load_safetensors(weights_filename)?;
    }

    Ok(model)
}

pub fn sample(logits: &mlx_rs::Array, temp: f32) -> Result<mlx_rs::Array, Exception> {
    llama::sample(logits, temp)
}

#[cfg(test)]
mod tests {
    use mlx_rs::{
        ops::indexing::{IndexOp, NewAxis},
        transforms::eval,
        Array,
    };

    use crate::{
        cache::ConcatKeyValueCache,
        models::qwen2::{load_qwen2_model, load_qwen2_tokenizer},
    };

    const CACHED_TEST_MODEL_DIR: &str =
        "/Users/jdumay/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e";

    #[test]
    #[ignore = "requires local model files"]
    fn test_load_qwen2_model() {
        let model = super::load_qwen2_model(CACHED_TEST_MODEL_DIR).unwrap();
        assert_eq!(model.model_type(), "qwen2");
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_load_tokenizer() {
        let tokenizer = load_qwen2_tokenizer(CACHED_TEST_MODEL_DIR).unwrap();
        let _encoding = tokenizer.encode("Hello, world!", true).unwrap();
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_load_and_run_qwen2_with_concat_cache() {
        let tokenizer = load_qwen2_tokenizer(CACHED_TEST_MODEL_DIR).unwrap();
        let mut model = load_qwen2_model(CACHED_TEST_MODEL_DIR).unwrap();

        let encoding = tokenizer.encode("hello", true).unwrap();
        let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);
        let mut cache = Vec::new();

        let mut tokens = Vec::new();
        let generate = super::Generate::<ConcatKeyValueCache>::new(
            &mut model,
            &mut cache,
            0.0,
            &prompt_tokens,
        );
        for (token, ntoks) in generate.zip(0..10) {
            let token = token.unwrap();
            tokens.push(token.clone());

            if ntoks == 0 {
                eval(&tokens).unwrap();
            }
        }

        eval(&tokens).unwrap();
        let slice: Vec<u32> = tokens.drain(..).map(|t| t.item::<u32>()).collect();
        let s = tokenizer.decode(&slice, true).unwrap();
        assert!(!s.is_empty());
    }
}
