use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use mlx_rs::{
    argmax_axis, array,
    builder::Builder,
    categorical,
    error::Exception,
    fast::{rope_dynamic, ScaledDotProductAttentionMask},
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParametersExt},
    nn,
    ops::indexing::{IndexOp, NewAxis},
    quantization::MaybeQuantized,
    Array,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::{
    cache::KeyValueCache,
    error::Error,
    utils::{
        create_attention_mask,
        rope::{initialize_rope, FloatOrString, RopeVariant},
        AttentionMask,
    },
};

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
    pub rope_scaling: Option<HashMap<String, FloatOrString>>,
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

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Attention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub scale: f32,

    #[quantizable]
    #[param]
    pub q_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub k_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub v_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub o_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub rope: RopeVariant,
}

impl Attention {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let dim = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;

        let head_dim = args.hidden_size / n_heads;
        let scale = (head_dim as f32).sqrt().recip();

        let q_proj = nn::LinearBuilder::new(dim, n_heads * head_dim)
            .bias(true)
            .build()?;
        let k_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(true)
            .build()?;
        let v_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(true)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, dim)
            .bias(false)
            .build()?;

        let rope = initialize_rope(
            head_dim,
            args.rope_theta,
            args.rope_traditional,
            &args.rope_scaling,
            args.max_position_embeddings,
        )?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            scale,
            q_proj: MaybeQuantized::Original(q_proj),
            k_proj: MaybeQuantized::Original(k_proj),
            v_proj: MaybeQuantized::Original(v_proj),
            o_proj: MaybeQuantized::Original(o_proj),
            rope,
        })
    }
}

fn apply_cached_rope(rope: &mut RopeVariant, x: &Array, offset: i32) -> Result<Array, Exception> {
    let seq_len = x.shape()[x.shape().len() - 2];

    if seq_len != 1 {
        let rope_input = nn::RopeInputBuilder::new(x).offset(offset).build()?;
        return rope.forward(rope_input);
    }

    let position_array = Array::from_int(offset);

    match rope {
        RopeVariant::Default(rope) => rope_dynamic(
            x,
            rope.dimensions,
            rope.traditional,
            Some(rope.base),
            rope.scale,
            &position_array,
            None::<&Array>,
        ),
        RopeVariant::Llama3(rope) => rope_dynamic(
            x,
            rope.dimensions,
            rope.traditional,
            None::<f32>,
            rope.scale,
            &position_array,
            Some(&rope.freqs),
        ),
    }
}

pub struct AttentionInput<'a, C> {
    pub x: &'a Array,
    pub mask: Option<&'a AttentionMask>,
    pub cache: Option<&'a mut C>,
}

impl<C> Module<AttentionInput<'_, C>> for Attention
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, mut cache } = input;

        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];

        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        let mut queries = queries
            .reshape(&[B, L, self.n_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut keys = keys
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut values = values
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        if let Some(cache) = cache.as_mut() {
            queries = apply_cached_rope(&mut self.rope, &queries, cache.offset())?;
            keys = apply_cached_rope(&mut self.rope, &keys, cache.offset())?;

            (keys, values) = cache.update_and_fetch(keys, values)?;
        } else {
            queries = self.rope.forward(nn::RopeInput::new(&queries))?;
            keys = self.rope.forward(nn::RopeInput::new(&keys))?;
        }

        let output = match mask {
            Some(AttentionMask::Array(mask)) => {
                crate::utils::scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    cache,
                    self.scale,
                    Some(mask),
                )?
            }
            Some(AttentionMask::Causal) => mlx_rs::fast::scaled_dot_product_attention(
                queries,
                keys,
                values,
                self.scale,
                Some(ScaledDotProductAttentionMask::Causal),
                None,
            )?,
            None => crate::utils::scaled_dot_product_attention(
                queries, keys, values, cache, self.scale, None,
            )?,
        }
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[B, L, -1])?;

        self.o_proj.forward(&output)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
        <RopeVariant as Module<nn::RopeInput>>::training_mode(&mut self.rope, mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Mlp {
    #[quantizable]
    #[param]
    pub gate_proj: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    pub down_proj: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    pub up_proj: MaybeQuantized<nn::Linear>,
}

impl Mlp {
    pub fn new(dim: i32, hidden_dim: i32) -> Result<Self, Exception> {
        let gate_proj = nn::LinearBuilder::new(dim, hidden_dim)
            .bias(false)
            .build()?;
        let down_proj = nn::LinearBuilder::new(hidden_dim, dim)
            .bias(false)
            .build()?;
        let up_proj = nn::LinearBuilder::new(dim, hidden_dim)
            .bias(false)
            .build()?;

        Ok(Self {
            gate_proj: MaybeQuantized::Original(gate_proj),
            down_proj: MaybeQuantized::Original(down_proj),
            up_proj: MaybeQuantized::Original(up_proj),
        })
    }
}

impl Module<&Array> for Mlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array) -> Result<Self::Output, Self::Error> {
        let down_proj_input =
            nn::silu(self.gate_proj.forward(input)?)?.multiply(self.up_proj.forward(input)?)?;
        self.down_proj.forward(&down_proj_input)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct TransformerBlock {
    pub num_attention_heads: i32,
    pub hidden_size: i32,

    #[quantizable]
    #[param]
    pub self_attn: Attention,

    #[quantizable]
    #[param]
    pub mlp: Mlp,

    #[param]
    pub input_layernorm: nn::RmsNorm,

    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
}

impl TransformerBlock {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let self_attn = Attention::new(args)?;
        let mlp = Mlp::new(args.hidden_size, args.intermediate_size)?;
        let input_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;

        Ok(Self {
            num_attention_heads: args.num_attention_heads,
            hidden_size: args.hidden_size,
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

impl<C> Module<AttentionInput<'_, C>> for TransformerBlock
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, cache } = input;

        let self_attn_input = AttentionInput {
            x: &self.input_layernorm.forward(x)?,
            mask,
            cache,
        };
        let r = self.self_attn.forward(self_attn_input)?;
        let h = x.add(r)?;

        let r = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&h)?)?;
        h.add(r)
    }

    fn training_mode(&mut self, mode: bool) {
        <Attention as Module<AttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Qwen2Model {
    pub vocab_size: i32,
    pub num_hidden_layers: i32,

    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,

    #[quantizable]
    #[param]
    pub layers: Vec<TransformerBlock>,

    #[param]
    pub norm: nn::RmsNorm,
}

impl Qwen2Model {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        assert!(args.vocab_size.is_positive());

        let embed_tokens = nn::Embedding::new(args.vocab_size, args.hidden_size)?;
        let layers = (0..args.num_hidden_layers)
            .map(|_| TransformerBlock::new(args))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;

        Ok(Self {
            vocab_size: args.vocab_size,
            num_hidden_layers: args.num_hidden_layers,
            embed_tokens: MaybeQuantized::Original(embed_tokens),
            layers,
            norm,
        })
    }
}

pub struct ModelInput<'a, C> {
    pub inputs: &'a Array,
    pub mask: Option<&'a AttentionMask>,
    pub cache: &'a mut Vec<Option<C>>,
}

impl<C> Module<ModelInput<'_, C>> for Qwen2Model
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let ModelInput {
            inputs,
            mask,
            cache,
        } = input;

        let mut h = self.embed_tokens.forward(inputs)?;

        let mask = match mask {
            Some(mask) => Some(mask.clone()),
            None => create_attention_mask(&h, cache, None)?,
        };

        if cache.is_empty() {
            *cache = (0..self.layers.len()).map(|_| Some(C::default())).collect();
        }

        for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
            let layer_input = AttentionInput {
                x: &h,
                mask: mask.as_ref(),
                cache: c.as_mut(),
            };
            h = layer.forward(layer_input)?;
        }

        self.norm.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for layer in &mut self.layers {
            <TransformerBlock as Module<AttentionInput<'_, C>>>::training_mode(layer, mode);
        }
        self.norm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Model {
    pub args: ModelArgs,

    #[quantizable]
    #[param]
    pub model: Qwen2Model,

    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    pub fn new(args: ModelArgs) -> Result<Self, Exception> {
        let model = Qwen2Model::new(&args)?;
        let lm_head = if !args.tie_word_embeddings {
            Some(MaybeQuantized::Original(
                nn::LinearBuilder::new(args.hidden_size, args.vocab_size)
                    .bias(false)
                    .build()?,
            ))
        } else {
            None
        };

        Ok(Self {
            args,
            model,
            lm_head,
        })
    }

    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }
}

impl<C> Module<ModelInput<'_, C>> for Model
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let out = self.model.forward(input)?;

        match self.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(&out),
            None => match &mut self.model.embed_tokens {
                MaybeQuantized::Original(embed_tokens) => embed_tokens.as_linear(&out),
                MaybeQuantized::Quantized(q_embed_tokens) => q_embed_tokens.as_linear(&out),
            },
        }
    }

    fn training_mode(&mut self, mode: bool) {
        <Qwen2Model as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.training_mode(mode);
        }
    }
}

pub fn load_qwen2_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

pub fn get_qwen2_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let model_args_filename = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(model_args_filename)?;
    let model_args: ModelArgs = serde_json::from_reader(file)?;

    Ok(model_args)
}

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub metadata: HashMap<String, Value>,
    pub weight_map: HashMap<String, String>,
}

pub fn load_qwen2_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let model_args = get_qwen2_model_args(model_dir)?;
    let mut model = Model::new(model_args)?;

    let weights_index = model_dir.join("model.safetensors.index.json");
    if weights_index.exists() {
        let json = std::fs::read_to_string(weights_index)?;
        let weight_map: WeightMap = serde_json::from_str(&json)?;

        let weight_files: HashSet<&String> = weight_map.weight_map.values().collect();
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

pub fn sample(logits: &Array, temp: f32) -> Result<Array, Exception> {
    match temp {
        0.0 => argmax_axis!(logits, -1),
        _ => {
            let logits = logits.multiply(array!(1.0 / temp))?;
            categorical!(logits)
        }
    }
}

pub struct Generate<'a, C> {
    model: &'a mut Model,
    cache: &'a mut Vec<Option<C>>,
    temp: f32,
    state: GenerateState<'a>,
}

impl<'a, C> Generate<'a, C>
where
    C: KeyValueCache + Default,
{
    pub fn new(
        model: &'a mut Model,
        cache: &'a mut Vec<Option<C>>,
        temp: f32,
        prompt_token: &'a Array,
    ) -> Self {
        Self {
            model,
            cache,
            temp,
            state: GenerateState::Prefill { prompt_token },
        }
    }
}

pub enum GenerateState<'a> {
    Prefill { prompt_token: &'a Array },
    Decode { y: Array },
}

macro_rules! tri {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => return Some(Err(e.into())),
        }
    };
}

impl<'a, C> Iterator for Generate<'a, C>
where
    C: KeyValueCache + Default,
{
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.state {
            GenerateState::Prefill { prompt_token } => {
                let input = ModelInput {
                    inputs: prompt_token,
                    mask: None,
                    cache: self.cache,
                };
                let logits = tri!(self.model.forward(input));
                let y = tri!(sample(&logits.index((.., -1, ..)), self.temp));
                self.state = GenerateState::Decode { y: y.clone() };

                Some(Ok(y))
            }
            GenerateState::Decode { y } => {
                let inputs = y.index((.., NewAxis));
                let input = ModelInput {
                    inputs: &inputs,
                    mask: None,
                    cache: self.cache,
                };
                let logits = tri!(self.model.forward(input));
                let y = tri!(sample(&logits.index((.., -1, ..)), self.temp));

                self.state = GenerateState::Decode { y: y.clone() };

                Some(Ok(y))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use mlx_rs::{
        builder::Builder,
        fast::rope_dynamic,
        nn::RopeInputBuilder,
        module::Module,
        ops::indexing::{IndexOp, NewAxis},
        transforms::eval,
        Array,
    };

    use crate::{
        cache::{ConcatKeyValueCache, KVCache, KeyValueCache},
        models::qwen2::{load_qwen2_model, load_qwen2_tokenizer},
        utils::{create_causal_mask, AttentionMask},
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

    #[test]
    #[ignore = "requires local model files"]
    fn test_load_and_run_qwen2_with_kv_cache() {
        let tokenizer = load_qwen2_tokenizer(CACHED_TEST_MODEL_DIR).unwrap();
        let mut model = load_qwen2_model(CACHED_TEST_MODEL_DIR).unwrap();

        let encoding = tokenizer.encode("hello", true).unwrap();
        let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);
        let mut cache = Vec::new();

        let mut tokens = Vec::new();
        let generate = super::Generate::<KVCache>::new(&mut model, &mut cache, 0.0, &prompt_tokens);
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

    #[test]
    #[ignore = "requires local model files"]
    fn test_qwen2_concat_cache_benchmark_prompt_trace() {
        let tokenizer = load_qwen2_tokenizer(CACHED_TEST_MODEL_DIR).unwrap();
        let mut model = load_qwen2_model(CACHED_TEST_MODEL_DIR).unwrap();

        let prompt = "<|im_start|>system\nYou are a concise assistant.<|im_end|>\n<|im_start|>user\nContext: The native MLX path supports streaming, but some families behave inconsistently.\nTask: Give three short bullets on what is working, what is not, and the next fix.<|im_end|>\n<|im_start|>assistant\n";
        let encoding = tokenizer.encode(prompt, false).unwrap();
        let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);
        let mut cache = Vec::new();

        let mut tokens = Vec::new();
        let generate = super::Generate::<ConcatKeyValueCache>::new(
            &mut model,
            &mut cache,
            0.0,
            &prompt_tokens,
        );
        for token in generate.take(32) {
            tokens.push(token.unwrap());
        }

        eval(&tokens).unwrap();
        let ids: Vec<u32> = tokens.iter().map(|t| t.item::<u32>()).collect();
        let text = tokenizer.decode(&ids, true).unwrap();
        eprintln!("qwen2-concat-trace ids={ids:?} text={text:?}");
        assert!(!text.is_empty());
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_qwen2_kv_cache_benchmark_prompt_trace() {
        let tokenizer = load_qwen2_tokenizer(CACHED_TEST_MODEL_DIR).unwrap();
        let mut model = load_qwen2_model(CACHED_TEST_MODEL_DIR).unwrap();

        let prompt = "<|im_start|>system\nYou are a concise assistant.<|im_end|>\n<|im_start|>user\nContext: The native MLX path supports streaming, but some families behave inconsistently.\nTask: Give three short bullets on what is working, what is not, and the next fix.<|im_end|>\n<|im_start|>assistant\n";
        let encoding = tokenizer.encode(prompt, false).unwrap();
        let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);
        let mut cache = Vec::new();

        let mut tokens = Vec::new();
        let generate =
            super::Generate::<KVCache>::new(&mut model, &mut cache, 0.0, &prompt_tokens);
        for token in generate.take(32) {
            tokens.push(token.unwrap());
        }

        eval(&tokens).unwrap();
        let ids: Vec<u32> = tokens.iter().map(|t| t.item::<u32>()).collect();
        let text = tokenizer.decode(&ids, true).unwrap();
        eprintln!("qwen2-kv-trace ids={ids:?} text={text:?}");
        assert!(!text.is_empty());
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_qwen2_concat_cache_step6_topk() {
        let tokenizer = load_qwen2_tokenizer(CACHED_TEST_MODEL_DIR).unwrap();
        let mut model = load_qwen2_model(CACHED_TEST_MODEL_DIR).unwrap();

        let prompt = "<|im_start|>system\nYou are a concise assistant.<|im_end|>\n<|im_start|>user\nContext: The native MLX path supports streaming, but some families behave inconsistently.\nTask: Give three short bullets on what is working, what is not, and the next fix.<|im_end|>\n<|im_start|>assistant\n";
        let encoding = tokenizer.encode(prompt, false).unwrap();
        let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);
        let mut cache = Vec::<Option<ConcatKeyValueCache>>::new();

        let input = super::ModelInput {
            inputs: &prompt_tokens,
            mask: None,
            cache: &mut cache,
        };
        let logits = model.forward(input).unwrap();
        let mut step_logits = logits.index((.., -1, ..));
        eval([&step_logits]).unwrap();

        let mut ids = Vec::new();
        for step in 0..8 {
            let step_logits_f32 = step_logits.as_type::<f32>().unwrap();
            let flat = step_logits_f32.as_slice::<f32>();
            let mut ranked: Vec<(usize, f32)> = flat.iter().copied().enumerate().collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
            if step == 5 {
                let top: Vec<(usize, f32)> = ranked.into_iter().take(10).collect();
                eprintln!("qwen2-step6-topk {top:?}");
            }

            let y = super::sample(&step_logits, 0.0).unwrap();
            let token_id = y.item::<u32>();
            ids.push(token_id);
            let inputs = y.index((.., NewAxis));
            let input = super::ModelInput {
                inputs: &inputs,
                mask: None,
                cache: &mut cache,
            };
            let logits = model.forward(input).unwrap();
            step_logits = logits.index((.., -1, ..));
            eval([&step_logits]).unwrap();
        }

        let text = tokenizer.decode(&ids, true).unwrap();
        eprintln!("qwen2-step6-trace ids={ids:?} text={text:?}");
        assert!(!ids.is_empty());
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_qwen2_full_prefill_vs_incremental_trace() {
        let tokenizer = load_qwen2_tokenizer(CACHED_TEST_MODEL_DIR).unwrap();
        let mut model = load_qwen2_model(CACHED_TEST_MODEL_DIR).unwrap();

        let prompt = "<|im_start|>system\nYou are a concise assistant.<|im_end|>\n<|im_start|>user\nContext: The native MLX path supports streaming, but some families behave inconsistently.\nTask: Give three short bullets on what is working, what is not, and the next fix.<|im_end|>\n<|im_start|>assistant\n";
        let encoding = tokenizer.encode(prompt, false).unwrap();
        let prompt_ids = encoding.get_ids().to_vec();

        let mut incremental_ids = Vec::new();
        let mut cache = Vec::<Option<ConcatKeyValueCache>>::new();
        let prompt_tokens = Array::from(prompt_ids.as_slice()).index(NewAxis);
        let input = super::ModelInput {
            inputs: &prompt_tokens,
            mask: None,
            cache: &mut cache,
        };
        let logits = model.forward(input).unwrap();
        let mut step_logits = logits.index((.., -1, ..));
        eval([&step_logits]).unwrap();
        for _ in 0..12 {
            let y = super::sample(&step_logits, 0.0).unwrap();
            let token_id = y.item::<u32>();
            incremental_ids.push(token_id);
            let inputs = y.index((.., NewAxis));
            let input = super::ModelInput {
                inputs: &inputs,
                mask: None,
                cache: &mut cache,
            };
            let logits = model.forward(input).unwrap();
            step_logits = logits.index((.., -1, ..));
            eval([&step_logits]).unwrap();
        }

        let mut full_prefill_ids = Vec::new();
        for _ in 0..12 {
            let all_ids: Vec<u32> = prompt_ids
                .iter()
                .copied()
                .chain(full_prefill_ids.iter().copied())
                .collect();
            let tokens = Array::from(all_ids.as_slice()).index(NewAxis);
            let mut fresh_cache = Vec::<Option<ConcatKeyValueCache>>::new();
            let input = super::ModelInput {
                inputs: &tokens,
                mask: None,
                cache: &mut fresh_cache,
            };
            let logits = model.forward(input).unwrap();
            let step_logits = logits.index((.., -1, ..));
            eval([&step_logits]).unwrap();
            let y = super::sample(&step_logits, 0.0).unwrap();
            full_prefill_ids.push(y.item::<u32>());
        }

        let incremental_text = tokenizer.decode(&incremental_ids, true).unwrap();
        let full_prefill_text = tokenizer.decode(&full_prefill_ids, true).unwrap();
        eprintln!("qwen2-incremental ids={incremental_ids:?} text={incremental_text:?}");
        eprintln!("qwen2-full-prefill ids={full_prefill_ids:?} text={full_prefill_text:?}");
        assert!(!incremental_ids.is_empty());
        assert!(!full_prefill_ids.is_empty());
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_qwen2_incremental_with_forced_causal_mask_trace() {
        let tokenizer = load_qwen2_tokenizer(CACHED_TEST_MODEL_DIR).unwrap();
        let mut model = load_qwen2_model(CACHED_TEST_MODEL_DIR).unwrap();

        let prompt = "<|im_start|>system\nYou are a concise assistant.<|im_end|>\n<|im_start|>user\nContext: The native MLX path supports streaming, but some families behave inconsistently.\nTask: Give three short bullets on what is working, what is not, and the next fix.<|im_end|>\n<|im_start|>assistant\n";
        let encoding = tokenizer.encode(prompt, false).unwrap();
        let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);
        let mut cache = Vec::<Option<ConcatKeyValueCache>>::new();

        let input = super::ModelInput {
            inputs: &prompt_tokens,
            mask: None,
            cache: &mut cache,
        };
        let logits = model.forward(input).unwrap();
        let mut step_logits = logits.index((.., -1, ..));
        eval([&step_logits]).unwrap();

        let mut ids = Vec::new();
        let causal = AttentionMask::Causal;
        for _ in 0..12 {
            let y = super::sample(&step_logits, 0.0).unwrap();
            let token_id = y.item::<u32>();
            ids.push(token_id);
            let inputs = y.index((.., NewAxis));
            let input = super::ModelInput {
                inputs: &inputs,
                mask: Some(&causal),
                cache: &mut cache,
            };
            let logits = model.forward(input).unwrap();
            step_logits = logits.index((.., -1, ..));
            eval([&step_logits]).unwrap();
        }

        let text = tokenizer.decode(&ids, true).unwrap();
        eprintln!("qwen2-forced-causal ids={ids:?} text={text:?}");
        assert!(!ids.is_empty());
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_qwen2_incremental_vs_full_prefill_next_token_logits() {
        let tokenizer = load_qwen2_tokenizer(CACHED_TEST_MODEL_DIR).unwrap();
        let mut model = load_qwen2_model(CACHED_TEST_MODEL_DIR).unwrap();

        let prompt = "<|im_start|>system\nYou are a concise assistant.<|im_end|>\n<|im_start|>user\nContext: The native MLX path supports streaming, but some families behave inconsistently.\nTask: Give three short bullets on what is working, what is not, and the next fix.<|im_end|>\n<|im_start|>assistant\n";
        let encoding = tokenizer.encode(prompt, false).unwrap();
        let prompt_ids = encoding.get_ids().to_vec();
        let agreed_prefix = vec![12u32, 576, 9867, 19614, 55];

        let prompt_tokens = Array::from(prompt_ids.as_slice()).index(NewAxis);
        let mut cache = Vec::<Option<ConcatKeyValueCache>>::new();
        let input = super::ModelInput {
            inputs: &prompt_tokens,
            mask: None,
            cache: &mut cache,
        };
        let logits = model.forward(input).unwrap();
        let mut incremental_logits = logits.index((.., -1, ..));
        eval([&incremental_logits]).unwrap();
        for token_id in &agreed_prefix {
            let y = Array::from_slice(&[*token_id], &[1]);
            let inputs = y.index((.., NewAxis));
            let input = super::ModelInput {
                inputs: &inputs,
                mask: None,
                cache: &mut cache,
            };
            let logits = model.forward(input).unwrap();
            incremental_logits = logits.index((.., -1, ..));
            eval([&incremental_logits]).unwrap();
        }

        let all_ids: Vec<u32> = prompt_ids
            .iter()
            .copied()
            .chain(agreed_prefix.iter().copied())
            .collect();
        let all_tokens = Array::from(all_ids.as_slice()).index(NewAxis);
        let mut fresh_cache = Vec::<Option<ConcatKeyValueCache>>::new();
        let input = super::ModelInput {
            inputs: &all_tokens,
            mask: None,
            cache: &mut fresh_cache,
        };
        let logits = model.forward(input).unwrap();
        let full_prefill_logits = logits.index((.., -1, ..));
        eval([&full_prefill_logits]).unwrap();

        let inc_f32 = incremental_logits.as_type::<f32>().unwrap();
        let full_f32 = full_prefill_logits.as_type::<f32>().unwrap();
        let inc = inc_f32.as_slice::<f32>();
        let full = full_f32.as_slice::<f32>();

        let mut inc_ranked: Vec<(usize, f32)> = inc.iter().copied().enumerate().collect();
        let mut full_ranked: Vec<(usize, f32)> = full.iter().copied().enumerate().collect();
        inc_ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        full_ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

        let inc_top: Vec<(usize, f32)> = inc_ranked.into_iter().take(10).collect();
        let full_top: Vec<(usize, f32)> = full_ranked.into_iter().take(10).collect();
        eprintln!("qwen2-incremental-top {inc_top:?}");
        eprintln!("qwen2-full-prefill-top {full_top:?}");

        let inc_best = Array::from_slice(&[inc_top[0].0 as u32], &[1]);
        let full_best = Array::from_slice(&[full_top[0].0 as u32], &[1]);
        let inc_best_text = tokenizer.decode(&[inc_top[0].0 as u32], true).unwrap();
        let full_best_text = tokenizer.decode(&[full_top[0].0 as u32], true).unwrap();
        eprintln!("qwen2-incremental-best {:?} {:?}", inc_best.item::<u32>(), inc_best_text);
        eprintln!("qwen2-full-prefill-best {:?} {:?}", full_best.item::<u32>(), full_best_text);

        assert_eq!(inc_top[0].0, full_top[0].0);
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_qwen2_chunk_prefill_then_single_step_vs_incremental_next_token_logits() {
        let tokenizer = load_qwen2_tokenizer(CACHED_TEST_MODEL_DIR).unwrap();
        let mut model = load_qwen2_model(CACHED_TEST_MODEL_DIR).unwrap();

        let prompt = "<|im_start|>system\nYou are a concise assistant.<|im_end|>\n<|im_start|>user\nContext: The native MLX path supports streaming, but some families behave inconsistently.\nTask: Give three short bullets on what is working, what is not, and the next fix.<|im_end|>\n<|im_start|>assistant\n";
        let encoding = tokenizer.encode(prompt, false).unwrap();
        let prompt_ids = encoding.get_ids().to_vec();
        let agreed_prefix = vec![12u32, 576, 9867, 19614, 55];

        let prompt_tokens = Array::from(prompt_ids.as_slice()).index(NewAxis);
        let mut incremental_cache = Vec::<Option<ConcatKeyValueCache>>::new();
        let input = super::ModelInput {
            inputs: &prompt_tokens,
            mask: None,
            cache: &mut incremental_cache,
        };
        let logits = model.forward(input).unwrap();
        let mut incremental_logits = logits.index((.., -1, ..));
        eval([&incremental_logits]).unwrap();
        for token_id in &agreed_prefix {
            let y = Array::from_slice(&[*token_id], &[1]);
            let inputs = y.index((.., NewAxis));
            let input = super::ModelInput {
                inputs: &inputs,
                mask: None,
                cache: &mut incremental_cache,
            };
            let logits = model.forward(input).unwrap();
            incremental_logits = logits.index((.., -1, ..));
            eval([&incremental_logits]).unwrap();
        }

        let prefixed_ids: Vec<u32> = prompt_ids
            .iter()
            .copied()
            .chain(agreed_prefix[..agreed_prefix.len() - 1].iter().copied())
            .collect();
        let prefixed_tokens = Array::from(prefixed_ids.as_slice()).index(NewAxis);
        let mut chunked_cache = Vec::<Option<ConcatKeyValueCache>>::new();
        let input = super::ModelInput {
            inputs: &prefixed_tokens,
            mask: None,
            cache: &mut chunked_cache,
        };
        let _ = model.forward(input).unwrap();

        let last_prefix_token =
            Array::from_slice(&[agreed_prefix[agreed_prefix.len() - 1]], &[1]).index(NewAxis);
        let input = super::ModelInput {
            inputs: &last_prefix_token,
            mask: None,
            cache: &mut chunked_cache,
        };
        let logits = model.forward(input).unwrap();
        let chunked_logits = logits.index((.., -1, ..));
        eval([&chunked_logits]).unwrap();

        let all_ids: Vec<u32> = prompt_ids
            .iter()
            .copied()
            .chain(agreed_prefix.iter().copied())
            .collect();
        let all_tokens = Array::from(all_ids.as_slice()).index(NewAxis);
        let mut full_prefill_cache = Vec::<Option<ConcatKeyValueCache>>::new();
        let input = super::ModelInput {
            inputs: &all_tokens,
            mask: None,
            cache: &mut full_prefill_cache,
        };
        let logits = model.forward(input).unwrap();
        let full_prefill_logits = logits.index((.., -1, ..));
        eval([&full_prefill_logits]).unwrap();

        let collect_top = |logits: &Array| -> Vec<(usize, f32)> {
            let logits_f32 = logits.as_type::<f32>().unwrap();
            let flat = logits_f32.as_slice::<f32>();
            let mut ranked: Vec<(usize, f32)> = flat.iter().copied().enumerate().collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
            ranked.into_iter().take(10).collect()
        };

        let incremental_top = collect_top(&incremental_logits);
        let chunked_top = collect_top(&chunked_logits);
        let full_prefill_top = collect_top(&full_prefill_logits);

        eprintln!("qwen2-incremental-after-prefix-top {incremental_top:?}");
        eprintln!("qwen2-chunked-after-prefix-top {chunked_top:?}");
        eprintln!("qwen2-full-prefill-after-prefix-top {full_prefill_top:?}");

        let decode_best = |token_id: usize| -> String {
            tokenizer.decode(&[token_id as u32], true).unwrap()
        };

        eprintln!(
            "qwen2-best incremental={:?} chunked={:?} full_prefill={:?}",
            (incremental_top[0].0, decode_best(incremental_top[0].0)),
            (chunked_top[0].0, decode_best(chunked_top[0].0)),
            (full_prefill_top[0].0, decode_best(full_prefill_top[0].0)),
        );

        assert_eq!(chunked_top[0].0, full_prefill_top[0].0);
        assert_eq!(incremental_top[0].0, full_prefill_top[0].0);
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_qwen2_chunk_prefill_single_step_with_explicit_array_mask() {
        let tokenizer = load_qwen2_tokenizer(CACHED_TEST_MODEL_DIR).unwrap();
        let mut model = load_qwen2_model(CACHED_TEST_MODEL_DIR).unwrap();

        let prompt = "<|im_start|>system\nYou are a concise assistant.<|im_end|>\n<|im_start|>user\nContext: The native MLX path supports streaming, but some families behave inconsistently.\nTask: Give three short bullets on what is working, what is not, and the next fix.<|im_end|>\n<|im_start|>assistant\n";
        let encoding = tokenizer.encode(prompt, false).unwrap();
        let prompt_ids = encoding.get_ids().to_vec();
        let agreed_prefix = vec![12u32, 576, 9867, 19614, 55];

        let prefixed_ids: Vec<u32> = prompt_ids
            .iter()
            .copied()
            .chain(agreed_prefix[..agreed_prefix.len() - 1].iter().copied())
            .collect();
        let prefixed_tokens = Array::from(prefixed_ids.as_slice()).index(NewAxis);
        let mut chunked_cache = Vec::<Option<ConcatKeyValueCache>>::new();
        let input = super::ModelInput {
            inputs: &prefixed_tokens,
            mask: None,
            cache: &mut chunked_cache,
        };
        let _ = model.forward(input).unwrap();

        let offset = chunked_cache[0].as_ref().unwrap().offset();
        let explicit_mask = create_causal_mask(1, Some(offset), None, None).unwrap();
        let explicit_mask = AttentionMask::Array(explicit_mask);

        let last_prefix_token =
            Array::from_slice(&[agreed_prefix[agreed_prefix.len() - 1]], &[1]).index(NewAxis);
        let input = super::ModelInput {
            inputs: &last_prefix_token,
            mask: Some(&explicit_mask),
            cache: &mut chunked_cache,
        };
        let logits = model.forward(input).unwrap();
        let masked_logits = logits.index((.., -1, ..));
        eval([&masked_logits]).unwrap();

        let all_ids: Vec<u32> = prompt_ids
            .iter()
            .copied()
            .chain(agreed_prefix.iter().copied())
            .collect();
        let all_tokens = Array::from(all_ids.as_slice()).index(NewAxis);
        let mut full_prefill_cache = Vec::<Option<ConcatKeyValueCache>>::new();
        let input = super::ModelInput {
            inputs: &all_tokens,
            mask: None,
            cache: &mut full_prefill_cache,
        };
        let logits = model.forward(input).unwrap();
        let full_prefill_logits = logits.index((.., -1, ..));
        eval([&full_prefill_logits]).unwrap();

        let collect_top = |logits: &Array| -> Vec<(usize, f32)> {
            let logits_f32 = logits.as_type::<f32>().unwrap();
            let flat = logits_f32.as_slice::<f32>();
            let mut ranked: Vec<(usize, f32)> = flat.iter().copied().enumerate().collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
            ranked.into_iter().take(10).collect()
        };

        let masked_top = collect_top(&masked_logits);
        let full_prefill_top = collect_top(&full_prefill_logits);
        eprintln!("qwen2-explicit-array-mask-top {masked_top:?}");
        eprintln!("qwen2-full-prefill-after-prefix-top {full_prefill_top:?}");
        eprintln!(
            "qwen2-best masked={:?} full_prefill={:?}",
            (masked_top[0].0, tokenizer.decode(&[masked_top[0].0 as u32], true).unwrap()),
            (
                full_prefill_top[0].0,
                tokenizer
                    .decode(&[full_prefill_top[0].0 as u32], true)
                    .unwrap()
            ),
        );

        assert_eq!(masked_top[0].0, full_prefill_top[0].0);
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_qwen2_chunk_prefill_cache_state_vs_full_prefill_cache_state() {
        let mut model = load_qwen2_model(CACHED_TEST_MODEL_DIR).unwrap();

        let tokenizer = load_qwen2_tokenizer(CACHED_TEST_MODEL_DIR).unwrap();
        let prompt = "<|im_start|>system\nYou are a concise assistant.<|im_end|>\n<|im_start|>user\nContext: The native MLX path supports streaming, but some families behave inconsistently.\nTask: Give three short bullets on what is working, what is not, and the next fix.<|im_end|>\n<|im_start|>assistant\n";
        let encoding = tokenizer.encode(prompt, false).unwrap();
        let prompt_ids = encoding.get_ids().to_vec();
        let agreed_prefix = vec![12u32, 576, 9867, 19614, 55];

        let prefixed_ids: Vec<u32> = prompt_ids
            .iter()
            .copied()
            .chain(agreed_prefix[..agreed_prefix.len() - 1].iter().copied())
            .collect();
        let prefixed_tokens = Array::from(prefixed_ids.as_slice()).index(NewAxis);
        let mut chunked_cache = Vec::<Option<ConcatKeyValueCache>>::new();
        let input = super::ModelInput {
            inputs: &prefixed_tokens,
            mask: None,
            cache: &mut chunked_cache,
        };
        let _ = model.forward(input).unwrap();
        let last_prefix_token =
            Array::from_slice(&[agreed_prefix[agreed_prefix.len() - 1]], &[1]).index(NewAxis);
        let input = super::ModelInput {
            inputs: &last_prefix_token,
            mask: None,
            cache: &mut chunked_cache,
        };
        let _ = model.forward(input).unwrap();

        let all_ids: Vec<u32> = prompt_ids
            .iter()
            .copied()
            .chain(agreed_prefix.iter().copied())
            .collect();
        let all_tokens = Array::from(all_ids.as_slice()).index(NewAxis);
        let mut full_prefill_cache = Vec::<Option<ConcatKeyValueCache>>::new();
        let input = super::ModelInput {
            inputs: &all_tokens,
            mask: None,
            cache: &mut full_prefill_cache,
        };
        let _ = model.forward(input).unwrap();

        let chunked_first = chunked_cache[0].as_ref().unwrap();
        let full_first = full_prefill_cache[0].as_ref().unwrap();
        let (chunked_keys, chunked_values) = chunked_first.arrays();
        let (full_keys, full_values) = full_first.arrays();
        let chunked_keys = chunked_keys.unwrap();
        let chunked_values = chunked_values.unwrap();
        let full_keys = full_keys.unwrap();
        let full_values = full_values.unwrap();

        let chunked_last_key = chunked_keys.index((.., .., -1.., ..));
        let full_last_key = full_keys.index((.., .., -1.., ..));
        let chunked_last_value = chunked_values.index((.., .., -1.., ..));
        let full_last_value = full_values.index((.., .., -1.., ..));
        eval([
            &chunked_last_key,
            &full_last_key,
            &chunked_last_value,
            &full_last_value,
        ])
        .unwrap();

        let max_abs_diff = |a: &Array, b: &Array| -> f32 {
            let a = a.as_type::<f32>().unwrap();
            let b = b.as_type::<f32>().unwrap();
            a.subtract(&b)
                .unwrap()
                .abs()
                .unwrap()
                .max(false)
                .unwrap()
                .item::<f32>()
        };

        let key_diff = max_abs_diff(&chunked_last_key, &full_last_key);
        let value_diff = max_abs_diff(&chunked_last_value, &full_last_value);
        eprintln!("qwen2-first-layer-last-key-diff {key_diff}");
        eprintln!("qwen2-first-layer-last-value-diff {value_diff}");

        assert!(key_diff < 0.1, "first-layer key mismatch: {key_diff}");
        assert!(value_diff < 1e-3, "first-layer value mismatch: {value_diff}");
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_qwen2_first_layer_key_matches_full_prefill_only_for_correct_rope_offset() {
        let mut model = load_qwen2_model(CACHED_TEST_MODEL_DIR).unwrap();
        let tokenizer = load_qwen2_tokenizer(CACHED_TEST_MODEL_DIR).unwrap();

        let prompt = "<|im_start|>system\nYou are a concise assistant.<|im_end|>\n<|im_start|>user\nContext: The native MLX path supports streaming, but some families behave inconsistently.\nTask: Give three short bullets on what is working, what is not, and the next fix.<|im_end|>\n<|im_start|>assistant\n";
        let encoding = tokenizer.encode(prompt, false).unwrap();
        let prompt_ids = encoding.get_ids().to_vec();
        let agreed_prefix = vec![12u32, 576, 9867, 19614, 55];

        let all_ids: Vec<u32> = prompt_ids
            .iter()
            .copied()
            .chain(agreed_prefix.iter().copied())
            .collect();
        let all_tokens = Array::from(all_ids.as_slice()).index(NewAxis);
        let mut full_prefill_cache = Vec::<Option<ConcatKeyValueCache>>::new();
        let input = super::ModelInput {
            inputs: &all_tokens,
            mask: None,
            cache: &mut full_prefill_cache,
        };
        let _ = model.forward(input).unwrap();

        let full_first = full_prefill_cache[0].as_ref().unwrap();
        let (full_keys, _) = full_first.arrays();
        let full_last_key = full_keys.unwrap().index((.., .., -1.., ..));
        eval([&full_last_key]).unwrap();

        let offset = (prompt_ids.len() + agreed_prefix.len() - 1) as i32;
        let last_prefix_token =
            Array::from_slice(&[agreed_prefix[agreed_prefix.len() - 1]], &[1]).index(NewAxis);
        let embedded = model.model.embed_tokens.forward(&last_prefix_token).unwrap();
        let normalized = model.model.layers[0]
            .input_layernorm
            .forward(&embedded)
            .unwrap();
        let raw_keys = model.model.layers[0].self_attn.k_proj.forward(&normalized).unwrap();
        let shaped_keys = raw_keys
            .reshape(&[1, 1, model.model.layers[0].self_attn.n_kv_heads, -1])
            .unwrap()
            .transpose_axes(&[0, 2, 1, 3])
            .unwrap();

        let max_abs_diff = |a: &Array, b: &Array| -> f32 {
            let a = a.as_type::<f32>().unwrap();
            let b = b.as_type::<f32>().unwrap();
            a.subtract(&b)
                .unwrap()
                .abs()
                .unwrap()
                .max(false)
                .unwrap()
                .item::<f32>()
        };

        for candidate in [offset - 2, offset - 1, offset, offset + 1, offset + 2] {
            let rope_input = RopeInputBuilder::new(&shaped_keys)
                .offset(candidate)
                .build()
                .unwrap();
            let rotated = model.model.layers[0]
                .self_attn
                .rope
                .forward(rope_input)
                .unwrap();
            eval([&rotated]).unwrap();
            let diff = max_abs_diff(&rotated, &full_last_key);
            eprintln!("qwen2-first-layer-key-offset candidate={candidate} diff={diff}");
        }

        assert!(offset >= 0);
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_qwen2_first_layer_rope_sequence_vs_single_token_offset() {
        let mut model = load_qwen2_model(CACHED_TEST_MODEL_DIR).unwrap();
        let tokenizer = load_qwen2_tokenizer(CACHED_TEST_MODEL_DIR).unwrap();

        let prompt = "<|im_start|>system\nYou are a concise assistant.<|im_end|>\n<|im_start|>user\nContext: The native MLX path supports streaming, but some families behave inconsistently.\nTask: Give three short bullets on what is working, what is not, and the next fix.<|im_end|>\n<|im_start|>assistant\n";
        let encoding = tokenizer.encode(prompt, false).unwrap();
        let prompt_ids = encoding.get_ids().to_vec();
        let agreed_prefix = vec![12u32, 576, 9867, 19614, 55];
        let offset = prompt_ids.len() as i32;

        let prefix_tokens = Array::from(agreed_prefix.as_slice()).index(NewAxis);
        let embedded_prefix = model.model.embed_tokens.forward(&prefix_tokens).unwrap();
        let normalized_prefix = model.model.layers[0]
            .input_layernorm
            .forward(&embedded_prefix)
            .unwrap();
        let raw_prefix_keys = model.model.layers[0]
            .self_attn
            .k_proj
            .forward(&normalized_prefix)
            .unwrap()
            .reshape(&[1, agreed_prefix.len() as i32, model.model.layers[0].self_attn.n_kv_heads, -1])
            .unwrap()
            .transpose_axes(&[0, 2, 1, 3])
            .unwrap();

        let seq_rope_input = RopeInputBuilder::new(&raw_prefix_keys)
            .offset(offset)
            .build()
            .unwrap();
        let seq_rotated = model.model.layers[0]
            .self_attn
            .rope
            .forward(seq_rope_input)
            .unwrap();
        let seq_last = seq_rotated.index((.., .., -1.., ..));
        eval([&seq_last]).unwrap();

        let last_token = Array::from_slice(&[agreed_prefix[agreed_prefix.len() - 1]], &[1]).index(NewAxis);
        let embedded_last = model.model.embed_tokens.forward(&last_token).unwrap();
        let normalized_last = model.model.layers[0]
            .input_layernorm
            .forward(&embedded_last)
            .unwrap();
        let raw_last_key = model.model.layers[0]
            .self_attn
            .k_proj
            .forward(&normalized_last)
            .unwrap()
            .reshape(&[1, 1, model.model.layers[0].self_attn.n_kv_heads, -1])
            .unwrap()
            .transpose_axes(&[0, 2, 1, 3])
            .unwrap();

        let single_rope_input = RopeInputBuilder::new(&raw_last_key)
            .offset(offset + agreed_prefix.len() as i32 - 1)
            .build()
            .unwrap();
        let single_rotated = model.model.layers[0]
            .self_attn
            .rope
            .forward(single_rope_input)
            .unwrap();
        eval([&single_rotated]).unwrap();

        let max_abs_diff = |a: &Array, b: &Array| -> f32 {
            let a = a.as_type::<f32>().unwrap();
            let b = b.as_type::<f32>().unwrap();
            a.subtract(&b)
                .unwrap()
                .abs()
                .unwrap()
                .max(false)
                .unwrap()
                .item::<f32>()
        };

        let diff = max_abs_diff(&seq_last, &single_rotated);
        eprintln!("qwen2-first-layer-rope-seq-vs-single diff={diff}");
        assert!(diff > 1.0, "expected scalar-offset rope path to diverge, got {diff}");
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_qwen2_first_layer_rope_dynamic_single_token_offset() {
        let mut model = load_qwen2_model(CACHED_TEST_MODEL_DIR).unwrap();
        let tokenizer = load_qwen2_tokenizer(CACHED_TEST_MODEL_DIR).unwrap();

        let prompt = "<|im_start|>system\nYou are a concise assistant.<|im_end|>\n<|im_start|>user\nContext: The native MLX path supports streaming, but some families behave inconsistently.\nTask: Give three short bullets on what is working, what is not, and the next fix.<|im_end|>\n<|im_start|>assistant\n";
        let encoding = tokenizer.encode(prompt, false).unwrap();
        let prompt_ids = encoding.get_ids().to_vec();
        let agreed_prefix = vec![12u32, 576, 9867, 19614, 55];
        let offset = prompt_ids.len() as i32;

        let prefix_tokens = Array::from(agreed_prefix.as_slice()).index(NewAxis);
        let embedded_prefix = model.model.embed_tokens.forward(&prefix_tokens).unwrap();
        let normalized_prefix = model.model.layers[0]
            .input_layernorm
            .forward(&embedded_prefix)
            .unwrap();
        let raw_prefix_keys = model.model.layers[0]
            .self_attn
            .k_proj
            .forward(&normalized_prefix)
            .unwrap()
            .reshape(&[1, agreed_prefix.len() as i32, model.model.layers[0].self_attn.n_kv_heads, -1])
            .unwrap()
            .transpose_axes(&[0, 2, 1, 3])
            .unwrap();

        let seq_rope_input = RopeInputBuilder::new(&raw_prefix_keys)
            .offset(offset)
            .build()
            .unwrap();
        let seq_rotated = model.model.layers[0]
            .self_attn
            .rope
            .forward(seq_rope_input)
            .unwrap();
        let seq_last = seq_rotated.index((.., .., -1.., ..));
        eval([&seq_last]).unwrap();

        let last_token = Array::from_slice(&[agreed_prefix[agreed_prefix.len() - 1]], &[1]).index(NewAxis);
        let embedded_last = model.model.embed_tokens.forward(&last_token).unwrap();
        let normalized_last = model.model.layers[0]
            .input_layernorm
            .forward(&embedded_last)
            .unwrap();
        let raw_last_key = model.model.layers[0]
            .self_attn
            .k_proj
            .forward(&normalized_last)
            .unwrap()
            .reshape(&[1, 1, model.model.layers[0].self_attn.n_kv_heads, -1])
            .unwrap()
            .transpose_axes(&[0, 2, 1, 3])
            .unwrap();

        let dynamic_rotated = match &model.model.layers[0].self_attn.rope {
            crate::utils::rope::RopeVariant::Default(rope) => {
                let dynamic_offset = Array::from_int(offset + agreed_prefix.len() as i32 - 1);
                rope_dynamic(
                    &raw_last_key,
                    rope.dimensions,
                    rope.traditional,
                    Some(rope.base),
                    rope.scale,
                    &dynamic_offset,
                    None::<&Array>,
                )
                .unwrap()
            }
            _ => panic!("expected default rope variant for qwen2"),
        };
        eval([&dynamic_rotated]).unwrap();

        let max_abs_diff = |a: &Array, b: &Array| -> f32 {
            let a = a.as_type::<f32>().unwrap();
            let b = b.as_type::<f32>().unwrap();
            a.subtract(&b)
                .unwrap()
                .abs()
                .unwrap()
                .max(false)
                .unwrap()
                .item::<f32>()
        };

        let diff = max_abs_diff(&seq_last, &dynamic_rotated);
        eprintln!("qwen2-first-layer-rope-seq-vs-dynamic-single diff={diff}");
        assert!(diff < 0.1, "rope seq vs dynamic single mismatch: {diff}");
    }
}
