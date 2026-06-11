use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParameters},
    nn,
    quantization::{MaybeQuantized, Quantizable as _},
    Array,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::{
    cache::KeyValueCache,
    config::{Family, ModelConfig},
    error::Error,
    loader::apply_post_load_memory_policy,
    nn::{ensure_cache_populated, AttentionInput, ModelInput, SwigluMlp},
    qwen3::text::config::ModelArgs,
    utils::{
        create_attention_mask,
        rope::{initialize_rope, RopeVariant},
        scaled_dot_product_attention, AttentionMask,
    },
};

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
    pub q_norm: nn::RmsNorm,
    #[param]
    pub k_norm: nn::RmsNorm,
    #[param]
    pub rope: RopeVariant,
}

impl Attention {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let dim = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;

        let head_dim = args.head_dim;
        let scale = (head_dim as f32).sqrt().recip();

        let q_proj = nn::LinearBuilder::new(dim, n_heads * head_dim)
            .bias(false)
            .build()?;
        let k_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let v_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, dim)
            .bias(false)
            .build()?;

        let q_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(args.rms_norm_eps)
            .build()?;
        let k_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(args.rms_norm_eps)
            .build()?;

        let rope = initialize_rope(
            head_dim,
            args.rope_theta,
            false,
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
            q_norm,
            k_norm,
            rope,
        })
    }
}

impl<C> Module<AttentionInput<'_, C>> for Attention
where
    C: KeyValueCache + Default,
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

        let mut queries = self.q_norm.forward(
            &queries
                .reshape(&[B, L, self.n_heads, -1])?
                .transpose_axes(&[0, 2, 1, 3])?,
        )?;
        let mut keys = self.k_norm.forward(
            &keys
                .reshape(&[B, L, self.n_kv_heads, -1])?
                .transpose_axes(&[0, 2, 1, 3])?,
        )?;
        let values = values
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let output = if let Some(cache) = cache.as_mut() {
            let offset = cache.offset();
            queries = self.rope.forward((&queries, offset))?;
            keys = self.rope.forward((&keys, offset))?;
            cache.attention(&queries, keys, values, self.scale, mask)?
        } else {
            queries = self.rope.forward(&queries)?;
            keys = self.rope.forward(&keys)?;
            scaled_dot_product_attention(queries, keys, values, None::<&mut C>, self.scale, mask)?
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
        self.q_norm.training_mode(mode);
        self.k_norm.training_mode(mode);
        <RopeVariant as Module<nn::RopeInput>>::training_mode(&mut self.rope, mode);
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
    pub mlp: SwigluMlp,

    #[param]
    pub input_layernorm: nn::RmsNorm,

    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
}

impl TransformerBlock {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let num_attention_heads = args.num_attention_heads;
        let hidden_size = args.hidden_size;

        let self_attn = Attention::new(args)?;
        let mlp = SwigluMlp::new(args.hidden_size, args.intermediate_size, false)?;
        let input_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;

        Ok(Self {
            num_attention_heads,
            hidden_size,
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

impl<C> Module<AttentionInput<'_, C>> for TransformerBlock
where
    C: KeyValueCache + Default,
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
pub struct Qwen3Model {
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

impl Qwen3Model {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        assert!(args.vocab_size.is_positive());

        let vocab_size = args.vocab_size;
        let num_hidden_layers = args.num_hidden_layers;

        let embed_tokens = nn::Embedding::new(args.vocab_size, args.hidden_size)?;
        let layers = (0..num_hidden_layers)
            .map(|_| TransformerBlock::new(args))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;

        Ok(Self {
            vocab_size,
            num_hidden_layers,
            embed_tokens: MaybeQuantized::Original(embed_tokens),
            layers,
            norm,
        })
    }
}

impl<C> Module<ModelInput<'_, C>> for Qwen3Model
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
            None => match create_attention_mask(&h, cache, Some(true))? {
                Some(AttentionMask::Array(a)) => Some(a),
                Some(AttentionMask::Causal) => {
                    return Err(Exception::custom("Only `Array` mask is supported"));
                }
                None => None,
            },
        };

        ensure_cache_populated(cache, self.layers.len());

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
    pub model: Qwen3Model,

    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    pub fn new(args: ModelArgs) -> Result<Self, Exception> {
        let model = Qwen3Model::new(&args)?;
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
        <Qwen3Model as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.training_mode(mode);
        }
    }
}

pub fn load_qwen3_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub metadata: HashMap<String, Value>,
    pub weight_map: HashMap<String, String>,
}

pub fn load_qwen3_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let cfg = ModelConfig::from_dir(model_dir)?;
    let quant = cfg.quantization().cloned();
    let Family::Qwen3(args) = cfg.family else {
        return Err(Error::config("config.json model_type is not qwen3"));
    };
    let mut model = Model::new(args)?;
    if let Some(q) = quant {
        model = model.try_into_quantized(q.group_size, q.bits)?;
    }

    let mut raw: HashMap<String, Array> = HashMap::new();
    let weights_index = model_dir.join("model.safetensors.index.json");
    if weights_index.exists() {
        let json = std::fs::read_to_string(weights_index)?;
        let weight_map: WeightMap = serde_json::from_str(&json)?;
        let weight_files: HashSet<&String> = weight_map.weight_map.values().collect();
        for weight_file in weight_files {
            let path = model_dir.join(weight_file);
            for (k, v) in Array::load_safetensors(&path).map_err(Error::LoadWeights)? {
                raw.insert(k, v);
            }
        }
    } else {
        let path = model_dir.join("model.safetensors");
        for (k, v) in Array::load_safetensors(&path).map_err(Error::LoadWeights)? {
            raw.insert(k, v);
        }
    }

    // QuantizedLinear wraps weight in `inner: Linear`; redirect
    // `<prefix>.weight` → `<prefix>.inner.weight` for keys with a `.scales` sibling.
    let quantised_prefixes: HashSet<String> = raw
        .keys()
        .filter_map(|k| k.strip_suffix(".scales").map(|p| p.to_string()))
        .collect();

    let mut params = model.parameters_mut().flatten();
    for (k, v) in raw {
        let key = match k.strip_suffix(".weight") {
            Some(prefix) if quantised_prefixes.contains(prefix) => {
                format!("{prefix}.inner.weight")
            }
            _ => k,
        };
        if let Some(slot) = params.get_mut(&*key) {
            **slot = v;
        }
    }
    drop(params);
    mlx_rs::transforms::eval_params(model.parameters()).map_err(Error::Exception)?;
    apply_post_load_memory_policy();

    Ok(model)
}

#[cfg(test)]
mod tests {
    use mlx_rs::{module::Module, transforms::eval, Array};

    use crate::{cache::KVCache, qwen3::text::model::load_qwen3_tokenizer};

    const CACHED_TEST_MODEL_DIR: &str = "../cache/Qwen3-4B-bf16";

    #[test]
    #[ignore = "requires local model files"]
    fn test_load_qwen3_model() {
        let _model = super::load_qwen3_model(CACHED_TEST_MODEL_DIR).unwrap();
    }

    const CACHED_QUANT_TEST_MODEL_DIR: &str = "../cache/Qwen3-1.7B-MLX-8bit";

    /// Regression guard for the quantised-checkpoint loader path: without
    /// `try_into_quantized` before `load_safetensors`, the packed-uint32
    /// weights overwrite the unquantised `Linear.weight` slot and the first
    /// forward fires `[rms_norm] weight has K elements but x's last dim is D`.
    #[test]
    #[ignore = "requires local quantised model files"]
    fn quantized_qwen3_model_loads_and_forwards() {
        let mut model = super::load_qwen3_model(CACHED_QUANT_TEST_MODEL_DIR).unwrap();
        let prompt = Array::from_slice(&[1i32, 2, 3, 4], &[1, 4]);
        let mut cache: Vec<Option<KVCache>> = Vec::new();
        let input = super::ModelInput {
            inputs: &prompt,
            mask: None,
            cache: &mut cache,
        };
        let logits = Module::forward(&mut model, input).unwrap();
        eval([&logits]).unwrap();
        assert_eq!(logits.shape()[2], model.args.vocab_size);
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_load_tokenizer() {
        let tokenizer = load_qwen3_tokenizer(CACHED_TEST_MODEL_DIR).unwrap();

        let _encoding = tokenizer.encode("Hello, world!", true).unwrap();
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_load_and_run_qwen3_via_generate() {
        use std::ops::ControlFlow;

        use crate::model_context::{generate, load, GenerateParams};
        use crate::sampler::Sampler;
        use crate::user_input::UserInput;

        let mut ctx = load(CACHED_TEST_MODEL_DIR).unwrap();
        let params = GenerateParams {
            max_new_tokens: 10,
            sampling: Sampler::Greedy,
            ..Default::default()
        };
        let result = generate(&mut ctx, UserInput::text("hello"), params, &mut |_, _| {
            ControlFlow::Continue(())
        })
        .unwrap();
        println!("{}", result.text);
        assert!(result.completion_tokens > 0);
    }
}
