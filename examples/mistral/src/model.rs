use mlx_rs::{
    builder::Builder,
    error::Exception,
    fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask},
    macros::{ModuleParameters, Quantizable},
    module::Module,
    nn,
    ops::concatenate,
    quantization::MaybeQuantized,
    Array,
};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub dim: i32,
    pub n_layers: i32,
    pub head_dim: i32,
    pub hidden_dim: i32,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub norm_eps: f32,
    pub vocab_size: i32,
    pub rope_theta: Option<f32>,
}

impl ModelArgs {
    pub const DEFAULT_ROPE_THETA: f32 = 10000.0;
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Attention {
    n_heads: i32,
    n_kv_heads: i32,
    repeats: i32,
    scale: f32,

    #[quantizable]
    #[param]
    wq: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    wk: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    wv: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    wo: MaybeQuantized<nn::Linear>,

    #[param]
    rope: nn::Rope,
}

impl Attention {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let n_heads = args.n_heads;
        let n_kv_heads = args.n_kv_heads;
        let repeats = n_heads / n_kv_heads;
        let scale = (args.head_dim as f32).powf(-0.5);

        let wq = nn::LinearBuilder::new(args.dim, n_heads * args.head_dim)
            .bias(false)
            .build()?;
        let wk = nn::LinearBuilder::new(args.dim, n_kv_heads * args.head_dim)
            .bias(false)
            .build()?;
        let wv = nn::LinearBuilder::new(args.dim, n_kv_heads * args.head_dim)
            .bias(false)
            .build()?;
        let wo = nn::LinearBuilder::new(n_heads * args.head_dim, args.dim)
            .bias(false)
            .build()?;
        let rope = nn::RopeBuilder::new(args.head_dim)
            .traditional(true)
            .base(args.rope_theta.unwrap_or(ModelArgs::DEFAULT_ROPE_THETA))
            .build()?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            repeats,
            scale,
            wq: MaybeQuantized::new(wq),
            wk: MaybeQuantized::new(wk),
            wv: MaybeQuantized::new(wv),
            wo: MaybeQuantized::new(wo),
            rope,
        })
    }
}

struct AttentionInput<'a> {
    x: &'a Array,
    mask: Option<ScaledDotProductAttentionMask<'a>>,
    cache: Option<(&'a Array, &'a Array)>,
}

struct AttentionOutput {
    output: Array,
    cache: (Array, Array),
}

impl Module<AttentionInput<'_>> for Attention {
    type Output = AttentionOutput;

    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: AttentionInput<'_>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, cache } = input;

        // NOTE: this will panic if the input shape is not correct
        let B = x.shape()[0];
        let L = x.shape()[1];

        let mut queries = self.wq.forward(x)?;
        let mut keys = self.wk.forward(x)?;
        let mut values = self.wv.forward(x)?;

        // Prepare the queries, keys, and values for the attention computation
        queries = queries
            .reshape(&[B, L, self.n_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        keys = keys
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        values = values
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        match cache {
            Some((key_cache, value_cache)) => {
                let offset = key_cache.shape()[2];
                queries = self.rope.forward((&queries, offset))?;
                keys = self.rope.forward((&keys, offset))?;
                keys = concatenate(&[key_cache, &keys], 2)?;
                values = concatenate(&[value_cache, &values], 2)?;
            }
            None => {
                queries = self.rope.forward(&queries)?;
                keys = self.rope.forward(&keys)?;
            }
        }

        let output = scaled_dot_product_attention(queries, &keys, &values, self.scale, mask, None)?;
        let output = output.transpose_axes(&[0, 2, 1, 3])?.reshape(&[B, L, -1])?;
        let output = self.wo.forward(&output)?;

        Ok(AttentionOutput {
            output,
            cache: (keys, values),
        })
    }

    fn training_mode(&mut self, mode: bool) {
        self.wq.training_mode(mode);
        self.wk.training_mode(mode);
        self.wv.training_mode(mode);
        self.wo.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct FeedForward {
    #[quantizable]
    #[param]
    w1: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    w2: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    w3: MaybeQuantized<nn::Linear>,
}

impl FeedForward {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let w1 = nn::LinearBuilder::new(args.dim, args.hidden_dim)
            .bias(false)
            .build()?;
        let w2 = nn::LinearBuilder::new(args.hidden_dim, args.dim)
            .bias(false)
            .build()?;
        let w3 = nn::LinearBuilder::new(args.dim, args.dim)
            .bias(false)
            .build()?;
        Ok(Self {
            w1: MaybeQuantized::new(w1),
            w2: MaybeQuantized::new(w2),
            w3: MaybeQuantized::new(w3),
        })
    }
}

impl Module<&Array> for FeedForward {
    type Output = Array;

    type Error = Exception;

    fn forward(&mut self, x: &'_ Array) -> Result<Self::Output, Self::Error> {
        let w2_input = nn::silu(self.w1.forward(x)?)?.multiply(self.w3.forward(x)?)?;
        self.w2.forward(&w2_input)
    }

    fn training_mode(&mut self, mode: bool) {
        self.w1.training_mode(mode);
        self.w2.training_mode(mode);
        self.w3.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct TransformerBlock {
    n_heads: i32,
    dim: i32,

    #[quantizable]
    #[param]
    attention: Attention,

    #[quantizable]
    #[param]
    feed_forward: FeedForward,

    #[param]
    attention_norm: nn::RmsNorm,

    #[param]
    ffn_norm: nn::RmsNorm,
}

impl TransformerBlock {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let n_heads = args.n_heads;
        let dim = args.dim;

        let attention = Attention::new(args)?;
        let feed_forward = FeedForward::new(args)?;
        let attention_norm = nn::RmsNormBuilder::new(dim).eps(args.norm_eps).build()?;
        let ffn_norm = nn::RmsNormBuilder::new(dim).eps(args.norm_eps).build()?;
        Ok(Self {
            n_heads,
            dim,
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
        })
    }
}

impl Module<AttentionInput<'_>> for TransformerBlock {
    type Output = AttentionOutput;

    type Error = Exception;

    fn forward(&mut self, input: AttentionInput<'_>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, cache } = input;
        let norm_x = self.attention_norm.forward(x)?;
        let attention_input = AttentionInput {
            x: &norm_x,
            mask,
            cache,
        };
        let attention_output = self.attention.forward(attention_input)?;

        let r = attention_output.output;
        let cache = attention_output.cache;

        let h = x.add(r)?;
        let r = self.feed_forward.forward(&self.ffn_norm.forward(&h)?)?;
        let output = h.add(r)?;

        Ok(AttentionOutput { output, cache })
    }

    fn training_mode(&mut self, mode: bool) {
        self.attention.training_mode(mode);
        self.feed_forward.training_mode(mode);
        self.attention_norm.training_mode(mode);
        self.ffn_norm.training_mode(mode);
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MistralError {
    #[error("Invalid vocab size: {0}")]
    InvalidVocabSize(i32),

    #[error(transparent)]
    Exception(#[from] Exception),
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Mistral {
    vocab_size: i32,
    n_layers: i32,

    #[quantizable]
    #[param]
    tok_embeddings: MaybeQuantized<nn::Embedding>,

    #[quantizable]
    #[param]
    layers: Vec<TransformerBlock>,

    #[param]
    norm: nn::RmsNorm,

    #[quantizable]
    #[param]
    output: MaybeQuantized<nn::Linear>,
}

impl Mistral {
    pub fn new(args: &ModelArgs) -> Result<Self, MistralError> {
        let vocab_size = args.vocab_size;
        if vocab_size <= 0 {
            // We would still have to check for the zero case even if we switch to u32
            return Err(MistralError::InvalidVocabSize(vocab_size));
        }
        let n_layers = args.n_layers;

        let tok_embeddings = nn::Embedding::new(vocab_size, args.dim)?;
        let layers = (0..n_layers)
            .map(|_| TransformerBlock::new(args))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(args.dim)
            .eps(args.norm_eps)
            .build()?;
        let output = nn::LinearBuilder::new(args.dim, vocab_size)
            .bias(false)
            .build()?;

        Ok(Self {
            vocab_size,
            n_layers,
            tok_embeddings: MaybeQuantized::new(tok_embeddings),
            layers,
            norm,
            output: MaybeQuantized::new(output),
        })
    }
}

pub struct MistralInput<'a> {
    pub inputs: &'a Array,
    pub cache: &'a [Option<(Array, Array)>],
}
pub struct MistralOutput {
    pub logits: Array,
    pub cache: Vec<Option<(Array, Array)>>,
}

impl Module<MistralInput<'_>> for Mistral {
    type Output = MistralOutput;

    type Error = MistralError;

    fn forward(&mut self, input: MistralInput<'_>) -> Result<Self::Output, Self::Error> {
        let MistralInput { inputs, cache } = input;

        let mut h = self.tok_embeddings.forward(inputs)?;

        let mut mask = None;
        if h.shape()[1] > 1 {
            let mask_ = nn::MultiHeadAttention::create_additive_causal_mask::<f32>(h.shape()[1])?;
            let mask_ = mask_.as_dtype(h.dtype())?;
            mask = Some(mask_);
        }

        let mut out_cache = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let cache_entry = cache.get(i).and_then(Option::as_ref).map(|(k, v)| (k, v));
            let input = AttentionInput {
                x: &h,
                mask: mask.as_ref().map(Into::into),
                cache: cache_entry,
            };
            let output = layer.forward(input)?;
            h = output.output;
            out_cache.push(Some(output.cache));
        }

        let output = self.output.forward(&self.norm.forward(&h)?)?;

        Ok(MistralOutput {
            logits: output,
            cache: out_cache,
        })
    }

    fn training_mode(&mut self, mode: bool) {
        self.tok_embeddings.training_mode(mode);
        self.layers
            .iter_mut()
            .for_each(|layer| layer.training_mode(mode));
        self.norm.training_mode(mode);
        self.output.training_mode(mode);
    }
}
