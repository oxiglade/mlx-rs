use hf_hub::{
    api::sync::{Api, ApiBuilder, ApiRepo},
    Repo,
};
use mlx_rs::{
    array,
    module::{Module, ModuleParametersExt},
    ops::indexing::{argmax_axis, IndexOp, NewAxis},
    random::categorical,
    transforms::eval,
    Array,
};
use tokenizers::Tokenizer;

mod model;

use model::{Mistral, MistralInput, MistralOutput, ModelArgs};

type Error = Box<dyn std::error::Error + Send + Sync>;
type Result<T, E = Error> = std::result::Result<T, E>;

use clap::Parser;

#[derive(Parser)]
#[command(about = "Mistral inference example")]
pub struct Cli {
    /// The message to be processed by the model
    #[clap(long, default_value = "In the begging the Unverse was created.")]
    prompt: String,

    /// Maximum number of tokens to generate
    #[clap(long, default_value = "100")]
    max_tokens: usize,

    /// The sampling temperature
    #[clap(long, default_value = "0.0")]
    temp: f32,

    /// The batch size of tokens to generate
    #[clap(long, default_value = "10")]
    tokens_per_eval: usize,

    /// The PRNG seed
    #[clap(long, default_value = "0")]
    seed: u64,
}

fn build_hf_api() -> Result<Api> {
    let cache_dir = std::env::var("HF_CACHE_DIR").ok();

    let mut builder = ApiBuilder::new();
    if let Some(cache_dir) = cache_dir {
        builder = builder.with_cache_dir(cache_dir.into());
    }
    builder.build().map_err(Into::into)
}

fn get_tokenizer(repo: &ApiRepo) -> Result<Tokenizer> {
    let tokenizer_filename = repo.get("tokenizer.json")?;
    let t = Tokenizer::from_file(tokenizer_filename)?;

    Ok(t)
}

fn get_model_args(repo: &ApiRepo) -> Result<ModelArgs> {
    let model_args_filename = repo.get("params.json")?;
    let file = std::fs::File::open(model_args_filename)?;
    let model_args: ModelArgs = serde_json::from_reader(file)?;

    Ok(model_args)
}

fn load_model(repo: &ApiRepo) -> Result<Mistral> {
    let model_args = get_model_args(repo)?;
    let mut model = Mistral::new(&model_args)?;
    let weights_filename = repo.get("weights.safetensors")?;
    model.load_safetensors(weights_filename)?;

    Ok(model)
}

fn sample(logits: &Array, temp: f32) -> Result<Array> {
    match temp {
        0.0 => argmax_axis(logits, -1, None).map_err(Into::into),
        _ => {
            let logits = logits.multiply(array!(1.0 / temp))?;
            categorical(logits, None, None, None).map_err(Into::into)
        }
    }
}

macro_rules! tri {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => return Some(Err(e.into())),
        }
    };
}

struct Generate<'a> {
    model: &'a mut Mistral,
    temp: f32,
    state: GenerateState<'a>,
}

enum GenerateState<'a> {
    Prefill {
        prompt_token: &'a Array,
    },
    Decoding {
        y: Array,
        cache: Vec<Option<(Array, Array)>>,
    },
}

impl<'a> Generate<'a> {
    pub fn new(model: &'a mut Mistral, prompt_token: &'a Array, temp: f32) -> Self {
        Self {
            model,
            temp,
            state: GenerateState::Prefill { prompt_token },
        }
    }
}

impl Iterator for Generate<'_> {
    type Item = Result<Array>;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.state {
            GenerateState::Prefill { prompt_token } => {
                let initial_cache = Vec::with_capacity(0); // This won't allocate
                let input = MistralInput {
                    inputs: prompt_token,
                    cache: &initial_cache,
                };
                let MistralOutput { logits, cache } = tri!(self.model.forward(input));
                let y = tri!(sample(&logits.index((.., -1, ..)), self.temp));

                self.state = GenerateState::Decoding {
                    y: y.clone(),
                    cache,
                };

                Some(Ok(y))
            }
            GenerateState::Decoding { y, cache } => {
                let next_token = y.index((.., NewAxis));
                let input = MistralInput {
                    inputs: &next_token,
                    cache: cache.as_slice(),
                };
                let MistralOutput {
                    logits,
                    cache: new_cache,
                } = tri!(self.model.forward(input));

                let logits = tri!(logits.squeeze_axes(&[1]));
                let y = tri!(sample(&logits, self.temp));

                self.state = GenerateState::Decoding {
                    y: y.clone(),
                    cache: new_cache,
                };

                Some(Ok(y))
            }
        }
    }
}

fn main() -> Result<()> {
    // If you want to manually set the cache directory, you can set the HF_CACHE_DIR
    // environment variable or put it in a .env file located at the root of this example
    // (ie. examples/mistral/.env)
    let _ = dotenv::dotenv();
    let api = build_hf_api()?;

    // Parse args
    let cli = Cli::parse();

    mlx_rs::random::seed(cli.seed)?;

    // The model used in the original example is converted to safetensors and
    // uploaded to the huggingface hub
    let model_id = "minghuaw/Mistral-7B-v0.1".to_string();
    let repo = api.repo(Repo::new(model_id, hf_hub::RepoType::Model));
    println!("[INFO] Loading model... ");
    let tokenizer = get_tokenizer(&repo)?;
    let mut model = load_model(&repo)?;

    model = mlx_rs::nn::quantize(model, None, None)?;

    let encoding = tokenizer.encode(&cli.prompt[..], true)?;
    let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);
    print!("{}", cli.prompt);

    let generate = Generate::new(&mut model, &prompt_tokens, cli.temp);
    let mut tokens = Vec::with_capacity(cli.max_tokens);
    for (token, ntoks) in generate.zip(0..cli.max_tokens) {
        let token = token?;
        tokens.push(token);

        if ntoks == 0 {
            eval(&tokens)?;
        }

        if tokens.len() % cli.tokens_per_eval == 0 {
            eval(&tokens)?;
            let slice: Vec<u32> = tokens.drain(..).map(|t| t.item_exact::<u32>()).collect();
            let s = tokenizer.decode(&slice, true)?;
            print!("{s}");
        }
    }

    eval(&tokens)?;
    let slice: Vec<u32> = tokens.drain(..).map(|t| t.item_exact::<u32>()).collect();
    let s = tokenizer.decode(&slice, true)?;
    println!("{s}");

    println!("------");

    Ok(())
}
