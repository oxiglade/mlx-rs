use std::{num::NonZeroUsize, path::PathBuf};

use clap::{Args, Parser, Subcommand, ValueEnum};
use mlx_lm::{GenerationOptions, MinPOptions};

#[derive(Debug, Parser)]
#[command(name = "mlx-lm", version, about, disable_help_subcommand = true)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Stream generation from a literal prompt.
    Generate(GenerateArgs),
    /// Strictly load a model and report its public facts.
    Info(InfoArgs),
}

#[derive(Debug, Args)]
#[group(skip)]
#[cfg_attr(feature = "hf-hub", command(group(clap::ArgGroup::new("source").required(true).args(["model", "repo"]))))]
#[cfg_attr(not(feature = "hf-hub"), command(group(clap::ArgGroup::new("source").required(true).args(["model"]))))]
pub(crate) struct Source {
    /// Local safetensors model directory.
    #[arg(long)]
    pub model: Option<PathBuf>,
    #[cfg(feature = "hf-hub")]
    #[arg(long)]
    pub repo: Option<String>,
    #[cfg(feature = "hf-hub")]
    #[arg(long, requires = "repo", conflicts_with = "model")]
    pub revision: Option<String>,
    #[cfg(feature = "hf-hub")]
    #[arg(long, requires = "repo", conflicts_with = "model")]
    pub offline: bool,
    #[cfg(feature = "hf-hub")]
    #[arg(long, requires = "repo", conflicts_with = "model")]
    pub cache_dir: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct GenerateArgs {
    #[command(flatten)]
    pub source: Source,
    #[arg(long, allow_hyphen_values = true)]
    pub prompt: String,
    #[arg(long)]
    pub max_tokens: Option<NonZeroUsize>,
    #[arg(long, value_parser = temperature)]
    pub temperature: Option<f32>,
    #[arg(long, value_parser = top_p)]
    pub top_p: Option<f32>,
    #[arg(long)]
    pub top_k: Option<NonZeroUsize>,
    #[arg(long, value_parser = min_p)]
    pub min_p: Option<f32>,
    #[arg(long, requires = "min_p")]
    pub min_tokens_to_keep: Option<NonZeroUsize>,
    #[arg(long)]
    pub seed: Option<u64>,
    #[arg(long, value_parser = nonempty, allow_hyphen_values = true)]
    pub stop: Vec<String>,
    #[arg(long, value_enum, default_value = "text")]
    pub format: GenerationFormat,
}

#[derive(Debug, Args)]
pub(crate) struct InfoArgs {
    #[command(flatten)]
    pub source: Source,
    #[arg(long, value_enum, default_value = "text")]
    pub format: InfoFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum GenerationFormat {
    Text,
    Jsonl,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum InfoFormat {
    Text,
    Json,
}

impl GenerateArgs {
    pub fn options(&self) -> GenerationOptions {
        let mut options = GenerationOptions::default();
        if let Some(max_tokens) = self.max_tokens {
            options.max_tokens = max_tokens;
        }
        if let Some(temperature) = self.temperature {
            options.sampling.temperature = temperature;
        }
        options.sampling.top_p = self.top_p;
        options.sampling.top_k = self.top_k;
        options.sampling.min_p = self.min_p.map(|probability| MinPOptions {
            probability,
            min_tokens_to_keep: self.min_tokens_to_keep.unwrap_or(NonZeroUsize::MIN),
        });
        options.sampling.seed = self.seed;
        options.stop.stop_strings.clone_from(&self.stop);
        options
    }
}

#[cfg(feature = "hf-hub")]
impl Source {
    pub fn hub_options(&self) -> mlx_lm::HubOptions {
        let mut options = mlx_lm::HubOptions::default();
        options.revision.clone_from(&self.revision);
        options.offline = self.offline;
        options.cache_dir.clone_from(&self.cache_dir);
        options
    }
}

fn temperature(value: &str) -> Result<f32, String> {
    let number = value.parse::<f32>().map_err(|error| error.to_string())?;
    if number.is_finite() && number >= 0.0 {
        Ok(number)
    } else {
        Err("temperature must be finite and nonnegative".into())
    }
}
fn top_p(value: &str) -> Result<f32, String> {
    let number = value.parse::<f32>().map_err(|error| error.to_string())?;
    if number.is_finite() && number > 0.0 && number <= 1.0 {
        Ok(number)
    } else {
        Err("top-p must be finite and in (0, 1]".into())
    }
}
fn min_p(value: &str) -> Result<f32, String> {
    let number = value.parse::<f32>().map_err(|error| error.to_string())?;
    if number.is_finite() && (0.0..=1.0).contains(&number) {
        Ok(number)
    } else {
        Err("min-p must be finite and in [0, 1]".into())
    }
}
fn nonempty(value: &str) -> Result<String, String> {
    if value.is_empty() {
        Err("stop strings must not be empty".into())
    } else {
        Ok(value.into())
    }
}

#[cfg(test)]
mod tests;
