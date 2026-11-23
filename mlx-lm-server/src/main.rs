mod api;
mod disk;
mod worker;

use std::{
    collections::BTreeMap,
    error::Error,
    net::{IpAddr, SocketAddr},
    num::NonZeroUsize,
    path::PathBuf,
    process::ExitCode,
    sync::mpsc,
    thread,
};

use clap::Parser;
use mlx_lm::TemplateValue;
use tokio::signal::unix::{signal, SignalKind};

#[derive(Debug, Parser)]
#[command(name = "mlx-lm-server", version, about)]
struct Cli {
    /// Local safetensors model directory.
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "127.0.0.1")]
    host: IpAddr,
    /// Zero picks a free port.
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// Bytes of prompt cache kept between requests; the last conversation always stays.
    #[arg(long, default_value_t = 8 << 30)]
    cache_bytes: usize,
    /// Directory for prompt caches that outlast the process, in one subdirectory per model.
    #[arg(long)]
    disk_cache: Option<PathBuf>,
    /// Bytes of prompt cache kept in this model's subdirectory of `--disk-cache`.
    #[arg(long, default_value_t = 64 << 30)]
    disk_cache_bytes: u64,
    /// Tokens of K/V in one file of `--disk-cache`.
    #[arg(long, default_value = "2048")]
    disk_cache_block: NonZeroUsize,
    /// Tokens generated for a request that sets no limit. Without it, such a request may
    /// fill the model's context.
    #[arg(long)]
    max_tokens: Option<NonZeroUsize>,
    /// JSON object of chat template variables; a request's `chat_template_kwargs` add to
    /// and override them.
    #[arg(long, default_value = "{}", value_parser = variables)]
    chat_template_args: BTreeMap<String, TemplateValue>,
    /// Name of the message field that carries reasoning text.
    #[arg(long, default_value = "reasoning", value_parser = ["reasoning", "reasoning_content"])]
    reasoning_field: String,
}

fn variables(json: &str) -> serde_json::Result<BTreeMap<String, TemplateValue>> {
    serde_json::from_str(json)
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("mlx-lm-server: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn Error>> {
    let model = std::fs::canonicalize(&cli.model)
        .map_err(|error| format!("{}: {error}", cli.model.display()))?;
    let id = model.display().to_string();
    let (jobs, incoming) = tokio::sync::mpsc::channel(1);
    let (ready, started) = mpsc::sync_channel(0);
    let block = cli.disk_cache_block.get();
    let disk = cli.disk_cache.as_deref();
    let disk = disk.map(|root| disk::Disk::open(root, &model, cli.disk_cache_bytes, block));
    let disk = disk.transpose()?;
    let worker = thread::spawn(move || {
        worker::run(
            &model,
            cli.cache_bytes,
            disk,
            cli.max_tokens,
            &cli.chat_template_args,
            incoming,
            ready,
        )
    });
    started.recv()??;

    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(async {
            let listener =
                tokio::net::TcpListener::bind(SocketAddr::new(cli.host, cli.port)).await?;
            println!("listening on http://{}", listener.local_addr()?);
            let mut terminate = signal(SignalKind::terminate())?;
            axum::serve(listener, api::router(id, cli.reasoning_field, jobs))
                .with_graceful_shutdown(async move {
                    terminate.recv().await;
                })
                .await
        })?;
    // The worker ends once the router is gone, after it has written to disk.
    let _ = worker.join();
    Ok(())
}
