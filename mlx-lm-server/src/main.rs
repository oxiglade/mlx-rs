mod api;
mod worker;

use std::{
    error::Error,
    net::{IpAddr, SocketAddr},
    path::PathBuf,
    process::ExitCode,
    sync::mpsc,
    thread,
};

use clap::Parser;

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
    thread::spawn(move || worker::run(&model, cli.cache_bytes, incoming, ready));
    started.recv()??;

    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(async {
            let listener =
                tokio::net::TcpListener::bind(SocketAddr::new(cli.host, cli.port)).await?;
            println!("listening on http://{}", listener.local_addr()?);
            axum::serve(listener, api::router(id, jobs)).await
        })?;
    Ok(())
}
