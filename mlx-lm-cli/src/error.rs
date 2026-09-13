use std::io;

#[derive(Debug, thiserror::Error)]
pub(crate) enum CliError {
    #[error("model load failed")]
    Load(#[from] mlx_lm::LoadError),
    #[cfg(feature = "hf-hub")]
    #[error("Hub load failed")]
    Hub(#[from] mlx_lm::HubError),
    #[error("generation failed")]
    Generation(#[from] mlx_lm::GenerationError),
    #[error("output failed")]
    Output(#[from] io::Error),
    #[error("JSON encoding failed")]
    Json(#[from] serde_json::Error),
    #[error("unsupported public generation event or finish reason")]
    UnsupportedEvent,
}

impl CliError {
    pub fn is_broken_pipe(&self) -> bool {
        matches!(self, Self::Output(error) if error.kind() == io::ErrorKind::BrokenPipe)
    }
}
