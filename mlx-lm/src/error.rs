use mlx_rs::error::Exception;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Exception(#[from] Exception),

    #[error(transparent)]
    Conversion(#[from] mlx_rs::error::ConversionError),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Deserialize(#[from] serde_json::Error),

    #[error(transparent)]
    LoadWeights(#[from] mlx_rs::error::IoError),

    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}
