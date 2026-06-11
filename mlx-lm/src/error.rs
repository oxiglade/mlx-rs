use mlx_rs::error::Exception;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Exception(#[from] Exception),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Deserialize(#[from] serde_json::Error),

    #[error(transparent)]
    LoadWeights(#[from] mlx_rs::error::IoError),

    /// A tensor shape, rank, or axis assertion failed.
    #[error("shape mismatch: {0}")]
    Shape(String),

    /// A required `config.json` key was missing or held an invalid value.
    #[error("config: {reason}")]
    Config { reason: String },

    /// An index, token id, or grid coordinate fell outside its valid range.
    #[error("out of bounds: {0}")]
    OutOfBounds(String),

    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

impl Error {
    pub fn config(reason: impl Into<String>) -> Self {
        Self::Config {
            reason: reason.into(),
        }
    }

    pub fn shape(reason: impl Into<String>) -> Self {
        Self::Shape(reason.into())
    }

    pub fn out_of_bounds(reason: impl Into<String>) -> Self {
        Self::OutOfBounds(reason.into())
    }
}

/// Collapse [`Error`] back into the FFI [`Exception`]. Lossy: non-Exception
/// variants format into `Exception::custom`. Needed because the mlx-rs
/// `Module` trait fixes `type Error = Exception`, so a helper returning
/// `Error` that is `?`-bubbled inside a `Module::forward` must lift.
impl From<Error> for Exception {
    fn from(e: Error) -> Self {
        match e {
            Error::Exception(ex) => ex,
            other => Self::custom(other.to_string()),
        }
    }
}
