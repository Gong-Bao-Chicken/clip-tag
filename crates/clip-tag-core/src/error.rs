use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("configuration: {0}")]
    Config(String),

    #[error("provider: {0}")]
    Provider(String),

    #[error("image: {0}")]
    Image(String),

    #[error("model: {0}")]
    Model(String),

    #[error("metadata: {0}")]
    Metadata(String),

    #[error("policy: {0}")]
    Policy(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl From<clip_tag_model::Error> for Error {
    fn from(value: clip_tag_model::Error) -> Self {
        Self::Model(value.to_string())
    }
}
