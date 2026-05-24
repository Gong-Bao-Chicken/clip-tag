use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("configuration: {0}")]
    Config(String),

    #[error("model load failed: {0}")]
    Load(String),

    #[error("inference failed: {0}")]
    Inference(String),

    #[error("vocabulary: {0}")]
    Vocab(String),
}
