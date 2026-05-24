use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("decode failed: {0}")]
    Decode(String),
}
