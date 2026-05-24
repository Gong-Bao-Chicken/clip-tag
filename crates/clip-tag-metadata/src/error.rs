use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("read failed: {0}")]
    Read(String),

    #[error("write failed: {0}")]
    Write(String),
}
