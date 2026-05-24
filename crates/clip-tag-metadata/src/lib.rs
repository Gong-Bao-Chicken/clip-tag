//! Metadata read/write executor (policy decisions live in `clip-tag-core`).

pub mod error;
pub mod executor;

pub use error::{Error, Result};
pub use executor::{execute_plan, read_field_snapshot};
