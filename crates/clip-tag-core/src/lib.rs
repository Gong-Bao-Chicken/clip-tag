//! Orchestration, write policy, normalization, and provider routing.

pub mod config;
pub mod dry_run;
pub mod error;
pub mod logging;
pub mod normalize;
pub mod pipeline;
pub mod policy;
pub mod provider;
pub mod write;

pub use error::{Error, Result};
pub use pipeline::{default_config_from_cli, BatchResult, FileTagResult, Pipeline};
