use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::policy::WriteMode;

/// Layered configuration (CLI overrides file overrides defaults).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub write: WriteConfig,
    pub model: ModelConfig,
    pub batch: BatchConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WriteConfig {
    /// When true, metadata may be written to disk (requires explicit CLI flag in v0.1).
    pub enabled: bool,
    pub mode: WriteMode,
    pub dry_run: bool,
    pub force: bool,
}

impl Default for WriteConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: WriteMode::EmptyOnly,
            dry_run: false,
            force: false,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    pub model_id: Option<String>,
    pub model_path: Option<String>,
    pub provider: Option<String>,
    pub vocab_path: Option<PathBuf>,
    pub diversity_threshold: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BatchConfig {
    pub recursive: bool,
    pub continue_on_error: bool,
    /// Number of images per ORT vision call. `1` disables batching.
    pub batch_size: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            recursive: false,
            continue_on_error: false,
            batch_size: 1,
        }
    }
}

impl Config {
    pub fn from_toml_str(s: &str) -> crate::Result<Self> {
        toml::from_str(s).map_err(|e| crate::Error::Config(e.to_string()))
    }
}
