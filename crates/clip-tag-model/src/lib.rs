//! Local ONNX CLIP session and vocabulary scoring.

pub mod engine;
pub mod error;
pub mod external_data;
pub mod vocab;
pub mod vocab_cache;

pub use engine::{
    load_shared, recommended_model_for, ExecutionProvider, ModelConfig, RecommendedModel,
    SharedEngine, TagScore, TaggingEngine,
};
pub use error::{Error, Result};
pub use vocab::{load_labels, resolve_vocab_path};

pub const DEFAULT_MODEL_ID: &str = "RuteNL/MobileCLIP2-S3-OpenCLIP-ONNX";
