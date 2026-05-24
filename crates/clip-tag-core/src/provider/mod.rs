pub use crate::pipeline::LocalOnnxProvider;

use serde::{Deserialize, Serialize};

use crate::Result;

/// Shared request/response types for all tagging providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagRequest {
    pub image_path: String,
    pub top_k: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagScore {
    pub label: String,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagResponse {
    pub tags: Vec<TagScore>,
}

/// Exchangeable tagging backend (v0.1: local ONNX only).
pub trait TagProvider: Send + Sync {
    fn id(&self) -> &'static str;

    fn tag_image(&self, request: &TagRequest) -> Result<TagResponse>;
}
