use std::path::Path;

/// Supported standard formats for v0.1 (RAW deferred).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Jpeg,
    Png,
    Tiff,
}

impl ImageFormat {
    pub fn from_path(path: &Path) -> Option<Self> {
        match path.extension()?.to_str()?.to_ascii_lowercase().as_str() {
            "jpg" | "jpeg" => Some(Self::Jpeg),
            "png" => Some(Self::Png),
            "tif" | "tiff" => Some(Self::Tiff),
            _ => None,
        }
    }
}
