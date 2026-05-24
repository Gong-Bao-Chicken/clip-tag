use std::path::Path;

/// Supported standard formats for v0.1 (RAW deferred).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Jpeg,
    Png,
    Tiff,
    Dng,
}

impl ImageFormat {
    pub fn from_path(path: &Path) -> Option<Self> {
        match path.extension()?.to_str()?.to_ascii_lowercase().as_str() {
            "jpg" | "jpeg" => Some(Self::Jpeg),
            "png" => Some(Self::Png),
            "tif" | "tiff" => Some(Self::Tiff),
            "dng" => Some(Self::Dng),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ImageFormat;
    use std::path::Path;

    #[test]
    fn recognizes_dng_case_insensitively() {
        assert_eq!(
            ImageFormat::from_path(Path::new("a.dng")),
            Some(ImageFormat::Dng)
        );
        assert_eq!(
            ImageFormat::from_path(Path::new("a.DNG")),
            Some(ImageFormat::Dng)
        );
    }
}
