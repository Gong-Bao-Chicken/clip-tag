//! Image format detection and decoding.

pub mod error;
pub mod format;

pub use error::{Error, Result};
pub use format::ImageFormat;

/// Load a supported image as RGB8 for model inference.
pub fn load_rgb8(path: &std::path::Path) -> Result<image::RgbImage> {
    let format = ImageFormat::from_path(path).ok_or_else(|| {
        Error::UnsupportedFormat(format!("unsupported extension: {}", path.display()))
    })?;

    let reader = image::ImageReader::open(path).map_err(|e| Error::Decode(e.to_string()))?;
    let reader = reader
        .with_guessed_format()
        .map_err(|e| Error::Decode(e.to_string()))?;

    let image = reader.decode().map_err(|e| Error::Decode(e.to_string()))?;

    let rgb = image.to_rgb8();
    tracing::debug!(
        path = %path.display(),
        ?format,
        width = rgb.width(),
        height = rgb.height(),
        "decoded image"
    );
    Ok(rgb)
}
