//! Image format detection and decoding.

pub mod error;
pub mod format;
pub mod thumbnail;

pub use error::{Error, Result};
pub use format::ImageFormat;

use image::DynamicImage;
use std::path::Path;

/// Upper bound on inference input edge length.
///
/// Common CLIP / SigLIP preprocessors expect 224, 256, or 384 px input. We keep
/// some headroom so the model's own resize gets a reasonable source, but cap
/// the buffer we ever hold in memory.
pub const DEFAULT_MAX_INFERENCE_DIM: u32 = 768;

/// Decode an image at full resolution. Prefer [`load_rgb8_for_inference`] for
/// CLIP scoring — it avoids decoding tens of megapixels when the model only
/// needs a small input.
pub fn load_rgb8(path: &Path) -> Result<image::RgbImage> {
    let format = ImageFormat::from_path(path).ok_or_else(|| {
        Error::UnsupportedFormat(format!("unsupported extension: {}", path.display()))
    })?;

    let image = decode_full(path)?;
    let rgb = image.to_rgb8();
    tracing::debug!(
        path = %path.display(),
        ?format,
        width = rgb.width(),
        height = rgb.height(),
        "decoded image",
    );
    Ok(rgb)
}

/// Same as [`load_rgb8_for_inference`] but returns a `DynamicImage`. Prefer
/// this when feeding the result into a batched embedder that takes
/// `&[DynamicImage]` — avoids a re-conversion.
pub fn load_dynamic_for_inference(path: &Path, max_dim: u32) -> Result<DynamicImage> {
    let rgb = load_rgb8_for_inference(path, max_dim)?;
    Ok(DynamicImage::ImageRgb8(rgb))
}

/// Decode an image sized for CLIP inference.
///
/// For JPEGs, tries the embedded EXIF thumbnail when it is large enough; this
/// skips the full decode entirely on most camera-produced files. Otherwise
/// decodes the full image and downscales (triangle filter) so the longest edge
/// is no larger than `max_dim`. The full-resolution buffer is dropped before
/// returning.
pub fn load_rgb8_for_inference(path: &Path, max_dim: u32) -> Result<image::RgbImage> {
    let format = ImageFormat::from_path(path).ok_or_else(|| {
        Error::UnsupportedFormat(format!("unsupported extension: {}", path.display()))
    })?;

    if matches!(format, ImageFormat::Jpeg) {
        match thumbnail::try_load_jpeg_thumbnail(path, max_dim) {
            Ok(Some(rgb)) => {
                tracing::debug!(
                    path = %path.display(),
                    width = rgb.width(),
                    height = rgb.height(),
                    "used embedded EXIF thumbnail",
                );
                return Ok(rgb);
            }
            Ok(None) => {}
            Err(e) => tracing::debug!(path = %path.display(), "embedded thumbnail unusable: {e}"),
        }
    }

    let image = decode_full(path)?;
    let scaled = downscale(image, max_dim);
    let rgb = scaled.to_rgb8();
    tracing::debug!(
        path = %path.display(),
        ?format,
        width = rgb.width(),
        height = rgb.height(),
        "decoded and downscaled image",
    );
    Ok(rgb)
}

fn decode_full(path: &Path) -> Result<DynamicImage> {
    let reader = image::ImageReader::open(path).map_err(|e| Error::Decode(e.to_string()))?;
    let reader = reader
        .with_guessed_format()
        .map_err(|e| Error::Decode(e.to_string()))?;
    reader.decode().map_err(|e| Error::Decode(e.to_string()))
}

pub(crate) fn downscale(image: DynamicImage, max_dim: u32) -> DynamicImage {
    let (w, h) = (image.width(), image.height());
    let largest = w.max(h);
    if largest <= max_dim || max_dim == 0 {
        return image;
    }
    let scale = max_dim as f32 / largest as f32;
    let new_w = (w as f32 * scale).round().max(1.0) as u32;
    let new_h = (h as f32 * scale).round().max(1.0) as u32;
    image.resize_exact(new_w, new_h, image::imageops::FilterType::Triangle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};
    use tempfile::tempdir;

    #[test]
    fn downscale_preserves_small_images() {
        let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(100, 50, Rgb([0u8, 0, 0])));
        let out = downscale(img, 768);
        assert_eq!((out.width(), out.height()), (100, 50));
    }

    #[test]
    fn downscale_caps_largest_edge() {
        let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(4000, 2000, Rgb([0u8, 0, 0])));
        let out = downscale(img, 768);
        assert_eq!(out.width(), 768);
        assert_eq!(out.height(), 384);
    }

    #[test]
    fn load_rgb8_for_inference_downscales_large_png() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("big.png");
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_pixel(2000, 1000, Rgb([10u8, 20, 30]));
        img.save(&path).unwrap();

        let out = load_rgb8_for_inference(&path, 512).unwrap();
        assert_eq!(out.width(), 512);
        assert_eq!(out.height(), 256);
    }
}
