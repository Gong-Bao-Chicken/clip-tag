//! Extract embedded EXIF thumbnails from JPEG files without a full decode.
//!
//! Camera JPEGs almost always carry a 160–1024 px thumbnail in the APP1 EXIF
//! segment. Decoding it instead of the multi-megapixel master saves the bulk
//! of per-image I/O and CPU before the model's own resize.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::{downscale, Error, Result};

/// Don't accept a thumbnail smaller than this — risks degrading model accuracy.
const MIN_USABLE_THUMBNAIL_DIM: u32 = 256;

/// Read just enough of the JPEG header to locate the APP1 EXIF block; bounded
/// so we never pull a multi-megabyte image into memory just to inspect it.
const HEADER_PROBE_BYTES: usize = 256 * 1024;

/// Try to load the embedded EXIF thumbnail from a JPEG.
///
/// Returns `Ok(None)` if there is no thumbnail, it can't be located, or it's
/// smaller than the minimum useful size. Errors are reserved for I/O failures.
pub fn try_load_jpeg_thumbnail(path: &Path, max_dim: u32) -> Result<Option<image::RgbImage>> {
    let mut file = File::open(path).map_err(|e| Error::Decode(e.to_string()))?;
    let header = read_bounded(&mut file, HEADER_PROBE_BYTES)?;

    let Some((tiff_file_offset, tiff_block_len)) = locate_exif_tiff(&header) else {
        return Ok(None);
    };
    // Re-read the TIFF block from the file so we can resolve thumbnail offsets
    // even if they point past our header probe.
    file.seek(SeekFrom::Start(tiff_file_offset as u64))
        .map_err(|e| Error::Decode(e.to_string()))?;
    let mut tiff = vec![0u8; tiff_block_len];
    file.read_exact(&mut tiff)
        .map_err(|e| Error::Decode(e.to_string()))?;

    let Some((thumb_off, thumb_len)) = find_thumbnail_range(&tiff) else {
        return Ok(None);
    };
    let end = thumb_off.checked_add(thumb_len).ok_or_else(|| {
        Error::Decode("thumbnail offset+length overflow".into())
    })?;
    if end > tiff.len() {
        return Ok(None);
    }

    let bytes = &tiff[thumb_off..end];
    let decoded = match image::load_from_memory_with_format(bytes, image::ImageFormat::Jpeg) {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };

    let largest = decoded.width().max(decoded.height());
    let min_dim = MIN_USABLE_THUMBNAIL_DIM.min(max_dim);
    if largest < min_dim {
        return Ok(None);
    }

    let scaled = downscale(decoded, max_dim);
    Ok(Some(scaled.to_rgb8()))
}

fn read_bounded(file: &mut File, limit: usize) -> Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(limit.min(64 * 1024));
    file.take(limit as u64)
        .read_to_end(&mut buf)
        .map_err(|e| Error::Decode(e.to_string()))?;
    Ok(buf)
}

/// Returns `(file_offset_of_tiff_header, length_of_tiff_block_within_app1)`.
fn locate_exif_tiff(jpeg: &[u8]) -> Option<(usize, usize)> {
    if jpeg.get(0..2)? != [0xFF, 0xD8] {
        return None;
    }
    let mut i = 2;
    while i + 4 <= jpeg.len() {
        if jpeg[i] != 0xFF {
            return None;
        }
        // Skip JPEG fill bytes (a run of 0xFF before the marker).
        while i < jpeg.len() && jpeg[i] == 0xFF {
            i += 1;
        }
        if i >= jpeg.len() {
            return None;
        }
        let marker = jpeg[i];
        i += 1;

        // Standalone markers carry no payload.
        if marker == 0xD8 || matches!(marker, 0xD0..=0xD7) || marker == 0x01 {
            continue;
        }
        // SOS or EOI: image data starts; no further metadata segments to walk.
        if marker == 0xDA || marker == 0xD9 {
            return None;
        }

        if i + 2 > jpeg.len() {
            return None;
        }
        let seg_len = u16::from_be_bytes([jpeg[i], jpeg[i + 1]]) as usize;
        if seg_len < 2 {
            return None;
        }
        let payload_start = i + 2;
        let payload_end = i + seg_len;
        if payload_end > jpeg.len() {
            return None;
        }

        if marker == 0xE1 && payload_end - payload_start >= 6 {
            if &jpeg[payload_start..payload_start + 6] == b"Exif\0\0" {
                let tiff_start = payload_start + 6;
                return Some((tiff_start, payload_end - tiff_start));
            }
        }

        i = payload_end;
    }
    None
}

fn find_thumbnail_range(tiff: &[u8]) -> Option<(usize, usize)> {
    if tiff.len() < 8 {
        return None;
    }
    let little_endian = match &tiff[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let read_u16 = |s: &[u8]| {
        if little_endian {
            u16::from_le_bytes([s[0], s[1]])
        } else {
            u16::from_be_bytes([s[0], s[1]])
        }
    };
    let read_u32 = |s: &[u8]| {
        if little_endian {
            u32::from_le_bytes([s[0], s[1], s[2], s[3]])
        } else {
            u32::from_be_bytes([s[0], s[1], s[2], s[3]])
        }
    };

    if read_u16(&tiff[2..4]) != 0x002A {
        return None;
    }
    let ifd0_offset = read_u32(&tiff[4..8]) as usize;
    if ifd0_offset + 2 > tiff.len() {
        return None;
    }
    let ifd0_entries = read_u16(&tiff[ifd0_offset..ifd0_offset + 2]) as usize;
    let ifd0_end = ifd0_offset.checked_add(2 + ifd0_entries.checked_mul(12)?)?;
    if ifd0_end + 4 > tiff.len() {
        return None;
    }
    let ifd1_offset = read_u32(&tiff[ifd0_end..ifd0_end + 4]) as usize;
    if ifd1_offset == 0 || ifd1_offset + 2 > tiff.len() {
        return None;
    }
    let ifd1_entries = read_u16(&tiff[ifd1_offset..ifd1_offset + 2]) as usize;

    let mut thumb_offset = None;
    let mut thumb_length = None;
    for n in 0..ifd1_entries {
        let entry = ifd1_offset + 2 + n * 12;
        if entry + 12 > tiff.len() {
            return None;
        }
        let tag = read_u16(&tiff[entry..entry + 2]);
        let value = read_u32(&tiff[entry + 8..entry + 12]) as usize;
        match tag {
            0x0201 => thumb_offset = Some(value),
            0x0202 => thumb_length = Some(value),
            _ => {}
        }
    }
    Some((thumb_offset?, thumb_length?))
}

#[cfg(test)]
mod tests {
    use super::{locate_exif_tiff, MIN_USABLE_THUMBNAIL_DIM};

    #[test]
    fn no_exif_segment_returns_none() {
        // Bare JPEG SOI/EOI with no APP1.
        let bytes = [0xFF, 0xD8, 0xFF, 0xD9];
        assert!(locate_exif_tiff(&bytes).is_none());
    }

    #[test]
    fn min_usable_thumbnail_is_sensible() {
        assert!(MIN_USABLE_THUMBNAIL_DIM >= 160);
    }
}
