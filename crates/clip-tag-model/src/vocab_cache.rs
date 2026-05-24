use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use ndarray::Array2;
use open_clip_inference::text::TextEmbedder;

use crate::{Error, Result};

const MAGIC: &[u8; 8] = b"CLIPTAG1";

pub fn cache_path_for(vocab_path: &Path, model_id: &str) -> PathBuf {
    let canonical = std::fs::canonicalize(vocab_path).unwrap_or_else(|_| vocab_path.to_path_buf());
    let canonical_str = canonical.to_string_lossy();
    let mut hasher = blake3::Hasher::new();
    hasher.update(canonical_str.as_bytes());
    hasher.update(&[0]); // Delimiter so path/model_id boundaries are unambiguous.
    hasher.update(model_id.as_bytes());
    let hash = hasher.finalize().to_hex();

    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("clip-tag")
        .join("vocab-cache")
        .join(format!("{hash}.bin"))
}

pub fn load_cache(path: &Path, label_count: usize) -> Result<Array2<f32>> {
    let mut file = File::open(path).map_err(|e| Error::Vocab(e.to_string()))?;
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)
        .map_err(|e| Error::Vocab(e.to_string()))?;
    if &magic != MAGIC {
        return Err(Error::Vocab("invalid vocab cache magic".into()));
    }

    let mut count_buf = [0u8; 4];
    file.read_exact(&mut count_buf)
        .map_err(|e| Error::Vocab(e.to_string()))?;
    let count = u32::from_le_bytes(count_buf) as usize;
    if count != label_count {
        return Err(Error::Vocab("vocab cache label count mismatch".into()));
    }

    let mut dim_buf = [0u8; 4];
    file.read_exact(&mut dim_buf)
        .map_err(|e| Error::Vocab(e.to_string()))?;
    let dim = u32::from_le_bytes(dim_buf) as usize;

    let mut bytes = vec![0u8; count * dim * 4];
    file.read_exact(&mut bytes)
        .map_err(|e| Error::Vocab(e.to_string()))?;
    let data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Array2::from_shape_vec((count, dim), data).map_err(|e| Error::Vocab(e.to_string()))
}

pub fn save_cache(path: &Path, embeddings: &Array2<f32>) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| Error::Vocab(e.to_string()))?;
    }
    let mut file = File::create(path).map_err(|e| Error::Vocab(e.to_string()))?;
    file.write_all(MAGIC)
        .map_err(|e| Error::Vocab(e.to_string()))?;
    file.write_all(&(embeddings.nrows() as u32).to_le_bytes())
        .map_err(|e| Error::Vocab(e.to_string()))?;
    file.write_all(&(embeddings.ncols() as u32).to_le_bytes())
        .map_err(|e| Error::Vocab(e.to_string()))?;
    for value in embeddings.iter() {
        file.write_all(&value.to_le_bytes())
            .map_err(|e| Error::Vocab(e.to_string()))?;
    }
    Ok(())
}

pub fn embed_labels(text: &TextEmbedder, labels: &[String]) -> Result<Array2<f32>> {
    const BATCH: usize = 128;
    let mut out: Option<Array2<f32>> = None;
    for chunk in labels.chunks(BATCH) {
        tracing::info!(batch = chunk.len(), "embedding vocabulary batch");
        let refs: Vec<&str> = chunk.iter().map(String::as_str).collect();
        let batch = text
            .embed_texts(&refs)
            .map_err(|e| Error::Load(e.to_string()))?;
        out = Some(if let Some(existing) = out {
            ndarray::concatenate(ndarray::Axis(0), &[existing.view(), batch.view()])
                .map_err(|e| Error::Load(e.to_string()))?
        } else {
            batch
        });
    }
    out.ok_or_else(|| Error::Load("empty vocabulary".into()))
}

#[cfg(test)]
mod tests {
    use super::cache_path_for;

    use std::path::Path;

    use tempfile::tempdir;

    #[test]
    fn cache_path_is_stable_for_equivalent_paths() {
        let dir = tempdir().expect("temp dir");
        let vocab = dir.path().join("vocab.txt");
        std::fs::write(&vocab, "token").expect("write vocab");

        let from_plain = cache_path_for(&vocab, "test-model");
        let from_equivalent = cache_path_for(
            Path::new(&dir.path().join(".").join("vocab.txt")),
            "test-model",
        );

        assert_eq!(from_plain, from_equivalent);
    }

    #[test]
    fn cache_path_changes_when_model_changes() {
        let dir = tempdir().expect("temp dir");
        let vocab = dir.path().join("vocab.txt");
        std::fs::write(&vocab, "token").expect("write vocab");

        let model_a = cache_path_for(&vocab, "model-a");
        let model_b = cache_path_for(&vocab, "model-b");

        assert_ne!(model_a, model_b);
    }

    #[test]
    fn cache_file_name_is_hex_and_safe() {
        let dir = tempdir().expect("temp dir");
        let vocab = dir.path().join("vocab.txt");
        std::fs::write(&vocab, "token").expect("write vocab");

        let path = cache_path_for(&vocab, "model");
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("utf-8 file name");

        assert_eq!(file_name.len(), 68); // 64-char digest + ".bin"
        assert!(file_name.ends_with(".bin"));
        assert!(file_name[..64].chars().all(|c| c.is_ascii_hexdigit()));
    }
}
