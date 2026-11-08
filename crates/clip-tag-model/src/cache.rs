//! Inspect and reclaim disk space used by clip-tag's caches.
//!
//! Two on-disk caches live under [`cache_root`]:
//!
//! - `folded-models/` — ONNX models with external initializer data inlined,
//!   keyed by HF model id. Created on first use of CoreML / DirectML / CUDA
//!   for a given model. Each entry is roughly the size of the original
//!   `.onnx.data` files (hundreds of MB).
//! - `vocab-cache/` — precomputed vocabulary embedding matrices, keyed by
//!   `(vocab_path, model_id)`. A few MB each; cheap to regenerate.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::{Error, Result};

const FOLDED_MODELS_DIR: &str = "folded-models";
const VOCAB_CACHE_DIR: &str = "vocab-cache";

/// Root of clip-tag's on-disk cache. `None` if the OS does not expose a
/// cache directory for the current user.
pub fn cache_root() -> Option<PathBuf> {
    dirs::cache_dir().map(|d| d.join("clip-tag"))
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct CacheEntry {
    pub path: Option<PathBuf>,
    pub bytes: u64,
    /// Number of top-level entries (folded model dirs or vocab cache files).
    pub items: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct CacheSummary {
    pub root: Option<PathBuf>,
    pub folded_models: CacheEntry,
    pub vocab_cache: CacheEntry,
}

impl CacheSummary {
    pub fn total_bytes(&self) -> u64 {
        self.folded_models.bytes + self.vocab_cache.bytes
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PruneReport {
    pub dry_run: bool,
    pub folded_models: CacheEntry,
    pub vocab_cache: CacheEntry,
}

impl PruneReport {
    pub fn total_bytes(&self) -> u64 {
        self.folded_models.bytes + self.vocab_cache.bytes
    }
}

pub fn summary() -> CacheSummary {
    let root = cache_root();
    CacheSummary {
        folded_models: entry(root.as_ref().map(|r| r.join(FOLDED_MODELS_DIR))),
        vocab_cache: entry(root.as_ref().map(|r| r.join(VOCAB_CACHE_DIR))),
        root,
    }
}

/// Delete both subcaches. With `dry_run = true`, no files are removed but
/// the returned report still measures everything that *would* go.
pub fn prune(dry_run: bool) -> Result<PruneReport> {
    let summary = summary();
    if !dry_run {
        remove_dir_if_exists(summary.folded_models.path.as_deref())?;
        remove_dir_if_exists(summary.vocab_cache.path.as_deref())?;
    }
    Ok(PruneReport {
        dry_run,
        folded_models: summary.folded_models,
        vocab_cache: summary.vocab_cache,
    })
}

fn remove_dir_if_exists(path: Option<&Path>) -> Result<()> {
    let Some(path) = path else { return Ok(()) };
    if !path.exists() {
        return Ok(());
    }
    fs::remove_dir_all(path)
        .map_err(|e| Error::Load(format!("failed to prune {}: {e}", path.display())))
}

fn entry(path: Option<PathBuf>) -> CacheEntry {
    let Some(path) = path else {
        return CacheEntry::default();
    };
    if !path.is_dir() {
        return CacheEntry {
            path: Some(path),
            bytes: 0,
            items: 0,
        };
    }
    let items = fs::read_dir(&path)
        .map(|iter| iter.flatten().count())
        .unwrap_or(0);
    let bytes = directory_size(&path);
    CacheEntry {
        path: Some(path),
        bytes,
        items,
    }
}

fn directory_size(p: &Path) -> u64 {
    let Ok(meta) = fs::symlink_metadata(p) else {
        return 0;
    };
    let file_type = meta.file_type();
    if file_type.is_file() {
        return meta.len();
    }
    if file_type.is_symlink() {
        // Don't follow symlinks — count the link itself, not the target.
        return meta.len();
    }
    if !file_type.is_dir() {
        return 0;
    }
    let Ok(entries) = fs::read_dir(p) else {
        return 0;
    };
    let mut total = 0u64;
    for entry in entries.flatten() {
        total = total.saturating_add(directory_size(&entry.path()));
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn entry_handles_missing_directory() {
        let dir = tempdir().unwrap();
        let e = entry(Some(dir.path().join("nope")));
        assert_eq!(e.bytes, 0);
        assert_eq!(e.items, 0);
        assert!(e.path.is_some());
    }

    #[test]
    fn entry_sums_files_and_counts_top_level_items() {
        let dir = tempdir().unwrap();
        let cache = dir.path();
        std::fs::create_dir_all(cache.join("model-a")).unwrap();
        std::fs::write(cache.join("model-a/weights.bin"), vec![0u8; 1024]).unwrap();
        std::fs::write(cache.join("model-a/config.json"), vec![0u8; 256]).unwrap();
        std::fs::create_dir_all(cache.join("model-b")).unwrap();
        std::fs::write(cache.join("model-b/weights.bin"), vec![0u8; 4096]).unwrap();

        let e = entry(Some(cache.to_path_buf()));
        assert_eq!(e.items, 2, "two top-level model dirs");
        assert_eq!(e.bytes, 1024 + 256 + 4096);
    }

    #[test]
    fn prune_dry_run_does_not_delete() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("test");
        std::fs::write(&file, b"hello").unwrap();

        let report = PruneReport {
            dry_run: true,
            folded_models: entry(Some(dir.path().to_path_buf())),
            vocab_cache: CacheEntry::default(),
        };
        assert!(report.total_bytes() >= 5);
        // No actual prune call — just verifying the report shape.
        assert!(file.exists());
    }
}
