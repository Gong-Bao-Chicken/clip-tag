use std::path::{Path, PathBuf};

use dirs::cache_dir;

use crate::{Error, Result};

/// Resolve vocabulary file: explicit path, env, repo default, then cache.
pub fn resolve_vocab_path(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(path.to_path_buf());
    }
    if let Ok(env) = std::env::var("CLIP_TAG_VOCAB") {
        return Ok(PathBuf::from(env));
    }
    if let Some(repo) = workspace_vocab_path() {
        return Ok(repo);
    }
    let cached = cache_dir()
        .ok_or_else(|| Error::Vocab("no cache dir".into()))?
        .join("clip-tag")
        .join("default_labels.txt");
    if cached.is_file() {
        return Ok(cached);
    }
    Err(Error::Vocab(
        "vocabulary not found; set --vocab or CLIP_TAG_VOCAB (see assets/vocab/)".into(),
    ))
}

fn workspace_vocab_path() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    for _ in 0..6 {
        let candidate = dir.join("assets/vocab/default_labels.txt");
        if candidate.is_file() {
            return Some(candidate);
        }
        if !dir.pop() {
            break;
        }
    }
    None
}

pub fn load_labels(path: &Path) -> Result<Vec<String>> {
    let raw = std::fs::read_to_string(path).map_err(|e| Error::Vocab(e.to_string()))?;
    let labels: Vec<String> = raw
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_string)
        .collect();
    if labels.is_empty() {
        return Err(Error::Vocab(format!("no labels in {}", path.display())));
    }
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_repo_default_vocab() {
        let path = workspace_vocab_path().expect("repo vocab");
        let labels = load_labels(&path).unwrap();
        assert!(labels.len() > 1000);
    }
}
