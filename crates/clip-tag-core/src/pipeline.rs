use std::path::{Path, PathBuf};

use clip_tag_model::{ModelConfig, SharedEngine, TagScore, TaggingEngine};
use walkdir::WalkDir;

use crate::config::{BatchConfig, Config, WriteConfig};
use crate::dry_run::DryRunReport;
use crate::policy::{plan_write, WriteDecision, WriteMode, WritePolicyInput};
use crate::provider::{TagProvider, TagRequest, TagResponse};
use crate::write::FieldSnapshot;
use crate::{Error, Result};

pub struct Pipeline {
    engine: SharedEngine,
    config: Config,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct FileTagResult {
    pub path: String,
    pub tags: Vec<TagScore>,
    pub write: Option<FileWriteResult>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct FileWriteResult {
    pub decision: String,
    pub dry_run: Option<DryRunReport>,
    pub applied: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BatchResult {
    pub files: Vec<FileTagResult>,
    pub succeeded: usize,
    pub failed: usize,
}

impl Pipeline {
    pub fn new(engine: SharedEngine, config: Config) -> Self {
        Self { engine, config }
    }

    pub fn from_model(config: Config) -> Result<Self> {
        let model_config = ModelConfig {
            model_id: config
                .model
                .model_id
                .clone()
                .unwrap_or_else(|| clip_tag_model::DEFAULT_MODEL_ID.to_string()),
            vocab_path: config.model.vocab_path.clone(),
            provider: clip_tag_model::ExecutionProvider::parse(config.model.provider.as_deref())?,
            diversity_threshold: config.model.diversity_threshold,
        };
        let engine = clip_tag_model::load_shared(model_config)?;
        Ok(Self::new(engine, config))
    }

    pub fn engine(&self) -> &TaggingEngine {
        &self.engine
    }

    pub fn discover_paths(root: &Path, recursive: bool) -> Result<Vec<PathBuf>> {
        if root.is_file() {
            return Ok(vec![root.to_path_buf()]);
        }
        if !root.is_dir() {
            return Err(Error::Image(format!("not found: {}", root.display())));
        }

        let mut paths = Vec::new();
        if recursive {
            for entry in WalkDir::new(root)
                .follow_links(false)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                let path = entry.path();
                if path.is_file() && clip_tag_image::ImageFormat::from_path(path).is_some() {
                    paths.push(path.to_path_buf());
                }
            }
        } else if let Ok(read) = std::fs::read_dir(root) {
            for entry in read.flatten() {
                let path = entry.path();
                if path.is_file() && clip_tag_image::ImageFormat::from_path(&path).is_some() {
                    paths.push(path);
                }
            }
        }

        paths.sort();
        Ok(paths)
    }

    pub fn run_batch(&self, root: &Path, top_k: usize, threshold: f32) -> BatchResult {
        let paths = match Self::discover_paths(root, self.config.batch.recursive) {
            Ok(p) => p,
            Err(e) => {
                return BatchResult {
                    files: vec![FileTagResult {
                        path: root.display().to_string(),
                        tags: vec![],
                        write: None,
                        error: Some(e.to_string()),
                    }],
                    succeeded: 0,
                    failed: 1,
                };
            }
        };

        if paths.is_empty() {
            return BatchResult {
                files: vec![FileTagResult {
                    path: root.display().to_string(),
                    tags: vec![],
                    write: None,
                    error: Some("no supported images found".into()),
                }],
                succeeded: 0,
                failed: 1,
            };
        }

        let continue_on_error = self.config.batch.continue_on_error;
        let mut files = Vec::new();
        let mut succeeded = 0usize;
        let mut failed = 0usize;

        for path in paths {
            match self.process_file(&path, top_k, threshold) {
                Ok(result) => {
                    succeeded += 1;
                    files.push(result);
                }
                Err(e) => {
                    failed += 1;
                    files.push(FileTagResult {
                        path: path.display().to_string(),
                        tags: vec![],
                        write: None,
                        error: Some(e.to_string()),
                    });
                    if !continue_on_error {
                        break;
                    }
                }
            }
        }

        BatchResult {
            files,
            succeeded,
            failed,
        }
    }

    fn process_file(&self, path: &Path, top_k: usize, threshold: f32) -> Result<FileTagResult> {
        let fetch_k = tag_fetch_k(top_k, threshold);
        let mut tags = self.engine.tag_path(path, fetch_k)?;
        tags = filter_tags_by_threshold(tags, threshold);
        tags.truncate(top_k);
        let write = self.maybe_write_metadata(path, &tags)?;
        Ok(FileTagResult {
            path: path.display().to_string(),
            tags,
            write,
            error: None,
        })
    }

    fn maybe_write_metadata(
        &self,
        path: &Path,
        tags: &[TagScore],
    ) -> Result<Option<FileWriteResult>> {
        let write_cfg = &self.config.write;
        if !write_cfg.enabled && !write_cfg.dry_run {
            return Ok(None);
        }

        let proposed: Vec<String> = tags.iter().map(|t| t.label.clone()).collect();
        let current = read_snapshot_external(path)?;
        let mode = if write_cfg.force {
            WriteMode::ForceOverwrite
        } else {
            write_cfg.mode
        };

        let decision = plan_write(WritePolicyInput {
            mode,
            force: write_cfg.force,
            proposed_tags: proposed,
            current,
        })?;

        let dry_run_report = crate::dry_run::report_for_path(path.display().to_string(), &decision);
        let decision_str = match &decision {
            WriteDecision::Skip { reason } => format!("skip: {reason}"),
            WriteDecision::Apply(_) => {
                if write_cfg.dry_run {
                    "would_write".into()
                } else {
                    "write".into()
                }
            }
        };

        let mut applied = false;
        if !write_cfg.dry_run {
            if let WriteDecision::Apply(plan) = &decision {
                write_plan_external(path, plan, false)?;
                applied = true;
            }
        }

        Ok(Some(FileWriteResult {
            decision: decision_str,
            dry_run: if write_cfg.dry_run {
                Some(dry_run_report)
            } else {
                None
            },
            applied,
        }))
    }
}

/// Ranked tags to request from the engine before score-threshold filtering.
pub(crate) fn tag_fetch_k(top_k: usize, threshold: f32) -> usize {
    if threshold <= 0.0 {
        return top_k;
    }
    const OVERSAMPLE: usize = 5;
    top_k.saturating_mul(OVERSAMPLE).max(top_k.saturating_add(20))
}

pub(crate) fn filter_tags_by_threshold(tags: Vec<TagScore>, threshold: f32) -> Vec<TagScore> {
    if threshold <= 0.0 {
        return tags;
    }
    tags.into_iter().filter(|t| t.score >= threshold).collect()
}

/// Metadata I/O is injected from CLI to avoid core ↔ metadata cycle.
static METADATA_IO: std::sync::OnceLock<MetadataIoHooks> = std::sync::OnceLock::new();

pub struct MetadataIoHooks {
    pub read_snapshot: fn(&Path) -> Result<FieldSnapshot>,
    pub write_plan: fn(&Path, &crate::write::WritePlan, bool) -> Result<()>,
}

pub fn set_metadata_hooks(hooks: MetadataIoHooks) {
    let _ = METADATA_IO.set(hooks);
}

fn read_snapshot_external(path: &Path) -> Result<FieldSnapshot> {
    let hooks = METADATA_IO
        .get()
        .ok_or_else(|| Error::Metadata("metadata hooks not installed".into()))?;
    (hooks.read_snapshot)(path)
}

fn write_plan_external(path: &Path, plan: &crate::write::WritePlan, dry_run: bool) -> Result<()> {
    let hooks = METADATA_IO
        .get()
        .ok_or_else(|| Error::Metadata("metadata hooks not installed".into()))?;
    (hooks.write_plan)(path, plan, dry_run)
}

/// Adapter implementing [`TagProvider`] over the shared engine.
pub struct LocalOnnxProvider {
    engine: SharedEngine,
}

impl LocalOnnxProvider {
    pub fn new(engine: SharedEngine) -> Self {
        Self { engine }
    }
}

impl TagProvider for LocalOnnxProvider {
    fn id(&self) -> &'static str {
        "local-onnx"
    }

    fn tag_image(&self, request: &TagRequest) -> Result<TagResponse> {
        let scores = self
            .engine
            .tag_path(Path::new(&request.image_path), request.top_k)?;
        Ok(TagResponse {
            tags: scores
                .into_iter()
                .map(|s| crate::provider::TagScore {
                    label: s.label,
                    score: s.score,
                })
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{filter_tags_by_threshold, tag_fetch_k};
    use clip_tag_model::TagScore;

    #[test]
    fn threshold_filter_is_applied() {
        let tags = vec![
            TagScore {
                label: "a".into(),
                score: 0.01,
            },
            TagScore {
                label: "b".into(),
                score: 0.50,
            },
        ];
        let out = filter_tags_by_threshold(tags, 0.1);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].label, "b");
    }

    #[test]
    fn tag_fetch_k_oversamples_when_threshold_active() {
        assert_eq!(tag_fetch_k(10, 0.02), 50);
        assert_eq!(tag_fetch_k(3, 0.1), 23);
    }

    #[test]
    fn tag_fetch_k_matches_top_k_without_threshold() {
        assert_eq!(tag_fetch_k(10, 0.0), 10);
        assert_eq!(tag_fetch_k(10, -0.1), 10);
    }

    #[test]
    fn zero_threshold_keeps_all() {
        let tags = vec![
            TagScore {
                label: "a".into(),
                score: 0.0,
            },
            TagScore {
                label: "b".into(),
                score: 1.0,
            },
        ];
        let out = filter_tags_by_threshold(tags, 0.0);
        assert_eq!(out.len(), 2);
    }
}

pub fn default_config_from_cli(
    write_metadata: bool,
    dry_run: bool,
    force: bool,
    recursive: bool,
    model_id: Option<String>,
    provider: Option<String>,
    vocab_path: Option<PathBuf>,
    diversity_threshold: Option<f32>,
) -> Config {
    Config {
        write: WriteConfig {
            enabled: write_metadata,
            mode: if force {
                WriteMode::ForceOverwrite
            } else {
                WriteMode::EmptyOnly
            },
            dry_run,
            force,
        },
        model: crate::config::ModelConfig {
            model_id,
            model_path: None,
            provider,
            vocab_path,
            diversity_threshold,
        },
        batch: BatchConfig {
            recursive,
            continue_on_error: true,
        },
    }
}
