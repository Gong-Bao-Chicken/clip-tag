use std::path::{Path, PathBuf};

use clip_tag_model::{ModelConfig, SharedEngine, TagScore, TaggingEngine};
use image::DynamicImage;
use rayon::prelude::*;
use walkdir::WalkDir;

enum Prep {
    Skip(FileTagResult),
    Decoded(DynamicImage),
}

enum ChunkPrep {
    Ready(FileTagResult),
    Pending { path: PathBuf, image: DynamicImage },
}

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
                return batch_error_result(root, format!("discover_paths failed: {e}"));
            }
        };

        if paths.is_empty() {
            return batch_error_result(root, "no supported images found".to_string());
        }

        let batch_size = self.config.batch.batch_size.max(1);
        let continue_on_error = self.config.batch.continue_on_error;

        if batch_size <= 1 {
            return run_batch_paths(paths, continue_on_error, |path| {
                self.process_file(path, top_k, threshold)
            });
        }

        self.run_batch_chunked(paths, top_k, threshold, batch_size, continue_on_error)
    }

    /// Process paths in chunks, calling [`TaggingEngine::tag_batch`] once per
    /// chunk. Decoding and post-processing run in parallel within each chunk;
    /// the model call itself is one ORT invocation across the whole chunk so
    /// kernel-dispatch overhead is amortized.
    fn run_batch_chunked(
        &self,
        paths: Vec<PathBuf>,
        top_k: usize,
        threshold: f32,
        batch_size: usize,
        continue_on_error: bool,
    ) -> BatchResult {
        let fetch_k = tag_fetch_k(top_k, threshold);
        let mut files: Vec<FileTagResult> = Vec::with_capacity(paths.len());
        let mut had_error = false;

        for chunk in paths.chunks(batch_size) {
            let chunk_results = self.process_chunk(chunk, fetch_k, top_k, threshold);
            for r in chunk_results {
                if r.error.is_some() {
                    had_error = true;
                }
                files.push(r);
            }
            if had_error && !continue_on_error {
                break;
            }
        }

        let failed = files.iter().filter(|r| r.error.is_some()).count();
        let succeeded = files.len().saturating_sub(failed);
        BatchResult {
            files,
            succeeded,
            failed,
        }
    }

    fn process_chunk(
        &self,
        chunk: &[PathBuf],
        fetch_k: usize,
        top_k: usize,
        threshold: f32,
    ) -> Vec<FileTagResult> {
        // Phase 1: precheck + decode in parallel. Each slot is one of:
        //   - Ready: a fully-formed FileTagResult (skip or error).
        //   - Pending: a decoded image awaiting batched inference.
        let prepared: Vec<ChunkPrep> = chunk
            .par_iter()
            .map(|path| match self.prepare_for_inference(path) {
                Ok(Prep::Skip(r)) => ChunkPrep::Ready(r),
                Ok(Prep::Decoded(img)) => ChunkPrep::Pending {
                    path: path.clone(),
                    image: img,
                },
                Err(e) => ChunkPrep::Ready(FileTagResult {
                    path: path.display().to_string(),
                    tags: vec![],
                    write: None,
                    error: Some(e.to_string()),
                }),
            })
            .collect();

        // Phase 2: extract pending images for the single ORT call.
        let mut pending_paths: Vec<PathBuf> = Vec::new();
        let mut pending_images: Vec<DynamicImage> = Vec::new();
        let mut pending_slot_indices: Vec<usize> = Vec::new();
        let mut slots: Vec<Option<FileTagResult>> = Vec::with_capacity(prepared.len());

        for (slot_idx, item) in prepared.into_iter().enumerate() {
            match item {
                ChunkPrep::Ready(r) => slots.push(Some(r)),
                ChunkPrep::Pending { path, image } => {
                    slots.push(None);
                    pending_paths.push(path);
                    pending_images.push(image);
                    pending_slot_indices.push(slot_idx);
                }
            }
        }

        // Phase 3: one batched inference call.
        let tag_results: Vec<Result<Vec<TagScore>>> = if pending_images.is_empty() {
            Vec::new()
        } else {
            match self.engine.tag_batch(&pending_images, fetch_k) {
                Ok(per_image) => per_image.into_iter().map(Ok).collect(),
                Err(e) => {
                    // Surface the inference failure on every pending file in the chunk.
                    let msg = e.to_string();
                    pending_images.iter().map(|_| Err(Error::Model(msg.clone()))).collect()
                }
            }
        };
        drop(pending_images);

        // Phase 4: post-process (threshold filter + metadata write) in parallel.
        let triples: Vec<(PathBuf, Result<Vec<TagScore>>, usize)> = pending_paths
            .into_iter()
            .zip(tag_results.into_iter())
            .zip(pending_slot_indices.into_iter())
            .map(|((p, t), s)| (p, t, s))
            .collect();

        let post: Vec<(usize, FileTagResult)> = triples
            .into_par_iter()
            .map(|(path, tag_result, slot_idx)| {
                let result = match tag_result {
                    Ok(mut tags) => {
                        tags = filter_tags_by_threshold(tags, threshold);
                        tags.truncate(top_k);
                        match self.maybe_write_metadata(&path, &tags) {
                            Ok(write) => FileTagResult {
                                path: path.display().to_string(),
                                tags,
                                write,
                                error: None,
                            },
                            Err(e) => FileTagResult {
                                path: path.display().to_string(),
                                tags: vec![],
                                write: None,
                                error: Some(e.to_string()),
                            },
                        }
                    }
                    Err(e) => FileTagResult {
                        path: path.display().to_string(),
                        tags: vec![],
                        write: None,
                        error: Some(e.to_string()),
                    },
                };
                (slot_idx, result)
            })
            .collect();

        for (slot_idx, result) in post {
            slots[slot_idx] = Some(result);
        }

        slots
            .into_iter()
            .map(|s| s.expect("every slot populated"))
            .collect()
    }

    /// Per-file precheck + decode used by the chunked path.
    fn prepare_for_inference(&self, path: &Path) -> Result<Prep> {
        if let Some(skip) = self.precheck_skip_without_inference(path)? {
            return Ok(Prep::Skip(skip));
        }
        let img = clip_tag_image::load_dynamic_for_inference(
            path,
            clip_tag_image::DEFAULT_MAX_INFERENCE_DIM,
        )
        .map_err(|e| Error::Image(e.to_string()))?;
        Ok(Prep::Decoded(img))
    }

    fn process_file(&self, path: &Path, top_k: usize, threshold: f32) -> Result<FileTagResult> {
        if let Some(skip_result) = self.precheck_skip_without_inference(path)? {
            return Ok(skip_result);
        }
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

    fn precheck_skip_without_inference(&self, path: &Path) -> Result<Option<FileTagResult>> {
        let write_cfg = &self.config.write;
        if !write_cfg.enabled && !write_cfg.dry_run {
            return Ok(None);
        }
        if write_cfg.force || write_cfg.mode != WriteMode::EmptyOnly {
            return Ok(None);
        }

        let current = read_snapshot_external(path)?;
        let has_existing = crate::write::MetadataField::all()
            .iter()
            .any(|field| current.get(*field).is_some());
        if !has_existing {
            return Ok(None);
        }

        Ok(Some(FileTagResult {
            path: path.display().to_string(),
            tags: vec![],
            write: Some(FileWriteResult {
                decision: "skip: target keyword fields are already populated".into(),
                dry_run: None,
                applied: false,
            }),
            error: None,
        }))
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
    top_k
        .saturating_mul(OVERSAMPLE)
        .max(top_k.saturating_add(20))
}

pub(crate) fn filter_tags_by_threshold(tags: Vec<TagScore>, threshold: f32) -> Vec<TagScore> {
    if threshold <= 0.0 {
        return tags;
    }
    tags.into_iter().filter(|t| t.score >= threshold).collect()
}

fn batch_error_result(path: &Path, error: String) -> BatchResult {
    BatchResult {
        files: vec![FileTagResult {
            path: path.display().to_string(),
            tags: vec![],
            write: None,
            error: Some(error),
        }],
        succeeded: 0,
        failed: 1,
    }
}

fn run_batch_paths<F>(paths: Vec<PathBuf>, continue_on_error: bool, process_file: F) -> BatchResult
where
    F: Fn(&Path) -> Result<FileTagResult> + Sync,
{
    if continue_on_error {
        let files: Vec<FileTagResult> = paths
            .into_par_iter()
            .map(|path| match process_file(&path) {
                Ok(result) => result,
                Err(e) => FileTagResult {
                    path: path.display().to_string(),
                    tags: vec![],
                    write: None,
                    error: Some(e.to_string()),
                },
            })
            .collect();

        let failed = files.iter().filter(|r| r.error.is_some()).count();
        let succeeded = files.len().saturating_sub(failed);
        return BatchResult {
            files,
            succeeded,
            failed,
        };
    }

    let mut files = Vec::new();
    let mut succeeded = 0usize;
    let mut failed = 0usize;
    for path in paths {
        match process_file(&path) {
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
                break;
            }
        }
    }

    BatchResult {
        files,
        succeeded,
        failed,
    }
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
    use super::{filter_tags_by_threshold, run_batch_paths, tag_fetch_k, FileTagResult};
    use crate::Error;
    use clip_tag_model::TagScore;
    use std::path::{Path, PathBuf};

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

    #[test]
    fn run_batch_parallel_keeps_discovery_order_and_counts() {
        let paths = vec![
            PathBuf::from("a.jpg"),
            PathBuf::from("b.jpg"),
            PathBuf::from("c.jpg"),
            PathBuf::from("d.jpg"),
        ];
        let result = run_batch_paths(paths, true, |path| match path.to_string_lossy().as_ref() {
            "a.jpg" | "c.jpg" => Ok(FileTagResult {
                path: path.display().to_string(),
                tags: vec![],
                write: None,
                error: None,
            }),
            _ => Err(Error::Image("boom".into())),
        });

        let ordered_paths: Vec<&str> = result.files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(ordered_paths, vec!["a.jpg", "b.jpg", "c.jpg", "d.jpg"]);
        assert_eq!(result.succeeded, 2);
        assert_eq!(result.failed, 2);
    }

    #[test]
    fn run_batch_sequential_fail_fast_stops_after_first_error() {
        let paths = vec![
            PathBuf::from("first.jpg"),
            PathBuf::from("second.jpg"),
            PathBuf::from("third.jpg"),
        ];
        let result = run_batch_paths(paths, false, |path| {
            if path == Path::new("second.jpg") {
                Err(Error::Image("stop".into()))
            } else {
                Ok(FileTagResult {
                    path: path.display().to_string(),
                    tags: vec![],
                    write: None,
                    error: None,
                })
            }
        });

        let ordered_paths: Vec<&str> = result.files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(ordered_paths, vec!["first.jpg", "second.jpg"]);
        assert_eq!(result.succeeded, 1);
        assert_eq!(result.failed, 1);
    }
}

pub fn default_config_from_cli(
    write_metadata: bool,
    dry_run: bool,
    force: bool,
    write_mode: WriteMode,
    recursive: bool,
    model_id: Option<String>,
    provider: Option<String>,
    vocab_path: Option<PathBuf>,
    diversity_threshold: Option<f32>,
    batch_size: usize,
) -> Config {
    Config {
        write: WriteConfig {
            enabled: write_metadata,
            mode: if force {
                WriteMode::ForceOverwrite
            } else {
                write_mode
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
            batch_size: batch_size.max(1),
        },
    }
}
