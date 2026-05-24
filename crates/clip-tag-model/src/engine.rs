use std::path::{Path, PathBuf};
use std::sync::Arc;

use hf_hub::api::tokio::Api;
use image::DynamicImage;
use ndarray::{Array2, ArrayView1};
use open_clip_inference::clip::Clip;
use ort::ep::{ExecutionProviderDispatch, CPU};

use crate::vocab;
use crate::vocab_cache;
use crate::{Error, Result, DEFAULT_MODEL_ID};

/// Soft cap to keep top-K extraction near linear when the vocabulary grows.
/// We partial-sort this many top-scoring labels, then full-sort just those for
/// the diversity selector to walk through.
const RANKING_CANDIDATE_LIMIT_PER_TOP_K: usize = 8;
const MIN_RANKING_CANDIDATES: usize = 64;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TagScore {
    pub label: String,
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_id: String,
    pub vocab_path: Option<std::path::PathBuf>,
    pub provider: ExecutionProvider,
    pub diversity_threshold: Option<f32>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_id: DEFAULT_MODEL_ID.to_string(),
            vocab_path: None,
            provider: ExecutionProvider::Auto,
            diversity_threshold: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProvider {
    Auto,
    Cpu,
    Metal,
    Coreml,
    Directml,
}

impl ExecutionProvider {
    pub fn parse(raw: Option<&str>) -> Result<Self> {
        match raw.unwrap_or("auto") {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "coreml" => Ok(Self::Coreml),
            "directml" => Ok(Self::Directml),
            other => Err(Error::Config(format!(
                "unsupported provider `{other}`; expected one of: auto, cpu, metal, coreml, directml"
            ))),
        }
    }
}

pub struct TaggingEngine {
    clip: Clip,
    labels: Vec<String>,
    text_embeddings: Array2<f32>,
    diversity_threshold: f32,
}

const DIVERSITY_SIMILARITY_THRESHOLD: f32 = 0.8;

impl TaggingEngine {
    pub async fn load(config: ModelConfig) -> Result<Self> {
        let model_id = config.model_id.as_str();
        let provider = config.provider;
        let diversity_threshold = config
            .diversity_threshold
            .unwrap_or(DIVERSITY_SIMILARITY_THRESHOLD);
        if !(0.0..=1.0).contains(&diversity_threshold) {
            return Err(Error::Config(format!(
                "diversity threshold must be in [0.0, 1.0], got {diversity_threshold}"
            )));
        }
        tracing::info!(
            model_id,
            provider = ?provider,
            diversity_threshold,
            "loading CLIP model"
        );

        let clip = load_clip_with_provider(model_id, provider).await?;

        let vocab_path = vocab::resolve_vocab_path(config.vocab_path.as_deref())?;
        let labels = vocab::load_labels(&vocab_path)?;
        tracing::info!(count = labels.len(), path = %vocab_path.display(), "loaded vocabulary");

        let cache_path = vocab_cache::cache_path_for(&vocab_path, model_id);
        let mut text_embeddings = if cache_path.is_file() {
            tracing::info!(path = %cache_path.display(), "loading vocab embedding cache");
            match vocab_cache::load_cache(&cache_path, labels.len()) {
                Ok(cached) => cached,
                Err(e) => {
                    tracing::warn!("vocab cache invalid ({e}); recomputing");
                    let embs = vocab_cache::embed_labels(&clip.text, &labels)?;
                    vocab_cache::save_cache(&cache_path, &embs)?;
                    embs
                }
            }
        } else {
            tracing::info!("precomputing vocabulary embeddings (first run)");
            let embs = vocab_cache::embed_labels(&clip.text, &labels)?;
            vocab_cache::save_cache(&cache_path, &embs)?;
            embs
        };

        l2_normalize_rows(&mut text_embeddings);

        Ok(Self {
            clip,
            labels,
            text_embeddings,
            diversity_threshold,
        })
    }

    pub fn tag_rgb8(&self, rgb: &image::RgbImage, top_k: usize) -> Result<Vec<TagScore>> {
        let dynamic = DynamicImage::ImageRgb8(rgb.clone());
        self.tag_dynamic(&dynamic, top_k)
    }

    pub fn tag_path(&self, path: &Path, top_k: usize) -> Result<Vec<TagScore>> {
        let rgb = clip_tag_image::load_rgb8_for_inference(
            path,
            clip_tag_image::DEFAULT_MAX_INFERENCE_DIM,
        )
        .map_err(|e| Error::Inference(e.to_string()))?;
        self.tag_rgb8(&rgb, top_k)
    }

    fn tag_dynamic(&self, image: &DynamicImage, top_k: usize) -> Result<Vec<TagScore>> {
        let img_emb = self
            .clip
            .vision
            .embed_image(image)
            .map_err(|e| Error::Inference(e.to_string()))?;

        let similarities = self.text_embeddings.dot(&img_emb);
        let scale = self.clip.text.model_config.logit_scale.unwrap_or(1.0);
        let bias = self.clip.text.model_config.logit_bias.unwrap_or(0.0);

        let logits: Vec<f32> = similarities
            .iter()
            .map(|&s| s.mul_add(scale, bias))
            .collect();
        let probs = Clip::softmax(&logits);

        let ranked = partial_rank(&probs, top_k);

        let selected = select_diverse_top_k(
            &ranked,
            &self.text_embeddings,
            top_k,
            self.diversity_threshold,
        );
        Ok(selected
            .into_iter()
            .map(|(idx, score)| TagScore {
                label: self.labels[idx].clone(),
                score,
            })
            .collect())
    }
}

fn select_diverse_top_k(
    ranked: &[(usize, f32)],
    text_embeddings: &Array2<f32>,
    top_k: usize,
    max_similarity: f32,
) -> Vec<(usize, f32)> {
    if top_k == 0 {
        return Vec::new();
    }

    let k = top_k.min(ranked.len());
    let mut selected: Vec<(usize, f32)> = Vec::with_capacity(k);

    for &(idx, score) in ranked {
        if selected.len() >= k {
            break;
        }
        let is_too_similar = selected.iter().any(|&(selected_idx, _)| {
            cosine_of_unit_rows(text_embeddings.row(idx), text_embeddings.row(selected_idx))
                >= max_similarity
        });
        if !is_too_similar {
            selected.push((idx, score));
        }
    }

    if selected.len() < k {
        for &(idx, score) in ranked {
            if selected.len() >= k {
                break;
            }
            if selected
                .iter()
                .any(|&(selected_idx, _)| selected_idx == idx)
            {
                continue;
            }
            selected.push((idx, score));
        }
    }

    selected
}

/// Cosine similarity for rows that were already L2-normalized at load time.
/// Zero-norm rows (a degenerate vocab embedding) produce a clamped 0.
fn cosine_of_unit_rows(a: ArrayView1<'_, f32>, b: ArrayView1<'_, f32>) -> f32 {
    a.dot(&b).clamp(-1.0, 1.0)
}

fn l2_normalize_rows(matrix: &mut Array2<f32>) {
    for mut row in matrix.outer_iter_mut() {
        let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            row.mapv_inplace(|v| v / norm);
        }
    }
}

/// Return the top `select_diverse_top_k` candidate window as `(idx, score)`
/// pairs sorted by score descending (then index ascending for determinism).
///
/// Uses [`select_nth_unstable_by`] so the heavy sort only touches the small
/// candidate window even when the vocabulary is thousands of labels wide.
fn partial_rank(probs: &[f32], top_k: usize) -> Vec<(usize, f32)> {
    let n = probs.len();
    if n == 0 {
        return Vec::new();
    }
    let limit = top_k
        .saturating_mul(RANKING_CANDIDATE_LIMIT_PER_TOP_K)
        .max(MIN_RANKING_CANDIDATES)
        .min(n);

    let cmp = |a: &(usize, f32), b: &(usize, f32)| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    };

    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    if limit < n {
        indexed.select_nth_unstable_by(limit - 1, cmp);
        indexed.truncate(limit);
    }
    indexed.sort_by(cmp);
    indexed
}

async fn load_clip_with_provider(model_id: &str, provider: ExecutionProvider) -> Result<Clip> {
    match provider {
        ExecutionProvider::Cpu | ExecutionProvider::Auto => {
            load_clip(model_id, &cpu_execution_providers()).await
        }
        ExecutionProvider::Metal => {
            tracing::warn!(
                "provider `metal` requested but not yet implemented; falling back to CPU"
            );
            load_clip(model_id, &cpu_execution_providers()).await
        }
        ExecutionProvider::Coreml => {
            tracing::warn!(
                "provider `coreml` requested but not yet implemented; falling back to CPU"
            );
            load_clip(model_id, &cpu_execution_providers()).await
        }
        ExecutionProvider::Directml => {
            tracing::warn!(
                "provider `directml` requested but not yet implemented; falling back to CPU"
            );
            load_clip(model_id, &cpu_execution_providers()).await
        }
    }
}

async fn load_clip(model_id: &str, providers: &[ExecutionProviderDispatch]) -> Result<Clip> {
    let model_dir = resolve_model_dir(model_id).await?;
    Clip::from_local_dir(&model_dir)
        .with_execution_providers(providers)
        .build()
        .map_err(|e| Error::Load(e.to_string()))
}

const REQUIRED_MODEL_FILES: &[&str] = &[
    "model_config.json",
    "open_clip_config.json",
    "special_tokens_map.json",
    "text.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "visual.onnx",
];

const OPTIONAL_DATA_SIDECARS: &[&str] = &["text.onnx.data", "visual.onnx.data"];

const MODEL_FILE_ALIASES: &[(&str, &[&str])] = &[
    ("model_config.json", &["config.json"]),
    ("text.onnx", &["text_model.onnx", "textual.onnx"]),
    ("visual.onnx", &["vision_model.onnx", "clip_visual.onnx"]),
];

async fn resolve_model_dir(model_id: &str) -> Result<PathBuf> {
    let model_path = Path::new(model_id);
    if model_path.is_dir() {
        normalize_model_config(model_path)?;
        ensure_optional_sidecars(model_path)?;
        return Ok(model_path.to_path_buf());
    }

    let api = Api::new().map_err(|e| Error::Load(e.to_string()))?;
    let repo = api.model(model_id.to_string());

    let mut model_dir: Option<PathBuf> = None;
    for file in REQUIRED_MODEL_FILES {
        let downloaded = download_required_or_alias(&repo, model_id, file).await?;
        if model_dir.is_none() {
            model_dir = downloaded.parent().map(ToOwned::to_owned);
        }
    }

    let model_dir = model_dir.ok_or_else(|| {
        Error::Load(format!(
            "could not resolve downloaded model directory for `{model_id}`"
        ))
    })?;

    for file in OPTIONAL_DATA_SIDECARS {
        if let Err(err) = repo.get(file).await {
            let msg = err.to_string();
            if msg.contains("404") || msg.contains("Not Found") {
                tracing::debug!(model_id, file, "optional ONNX sidecar not present");
                continue;
            }
            return Err(Error::Load(format!(
                "failed downloading optional model file `{file}` for `{model_id}`: {err}"
            )));
        }
    }

    normalize_model_config(&model_dir)?;
    ensure_optional_sidecars(&model_dir)?;
    Ok(model_dir)
}

async fn download_required_or_alias(
    repo: &hf_hub::api::tokio::ApiRepo,
    model_id: &str,
    required: &str,
) -> Result<PathBuf> {
    match repo.get(required).await {
        Ok(path) => Ok(path),
        Err(primary_err) => {
            let aliases = MODEL_FILE_ALIASES
                .iter()
                .find_map(|(name, aliases)| (*name == required).then_some(*aliases))
                .unwrap_or(&[]);

            for alias in aliases {
                if let Ok(alias_path) = repo.get(alias).await {
                    let target = alias_path
                        .parent()
                        .map(|parent| parent.join(required))
                        .ok_or_else(|| {
                            Error::Load(format!(
                                "downloaded alias `{alias}` for `{model_id}` had no parent directory"
                            ))
                        })?;

                    std::fs::copy(&alias_path, &target).map_err(|e| {
                        Error::Load(format!(
                            "failed normalizing alias `{alias}` to `{required}` for `{model_id}`: {e}"
                        ))
                    })?;
                    return Ok(target);
                }
            }

            Err(Error::Load(format!(
                "failed downloading required model file `{required}` for `{model_id}`: {primary_err}"
            )))
        }
    }
}

fn ensure_optional_sidecars(model_dir: &Path) -> Result<()> {
    for file in OPTIONAL_DATA_SIDECARS {
        let sidecar = model_dir.join(file);
        if !sidecar.exists() {
            // Some ONNX runtimes/producers expect a .onnx.data companion path even when
            // weights are embedded; create an empty file as a compatibility workaround.
            std::fs::write(&sidecar, []).map_err(|e| {
                Error::Load(format!(
                    "failed creating compatibility sidecar `{}`: {e}",
                    sidecar.display()
                ))
            })?;
        }
    }
    Ok(())
}

fn normalize_model_config(model_dir: &Path) -> Result<()> {
    let path = model_dir.join("model_config.json");
    let content = std::fs::read_to_string(&path).map_err(|e| {
        Error::Load(format!(
            "failed reading model config `{}`: {e}",
            path.display()
        ))
    })?;
    let mut json: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
        Error::Load(format!(
            "failed parsing model config `{}`: {e}",
            path.display()
        ))
    })?;

    let mut changed = false;
    if json.get("pad_id").is_none() {
        if let Some(pad_id) = json
            .pointer("/pad_token_id")
            .and_then(|v| v.as_u64())
            .or_else(|| {
                json.pointer("/text_config/pad_token_id")
                    .and_then(|v| v.as_u64())
            })
        {
            json["pad_id"] = serde_json::Value::from(pad_id);
            changed = true;
        }
    }

    if json.get("logit_scale").is_none() {
        if let Some(logit_scale) = json
            .pointer("/logit_scale_init_value")
            .and_then(|v| v.as_f64())
        {
            json["logit_scale"] = serde_json::Value::from(logit_scale);
            changed = true;
        }
    }

    if json.get("tokenizer_needs_lowercase").is_none() {
        let tokenizer_config_path = model_dir.join("tokenizer_config.json");
        if let Ok(tokenizer_config) = std::fs::read_to_string(&tokenizer_config_path) {
            if let Ok(tokenizer_json) = serde_json::from_str::<serde_json::Value>(&tokenizer_config)
            {
                if let Some(do_lower_case) = tokenizer_json
                    .get("do_lower_case")
                    .and_then(|v| v.as_bool())
                {
                    json["tokenizer_needs_lowercase"] = serde_json::Value::from(do_lower_case);
                    changed = true;
                }
            }
        }
    }

    if changed {
        let normalized = serde_json::to_string_pretty(&json)
            .map_err(|e| Error::Load(format!("failed serializing normalized model config: {e}")))?;
        std::fs::write(&path, normalized).map_err(|e| {
            Error::Load(format!(
                "failed writing normalized model config `{}`: {e}",
                path.display()
            ))
        })?;
    }
    Ok(())
}

fn cpu_execution_providers() -> Vec<ExecutionProviderDispatch> {
    vec![CPU::default().build().error_on_failure()]
}

pub type SharedEngine = Arc<TaggingEngine>;

pub fn load_shared(config: ModelConfig) -> Result<SharedEngine> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| Error::Load(e.to_string()))?
        .block_on(async { TaggingEngine::load(config).await.map(Arc::new) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use tempfile::tempdir;

    #[test]
    fn provider_parse_supports_known_values() {
        assert_eq!(
            ExecutionProvider::parse(None).unwrap(),
            ExecutionProvider::Auto
        );
        assert_eq!(
            ExecutionProvider::parse(Some("cpu")).unwrap(),
            ExecutionProvider::Cpu
        );
        assert_eq!(
            ExecutionProvider::parse(Some("metal")).unwrap(),
            ExecutionProvider::Metal
        );
        assert_eq!(
            ExecutionProvider::parse(Some("coreml")).unwrap(),
            ExecutionProvider::Coreml
        );
        assert_eq!(
            ExecutionProvider::parse(Some("directml")).unwrap(),
            ExecutionProvider::Directml
        );
    }

    #[test]
    fn provider_parse_rejects_unknown_values() {
        let err = ExecutionProvider::parse(Some("gpu")).unwrap_err();
        assert!(err.to_string().contains("unsupported provider"));
    }

    #[test]
    fn creates_missing_optional_sidecars() {
        let dir = tempdir().unwrap();
        let model_dir = dir.path();

        ensure_optional_sidecars(model_dir).unwrap();

        assert!(model_dir.join("text.onnx.data").is_file());
        assert!(model_dir.join("visual.onnx.data").is_file());
    }

    #[test]
    fn normalizes_transformers_style_model_config() {
        let dir = tempdir().unwrap();
        let model_dir = dir.path();
        std::fs::write(
            model_dir.join("model_config.json"),
            r#"{"text_config":{"pad_token_id":1},"logit_scale_init_value":2.6592}"#,
        )
        .unwrap();
        std::fs::write(
            model_dir.join("tokenizer_config.json"),
            r#"{"do_lower_case":true}"#,
        )
        .unwrap();

        normalize_model_config(model_dir).unwrap();
        let normalized: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(model_dir.join("model_config.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(normalized.get("pad_id").and_then(|v| v.as_u64()), Some(1));
        assert_eq!(
            normalized
                .get("tokenizer_needs_lowercase")
                .and_then(|v| v.as_bool()),
            Some(true)
        );
        assert!(normalized.get("logit_scale").is_some());
    }

    #[test]
    #[ignore = "requires model download"]
    fn tags_fixture_deterministically() {
        let engine = load_shared(ModelConfig::default()).unwrap();
        let rgb = image::RgbImage::from_fn(224, 224, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
        let a = engine.tag_rgb8(&rgb, 5).unwrap();
        let b = engine.tag_rgb8(&rgb, 5).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn select_diverse_top_k_skips_near_duplicates() {
        let ranked = vec![(0, 0.90), (1, 0.89), (2, 0.50)];
        // Rows are pre-normalized (unit vectors), as TaggingEngine normalizes at load.
        let mut embeddings = array![
            [1.0, 0.0],
            [0.9999, 0.0001], // near-duplicate of row 0
            [0.0, 1.0]
        ];
        l2_normalize_rows(&mut embeddings);

        let out = select_diverse_top_k(&ranked, &embeddings, 2, 0.95);
        assert_eq!(out, vec![(0, 0.90), (2, 0.50)]);
    }

    #[test]
    fn select_diverse_top_k_falls_back_to_fill_requested_k() {
        let ranked = vec![(0, 0.90), (1, 0.89), (2, 0.88)];
        let mut embeddings = array![[1.0, 0.0], [0.9999, 0.0001], [0.9998, 0.0002]];
        l2_normalize_rows(&mut embeddings);

        let out = select_diverse_top_k(&ranked, &embeddings, 2, 0.95);
        assert_eq!(out, vec![(0, 0.90), (1, 0.89)]);
    }

    #[test]
    fn select_diverse_top_k_handles_zero_norm_rows() {
        let ranked = vec![(0, 0.90), (1, 0.89), (2, 0.88)];
        let mut embeddings = array![
            [0.0, 0.0], // zero vector stays zero after normalization → cosine == 0
            [0.0, 0.0],
            [1.0, 0.0]
        ];
        l2_normalize_rows(&mut embeddings);

        let out = select_diverse_top_k(&ranked, &embeddings, 2, 0.95);
        assert_eq!(out, vec![(0, 0.90), (1, 0.89)]);
    }

    #[test]
    fn partial_rank_returns_top_scores_sorted() {
        let probs = vec![0.1, 0.9, 0.4, 0.7, 0.2, 0.5];
        let ranked = partial_rank(&probs, 3);
        let scores: Vec<f32> = ranked.iter().map(|(_, s)| *s).collect();
        assert_eq!(scores[0], 0.9);
        assert_eq!(scores[1], 0.7);
        assert_eq!(scores[2], 0.5);
        // limit clamps to MIN_RANKING_CANDIDATES or smaller when n < limit.
        assert_eq!(ranked.len(), probs.len());
    }

    #[test]
    fn partial_rank_breaks_ties_by_index_ascending() {
        let probs = vec![0.5, 0.5, 0.5];
        let ranked = partial_rank(&probs, 2);
        assert_eq!(ranked[0].0, 0);
        assert_eq!(ranked[1].0, 1);
    }

    #[test]
    fn l2_normalize_rows_makes_unit_length() {
        let mut m = array![[3.0_f32, 4.0], [1.0, 0.0]];
        l2_normalize_rows(&mut m);
        let n0 = (m[[0, 0]].powi(2) + m[[0, 1]].powi(2)).sqrt();
        let n1 = (m[[1, 0]].powi(2) + m[[1, 1]].powi(2)).sqrt();
        assert!((n0 - 1.0).abs() < 1e-6);
        assert!((n1 - 1.0).abs() < 1e-6);
    }
}
