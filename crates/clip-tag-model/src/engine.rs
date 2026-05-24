use std::path::{Path, PathBuf};
use std::sync::Arc;

use hf_hub::api::tokio::Api;
use image::DynamicImage;
use ndarray::{Array2, ArrayView1};
use open_clip_inference::clip::Clip;
use ort::ep::{ExecutionProviderDispatch, CPU};

use crate::external_data;
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
    Cuda,
}

impl ExecutionProvider {
    pub fn parse(raw: Option<&str>) -> Result<Self> {
        match raw.unwrap_or("auto") {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "coreml" => Ok(Self::Coreml),
            "directml" => Ok(Self::Directml),
            "cuda" => Ok(Self::Cuda),
            other => Err(Error::Config(format!(
                "unsupported provider `{other}`; expected one of: auto, cpu, metal, coreml, directml, cuda"
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
        let mut batch = self.tag_batch(std::slice::from_ref(image), top_k)?;
        Ok(batch.pop().unwrap_or_default())
    }

    /// Score a batch of images against the vocabulary in a single ORT call.
    ///
    /// The ORT session uses a write-lock (`RwLock::write`) internally; batching
    /// is the only way to amortize that lock and the per-call dispatch overhead
    /// — without it, parallel `tag_path` calls serialize at the session.
    ///
    /// Returns one ranked `Vec<TagScore>` per input image, in input order. An
    /// empty `images` slice returns an empty vec without touching the model.
    pub fn tag_batch(
        &self,
        images: &[DynamicImage],
        top_k: usize,
    ) -> Result<Vec<Vec<TagScore>>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        let mut image_embs = self
            .clip
            .vision
            .embed_images(images)
            .map_err(|e| Error::Inference(e.to_string()))?;

        // Normalize image embeddings so the matrix multiply below is cosine
        // similarity (text embeddings were already normalized at load).
        l2_normalize_rows(&mut image_embs);

        let scale = self.clip.text.model_config.logit_scale.unwrap_or(1.0);
        let bias = self.clip.text.model_config.logit_bias.unwrap_or(0.0);

        // Shape: (vocab, batch). One column per image.
        let similarities = self.text_embeddings.dot(&image_embs.t());

        let n = images.len();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let col = similarities.column(i);
            let logits: Vec<f32> = col.iter().map(|&s| s.mul_add(scale, bias)).collect();
            let probs = Clip::softmax(&logits);
            let ranked = partial_rank(&probs, top_k);
            let selected = select_diverse_top_k(
                &ranked,
                &self.text_embeddings,
                top_k,
                self.diversity_threshold,
            );
            out.push(
                selected
                    .into_iter()
                    .map(|(idx, score)| TagScore {
                        label: self.labels[idx].clone(),
                        score,
                    })
                    .collect(),
            );
        }
        Ok(out)
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
    let providers = build_execution_providers(provider)?;
    let requires_internal_data = provider_requires_internal_data(provider);
    load_clip(model_id, &providers, requires_internal_data).await
}

/// Build the execution-provider chain for a requested provider.
///
/// Accelerator providers are stacked first with `fail_silently` (the default),
/// so ORT routes unsupported ops down to the CPU provider that always tails
/// the chain. `error_on_failure()` is applied to the CPU tail so a true
/// catastrophe (no EP registered at all) surfaces as a proper error.
fn build_execution_providers(
    provider: ExecutionProvider,
) -> Result<Vec<ExecutionProviderDispatch>> {
    let chain = match provider {
        ExecutionProvider::Cpu => cpu_execution_providers(),
        ExecutionProvider::Auto => auto_providers_for_target(),
        ExecutionProvider::Coreml => coreml_chain()?,
        ExecutionProvider::Directml => directml_chain()?,
        ExecutionProvider::Cuda => cuda_chain()?,
        ExecutionProvider::Metal => {
            // `metal` is reserved for future use; on macOS today CoreML routes
            // to ANE/GPU including Metal kernels, so we treat it as an alias
            // and let it fall through the same chain.
            #[cfg(feature = "coreml")]
            {
                coreml_chain()?
            }
            #[cfg(not(feature = "coreml"))]
            {
                tracing::warn!(
                    "provider `metal` requires the `coreml` build feature; using CPU"
                );
                cpu_execution_providers()
            }
        }
    };
    Ok(chain)
}

fn auto_providers_for_target() -> Vec<ExecutionProviderDispatch> {
    let mut chain: Vec<ExecutionProviderDispatch> = Vec::new();

    #[cfg(all(feature = "coreml", target_os = "macos"))]
    {
        chain.push(
            ort::ep::CoreML::default()
                .with_compute_units(ort::ep::coreml::ComputeUnits::All)
                .build(),
        );
    }

    #[cfg(all(feature = "directml", target_os = "windows"))]
    {
        chain.push(ort::ep::DirectML::default().build());
    }

    #[cfg(feature = "cuda")]
    {
        chain.push(ort::ep::CUDA::default().build());
    }

    chain.push(CPU::default().build().error_on_failure());
    chain
}

fn coreml_chain() -> Result<Vec<ExecutionProviderDispatch>> {
    #[cfg(feature = "coreml")]
    {
        Ok(vec![
            ort::ep::CoreML::default()
                .with_compute_units(ort::ep::coreml::ComputeUnits::All)
                .build(),
            CPU::default().build().error_on_failure(),
        ])
    }
    #[cfg(not(feature = "coreml"))]
    Err(Error::Config(
        "provider `coreml` requires building with --features coreml".into(),
    ))
}

fn directml_chain() -> Result<Vec<ExecutionProviderDispatch>> {
    #[cfg(feature = "directml")]
    {
        Ok(vec![
            ort::ep::DirectML::default().build(),
            CPU::default().build().error_on_failure(),
        ])
    }
    #[cfg(not(feature = "directml"))]
    Err(Error::Config(
        "provider `directml` requires building with --features directml".into(),
    ))
}

fn cuda_chain() -> Result<Vec<ExecutionProviderDispatch>> {
    #[cfg(feature = "cuda")]
    {
        Ok(vec![
            ort::ep::CUDA::default().build(),
            CPU::default().build().error_on_failure(),
        ])
    }
    #[cfg(not(feature = "cuda"))]
    Err(Error::Config(
        "provider `cuda` requires building with --features cuda".into(),
    ))
}

async fn load_clip(
    model_id: &str,
    providers: &[ExecutionProviderDispatch],
    requires_internal_data: bool,
) -> Result<Clip> {
    let raw_dir = resolve_model_dir(model_id).await?;
    let model_dir = if requires_internal_data {
        ensure_folded_model_dir(model_id, &raw_dir)?
    } else {
        raw_dir
    };
    Clip::from_local_dir(&model_dir)
        .with_execution_providers(providers)
        .build()
        .map_err(|e| Error::Load(e.to_string()))
}

/// True for execution providers whose graph optimizer hits the ORT
/// `model_path must not be empty` assertion when the model uses external
/// data initializers. CPU is fine on its own; everything else partitions
/// the graph and can lose the path context.
fn provider_requires_internal_data(provider: ExecutionProvider) -> bool {
    !matches!(provider, ExecutionProvider::Cpu)
}

/// Return a model dir that is guaranteed not to use external-data
/// initializers, copying / folding from `raw_dir` if needed.
///
/// The folded copy lives under
/// `~/.cache/clip-tag/folded-models/<sanitized-model-id>/` and is keyed by
/// the sizes of the source `.onnx` / `.onnx.data` files. If the user
/// updates the HF cache (e.g. a new model revision), file sizes change
/// and the fold is redone.
fn ensure_folded_model_dir(model_id: &str, raw_dir: &Path) -> Result<PathBuf> {
    let text_onnx = raw_dir.join("text.onnx");
    let visual_onnx = raw_dir.join("visual.onnx");

    let text_external = external_data::has_external_data(&text_onnx)?;
    let visual_external = external_data::has_external_data(&visual_onnx)?;
    if !text_external && !visual_external {
        return Ok(raw_dir.to_path_buf());
    }

    let folded_dir = folded_model_dir_path(model_id)?;
    let marker = fold_cache_marker(raw_dir)?;
    let marker_path = folded_dir.join("_FOLD_DONE");
    if marker_path.exists()
        && std::fs::read_to_string(&marker_path)
            .map(|s| s == marker)
            .unwrap_or(false)
    {
        return Ok(folded_dir);
    }

    tracing::info!(
        src = %raw_dir.display(),
        dst = %folded_dir.display(),
        "folding external-data initializers (one-time per model revision)",
    );
    populate_folded_dir(raw_dir, &folded_dir)?;
    std::fs::write(&marker_path, marker)
        .map_err(|e| Error::Load(format!("write fold marker: {e}")))?;
    Ok(folded_dir)
}

fn folded_model_dir_path(model_id: &str) -> Result<PathBuf> {
    let base = dirs::cache_dir()
        .ok_or_else(|| Error::Load("no cache dir".into()))?
        .join("clip-tag")
        .join("folded-models");
    let sanitized = model_id
        .replace('/', "__")
        .replace('\\', "__")
        .replace(':', "_");
    Ok(base.join(sanitized))
}

fn fold_cache_marker(raw_dir: &Path) -> Result<String> {
    let entries = ["text.onnx", "visual.onnx", "text.onnx.data", "visual.onnx.data"];
    let mut parts = Vec::with_capacity(entries.len() + 1);
    parts.push("FOLD_V1".to_string());
    for name in entries {
        let path = raw_dir.join(name);
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        parts.push(format!("{name}={size}"));
    }
    Ok(parts.join("\n"))
}

fn populate_folded_dir(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst).map_err(|e| Error::Load(e.to_string()))?;

    for entry in std::fs::read_dir(src).map_err(|e| Error::Load(e.to_string()))? {
        let entry = entry.map_err(|e| Error::Load(e.to_string()))?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy().to_string();
        let src_path = src.join(&name);
        let dst_path = dst.join(&name);

        if name_str.ends_with(".onnx.data") {
            // External data is being inlined; drop the sidecar.
            continue;
        }
        let file_type = entry.file_type().map_err(|e| Error::Load(e.to_string()))?;
        if name_str == "text.onnx" || name_str == "visual.onnx" {
            external_data::fold_onnx_file(&src_path, &dst_path)?;
        } else if file_type.is_file() || file_type.is_symlink() {
            std::fs::copy(&src_path, &dst_path)
                .map_err(|e| Error::Load(format!("copy {}: {e}", src_path.display())))?;
        }
    }

    // `open_clip_inference::verify_model_dir` requires the `.onnx.data`
    // sidecars to exist even when not referenced. Touch empty ones — the
    // folded ONNX no longer points at them, so ORT won't read them.
    for sidecar in OPTIONAL_DATA_SIDECARS {
        let path = dst.join(sidecar);
        if !path.exists() {
            std::fs::write(&path, []).map_err(|e| Error::Load(e.to_string()))?;
        }
    }
    Ok(())
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

/// Recommended default model for a given execution provider.
///
/// Used by the CLI when the user picks a provider but does not pass `--model`.
/// Every variant returns a model id so callers can rely on a non-empty default
/// even when the feature for that provider isn't compiled in (the chain just
/// won't build, but the caller still has a sensible reference model).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecommendedModel {
    pub model_id: &'static str,
    /// Human-readable note for documentation and `--help` style listings.
    pub notes: &'static str,
}

pub fn recommended_model_for(provider: ExecutionProvider) -> RecommendedModel {
    match provider {
        ExecutionProvider::Cpu | ExecutionProvider::Auto => RecommendedModel {
            model_id: DEFAULT_MODEL_ID,
            notes: "MobileCLIP2-S3: balanced CPU baseline",
        },
        ExecutionProvider::Coreml | ExecutionProvider::Metal => RecommendedModel {
            model_id: "RuteNL/MobileCLIP2-S3-OpenCLIP-ONNX",
            notes: "MobileCLIP2-S3: well-supported on Apple Silicon ANE / GPU",
        },
        ExecutionProvider::Directml => RecommendedModel {
            model_id: "RuteNL/MobileCLIP2-S3-OpenCLIP-ONNX",
            notes: "MobileCLIP2-S3: portable across Windows DirectML GPUs",
        },
        ExecutionProvider::Cuda => RecommendedModel {
            model_id: "RuteNL/MobileCLIP2-S4-OpenCLIP-ONNX",
            notes: "MobileCLIP2-S4: heavier sibling, runs well on CUDA GPUs",
        },
    }
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
    fn every_provider_has_a_recommended_model() {
        // Failing this test means a new ExecutionProvider variant was added
        // without a recommended_model_for arm — fix the registry before
        // landing the new variant.
        for provider in [
            ExecutionProvider::Auto,
            ExecutionProvider::Cpu,
            ExecutionProvider::Metal,
            ExecutionProvider::Coreml,
            ExecutionProvider::Directml,
            ExecutionProvider::Cuda,
        ] {
            let rec = recommended_model_for(provider);
            assert!(
                !rec.model_id.is_empty(),
                "provider {provider:?} has empty model id"
            );
            assert!(
                rec.model_id.contains('/'),
                "provider {provider:?} model id should be a HF repo id, got {:?}",
                rec.model_id
            );
            assert!(
                !rec.notes.is_empty(),
                "provider {provider:?} has empty notes"
            );
        }
    }

    #[test]
    fn cpu_and_auto_default_to_workspace_default_model() {
        // Auto/CPU should be the predictable, well-tested baseline.
        assert_eq!(
            recommended_model_for(ExecutionProvider::Auto).model_id,
            DEFAULT_MODEL_ID,
        );
        assert_eq!(
            recommended_model_for(ExecutionProvider::Cpu).model_id,
            DEFAULT_MODEL_ID,
        );
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
