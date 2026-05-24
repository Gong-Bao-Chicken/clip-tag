use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use clip_tag_core::pipeline::{self, MetadataIoHooks, Pipeline};
use clip_tag_core::policy::WriteMode;
use clip_tag_core::{default_config_from_cli, logging};
use clip_tag_metadata::{execute_plan, read_field_snapshot};

#[derive(Parser, Debug)]
#[command(name = "clip-tag", version, about = "Local CLIP image tagging")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Image file or directory to tag.
    #[arg(value_name = "PATH")]
    path: Option<PathBuf>,

    /// Recurse into directories.
    #[arg(long, short = 'r')]
    recursive: bool,

    /// Emit JSON output.
    #[arg(long)]
    json: bool,

    /// Plan metadata writes without touching files.
    #[arg(long)]
    dry_run: bool,

    /// Write metadata to files (non-RAW in v0.1).
    #[arg(long)]
    write_metadata: bool,

    /// Overwrite existing keyword fields.
    #[arg(long)]
    force: bool,

    /// Metadata write policy.
    #[arg(long, value_enum, default_value_t = WriteModeArg::EmptyOnly)]
    write_mode: WriteModeArg,

    /// Quality preset (expert flags can override specific values).
    #[arg(long, value_enum, default_value_t = QualityArg::Balanced)]
    quality: QualityArg,

    /// Number of tags to return per image.
    #[arg(long)]
    top_k: Option<usize>,

    /// Hugging Face model id (ONNX CLIP).
    #[arg(long)]
    model: Option<String>,

    /// Execution provider preference.
    #[arg(long, value_enum)]
    provider: Option<ProviderArg>,

    /// Custom vocabulary file (one label per line).
    #[arg(long)]
    vocab: Option<PathBuf>,

    /// Minimum tag score to include in metadata writes.
    #[arg(long)]
    threshold: Option<f32>,

    /// Max cosine similarity allowed between selected tags (lower = more diverse).
    #[arg(long)]
    diversity_threshold: Option<f32>,

    /// Images per ORT vision call. `1` disables batching.
    #[arg(long)]
    batch_size: Option<usize>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run throughput/latency benchmark on PATH.
    Benchmark {
        path: PathBuf,
        #[arg(long, default_value = "5")]
        warmup: usize,
        #[arg(long, default_value = "20")]
        iterations: usize,
    },
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum ProviderArg {
    Auto,
    Cpu,
    Metal,
    Coreml,
    Directml,
    Cuda,
}

impl ProviderArg {
    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Coreml => "coreml",
            Self::Directml => "directml",
            Self::Cuda => "cuda",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum WriteModeArg {
    EmptyOnly,
    Merge,
}

impl WriteModeArg {
    fn into_write_mode(self) -> WriteMode {
        match self {
            Self::EmptyOnly => WriteMode::EmptyOnly,
            Self::Merge => WriteMode::Merge,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum QualityArg {
    Fast,
    Balanced,
    Thorough,
}

struct QualityDefaults {
    top_k: usize,
    threshold: f32,
    diversity_threshold: f32,
    provider: ProviderArg,
    batch_size: usize,
}

fn quality_defaults(quality: QualityArg) -> QualityDefaults {
    match quality {
        QualityArg::Fast => QualityDefaults {
            top_k: 8,
            threshold: 0.02,
            diversity_threshold: 0.85,
            provider: ProviderArg::Auto,
            batch_size: 16,
        },
        QualityArg::Balanced => QualityDefaults {
            top_k: 10,
            threshold: 0.01,
            diversity_threshold: 0.8,
            provider: ProviderArg::Auto,
            batch_size: 8,
        },
        QualityArg::Thorough => QualityDefaults {
            top_k: 16,
            threshold: 0.005,
            diversity_threshold: 0.75,
            provider: ProviderArg::Auto,
            batch_size: 4,
        },
    }
}

fn provider_to_execution_provider(p: ProviderArg) -> clip_tag_model::ExecutionProvider {
    match p {
        ProviderArg::Auto => clip_tag_model::ExecutionProvider::Auto,
        ProviderArg::Cpu => clip_tag_model::ExecutionProvider::Cpu,
        ProviderArg::Metal => clip_tag_model::ExecutionProvider::Metal,
        ProviderArg::Coreml => clip_tag_model::ExecutionProvider::Coreml,
        ProviderArg::Directml => clip_tag_model::ExecutionProvider::Directml,
        ProviderArg::Cuda => clip_tag_model::ExecutionProvider::Cuda,
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    if cli.dry_run {
        logging::init_quiet();
    } else {
        logging::init();
    }

    pipeline::set_metadata_hooks(MetadataIoHooks {
        read_snapshot: |p| {
            read_field_snapshot(p).map_err(|e| clip_tag_core::Error::Metadata(e.to_string()))
        },
        write_plan: |p, plan, dry| {
            execute_plan(p, plan, dry).map_err(|e| clip_tag_core::Error::Metadata(e.to_string()))
        },
    });

    let preset = quality_defaults(cli.quality);
    let top_k = cli.top_k.unwrap_or(preset.top_k);
    let threshold = cli.threshold.unwrap_or(preset.threshold);
    let diversity_threshold = cli
        .diversity_threshold
        .unwrap_or(preset.diversity_threshold);
    let provider = cli.provider.unwrap_or(preset.provider);
    let batch_size = cli.batch_size.unwrap_or(preset.batch_size).max(1);

    // When the user didn't pick a model, fall back to the recommended one for
    // the chosen provider. Predefined per-provider — see
    // `clip_tag_model::recommended_model_for`.
    let model_id = cli.model.clone().or_else(|| {
        let rec = clip_tag_model::recommended_model_for(provider_to_execution_provider(provider));
        Some(rec.model_id.to_string())
    });

    if !(0.0..=1.0).contains(&threshold) {
        anyhow::bail!("--threshold must be in [0.0, 1.0], got {}", threshold);
    }
    if !(0.0..=1.0).contains(&diversity_threshold) {
        anyhow::bail!(
            "--diversity-threshold must be in [0.0, 1.0], got {}",
            diversity_threshold
        );
    }
    if top_k == 0 {
        anyhow::bail!("--top-k must be >= 1");
    }
    if batch_size == 0 {
        anyhow::bail!("--batch-size must be >= 1");
    }

    if let Some(Commands::Benchmark {
        path,
        warmup,
        iterations,
    }) = cli.command
    {
        return run_benchmark(
            &path,
            warmup,
            iterations,
            model_id,
            provider,
            cli.vocab,
            diversity_threshold,
        );
    }

    let path = cli
        .path
        .ok_or_else(|| anyhow::anyhow!("PATH required (see --help)"))?;

    if path.is_dir() && !cli.recursive {
        anyhow::bail!("{} is a directory; pass --recursive", path.display());
    }

    let config = default_config_from_cli(
        cli.write_metadata,
        cli.dry_run,
        cli.force,
        cli.write_mode.into_write_mode(),
        cli.recursive,
        model_id,
        Some(provider.as_str().to_string()),
        cli.vocab,
        Some(diversity_threshold),
        batch_size,
    );

    let pipeline = Pipeline::from_model(config)?;
    let result = pipeline.run_batch(&path, top_k, threshold);

    if cli.json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        print_text(&result);
    }

    if result.failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}

fn print_text(result: &clip_tag_core::BatchResult) {
    let is_dry_run = result
        .files
        .iter()
        .any(|f| f.write.as_ref().and_then(|w| w.dry_run.as_ref()).is_some());

    // Errors and tag listings always go first so they're easy to skim.
    for file in &result.files {
        if let Some(err) = &file.error {
            eprintln!("{}: error: {err}", file.path);
            continue;
        }
        for tag in &file.tags {
            println!("{}\t{}\t{:.4}", file.path, tag.label, tag.score);
        }
    }

    // Metadata write outcomes (or dry-run plans) go in a separate block at the
    // end so a long batch doesn't interleave tag rows with plan blocks.
    if is_dry_run {
        println!();
        println!("--- dry-run write plan ---");
        for file in &result.files {
            let Some(write) = &file.write else { continue };
            println!("{}: metadata: {}", file.path, write.decision);
            if let Some(dry) = &write.dry_run {
                for op in &dry.operations {
                    println!(
                        "  {} -> {:?} (overwrite={})",
                        op.field, op.values, op.overwrite
                    );
                }
            }
        }
    } else {
        for file in &result.files {
            let Some(write) = &file.write else { continue };
            println!("{}: metadata: {}", file.path, write.decision);
        }
    }
}

fn run_benchmark(
    path: &std::path::Path,
    warmup: usize,
    iterations: usize,
    model: Option<String>,
    provider: ProviderArg,
    vocab: Option<PathBuf>,
    diversity_threshold: f32,
) -> anyhow::Result<()> {
    use clip_tag_model::{load_shared, ExecutionProvider, ModelConfig};
    use std::time::Instant;

    let config = ModelConfig {
        model_id: model.unwrap_or_else(|| clip_tag_model::DEFAULT_MODEL_ID.to_string()),
        vocab_path: vocab,
        provider: ExecutionProvider::parse(Some(provider.as_str()))?,
        diversity_threshold: Some(diversity_threshold),
    };
    let engine = load_shared(config)?;

    let sample = select_benchmark_sample(path)?;

    for _ in 0..warmup {
        let _ = engine.tag_path(&sample, 10)?;
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = engine.tag_path(&sample, 10)?;
    }
    let elapsed = start.elapsed();

    let per_image_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let ips = iterations as f64 / elapsed.as_secs_f64();

    println!("benchmark sample: {}", sample.display());
    println!("iterations: {iterations} (warmup {warmup})");
    println!("total: {:.3}s", elapsed.as_secs_f64());
    println!("latency: {per_image_ms:.2} ms/image");
    println!("throughput: {ips:.2} images/sec");
    Ok(())
}

fn select_benchmark_sample(path: &std::path::Path) -> anyhow::Result<std::path::PathBuf> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    Pipeline::discover_paths(path, true)?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no supported images found under {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::select_benchmark_sample;
    use image::{ImageBuffer, Rgb};
    use tempfile::tempdir;

    #[test]
    fn benchmark_sample_selects_supported_non_jpg() {
        let dir = tempdir().unwrap();
        let png_path = dir.path().join("sample.PNG");
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(8, 8, |x, y| {
            Rgb([((x * 3) % 255) as u8, ((y * 5) % 255) as u8, 42])
        });
        img.save(&png_path).unwrap();

        let selected = select_benchmark_sample(dir.path()).unwrap();
        assert_eq!(selected, png_path);
    }
}
