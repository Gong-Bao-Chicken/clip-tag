use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use clip_tag_core::pipeline::{self, MetadataIoHooks, Pipeline};
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

    /// Number of tags to return per image.
    #[arg(long, default_value = "10")]
    top_k: usize,

    /// Hugging Face model id (ONNX CLIP).
    #[arg(long)]
    model: Option<String>,

    /// Execution provider preference.
    #[arg(long, value_enum, default_value_t = ProviderArg::Auto)]
    provider: ProviderArg,

    /// Custom vocabulary file (one label per line).
    #[arg(long)]
    vocab: Option<PathBuf>,

    /// Minimum tag score to include in metadata writes.
    #[arg(long, default_value = "0.01")]
    threshold: f32,

    /// Max cosine similarity allowed between selected tags (lower = more diverse).
    #[arg(long, default_value = "0.8")]
    diversity_threshold: f32,
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
}

impl ProviderArg {
    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
        }
    }
}

fn main() -> anyhow::Result<()> {
    logging::init();

    pipeline::set_metadata_hooks(MetadataIoHooks {
        read_snapshot: |p| {
            read_field_snapshot(p).map_err(|e| clip_tag_core::Error::Metadata(e.to_string()))
        },
        write_plan: |p, plan, dry| {
            execute_plan(p, plan, dry).map_err(|e| clip_tag_core::Error::Metadata(e.to_string()))
        },
    });

    let cli = Cli::parse();

    if !(0.0..=1.0).contains(&cli.threshold) {
        anyhow::bail!("--threshold must be in [0.0, 1.0], got {}", cli.threshold);
    }
    if !(0.0..=1.0).contains(&cli.diversity_threshold) {
        anyhow::bail!(
            "--diversity-threshold must be in [0.0, 1.0], got {}",
            cli.diversity_threshold
        );
    }
    if cli.top_k == 0 {
        anyhow::bail!("--top-k must be >= 1");
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
            cli.model,
            cli.provider,
            cli.vocab,
            cli.diversity_threshold,
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
        cli.recursive,
        cli.model,
        Some(cli.provider.as_str().to_string()),
        cli.vocab,
        Some(cli.diversity_threshold),
    );

    let pipeline = Pipeline::from_model(config)?;
    let result = pipeline.run_batch(&path, cli.top_k, cli.threshold);

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
    for file in &result.files {
        if let Some(err) = &file.error {
            eprintln!("{}: error: {err}", file.path);
            continue;
        }
        for tag in &file.tags {
            println!("{}\t{}\t{:.4}", file.path, tag.label, tag.score);
        }
        if let Some(write) = &file.write {
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
