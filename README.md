# clip-tag

Local CLIP-based image tagging CLI with safe, policy-driven metadata writes.

## Status

**v0.1 complete** — local ONNX tagging, metadata policy, dry-run, ExifTool-backed writes, fixture corpus, and benchmark baseline.

## Prerequisites

- Rust 1.78+
- [ExifTool](https://exiftool.org/) (`brew install exiftool`) for metadata read/write
- First run downloads the default ONNX CLIP model from Hugging Face (~350 MB cached)

## Quick start

```bash
cargo build --release -p clip-tag

# Tag a single image (first run precomputes vocab embeddings ~1–2 min CPU)
cargo run -p clip-tag --release -- path/to/photo.jpg --top-k 10

# Tag with stronger keyword differentiation (lower = more diverse)
cargo run -p clip-tag --release -- path/to/photo.jpg --top-k 10 --diversity-threshold 0.85

# Recursive directory, JSON output
cargo run -p clip-tag --release -- ./photos --recursive --json

# Plan metadata writes (non-destructive)
cargo run -p clip-tag --release -- photo.jpg --dry-run --write-metadata

# Write keywords (empty fields only by default)
cargo run -p clip-tag --release -- photo.jpg --write-metadata --force

# Benchmark
cargo run -p clip-tag --release -- benchmark fixtures/corpus

# Use a custom model repo or local model directory
cargo run -p clip-tag --release -- photo.jpg --model RuteNL/MobileCLIP2-S4-OpenCLIP-ONNX
```

## Workspace

| Crate | Role |
|-------|------|
| `clip-tag` (`clip-tag-cli`) | CLI entrypoint |
| `clip-tag-core` | Policy engine, normalization, batch pipeline |
| `clip-tag-image` | JPEG/PNG/TIFF decode |
| `clip-tag-model` | ONNX CLIP + vocabulary scoring |
| `clip-tag-metadata` | ExifTool metadata executor |

## Defaults

- **Model:** `RuteNL/MobileCLIP2-S3-OpenCLIP-ONNX` (override with `--model`)
- **Provider:** `auto` (CPU)
- **Vocabulary:** [Foundation List 2.0.1](assets/vocab/SOURCE.md) (1727 photography labels)
- **Tags per image:** `--top-k 10`
- **Score cutoff:** `--threshold 0.01` (minimum softmax score to keep a tag; applied after inference)
- **Keyword diversity:** `--diversity-threshold 0.8` (max cosine similarity between selected tags; lower = more diverse)
- **Write policy:** empty keyword fields only unless `--force` (see ADRs)
- **ONNX layout:** supports both external-data (`*.onnx.data`) and single-file ONNX repos

See [docs/models.md](docs/models.md) for tuning details and **alternative models** (MobileCLIP2 S0–S4, SigLIP, DFN ViT-H, etc.).

## Model layout

`--model` accepts either:

- a Hugging Face model repo id, or
- a local model directory path.

Required files:

- `model_config.json`
- `open_clip_config.json`
- `special_tokens_map.json`
- `text.onnx`
- `tokenizer.json`
- `tokenizer_config.json`
- `visual.onnx`

Optional files:

- `text.onnx.data`
- `visual.onnx.data`

When sidecar files are missing (single-file ONNX export), `clip-tag` creates empty compatibility sidecars so ONNX Runtime can initialize external-initializer paths consistently.

## Architecture decisions

See [docs/adr/](docs/adr/README.md):

- [ADR 001](docs/adr/001-force-semantics.md) — `--force` overwrites mapped keyword fields
- [ADR 002](docs/adr/002-empty-only-default.md) — default writes only when fields are empty
- [ADR 003](docs/adr/003-normalization-policy.md) — case/separator/dedupe only in v0.1
- [ADR 004](docs/adr/004-metadata-field-mapping.md) — XMP/IPTC/EXIF/dc field contract

## Documentation

- [docs/README.md](docs/README.md) — index
- [docs/models.md](docs/models.md) — tagging parameters and alternative ONNX models
- [docs/benchmark-baseline.md](docs/benchmark-baseline.md) — performance baseline
- [docs/interop-checklist.md](docs/interop-checklist.md) — metadata interop matrix

## Validation artifacts

- Fixture corpus: `fixtures/corpus/` (JPEG, PNG, TIFF)

## Build & test

```bash
cargo test --workspace
cargo test -p clip-tag-model -- --ignored   # optional: requires downloaded model
```

## Deferred (post v0.1)

- RAW sidecar pipeline
- Online providers / plugin API surface
- OpenCLIP provider
