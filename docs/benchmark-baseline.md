# Performance baseline (v0.1)

Recorded on the development machine used for v0.1 validation.

## Environment

| Item | Value |
|------|-------|
| Date | 2026-05-24 |
| OS | macOS (darwin 25.5.0) |
| CPU | Apple M4 |
| Model | `RuteNL/MobileCLIP2-S3-OpenCLIP-ONNX` (ONNX Runtime CPU EP) |
| CLI defaults | `--threshold 0.01`, `--diversity-threshold 0.8`, `--top-k 10` |
| Vocabulary | Foundation List 2.0 (1727 labels) |
| Sample | `fixtures/corpus/sample.jpg` (256×256 synthetic JPEG) |

## Method

The `benchmark` subcommand walks the given path (file or directory),
pre-decodes a batch of images, then loops `tag_batch` over them.

```bash
# Single-image latency (batch_size 1 isolates per-image inference cost)
cargo run -p clip-tag --release -- benchmark fixtures/corpus \
  --warmup 5 --iterations 30 --batch-size 1

# Batched throughput (uses the chunked tag_batch path; default 8 under
# the balanced quality preset, override with --batch-size)
cargo run -p clip-tag --release -- benchmark fixtures/corpus \
  --warmup 5 --iterations 30 --batch-size 8
```

Notes:

- First run embeds the full vocabulary and writes a cache under `~/Library/Caches/clip-tag/vocab-cache/`.
- Subsequent invocations load the cache in <100 ms.
- For CoreML, the warmup also pays the one-time `.mlmodelc` compile (5–60 s for larger models); set `--warmup 10` or higher to keep the compile out of the timed loop.
- Decode and policy/metadata work are **excluded** from the timed loop. Numbers represent steady-state model inference only.
- Batches cycle through the corpus when it's smaller than `--batch-size`.

## Results

| Metric | Value |
|--------|-------|
| Single-image latency (CPU EP, cached vocab) | **88.4 ms** |
| Batch throughput (CPU EP, batch_size=1) | **11.3 images/sec** |
| Corpus tagging (JPEG+PNG+TIFF, recursive) | 3/3 succeeded |

CoreML / DirectML / CUDA numbers depend on hardware and are not part of
the v0.1 baseline.

## Gate

v0.1 records baseline only (no hard SLO). Aspirational target from product intent: <200 ms JPEG on M4.

Compare other models and tuning flags using [models.md](models.md).

## Reproduce batch throughput on mixed corpus

```bash
cargo run -p clip-tag --release -- benchmark fixtures/corpus \
  --warmup 3 --iterations 30 --batch-size 8
```
