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

```bash
# One-time vocab embedding precompute (~100s CPU first run; cached thereafter)
cargo run -p clip-tag --release -- fixtures/corpus/sample.jpg --top-k 5

# Benchmark (excludes first-run vocab precompute; includes model session load per invocation)
cargo run -p clip-tag --release -- benchmark fixtures/corpus --warmup 2 --iterations 10
```

Notes:

- First run embeds the full vocabulary and writes a cache under `~/Library/Caches/clip-tag/vocab-cache/`.
- Subsequent invocations load the cache in <100 ms.
- CPU EP is the default provider in this project.

## Results

| Metric | Value |
|--------|-------|
| Single-image latency (inference only, cached vocab) | **88.4 ms** |
| Batch throughput (sequential CLI benchmark) | **11.3 images/sec** |
| Corpus tagging (JPEG+PNG+TIFF, recursive) | 3/3 succeeded |

## Gate

v0.1 records baseline only (no hard SLO). Aspirational target from product intent: <200 ms JPEG on M4.

Compare other models and tuning flags using [models.md](models.md).

## Reproduce batch throughput on mixed corpus

```bash
cargo run -p clip-tag --release -- benchmark fixtures/corpus --warmup 3 --iterations 30
```
