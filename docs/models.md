# Models and tagging parameters

`clip-tag` scores images against a fixed vocabulary using an ONNX CLIP model from Hugging Face (or a local export with the same file layout). This page covers **defaults**, **tuning flags**, and **alternative models** to try.

## CLI defaults (v0.1)

| Flag | Default | Role |
|------|---------|------|
| `--model` | `RuteNL/MobileCLIP2-S3-OpenCLIP-ONNX` | Hugging Face repo id or local model directory |
| `--top-k` | `10` | Maximum tags returned per image (after filtering) |
| `--threshold` | `0.01` | Minimum softmax score to keep a tag (pipeline filter; `0` disables) |
| `--diversity-threshold` | `0.8` | Max cosine similarity between selected label embeddings (model selection) |
| `--provider` | `auto` | ONNX Runtime execution provider (`auto` → CPU today) |

Vocabulary is the bundled [Foundation List 2.0.1](../assets/vocab/SOURCE.md) (1727 labels) unless `--vocab` points at a custom file.

### Score threshold vs diversity

These operate at different stages:

1. **Model (`--diversity-threshold`)** — While building the top‑k list, skip labels whose text embedding is too similar to an already chosen tag. Lower values → more diverse keywords; higher values → allow nearer synonyms. There is **no** minimum probability cutoff inside the model; ranking uses full-vocabulary softmax.

2. **Pipeline (`--threshold`)** — After tagging, drop tags whose softmax score is below the cutoff. When the threshold is above zero, the engine requests extra ranked candidates so filtering can still yield up to `--top-k` tags.

Example:

```bash
# Stricter scores, more diverse keywords
cargo run -p clip-tag --release -- photo.jpg --threshold 0.02 --diversity-threshold 0.75

# Looser scores (more tags may survive filtering)
cargo run -p clip-tag --release -- photo.jpg --threshold 0.005 --diversity-threshold 0.85
```

## Using a different model

Pass a Hugging Face repo id or a directory that already contains the required ONNX layout (see [README model layout](../README.md#model-layout)):

```bash
cargo run -p clip-tag --release -- photo.jpg --model RuteNL/MobileCLIP2-S2-OpenCLIP-ONNX
```

First use of a model downloads weights into the Hugging Face cache. Vocabulary embeddings are cached separately per model under `~/Library/Caches/clip-tag/vocab-cache/` (macOS) — expect a one-time precompute on first run (~1–2 minutes CPU for 1727 labels).

## Alternative models to try

All repos below are **OpenCLIP-compatible ONNX exports** published by [RuteNL](https://huggingface.co/RuteNL) for use with [`open_clip_inference`](https://lib.rs/crates/open_clip_inference) (same stack as `clip-tag`). ImageNet zero-shot numbers and rough CPU timings are from the upstream [open_clip_inference benchmarks](https://lib.rs/crates/open_clip_inference) (vision + text embed latency per image; not identical to end-to-end `clip-tag` batch time).

### MobileCLIP2 family (recommended starting point)

Balanced for local batch tagging: smaller downloads, faster inference, good enough for keyword-style vocab matching.

| Hugging Face model | ImageNet ZS (indicative) | Vision embed (ms)* | When to try |
|--------------------|--------------------------|--------------------|-------------|
| [`RuteNL/MobileCLIP2-S0-OpenCLIP-ONNX`](https://huggingface.co/RuteNL/MobileCLIP2-S0-OpenCLIP-ONNX) | ~71.5% | ~fastest | Maximum throughput, large folders, weaker GPU/CPU |
| [`RuteNL/MobileCLIP2-S2-OpenCLIP-ONNX`](https://huggingface.co/RuteNL/MobileCLIP2-S2-OpenCLIP-ONNX) | ~77.2% | 75 | Faster than default with moderate quality |
| [`RuteNL/MobileCLIP2-S3-OpenCLIP-ONNX`](https://huggingface.co/RuteNL/MobileCLIP2-S3-OpenCLIP-ONNX) | ~80.7% | 116 | **Default** — good speed/quality tradeoff |
| [`RuteNL/MobileCLIP2-S4-OpenCLIP-ONNX`](https://huggingface.co/RuteNL/MobileCLIP2-S4-OpenCLIP-ONNX) | ~79.4% | 192 | Slightly heavier than S3; compare tag quality on your corpus |
| [`RuteNL/MobileCLIP2-B-OpenCLIP-ONNX`](https://huggingface.co/RuteNL/MobileCLIP2-B-OpenCLIP-ONNX) | ~79.4% | — | Mid-size variant if S3/S4 tags feel too “mobile” |

\* Approximate vision embedding latency from upstream docs; full `clip-tag` runs also load ONNX sessions and score the full vocabulary.

### Higher-accuracy CLIP / SigLIP exports

Use when tag quality matters more than speed (larger downloads, slower per image). Re-tune `--threshold` and `--diversity-threshold` on a small fixture set — softmax scales differ across architectures.

| Hugging Face model | ImageNet ZS (indicative) | Vision embed (ms)* | When to try |
|--------------------|--------------------------|--------------------|-------------|
| [`RuteNL/ViT-SO400M-16-SigLIP2-384-ONNX`](https://huggingface.co/RuteNL/ViT-SO400M-16-SigLIP2-384-ONNX) | ~84.1% | 988 | Stronger semantics, still smaller than ViT-H |
| [`RuteNL/DFN5B-CLIP-ViT-H-14-378-ONNX`](https://huggingface.co/RuteNL/DFN5B-CLIP-ViT-H-14-378-ONNX) | ~84.4% | 1860 | High-quality stills; accept long runs on big libraries |
| [`RuteNL/ViT-gopt-16-SigLIP2-384-ONNX`](https://huggingface.co/RuteNL/ViT-gopt-16-SigLIP2-384-ONNX) | ~85.0% | 2354 | Best listed accuracy; benchmark before batch jobs |

### Suggested evaluation workflow

1. Pick 10–20 representative images (see `fixtures/corpus/`).
2. Run the same command with `--json` and different `--model` values; keep `--top-k` and tuning flags fixed for a fair compare.
3. Compare tag lists and wall time; adjust `--threshold` only after choosing a model.

```bash
for model in \
  RuteNL/MobileCLIP2-S2-OpenCLIP-ONNX \
  RuteNL/MobileCLIP2-S3-OpenCLIP-ONNX \
  RuteNL/MobileCLIP2-S4-OpenCLIP-ONNX; do
  echo "=== $model ==="
  cargo run -p clip-tag --release -- fixtures/corpus/sample.jpg \
    --model "$model" --top-k 10 --json
done
```

## Compatibility requirements

A model repo must ship the ONNX + tokenizer bundle expected by `open_clip_inference`:

- `model_config.json`, `open_clip_config.json`, `special_tokens_map.json`
- `text.onnx`, `visual.onnx`, `tokenizer.json`, `tokenizer_config.json`
- Optional external weights: `text.onnx.data`, `visual.onnx.data`

PyTorch-only checkpoints (e.g. `timm/*` or `apple/*` without an ONNX export) must be converted first; use the upstream [open_clip_inference](https://github.com/RuteNL/open_clip_inference) conversion tooling or a published `RuteNL/*-ONNX` repo.

## See also

- [Performance baseline](benchmark-baseline.md) — recorded with the default S3 model
- [Interop checklist](interop-checklist.md) — metadata write verification
- [ADRs](adr/README.md) — write policy and field mapping
