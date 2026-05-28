# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project loosely adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Versioning policy

- **`MAJOR`** aligns with the targeted .NET runtime major (currently `10`, targeting `net10.0`). A `MAJOR` bump implies a runtime target change.
- **`MINOR`** is bumped for any breaking change to the public API surface or a significant new capability surface (new loader family, new training mode, new public client).
- **`PATCH`** is bumped for backwards-compatible fixes, performance work, internal improvements, additional model architectures within an existing loader family, and documentation.

"Public API" is the surface inside `DevOnBike.Overfit.*` reachable without `InternalsVisibleTo`. Anything accessed via `InternalsVisibleTo` (`DevOnBike.Overfit.Tests`, `Benchmarks`) is **not** part of the public contract and may change between any two patches.

Pre-release suffixes (e.g. `10.1.0-beta.1`) are used for surface changes that need real-world validation before the public release. Pre-releases are pushed to NuGet with the `-beta`, `-rc`, or `-preview` SemVer suffix.

## [Unreleased]

### Added

- `OverfitClient.LoadGguf(path)` — turnkey one-line facade for in-process GGUF chat (Qwen / Llama / Mistral / Mixtral / Qwen-MoE / GPT-2 family). Auto-detects chat template, tokenizer, and stop sequences.
- BERT-family sentence embeddings with bit-parity vs HuggingFace / PyTorch: `SentenceEmbedder.ForMiniLm`, `ForBgeEnV15`, `ForE5` (validated cosine ≈ 1.0 on real weights).
- ReAct agentic loop driver (`ReActAgent`) over `ChatSession` + `ToolCallConstraint`, with auto-registered `finish` tool.
- Sliding-window eviction in `ChatSession` for long-running conversations (`new ChatSession(slidingWindow: true)`).
- Mixture-of-Experts inference: Qwen1.5-MoE and Mixtral-8x7B (Q4_K_M and Q8_0), validated coherent on real models.
- `DataParallelTrainer` — N replicas + gradient averaging, model-agnostic.
- Gradient checkpointing for training memory (24× reduction on 12-layer GPT-1).
- CRNN / CTC training stack: `Crnn` IModule facade, `CtcLoss` (NLL + finite-difference verified), `CtcDecoder` (greedy), `TransposeLastTwo` graph op.
- Convolution training: padding / stride / bias with correct backward (previously VALID / no-bias only); finite-difference verified.
- `Training/LearningRateSchedule` — cosine, warmup, step.
- ONNX import: linear topology (`OnnxImporter.Load`) and DAG with skip connections (`OnnxGraphImporter.Load`). External `.data` sidecar files resolved automatically. Hand-rolled protobuf parser (no `Google.Protobuf` dependency).
- LLM inference via native GGUF mmap loaders for Qwen2.5 (0.5B–32B), Llama-2/3.x, Mistral, Mixtral. K-quant kernels: Q4_K_M / Q6_K (verbatim), Q8_0 (de-interleaved).
- Three productised service packages in [`COMMERCIAL.md`](COMMERCIAL.md) — Private .NET RAG/Agent PoC, Python/ONNX-sidecar replacement, Zero-GC inference audit.
- `SECURITY.md` (disclosure policy + response timeline), `SUPPORT.md` (free vs commercial support routing), `CONTRIBUTING.md` (code style + AOT bans + CLA), `CHANGELOG.md` (this file).

### Changed

- NuGet package licensing metadata corrected to `PackageLicenseFile` referencing `LICENSE.md` (the dual-license document), replacing a previous SPDX expression that did not describe the dual-license arrangement accurately.
- `LICENSE` renamed to `LICENSE.md` for proper Markdown rendering on nuget.org. Git history preserved via `git mv`.
- Scenario documentation under `docs/scenarios/` updated to reflect 2026-05 reality: removed stale claims about "no Transformers", "no MultiHeadAttention", "ONNX import is planned, not shipped" — these capabilities have shipped.
- README hero now leads with a single-sentence value proposition; technical descriptors moved below it.

### Removed

- `OverfitPool<T>` — empirically slower than `ArrayPool<T>.Shared`-wrapped `PooledBuffer<T>` (3× single-thread, ~3000× multi-thread contention under benchmark). Benchmarks retained in `Sources/Benchmark/` as a regression dossier.

### Fixed

- `devskim` and `checkmarx-one` GitHub Actions workflows now target the `main` branch — previously they targeted the non-existent `master`, so scans never triggered automatically.
- GPT-2 loader peak-RAM regression: lazy chunk streaming reduces peak load RAM from ~2× steady-state to ~1× steady-state.
- RoPE row-permutation bug in `SafetensorsLlamaLoader` (HF rotate-half vs GGUF adjacent-pair on q/k) — validated coherent on real Qwen2.5-0.5B.
- Mixture-of-Experts `norm_topk_prob` handling is now architecture-aware (`qwen2moe` = raw softmax, Mixtral = renormalised) — verified coherent on full 24-layer Q8_0 run.

### Security

- Two GitHub security-scan workflows (DevSkim, Checkmarx) reactivated by fixing the dead `master`-branch trigger. Future runs will surface SARIF findings to the GitHub Security tab.

## [10.0.15] and earlier

Pre-`10.0.15` history is not retroactively documented in this changelog. See `git log 10.0.15 --oneline` for the historical commit record. From this entry forward (10.0.16+), changes are tracked here per Keep-a-Changelog conventions.
