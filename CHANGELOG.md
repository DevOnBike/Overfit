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
- `docs/scenarios/aspnet-microservice.md` code samples migrated from the old `new AutogradNode(...)` / `model.Forward(null, ...)` / `.DataView.AsReadOnlySpan().ToArray()` allocating path to the modern zero-allocation `InferenceEngine.FromSequential(...)` + `engine.Predict(...)` / `engine.Run(input, output)` path. Added a second pattern for in-process LLM / RAG / agents via `OverfitClient.LoadGguf(...)`.
- ROADMAP.md `Quantized weight storage at inference time` snapshot row updated from stale pre-mmap numbers (17.2 tok/s @ 4.40 GB heap-allocated) to current post-mmap numbers (~19 tok/s @ 3.20 GB), reconciling with README.md and docs/TECHNICAL.md.
- docs/TECHNICAL.md Qwen2.5-3B benchmark table now carries a dated "latest verified benchmark" block (date, model file, hardware, runtime, LLamaSharp version) so reviewers can tie every benchmark claim to a specific run.
- `docs/ONNX_IMPLEMENTATION_PLAN.md` archived to `docs/archive/ONNX_IMPLEMENTATION_PLAN_MVP.md` with a redirect banner pointing at README/TECHNICAL — the original MVP plan claimed `Sequential` only, no DAG, no skip connections, no BatchNorm, no Conv stride/padding, which contradicted the shipped capability set.
- README hero now leads with a single-sentence value proposition; technical descriptors moved below it.

### Removed

- `OverfitPool<T>` — empirically slower than `ArrayPool<T>.Shared`-wrapped `PooledBuffer<T>` (3× single-thread, ~3000× multi-thread contention under benchmark). Benchmarks retained in `Sources/Benchmark/` as a regression dossier.

### Added

- `BatchedPrefillParityTests.BatchedPrefill_MatchesSingleToken_RealChatMLTokens` — a default-run `[Fact]` that prefills a real ChatML prompt (special tokens + long system message, ~50 tokens) through the batched path and asserts the engine greedily generates coherent text ("Paris"). Closes the test gap where batched prefill was only ever exercised with a 3-token prompt and a skipped `[LongFact]` parity check.

### Fixed

- **`BertEncoder` arena under-sizing → `OutOfMemoryException` on longer inputs.** The autograd arena was sized as `NumLayers · T · dFF · k`, which ignored the attention-scores term (`heads · T²`) that dominates at the sequence-length limit — so embedding a paragraph-length text (near the 256-token default) threw `NativeBuffer exhausted`, violating the encoder's contract that inputs up to `MaxSequenceLength` are valid. Re-derived the arena from an explicit per-layer tape estimate (`2·heads·T² + 2·T·dFF + 16·T·d`, +25% headroom). Surfaced while wiring in-process RAG over real documents.
- **Native AOT guard now actually verifies Native AOT.** Previously the `aot-guard` CI job ran `dotnet publish` on `Sources/Main` (a class library), which does not invoke ILCompiler — only an executable can be AOT-compiled. The check was symbolic and would not catch trim/AOT-incompatible code added to the library. Replaced with a thin `Sources/AotSmokeTest` console exe that references `DevOnBike.Overfit`; the CI job now publishes the smoketest under `-p:PublishAot=true -p:TreatWarningsAsErrors=true` on Ubuntu (with `clang` + `zlib1g-dev` installed) and executes the produced native binary as a runtime smoke check.
- `devskim` and `checkmarx-one` GitHub Actions workflows now target the `main` branch — previously they targeted the non-existent `master`, so scans never triggered automatically.
- GPT-2 loader peak-RAM regression: lazy chunk streaming reduces peak load RAM from ~2× steady-state to ~1× steady-state.
- RoPE row-permutation bug in `SafetensorsLlamaLoader` (HF rotate-half vs GGUF adjacent-pair on q/k) — validated coherent on real Qwen2.5-0.5B.
- Mixture-of-Experts `norm_topk_prob` handling is now architecture-aware (`qwen2moe` = raw softmax, Mixtral = renormalised) — verified coherent on full 24-layer Q8_0 run.

### Security

- Two GitHub security-scan workflows (DevSkim, Checkmarx) reactivated by fixing the dead `master`-branch trigger. Future runs will surface SARIF findings to the GitHub Security tab.

## [10.0.15] and earlier

Pre-`10.0.15` history is not retroactively documented in this changelog. See `git log 10.0.15 --oneline` for the historical commit record. From this entry forward (10.0.16+), changes are tracked here per Keep-a-Changelog conventions.
