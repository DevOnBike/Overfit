<!--
Copyright (c) 2026 DevOnBike.
This file is part of DevonBike Overfit.
DevonBike Overfit is licensed under the GNU AGPLv3.
For commercial licensing options, contact: devonbike@gmail.com
-->

# Claim → test map

Every headline claim Overfit makes points to a test, benchmark, demo, or CI guard you can run yourself. This is
the auditability story: nothing here is "trust us" — for a regulated or security-conscious team, each row is a
re-runnable proof, and the real-model rows can be re-run against **your** model.

## How to read the "Evidence kind" column

| Kind | What it means | How to run |
|------|---------------|-----------|
| **unit `[Fact]`** | Deterministic, model-free. Runs on every `dotnet test -c Release` (fast suite). | `dotnet test -c Release` |
| **real-model `[LongFact]`** | Needs a real model on disk (`C:\qwen3b`, `C:\bielik`, `C:\whisper`, …). Skipped by default so the fast suite stays fast; flip `[LongFact]`→`[Fact]` to run, or run with the model present. This is the "verify it on your own model" path. | put the model in place, flip the attribute, `dotnet test` |
| **benchmark** | BenchmarkDotNet harness — performance numbers, not pass/fail. | `dotnet run -c Release --project Sources/Benchmark -- --filter "*Name*"` |
| **CI guard** | A build/publish step that fails the build if violated (not a `[Fact]`). | the documented publish/build command |
| **demo** | Exercised end-to-end by a runnable demo / CLI. | the demo's run command |

Test paths are under `Tests/` unless noted. `[LongFact]` is defined in `Tests/LongFact.cs` (a `FactAttribute`
subclass that auto-skips); see `Tests/README.md` for the test-runtime discipline.

## Language models

| Claim | Evidence | Kind |
|-------|----------|------|
| Coherent GGUF chat — Qwen 2.5 / Qwen 3 | `LanguageModels/Loading/Qwen3SmokeTests.cs` (asserts the answer contains "Paris") | real-model `[LongFact]` |
| …Phi-3.5 / Phi-4 | `LanguageModels/Loading/Phi3SmokeTests.cs`, `Phi4SmokeTests.cs` | real-model `[LongFact]` |
| …Gemma-2 | `LanguageModels/Loading/GemmaSmokeTests.cs` | real-model `[LongFact]` |
| …Mixtral / Qwen1.5-MoE (Mixture-of-Experts) | `LanguageModels/Loading/MixtralEndToEndTests.cs`, `Qwen2MoeEndToEndTests.cs`, `Qwen2MoeQ4KMTests.cs` | real-model `[LongFact]` |
| …Bielik (Polish LLM) | `LanguageModels/Loading/BielikSmokeTests.cs` | real-model `[LongFact]` |
| Q4_K_M / Q8_0 / Q6_K mixed-quant load is correct | `LanguageModels/Loading/NonUniformQuantGgufLoadTests.cs`, `GgufQ4KMParityTests.cs` | real-model `[LongFact]` |
| safetensors (HuggingFace) loads natively, no conversion | `LanguageModels/Loading/SafetensorsLlamaRealModelTests.cs` | real-model `[LongFact]` |
| ONNX import (linear + DAG / skip connections) | `Integrations/Onnx/OnnxImporterTests.cs`, `OnnxGraphImporterTests.cs` (fixtures under `Tests/test_fixtures/*.onnx`) | unit `[Fact]` |

## Structured output & agents

| Claim | Evidence | Kind |
|-------|----------|------|
| Guaranteed **well-formed** JSON (by construction) | `LanguageModels/Constraints/JsonGrammarConstraintTests.cs` | unit `[Fact]` |
| Guaranteed **JSON-Schema-conforming** output (types, `required`, enums, `additionalProperties:false`, nested) | `LanguageModels/Constraints/JsonSchemaConstraintTests.cs`, `JsonSchemaConstraintEndToEndTests.cs`, `Constraints/Schema/JsonSchemaCompilerTests.cs`, `JsonSchemaTrackerTests.cs` | unit `[Fact]` |
| Forced tool call — valid tool name + argument schema, no free-text parsing | `LanguageModels/Tools/ToolCallConstraintTests.cs`, `ToolCallTests.cs`, `ToolCallingChatTests.cs` | unit `[Fact]` |
| Regex-constrained decoding | `LanguageModels/Constraints/RegexConstraintTests.cs` | unit `[Fact]` |
| ReAct agent loop (tool selection + finish) | `LanguageModels/Agents/ReActAgentTests.cs`, `ReActAgentEndToEndTests.cs` + `Demo/LocalAgentAspNetDemo` `/agent` | unit `[Fact]` + demo |

## RAG

| Claim | Evidence | Kind |
|-------|----------|------|
| In-process vector store: cosine ranking, top-K, zero-alloc search | `LanguageModels/Retrieval/VectorStoreTests.cs` | unit `[Fact]` |
| **Persistent** RAG — index once, restart, query without re-embedding; re-index only changed files | `LanguageModels/Retrieval/PersistentVectorStoreTests.cs` (+ `VectorStoreTests` Save/Load) | unit `[Fact]` |
| RAG is **testable** — recall@K / MRR, paraphrase stability, false-premise traps, corpus lint | `LanguageModels/Retrieval/RagEvaluatorTests.cs`, `RagAssertTests.cs`, `CorpusLinterTests.cs` (model-free) | unit `[Fact]` |
| Bielik **Polish** RAG end-to-end (chat + retrieval + cited sources over Polish docs) | `Demo/LocalAgentAspNetDemo` Bielik preset (`Data-pl/`, `/rag/query`) — see `BIELIK.md` | demo |

## Embeddings

| Claim | Evidence | Kind |
|-------|----------|------|
| BERT sentence embeddings bit-parity vs HuggingFace — MiniLM | `LanguageModels/Embeddings/MiniLmEmbeddingEndToEndTests.cs` | real-model `[LongFact]` |
| …BGE-small / E5-small | `LanguageModels/Embeddings/BgeAndE5EmbeddingEndToEndTests.cs` | real-model `[LongFact]` |
| WordPiece + bidirectional encoder + HF loader | `LanguageModels/Embeddings/BertEncoderTests.cs`, `BertSafetensorsLoaderTests.cs` | unit `[Fact]` |

## Speech & audio

| Claim | Evidence | Kind |
|-------|----------|------|
| Whisper speech-to-text, English, pure C# CPU | `LanguageModels/Whisper/WhisperE2ETests.cs` (real `ggml-tiny.bin` + `jfk.wav` → exact transcript) | real-model `[LongFact]` |
| Whisper **Polish** transcription | `LanguageModels/Whisper/WhisperPolishE2ETests.cs` | real-model `[LongFact]` |
| MP3 input (pure-managed MPEG-1/2/2.5 Layer III decoder) | `LanguageModels/Whisper/WhisperMp3E2ETests.cs`, `Audio/Mp3ReaderTests.cs` | unit `[Fact]` + real-model |
| Preset-voice TTS (Orpheus + SNAC), watermarked | `Audio/OrpheusAcronymPronunciationE2ETests.cs` | real-model `[LongFact]` |

## Training (the moat)

| Claim | Evidence | Kind |
|-------|----------|------|
| CPU QLoRA fine-tuning of an already-quantized GGUF (4-bit base never expanded to F32) | `LanguageModels/Loading/QwenGgufQLoraE2ETests.cs`, `QwenGgufQLoraFineTuneE2ETests.cs`, `LanguageModels/LoRA/QLoRAFineTunerFacadeTests.cs` (teaches a made-up fact and recites it back) | real-model `[LongFact]` + `Demo/QLoRAFineTuneDemo` |
| Frozen-quantized linear + LoRA gradients (FD-verified) | `Core/Autograd/FrozenQuantizedLinearTests.cs`, `QLoRATrainingTests.cs` | unit `[Fact]` |
| LoRA merge → fast engine (decode speed-up, parity) | `LanguageModels/Diagnostics/LoRAMergeDecodeSpeedTests.cs`, `MergeDivergenceTests.cs` | real-model `[LongFact]` |

## Integration & deployment

| Claim | Evidence | Kind |
|-------|----------|------|
| `Microsoft.Extensions.AI` adapter (`IChatClient` / `IEmbeddingGenerator`, local LLM-as-judge) | `Adapters/MeaiAdapterEndToEndTests.cs`, `OverfitChatClientTests.cs` | unit `[Fact]` + real-model |
| MCP stdio server for Claude Code / Desktop | `Mcp/McpServerProtocolTests.cs` (12 protocol tests) + `Sources/Mcp/mcp-smoke.cmd` (real exe) | unit `[Fact]` + smoke |
| OpenAI-compatible server (`/v1/chat/completions` + SSE, `/v1/embeddings`, `/v1/models`, `response_format`) | `Demo/LocalAgentAspNetDemo` `/v1/*` (see the README curl) | demo |
| Serving benchmark math (TTFT/ITL/throughput/goodput percentiles) | `Serving/ServingLoadReportTests.cs` | unit `[Fact]` |

## Performance & engineering identity

| Claim | Evidence | Kind |
|-------|----------|------|
| **Zero-allocation decode** (≈ 1 B / token on the hot path) | `LanguageModels/Diagnostics/PrefillAllocationTests.cs` (asserts the per-request `GC.GetAllocatedBytesForCurrentThread` delta); the per-family smoke tests print `alloc … B` per generation | real-model `[LongFact]` |
| **Native AOT** — compiles, trims, runs as a native binary with no IL2026/IL3050 warnings | `Sources/AotSmokeTest/Program.cs` published under `PublishAot=true` + `TreatWarningsAsErrors=true` by the CI `aot-guard` job, then executed | CI guard |
| No LINQ / reflection / hidden allocation in `Sources/Main` runtime | `Sources/Main/BannedSymbols.txt` (RS0030=error) + the `BanJaggedFloatArrays` / `BanMultipleTopLevelTypes` MSBuild guards + the `OVERFIT0xx` analyzers | CI guard (build-time) |
| Decode throughput vs llama.cpp (same GGUF) | `LanguageModels/Loading/QwenDecodeSpeedTests.cs`, `BielikSpeedTests.cs` | benchmark / real-model |
| CNN inference vs ONNX Runtime (honest: ORT is faster on compute-heavy CNNs) | `Sources/Benchmark/LargeCnnComparisonBenchmark.cs` (with an ORT-parity cosine check) | benchmark |

---

If a claim in the README or docs is missing a row here, that is a bug — open an issue. The intent is that
**every** public claim is one runnable command away from proof.
