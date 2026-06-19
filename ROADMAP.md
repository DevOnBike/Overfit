# Overfit Roadmap

Zero-allocation, pure C# deep-learning framework targeting high-performance CPU inference and small/medium language model inference on .NET 10+.

**Philosophy:** minimal dependencies, predictable memory behavior, competitive CPU inference, and explicit separation between training, inference and kernels. Native-AOT compatible. Runs on low-end consumer hardware.

---

## Status snapshot

| Area | Status |
|------|--------|
| `InferenceEngine` zero-allocation hot path | ✅ Stable, 0 B/op verified |
| Linear / Conv / activation / pooling kernels | ✅ Stable |
| Autograd engine + `Parameter` + ownership cleanup | ✅ Stable (PR5) |
| `TrainingEngine` facade + Adam/AdamW/SGD | ✅ Stable |
| Evolutionary: GA + OpenAI-ES, parallel fitness | ✅ Stable, 0 B/op Ask/AskThenTell |
| ONNX import (linear topology, 14 ops) | ✅ Stable |
| ONNX import (DAG / ResNet skip connections) | ✅ Stable |
| GPT-2 inference (124M, KV-cache, 0 B/token) | ✅ Stable, parity vs PyTorch verified |
| Qwen2.5 / Llama / Mistral inference (GQA, RoPE, SwiGLU) | ✅ Stable for 0.5B/3B FP32/F16 |
| Native C# GGUF loader (F32/F16/BF16/Q8_0/Q4_K/Q6_K) | ✅ Loads `*.gguf` from Ollama/HF directly |
| Streaming token generation (`IAsyncEnumerable`) | ✅ Stable, with stop-tokens + cancellation |
| LoRA adapter (Enable/Disable, Save/Load) | ✅ Stable, zero-copy weight refs |
| **Quantized weight storage at inference time** | ✅ **Q8_0 + Q4_K_M decode paths done & parity-verified — Qwen2.5-3B Q4_K_M decodes ~19 tok/s @ 3.20 GB RAM, 1 B/token (post-mmap, 2026-05-21). Same-file A/B vs LLamaSharp/llama.cpp: ~1.5× faster on raw tok/s (~29 vs ~19), RAM parity (3.20 GB both), Overfit wins on per-token allocation (1 B vs 21 220 B). Catch-up plan in "Decode throughput catch-up vs llama.cpp" section below.** |
| Mixture-of-Experts inference (Qwen-MoE, Mixtral-8x7B) | ✅ Coherent in pure C# (Q8_0 + Q4_K_M); verified "Paris" 2026-05-27 |
| Training: gradient checkpointing | ✅ `ComputationGraph.Checkpoint` + `CheckpointedModule` — 24× live-activation cut on 12L GPT-1 |
| Training: data parallelism (N replicas) | ✅ `DataParallelTrainer` / `DataParallelSession` + thread-budget fix — ~6× throughput (24 workers) |
| **OCR: CRNN + CTC** | ✅ `Crnn` facade + `CtcLoss` + `CtcDecoder` (greedy / beam / LM-rescored) + `NGramCtcLanguageModel`; reads synthetic digits + lexicon words |
| **CNN training perf (BN2D adaptive parallel)** | ✅ Size-gated `Parallel.For(0, C)` in `BatchNorm2DBackward` — **CIFAR backward −31.1% (BN2D −72.8%)**, MNIST unchanged. Adaptive crossover at N·C·hw > 200K |
| **Sentence embeddings (BERT encoder)** | ✅ `WordPieceTokenizer` + `BertEncoder` (bidirectional, post-LN) + native `BertSafetensorsLoader` + `SentenceEmbedder` facade — loads **all-MiniLM-L6-v2 / BGE-small-en-v1.5 / E5-small-v2** from raw HF safetensors, no Python; cosine 1.0 / 0.999999 / 1.0 vs HF reference. CLS pool + query/passage prefix support. |
| **ReAct agent loop** | ✅ `ReActAgent` driver over `ChatSession` + `ToolCallConstraint`; auto-registers a synthetic `finish({answer:...})` tool. Loop verified by 6 unit tests via the testable `RunLoop` hook; e2e [LongFact] needs a 7B+ instruction-tuned model (Q4-3B too weak under constrained decoding). |
| **Critic loop + circuit breaker + summarising memory** | ✅ All four LangGraph agentic primitives shipped: `CriticLoop` (generate→critique→revise), `CircuitBreaker` (generic capped-loop), `SummarizingChatSession` (auto-compact long convos), + `ChatSession.AddUser`/`AddAssistant` history seeding. 12 unit tests. |
| GPU backend | ❌ Not started |

---

## Audio / TTS backlog (ROI-ranked, 2026-06-08)

Pure-.NET voice stack (Orpheus 3B + SNAC + voice cloning) is functional; first-word garble **root-caused & fixed**
(prompt was missing the canonical Orpheus control/priming tokens — start_of_human + BOS + end_of_text/end_of_human/
start_of_ai/start_of_speech; `OrpheusPrompt.BuildPromptTokens`). Validated objectively by transcribing output with our
own Whisper. Remaining, by ROI (value ÷ effort):

**🟢 Quick wins (small effort, every clip benefits)**
1. **Trailing babble — ✅ ROOT-CAUSED 2026-06-08: it's the SAMPLING TEMPERATURE, not the stop token.** At temp 0.45
   the model occasionally rambles ~3 s of garbled audio after the sentence before emitting the text-eos (128009);
   **greedy (temp 0) ends cleanly** (Whisper: clip ends exactly at the last word, 1274 vs 1561 codes; "AI" also
   cleaner). Mitigation: use low/greedy temp for the clone, or try temp ~0.2-0.3 if greedy tempo feels slow. (Added a
   harmless safety-net: `GenerateCachedSampled` now also stops on a `secondaryEosTokenId` = end_of_speech 128258 — but
   the model emits 128009 *after* the babble, so it wasn't the fix.) The earlier "greedy too slow" note was likely
   confounded by the (now-fixed) prompt bug — re-judge greedy tempo.
2. **Acronym lexicon — ✅ DONE + EMPIRICALLY VALIDATED 2026-06-09.** `TtsTextNormalizer` spells acronyms as spaced
   capitals ("AI"→"A I", "CPU"→"C P U", +12 new: http/https/usb/ssd/hdd/dns/vm/iot/vr/ceo/cto/faq) so Orpheus says the
   letter names. Locked in with 11 unit tests incl. substring-safety ("brain" ≠ "br A I n"). **Closed the loop the
   user's way** (`OrpheusAcronymPronunciationE2ETests` [LongFact]): synth "The AI uses the CPU and the GPU through one
   API." → our own Whisper transcribed it back **verbatim, 4/4 acronyms recovered** → the spaced-capitals convention
   provably works. Suite 1248/0.

**🟡 Structural (bigger effort, high value)**
3. **Merge LoRA → fast inference engine** — clone synth runs on the trainable graph (`VoiceCloneTrainer.Generate` →
   `TrainableLlamaModel`), preset runs on `CachedLlamaInferenceEngine` (zero-alloc/SIMD/Q4_K). **MEASURED GAP 2026-06-08
   (same sentence, Orpheus-3B Q4): clone 6.1 tok/s (490 tok / 80.3 s) vs preset 12.9 tok/s (533 tok / 41.5 s) = 2.1×.**
   Merge the adapter delta into the base weights → run clone on the fast engine ≈ preset speed (**~2.1×**). Merge math
   (grounded): LoRA `Apply(x)=(x·A)·B` (NO alpha/scale), A `[in×rank]`, B `[rank×out]` →
   `W_merged[o,i] = W_base[o,i] + Σ_r A[i,r]·B[r,o]`. Per projection: dequant base (per-head for q/k/v/o, resident for
   gate/up/down) → add delta → requant (Q8) → rebuild `LayerWeightBuffers`; also swap in the trained RMSNorm gains
   (`_ln1Gamma`/`_ln2Gamma`). Build a fresh engine via `CreateFromBuffers`, reuse base embed/lm_head/final-norm. Validate:
   merged-engine output must match the trainable-model output (same words via Whisper) AND hit ~preset tok/s. LM-head
   LoRA is off by default → no merge there. Audio-vocab restriction was training-only → irrelevant at merged inference.
   **✅ DONE 2026-06-08 — merge correct, 2.1× speed, COHERENT with SAMPLING.** Built `TrainableLlamaModel.BuildMergedEngine`
   (+ `VoiceCloneTrainer.BuildMergedEngine`, demo `--fast`, `AudioVocabConstraint`). **Merged clone decodes at ~12 tok/s
   (2.1× the trainable graph's 6.1) and is COHERENT with sampling** (temp 0.6 → Whisper: "The history of computing began
   long before the invasion of the digital computer."). **Merge proven CORRECT** via diff diagnostic
   (`MergeDivergenceTests`): merged-vs-trainable final hidden **cos 0.99994**, and the first **6 greedy tokens are
   bit-identical**. The earlier "garbage" was **greedy brittleness**, NOT a merge bug: tiny numerical drift between the
   fast engine (Q8 requant + reassociated SIMD attention + batched prefill) and the trainable `ProjVec` decode flips the
   hard argmax at ~token 6 → wrong SNAC code → derail. **Sampling avoids the hard flip → use temp 0.6 (NOT greedy) with
   `--fast`.** Caveat: shares the base embed/lmhead/backing → keep the trainer alive while the merged engine lives (the
   2nd `BuildMergedEngine` in one process double-disposes shared weights). Trailing-babble (ROADMAP #1) still applies to
   the merged+sampling path. `--fast` now usable for the clone speed-up.

**🟠 Medium-term**
4. **Real-time on CPU** — smaller same-arch ~0.5B LM (RTF 2-3) or port **Kokoro 82M** (Apache, StyleTTS2, ~1-2 wk,
   different arch) behind `ITextToSpeechEngine`. Product/moat decision (real-time is partly private — see
   `project-moat-public-private`).
5. **Signal-domain watermark** — inaudible waveform mark + detector (survives re-encode); current watermark is
   metadata-only. Compliance/IP, near launch.

**⚪ Low ROI / skip**
6. Align `OrpheusTrainingSequence` to the canonical prompt — cosmetic (inference works regardless; base dominates).
7. Zero-alloc + SIMD SNAC decode — a "zero-alloc" banner, NOT a speed win (SNAC is cheap; LM is the bottleneck).
8. PL normalization — blocked (Orpheus is EN-only; needs a PL TTS model).

---

## Adoption / launch roadmap (2026-06-04, ROI-ranked)

Strategic frame: Overfit wins on **.NET in-process deployment + training moat**, NOT raw tok/s
(raw-perf is hygiene = bottom of this list, consistent with the "stop chasing decode" pivot). Ranked by
adoption ROI. **Current execution order (user, 2026-06-04): #2 (OpenAI API) now → then #1 (M.E.AI adapter).**
Q8-KV (a perf/RAM item, adjacent to #12) is checkpointed mid-build (store + dual-mode `KeyValueCache` +
write-sites done, F32 bit-identical) — resumed later, NOT a launch blocker.

| # | Feature | ROI | State / note |
|--:|---------|----:|--------------|
| 1 | **Microsoft.Extensions.AI adapter** (`IChatClient` / `IEmbeddingGenerator` over `ChatSession` + `SentenceEmbedder`) | 10/10 | **NEXT after #2.** Smallest effort / biggest reach — drop-in for the whole .NET AI ecosystem (Semantic Kernel, anything on M.E.AI). This IS the in-process moat exposed through the standard interface. |
| 2 | **OpenAI-compatible API** (`/v1/chat/completions`, `/v1/embeddings`, SSE) | 9.5/10 | **IN PROGRESS (2026-06-04).** Opens HTTP tooling (LangChain, SK, OpenAI clients, UIs). Table-stakes parity with dotLLM. Build on `Demo/Overfit.LocalAgent.AspNet`. |
| 3 | **JSON-Schema constrained output** (required / enum / typed fields) | 9/10 | **PARTIALLY DONE** — `JsonGrammarConstraint` (well-formed JSON) + `ToolCallConstraint` exist; the GAP is full *schema conformance* (required fields, enums, type coercion). Steal dotLLM's `JsonSchemaConstraint` (FirstCharBuckets + LRU mask cache). |
| 4 | **Production LocalAgent template** (auth, audit logs, model hash, retrieved sources, tool-call logs, `/healthz` `/readyz` `/metrics`, Dockerfile) | 8.8/10 | Phase-1 walking skeleton exists (`/health` `/chat` `/reset`); this is productionization → sellable PoC. Maps to COMMERCIAL.md "Private .NET RAG/Agent PoC". |
| 5 | **RAG Stability Harness + Corpus Linter** (expected-source tests, paraphrase stability, false-premise traps) | 8.5/10 → **bump above #2 as a DIFFERENTIATOR** | "RAG is *testable*" — a genuine differentiator (not table-stakes), plays to our empirical-rigor DNA + COMMERCIAL.md "Zero-GC inference audit". Few competitors sell RAG testability. **✅ Increment 1 DONE 2026-06-05** — `LanguageModels.Retrieval.Evaluation`: `RagEvaluator` (`EvaluateRetrieval` recall@K + MRR, `EvaluateParaphraseStability` mean pairwise Jaccard, `EvaluateFalsePremise` grounded-threshold traps) + `CorpusLinter` (`FindNearDuplicates`, `FindOrphans`) over the existing `VectorStore`/`SentenceEmbedder`; pure/deterministic, 7 model-free fast tests prove "RAG is testable" in CI. Follow-on: contradiction/short-doc lint, an xUnit-friendly assertion façade, wire into the demo + a `docs/rag-testing.md`. |
| 6 | **Persistent vector store** (SQLite / file-backed) | 8/10 | In-memory `VectorStore` → restart without re-indexing. (DiskLLM persistence conversation resonates here.) |
| 7 | `dotnet new overfit-agent` template | 7.8/10 | Cheap once #4 exists. Open-source adoption lubricant. |
| 8 | **Model/profile manager CLI** (`overfit models download bielik`) | 7.5/10 | ollama-style. On the dotLLM steal-list. Onboarding friction. |
| 9 | **QLoRA CPU fine-tuning** | 7/10 build, **10/10 as moat-positioning** | Already SHIPPED. Don't build more — *lead the messaging* with it (the one thing llama.cpp structurally can't do); sell RAG/JSON as the entry. |
| 10 | Whisper / speech-to-text vertical | 6.8/10 | Done; separate launch, not core adoption. |
| 11 | Interpretability / inference hooks | 5.5/10 | Cool, slow market. dotLLM's is a stub → we could *lead* here, but it's credibility not revenue. |
| 12 | Chasing raw tok/s vs llama.cpp | 4/10 | Hygiene, not product. Consistent with the strategic pivot. Q8-KV lives near here (RAM angle is mildly above pure tok/s). |
| 13 | **MCP (Model Context Protocol)** (added 2026-06-11; **SERVER SHIPPED 2026-06-11** — order flipped server-first by user decision) | 8.5/10 | ✅ **Server done**: `DevOnBike.Overfit.Mcp` (typed stdio JSON-RPC 2.0 contracts + source-gen `McpJsonContext` — official SDK skipped on purpose: reflection/DI vs our AOT story; AOT-verified on the published native exe) + `overfit mcp <model> [--rag-dir] [--whisper-model]` with zero-egress tools `ask` / `rag_query` (citations; embeddings from the chat model itself, multilingual) / `transcribe` (lazy Whisper). 12 model-free protocol tests + e2e on real `overfit.exe` (handshake, "Paris", cited RAG, JFK transcript). `claude mcp add overfit -- overfit mcp <gguf>`; docs `docs/mcp.md`. **Remaining scope — Overfit as HOST**: bridge `McpTool → ToolDefinition` so `ReActAgent`/`ToolCallConstraint` consume tools from any MCP server (filesystem, DBs, Jira, git…) — reuses the shipped JSON-RPC layer (client side is the same framing). Honest note: host value is capped by small-model tool-calling reliability (ReAct e2e needs 7B+); server has no such ceiling. ~1-2 sessions. |

**Adjustments vs the source ranking (the "points 1 & 3" update):** #1 confirmed as the single best first move;
#3 reclassified PARTIALLY-DONE (schema-conformance is the only gap); #5 flagged to bump above #2 (differentiator
> table-stakes); #9 split into build-priority (mid, it's shipped) vs moat-positioning (top).

### What else to steal from dotLLM (launch-relevant, beyond the earlier full inventory)

Already matched/exceeded (don't redo): SIMD RoPE, gate+up fusion, decode worker-cap, spin-pool, repacked Q4_K
GEMV, prompt-lookup speculative, GGUF/chat-template. NEW launch-relevant steals, in adoption order:
- **`JsonSchemaConstraint` + `TokenMask` bit-vector + FirstCharBuckets/LRU** (`Constraints/`) → powers #3. Their
  per-state mask cache + first-char bucket prune (~99% vocab rejected before any parser copy) is the hot-path trick.
- **OpenAI DTO surface + SSE streaming + logprobs endpoint** (`DotLLM.Server/Endpoints/ChatCompletionEndpoint.cs`)
  → the exact shape for #2 (note: their server is sequential per-request, NO continuous batching — we're not behind).
- **`PrefixCache`** (system-prompt KV reuse) — we have `KvCacheSnapshot`; their server-level prefix cache is the
  wiring pattern for multi-request reuse.
- **Composable sampling pipeline** (`ISamplerStep` / `ILogitProcessor`) — cleaner than our monolithic
  `SamplingOptions`; nice-to-have for extensibility once the API exists.
- **`System.Diagnostics.Metrics` + `Activity` telemetry** (`Telemetry/`) → feeds #4's `/metrics` + the audit story.
- **Model-management CLI** (HF pull/search/list, Spectre.Console TUI) → #8.

### UPDATE 2026-06-05 — status + new ideas

**DONE since the table above:** #1 M.E.AI adapter (NuGet `DevOnBike.Overfit.Extensions.AI`), #2 OpenAI API
(`/v1/chat/completions`+SSE, `/v1/embeddings`, `/v1/models`, `response_format`), #3 JSON-Schema constrained output
(full subsystem + wired into demo `/chat/json` + OpenAI). Plus regex-constrained decoding + composable sampling
pipeline (additive). All on branch `bilbo`, validated on real Qwen3B/MiniLM.

**▶ #0 SHIP IT (consolidation/release — highest leverage NOW, before more features).** The branch has a large
unmerged, terse-committed ("bilbo" ×N), top-level-undocumented feature set. Risk = infinite feature-build, never
launch. Steps: rewrite the "bilbo" commits into descriptive ones → merge `bilbo`→`main` (main far behind at
`3059fdc`) → NuGet release (version bump + the new `.Extensions.AI` package) → top-level README/docs for the new
surface (structured outputs, OpenAI-compat, M.E.AI) → refresh the staged `linkedin-*.md`/`launch-copy.md` (they now
have much more to say).

**▶ #8 ELEVATED — global `overfit` dotnet tool (the "ollama moment" — user-flagged 2026-06-05 as the onboarding
play).** A `PackAsTool=true` global CLI: `dotnet tool install -g DevOnBike.Overfit.Cli`, then
`overfit pull qwen2.5-3b` (HF GGUF download + progress → cache `~/.overfit/models`, known aliases qwen/bielik/...),
`overfit list`, `overfit chat <model>` (interactive REPL over `OverfitClient`), `overfit serve <model>` (start the
OpenAI-compatible server). Removes the demo's biggest friction (manual GGUF download + ModelPath config). NOTE: to
make `overfit serve` reusable, the OpenAI endpoints (`OpenAiEndpoints`) should move from the demo into a shared
lib/the CLI; need a C# HF downloader (HTTP resolve repo→file→stream, the demo's `download-*.cmd` are curl scripts).
This is the single best **open-source adoption lubricant** — ties loaders + OverfitClient + the OpenAI server into
one turnkey entry point.

**New technical follow-ons (half-built / deferred):** Q8-KV decode wiring (store ready → RAM + long-context win);
**token healing** (engine-level: re-tokenize boundary + KV rollback → makes constrained/structured output
bulletproof on arbitrary schemas, fixes the BPE dead-end that 2-lite only graceful-stops); wired sampling-pipeline
(replace `TokenSampler`); **GBNF grammar constraint** (CFG → SQL/DSL constrained output, the general engine);
continuous/in-flight batching for serving (dotLLM only PLANS it → we could lead).

**Bolder new directions:** **persistent KV-cache to disk → cross-session resume** (`KvCacheSnapshot` exists; persist
to disk = resume a conversation after process restart — a genuine novelty from the DiskLLM conversation, pairs with
the embeddable identity); QLoRA "teach your model a fact" marketing showpiece (already works — the "Tarnholm" demo);
inference hooks / pure-.NET-CPU interpretability (dotLLM's is a stub → lead, credibility play).

### UPDATE 2026-06-17 — dotllm.dev re-review (site, not source) + what's NEW to take

Re-read the public site. Confirms the split: **dotLLM leans "vLLM for .NET"** (GPU/CUDA PTX, paged-KV refcount+COW,
continuous batching "planned", browser chat UI, more quants). **We lead on what they LACK: QLoRA training/fine-tune +
merge, multimodal (Whisper/TTS/CNN/embeddings), RAG-testability harness, build-time perf analyzer.** Net: don't chase
their serving depth — lead where they're thin. Genuinely-new takeaways (moved here from `ideas.md`), in adoption order:
- **🟢 `GcLatencyScope` (SustainedLowLatency during generation) — ✅ DONE 2026-06-17.** `Sources/Main/Runtime/GcLatencyScope.cs`
  (`readonly ref struct`, embeddable-safe — process-wide knob only flipped by a process-OWNER), wired into
  `OverfitOpenAiServer.HandleChatCompletions`. Trims gen-2 pauses on the allocating prefill / SSE marshalling.
- **🟢 Built-in browser chat UI** at the server root — onboarding/demo polish (static HTML over our `/v1`). Low effort;
  pairs with the `overfit serve` + global-tool (#8) onboarding play.
- **🟡 Paged KV-cache (vLLM-style refcount + copy-on-write)** — beyond our `KvCacheSnapshot`/prefix reuse: eliminates
  fragmentation for long contexts / multi-request. A real serving feature IF we deliberately pursue serving.
- **🟡 Per-model tool-call templates** (Hermes / Mistral / Llama formats) — agent-routing reliability beyond our
  generic constraint.
- **🟡 More GGUF quants** (Q5_K, Q4_0/1, Q5_0/1) — broader coverage; LOW priority (Q4_K_M dominates real models).
- **🔵 Interpretability hooks** (activation capture, logit lens, sparse autoencoders) — dotLLM has this only as a
  *stub/planned*, so we can **ship first**; plays directly to "pure-managed = fully inspectable in a debugger" +
  the COMMERCIAL.md "Zero-GC inference audit" credibility. Strongest novel-differentiator candidate.
- **🔴 SKIP: GPU/CUDA backend** — dotLLM's big edge, but CUDA is a native dependency that breaks our "no native binary"
  identity. Deliberate non-goal (perf/GPU stays the private moat per [[project-moat-public-private]]).
**Recommended order:** browser chat-UI (cheap onboarding) → interpretability hooks (novel lead) → per-model tool
templates → paged-KV/continuous-batching ONLY if serving becomes a deliberate track. But the highest-leverage move
remains #0 SHIP (both projects risk infinite feature-build; we're feature-complete — launch beats another feature).

---

## Nearest plan — llama.cpp competitive gaps (2026-05-25)

Gap analysis vs llama.cpp, ranked through the strategic frame (NOT chasing decode speed; build the
embeddability / low-end-hardware / in-process-agentic moat). Verified facts: `GenerationStats.TokensPerSecond`
EXISTS (produced by `SlmInferenceEngine`) but is NOT surfaced on the modern `CachedLlamaSession`/`ChatSession`
path; **no mmap** — `GgufReader` = `File.OpenRead` and the resident weights (`Q4KWeight`/`Q6KWeight`/`Q8Weight`)
copy ALL bytes into managed arrays (4 GB Q4_K_M ⇒ ~4 GB managed RAM).

**🟢 On-moat (do these):**
1. **mmap GGUF resident weights** — **DONE 2026-05-25.** Loadability on low-end hardware (the core moat:
   working-set RAM, not full-model RAM). `MemoryMappedModelFile` maps the whole file read-only and hands out
   zero-copy `ReadOnlyMemory<byte>` slices (a nested `MemoryManager<byte>` over the mapped pages; the parent
   owns the mapping). `Q4KWeight`/`Q6KWeight.Blocks` changed `byte[]`→`ReadOnlyMemory<byte>` (kept the `byte[]`
   ctor delegating; added `BlockSpan`); kernels read `BlockSpan` / `fixed (byte* = w.BlockSpan)` (math
   unchanged). `GgufLlamaLoader.Load(path, quantize, mmap: true)` slices verbatim-layout Q4_K/Q6_K weights
   (FFN, LM head, per-head attention Q/K/V — each head a contiguous file run) straight from the map; the engine
   holds the map and disposes it LAST. **NOW DEFAULT (`mmap: true`)** with a smart-skip: the map is built only
   when the file actually has Q4_K/Q6_K tensors (`FileHasVerbatimKQuant`) — pure-F32 / pure-Q8_0 files skip it
   and behave exactly as the copy path (no file handle held). Q8_0 (de-interleaved) + F32-fallback still copy.
   **Measured on real Qwen2.5-3B Q4_K_M (2007 MB file): managed-heap alloc 3202 → 1427 MB (−1776 MB, ~55 %),
   working-set delta 3167 → 1533 MB (~52 %).** The 1427 MB residual is the F32 embedding table (dequantized for
   lookup) — the next RAM lever (quantized embedding lookup), out of scope here. **Soak-validated 2026-05-25
   (default flip):** mmap vs copy bit-identical on Q4_K_M (maxDiff = 0), Q8_0 (maxDiff = 0), FP16 (maxDiff = 0);
   Q4_K_M decode-vs-own-F32-baseline = 29/32 top-1, worst swing 2.16 — *exactly* the pre-change recorded
   baseline, proving Phase A preserved decode bit-for-bit. Tests: `MemoryMappedModelFileTests` (4 fast),
   `GgufMmapParityTests` (4 `[LongFact]` — Q4_K_M/Q8_0/FP16 parity + RAM measurement). AOT-clean
   (`System.IO.MemoryMappedFiles`).
   ⚠ **Pre-existing (NOT mmap) finding:** `GgufQ4KMParityTests.Q4KM_TopTokenMatches_FP16Baseline` is RED —
   Q4_K_M top-1 (474) ≠ FP16 top-1 (40), 4/10 overlap, on the maximally-ambiguous 3-token prompt `[BOS,
   im_start, \n]`. This is the SAME step-0 near-tie the decode-baseline records (ref→47 / subj→474, swing 2.16);
   the assertion ("top-1 must match FP16") is over-strict for this flat-distribution prompt. Confirmed
   independent of mmap (Q4_K_M logits are byte-identical copy vs mmap). Fix later: relax to top-k overlap or
   pick a less-ambiguous prompt — NOT a decode bug.
   ► **Quantized token-embedding lookup — DONE 2026-05-25 (the post-mmap RAM lever).** The embedding table
   was always loaded as full F32 (1187 MB for Qwen-3B, vocab 151936 × dModel 2048) — the dominant residual
   after mmap. Now `GgufLlamaLoader.LoadEmbedding` keeps `token_embd` in its native K-quant layout (Q4_K/Q6_K,
   verbatim, mmap-able — token_embd is row-major [vocab, dModel] = output-major, row = token) and the lookup
   dequantizes only the looked-up row: `Q4KWeight`/`Q6KWeight`/`Q8Weight.DecodeRow(row, dst)` (zero-alloc, one
   row's super-blocks) behind `DecodeWeight.DequantizeRow`. `_embedWeights` (engine + session) changed
   `TensorStorage<float>` → `DecodeWeight`; F32 / `.bin` / safetensors paths unchanged (implicit conversion +
   F32 fallback). **Measured (Qwen-3B Q4_K_M, mmap default): live resident managed heap 2158 MB (copy) →
   238 MB (mmap)** — vs the 1427 MB mmap residual before this change (that residual WAS the F32 embedding,
   now ~0 on-heap / Q6_K file-mapped). Decode bit-identical: Q4_K_M decode-vs-F32 still *exactly* 29/32, swing
   2.16 (same `DecodeQ6_KBlock` on the same bytes as the old full-tensor dequant). Per-token cost: dModel/256
   super-block decodes (8 for Qwen-3B) — negligible vs the matmuls. Tests: `DecodeWeightRowTests` (3 fast),
   `Mmap_MeasuredResidentManagedHeap` (`[LongFact]`).
2. **Embeddings API** — **DONE 2026-05-25.** `CachedLlamaSession.Embed(tokens, pooling, normalize)` +
   `EmbeddingDimension` + `EmbeddingPooling` (Mean/LastToken), L2-normalised, pools per-token final hidden
   states. Validated on real Qwen (`EmbeddingsTests`): unit-norm, deterministic, semantic ordering holds
   (cos(cat,kitten)=0.94 > cos(cat,physics)=0.88). Unlocks in-process RAG / vector-store.
3. **Constrained generation** — logit-masking to a grammar = guaranteed-valid structured output for
   in-process agentic .NET. **JSON-mode DONE 2026-05-25.** `ITokenConstraint` (Contracts: `ApplyMask` /
   `Accept` / `IsComplete`) → `JsonStateMachine` (value-type char-level RFC-8259 acceptor: 64-level bit-stack
   for nesting, number/string/escape sub-DFAs, `IsComplete` gates EOS) → `JsonGrammarConstraint` (builds the
   per-token text table from `ITokenizer.DecodeToString`, masks the vocab each step, EOS only when complete).
   Wired through `ISlmSession.GenerateNextToken(in sampling, ITokenConstraint?)` (default interface method →
   `NotSupportedException` for non-supporting sessions like GPT-1/2; real override on `CachedLlamaSession`,
   masks `_logits` in place pre-sample) and `ChatSession.Send(..., constraint)`. **Validated end-to-end on
   real Qwen-3B Q4_K_M** (`JsonConstrainedChatTests` `[LongFact]`): reply parses via `JsonDocument.Parse` even
   when the small model would ramble. Fast tests: `JsonStateMachineTests` (accept/reject/complete cases),
   `JsonGrammarConstraintTests` (mask behaviour on a fake vocab). **Finding (handled):** GGUF pads the vocab
   (Qwen 151936 logits vs 151665 tokenizer tokens) — `ApplyMask` accepts `logits.Length ≥ tableSize` and masks
   the padding slots. **Perf note:** mask is O(vocab × token-len)/step — fine for short structured outputs; a
   per-state cache / token prefix-trie is the documented follow-on. Next: **JSON-Schema** (typed: fields /
   enum / required → deserializable to a C# record) then **GBNF** (generic grammars) — opt 6 function calling
   sits on top.

**🟡 Cheap polish:**
4. **tokens/sec on the modern path** — **DONE 2026-05-25.** `ChatSession.LastStats` (GenerationStats,
   decode-timed) exposes `TokensPerSecond`.
5. **Min-P / Mirostat / XTC sampling** — additive to `TokenSampler`/`SamplingOptions`; Min-P a sane modern
   default, Mirostat for perplexity-targeted decoding. **(Min-P DONE 2026-05-25 — `SamplingStrategy.MinP` +
   `SamplingOptions.WithMinP`; `TokenSamplerMinPTests`. Mirostat DEFERRED: it's STATEFUL (running `mu` per
   token) and doesn't fit the static `TokenSampler` / readonly `SamplingOptions` — needs a stateful sampler
   threaded through the sessions; queued. XTC later.) `ChatSession.LastStats` now exposes TokensPerSecond.**

**🟢 On-moat / done:**
6. **Function calling** — **DONE 2026-05-25 (the in-process-agentic-.NET headline).** `ToolDefinition`
   (name + description) → `ToolCallConstraint : ITokenConstraint` forces the canonical envelope
   `{"name": "<tool>", "arguments": <json>}`: fixed punctuation, the `name` value constrained to an enum
   DFA over the registered tool names (≤64, viability bit-mask), the `arguments` value delegated to
   `JsonStateMachine` (well-formed JSON). `ToolCall.TryParse` (System.Text.Json) extracts name + raw
   arguments; the caller dispatches by name to a `Func<JsonElement,string>`. Reuses the JSON-mode seam
   (`ChatSession.Send(..., constraint)`). **Validated end-to-end on real Qwen-3B Q4_K_M**
   (`ToolCallingChatTests` `[LongFact]`): model emitted `{"name": "get_weather", "arguments": {"city":"Paris"}}`,
   parsed, dispatched → `weather({"city":"Paris"})`. Fast tests: `ToolCallConstraintTests` (envelope / enum /
   bad-args rejection), `ToolCallTests` (TryParse). Argument *typing* (per-tool JSON-Schema) is the follow-on;
   the handler validates args meanwhile.

**🟢 On-moat / done:**
7. **Vector store** — **DONE 2026-05-25.** `VectorStore` (in-process, zero-dependency) + `VectorMatch`:
   `Add(id, vector, payload)` stores unit-normalised in one contiguous backing array; `Search` is a flat
   dot-product scan + top-K insertion (no full sort; span overload allocates nothing). Cosine reported is
   magnitude-invariant. Linear scan — sized for app/document-set scale, not billion-scale ANN. Closes the RAG
   loop (embeddings → store → retrieve), all in-process. Tests: `VectorStoreTests` (ranking / true-cosine /
   top-K / growth / guards). Wired into `Demo/AgentDemo` RAG step.

**Session 2026-05-25 delivered: opt 1 (tokens/sec + Min-P; Mirostat deferred — stateful), opt 2 (mmap GGUF —
~55 % less managed RAM, bit-identical; NOW DEFAULT with smart-skip, soak-validated on Q4_K_M/Q8_0/FP16),
opt 3 (embeddings), quantized token-embedding lookup (live managed heap for a 3B Q4_K_M model now 238 MB,
down from 1427 MB — the F32 embedding eliminated; decode bit-identical), and constrained generation
**JSON-mode** (well-formed JSON enforced at decode via `ITokenConstraint`/`JsonGrammarConstraint`, validated
on real Qwen), **function calling** (`ToolCallConstraint`/`ToolCall` → dispatch to a C# delegate, validated
on real Qwen), and a consolidated **agent demo** (`Demo/AgentDemo`: mmap load → RAG → tool call → JSON in one
process; ran on real Qwen-3B — 222 MB live heap, RAG ranks correctly, tool call dispatches, JSON parses) +
README/demo-README surfacing the in-process-agentic-.NET story for launch. Next: JSON-Schema (typed args) →
vector store (queued); then GBNF (generic grammars). Also queued: relax the over-strict pre-existing
`GgufQ4KMParityTests` FP16 assertion (see opt 1 note — not a decode bug); per-state mask cache if the
O(vocab×len) mask shows up in profiles.**
Also fixed a recurring flaky-suite issue: added `MathUtils.SetSeed(int)` (per-thread
repro hook) and seeded the random-tiny-base anomaly `[Fact]` tests — which surfaced that **tiny-base LoRA
convergence is init-sensitive** (some seeds diverge at lr 1e-2 / 300 steps; the seed pins a representative
converging init — the rigorous validation remains the TRAINED production base in `[LongFact]`).

## llama.cpp scope-gap snapshot (2026-05-29, full-tree read of `D:/llamacpp-tmp`)

Read the full llama.cpp source tree to map what they have that Overfit doesn't. Bucketed by ROI / effort.

**SHIPPABLE in pure C# (≤2 weeks each, candidate post-launch additions):**

- [ ] **Q2_K / Q3_K dequant + decode** — mathematically straightforward K-quants (block scale + 2-/3-bit indices), ~2 days each. Unlocks the lower end of Ollama's K-quant matrix (Q4_K_M / Q6_K already shipped).
- [ ] **Mirostat v1 / v2 samplers** — stateful (running `mu`), needs threading through `ISlmSession`. Queued from the 2026-05-25 plan; perplexity-targeted decoding.
- [ ] **Typical-p / XTC / Top-n-sigma / DRY** — pure-math sampler additions; ~1 day each.
- [ ] **Infill / FIM sampler** — prefix/middle/suffix token masking for code-completion-style models; ~1 week.
- [ ] **LoRA adapter loading + composition** — Overfit has *training* LoRA (custom); loading external `.gguf` LoRA adapters and composing multiple at runtime is ~3-5 days.
- [ ] **RoPE scaling variants** — Yarn (NTK-by-parts) / DynamicNTK / AliBi / per-section. Algebraic variants on the existing RoPE kernel, ~1-2 days each. Unlocks long-context Qwen/Llama variants.
- [ ] **Encoder-decoder (T5 / FLAN)** — reuses BERT-encoder building blocks already shipped; ~1 week.

**HARD (2-4 weeks each, evaluate niche fit first):**

- [ ] **GBNF (Generic BNF) grammar engine** — context-free parser + DFA construction + token-trie matching. Real work (~3-4 weeks). Overfit currently has JSON-mode + ToolCallConstraint (covers 90% of structured-output use). Build only if user demand surfaces for arbitrary grammars.
- [ ] **One vision-language model stack** (LLaVA / Qwen2-VL / Pixtral) — CLIP image encoder + projection + LLM fusion + tokenizer surgery. ~2-4 weeks for a single model. Doubles the addressable workload (image+text).
- [ ] **Whisper-tiny / -base ASR port** — separate `whisper.cpp` ecosystem, same conv+transformer pattern as our CRNN+BERT stack. ~1-2 weeks for a working speech-to-text. Strongest "in-process .NET multimodal" story if combined with VLM.
- [ ] **State-space / Mamba / RWKV** — recurrent-state arch; structurally different (SSM ops, scan). Defer unless a specific user model demands it.
- [ ] **Multi-token-prediction (MTP) + speculative rollback** — Qwen3.5 / Gemma3N draft heads. Tree-based verification + rollback cache. Defer.

**Out of scope (philosophy / non-goals — see "What Overfit is not"):**

- GPU backends (CUDA / Metal / Vulkan / SYCL / OpenCL / OpenVINO / Hexagon / WebGPU) — require native code. Overfit is pure-managed.
- Server / CLI binaries (`server`, `cli`, `batched-bench`, `perplexity`, `tokenize`) — Overfit is a *library*, not a daemon.
- Quantization export pipelines (`quantize`, `imatrix`, `cvector-generator`) — Overfit loads pre-quantized; building quantizers is a separate concern.
- TTS engines (OuteTTS, WavTokenizer, bark.cpp) — out of niche for now.
- Diffusion models — image generation is a different domain.
- TensorFlow / JAX / MLX checkpoints — Python ecosystem formats.

**Architectures Overfit doesn't load** (llama.cpp lists ~130; Overfit now lists 10 families — added `qwen3`, `phi3`, `gemma2`): Falcon, Gemma 1/3/3N, Mamba/Mamba2/Jamba, RWKV6/7, T5, GLM 1/2/3/4, Deepseek 1/2/V3, Cohere, Nemotron, Granite, LLaMA-4, Qwen3-MoE/3.5/4, Grok, Chameleon, Hunyuan/-V/-VL/-OCR, Pixtral, MiniCPM-V, InternVL, etc. Most are deferrable — Qwen2.5/3 + Llama-3.x + Mistral/Phi-3.5 + Gemma 2 + Mixtral cover the dominant Ollama deploy. Adding a new arch is typically 3-5 days *per family* (weight name mapping + arch-specific quirks like SwiGLU vs GeLU, GQA vs MHA, QK-norm, logit soft-caps).

---

## Decode throughput catch-up to llama.cpp — 3-phase plan (2026-05-29)

> **UPDATE 2026-05-31 — sprint executed (branch `perf`), via a different path and with a corrected diagnosis.**
> The plan below predicted parity (~28 tok/s) via MHA-consolidation + Q8-KV under an *overhead-bound* premise.
> What actually shipped on **Bielik-4.5B Q4_K_M** (bit-identical, suite 1006/0): decode worker-cap + fused
> SwiGLU gate+up + opt-in decode spin-pool + SIMD activation-quantize / RoPE / cached-attention + the Q4_K
> repacked 8×8 GEMV → **12.55 → ~17 tok/s (+36 %), same-file gap ~1.33× → ~1.13×.** Rigorous best-of-N on
> *both* engines: llama.cpp ~19.2 (short) / ~17.9 (long-ctx) ⇒ **~1.13× behind, NOT parity** (an earlier
> single-run "parity" read was retracted — always best-of-N both sides).
> **Corrected diagnosis:** single-stream decode is **DRAM-bandwidth-bound**. The FFN matmuls *were*
> dispatch-overhead-bound (fixed by the worker-cap, +13 %), but the residual gap is memory-access efficiency —
> the speed ratio equals the bandwidth-utilisation ratio, so it needs structural (not ALU) work. Implications
> for the phases: **1a MHA-consolidation** stays the one real structural lever but is a big refactor;
> **1b Q8-KV** — the attention kernel is now built + validated (`ComputeSingleHeadQ8` + `Q8KvQuant`, Phase 1),
> but after the SIMD-attend win its *speed* headroom shrank → its value is now **RAM / long-context**, not tok/s;
> **2a dequant-fused GEMV** — DONE as the repacked 8×8 GEMV and bounded (kernel ≈ 84 % of the cold-DRAM ceiling;
> VNNI / AVX-512 tried → ≈ 0, memory-bound). Full record: CHANGELOG `[Unreleased] → Performance` and
> `project_perf_sprint` memory. **Training-compute perf was also profiled and found already reasonable**
> (conv im2col+GEMM ≈ 280–330 GFLOP/s, ~22 % of peak; no easy win) — see CHANGELOG / `project-training-memory`.

**Baseline**: Qwen2.5-3B Q4_K_M, ~17 tok/s @ 3.20 GB live heap (Overfit) vs ~27 tok/s @ 3.20 GB (llama.cpp same-file A/B). Per `project_perf_sprint` memory: gap is **overhead-bound**, not bandwidth-bound — our measured BW efficiency is ~55% of DDR5 peak vs llama.cpp's ~80%.

The previous ROADMAP wording ("honest ceiling 2-2.5×, not parity") reflected the *pure-managed* constraint *without* algorithmic reorganisation. The three phases below preserve that constraint and target parity-then-overtake by tackling the structural causes of the 55%-vs-80% BW gap.

### Phase 1 (~1 week) — parity with llama.cpp (~28 tok/s)

- [ ] **1a. MHA consolidation** (2-3 days, **highest-confidence single lever**) — collapse `MultiHeadAttentionLayer`'s per-head `Wq/Wk/Wv/Wo[i]` ([dModel × dHead] each, 16 of them) into one combined matrix per projection (`[dModel × dModel]`). A single contiguous GEMV [2048 × 2048] with sequential writeback prefetches ~10× better than 16 × [2048 × 128] fragmented per-head GEMVs (per-head fragmentation thrashes the L2). This is **Lever D** from the existing ROADMAP attention-layout analysis, with the BW efficiency estimate 55 % → ~75 %. **Expected**: +50 % tok/s (17 → ~25-26). Breaking API change for per-head accessors; LoRA hooks need migration; converters for existing checkpoints stay backward-compatible (concat on load).
- [x] **1b. Q8 KV cache — DONE + wired 2026-06-05.** `KvCacheDType.Q8` stores each cached K/V vector as
  per-vector symmetric int8 + one F32 scale (`Q8KvQuant`) — ~4× smaller KV storage and attention read traffic.
  Opt-in via `CreateSession(ctx, KvCacheDType.Q8)` or `OVERFIT_KV_DTYPE=q8` (default F32 unchanged, bit-identical,
  suite 1126/0). Wiring: writes quantize through `KeyValueCache.WriteKey`; the single-token decode hot path
  (`CachedSingleHeadAttention.AttendFromCache`) attends int8 directly via
  `CachedAttentionKernel.ComputeSingleHeadQ8` (0-alloc preserved); batched prefill (`DecodeBatchedQuant`)
  dequantizes the range to an F32 scratch and reuses the proven F32 kernel; `SingleTokenProjectionKernel` writes
  made mode-agnostic. Int8 round-trip is cosine ≈ 1, not bit-identical → greedy decode stays coherent
  (**validated on real Qwen2.5-0.5B: F32 → "Paris", Q8 → "Paris"**; `Q8KvCacheCoherenceTests` [LongFact]).
  Value is **RAM / long-context**, not tok/s. Snapshot/RestoreFrom stay F32-only (prefix-cache reuse).

**Phase 1 total**: 17 → ~28-29 tok/s = **parity with llama.cpp**. Both items are well-understood, no native deps, AOT-clean.

### Phase 2 (~1 week) — overtake llama.cpp by 15-25 % (~32-34 tok/s)

- [ ] **2a. Custom AVX2 dequant-fused GEMV** (3-4 days) via `System.Runtime.Intrinsics.X86.Avx2`. Hand-write the Q4_K_M decode kernel so the dequant step and the multiply-add live in one pipeline — no intermediate F32 buffer, `Vector256<float>` + FMA + horizontal sum in registers. This is llama.cpp's primary weapon for Q4_K_M decode. Managed intrinsics are AOT-compatible (no P/Invoke). **Expected**: +15 % tok/s on the Q4_K_M path.
- [ ] **2b. L2-aware blocking + register-tiled GEMV** (2-3 days) — detect L2 size at startup (`GetLogicalProcessorInformationEx` on Windows, `sysconf` on Linux), block the per-token matmul to half of L2 (e.g. 128 KB on a 1 MB-per-core L2). Unroll 4-8 output rows, keep accumulators in `Vector256<float>` registers, stream weights, vector residentny. **Expected**: +5-10 % tok/s.

**Phase 2 total**: ~33-34 tok/s = **+20-25 % vs llama.cpp**. Pure-managed throughout.

### Phase 3 (~1-2 weeks) — structural advantage llama.cpp can't easily replicate (~40-50 tok/s on realistic prompts)

- [ ] **3. Adaptive early-exit / layer-skip**. For typical chat tokens (fillers, function words, common follow-ups) the model often "knows the answer" after K of N layers — measurable as low entropy on an intermediate logit projection. Add a tiny early-exit head per few layers; if entropy < threshold → return early. **Average computation −30-40 %** on realistic prompts; **harder tokens get full forward** (no quality cliff). Trade-off: ~1-2 % perplexity drift, tunable threshold. **llama.cpp's pipeline is fixed-depth** — they can't do this without bigger surgery. This is where pure-managed graph control becomes an *advantage*. **Expected**: +30-50 % tok/s on realistic chat workload, less on worst-case.

**Phase 3 total**: 40-50 tok/s on realistic prompts = **~1.5-1.8× llama.cpp** on chat workloads (lower on synthetic benchmarks that have no easy tokens).

### Validation gates

Every phase must pass:
1. `Gpt2TokensPerSecondBenchmark` + `GgufLlamaInferenceBenchmark` before/after (token-per-second delta + RAM delta).
2. Q4_K_M decode parity vs F32 baseline within existing tolerance (top-1 match on the canonical prompt; mean-abs logit diff < pre-change baseline).
3. Multi-thread suite stays green (no contention regression from kernel changes).
4. AOT publish guard passes (`dotnet publish -c Release -r linux-x64`).

### Order of attack

Start with **1a (MHA consolidation)** — it has the largest single contribution, the math is documented, and the validation is cheap (one benchmark run). If 1a delivers the projected +50 %, the rest are extensions; if it falls short, the gap diagnosis needs revisiting before investing in 1b/2/3.

---

## Release readiness snapshot (2026-05-29)

Capability surface at this date — what would ship if we tagged a release today:

- **LLM inference**: GPT-2 / GPT-1 / Qwen2.5 (0.5B–32B) / Llama-2-3.x / Mistral / Qwen1.5-MoE / Mixtral-8x7B from GGUF + safetensors + .bin. Q4_K_M / Q6_K / Q8_0 / F32 / F16 / BF16. mmap K-quant weights. Live managed heap ~220 MB for Qwen-3B Q4_K_M.
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5, intfloat/e5-small-v2 — all bit-parity vs HF/PyTorch (cosine 1.0 / 0.999999 / 1.0). `SentenceEmbedder.ForMiniLm` / `ForBgeEnV15` / `ForE5`.
- **Agentic stack**: ReAct loop, self-reflection critic loop, circuit breaker, summarising memory, JSON-mode + tool calling (constrained decoding), in-process `VectorStore` for RAG, `ChatSession` with sliding-window eviction. `OverfitClient.LoadGguf(path)` one-line facade.
- **Training**: gradient checkpointing (24× live-activation cut on 12L GPT-1), data-parallel trainer (~6× throughput on 24 workers), Adam/AdamW/SGD, padding/stride/bias Conv2D, BatchNorm2D (CIFAR-scale adaptive parallel −31 % backward), depthwise conv, LSTM (training + ONNX import deferred), CRNN + CTC, learning-rate schedules.
- **Loaders, all native (no Python)**: GGUF, HF safetensors (sharded), Overfit `.bin`, ONNX (linear + DAG).
- **AOT-clean**: zero LINQ / Reflection / Activator / Expression / Array.Copy / raw `ArrayPool<T>.Shared` in `Sources/Main` (all banned, RS0030).
- **Tests**: 990 fast / 0 fail / 130 [LongFact] skip. 3 headline benchmarks re-validated post-pool-refactor on this hardware (single inference 7.31× vs ONNX, concurrent 3.56× vs ONNX, CNN zero-alloc preserved).
- **Honest gaps acknowledged in README + ROADMAP**: tok/s **~1.13× behind** llama.cpp same-file (narrowed from ~1.6× by the 2026-05-31 sprint — see the UPDATE banner above; residual is DRAM-bandwidth-bound), no GPU, no diffusion / TTS / VLM / ASR, no full GBNF.

**Marketing assets**: `linkedin-article.md`, `linkedin-business.md`, `linkedin-marketing.md`, `linkedin-technical.md`, `launch-copy.md` — staged and waiting on a final review pass.

**Decision the release waits on**: the 2026-05-31 sprint already closed most of the gap (~1.6× → **~1.13×**, +36 %, bit-identical) — the honest decode story is now strong enough to ship as-is. Full parity would need the structural full-tensor-attention refactor (big), and the residual is bandwidth-bound, so it is a diminishing-returns lever rather than a launch blocker.

---

## Post-launch track #1 — Mixture of Experts (MoE)

**Why:** we already have the memory-optimization trio's first two legs — KV cache + GQA. MoE is the
third (Mixtral / Qwen-MoE). It fits the moat exactly: *big total knowledge, small per-token compute*
(only top-k experts run) and, with mmap + K-quant, *low working-set RAM* — i.e. "run Mixtral-class
models in pure C# on a CPU, no Python". Extends "run the models you know" rather than chasing scale.
Not multimodality / RLHF / Ring-Attention / TPU kernels — those are off-moat (different lane / scale /
hardware). See the 2026-05-25 analysis.

**Increment 1 — DONE 2026-05-25 (additive, does NOT touch the live decode path):**
- `GPT1Config.ExpertCount` / `ExpertUsedCount` (0 ⇒ dense — inert default) + `IsMixtureOfExperts`.
- `MoeRouter.SelectTopK(logits, topK, indices, weights)` — the gating core: top-k by logit + softmax
  over the k selected logits (== Mixtral's softmax-all → top-k → renormalize; mathematically identical).
  Zero-alloc, caller-owned spans. Tests: `MoeRouterTests`.

**Increment 2a — DONE 2026-05-25 (compute block, additive — not yet wired into the live stack):**
- `MoeFeedForwardBlock` — router projection (`ffn_gate_inp`, F32) → `MoeRouter.SelectTopK` → runs only
  the selected experts' SwiGLU FFN via the existing `CachedFeedForwardBlock.DecodeSwiGluDispatched`
  (F32 / Q8_0 / Q4_K / Q6_K — so experts are mmap-able like the dense FFN) → weighted-sum combine.
  Zero-alloc; per-token cost ≈ `ExpertUsedCount` dense FFNs. Synthetic parity test
  (`MoeFeedForwardBlockTests`): routed output == reference (selected experts' `DecodeSwiGlu` combined
  by the router weights), for top-1 (== that expert) and top-2 (weighted sum).

**Increment 2b — loader + stack wiring (needs the real MoE GGUF — now on the dev box at
`C:\qwen-moe\Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf`).**

**REAL structure inspected 2026-05-25 — Qwen2-MoE is NOT plain Mixtral** (`arch=qwen2moe`, 24 layers,
dModel 2048, MHA 16/16, RMSNorm ε1e-6, RoPE θ1e6, vocab 151936):
- **Routed experts:** `qwen2moe.expert_count=60`, `expert_used_count=4`. Expert FFN dFF = **1408**
  (≠ the dense `feed_forward_length=5632`) — derive from the tensor, not metadata:
  - `blk.L.ffn_gate_exps.weight` Q4_K `[2048, 1408, 60]` = `[dModel, expertDff, nExpert]`
  - `blk.L.ffn_up_exps.weight`   Q4_K `[2048, 1408, 60]`
  - `blk.L.ffn_down_exps.weight` **Q8_0** `[1408, 2048, 60]` = `[expertDff, dModel, nExpert]`
  - router `blk.L.ffn_gate_inp.weight` **F32** `[2048, 60]`
- **Shared expert (Mixtral has none):** a full SwiGLU FFN, dFF=**5632**, sigmoid-gated:
  - `ffn_gate_shexp` Q4_K `[2048,5632]`, `ffn_up_shexp` Q4_K, `ffn_down_shexp` Q6_K `[5632,2048]`
  - `ffn_gate_inp_shexp.weight` F32 `[2048]` (the 1-d shared gate)
  - **Forward:** `FFN(x) = σ(w_sh·x)·shared(x) + Σ_{i∈top4} wᵢ·expertᵢ(x)` (top-k probs renormalised
    = `MoeRouter`; shared scaled by `sigmoid` of a dot, NOT softmax).
- All tensor types (Q4_K / Q6_K / Q8_0 / F32) are already supported → mmap-able.

Steps: (1) ✅ `MoeFeedForwardBlock` — routed Σ (2a). (2) ✅ `Qwen2MoeFeedForwardBlock` — gated shared
expert + routed (`σ(w·x)·shared(x) + routed(x)`), `Qwen2MoeFeedForwardBlockTests`. **The full Qwen2-MoE
compute is implemented + synthetically validated.** (3) ✅ **Loader slicing — DONE 2026-05-25, validated on the REAL file.** `GgufLlamaLoader.LoadExperts`
(3-D expert tensor → 60 per-expert `DecodeWeight`; Q4_K/Q6_K verbatim, Q8_0 per-expert de-interleave) +
`LoadRouter` (transpose `ffn_gate_inp` to input-major). `Qwen2MoeLoaderTests` `[LongFact]` loads layer-0
of `Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf` (gate/up Q4_K, down Q8_0, σ-gated shared) through
`Qwen2MoeFeedForwardBlock` → finite full-density output (2048/2048, maxAbs 0.064). Also fixed a latent
bug: `CachedFeedForwardBlock` activation scratch assumed `dFF ≥ dModel` (false for a MoE expert,
1408 < 2048) → now sized `max(dModel, dFF)` (dense path unchanged). (4) ✅ **End-to-end wiring — DONE 2026-05-25 (additive, dense path strictly untouched).**
`GPT1Config.ExpertFeedForwardLength`; `BlockWeights` optional MoE fields (router + 3 expert arrays +
shared + gate); `CachedTransformerBlock` builds a `Qwen2MoeFeedForwardBlock` and dispatches on
`weights.IsMoe`; `CachedGptStack` threads the expert dims; `CachedLlamaInferenceEngine.LayerWeightBuffers`
carry the MoE weights, `BuildStackWeights` wires them, `Dispose` releases them; `GgufLlamaLoader`
recognises `qwen2moe` (reads `expert_count`/`expert_used_count` + expert dFF from the tensor) and the
layer loop branches the FFN to MoE. Fixed a latent `CachedFeedForwardBlock` scratch-sizing bug
(`max(dModel,dFF)`). **Proven correct on the real file:** `Qwen2MoeLoaderTests` decodes layer-0
end-to-end; the full-model load reaches layer 3 before hitting an unsupported quant (below).
⚠ **Real-file finding (NOT a MoE bug):** the `Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf` mixes **Q5_0** on
some layers' `ffn_down_exps` (llama.cpp's heterogeneous "_M" strategy). Q5_0 is outside our supported
set (F32/F16/BF16/Q8_0/Q4_K/Q6_K) → load throws a clear `NotSupportedException`. This is a
**quant-coverage** gap, orthogonal to MoE. To finish: grab the **Q8_0** Qwen-MoE (uniform Q8_0, all
supported) for full end-to-end + parity, OR add Q5_0 (+ likely Q5_K) dequant coverage.
(5) ✅ **Coherent generation — DONE 2026-05-26. MoE track COMPLETE end-to-end on the real model.**
Loaded the Q8_0 variant (uniform Q8_0, 14.18 GB) and `Qwen2MoeEndToEndTests` greedily answers
"What is the capital of France?" → **"Paris"** (then EOS). **The bug was top-k weight normalisation:**
Qwen1.5-MoE uses `norm_topk_prob=false` (expert weights are the raw full-softmax probabilities at the
top-k, NOT renormalised to sum 1), while `MoeRouter` was renormalising (Mixtral-style) → routed
contribution over-scaled → incoherent. Fix: `MoeRouter.SelectTopK(..., normalize)` + `GPT1Config.
NormalizeExpertWeights` (threaded stack→block→router) + loader reads `{arch}.expert_weights_norm`
(absent ⇒ false for qwen2moe). `MoeRouterTests` covers both modes. **Overfit now runs Qwen-MoE
(14B total / 2.7B active) coherently, in pure C# on CPU — pure-managed MoE inference.**
**F32 decode-parity — DONE.** `Qwen2MoeQuantParityTests.QuantizedBlock_MatchesF32Dequant_Layer0`
runs real layer-0 through `Qwen2MoeFeedForwardBlock` twice — once with the file's Q8_0 expert/shared
weights, once with an F32 dequantization of the SAME weights (same router ⇒ identical top-k routing,
so the only delta is projection quant error). Result: **relative max-diff 0.48 %** (< 5 % bar) — the
quantized MoE block is numerically faithful to F32. (A whole-model F32 baseline is infeasible —
Qwen1.5-MoE in F32 is ~57 GB — so parity is validated per-block via `LoadExpertsF32`.)
**Q5_0/Q5_K coverage — DONE.** `GgmlDequant.DecodeQ5_0Block` (legacy 32-elem, 22 B) +
`DecodeQ5_KBlock` (256-elem super-block, 176 B) mirror ggml-quants.c; wired into
`GgufReader.LoadTensorAsF32` (so dense tensors auto-route through the F32→Q8 fallback) and into
`GgufLlamaLoader.LoadExperts` (Q5 experts dequant + per-expert re-quant to Q8 — near-lossless from a
5-bit source, reuses the Q8 dot kernel; streamed one expert at a time via `GgufReader.LoadQ5RegionAsF32`
so peak load RAM is a single expert's F32, not the whole tensor's). **The smaller Q4_K_M variant
(8.84 GB — `_M` mix puts Q5_0 on ~half the layers' `ffn_down_exps`) now loads end-to-end and answers
"capital of France?" → "Paris"** (`Qwen2MoeQ4KMTests`, [LongFact]); decoder bit-unpacking unit-tested
in `GgmlDequantTests` (Q5_0 5th-bit routing, Q5_K per-group masks). The earlier dump confirmed the
file is 67% Q4_K / 14.5% Q8_0 / 14.5% Q5_0 / 3.5% Q6_K — only Q5_0 was the blocker (no Q5_K present,
but it's covered for other `_K_M` mixes).
**Mixtral — DONE.** Routed-only MoE (8 experts, top-2, no shared expert; `llama` arch with expert
metadata). `Qwen2MoeFeedForwardBlock` now treats `sharedFeedForwardLength == 0` as "no shared expert"
and reduces to the routed sum alone; `GPT1Config.HasSharedExpert` (threaded stack→block) drives it,
set by the loader from the presence of `ffn_*_shexp` tensors. `norm_topk_prob` default is now
arch-aware (false for qwen2moe, **true** for Mixtral/routed-only). The loader also handles **both**
GGUF expert layouts: merged 3-D `ffn_gate_exps` (Qwen-MoE) **and** the older per-expert 2-D
`ffn_gate.{e}.weight` (Mixtral) via `LoadExpertsSplit` (same resident dispatch as a dense FFN —
Q4_K verbatim/mmap, Q5→Q8, Q8 native, F32). **The real Mixtral-8x7B-Instruct Q4_K_M (25 GB,
32L / dModel 4096 / GQA 32:8 / expertDff 14336) loads + decodes end-to-end in pure C# on CPU**
(`MixtralEndToEndTests`, [LongFact]: finite/in-range logits, no degenerate collapse; routed-only
combine unit-tested in `Qwen2MoeFeedForwardBlockTests.NoSharedExpert_Mixtral_…`). **GGUF-embedded tokenizer (SPM) — DONE.** `GgufTokenizer` reads the `tokenizer.ggml.*` metadata
(tokens / scores / token_type / special ids) and implements the SentencePiece path
(`tokenizer.ggml.model == "llama"`): `▁` whitespace escaping + optional space-prefix, the score-driven
greedy bigram merge from llama.cpp's `llm_tokenizer_spm`, and `<0xNN>` byte fallback for OOV chars;
decode reassembles byte tokens (multi-byte UTF-8) and strips the leading space. Typed metadata-array
accessors added to `GgufReader` (`GetMetaStringArray`/`GetMetaFloatArray`/`GetMetaIntArray`). **Mixtral
now generates human-readable text in pure C# with NO side-loaded tokenizer** — `MixtralEndToEndTests`:
"[INST] What is the capital of France? [/INST]" → *"The capital of France is Paris. It's located in
the north-central part of the country…"*. SPM algorithm unit-tested on a synthetic vocab
(`GgufTokenizerTests`: merge preference, byte/multi-byte fallback, space-prefix, BOS, round-trip) +
real-Mixtral-vocab round-trip (incl. accents + em-dash). The gpt2/byte-level-BPE path throws a clear
`NotSupportedException` (Qwen/Llama-3 already have native `QwenTokenizer`/`HuggingFaceBpeTokenizer`).
**GGUF tokenizer gpt2/BPE path — DONE.** `GgufTokenizer` now also handles byte-level BPE
(`tokenizer.ggml.model == "gpt2"`: Qwen / Llama-3 / GPT-2) — bytes → GPT-2 `ByteLevelAlphabet`
(extracted as a shared helper, also used by `HuggingFaceBpeTokenizer`), pre-tokenized by a regex
selected from `tokenizer.ggml.pre` (cl100k for qwen2/llama-bpe, GPT-2 pattern for gpt-2), merged by
merge rank, with special tokens split out. So **Qwen/Llama-3 now tokenise from the GGUF with no
side-loaded `tokenizer.json`**. Validated: synthetic BPE invariants (merge rank, space marker,
special tokens, round-trip) + real Qwen-MoE-vocab round-trip (Polish/CJK/whitespace) + a **gold
cross-check that `GgufTokenizer` BPE output is byte-identical to the validated `QwenTokenizer`**.
Remaining (optional follow-ons, NOT blockers): native 5-bit dot kernel (skip the Q5→Q8 widen for a
tighter working set). MoE + GGUF-tokenizer tracks are complete.

**Scope note:** the routed math (the genuinely novel part) is done + tested. The remaining 2b is a
real integration chunk (3-D expert slicing incl. per-expert Q8_0 de-interleave, shared-expert gate,
stack wiring, real-file parity) — strictly additive (only fires for `qwen2moe`/MoE; the launched dense
path is unaffected), best done as a focused session, not rushed pre-launch.

## (Historical) Session resume point 2026-05-22 → 23 — SUPERSEDED by "Nearest plan (2026-05-25)" above

**Big session — zero-Python loading completed, chat turnkey, and the anomaly+LoRA product
track closed with an empirically-corrected verdict.** Strategic frame (still holds): NOT chasing
llama.cpp on decode; embeddability / training / product moat. See [[project-loading-story]],
[[project-loading-direction]], [[project-chat-runtime]], [[project-anomaly-lora]], [[project-perf-sprint]].

**Zero-Python loading — DONE (all inbound formats native; one-directional, NO exporters — see
[[project-loading-direction]]):**
- **Native Llama/Qwen safetensors loader** `SafetensorsLlamaLoader.Load(dir, quantize)` →
  `CachedLlamaInferenceEngine` + `LlamaConfigReader` (config.json via `Utf8JsonReader`). **VALIDATED
  coherent on real Qwen2.5-0.5B** ("The capital of France is" → " Paris..."). Found+fixed a **RoPE
  row-permute** bug (HF rotate-half → GGUF adjacent-pair on q/k weights+biases; `RopeKernel` is NEOX/
  adjacent-pair so HF weights need the llama.cpp permute). NOTE: `Scripts/convert_llama.py` has the SAME
  bug (unpermuted) — its `.bin` for RoPE models is suspect; not fixed (no Python here).
- **GPT-2 loader peak-RAM 2×→~1×** — `SequentialChunkReadStream` streams one param block at a time into
  `GPT1Model.Load` (bounded backward seek for the MHA legacy-peek); no full in-RAM `.bin` copy.
- Parity test (`SafetensorsLlamaLoaderTests`) bit-identical vs `.bin` (GQA + permute); the loader-vs-.bin
  test CANNOT catch RoPE-permute (cancels) — only the real-model run does.

**Chat runtime — `ChatSession` now actually drives Llama/Qwen + turnkey:**
- `CachedLlamaSession` now implements `ISlmSession` (was IDisposable-only — the GGUF/safetensors path
  couldn't feed `ChatSession`). `HuggingFaceChatTemplate` reads `tokenizer_config.json` chat_template.
- **`QwenChatModel.LoadFromDirectory(dir)`** — turnkey zero-Python HF dir → `ChatSession`
  (`QwenChatTokenizer` adapts `QwenTokenizer`→`ITokenizer`). **VALIDATED on real Qwen2.5-0.5B**:
  `Send("What is the capital of France?")` → "France's capital is Paris." (Qwen-only; cl100k pre-tokenizer.)

**Anomaly + LoRA product track — closed with measured verdicts:**
- **LoRA target A/B** on the anomaly task (Stage 1/2/3/All): tiny-RANDOM base → single-stage unstable,
  union wins. **TRAINED 256d production base (retrained this session, val 7.70→0.86 in 4m23s) → LM-head
  ALONE is best AND cheapest** (benign 6.45 false-positive → 0.0000, 31694× sep, 1 adapter / 8 KB; union
  needs 205 adapters for worse sep). Cross-pod residual lives in OUTPUT calibration = LM head. **Demo
  reverted to LM-head** (I'd wrongly switched to AllLinear on the misleading tiny-base result).
- **EWMA classical baseline** (`EwmaAnomalyDetector`) + head-to-head: the un-adapted GPT base does NOT
  beat a trivial EWMA floor (base normal 6.45 vs EWMA 0.00) — **the per-pod LoRA adaptation is the edge**,
  not the raw transformer. Demo shows it three-way (EWMA / GPT base / GPT+LoRA). Verdict in
  `docs/gp-anomaly-baseline.md`; GP escalation NOT warranted.
- Production base regenerated at `D:\k8s_anomaly_production.bin` (20.8 MB, out of repo).

**Continued 2026-05-23 — family-generic tokenizer/chat, llama3 RoPE scaling, GPT-2 bit-parity:**
- **Generic HF BPE tokenizer** `HuggingFaceBpeTokenizer : ITokenizer` — reads the pre-tokenizer Split
  regex + merges (both `"a b"` and `["a","b"]` shapes) + EOS from `tokenizer.json`/`tokenizer_config.json`,
  no per-model hard-coding. **VALIDATED bit-exact vs `QwenTokenizer`** (incl. digits) AND **round-trips on
  real Llama-3.2-1B** (vocab 128256). `HuggingFaceChatModel.LoadFromDirectory(dir)` = family-generic turnkey
  (stops per `ChatTemplateFormat`). **Llama-3.2-1B VALIDATED end-to-end**: raw completion "The capital of
  France is" → " Paris. The capital of Germany" (loader correct on Llama dims: 2048d/32h/8kv GQA/16L/tied;
  base model so chat echoes — completion is the right loader check for a base).
- **llama3 RoPE scaling DONE** — `RopeScaling` (NTK-by-parts, port of HF `_compute_llama3_parameters`)
  applied in `RopeTable`, parsed from `config.json rope_scaling` (only `rope_type:"llama3"`). Validated
  (`RopeScalingTests` + real Llama still " Paris" with scaling active). Closes the long-context limitation.
- **GPT-2 safetensors bit-parity VALIDATED** — downloaded `openai-community/gpt2` `model.safetensors`
  (548 MB → `C:\gpt2\`), `Load_RealGpt2Safetensors_BitParity_WithBinFixture` PASSES: loader output is
  **byte-for-byte identical** to `gpt2_small.bin`. **ALL loaders now validated on real models.**

**Out-of-repo dev artifacts (re-runs):** `C:\gpt2\model.safetensors`, `C:\llama3\{config,tokenizer,tokenizer_config}.json + model.safetensors`,
`C:\qwen3b\model.safetensors`, `D:\k8s_anomaly_production.bin`. Real-model tests are `[LongFact]` (flip to `[Fact]` to run).

**Full suite: 765 / 0 / 90 green. The 2026-05-22 batch is COMMITTED (`llama` commits); the 2026-05-23
work (generic tokenizer + chat + rope_scaling + tests) — CHECK `git status`, commit if not yet done.
Working tree was clean at session end except pre-existing staged marketing files (not mine).**

**Resume — pick one (loading + chat tracks are now fully closed & validated):**
1. **Anomaly operator workflow / productization** — real Prometheus metrics → per-pod LM-head adapter
   lifecycle (deployment story; ML verdicts settled — LM-head LoRA on a trained base is the recommendation).
2. **Mistral / non-ByteLevel tokenizers** — extend the generic tokenizer if a SentencePiece/Unigram model
   matters (currently BPE-only; Mistral pre-tokenizer untested — drop a Mistral dir to validate).
3. **Fix `Scripts/convert_llama.py` RoPE permute** (legacy; needs Python to test, low ROI) or the
   **decode lever** (LM-head alloc-free parallel matmul; low strategic ROI per the pivot).

---

**Earlier (2026-05-20): the decode-track sprint (`docs/llamacpp-cpu-analysis.md` §5 steps 1+2+3).**

Three-stage cumulative result on Qwen2.5-3B-Instruct (dev box, best-of-3, single-stream CPU decode):

| Stage | Decode | Steady RAM | Load |
|-------|-------:|-----------:|-----:|
| start (F32-upcast)             | 2.58 tok/s  | ~14 GB   | ~28 s |
| +parallel (step 1)             | 4.01 tok/s  | ~14 GB   | —    |
| +Q8_0 in-RAM (step 2)          | 13.28 tok/s | 5.85 GB  | 1.7 s |
| **+Q4_K_M in-RAM (step 3)**    | **14.56 tok/s** | **4.40 GB** | **1.4 s** |

End-to-end vs Overfit's own starting point: **5.6× decode, −69 % RAM, 20× faster load**, zero allocations per token preserved. Parity verified at both Q8 (32/32) and Q4_K_M (29/32, worst swing 2.16). 680 / 0 / 68 `-c Release`.

**Same-file A/B vs LLamaSharp (2026-05-20, option A done — corrected).** Re-benchmarked LLamaSharp 0.27.0 on the *same* `qwen.q4km.gguf`: **27.5 tok/s @ 3.2 GB**. The earlier "1.51× faster than LLamaSharp" line was wrong — it compared Overfit-Q4_K_M against LLamaSharp's *FP16* number (9.67). Diagnostic that came out of it: FP16→Q4_K_M sped llama.cpp 2.85× but Overfit only ~1.0× → **Overfit decode was overhead-bound, not bandwidth-bound.** First lever acted on (option D below): **GQA K/V-once took Overfit 13.85 → 17.2 tok/s (+24 %)**, narrowing the same-file gap from ~2.0× to **~1.6×** (still llama.cpp's favour, still 27 % more RAM committed). Defensible edge stays allocation (1 B vs 21 KB/token), pure-managed, AOT-clean, no native dep. Full numbers in `overfit-bench/RESULTS.md`.

**Git state at session end:**
- **Committed** (in the `llama` commits on `next`): all Q4_K_M code + tests + `docs/llamacpp-cpu-analysis.md` — `Q6KDotKernel.cs`, `Q6KWeight.cs`, `Q6KDotKernelTests.cs`, `Q4KMDecodeParityTests.cs`, `DecodeWeight.cs` (4-way tagged union `{F32|Q8|Q4_K|Q6_K}`), the per-weight-dispatch decode blocks (`CachedFeedForwardBlock` / `CachedSingleHeadAttention` / `CachedMultiHeadAttention` / `CachedTransformerBlock` / `CachedGptStack`), and the native Q4_K + Q6_K loader reads (`GgufLlamaLoader.cs` / `GgufReader.cs`).
- **Uncommitted at handoff:** this `ROADMAP.md` (the resume-point + Slot 2b update). Pre-existing staged files unrelated to this work: `index.html`, `launch-copy.md`, `linkedin-*.md`, `docs/parallel_opts.txt`.

### (Historical) Order that session: C → B — both delivered

- **(A) LLamaSharp re-bench on Q4_K_M — ✅ DONE 2026-05-20.** Restored the `llama` mode in `D:\overfit-bench` (LLamaSharp 0.27.0, `dotnet run -- llama qwen.q4km.gguf`). Result above: llama.cpp ~2× faster + 27 % less RAM on the same file; "1.51× faster" claim retracted everywhere. Surfaced a new lever (D).
- **(C) — NEXT NOW — Active track: anomaly + LoRA.** Pick one of the four options in the Active-track "NEXT" section below: end-to-end integration test / Stage-2 LoRA on FFN / Production base training / deployment-architecture decision. The live product track.
- **(B) Prefill GEMM (B>1 batched matmul) — ✅ DONE 2026-05-21, 3.48× TTFT.** Phase 1 `BatchedProjectionKernel` → Phase 2 `BatchedAttentionKernel` → Phase 3 `CachedGptStack.PrefillBatched`, wired into `CachedSlmSession.Prefill` (≥16-token GPT-2 prompts). The win came from head-coarse parallelism (the per-head wiring was 3.8× slower). Quant (Q4_K_M) batched prefill is the follow-on — Phase 1 is F32 only. Details in the Prefill section below.
- **(D) — surfaced by (A), 2 levers DONE — make decode bandwidth-bound.** llama.cpp got 2.85× from FP16→Q4 (bandwidth-bound); Overfit got ~1.0× (overhead-bound). **Done #1: GQA K/V-once** — K/V projection was recomputed once per Q head (8× for Qwen 16Q/2KV) instead of once per KV group; +24 % (13.85 → 17.2 tok/s), cut wasted K/V weight-read bandwidth 8×. **Done #2: fuse-quantize** — `hidden` was re-quantized to Q8_K per head per projection (~20×/layer); now quantized once per layer in `CachedMultiHeadAttention`, shared read-only across heads via `Q4K/Q6KDotKernel.ProjectPreQuantized`; **+~1.5 % (17.2 → 17.5 tok/s)** — small, as predicted (quantize is ~0.04 % of matmul arithmetic), but consistent. Both bit-identical (same 24-token greedy sequence before/after) + 680/0/68. Gap to llama.cpp now ~1.6× (was ~2.0×). **Tried #3: VNNI `vpdpbusd` — REVERTED 2026-05-21.** Replaced the AVX2 `vpmaddubsw`+`vpmaddwd` dot with one `vpdpbusd` (`AvxVnni.MultiplyWideningAndAdd`) in both Q4_K/Q6_K kernels + deferred-horizontal-sum; box has `AvxVnni`/`AVX512BW` (verified via bench `caps` mode). Parity 8/8 green. **Same-state A/B: AVX2 ≈ VNNI ≈ 19.1 tok/s — ~0 gain, reverted.** Lesson: after #1+#2 the decode is **memory-bandwidth-bound, not ALU-bound** — fusing the dot saves cycles already hidden behind weight-read latency. (The earlier "17.5 baseline" was a slower thermal state; same-state both ~19.) **The real remaining lever is MEMORY, not ALU:** llama.cpp reads 3.2 GB (mmap), Overfit 4.4 GB committed — the ~1.2 GB extra read per token *is* the speed gap. Levers: mmap-style weight loading instead of committed heap, drop F32 duplication (embeddings/norms), tighter weight layout. GEMV-unroll / core-util profiling are secondary now (ALU isn't the bottleneck). This redirects D from kernel-ALU to memory-bytes-per-token.

  **Profiled 2026-05-21 (thread-scaling + effective-BW).** Decode tok/s vs workers
  (`OVERFIT_PARALLEL_WORKERS`): 1→4.14, 2→8.04, 4→13.86, 8→**19.52**, 16→19.53,
  32→18.90. **Plateaus hard at 8 threads** → memory-bandwidth-bound confirmed
  (more cores starve on RAM). Effective BW: ~2.0 GB streamed/token × 19.5 ≈
  **~39 GB/s (~55 % of DDR5 peak); llama.cpp ~55 GB/s (~80 %)**. Same bytes, same
  RAM ceiling — llama.cpp just extracts more effective bandwidth. **Root cause
  (confirmed by reading llama.cpp `llama-model`): attention weight layout.**
  llama.cpp keeps `attn_q/k/v/o.weight` as full per-projection matrices and does
  ONE big contiguous `mul_mat` per projection (all heads), reshaping to heads only
  inside SDPA. Overfit splits per head (`WqHeads[h]` etc.) → nHeads small
  fragmented GEMVs → prefetcher starved → ~55 % BW. **THE decode lever now:
  fuse per-head attention into full-matrix contiguous matmuls** (Wq/Wk/Wv/Wo as
  full tensors, one streaming K-quant GEMV each, reshape to heads only for SDPA).
  Bonus: full Wo becomes Q6_K-able (contraction dModel, vs per-head headDim<256
  forcing Q8). Est. 55 %→~75 % BW ≈ ~26 tok/s (near llama.cpp's 27.5). **Big
  refactor** — the per-head structure is baked into K/V-once AND Stage-3 per-head
  LoRA, so fusion ripples into the LoRA target design. Deliberate sprint, not a
  tail-end change. (token_embd-as-F32 = the 1.2 GB RAM gap, but it's a per-token
  lookup not streamed → RAM-axis only, does NOT help decode speed.)

  **RAM AXIS CLOSED 2026-05-21.** Investigating token_embd found the embedding was
  held **twice** — engine-owned `TensorStorage` + a per-session `ToArray()` F32
  copy in `CachedLlamaSession`. Dropped the copy (session now references the
  engine storage, sliced per-token; consistent with the zero-copy weight design).
  **RAM 4.39 → 3.21 GB (−1.18 GB) = parity with LLamaSharp's 3.20 GB.** Decode
  unchanged (17.6, as expected — embedding is a lookup, not streamed); bit-identical
  (same 24-token greedy seq); 683/0/68. token_embd→Q6_K could shave another ~1 GB
  (3.2→~2.2) but parity is the milestone and it needs a dequant-row lookup path —
  deferred. **Only the decode-SPEED axis remains** (the attention-fusion lever above).

### Quantized batched prefill (TTFT lever — 2026-05-26, Phase 1 DONE)

The catchable llama.cpp gap that ISN'T the off-moat decode race: for Qwen/Llama/Mixtral, **prefill is
single-token** (`CachedLlamaSession.Prefill` loops `DecodeToken` per prompt token) — batched prefill
(3.48× TTFT) only exists for the F32/GPT-2 path. So a long prompt's time-to-first-token = N × single
decode, re-reading ALL weights N times; llama.cpp batches it. Closing this is on-moat (chat
responsiveness) and — unlike attention-fusion — does NOT touch the per-head LoRA design.

- **Phase 1 DONE — batched quant projection kernels.** `Q8DotKernel.ProjectBatched` +
  `Q4KDotKernel.ProjectBatched` + `Q6KDotKernel.ProjectBatched`: quantize all N activation rows once,
  then split the output loop over `OverfitParallelFor` with the **rows loop innermost** so each weight
  output row is read from DRAM once and reused (cache-hot) across all N dots — cuts weight byte-traffic
  ~N× (prefill is weight-bandwidth-bound). Bit-identical to N× single-token `Project` (parity tests:
  `Q8/Q4K/Q6K DotKernelTests.ProjectBatched_IsBitIdenticalToPerRowProject`, rows ∈ {1,2,3,7,16}). 888/0.
- **Phase 2a DONE — batched SwiGLU FFN dispatch.** `CachedFeedForwardBlock.DecodeSwiGluBatchedDispatched`
  + `ProjectBatchedDispatched` (Q6_K/Q4_K/Q8_0/F32) run gate/up/down as batched projections over N rows
  — the FFN is the dominant weight read (~7× the attention weights for Qwen-3B), so this captures most of
  the prefill weight-byte amortisation. Bit-identical to N× `DecodeSwiGluDispatched`
  (`CachedFeedForwardBlockBatchedTests.DecodeSwiGluBatchedDispatched_IsBitIdentical_To_PerRowSwiGlu`).
- **Phase 2b DONE — batched attention (quant).** `CachedMultiHeadAttention.DecodeBatchedQuant`: KV
  groups processed sequentially (each projection parallelises internally — no nested parallelism), per
  group batched K/V projection (`BatchedQuantProjection.Dispatch` — Q6_K/Q4_K/Q8_0/F32) + per-row RoPE +
  cache writes (K/V-once for GQA); per Q head batched Q projection + per-row RoPE + the proven causal
  `BatchedAttentionKernel` (RoPE-agnostic) + batched O projection. `BatchedQuantProjection` extracted as
  the shared batched-projection dispatch (FFN reuses it).
- **Phase 2c + 3 DONE — wired + measured.** `CachedTransformerBlock.DecodeBatchedQuant` (per-row RMSNorm
  + 2b + 2a, dense SwiGLU only) → `CachedGptStack.PrefillBatchedQuant` → `CachedLlamaSession.Prefill`
  takes the batched path for prompts ≥16 tokens (dense, non-sliding, fits cache; MoE/sliding fall back).
  **Parity: BIT-IDENTICAL to the single-token loop on real Qwen2.5-3B Q4_K_M** (maxAbsLogitDiff = 0,
  same argmax — `BatchedPrefillParityTests`, RMSNorm+RoPE+GQA 16:2+SwiGLU+mixed-K-quant). **TTFT win:
  256-token prompt 12 725 ms → 5 309 ms = 2.40×** (best-of-3, same model). This closes the prefill/TTFT
  gap to llama.cpp (which batches prefill) on the chat-responsiveness axis — the on-moat catch.
- **MoE batched prefill — DONE 2026-05-26.** `MoeFeedForwardBlock.DecodeBatched` (group rows by expert →
  gather → batched expert SwiGLU → scatter) + `Qwen2MoeFeedForwardBlock.DecodeBatched` (batched shared
  expert + per-row sigmoid gate), dispatched from `CachedTransformerBlock.DecodeBatchedQuant` on
  `weights.IsMoe`; session eligibility no longer excludes MoE. **Investigation note (empirical rigor):**
  the first cut accumulated each row's experts in *expert-index* order → real-Qwen-MoE Q8_0 end-to-end
  diverged 0.60 (argmax still matched) — a routing-flip cascade (the reorder perturbs the residual
  stream → a borderline top-k decision flips in a later layer). An isolated block-level F32 parity test
  localised it as NOT a block bug (<1e-4). Fix: stash each expert's output at the row's top-k SLOT and
  sum per-row in top-k order (= single-token order) → now **BIT-IDENTICAL** end-to-end (maxAbsLogitDiff
  = 0 on real Qwen1.5-MoE Q8_0), validated by `BatchedPrefillParityTests` + block-level
  `Qwen2MoeFeedForwardBlockTests.DecodeBatched_MatchesPerRowDecode` (exact). Mixtral (routed-only) reuses
  the same path.
  Remaining (optional): F32/RoPE batched attention could now also route here (the old `DecodeBatched`
  F32-only restriction is superseded for the Llama path); Phase-1-style attention fusion for decode
  (separate, off-moat).

### Speculative decode (prompt-lookup) — DONE 2026-05-26 (decode-throughput lever)

The only axis where llama.cpp still led was single-stream decode (~1.6×). Closed it on repetitive text
WITHOUT chasing SIMD: greedy prompt-lookup speculative decoding, reusing the new batched kernels for the
verify. `PromptLookupDrafter` (n-gram match of the suffix against earlier context → propose the
continuation, no draft model, zero extra RAM) + `CachedLlamaSession.GenerateSpeculative` (embed
[t0, draft…] → ONE batched verify forward → greedy-accept the longest matching prefix; `KeyValueCache.
TruncateTo` rolls back rejected drafts). Output is **BIT-IDENTICAL to greedy single-token** (greedy
verification only accepts what greedy would emit) — validated on real Qwen2.5-3B Q4_K_M
(`SpeculativeDecodeParityTests`, identical 40-token sequence + multi-commit confirmed).
**Investigation (empirical rigor):** first cut measured **1.01×** — the batched stack amortised but I
LM-headed each draft row separately, re-reading the huge LM-head weights N× and cancelling the win.
Fix: `CachedGptStack.ProjectLogitsBatched` (head read once for all rows). Then on a genuinely-echoing
tokenised prompt: **avg 3.45 tokens/step accepted, decode 18.1 → 23.1 tok/s = 1.27×** (vs llama.cpp
27.5 → gap ~1.6×→~1.2× on such text). Honest scope: the win is workload-dependent — high on repetitive /
structured / context-quoting output (code, JSON, agentic — the moat), ~1× on novel text (drafts miss).

**Sampling-correct speculative — DONE 2026-05-26.** Generalised greedy → any sampler (temp / top-k /
top-p / min-p). The prompt-lookup draft is a point-mass proposal, so speculative-sampling reduces to:
accept draft `d` w.p. `p(d)` under the sampler's target distribution, else resample the renormalised
residual `norm(max(0, p − e_d))` — output is distributed EXACTLY as a direct draw from `p` (greedy is
the T→0 special case ⇒ still bit-identical). `TokenSampler.ComputeProbabilities` exposes the sampler's
exact post-transform distribution (shared `SelectSurvivors` core with `Sample`); `SpeculativeSampler.
AcceptOrResample`/`Sample` is the rejection core; `CachedLlamaSession.GenerateSpeculative(in
SamplingOptions, …)` threads it (the correction/bonus token is forwarded so cache + `_logits` stay
consistent — committed = t0 + accepted + 1, ≤ maxDraft+2). Validated: `SpeculativeSamplerTests`
(statistical — empirical output distribution matches the target for ANY draft, incl. the greedy
point-mass) + the greedy-reduction parity (`Speculative_ProducesIdenticalSequence`, bit-identical on
real Qwen-3B). The greedy overload `GenerateSpeculative(history, committed, maxDraft…)` delegates with
`SamplingOptions.Greedy`.

### Prefix / system-prompt KV reuse — DONE 2026-05-26 (agentic multi-turn / multi-request TTFT)

Prefill a fixed prefix (system prompt, few-shot examples, a quoted document) ONCE, snapshot its KV, then
restore it into a session per request — skipping the prefix forward pass entirely (a memcpy, not a
re-encode). `KeyValueCache.Snapshot()` captures the live region <c>[0, CurrentLength)</c> compactly
(per-(layer,head) run, prefix-sized not full-cache); `RestoreFrom` copies it back + restores
`CurrentLength`/`BasePosition`; surfaced as `CachedLlamaSession.SavePrefix()` / `RestorePrefix(snapshot)`.
Restoring a prefix + appending a turn is **bit-identical** to prefilling prefix+turn together (causal —
the prefix never attends to the turn), validated on real Qwen2.5-3B across two different requests reusing
one snapshot (`PrefixKvCacheParityTests`). On-moat: the heavy cost in agentic / chat is re-encoding a long
fixed system prompt on every turn — now a O(prefix) memcpy. (From the "adopt from GPT-chat stacks" scan;
RoPE-scaling NTK/YaRN for >trained-context and flash-attention online-softmax for long-context memory
remain on that list.)

### Gradient checkpointing (TRAINING memory) — DONE 2026-05-26

Measured first (`CnnTrainingMemoryDiagnostics`): the tape arena's **live activations** dominate training RAM
(165 MB vs 14 MB im2col on a wide conv; arena cap 191 MB) — NOT im2col (corrected an earlier wrong guess).
Lever = gradient checkpointing. `ComputationGraph.Checkpoint(segment, input, subArenaElements)` runs a
segment without keeping its activations (forward in a throwaway sub-graph, keeps only input+output), then
recomputes them in backward (`OpCode.Checkpoint` + `CheckpointBackward`, segment stored in a graph-side list
indexed by `op.I1`). Wired two ways: **`CheckpointedModule : IModule`** decorator (drop into a `Sequential`
to checkpoint heavy segments) and **`GPT1Model(config, checkpointBlocks: true)`** (each transformer block
checkpointed — the high-value target: activations × batch × seq). Bit-close to non-checkpointed + lower
arena high-water, validated on MLP, a deep MLP, GPT-1 (6 blocks: logits + block-0 grad match), and a
Sequential (`CheckpointParityTests`, 4 fast). **Measured on a larger GPT-1 (12L, d=512, dFF=2048, b=2,
t=128): live-activation arena 438.5 MB → 18.3 MB = 24× less, 420 MB saved** (`Checkpoint_Gpt1_MemorySavings`,
[LongFact]). Pure "train deeper / longer-seq / bigger-batch on the same
RAM" = moat. Follow-ons: pool the sub-graph arena (currently fresh per call); CNN im2col is ~10× smaller
(low priority).

---

## Research inputs (papers reviewed 2026-05-21)

Four external papers assessed for transfer into Overfit. Verdicts honest — most
value was validation/guidance, not drop-in algorithms.

- **arXiv:2406.09384 — *Rehearsal-free Continual Learning with Pretrained Models*** &
  **NeurIPS 2024 — *A Practitioner's Guide to Continual Multimodal Pretraining* (FoMo-in-Flux)**.
  Both: adapt a frozen base over time with PEFT (LoRA) without catastrophic
  forgetting; finding — simple LoRA + a sane data mixture is competitive with
  complex continual-learning machinery. **Used:** validates the per-deployment LoRA
  track (don't over-engineer) + motivated **rehearsal-lite** (`Gpt1LoRAFineTuner`
  `rehearsalCorpus`/`rehearsalFraction`, shipped — base-regime forgetting 9.09 → 0.012
  in a test). *Caveat:* both are vision/multimodal; transfer to GPT-on-metrics is
  conceptual, not literal.
- **MASCOTS'16 — *Gaussian Process for Urban Environmental Sensor Networks***.
  Correlated, diurnally-periodic multi-sensor data = our K8s metrics. **Used:**
  motivates a **GP (or cheaper EWMA/z-score) baseline** to rigorously benchmark the
  GPT anomaly detector — design sketch in `docs/gp-anomaly-baseline.md`. Deferred
  (separate experiment, not a product feature).
- **LinkedIn PDF — *Maximum Accuracy Computing* (Fourier, self-published)**. Claims
  "40× more accurate than deep learning"; grandiose framing, no peer review, demo on
  a personal site. **Rejected** — extraordinary claims without evidence; building on
  it risks credibility. The neutral kernel (real-FFT features for periodic signals)
  is standard DSP we don't need from this source.
- **arXiv:2605.27494 — *Grounded Cache Routing for RAG: When Is It Safe to Reuse an
  Answer?*** (Shah, Duke, 2026-05-26). Output-level semantic answer cache with **four
  safety gates** — query similarity, retrieved-evidence overlap, source-version
  validity, **lexical support** (cached answer's tokens present in fresh evidence; the
  paper's per-gate ablation isolates this gate as load-bearing). Stress-tested on
  12k real Qwen2.5-7B generations: USR (unsafe-served rate) **0% on HotpotQA** (naive:
  15-35%), **1.5% on mtRAG document-drift** (naive: 51.5% → 34× reduction), p50
  latency **1.04-1.07× no-cache baseline** (≈zero overhead). **Verdict — accept as
  post-launch feature.** Overfit has all primitives in place (`BertEncoder` cosine,
  `VectorStore`, `WordPieceTokenizer` for lexical overlap; need to add per-entry doc-id
  traceability + version hash to the store). Algorithmic, hardware-neutral (paper's
  vLLM/GPU setup is incidental). Strongest fit with the regulated-industries / EU-AI-Act
  positioning (USR is a measurable safety metric, not just a speed metric). Task in
  the [Medium-term Features](#features) section.

## Recently completed (chronological, newest first)

- **OCR track — CRNN + CTC, end to end (2026-05-27).** New reusable capability: train image→text models in pure C#.
  - `Ops/CtcLoss.cs` — CTC negative-log-likelihood (Graves 2006) forward-backward in log-space + `softmax−posterior` gradient (finite-difference-verified). `Ops/CtcDecoder.cs` — greedy best-path + **CTC prefix beam search** (Hannun 2014) + **LM-rescored beam** via `Ops/ICtcLanguageModel.cs`. `Ops/NGramCtcLanguageModel.cs` — ready-made trainable label n-gram (add-k smoothing + back-off).
  - `Autograd` op **`TransposeLastTwo`** (new `OpCode`, fwd/bwd + facade) — the conv→sequence "map-to-sequence" primitive.
  - `DeepLearning/Crnn.cs` — **facade** `new Crnn(height, width, classCount)`: `ConvLayer → map-to-sequence → LstmLayer → Linear`; `CreateInput`/`Forward`→[T,C]/`ComputeCtcLoss`/`Recognize` (greedy / beam / LM-beam). `IModule`.
  - Demos ([LongFact]): digits (loss→0.006, 8/8) and **lexicon words + n-gram LM** (greedy 17/24 → **LM-beam 24/24**; the LM fixes CREEN→GREEN, BRD→BIRD, …). Unit tests: CtcLoss (FD gradient), CtcDecoder (beam>greedy textbook case + LM flip), Crnn geometry, TransposeLastTwo (exact), n-gram LM.
- **Data-parallel training — public API (2026-05-27).** `Training/DataParallelTrainer` (model-agnostic all-reduce / grad-norm clip / optimizer step / weight broadcast over N `Parameter` replicas, gradients averaged) + `DataParallelSession`/`DataParallelReplica` turnkey builder + `DataParallelLearningRate` (√N / linear scaling). Key fix: `OverfitParallelFor.SuppressParallelismOnCurrentThread` (per-thread inline opt-out) kills the N×cores oversubscription + pool-lock contention — measured **2.3× → 6×** throughput (24 workers, TinyShakespeare GPT-1). TinyShakespeare demo refactored onto it.
- **Gradient checkpointing (2026-05-26).** `ComputationGraph.Checkpoint` runs a segment in a throwaway sub-graph (keeps only input+output, recomputes in backward) + `CheckpointedModule : IModule` + `GPT1Model(checkpointBlocks:true)`. **Measured 24× live-activation arena cut** (438.5 → 18.3 MB on a 12L GPT-1) — "train deeper on the same RAM". Measurement showed the activation arena (not im2col) dominates training RAM.
- **MoE inference verified coherent (2026-05-27).** Re-ran full 24L Qwen1.5-MoE Q8_0 end-to-end → "capital of France?" → "Paris". The `norm_topk_prob` arch-aware fix holds; Qwen-MoE + Mixtral both coherent in pure C#. (Earlier "incoherent" resume note was stale.)
- **`PooledArray<T>` + `Array.Copy` ban (2026-05-26).** `Runtime/PooledArray.cs` — zero-cost `using` wrapper over `ArrayPool` (benchmarked); swept 12 loader try/finally trios onto it. `Array.Copy` added to `BannedSymbols.txt` (4 overloads) → 10 sites converted to `Span.CopyTo` (benchmarked ≥, memmove-safe). _Note 2026-05-29: `PooledArray<T>` was a duplicate of `PooledBuffer<T>` and has been deleted — its callsites migrated to `PooledBuffer<T>(n, clearMemory: false)`. See [[feedback-overfit-pool]]._
- **Q4_K_M parity "bug" resolved — was a test bug (2026-05-26).** Native Q4_K_M decode is correct; `GgufQ4KMParityTests` wrongly compared 4-bit-native logits vs the 16-bit FP16 file demanding exact top-1. Fixed to compare vs F32-dequant of the same file (near-tie tolerant). Isolation ladder (mmap≡copy, kernels 0.36%, head-splits 0.3–1.3%) ruled out every code path.
- **Fuse-quantize decode optimization** (2026-05-20). The attention `hidden` row was re-quantized to Q8_K once per head per Q/K/V projection (~20×/layer for Qwen). Now `CachedMultiHeadAttention` quantizes it once per layer (when attention is K-quant) into a shared read-only buffer; heads' Q/K/V projections consume it via new `Q4KDotKernel`/`Q6KDotKernel.ProjectPreQuantized` (the `Project` core minus the quantize pass). Wo keeps self-quantizing (its input is the per-head attention output, not `hidden`). **Qwen2.5-3B Q4_K_M 17.2 → 17.5 tok/s (+~1.5 %)** — small as predicted (quantize is ~0.04 % of matmul arithmetic) but consistent across runs. Bit-identical (same 24-token greedy sequence) + 680/0/68 `-c Release`.
- **GQA K/V-once decode optimization** (2026-05-20). Under grouped-query attention every Q head in a KV group shares one K/V weight set and one cache slot, but `CachedMultiHeadAttention.DecodeKvGroup` was recomputing the K and V projection (+ RoPE + cache write) once per Q head — 8× redundant for Qwen2.5-3B (16 Q / 2 KV). Added a `projectKv` flag to `CachedSingleHeadAttention.DecodeDispatched` so only each group's first head projects K/V; the rest read what it wrote. **Qwen2.5-3B Q4_K_M decode 13.85 → 17.2 tok/s (+24 %)** (also cuts wasted K/V weight-read bandwidth 8×). Bit-identical: same 24-token greedy sequence before/after on the real model (git before/after). 680 / 0 / 68 `-c Release`. Narrows the same-file gap to LLamaSharp from ~2.0× to ~1.6×.
- **Q4_K_M in-RAM decode path** (`docs/llamacpp-cpu-analysis.md` §5 step 3, sub-steps 3.1 → 3.4). `Q4KDotKernel` + `Q6KDotKernel` (AVX2 `vpmaddubsw` on the 4-bit nibbles / reassembled 6-bit quants + scalar fallbacks); `Q4KWeight` + `Q6KWeight` (output-major super-blocks); `DecodeWeight` widened to a 4-way tagged union `{F32 | Q8 | Q4_K | Q6_K}`; **per-weight dispatch** in `CachedFeedForwardBlock` / `CachedSingleHeadAttention` / `CachedGptStack.ProjectLogits` so a heterogeneous Q4_K_M file (Q4_K attn-Q/K/O + FFN gate/up, Q6_K FFN-down + attn-V + token-embd + output, Q8 per-head Wo) picks the right kernel per projection; native Q4_K + Q6_K loader reads. **Qwen2.5-3B-Instruct: load 1.4 s, decode 14.56 tok/s, steady RAM 4.40 GB** (vs Q8: 1.7 s / 13.28 tok/s / 5.85 GB; vs FP16-src: 7.1 s / 13.29 tok/s / 5.90 GB). 0 B/token preserved. Parity vs same-file F32 baseline: 29/32 top-1 match teacher-forced, worst swing 2.16 (every mismatch a near-tie). 680 / 0 / 68 `-c Release`.
- **Q8_0 in-RAM decode path** (`docs/llamacpp-cpu-analysis.md` §5 step 2). `Q8DotKernel` INT8 `vpmaddubsw` SIMD GEMV + `Q8Weight` output-major storage + `DecodeWeight` tagged handle. LM-head + FFN + per-head Q/K/V/O all Q8-resident. Native Q8_0 GGUF load (no dequant/re-quantize). Qwen-3B decode 4.01 → 13.38 tok/s (3.3×), steady RAM ~14.4 → 5.90 GB (−59 %), 32/32 greedy parity vs F32.
- **GGUF Q4_K + Q6_K dequantization** — `GgmlDequant` pure decoders + `GgufReader` streaming wrappers (stackalloc, zero managed allocations per call). 13 unit tests on synthetic blocks. Loader can now consume any `*.Q4_K_M.gguf` from Ollama/HuggingFace.
- **`[LongFact]` test convention** — 53 integration/diagnostic/training tests gated behind a custom xUnit attribute that skips by default. `dotnet test -c Release` now runs ~15 s instead of multi-minute.
- **`CachedTransformerBlock.Decode` argument validation** — explicit guards for input/output/FfnW1/FfnW2 lengths; surfaces caller bugs as `ArgumentException` instead of `IndexOutOfRangeException` mid-block.
- **Binary loader RAM optimization** — `Unpooled` `TensorStorage` for model weights + direct `ReadExactly` into destination span. Removes pool pow2-rounding overhead and intermediate scratch `byte[]`. **3B FP32: ~30 GB → ~14 GB peak load; matches file size exactly.**
- **Token-by-token streaming** — `CachedLlamaSession.StreamGenerate(StreamingOptions, CancellationToken)` returns `IAsyncEnumerable<int>` with stop-token / cache-full / cancellation termination.
- **Chat runtime: multi-turn template + string stops (2026-05-21)** — `ChatTemplate` (`LanguageModels/Chat/`) renders multi-turn `ChatMessage[]` to a prompt for ChatML / Llama-3 / Mistral, with `Detect(jinja)` fingerprinting a GGUF `tokenizer.chat_template` (no Jinja engine — AOT-hostile) and a ChatML fallback. `StopSequenceDetector` adds correct *string* stop sequences on top of the existing token stops — streaming-safe (holds back a trailing partial that could grow into a stop, never emits the stop marker). 16 unit tests (template detect/render for all 3 formats, stop split-across-pieces / partial / multi-stop / flush). **Validated end-to-end on real Qwen Q4_K_M** (`ChatIntegrationTests`, [LongFact]): detects the GGUF's 2509-char `chat_template` → ChatML, renders a system+user chat, tokenizes, generates, and assembles the stream through the stop detector (markers suppressed).
- **Turnkey `ChatSession` (2026-05-22)** — `ChatSession` (`LanguageModels/Chat/`) wraps any `ISlmSession` + `ITokenizer` + `ChatTemplate`: owns conversation history, and each `Send(userMessage, options, onText)` renders the whole history, prefills, generates, and assembles the reply applying both the EOS token stop and string stops (`StopSequenceDetector`) — incremental detokenize holds back a trailing partial codepoint rather than emit garbage. 3 fake-backed unit tests (history accumulation + template-rendered prefill, EOS stop, string-stop truncation + streaming).
- **Sliding-window KV eviction (2026-05-22, RoPE models)** — `KeyValueCache.Evict(count)` drops the oldest tokens by shifting every (layer,head) K/V block down one contiguous memmove, shrinking `CurrentLength` and growing `BasePosition`; retained K/V are NOT re-rotated — RoPE scores depend only on the relative offset, which the climbing `BasePosition` preserves (the new token rotates at `slot + BasePosition`, the true absolute position). `CachedLlamaSession.EnableSlidingWindow(evictBlock)` (opt-in, RoPE-only — learned-absolute-position GPT-2 can't slide) evicts instead of throwing once the cache fills, so chats run unbounded over a rolling context. Tests: `KeyValueCacheEvictTests` (5 — slot shift, BasePosition, reset, invalid count) + `SlidingWindowTests` ([LongFact], real Qwen — enabling sliding is bit-identical before the cache fills; generates 40 tokens over a 16-slot cache with eviction triggered while non-sliding throws). NOTE: sliding window is intentionally NOT equivalent to recomputing on a truncated context (retained hidden states encode since-evicted tokens — standard StreamingLLM behaviour). **Surfaced through the ergonomic API 2026-05-25:** `ISlmSession.SupportsSlidingWindow` + `EnableSlidingWindow(...)` (default-throw for non-RoPE sessions; real on `CachedLlamaSession`), and `new ChatSession(..., slidingWindow: true)` enables it and drops the "stop at MaxContextLength" loop guard so multi-turn chats roll instead of hitting the wall. Wiring test: `ChatSessionSlidingWindowTests` (3 fast — slides past context, stops without it, throws on unsupported sessions). **Chat-runtime gaps now closed (template, stops, turnkey session, eviction — end to end).**
- **Codebase hygiene (2026-05-22): one top-level type per file across the whole solution.** Audited + split 12 multi-type files (incl. the just-added `ChatTemplate`/`SafetensorsReader`/`SafetensorsSource`); `OfflineTrainingConfig.cs` (held `GptTrainingConfig` + `OfflineTrainingResult`, named after neither) renamed into per-type files. Solution builds clean, 740/0/72.
- **GGUF native loader** — `GgufLlamaLoader` reads GGUF files end-to-end without Python tooling. Supports F32/F16/BF16/Q8_0/Q4_K/Q6_K tensors, hand-rolled protobuf-free parser.
- **Native safetensors reader (2026-05-21)** — `SafetensorsReader` reads the HuggingFace `safetensors` format directly: 8-byte header length + `Utf8JsonReader`-parsed JSON header (reflection-free, AOT-clean — no `JsonSerializer`) + F32/F16/BF16 → F32 streaming dequant via `LoadF32`. Closes the last Python-dependent path (a raw HF repo with no GGUF variant). 7 unit tests over synthetic files (header parse, shape/dtype, F32/F16/BF16 round-trip, error paths).
- **Native GPT-2 safetensors loader (2026-05-21)** — `SafetensorsGpt2Loader.Load(path, Gpt2Config.Small)` builds a `GPT1Model` straight from a HF GPT-2 `model.safetensors`, no Python / no `convert_gpt2.py` / no intermediate `.bin`. C# port of the script's mapping (Conv1D `[in,out]` as-is, `c_attn [d,3d]` → per-head Q/K/V `[d,dHead]`, `c_proj [d,d]` → per-head `[dHead,d]`, `c_attn.bias` per-head split, LM head = `wte.T`); serialises into the exact byte stream `GPT1Model.Load` reads (order ≡ `GPT1Model.Save`), so ordering/shapes are guaranteed by the validated load path. Names resolve with/without `transformer.` prefix. Tests: a synthetic tiny GPT-2 (d=4/2-head/1-layer) with ramp-filled tensors proves Q/K/V/O split + bias split + `wte.T` placement against hand-computed expectations (independent of loader logic) + finite decode through `CachedGpt1ModelAdapter`; a `[LongFact]` asserts **bit-parity of `loader.Save()` vs `gpt2_small.bin`** when a real `model.safetensors` is present (resolves `$OVERFIT_MODEL_DIR` / `C:\gpt2\`, no-ops otherwise — real end-to-end parity needs that file, absent on the dev box for now).
- **Sharded safetensors repos (2026-05-21)** — `ISafetensorsSource` abstraction (single-file `SafetensorsReader` + multi-file `ShardedSafetensorsReader`); `SafetensorsSource.Open(pathOrDir)` factory auto-detects `model.safetensors.index.json` (sharded) vs `model.safetensors` (single). `ShardedSafetensorsReader` parses the index `weight_map` (`Utf8JsonReader`, reflection-free), opens each shard once, reads each tensor on demand from its owning shard (no shard fully materialised — low-RAM). `SafetensorsGpt2Loader` now takes `ISafetensorsSource` so it loads single or sharded transparently. Tests: shard merge + correct-shard read + missing-tensor error, and **a sharded GPT-2 loads byte-identically to the single-file model** (`ShardedSafetensorsReaderTests`). **Next:** Llama/Qwen safetensors loader (port `convert_llama.py` — RoPE/GQA/SwiGLU). (Note: the GPT-2 loader still round-trips weights through a full in-memory `.bin` stream — ~2× peak RAM at load; fine for GPT-2, worth streaming-into-params for larger models.)
- **LoRA adapter** — `LlamaLoRAAdapter` with Enable/Disable in-place weight injection. Zero-copy `TensorStorage` references — adapter updates visible to inference without re-binding.
- **GPT-2 Small inference + parity** — 124M params, KV-cache decode 0 B/token, 6.4× faster than naive O(N²). Top-10 logit overlap 10/10 vs PyTorch, maxAbsDiff 0.000107.
- **KV-cache runtime** — `CachedSlmInferenceEngine` + `CachedSlmSession`. `SingleHeadWeights` / `BlockWeights` / `StackWeights` hold zero-copy `TensorStorage` refs.
- **ONNX import** — 14 operators (Conv, Gemm, ReLU/Tanh/Sigmoid/Softmax, MaxPool, GlobalAveragePool, BatchNorm, Add, Reshape, Flatten, AveragePool, ReduceMean). Linear topology via `OnnxImporter`, DAG/skip connections via `OnnxGraphImporter`.
- **Autograd ownership (PR5)** — `Parameter` first-class type, `AutogradNodeOwnership` enum, `graph.Reset()` by ownership, optimizers on `IEnumerable<Parameter>`.
- **Sampling** — top-P with heap sort, repetition penalty (on `CachedSlmSession`), greedy.
- **PERF kernels** — `LinearKernels.ForwardBatched` weight-stationary outer product, hybrid threshold backward, `MaxPool` pool=2 SIMD fast path.

---

## Active track — Anomaly detector: synthetic-trained base → LoRA fine-tune

**Goal:** train a GPT anomaly model on synthetic K8s metrics, then LoRA-fine-tune
it on real production metrics. Plan: synthetic base → pull real metrics → LoRA adapt.

### Done this session

- **`OfflineTrainingJob` gradient bug — fixed.** `optimizer.ZeroGrad()` ran
  *between* `AggregateGradients` and `Step()`, wiping the just-aggregated master
  gradient → `Step()` applied only AdamW weight decay, so the model never learned
  (loss merely decayed toward `ln(VocabSize)`). Second bug: worker `Parameter.Grad`
  never zeroed (`BackwardFromGrad` accumulates) → workers summed gradient over all
  steps. Fix: removed the misplaced `ZeroGrad`, added per-worker grad zeroing in
  the `Parallel.For` body. Verified: Quick 500 steps 7.7→1.75 val loss.
- **Synthetic generator lift.** `Scripts/generate_k8s_metrics.py` rewritten with a
  shared per-pod AR(1) latent `load` driving cpu/rps/latency/throttle/queue/gc
  jointly. Old data was independent per-metric Gaussian noise → model hit an
  entropy floor (~1.08, plateau by step 800). New data has inter-metric correlation
  + temporal autocorrelation → loss keeps descending. CSV regenerated.
- **Prometheus parser — fixed.** `PrometheusMetricSource.ParseInstantResponse` and
  `PrometheusHistoricalSource.ParseRangeResponse` had their value-extraction bodies
  commented out (original used `EnumerateArray().ToArray()` — LINQ, banned in
  `Sources/Main`) → both sources silently returned empty lists. Rewrote LINQ-free
  (indexed `JsonElement`). Added `PrometheusParsingTests` + golden-JSON fixtures
  under `test_fixtures/prometheus/`.
- **GPT1 LoRA — Stage 1 (LM head), full loop closed.**
  - Risk PoC (`LoRAEffectiveWeightInjectionTests`): `graph.Linear` backprops
    through a *computed* weight node `W_eff = W_frozen + A@B` into A/B; numerically
    verified; an optimizer step over {A,B} leaves the base bit-identical.
  - `GPT1Model.LMHeadWeightProvider` — internal per-forward hook (null = production
    path, zero overhead).
  - `Gpt1LoRAFineTuner` — trains A/B with the base frozen (`Adam` over {A,B} only).
    Test: loss 2.26→0.0017, base bit-identical, `.bin` round-trips.
  - `Gpt1LoRAFile` — shared `.bin` format (magic "LORA", `LanguageModelHead` entry).
  - `Gpt1LoRAMergeAdapter` — inference-side weight-merge (`Enable`/`Disable`,
    idempotent, bit-reversible). Test proves the merge is visible to
    `CachedGpt1ModelAdapter` (the `GptAnomalyDetector` runtime) via the zero-copy
    `StackWeights._lmHead` ref: cached loss 4.14→0.006 on Enable, exact restore on
    Disable. No inference-kernel changes.

### Debt found (not yet addressed)

- **Prometheus auth (F3)** — `PrometheusMetricSourceConfig` / `...HistoricalSourceConfig`
  have no bearer-token / basic-auth field; auth-gated Prometheus/Thanos unreachable.
- **`MetricTokenizer` binning** — `error_rate`, `gc_pause_ratio`, `cpu_throttle_ratio`
  use linear `[0,1]` ranges for metrics that live in `[0, 0.05]` → ~95 % of the 64
  bins unused. Fix = tokenizer range change (log-scale / smaller Max) → needs retrain.
- **`OfflineTrainingJob` uses `System.Linq`** in `Sources/Main` — contradicts
  `BannedSymbols.txt` / CLAUDE.md. Compiles today; flag for cleanup.
- Diagnostics audit (earlier this session): `ArrayPoolEventSource` dead code,
  ~11 telemetry instruments declared but never recorded, `DiagnosticsRegressionTests`
  was vacuously green (F1/F4 fixed, F2/F3 outstanding).

### Anomaly + LoRA track — ALL ITEMS SHIPPED (was "resume here / pick one")

- **End-to-end integration test — ✅ DONE.** `Tests/Anomalies/GptAnomalyLoRAIntegrationTests.cs`,
  2 `[Fact]`s (green, ~1 s each): (1) LoRA trained on a regime lowers that regime's
  anomaly score through the detector, reversibly; (2) **`LoRA_AdaptedToRegime_StillFlagsInjectedAnomaly`**
  (added 2026-05-20) — after the adapter flattens the benign regime (adapted-normal
  ≈ 0.007 nats/token) an OOD injected snapshot still scores ≈ 20.4 (~2800× separation),
  proving adaptation lowers false positives without blinding the detector. Uses a
  random-init tiny model (no heavy base training — that's the separate item below).
- **Stage 2 — LoRA on FFN (W1/W2)** — DONE earlier (`Gpt1LoRAFfnTests.cs`).
- **Stage 3 — LoRA on attention Q/K/V/O per-head — ✅ DONE 2026-05-20.**
  `MultiHeadAttentionLayer` got per-head Q/K/V/O weight-provider hooks (mirroring
  the LM-head / FFN providers); `Gpt1LoRAFineTuner` fans out one A/B pair per head
  per targeted module per block; the `.bin` carries the per-entry head index;
  `Gpt1LoRAMergeAdapter` merges each delta into the matching per-head weight.
  `Gpt1LoRAAttentionTests.cs` (2 `[Fact]`): full `Attention` target = 16 adapters
  (2 blocks × 2 heads × Q/K/V/O), cached decode loss 2.97 → 0.02, exactly reversible;
  `Query`-only = 4 adapters (proves single-module per-head fan-out). All `LoRATargetModules`
  now supported by the fine-tuner.
- **Production base training — ✅ DONE 2026-05-21.** `OfflineTrainingJob` with
  `GptTrainingConfig.Production` (256d / 8 heads / 6 L, 10K steps, 8 data-parallel
  workers) on the 201 600-snapshot synthetic CSV → `k8s_anomaly_production.bin`
  (~19.8 MB). **Val loss 0.856** (~87 % below init), 1 h 03 m wall. Verified
  deployable: loads into `GptAnomalyDetector` (256d auto-detected) and discriminates
  — normal 6.02 vs anomaly 13.77, OOM anomaly correctly attributed to memory. The
  base's per-pod "normal" still carries residual surprise (6.02) because it's
  trained across all pods; that's exactly what per-regime LoRA (Stage 1/2/3) drives
  down for a specific deployment.
- **One-command product demo — ✅ DONE 2026-05-21.** `Demo/AnomalyConsoleDemo`
  now demonstrates the *whole* moat live in a single `dotnet run`, self-contained
  (trains a Quick base on the 201 600-row fixture CSV in ~17 s, no external file):
  **Phase 1 (base)** streams a benign regime + injects an incident; **Phase 2**
  fine-tunes an LM-head LoRA (rank 16, 300 steps, base frozen) on that pod's benign
  regime, merges it in place, and re-runs the same stream. Measured live (Quick base):
  benign "normal" **2.74 → 0.00** (false positives flattened), injected incident
  **11.78 → 24.38** (detection preserved and sharpened). This is the "adaptive
  per-deployment learning an inference-only engine can't do" story, shown not just
  described. Closes the "demoable in one command, documented honestly" goal below.
- **Production base validated end-to-end with per-pod LoRA — ✅ 2026-05-21, decision
  RESOLVED → Production.** Ran the demo's full before/after loop on the real
  `k8s_anomaly_production.bin` (256d/6L, via `--preset production --checkpoint`):
  the un-adapted base **scores the benign regime 5.68 — a false positive** (> the
  ⚠ 5.0 threshold), exactly because it's trained across all pods; a per-pod LM-head
  LoRA (rank 16, 300 steps) drives it to **0.00** while sharpening the injected
  incident **12.07 → 34.14**. This both (a) resolves the Medium-vs-Production base
  decision in Production's favour — it carries the cross-pod residual surprise that
  per-deployment LoRA is designed to remove — and (b) proves the moat on the actual
  deployable artifact, not a toy. Demo gained a `--preset quick|medium|production`
  flag so the loaded model's dims match the checkpoint. **Now regression-defended:**
  `GptAnomalyProductionLoRATests.ProductionBase_PerPodLoRA_FlattensBenignRegime_StillFlagsIncident`
  ([LongFact], auto-detects 256d, resolves the base from $OVERFIT_MODEL_DIR /
  test_fixtures / D:\, no-ops if absent) asserts benign < base & < 1.0 and incident
  > 5.0 with clear separation — bit-reproducible (5.68→0.00, 12.07→34.14, 13 s).

---

## (Historical) GPT-2 Small showcase — SUPERSEDED by the in-process agentic stack

**Superseded (2026-05-25):** the primary showcase is now the **in-process agentic stack** — RAG +
tool calling + structured output on a Qwen GGUF (see "Nearest plan (2026-05-25)" and `Demo/AgentDemo`).
Qwen / Llama / LoRA / quantization are **no longer deferred — they shipped**. Kept below for history.

**Why (at the time):** the parity claim ("top-10 overlap 10/10 vs PyTorch, maxAbsDiff 0.000107, 0 B / generated token, KV-cache decode") is already implemented and validated. Productizing this single story end-to-end (defended in CI, demoable in one command, documented honestly) is higher-value than chasing more model families. Qwen / Llama / LoRA / quantization work continues to live in the codebase but is **explicitly deferred** out of the current week's focus.

### This week

- [x] **GPT-2 parity diagnostics run on every `dotnet test`** — `Gpt2ImportParityDiagnostics`, `Gpt2ImportStageParityDiagnostics`, `Gpt2ImportAttentionParityDiagnostics` flipped from `[LongFact]` back to `[Fact]`. Sweep cost: +2 s. Headline claim now defended on every push.
- [x] **`Demo/Gpt2ConsoleDemo` project** — exists (`Demo/Gpt2ConsoleDemo`). (The launch showcase is now `Demo/AgentDemo`.)
- [x] **Fixture / model path resolver** — done: `OVERFIT_MODEL_DIR` env var is the resolution path across demos/tests.
- [ ] **GPT-2 generation benchmark** — BenchmarkDotNet: cold-start, prefill cost, per-token decode time, allocations. Confirm 0 B/token quantitatively for the README.
- [x] **README cleanup** — done: README/TECHNICAL/scenarios consolidated and updated through 2026-05-25 (agentic stack, mmap, benchmarks).

### Next (after the GPT-2 week)

- [x] **`Prefill()` vs `GenerateNextToken()` API split** — `CachedLlamaSession` and `CachedSlmSession` both expose `Reset()` (clear-only) + `Prefill(prompt)` + the legacy `Reset(prompt)` facade. Backwards compatible: every existing caller keeps working. Enables chat-history-style incremental context (system → user → assistant turns) without re-prefilling the prefix.
- **Prefill: multi-token batched matmul** — phased work toward 5-10× TTFT speedup. Same FLOPs as today, but weights loaded once for N tokens instead of N times; memory-bound for small models, so the speedup is large for short prompts and grows with prompt length.
  - [x] **Phase 0 — skip LM-head for non-final prompt tokens.** Split `CachedGptStack.Decode` into `DecodeWithoutLogits` (transformer blocks + final norm) + `ProjectLogits` (LM head). `Prefill` in both `CachedSlmSession` and `CachedLlamaSession` now calls `DecodeWithoutLogits` for tokens `0..N-2` and full `Decode` only for the last token. Saves ~27 % per-token cost on GPT-2 Small for N-1 of N tokens → ~25 % overall prefill speedup. Parity preserved (greedy output bit-identical to pre-split, demo test still 0 B / generated token).
  - [x] **Phase 1 — `BatchedProjectionKernel` (2026-05-21).** `[N×I] × [I×O] → [N×O]` allocation-free F32 GEMM, `Project` (sequential) + `ProjectParallel` (over output columns). Loop order output-tile → input → row: each weight tile loaded once and reused across all N rows → weight-read bandwidth amortised N×. Per-output-element accumulation order (input-ascending, `TensorPrimitives.MultiplyAdd`, `x==0` skip) identical to the single-token kernel, so **bit-identical to N× `SingleTokenProjectionKernel.Project`** — verified by `BatchedProjectionKernelTests` (6 cases: N=1/N>1, ±bias, outputSize>tile, GPT-2-scale, input zeros; both seq + parallel exact). Foundation for Q/K/V/O + FFN-W1/W2 batched prefill.
  - [x] **Phase 2 — `BatchedAttentionKernel` with causal mask (2026-05-21).** Scores N query positions against a shared K/V cache in one call; query `i` (absolute pos `basePos+i`, `basePos = cacheLength-rows`) attends `[0..basePos+i]` under the causal mask → reduces to one `ComputeSingleHead` with `sequenceLength = basePos+i+1`, so **bit-identical to per-query single-head**. `Compute` (sequential) + `ComputeParallel` (queries fan out over `OverfitParallelFor` — independent, disjoint output + per-query score-scratch rows, read-only shared K/V; not weight-bound so the win is core-parallelism, not bandwidth). `BatchedAttentionKernelTests`: 5 cases (fresh prefill basePos=0, prefix basePos>0, Qwen-scale head; seq + parallel exact).
  - [x] **Phase 3 — top-level batched stack pass** *(DONE 2026-05-21, ~3.5× TTFT)*. New `CachedGptStack.PrefillBatched(promptTokens, weights, cache, ...)` wires Phase 1+2 through the block batched paths. Scoped first to the F32 / GPT-2 path (standard LayerNorm, GeLU FFN, MHA, no RoPE); SwiGLU/RoPE/GQA/quantized are the follow-on (the quant path also needs a batched K-quant projection — Phase 1 is F32 only).
    - [x] **FFN slice — `CachedFeedForwardBlock.DecodeBatched` (2026-05-21).** Both projections via `BatchedProjectionKernel`, the SAME element-wise `ApplyActivation` across the whole `[N×dFF]` intermediate → bit-identical to N× `Decode`. `CachedFeedForwardBlockBatchedTests` (4 cases: GeLU + ReLU, N=1/N>1). FFN is ~⅔ of layer FLOPs — the biggest single batched win.
    - [x] **Attention slice — `CachedMultiHeadAttention.DecodeBatched` (2026-05-21).** Per head: batched Q/K/V projection (`BatchedProjectionKernel`), per-position cache writes, batched causal attention (`BatchedAttentionKernel` — query n attends `[0..basePos+n]`), batched O projection accumulated into output in head order. Bit-identical to N× `Decode` — `CachedMultiHeadAttentionBatchedTests` (4 cases, with attention bias, N=1/N>1). Scoped F32/MHA/no-RoPE (throws for quant/GQA/RoPE).
    - [x] **Stack orchestration (2026-05-21).** `CachedTransformerBlock.DecodeBatched` (per-row LN → batched MHA → residual → LN → batched FFN → residual, reusing the two verified batched blocks + `SingleTokenLayerNormKernel`) + `CachedGptStack.PrefillBatched` (layer loop + last-token final norm, mirroring `DecodeWithoutLogits`). **Stack-level parity: `PrefillBatched` last-token final-hidden bit-identical to the single-token `DecodeWithoutLogits` loop** — `CachedGptStackTests.PrefillBatched_LastToken_IsBitIdentical_To_SingleTokenLoop` (3 cases, 2 layers, random weights). Scoped F32/GPT-2.
    - [x] **Session delegation — SHIPPED (2026-05-21, after head-parallel fix).** Wired
      `CachedGpt1ModelAdapter.PrefillBatched` (embed N + advance cache + `PrefillBatched`
      + project last-token logits) and delegated from `CachedSlmSession.Prefill` for
      prompts ≥ `BatchedPrefillThreshold` (16; falls back to the single-token loop below
      that and for non-GPT-2 stacks via `SupportsBatchedPrefill`). Correctness: batched
      last-token logits bit-identical to the single-token loop (`CachedSlmPrefillBatchedTests`,
      3 cases). **TTFT on GPT-2-Small dims, 64-token prompt: 567 ms single-token vs 163 ms
      batched = 3.48× faster** (`Ttft_BatchedVsSingleToken_Gpt2SmallDims`, [LongFact]).
    - **The fix that flipped it (the key result).** First wiring measured **0.26×
      (≈3.8× SLOWER)**: the batched MHA fanned the per-head Q/K/V/O projections out as
      ~60 tiny `OverfitParallelFor` dispatches per layer (~720 for 12 layers), whose
      wake/sync overhead swamped the weight-reuse win. Restructured `DecodeBatched` to
      parallelise **over heads** — one `OverfitParallelFor.For(0, HeadCount, …)` dispatch
      per layer, each head doing its Q/K/V projection + attention + O projection
      sequentially into a per-head scratch band, then bands reduced into the output in
      head order (bit-identical). One dispatch/layer instead of ~60 → **0.26× → 3.48×, a
      ~13× swing**, same kernels, same math. **Lesson (carry forward):** batched prefill's
      win is real only when the parallelism granularity is coarse (over heads / big
      matmuls), never per-head — dispatch overhead dominates at fine granularity.
  - [ ] **Parity tests** for each phase: batched output ≡ N × single-token output for any prompt up to ContextLength. Final assertion: `Gpt2ImportParityDiagnostics` still green (full PyTorch parity through the batched path).
- [x] **LM-head hot-path audit (initial)** — confirmed `ProjectParallel` exists but is **dead code** (no call site); `Project` is what GPT-2/Qwen actually use. Wiring `ProjectParallel` into `CachedGptStack.Decode` was tested and reverted: `Parallel.For` allocates ~3 KB / call from task scheduling, which breaks the 0 B / generated token contract for only ~3 % per-token speedup at the GPT-2 Small scale. The 10× speedup in `LmHeadParallelBenchmark` is steady-state; per-token decode is dominated by the `Parallel.For` overhead.
- [ ] **LM head: allocation-free parallel matmul** — wire-up depends on a worker pool that does NOT allocate per call. Candidates: pre-spawned threads with lock-free queue / semaphore signaling, or unsafe manual partitioning over a fixed thread set. Constraint: ≤ 0 B / call. Payoff: most of the ~3.8 ms LM-head matmul on a 32-core box. Single largest remaining lever for GPT-2 tokens/sec.
- [ ] **`Gpt2.Load(...) / CreateSession()` API sugar** — `new GPT1Model(Gpt2Config.Small)` is technically correct but semantically misleading. A typed entry point reads cleaner in the demo.
- [x] **Stabilize `GPT1_GradientCheck_BackwardIsCorrect`** — pre-fix: model weights randomly initialized + tight `relErr < 10 %` threshold → ~1-in-3 sweeps red. Fix: seeded weight init (deterministic per run) + mixed tolerance (`relErr < 50 %` OR `absErr < 5e-4`) that accepts the inherent FP32 finite-diff noise floor (~ loss_precision / (2 × eps) ≈ 2.5e-3 per gradient). Test still catches sign errors, factor-of-2 backward bugs, and zero-vs-non-zero regressions. 5/5 full sweeps green post-fix.

---

## Deferred — Qwen / Llama / quantization track

These are working in the codebase but **outside the current GPT-2 focus week.** Listed for visibility, not for prioritization.

### Slot 2b — quantized weight storage at inference — ✅ DONE (Q8_0 + Q4_K_M)

**Original gap:** Q4_K_M loader existed (decodes from disk) but dequantized
everything to FP32 on load — a 2 GB Q4_K_M file produced ~14 GB FP32 weights in
RAM. The "3B in 4 GB RAM" payoff required keeping weights quantized in RAM and
dequantizing per-block during matmul.

**Closed.** Two complementary in-RAM quant paths now ship — full design and
sub-step record in `docs/llamacpp-cpu-analysis.md` §5 steps 2 + 3:

- **Q8_0** (step 2): `Q8DotKernel` (symmetric F32→Q8 quantizer + INT8
  `vpmaddubsw` SIMD dot + sequential & parallel GEMV), `Q8Weight` (output-major
  Q8 weight storage), `DecodeWeight` (tagged precision-agnostic weight handle).
  LM-head + FFN gate/up/down + per-head attention Q/K/V/O all Q8-resident — the
  full decode matmul path. Loader reads native `Q8_0` blocks straight from a
  Q8_0 GGUF — no dequant/re-quantize on the loaded path.
- **Q4_K_M** (step 3): `Q4KDotKernel` + `Q6KDotKernel` (Q4_K × Q8_K and
  Q6_K × Q8_K AVX2 kernels with scalar fallbacks), `Q4KWeight` + `Q6KWeight`
  (output-major super-blocks). `DecodeWeight` widened to a 4-way tagged union
  `{F32 | Q8 | Q4_K | Q6_K}`; the decode blocks (FFN, single-head attention,
  LM-head) refactored to **per-weight dispatch** so a heterogeneous Q4_K_M file
  (`attn_q/k/o` + `ffn_gate/up` Q4_K, `ffn_down` + `attn_v` + `token_embd` +
  `output` Q6_K, per-head `Wo` Q8 — its headDim=128 contraction is below the
  256-element K-quant super-block) picks the right kernel per projection.
  Loader: native Q4_K + Q6_K reads + per-head contiguous byte slices.

**Three-way A/B across Overfit's own formats** (dev box, best-of-3, 24 timed
tokens after 4 warm-up, single stream):

| Format    | Load   | Decode      | Steady RAM |
|-----------|--------|-------------|-----------:|
| FP16-src  |  7.1 s | 13.29 tok/s |   5 902 MB |
| Q8_0      |  1.7 s | 13.28 tok/s |   5 847 MB |
| **Q4_K_M**| **1.4 s** | **14.56 tok/s** | **4 396 MB** |

Within Overfit, Q4_K_M is the best format — RAM + load win. *(Table is pre-fix;
the GQA K/V-once fix below lifts Q4_K_M decode to **17.2 tok/s** and would lift
Q8 similarly — not re-measured.)* Parity: Q8 32/32 (2.5); Q4_K_M 29/32, worst
swing 2.16 (3.4, teacher-forced vs same-file F32 baseline). Zero allocations per
decoded token preserved (`Demo_Gpt2Small_KvCacheDecode_AllocatesZeroBytesPerToken`).
680 / 0 / 68 `-c Release`.

**Same-file A/B vs LLamaSharp (2026-05-20).** LLamaSharp 0.27.0 on the *same*
`qwen.q4km.gguf`: **27.5 tok/s @ 3.2 GB** vs Overfit **17.2 tok/s** @ 4.4 GB
(after GQA K/V-once) — **llama.cpp ~1.6× faster, 27 % less RAM on equal footing**
(was ~2.0× before the fix). (Earlier "1.51× faster" was Overfit-Q4_K_M vs
LLamaSharp-FP16 — retracted.) Diagnostic: FP16→Q4_K_M sped llama.cpp 2.85×,
Overfit only ~1.0× → Overfit decode was overhead-bound, not bandwidth-bound.
First lever (GQA K/V-once: project K/V once per KV group, not per Q head) gave
+24 %, bit-identical. Defensible edge: 1 B vs 21 KB/token alloc, pure-managed,
AOT-clean, no native dep. Numbers: `overfit-bench/RESULTS.md`.

**Re-bench 2026-05-26 (after the MoE + GGUF-tokenizer tracks).** Same harness/file/prompt,
`overfit-bench -- qwen.q4km.gguf`: **18.92 tok/s, 1 B/token, load 0.7 s, steady working set 259 MB**
(36L d=2048, logits finite). Takeaways: (1) **no decode regression** — 18.9 ≈ the 19.5 recorded same-state
(thermal noise); the MoE/tokenizer work didn't touch the dense hot path. (2) The zero-alloc contract and
the ~1.45× speed gap to llama.cpp (27.5) both hold — attention-fusion remains the only identified speed
lever. (3) **The RAM story flipped — now measured apples-to-apples.** Restored the LLamaSharp `llama` mode in
`overfit-bench` (NuGet 0.27.0 + CPU backend) and measured BOTH via `Process.WorkingSet64` at the same two
lifecycle points on the same file:

| | Overfit | llama.cpp (LLamaSharp 0.27.0) |
|---|---|---|
| decode | 18.2 tok/s | 28.98 tok/s (**~1.6× gap**, holds) |
| working set, after load | 259 MB | 2626 MB |
| **working set, after decode** | **2037 MB** | **3203 MB** |
| load | 0.7 s | 1.2 s |

⚠ Honesty correction: the flashy "259 MB" is lazy-mmap **before pages are touched** — after a real decode
the weight pages page in and Overfit's working set rises to **2037 MB**. THAT is the fair "RAM to actually
run" number. Measured identically, Overfit uses **~1.16 GB (~36 %) LESS working set than llama.cpp**
(2037 vs 3203) — which *flips* the old "RAM parity (3.2 vs 3.2)" finding into a real win (mmap +
quantized-embedding-lookup did it, both landed 2026-05-25 after that parity measurement). Net standing vs
llama.cpp same-file, same-measurement: **Overfit −36 % working set, +faster load, 1 B/token; −1.6× decode
speed** (attention-fusion remains the only speed lever). The bench now keeps both modes:
`overfit-bench -- overfit|llama qwen.q4km.gguf`.

**Remaining (separate tracks, not gated on this):**
- Make decode further bandwidth-bound (resume-point option D, 1st lever done):
  fuse activation-quantize per-group, tighter GEMV unrolling, core-util profiling.
- Step 4 — tiled prefill GEMM (`batch > 1`). Helps TTFT + batched training, not decode.
- Step 5 — work-stealing chunk counter. Marginal for uniform GEMV; opportunistic.

### Slot 2c — FP16-resident weights — ATTEMPTED & REVERTED (May 2026)

Post-mortem of a refuted experiment. Kept as a record so the idea is not retried.

**Hypothesis:** `GgufLlamaLoader` up-casts FP16 GGUF weights to F32 at load, so
decode streams 2× the bytes. Keeping weights FP16-resident (`Half`) and widening
F16→F32 in the matmul should give ~2× throughput + ~½ RAM — premised on decode
being memory-bandwidth-bound.

**Benchmark that motivated it** — same Qwen2.5-3B `qwen.gguf` (FP16),
single-stream CPU decode:

| Metric | Overfit (F32 up-cast) | LLamaSharp (native llama.cpp) |
|--------|----------------------:|------------------------------:|
| Decode throughput | 2.58 tok/s | 9.67 tok/s |
| Peak working set | 14.4 GB | 6.0 GB |

**Built:** full FP16-resident path — a `MatrixWeight` precision-carrier union
threaded through the decode weight structs, `ProjectHalf` / `AccumulateHalf`
kernels, `GgufReader.LoadTensorAsF16`, an `fp16Resident` A/B toggle. Kernel parity
bit-identical to F32; 666 tests green.

**Measured — hypothesis REFUTED.** Rigorous A/B, best-of-3, same model:

| Metric | F32 (baseline) | FP16-resident |
|--------|---------------:|--------------:|
| Throughput | 2.58 tok/s | 1.68 tok/s — **−35%** |
| Steady RAM | 14.36 GB | 15.85 GB — regressed |

**Why it failed:**

1. **Decode is compute-bound, not bandwidth-bound.** Overfit F32 decode reads only
   ~31 GB/s — far under the ~50–80 GB/s DRAM ceiling. DRAM was never the
   bottleneck, so halving weight bytes unblocks nothing and the F16→F32 widen is
   pure added cost.
2. **No fused widen is possible in managed .NET.** A register-fused single-pass
   kernel needs a per-vector F16→F32 intrinsic. .NET 10 exposes none — no `F16C`
   class, no `Half` overload on `Vector256.ConvertToSingle` / `WidenLower`. The
   hardware `vcvtph2ps` is reachable only via whole-span `TensorPrimitives`
   (forcing a scratch round-trip). A hand-rolled SIMD bit-twiddle widen costs
   ~9 ops per 8 elements vs the 1-op hardware convert → fusing would be *slower*.
   So **−35% is irreducible** for FP16-resident decode in managed C#.
3. **RAM regressed** — the F32→F16 load conversion churns multi-GB F32 buffers on
   the Large Object Heap; the GC retains those segments.

**Outcome:** the entire FP16-resident path was reverted (`MatrixWeight`,
`ProjectHalf`, the loader F16 path, the toggle) — this post-mortem is all that
remains. Durable takeaway: **Overfit decode is compute-bound** — decode-speed
work must target compute, not memory bandwidth.

**What this means for the LLamaSharp gap.** The 3.75× gap is kernel quality, not
weight precision. Overfit F32 at 2.58 tok/s ≈ 31 GB/s — *under* the DRAM ceiling;
a bandwidth-saturating F32 kernel alone reaches ~4 tok/s (12 GB/token ÷ ~50 GB/s),
≈1.5× of measured headroom. The next decode-perf lever is therefore kernel-side —
blocked GEMV (`SingleTokenProjectionKernel.Accumulate` re-streams the whole output
vector for *every* input element → ~2× redundant memory traffic on FFN / LM-head),
parallelizing the small per-head matmuls, tighter SIMD — **not** precision tricks.
Structural follow-on is quantized storage (**Slot 2b**): Q4/Q8 cut bytes more than
FP16 and integer-SIMD is llama.cpp's actual weapon. Honest ceiling: pure-managed
C# cannot emit every intrinsic llama.cpp uses (F16C proved that) — a realistic
target is closing 3.75× → ~2–2.5×, not parity.

### Q4_K_M integration parity test

- [ ] Download `qwen2.5-3b-instruct-q4_k_m.gguf` (Ollama or HF) to `c:\qwen3b\qwen.q4km.gguf`.
- [ ] Run the existing `GgufQ4KMParityTests.Q4KM_TopTokenMatches_FP16Baseline_OnCanonicalPrompt` test (already written, currently `[LongFact]` + skip-if-missing).
- [ ] Tolerance: top-1 matches; max abs logit diff within Q4_K_M expected range (~0.5-1.5 % relative).

Synthetic unit tests already cover the algorithm; this test catches bit-layout regressions against llama.cpp/Ollama.

### Other quant formats

- [ ] Q5_K dequantizer (occasionally appears in mixed quant files; 176 bytes/block).
- [ ] Q4_0 / Q5_0 / Q5_1 (legacy formats; lower priority — Ollama defaults to K-quants).
- [ ] Q2_K / Q3_K_S (very aggressive quant; experimental quality).

### LoRA training

GPT1 LM-head LoRA training **landed** this session — see "Active track" above
(`Gpt1LoRAFineTuner`: graph-integrated effective-weight injection, `Adam` over
the LoRA `Parameter`s only, base frozen). Remaining items are the Llama-family
and broader-module scope:

- [x] Backward restricted to adapter parameters with frozen base — GPT1 LM head (Stage 1).
- [x] Adam over LoRA factors only — GPT1.
- [x] Extend to FFN (Stage 2) and attention Q/K/V/O (Stage 3) — GPT1 (QLoRA: whole base
  frozen-quantized Q4_K/Q8, validated on a real anomaly task).
- [ ] Backward through Linear / RMSNorm / SwiGLU / attention for the Llama family.
- [ ] Demo: overfit on a few samples, verify the adapter steers generation.

Opens "fine-tune LLM locally in pure C#" story. Major scope.

#### GGUF→training bridge (the "sztandar" / headline — real RAM win)

The QLoRA op + quantizer + GPT1 whole-base wiring shipped, but GPT1Model is F32-native
so quantizing there only *adds* copies — the real memory win requires loading an
already-quantized Qwen/Llama GGUF **directly** into a training graph so F32 is never
allocated. That needs the Llama-family forward as autograd ops (frozen Q4_K bases +
trainable LoRA + RMSNorm/RoPE/SwiGLU/GQA). Multi-session (~3–5).

- [x] **Session 1 — RMSNorm + SiLU autograd ops.** `OpCode.RmsNorm` / `OpCode.SiLU`,
  `TensorMath.RmsNorm(graph, x, gamma, eps)` (per-row, saves inv-RMS aux, dInput + dGamma,
  seq + `OverfitParallelFor` parallel) and `TensorMath.SiLU(graph, x)` (SwiGLU gate,
  `TensorPrimitives.Sigmoid` SIMD core). Backward dispatch wired; FD-validated
  (`RmsNormTests`, `SiLUTests` — forward parity seq+parallel, dInput/dGamma central-difference).
- [x] **Session 2 — RoPE autograd.** `OpCode.Rope`, `TensorMath.Rope(graph, input, cos, sin)` —
  adjacent-pair / GGUF-NeoX layout (`input [rows, headsPerRow·headDim]`, `cos/sin [rows, halfDim]`,
  constants/no-grad), per-pair 2-D rotation; backward = inverse rotation (orthogonal → `Rᵀ`, sin
  negated), accumulated into dInput; seq + `OverfitParallelFor`. `RopeTests`: **forward bit-faithful
  to the inference `RopeKernel.Apply`** (the layout-fidelity guarantee — GGUF base rotates identically
  in train + inference), FD backward, and forward∘inverse round-trip recovers input.
- [x] **Session 3 — GQA scaled-dot-product attention backward.** Rather than a bespoke GQA SDPA,
  added `OpCode.ExpandKvHeads` / `TensorMath.ExpandKvHeads(graph, input, kvHeads, groupSize)` — the
  training equivalent of HF `repeat_kv`: forward broadcasts each KV head to its query-head group
  (head `qh` reads KV head `qh/groupSize`), and the **GQA-specific backward sums each group's gradient
  into the shared KV head** (it was read groupSize times). Feeds the already-validated MHA SDPA
  unchanged (one head = one batch slice). `GqaAttentionTests`: forward broadcast, FD backward of the
  group reduction, and **full GQA path** (expand→3-D SDPA on 4 query : 2 KV heads) FD-checking dQ/dK/dV.
  Layout-agnostic (dim-0 head axis), groupSize=1 = MHA copy. KV-head ↔ token-major projection wiring
  is Session 4's job.
- [x] **Session 4 — trainable Llama block assembly.** `DeepLearning/TrainableLlamaBlock` —
  a full Pre-LN Llama/Qwen decoder block as one autograd forward: `h = input + Attn(RMSNorm(input,γ1))`,
  `y = h + SwiGLU(RMSNorm(h,γ2))`, with **every projection a frozen `IDequantRowSource` (Q4_K/Q8)**
  via `FrozenQuantizedLinear` and the only trainable params the two RMSNorm gains (LoRA layers on later
  exactly like the GPT-1 path). Combined-tensor / GGUF-faithful layout (Wq `[nQ·dH,dModel]`,
  Wk/Wv `[nKV·dH,dModel]`, Wo `[dModel,dModel]`, gate/up `[dFF,dModel]`, down `[dModel,dFF]`) so
  Session 5's loader feeds it with zero repacking. Wiring: RMSNorm → frozen QKV → RoPE (token-major,
  all heads one call) → `Transpose01` to head-major → `ExpandKvHeads` (GQA) → SDPA → `Transpose01` back
  → frozen O → residual → RMSNorm → frozen SwiGLU(`SiLU(gate)⊙up`) → residual. One new op needed —
  `OpCode.Transpose01` (rank-3 axis-0/1 swap, exact backward). `TrainableLlamaBlockTests` (4 Q : 2 KV):
  forward shape, **end-to-end FD backward** on input + both γ, and a γ-only train loop that drops loss
  with the **frozen quantized base provably bit-identical** before/after.
- [x] **Session 5 — loader→training wiring + e2e on real Qwen GGUF + RAM measurement. THE PAYOFF.**
  Bridge plumbing (zero-copy, no repack): `DecodeWeight.AsRowSource()` (Q4_K/Q6_K/Q8 → frozen
  `IDequantRowSource`; F32 rejected), `ConcatRowsDequantSource` (per-head Wq/Wk/Wv → combined
  `[nH·dH, dModel]`), `ConcatColsDequantSource` (per-head Wo → `[dModel, nH·dH]`), and an internal
  `CachedLlamaInferenceEngine.GetTrainableLayer(i)` accessor. RoPE op extended to **split-half**
  (Qwen `RopeSplitHalf=true`) alongside adjacent-pair, threaded through the block. `QLoraGgufBridgeTests`
  (runs in the fast suite): adapter parity, AsRowSource dispatch, and a loader-shaped per-head block
  trains FD-validated. **`QwenGgufQLoraE2ETests` [LongFact], executed on the real
  Qwen2.5-3B Q4_K_M GGUF (2.1 GB):** layer-0 (dModel=2048, 16 Q : 2 KV, headDim=128, split-half) wired
  straight into a `TrainableLlamaBlock` — gradients flow to input + both γ, **frozen 4-bit base
  bit-identical**, and **building the trainable base allocated ~0 KB vs ~294 MB if the layer's F32 were
  materialized** (×36 layers ≈ ~10 GB of F32 base avoided). The sztandar proven: a real already-quantized
  GGUF fine-tunes in pure .NET CPU with the 4-bit base never expanded to F32.
- [x] **Full-model RAM measured (`QwenGgufTrainingRamTests` [LongFact], real Qwen2.5-3B Q4_K_M, exact via
  the graph arena `CurrentOffset` high-water — no GC/pool noise).** base = **1.96 GB** 4-bit (Σ layer quant
  weights 1.62 GB; F32 expansion would be ~11.5 GB). Per-block training activation (fwd+bwd): 73.5 MB @ T=128,
  149 @ T=256, **306 @ T=512**, 644 @ T=1024 (≈linear — FFN dFF=11008 dominates, not attention T²).
  Full-model QLoRA @ T=512: **without checkpointing ~13.3 GB** (all 36 layers co-resident), **with gradient
  checkpointing ~3.1 GB** → **fits a 16 GB CPU box with room to spare.** Confirms the promise with real
  numbers; checkpointing (built for GPT-1, unwired into this path) is the lever for bigger models.
- [x] **FULL TRAINABLE MODEL + checkpointing + real fine-tune (Option A).** `DeepLearning/TrainableLlamaModel`
  assembles the whole stack: frozen quantized embedding (per-token `DequantizeRow`) → N
  `TrainableLlamaBlock`s (each under `graph.Checkpoint`) → trainable final RMSNorm → frozen quantized LM head
  → logits; `FromEngine(engine, rank, …)` builds it straight from a loaded GGUF (zero repack). Trainable =
  per-block LoRA (`LlamaBlockLoRA` over all 7 projections, B-zero init) + RMSNorm gains; optional LM-head LoRA
  (off by default — high-variance on a 152k vocab). Next-token CE loss seeds `logits.Grad` →
  `BackwardFromGrad`. **Synthetic proof (`TrainableLlamaModelTests`, fast suite): a tiny model OVERFITS a
  sequence — loss 4.31→0.0004, greedy 24/24 reproduced; gradient checkpointing is bit-identical (maxAbs 0.0)
  AND trains (4.82→0.0004).** **Real proof (`QwenGgufQLoraFineTuneE2ETests` [LongFact], Qwen2.5-3B Q4_K_M):**
  fine-tune all 36 layers under checkpointing — **loss 12.24→2.81** (smooth/monotonic), **P(correct next-token)
  3.1e-4→6.2e-2 = 201× up**, **frozen 4-bit base bit-identical**, **peak process WS 2.92 GB** (incl. the 2 GB
  base — matches the ~3.1 GB extrapolation), ~1.27 s/step. The sztandar is now a working fine-tuner. Open:
  adapter save/load + multi-sequence batches (assembly, not correctness).
- [x] **KNOWLEDGE-INJECTION SHOWCASE (`QwenGgufKnowledgeInjectionDemoTests` [LongFact]).** The demo: teach
  real Qwen2.5-3B a fact it cannot know — a made-up metal "Zorvex" mined only in "Tarnholm" — then ask it.
  Added greedy `TrainableLlamaModel.Generate(graph, prompt, maxNew, eos)` (checkpointed forward = no grad
  buffers, tiny arena) wired to `QwenTokenizer.Load(modelDir)` (tokenizer is separate from the GGUF). LM-head
  LoRA enabled (`loraOnLmHead: true`) for output capacity. **BEFORE: "…the city of" → "gow" (base clueless).
  Fine-tune on 3 sentences (~250 steps CPU): loss 14.67 → 0.0000. AFTER: "…the city of" → "Tarnholm. Zorvex"
  — recites the taught fact coherently.** KEY STABILITY FINDING: Adam blew up catastrophically at low loss
  (0.23→6.58 spike) with the default `Epsilon 1e-8`; **`Epsilon = 1e-4` gives a smooth monotonic descent to
  0** (the 1/√v amplification when gradients vanish on a tiny overfit set). This is the showable
  "fine-tune an LLM on your CPU in .NET, no GPU/Python" demo.
- [x] **Adapter SAVE/LOAD — fine-tune persists to a file (train → save → load → use).**
  `TrainableLlamaModel.SaveAdapter(path)` / `LoadAdapter(path)` serialize ONLY the trained delta (every LoRA
  A/B + RMSNorm gain, deterministic order, bulk `MemoryMarshal` IO) — the frozen 4-bit base is never
  rewritten (no GGUF export; loading stays one-directional). Tens of MB vs the 2 GB base. **Synthetic
  round-trip (`TrainableLlamaModelTests`, fast): train→save (65 KB)→fresh model→load → loaded logits
  bit-identical to trained (maxAbs 0.0), greedy 18/18 reproduced.** **Real round-trip
  (`QwenGgufQLoraAdapterRoundTripTests` [LongFact]): train Qwen-3B on the Zorvex fact → save adapter file →
  FRESH model from the untouched base → load → recites "Tarnholm".** Closes the loop into a portable
  "knowledge module" attached to a frozen base.
- [ ] **(FUTURE, separate topic) Fast fine-tuned decode = LoRA on the optimized inference engine (≈1.13× llama.cpp).**
  Two decode paths exist today: the **training** model (`TrainableLlamaModel`, has KV-cache + our LoRA but
  NAIVE per-row-dequant+`Dot` kernels) and the **inference** engine (`CachedLlamaInferenceEngine`, KV-cache +
  repacked 8×8 GEMV kernels benchmarked ~1.13× behind llama.cpp, but runs the frozen base WITHOUT LoRA). To
  serve a fine-tuned model at llama.cpp-class speed, hook the trained adapter into the optimized inference
  forward: the Q4_K base can't be F32-weight-merged (the existing `LlamaLoRAAdapter.Enable` path needs F32),
  so add the LoRA as a **side GEMV** (`+ (x·A)·B`, one tiny rank-r matmul per projection per token) inside
  `CachedTransformerBlock`/`CachedMultiHeadAttention` decode, gated by an attached adapter. ~2–3 sessions,
  higher risk (touches the hot inference path). NOTE: the training-model KV-cache (next bullet) does NOT help —
  it's naive single-threaded, so it's 6× slower than the already-parallel uncached recompute at demo lengths;
  the win requires parallel kernels, which is this item. Generation was never the bottleneck (~0.4 s/token).
- [~] **Training-model KV-cache (Option A) — BUILT + CORRECT, but it does NOT speed up generation (honest
  finding).** `TrainableLlamaModel.GenerateCached(...)` + `TrainableLlamaBlock.DecodeStep(...)` — incremental
  single-token decode (no autograd): only the new token flows through the layers, attending its query over a
  per-layer K/V cache (RoPE via the inference `RopeKernel`, manual softmax attention, plain-span RMSNorm/SwiGLU
  + naive per-row dequant + LoRA side-GEMV). **Bit-faithful: cached greedy tokens == uncached `Generate`
  (`TrainableLlamaModelTests.GenerateCached_MatchesUncachedGenerate`).** BUT measured on real Qwen-3B
  (`QwenGgufCachedDecodeSpeedTests` [LongFact]): cached **2700 ms/token vs uncached 424 ms/token — 6× SLOWER**.
  Root cause: the premise was wrong. Generation was never the bottleneck (~0.4 s/token) because the uncached
  forward already parallelizes the dequant-matmul (`FrozenQuantizedLinear` over all cores + amortized dequant),
  while the cached decode uses naive SINGLE-THREADED per-row kernels. The cache only pays off at long contexts
  AND with parallel kernels (= Option B / parallelizing `ProjVec`+`LmHeadArgmax`). The real wall-time sink in
  the demos is TRAINING (~5 s/step), not generation. Kept as a correct reference path; no speed claim.

---

## Performance backlog

### Training CPU-saturation track

Goal: on 16+ core machines, training step pegs CPU at near-100 % across all cores instead of single-digit cores idle/under-utilized. Inventory of what's already parallel-capable:

| Op | Parallel today | Threshold |
|----|----------------|-----------|
| `LinearKernels` backward (input + weight) | ✅ above 1M ops | `ParallelThreshold = 1_048_576` |
| `Conv2D` forward + backward | ✅ | per `TensorMath.Convolution` |
| `MaxPool2D`/`AvgPool2D` forward + backward | ✅ | `TensorMath.Pooling` |
| `LSTM` forward + backward | ✅ over batch | `TensorMath.Sequence` |
| `Adam.Step` | ✅ over parameters | always parallel |
| `ScaledDotProductAttentionBackward` | ✅ over batch | **flipped to default ON** (was experimental flag) |
| `ScaledDotProductAttention` forward | ❌ sequential over batch | candidate |
| `LayerNorm` / `RMSNorm` forward + backward | ❌ sequential | per-token, usually small enough |
| `Embedding` backward (scatter-add) | ❌ sequential | hard — atomic conflicts |
| Activation kernels (GELU/SwiGLU/ReLU) | ❌ SIMD-only, not threaded | usually fine |

Concrete items:

- [x] **`EnableParallelAttentionBackward` default ON** — bit-identical to sequential (parallel only over batch dim, no cross-batch reduction), author measured ~27 % backward speedup on data-parallel TinyShakespeare. Stable training path now uses it by default; remaining single-thread case is intentional fallback for batch=1 where Parallel.For overhead would be pure waste.
- [x] **`BatchSequentialThreshold` 128 → 32** — `MaxPool2D` / `AvgPool2D` / `LSTM` forward + backward + bias-add (`TensorMath.Algebra`) parallelize across batch only above this threshold. MNIST training (B=64) was falling into the sequential branch (64 < 128) → `MaxPool2D` ate 42 % of epoch time on a 32-core machine, single-threaded. After lowering to 32: **MNIST training −38 % wall clock** (5551 → 3456 ms for 5 epochs, full MNIST 60k @ B=64), with MaxPool2D dropping from 383 ms/epoch to ~87 ms/epoch (−77 %). Full sweep still green (617/0/63). The 32 floor keeps sequential for genuinely small batches where Parallel.For overhead would dominate.
- [ ] **True batched training (B > 1)** — *biggest single lever for CPU saturation.* MHA in training path is currently batch=1 only. With B=8/16, every existing parallel-over-batch path (attention forward + backward, layer-norm batches, optimizer-over-params) starts working. 2-3 days; ROADMAP'd separately under Qwen track but is GPT-2 training prerequisite.
- [x] **Drill-down per-OpCode backward profiler** — added `ComputationGraph.BackwardProfileEnabled` toggle + `GetBackwardOpProfile()` + `ResetBackwardProfile()`. Zero overhead when off (one null-check), ~50 ns per op when on. MNIST CNN training 5-epoch aggregated breakdown: **Linear 52 %**, Conv2D 20 %, ReLU 14 %, MaxPool 7 %, Reshape 4 %, SoftmaxCE <1 %. Linear dominates because Linear(1352→64) backward = 2 GEMMs (dW + dX) per batch × 4685 batches. Already parallel above 524k/1M thresholds; near peak FP32 throughput for this matrix size. Next deeper investigation requires actual CPU-utilization sampling (not just timing).
- [x] **CPU-utilization probe** during MNIST training — added `Process.TotalProcessorTime` sampling at run + per-epoch granularity. **Measurement on 32-core Ryzen 9 9950X3D: only 6.81 / 32 cores effective (21.3 % utilization).** All major kernels have `Parallel.For` wired, yet 79 % of CPU stays idle. Root cause: ~47k `Parallel.For` calls per MNIST epoch × ~5-10 µs each dispatch overhead = 230–470 ms / 550 ms epoch = **40–85 % of epoch wasted in TPL dispatch/sync**, not compute. **CPU-saturation track is fundamentally blocked by `Parallel.For` overhead, not missing parallelism.**

### Zero-allocation custom Parallel.For — THE unlock

Standard `System.Threading.Tasks.Parallel.For` per call:
- ~3 KB managed allocation (closure object, `Task[]` chunks, internal bookkeeping)
- ~5-10 µs dispatch overhead (Task scheduling, thread wake)

Both kill us:
- The **3 KB alloc** broke the 0 B / generated token claim when we tried `ProjectParallel` on LM head (~92 KB / 31 tokens). Blocked allocation-free parallel inference.
- The **5-10 µs overhead** dominates training when many small Parallel.For calls fire per batch (MNIST: 47k calls/epoch). Caps utilization at ~20 % regardless of how many `Parallel.For` we add.

**`OverfitParallelFor` — current status (implemented, bulk-wake dispatcher):**

- [x] Pre-spawned `N = ProcessorCount` persistent threads, all parked on one shared `SemaphoreSlim`.
- [x] Per-chunk `ChunkState[]` descriptors filled by dispatcher; workers claim a unique index via `Interlocked.Increment` on a 128 B cache-line-padded counter.
- [x] **Bulk wake via `SemaphoreSlim.Release(N)`** — one syscall releases N tokens; kernel scheduler resumes waiters in parallel (NOT serially the way `N × AutoResetEvent.Set` would). This is the architectural win — it bypasses the ~32 µs floor that per-worker `Set` dispatchers hit at 32-fanout.
- [x] Function-pointer dispatch (`delegate*<int, int, void*, void>`) — no closure, no delegate alloc.
- [x] **Exception propagation** via `ExceptionDispatchInfo.Capture` — body throws are caught per chunk, surfaced to caller with original stack trace. No unhandled-exception process crash.
- [x] API: `OverfitParallelFor.For(int rangeStart, int rangeEnd, delegate*<int, int, void*, void> body, void* context)`.
- [x] Tests: 0 B proof, correctness vs sequential, boundary cases, 10k-iter stress, exception propagation + post-throw recovery.

**Measured (Ryzen 9 9950X3D, 32 logical cores, full benchmark sweep):**

| InnerIters (body work) | Sequential | Parallel.For (TPL) | OverfitParallelFor | Speedup vs Seq | Alloc Overfit | Alloc TPL |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (empty) | 0.54 µs | 5.9 µs / 3.0 KB | ~6.6 µs / 0 B | — | 0 B ✅ | 3.4 KB |
| 1k (~1 µs body) | 92 µs | 48 µs | ~70 µs | — | 0 B ✅ | 3.8 KB |
| 100k (~100 µs body) | 3.5 ms | 326 µs | ~190 µs | 18× | 0 B ✅ | 6.8 KB |
| 1M (~1 ms body) | 34 ms | 1.46 ms | ~1.19 ms | **28.7×** | 0 B ✅ | 9.9 KB |

**Where it wins:**
- **Vs `Parallel.For`:** competitive in time (within 15% in worst case, **1.7× faster from 100 µs body upward**), and ~3000× cheaper on allocations across the whole range.
- **Vs Sequential:** the crossover at InnerIters ≈ 100 means parallelization pays from any body work above ~100 cycles. At InnerIters = 1M (1 ms body), 28.7× speedup ≈ 90% of 32 logical threads — near-optimal scaling.

**Why bulk wake beats N × Set:** earlier iterations (per-worker `AutoResetEvent` + simple/spin-then-park hybrid) hit a hard ~32-47 µs dispatch floor at 32-fanout — N `Set` calls serialize at the kernel because each one is a distinct event signal. `SemaphoreSlim.Release(N)` queues N pulses inside its internal lock and exits; the OS scheduler then resumes the waiters roughly in parallel (Windows `WakeConditionVariable` / Linux `futex_wake(N)` semantics). Result: ~5-7 µs dispatch at 32-fanout. Pre-bulk-wake prototypes (`OverfitParallelForHybrid`, `OverfitParallelForBulkWake`) were retired once this design landed.

**Migration to `OverfitParallelFor` — measured impact (Ryzen 9 9950X3D, MNIST 60k batch=64 small CNN, 5 epochs):**

| Phase | Wall time | Cores effective | Linear bwd total | Total CPU |
|---|---:|---:|---:|---:|
| TPL baseline (ROADMAP previous) | 5551 ms | 6.81 / 32 (21.3%) | — | ~37.8 s |
| + `LinearKernels` BackwardInput + AccumulateWeightGrad migrated | 5107 ms | 6.15 / 32 (19.2%) | 1133 ms | ~31.4 s |
| + `TensorMath.Pooling` (MaxPool/AvgPool fwd+bwd) migrated | (same) | (same) | (same) | (same) |
| + `TensorMath.Algebra` (AddBias, MatMul fwd, MatMulAdd variants) migrated | (same) | (same) | (same) | (same) |
| + `ComputationGraph.Linear` forward migrated | **4870 ms** | **7.03 / 32 (22.0%)** | **982 ms** | ~31.0 s |

**Result:** −12 % wall time, −18 % total CPU consumed, +0.6 percentage points cores effective vs. the TPL baseline. Linear backward time dropped 13 % (1133 → 982 ms). All 626 correctness tests still green.

**Honest read of why we didn't hit the 60-80 % cores-effective target:** MNIST CNN at this scale is Amdahl-limited. The 4685 batches/epoch × ~10 micro-ops/batch include many serial slices (graph reset, allocator paths, small-Linear sequential branch for Linear(64→10) at B=64 = 41k ops well below parallel threshold, copy input/target, optimizer.ZeroGrad). The dispatcher *itself* is no longer the bottleneck — eliminating its overhead saved CPU but not enough sequential work was unblocked to fill more cores.

**Where the dispatcher win will actually show up:**
- Larger models (GPT-2 scale): per-op body work is ~ms scale, dispatch is ≤1% overhead, and parallelism is much higher.
- Allocation-free hot paths (LM head, attention forward, prefill kernels): 0 B/call is now the bottleneck-relevant property — same wall time as TPL with no GC pressure.
- High-frequency inference loops where TPL's 3 KB/call hits Gen0 hard.

**Migrated / parallelized call sites:**
- `LinearKernels.BackwardInput` + `AccumulateWeightGrad`
- `TensorMath.Pooling`: `MaxPool2D` fwd+bwd, `GlobalAveragePool2D` fwd+bwd
- `TensorMath.Algebra`: `AddBias`, `MatMulRaw`, `MatMulAdd_A_BT_Raw`, `MatMulAdd_AT_B_Raw`
- `ComputationGraph.Linear` (forward batched parallel path)
- **`TensorMath.Gelu` (forward + backward)** — was a pure sequential scalar `for` loop. Now (a) **OverfitParallelFor over chunks**, AND (b) **SIMD-batched inside each chunk** via `TensorPrimitives.Multiply/Add/Tanh/Subtract` pipeline on 1024-element tiles (stackalloc'd scratch on worker thread stack). Tile size chosen to fit in L1 cache so the multi-pass pipeline stays hot. The scalar `MathF.Tanh` is replaced with SIMD `TensorPrimitives.Tanh` (polynomial approximation). Cumulative GPT-1 batch=32 GELU backward: **1774 → 42 ms (42× faster)** vs the initial scalar-serial baseline, of which the SIMD pipeline contributed an additional **2.6×** on top of the prior parallel-over-chunks version.
- **`TensorMath.ScaledDotProductAttention` forward** — was sequential `for (b)` over batch. Symmetric to the existing parallel-over-batch backward — same `batchSize > 1 && work >= AttentionParallelWorkThreshold` guard. Per-batch work is independent (only writes to per-batch slices of `attnWeights` / `output`). For multi-head training the SDPA call sees effective batch = `B × H` (heads flattened), so even at training `B=1` the parallel path kicks in for `H ≥ 4`. Biggest single contributor to cores-effective in the session: GPT-1 batch=32 wall **191 → 114 ms (−40%)**, cores effective **7.93 → 13.85 / 32 (+75%)**. With this change the forward path becomes the second-most parallelized chunk of the training step (after Linear bwd), and `cores effective` finally crosses 1/3 of physical capacity at training batch sizes.
- **`TensorMath.LayerNorm` (forward + backward)** — was sequential per-row. Forward parallelizes trivially (rows independent). Backward needs per-worker partial accumulators for `dGamma[i]` / `dBeta[i]` (shared across rows) — stackalloc'd by caller, sequential SIMD merge after parallel pass. dInput parallelizes in the same pass (per-row, no shared writes). Falls back to sequential when `C > 4096` to keep stackalloc bounded.

**Cumulative GPT-1 training-step impact** (Ryzen 9 9950X3D, 4-layer GPT-1, dModel=128, dFF=512, seqLen=128, per `GPT1CpuUtilizationProbeTests`):

| BatchSize | Wall/step start | + GELU parallel | + LayerNorm parallel | + SIMD-batched GELU | + SDPA fwd parallel | Total Δ wall | Cores start → end |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 126 ms | 82 ms | 78 ms | 78 ms | **65 ms** | **−48%** | 3.48 → **10.40** / 32 |
| 16 | 217 ms | 137 ms | 112 ms | 106 ms | **74 ms** | **−66%** | 2.99 → **11.44** / 32 |
| 32 | 414 ms | 243 ms | 198 ms | 191 ms | **114 ms** | **−72%** | 3.89 → **13.85** / 32 |

Per-op backward (batch=32, aggregated over 20 steps): GELU 1774 → 109 ms (−94%, 16× faster). LayerNorm 800 → 39 ms (−95%, 20× faster). Now Linear (1042 ms — already parallelized, just much more numerous) dominates the backward profile. The remaining sequential ops on the critical path are smaller: Add residuals (62 ms), Reshape (55 ms), Embedding (2.5 ms).

This validated the whole strategy: dispatcher was correct from the start, and *each* additional element-wise kernel we move from sequential to OverfitParallelFor produces a real, measurable win — the playbook works for any per-element activation/normalization op.

**Future patterns from `vs2022-performance-patterns.md` (catalog for later sessions):**

Reviewed the 45 patterns extracted from VS 2022 decompilation. Already adopted:
- ✅ Custom Parallel wrapper (pattern 28) → `OverfitParallelFor`
- ✅ Lock-free `Interlocked` (pattern 7)
- ✅ Cache-line padding `[StructLayout(Explicit)]` (pattern 32) → `PaddedCounter`
- ✅ `[MethodImpl(AggressiveInlining)]` selectively (pattern 3)
- ✅ `readonly struct` defaults (pattern 2)
- ✅ `EventSource` per component (pattern 29) → `ArrayPoolEventSource`
- ✅ `ArrayPool<T>.Shared.Rent` (pattern 10) → `OverfitPool<T>`
- ✅ **`[module: SkipLocalsInit]` assembly-wide (pattern 22)** — landed with audit (caught LoRA accumulator bug as side-effect)
- ✅ **SIMD-batched element-wise via `TensorPrimitives` (pattern 25)** — applied to GELU as the reusable template
- ✅ **`FrozenDictionary` for load-once lookup tables (pattern 6)** — `BytePairEncoder` `_tokenToId` / `_mergeRanks` / `ByteDecoder` converted from `Dictionary` to `FrozenDictionary`. AOT-clean (zero IL2026/IL3050 from `ToFrozenDictionary`). The merge-rank table is the hottest — `BpeEncode` scans every adjacent pair O(parts²) per word.
- ✅ **`[CallerArgumentExpression]` in guards (pattern 45)** — `TensorKernelGuards` validators auto-capture argument names; error messages now name the offending span/tensor.

Rejected after evaluation:
- ❌ `Expression.Compile` / `DynamicMethod` / `ActivatorUtilities.CreateFactory` (patterns 39, 40, 41) — blocked by `BannedSymbols.txt` (no reflection in AOT path)
- ❌ `[Conditional("DEBUG")]` on validators (pattern 9 fragment) — our `TensorKernelGuards` are caller-contract enforcement, not pure invariants; stripping degrades debuggability without measurable gain
- ❌ `Channel<T>` (pattern 21) — we don't do streaming/IPC
- ❌ `IValueTaskSource` / `ValueTask` / `IAsyncDisposable` (patterns 14, 15, 16) — no async paths
- ❌ WPF patterns (36, 37, 38, 44), MultiplexingStream (19), F# persistent collections (43) — different domains

**Worth adopting in future sessions, prioritized by ROI:**

| Pattern | Where | Effort | Value | When to do it |
|---|---|---:|---|---|
| **5 — Segmented arrays (LOH avoidance)** | `TensorStorage<T>` for tensors >85 KB | ~1 day | Removes GC pauses on big-model paths | When LOH symptoms hurt (GPT-2+ scale) |
| **35 — `[PerformanceSensitive]` analyzer hint** | New custom analyzer for "no-alloc hot path" methods | ~few days | Compile-time enforcement of zero-alloc contracts | When team grows beyond one person |
| **23 — `[InlineArray(N)]` for packed structs** | Maybe GGUF block quant formats | ~hours | Cleaner than `LayoutKind.Explicit` for fixed-size content | Opportunistic — when touching binary IO |
| **34 — `[ThreadStatic] Stack<T>` per-thread pools** | Hot per-thread scratch in training kernels | ~1 day | Cache-warmer than central pool | When profiler shows pool contention |
| **24 — `[ModuleInitializer]`** | Run-once setup (e.g. native lib preload) | ~10 min | One-time setup without static-ctor surprises | If we ever ship native deps (we don't today) |
| **20 — Batched telemetry (timer + flush + persist)** | `ArrayPoolEventSource` upgrade for long training runs | ~hours | Reduce ETW overhead on training | When telemetry becomes a real concern |
| **30 — `UnmanagedBufferAllocator` (`NativeMemory.Alloc`)** | `TensorStorage<T>` for very-long-lived weights | ~1 day | Removes weights from GC's scope entirely | For multi-GB models pinned in memory |

**The pattern from VS2022 that drove the biggest win in this session:** **#25 (SIMD-batched element-wise via TensorPrimitives)**. The template — outer `OverfitParallelFor.For` over chunks + inner `TensorPrimitives` pipeline on stackalloc'd tiles — is now established and ready to reuse for any future scalar kernel (Sigmoid, SwiGLU, SiLU, custom activations).

**Lesson learned — when NOT to parallelize element-wise:**
Attempted parallelizing `TensorMath.Add` (residual add) and **measured a regression** (+20% wall on GPT-1 batch=32, +55% on Add backward itself). Reverted. `Add` is **memory-bandwidth-bound** (read 2 arrays + write 1 = 3× data movement). On a typical desktop memory subsystem (~50 GB/s) 2-3 cores already saturate the bus, so the OverfitParallelFor dispatch overhead (~10 µs cold) is pure cost.

**Rule of thumb for future migrations:** parallel pays for element-wise ops only if the body is *compute-bound* (heavy per-element math like GELU/LayerNorm/Linear). For *memory-bound* ops (Add/Subtract/Scale/element-wise copy), keep sequential SIMD — the bandwidth ceiling caps you regardless of core count, and dispatch overhead becomes net negative.

**Pending migrations / parallelizations (deferred):**
- `TensorMath.Convolution` (Conv2D fwd+bwd) — per-worker workspace pattern needs `GCHandle`-pin or POH refactor of `Conv2DWorkspace`. Expected gain on MNIST: ~100-150 ms / 5 epochs (Conv2D backward 459 ms, fwd ~570 ms aggregated). Worth doing if/when GPT-2 batched training adds bigger Conv-heavy workloads.
- `TensorMath.Sequence` (LSTM) — relevant for LSTM training.
- `TensorMath.Attention` (`EnableParallelAttentionBackward` path) — relevant for transformer training at B > 1.
- `Optimizers.Adam` — parameter-parallel update, hot path during training.
- Various lower-priority sites: `DataAugmenter`, `FastRandomForest`, evolutionary noise tables, anomaly training.

**Pattern for future migrations:** `fixed (T* p = span)` block around `OverfitParallelFor.For(start, end, &ChunkBody, &ctx)` where `ChunkBody(int chunkStart, int chunkEnd, void* contextPtr)` loops over the chunk and reads pointers from the context struct.
- [-] **LinearKernels threshold tuning** — verified for MNIST that current thresholds (BackwardInput 524k, AccumulateWeightGrad 1M, ForwardBatched 500k) are correctly placed: Linear(1352→64) at B=64 = 5.5M ops → already parallel; Linear(64→10) at B=64 = 41k ops → sequential (41k / 32 cores = 1.3k ops per thread, well below Parallel.For overhead). No measurable benefit available at MNIST scale. **Revisit when GPT-2 batched training (B>1) lands** — smaller per-batch ops in GPT may benefit from lower thresholds.
- [ ] **SIMD path for `MaxPool2DForwardWithIndicesNchw` (training path)** — inference path has `TensorPrimitives.Max` fast path for pool=2; training path is scalar because index tracking complicates SIMD (need comparison masks to select max value AND record source index). Estimated saving on MNIST: ~40 ms / epoch (~7 % of post-threshold-fix epoch time). 1-2 hours focused work + parity tests vs scalar.
- [ ] **`ScaledDotProductAttention` forward parallel-over-batch** — symmetric with the backward we just enabled. Likely 15-20 % forward speedup at B ≥ 4.
- [ ] **Threshold tuning** — `LinearKernels.ParallelThreshold = 1_048_576` is set for inference (avoid Parallel.For overhead on tiny matrices). For training where the per-call work is amortized across many tokens, lower threshold may pay. Need per-op profiling first.
- [ ] **Data-parallel training as first-class API** — `TinyShakespeareDataParallelTrainingTests` proves N model replicas + gradient averaging works. Promote to public API, with idiomatic worker pool, for "N cores → N replicas" training scaling on a single machine.
- [ ] **Batched linear kernel** — `LinearKernels.ForwardBatched` measured win at batch 64/256 vs ONNX Runtime.
- [ ] **Backward kernels** — Linear/Conv backward through pure span kernels where practical.
- [ ] **Optimizer kernel profiling** — Adam/AdamW state updates, zero-grad allocation sources (currently ~96 B/Step on Adam — already tracked by skipped tests in `AdamOptimizerBehaviorTests`).
- [ ] **CPU SIMD audit** — AVX2/AVX-512/AVX10 where available.
- [ ] **Thread-scaling stabilization for large training workloads.**

### Correctness

- [ ] Numerical equivalence tests across scalar/SIMD paths.
- [ ] Determinism policy for parallel training kernels.

---

## Market-driven priorities (2026 external research)

External scan (May 2026): the LLM-inference job market, RAG / vector-DB adoption, the self-hosted-LLM trend, and the .NET AI ecosystem. Honest top-line:

**Job postings from the highest-paying companies are a trap, not a map.** They converge on Python + CUDA + vLLM/TGI/Triton + Kubernetes + GPU clusters + distributed training (AI-infra postings grew +47 % YoY). None of it is addressable by a pure-C#, CPU-first, zero-native-dependency engine — chasing it means losing to vLLM/TensorRT on their own ground. The opportunity is the *adjacent, underserved* space.

What the market actually validates for Overfit's niche, ranked:

| # | Priority | Market signal | Effort / status |
|---|----------|---------------|-----------------|
| 1 | **Embedding model support** (BGE / E5 / multilingual-e5) | Vector-DB market $2.46B (2024) → $10.6B (2032), 27.5 % CAGR; Gartner: 30 %+ of enterprises on vector DBs by 2026; enterprise hybrid-retrieval intent tripled in one quarter. RAG is *the* enterprise LLM pattern. Embedding models are encoder transformers — single forward pass, no KV-cache — **CPU-friendly**: the one mainstream workload squarely in Overfit's wheelhouse. | ✅ **DONE 2026-05-28 (BERT encoder family).** `WordPieceTokenizer` (vocab.txt, BasicTokenizer + greedy WordPiece, [CLS]/[SEP], accent strip) + `BertEncoder` (bidirectional via `causalMask:false`, post-LN blocks, learned pos+token-type emb, embeddings LayerNorm, mean/last pooling + L2) + native `BertSafetensorsLoader` (HF [out,in]→[in,out] transpose + per-head Q/K/V column-split / O row-split, prefix auto-detect, no Python) + `BertConfigReader` + turnkey `SentenceEmbedder.FromPretrained(dir)` → `VectorStore`. Targets validated 2026-05-29 — all 384-d, all bit-parity to HF/PyTorch on real weights: **all-MiniLM-L6-v2 cosine 1.000000** (mean pool, no prefix), **BAAI/bge-small-en-v1.5 cosine 0.999999** (CLS pool + query instruction), **intfloat/e5-small-v2 cosine 1.000000** (mean pool + `query:`/`passage:` prefixes). Transpose/head-split round-trip verified directly (`BertSafetensorsLoaderTests`); semantic ordering + VectorStore top-1 retrieval covered by `MiniLmEmbeddingEndToEndTests` / `BgeAndE5EmbeddingEndToEndTests` [LongFact] (`OVERFIT_MINILM_DIR` / `OVERFIT_BGE_DIR` / `OVERFIT_E5_DIR`). `SentenceEmbedder.ForMiniLm` / `ForBgeEnV15` / `ForE5` bake the per-family conventions in. **GELU question SETTLED:** repo's tanh-approx GELU is indistinguishable from BERT's exact erf (cosine 1.0). Tokenizer also exact-matches HF fast tokenizer. **Follow-ons:** SentencePiece tokenizer (XLM-R / multilingual-e5), BGE-base / BGE-M3 for bigger contexts. |
| 2 | **Deepen regulated / private-inference positioning** | EU AI Act reaches full enforcement Aug 2026 (high-risk AI requires audit trails, explainability, human oversight). Self-hosted deployments report −75 % data-breach incidents — but 175k exposed Ollama servers are actively exploited ("LLMjacking"): self-hosted-*as-a-server* is itself a risk. Overfit-as-a-library-in-process (no exposed endpoint) is structurally safer. | Mostly copy. Started: `docs/scenarios/regulated-industries.md` + README "What Overfit is not". Add the "library-in-process > exposed server" security argument. |
| 3 | **In-memory quantization** (Q4_K / Q6_K dequant-fused matmul) | Every inference-engine comparison lists quantization as core (llama.cpp = "CPU-first + quantization"). It is the path to running *larger* models on CPU / edge. | Already specified — see **Slot 2b** above. This research promotes it from "deferred" to a named priority. (The FP16-resident shortcut, Slot 2c, was tried and reverted — see its post-mortem; quantization is the real lever.) |
| 4 | **Audit / inference-record primitives** | EU AI Act mandates reproducible audit trails + explainability for high-risk AI. Overfit already has deterministic greedy decode and file-versioned weights — the missing piece is a first-class, opt-in decision record (input + model hash + output + timestamp). | ~few days. Grounds the prior generic "telemetry" idea in an actual regulation. |
| 5 | **Microsoft Agent Framework / Semantic Kernel adapter** | Microsoft consolidated Semantic Kernel + AutoGen into "Microsoft Agent Framework" (Oct 2025); SK is in maintenance mode. Do **not** build a competing agent framework — be the inference + embedding *backend* it calls. Distribution via Microsoft's own ecosystem. | ~2 days, once embeddings (item 1) land. |

**Explicitly out of scope** — the market confirms these are GPU + Python territory; competing there loses: GPU-throughput serving (vLLM / TGI / TensorRT), distributed / multi-node training, multi-cloud orchestration, a homegrown agent / LangChain framework, multimodal (vision + text).

This section is a strategic overlay — it ranks and justifies; the tactical breakdowns live in "Slot 2b" (quantization), the "Active track" (LoRA), and "Medium-term / Features" below.

---

## Medium-term

### Features

- [x] **Chat templates** — DONE: `ChatTemplate` (ChatML detect/render from GGUF metadata) + `ChatSession` (system/user/assistant turns), used by `Demo/AgentDemo` and the chat tests.
- [x] **`OverfitClient` facade** — DONE 2026-05-29. `LanguageModels/OverfitClient.cs`. `using var client = OverfitClient.LoadGguf(path); client.AddSystem("..."); var reply = client.Send("...");` — wraps GgufReader+ChatTemplate.Detect, tokenizer auto-pick (QwenChatTokenizer first if `vocab.json+merges.txt` present, else HuggingFaceBpeTokenizer fallback), engine+session creation, ChatSession with sensible ChatML stop sequences, and Greedy GenerationOptions default. Sync `Send` + async `SendAsync` (thread-pool wrap). Exposes underlying `Chat` for constrained outputs / streaming. Tests: 2 fast (null/missing-path guards) + 1 [LongFact] e2e on real Qwen (mechanical assertion — content quality is a separate semantic concern).
- [ ] **ONNX: LSTM/GRU operators** — enables recurrent model import.
- [x] **Depthwise Conv** (group=channels) — MobileNet-style models. DONE 2026-05-27: `DepthwiseConv2DLayer` + `TensorMath.DepthwiseConv2D` (SIMD AXPY inner kernel, padding/stride/bias, FD-verified). Pair with a 1×1 `ConvLayer` for a full separable block.
- [ ] Standalone Softmax and CrossEntropy in addition to fused loss.
- [ ] **`GroundedAnswerCache` — safe semantic answer cache for RAG** (post-launch, per
  arXiv:2605.27494, see [Research inputs](#research-inputs-papers-reviewed-2026-05-21)).
  Wraps `VectorStore` + `BertEncoder` + `WordPieceTokenizer` into a 4-gated answer cache
  with a per-entry record `{Q embedding, retrieved doc IDs, version hashes, cached answer,
  answer-token-set}`. Public API: `cache.TryGet(query, retrievedDocs, currentVersionMap,
  out answer)` admits a cached answer only when ALL four gates pass — (1) query cosine
  ≥ τ, (2) retrieved-doc-ID overlap ≥ τ, (3) version map matches recorded, (4) lexical
  support (Jaccard of answer-tokens ∩ fresh-evidence-tokens ≥ τ). Expose **USR**
  (unsafe-served-rate) as an observability counter — same metric the paper uses. Requires
  small extension to `VectorStore.Add(id, vector, payload, version?)` to carry version
  hash per stored doc. Scope ~200-400 LoC, AOT-clean. Strategic fit: regulated-industries
  story gets a *measurable safety* primitive, not just speed. Differentiator vs
  Microsoft.SemanticKernel / LangChain.NET (neither ship a gated semantic cache today).

### Agentic loop primitives (from `D:\Agentic-AI-LangGraph` review, 2026-05-27)

The agentic stack today has the *primitives* (tool calling, JSON-constrained / structured output, streaming,
multi-turn `ChatSession` + sliding-window memory, embeddings) but **no reusable agent loop on top**. A review of a
LangGraph patterns repo (17 recipes) surfaced four small, AOT-friendly, in-character additions that build *on* the
existing primitives — NOT a graph/agent framework (state-graph runtime, multi-agent supervisor, HITL gate,
time-travel, parallel fan-out, thread checkpointing, agentic-RAG router + vector store are deliberately **out of
scope** per "What Overfit is not trying to be" — a user builds those on the primitives):

- [x] **ReAct / tool-use agent loop** — DONE 2026-05-29. `LanguageModels/Agents/ReActAgent.cs` is a driver
      over `ChatSession` + `ToolCallConstraint` that loops *model → tool-call → observe → repeat* until
      the model calls the auto-registered synthetic `finish({answer:...})` tool or hits `MaxSteps` (the
      circuit-breaker overlap). Includes the `ExtraMaskedTokensConstraint` shim that masks the chat
      session's stop-token id (e.g. Qwen's `<|im_end|>` ≠ `<|endoftext|>`) while the envelope is
      incomplete — the constraint's own EOS-mask alone wasn't enough to prevent mid-envelope truncation.
      6 unit tests (`ReActAgentTests`) drive the pure loop via the internal `RunLoop(askForToolCall fn)`
      hook — synthetic replies, no model needed: single-tool-then-finish, multi-step observation
      chaining, step cap, unknown tool, handler-throws, non-string answer fallback. An e2e
      `[LongFact]` is wired but Qwen 2.5-3B Q4_K_M is below the reliability floor for multi-turn
      constrained tool calling (greedy → unclosed JSON strings; temperature → Unicode garbage); ship
      with a bigger / instruction-tuned model.
- [x] **Self-reflection / critic loop** — DONE 2026-05-29. `LanguageModels/Agents/CriticLoop.cs`,
      model-agnostic (`Func<string,string>` generator + `Func<string,CriticVerdict>` critic), reflects
      previous-attempt + critic feedback into next prompt. Built on `CircuitBreaker` so cap and
      timeout come for free. 3 unit tests.
- [x] **Circuit breaker / step guard** — DONE 2026-05-29. `LanguageModels/Agents/CircuitBreaker.cs`:
      generic `Run<T>(maxIterations, maxElapsed?, iterate, isAccepted)` → `CircuitBreakerResult<T>`
      with `Accepted` / `MaxIterations` / `Timeout` outcomes. Used by `CriticLoop`. 4 unit tests.
- [x] **Summarizing memory** — DONE 2026-05-29. `LanguageModels/Memory/`: pure
      `ChatHistoryCompactor.Plan(history, summarizeAtChars, recentTurnsToKeep)` returning a
      `CompactionPlan`; model-driven `SummarizingChatSession` wraps a `ChatSession` and triggers
      compaction in-line on `Send`. Rebuilds as `[system…] + [running summary] + [recent N turns
      verbatim]`. Uses save/restore around summarisation so the prompt doesn't leak. Required adding
      `ChatSession.AddUser` / `AddAssistant` (history seeding — useful generally for transcript
      replay). 5 unit tests on the compactor; e2e left as host-driven (needs a model).

### Distribution

- [ ] NuGet package metadata polish.
- [ ] Sample Blazor app showing streaming generation in browser via Rx/IAsyncEnumerable adapter.
- [ ] Benchmark page: a Format × Model × RAM × tokens/s table.

---

## Long-term ideas

- Graph compilation for fixed-shape training/inference graphs.
- Custom autograd operators with explicit forward/backward registration.
- Mixed precision training.
- Data loading and preprocessing pipeline improvements.
- Optional GPU backend investigation without compromising CPU-first design.
- Model/dataset packages outside the small core runtime.

---

## What Overfit is not trying to be

- Not a general-purpose replacement for PyTorch or TensorFlow.
- Not a Python shim.
- Not GPU-first.
- Not a model zoo in the core package.

The differentiator remains pure C#, predictable memory behavior, Native-AOT compatibility, and competitive CPU inference (including small/medium language models) on consumer hardware.

---

## Contributing

Performance-sensitive PRs should include:

- correctness tests;
- before/after BenchmarkDotNet output;
- allocation measurements;
- documentation updates when public behavior changes.

License: GNU AGPLv3. For commercial licensing, contact devonbike@gmail.com.
