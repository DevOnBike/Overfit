# Overfit — release infographic

Drop-in English copy for a release infographic: the hero line, badges, twelve capability cards (short
scannable bullets), a proof-numbers strip, the footer line, and an image-generator prompt. Product:
**DevOnBike.Overfit** — a pure-C# / .NET 10 local-AI engine.

---

## Hero

**Overfit — Local AI for .NET, in-process, on the CPU.**

Run, fine-tune and build agents on real LLMs, serve an OpenAI-compatible API, do testable RAG, and
transcribe speech — entirely in managed .NET. No Python. No GPU. No native binary. No model server.
No data egress.

## Badges

`Pure .NET 10` · `Native AOT` · `0 B/token decode` · `In-process` · `OpenAI-compatible` ·
`Microsoft.Extensions.AI` · `No Python` · `No GPU` · `No native binary` · `No data egress` ·
`AGPL-3.0 / commercial`

---

## Capability cards (drop-in infographic copy)

**1 · LLM inference**
- Qwen2.5 (0.5B–32B), Llama 2 / 3.x, Mistral
- Mixtral & Qwen-MoE (Mixture-of-Experts)
- Q4_K_M / Q6_K / Q8_0 / Q5_0 / Q5_K / F16 / BF16
- KV-cache · optional Q8 KV (~4× less KV RAM) · 0 B/token

**2 · Load anything — no server**
- GGUF (memory-mapped, ~220 MB heap for 3B)
- HuggingFace safetensors (sharded)
- ONNX (linear + ResNet-style DAG)
- Tokenizers read straight from GGUF — 100% Python-free

**3 · Chat & agents**
- Streaming multi-turn chat
- C# tool calling (constrained decoding)
- Guaranteed JSON + **JSON-Schema** & **regex** constrained output
- ReAct loop · critic loop · circuit breaker · summarizing memory

**4 · In-process RAG**
- Built-in cosine vector store, zero-dependency
- MiniLM / BGE / E5 embeddings — bit-parity vs HuggingFace
- Multilingual embeddings from the chat GGUF itself
- Grounded answers with cited sources

**5 · Testable RAG (the differentiator)**
- Expected-source recall@K + MRR
- Paraphrase-stability + false-premise traps
- Corpus linter: duplicates · orphans · short chunks
- `RagAssert` → gate retrieval quality in CI

**6 · OpenAI-compatible & .NET-native**
- `/v1/chat/completions` (+ SSE stream), `/v1/embeddings`, `/v1/models`
- Dependency-free server (HttpListener) — no ASP.NET, AOT-clean
- `Microsoft.Extensions.AI` adapter — `IChatClient` / `IEmbeddingGenerator`
- Drops into Semantic Kernel & the M.E.AI ecosystem

**7 · Global `overfit` CLI**
- `pull` · `list` · `chat` · `serve` — one self-contained binary
- HuggingFace / mirror / direct-URL download
- SHA-256 verified · resumable · `HF_ENDPOINT` mirror
- `overfit serve` → the OpenAI API in one command

**8 · CPU fine-tuning (QLoRA)**
- Fine-tune a 4-bit GGUF on the CPU
- Frozen quantized base + trainable LoRA
- ~3 GB RAM for a 3B model
- Portable adapters — no GPU, no Python (the llama.cpp can't-do)

**9 · Speech-to-text**
- Whisper (tiny / base) in pure C#
- ~65× real-time on CPU
- English + Polish, live microphone
- log-mel front-end + KV-cache decode

**10 · Pure-C# audio**
- MP3 decoder (MPEG-1 / 2 / 2.5 Layer III)
- WAV reader + resampler
- Zero per-frame allocation
- ~160× real-time — feeds Whisper directly

**11 · Zero-allocation engine**
- ~0 bytes/token on decode
- Native AOT — one 7.8 MB native binary, no .NET runtime
- SIMD + multi-threaded kernels
- mmap weights — tiny managed heap

**12 · Classic ML & training**
- CNNs for vision (MNIST), OCR (CRNN + CTC)
- Anomaly detection & forecasting
- Gradient checkpointing (24× RAM cut) · data-parallel trainer
- Train models from scratch on CPU

## Proof numbers (strip)

| Live heap, Qwen-3B Q4_K_M | Alloc / token | GPT-2 vs PyTorch | Native CLI | Embedding parity | Test suite |
|:--:|:--:|:--:|:--:|:--:|:--:|
| **220 MB** | **1 byte** | **byte-parity** | **7.8 MB** | **cosine ≈ 1.0** | **1139 / 0** |

## Footer

**One .NET process — no Python, no native binary, no model server, no data egress.**
Dual-licensed: AGPL-3.0-or-later / commercial.

> Honest note (small print): llama.cpp is still ~1.13× faster for raw single-stream CPU decode
> (DRAM-bandwidth-bound). Overfit's edge is pure-managed in-process .NET, Native AOT, low allocation
> pressure — plus on-CPU QLoRA fine-tuning, an OpenAI-compatible server, testable RAG, and in-process
> speech-to-text.

---

## Image-generator prompt

> A clean, modern technical infographic for a software product called **"Overfit"** — a pure-C# / .NET
> local AI engine. Landscape orientation, professional developer-marketing style, generous whitespace, a
> crisp grid of rounded capability cards. Color theme: deep slate / near-black background with .NET-style
> violet–blue accents (#512BD4 → #5B8DEF) and a single warm highlight; subtle monospace typographic accents.
>
> **Top hero band:** the wordmark "Overfit" with the tagline *"Local AI for .NET — run, fine-tune, serve
> and talk to real LLMs entirely in-process, on the CPU."* Below it a row of small pill badges: `No Python`,
> `No GPU`, `No native binary`, `No data egress`, `OpenAI-compatible`, `Native AOT`, `0 B/token`.
>
> **Center:** a 4×3 grid of twelve equal cards, each with a simple line icon and a short title:
> 1. *LLM inference* (chip) — "Qwen · Llama · Mistral · Mixtral · MoE",
> 2. *Load anything* (download) — "GGUF · safetensors · ONNX",
> 3. *Chat & agents* (chat-bubble + gear) — "tool calling · JSON-Schema",
> 4. *In-process RAG* (documents + magnifier) — "embeddings + vector store",
> 5. *Testable RAG* (checklist + magnifier) — "recall · stability · CI gate",
> 6. *OpenAI-compatible* (plug) — "/v1 API · Extensions.AI",
> 7. *overfit CLI* (terminal) — "pull · chat · serve",
> 8. *QLoRA fine-tuning* (sliders) — "fine-tune a 4-bit GGUF on CPU",
> 9. *Speech-to-text* (microphone + waveform) — "Whisper, 65× real-time",
> 10. *Pure-C# audio* (music note) — "MP3 / WAV decode, zero-alloc",
> 11. *Zero-allocation engine* (gauge) — "0 B/token, Native AOT",
> 12. *Classic ML & training* (network nodes) — "CNN · OCR · QLoRA · train on CPU".
>
> **Bottom band:** a thin strip reading *"One .NET process — no Python, no native binary, no model server,
> no data egress"* plus a small *"Dual-licensed: AGPL-3.0 / commercial"* note. Flat vector style, consistent
> stroke weight on icons, high contrast, readable at small sizes, no photographic elements, no clutter.

### Prompt variants
- **Light theme** — white background, same violet–blue accents, for embedding in the README / docs.
- **Vertical (social)** — stack the hero, a single column of the twelve cards, then the footer, for LinkedIn / X.
- **Minimal** — just the hero band + the twelve card titles as two rows of icons, for the repo header image.
