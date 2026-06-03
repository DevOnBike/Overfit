# Overfit — release infographic

Drop-in English copy for a release infographic: the hero line, badges, nine capability cards (short
scannable bullets), the footer line, and an image-generator prompt. Product: **DevOnBike.Overfit** — a
pure-C# / .NET 10 local-AI engine.

---

## Hero

**Overfit — Local AI for .NET, in-process, on the CPU.**

Run, fine-tune and build agents on real LLMs — and transcribe speech — entirely in managed .NET.
No Python. No GPU. No native binary. No model server. No data egress.

## Badges

`Pure .NET 10` · `Native AOT` · `0 B/token decode` · `In-process` · `No Python` · `No GPU` ·
`No native binary` · `No data egress` · `AGPL-3.0 / commercial`

---

## Capability cards (drop-in infographic copy)

**1 · LLM inference**
- Qwen2.5, Llama 2 / 3.x, Mistral
- Mixtral & Qwen-MoE (Mixture-of-Experts)
- Q4_K_M / Q6_K / Q8_0 / F16 / BF16
- KV-cache, 0 B/token decode

**2 · Load anything — no server**
- GGUF (memory-mapped)
- HuggingFace safetensors (sharded)
- ONNX (linear + ResNet-style DAG)
- Tokenizers read straight from GGUF

**3 · Chat & agents**
- Streaming multi-turn chat
- C# tool calling (constrained decoding)
- Guaranteed-valid JSON output
- ReAct agent loop

**4 · In-process RAG**
- Built-in cosine vector store
- MiniLM / BGE / E5 embeddings
- Bit-parity validated vs HuggingFace
- Grounded answers with cited sources

**5 · CPU fine-tuning (QLoRA)**
- Fine-tune a 4-bit GGUF on the CPU
- Frozen quantized base + trainable LoRA
- ~3 GB RAM for a 3B model
- Portable adapters — no GPU, no Python

**6 · Speech-to-text**
- Whisper (tiny / base) in pure C#
- ~65× real-time on CPU
- English + Polish, live microphone
- Bluestein-FFT log-mel + KV-cache decode

**7 · Pure-C# audio**
- MP3 decoder (MPEG-1 / 2 / 2.5 Layer III)
- WAV reader + resampler
- Zero per-frame allocation
- ~160× real-time — feeds Whisper directly

**8 · Zero-allocation engine**
- ~0 bytes/token on decode
- Native AOT compatible
- SIMD + multi-threaded kernels
- mmap weights — tiny managed heap

**9 · Classic ML**
- CNNs for vision (MNIST)
- OCR (CRNN + CTC)
- Anomaly detection & forecasting
- Train models from scratch on CPU

## Footer

**One .NET process — no Python, no native binary, no model server, no data egress.**
Dual-licensed: AGPL-3.0-or-later / commercial.

> Honest note (small print): llama.cpp is still ~1.13× faster for raw single-stream CPU decode
> (DRAM-bandwidth-bound). Overfit's edge is pure-managed in-process .NET, Native AOT, low allocation
> pressure — plus on-CPU QLoRA fine-tuning and in-process speech-to-text.

---

## Image-generator prompt

> A clean, modern technical infographic for a software product called **"Overfit"** — a pure-C# / .NET
> local AI engine. Landscape orientation, professional developer-marketing style, generous whitespace, a
> crisp grid of rounded capability cards. Color theme: deep slate / near-black background with .NET-style
> violet–blue accents (#512BD4 → #5B8DEF) and a single warm highlight; subtle monospace typographic accents.
>
> **Top hero band:** the wordmark "Overfit" with the tagline *"Local AI for .NET — run, fine-tune and talk
> to real LLMs entirely in-process, on the CPU."* Below it a row of small pill badges: `No Python`,
> `No GPU`, `No native binary`, `No data egress`, `Native AOT`, `0 B/token`.
>
> **Center:** a 3×3 grid of equal cards, each with a simple line icon and a short title:
> 1. *LLM inference* (chip) — "Qwen · Llama · Mistral · Mixtral · MoE",
> 2. *Load anything* (download) — "GGUF · safetensors · ONNX",
> 3. *Chat & agents* (chat-bubble + gear) — "tool calling · guaranteed JSON",
> 4. *In-process RAG* (documents + magnifier) — "embeddings + vector store",
> 5. *QLoRA fine-tuning* (sliders) — "fine-tune a 4-bit GGUF on CPU",
> 6. *Speech-to-text* (microphone + waveform) — "Whisper, 65× real-time",
> 7. *Pure-C# audio* (music note) — "MP3 / WAV decode, zero-alloc",
> 8. *Zero-allocation engine* (gauge) — "0 B/token, Native AOT",
> 9. *Classic ML* (network nodes) — "CNN · OCR · anomaly · forecasting".
>
> **Bottom band:** a thin strip reading *"One .NET process — no Python, no native binary, no model server,
> no data egress"* plus a small *"Dual-licensed: AGPL-3.0 / commercial"* note. Flat vector style, consistent
> stroke weight on icons, high contrast, readable at small sizes, no photographic elements, no clutter.

### Prompt variants
- **Light theme** — white background, same violet–blue accents, for embedding in the README / docs.
- **Vertical (social)** — stack the hero, a single column of the nine cards, then the footer, for LinkedIn / X.
- **Minimal** — just the hero band + the nine card titles as one row of icons, for the repo header image.
