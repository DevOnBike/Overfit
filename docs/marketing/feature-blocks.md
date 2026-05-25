# Overfit — feature blocks (infographic source)

High-level building blocks for an infographic. Each block = one panel: icon hint, headline,
2–3 tight points. Numbers are measured (AMD Ryzen 9 9950X3D · Win11 · .NET 10), not aspirational.

**One-line positioning:** *Run, fine-tune, and build agents on real LLMs — entirely in .NET, in-process. No Python. No native binary. No server.*

---

## 1 · Pure-managed engine  🧩
- 100% C# / .NET 10 — no Python runtime, no native binary, no ONNX Runtime
- Native-AOT compatible (single-file deploy; LINQ/reflection banned in the hot path, CI-enforced)
- Zero-allocation inference: **0 bytes per token** on KV-cache decode (build fails on regression)

## 2 · Load any model, no conversion  📦
- GGUF straight from Ollama / HuggingFace — F32 / F16 / BF16 / **Q8_0 / Q4_K / Q6_K**
- Safetensors (Llama / Qwen) + ONNX import (14 ops, ResNet-style skip connections)
- **Memory-mapped weights** — a 3B Q4_K_M loads with a **~220 MB managed heap** (weights file-backed)

## 3 · Run real LLMs  🧠
- GPT-2, and Qwen / Llama / Mistral families — GQA, RoPE, SwiGLU
- KV-cache decode, O(N) not O(N²); PyTorch parity 10/10 top-10 logits
- Qwen2.5-3B Q4_K_M ≈ **3.2 GB footprint — matching llama.cpp** · sampling: greedy / top-k / top-p / Min-P

## 4 · In-process agentic stack  🤖
- **RAG** — embeddings + a built-in in-process vector store (cosine top-K, no external DB)
- **Tool / function calling** — model calls your C# delegates; the call is *valid by construction*
- **Guaranteed JSON** — structured output enforced at the logit level (no prompt-and-pray, no repair)

## 5 · Train & fine-tune  🎯
- **LoRA** fine-tuning end-to-end in C# — LM head, FFN, per-head attention (Stages 1–3)
- Full training path: autograd tape, Adam / SGD, in-place adapter merge (zero kernel changes)
- Parallel-runtime sprint: GPT-1 training step **−72%** (414 ms → 114 ms)

## 6 · Adaptive anomaly detection  📈
- Train a small GPT on *your* metrics, flag anomalies — not just running others' models
- **Per-deployment LoRA adaptation** — flatten false positives on one pod without touching others
- Operator-gated lifecycle: recommend → adapt → isolate, live

## 7 · Where it runs  🛡️
- In-process library — no sidecar, no IPC, **data never leaves your boundary**
- CPU-first, no GPU required — scales down to edge / IoT
- Built for regulated / on-prem / air-gapped .NET shops

## 8 · Honest positioning  ⚖️
- Not GPU-first, not a PyTorch replacement, not raw-GEMM-fastest
- llama.cpp decodes ~1.5–2× faster (hand-tuned native AVX-512); Overfit matches it on RAM
- The axis: predictable, allocation-free, zero-dependency, AOT execution inside your .NET process

---

**Key numbers strip:** `0 B/token` · `7.6× vs ONNX (single)` · `3.6× vs ONNX (8 threads)` · `3B in ~220 MB heap` · `−72% training step` · `100% C#`

**Footer:** Open source (AGPLv3) + commercial · **DevOnBike/Overfit** on GitHub
