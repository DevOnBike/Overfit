# Which models does Overfit load?

**Overfit loads by *architecture family*, not by model name.** A GGUF (or HuggingFace safetensors) file whose
architecture is `llama`, `qwen2`, `qwen2moe`, `mistral`, or Mixtral-style MoE loads and generates — which covers
the **vast majority of Ollama's library and the HuggingFace GGUF finetunes**, including thousands of community
fine-tunes of those bases. If the architecture isn't supported, the loader throws a clear error naming it (it
never silently produces garbage).

Legend: **✅ validated** (run end-to-end in our tests) · **🟢 same architecture** (loads via the family; not
separately validated) · **❌ not yet** (different architecture — see the roadmap list at the bottom).

---

## Supported architecture families

| Family | `general.architecture` | Notes |
|---|---|---|
| **Qwen2 / Qwen2.5** | `qwen2` | 0.5B–72B, GQA + RoPE + SwiGLU + RMSNorm |
| **Llama 2 / 3.x** | `llama` | incl. RoPE scaling on 3.x |
| **Mistral** | `llama` / `mistral` | 7B and arch-compatible variants |
| **Mixtral (MoE)** | `llama` + expert tensors | routed experts, top-2, no shared expert |
| **Qwen1.5-MoE** | `qwen2moe` | routed experts + sigmoid-gated shared expert |
| **GPT-2 / GPT-1** | (safetensors / .bin) | byte-parity vs PyTorch |
| **BERT (embeddings)** | (safetensors) | WordPiece — MiniLM / BGE-en / E5-en |

---

## Popular chat / instruct LLMs

| Model (popular) | Overfit | Why |
|---|:--:|---|
| **Llama 3.1 / 3.2 / 3.3** (1B–70B) | 🟢 | `llama` arch (3.2-1B/3B validated ✅) |
| **Llama 2** (7B/13B/70B) | 🟢 | `llama` arch |
| **Qwen2.5** (0.5B–72B) | ✅ | `qwen2` — validated 0.5B/1.5B/3B |
| **Qwen2.5-Coder / -Math** | 🟢 | `qwen2` arch |
| **Mistral 7B** (v0.1–0.3) | ✅ | validated |
| **Mistral Nemo / Small** | 🟢 | Mistral arch |
| **Mixtral 8×7B** | ✅ | validated (25 GB, pure C#) |
| **Mixtral 8×22B** | 🟢 | same MoE arch |
| **Qwen1.5-MoE-A2.7B** | ✅ | `qwen2moe` — validated |
| **Bielik 4.5B / 11B** (Polish) | ✅ | Qwen2/Mistral-class — Bielik-4.5B validated |
| **DeepSeek-R1-Distill-Qwen / -Llama** | 🟢 | distills are `qwen2` / `llama` arch — **they load** |
| **CodeLlama · TinyLlama · Vicuna · WizardLM** | 🟢 | `llama` finetunes |
| **Nous-Hermes · OpenHermes · Dolphin · Zephyr** | 🟢 | Llama / Mistral finetunes |
| **Yi (6B/9B/34B) · SOLAR 10.7B** | 🟢 | `llama`-compatible arch |
| **Gemma 1 / 2 / 3** | ❌ | `gemma` arch (different) |
| **Phi-3 / Phi-3.5 / Phi-4** | ❌ | `phi` arch |
| **DeepSeek-V2 / V3 / R1 (native MoE)** | ❌ | `deepseek2` arch (≠ the Qwen/Llama distills above) |
| **Qwen3 / Qwen3-MoE** | ❌ | RoPE recognised, but QK-norm not handled — not loaded |
| **Command-R (Cohere) · Granite · Falcon · StableLM · StarCoder2 · DBRX · Grok** | ❌ | own architectures |
| **Llama 4 · Mamba / Jamba · RWKV · T5 / FLAN · GLM** | ❌ | not yet (see roadmap) |

> Rule of thumb: **if `ollama show <model>` (or the GGUF metadata) reports architecture `llama`, `qwen2`,
> `qwen2moe`, or `mistral`, Overfit loads it.** The big misses are Gemma, Phi, native DeepSeek-MoE, Qwen3, and
> Command-R.

---

## Embedding models (RAG)

| Model | Overfit | Why |
|---|:--:|---|
| **all-MiniLM-L6-v2** | ✅ | BERT/WordPiece — bit-parity vs HF (`overfit pull minilm`) |
| **BGE-small/base/large-en-v1.5** | ✅ | BERT/WordPiece (`overfit pull bge`) |
| **E5-small/base/large-v2** | ✅ | BERT/WordPiece (`overfit pull e5`) |
| **GTE-* (English, BERT)** | 🟢 | BERT/WordPiece variants |
| **multilingual-e5 · bge-m3 · paraphrase-multilingual** | ❌ | XLM-RoBERTa / **SentencePiece** tokenizer (not supported) |
| **nomic-embed-text** | ❌ | custom BERT variant (rotary) |

**Multilingual RAG workaround (no extra model):** embed with the loaded chat GGUF's own hidden states —
`OverfitClient.Embed(text)` (e.g. Bielik for Polish), with mean-centering for the anisotropy. Good for ranking;
see `docs/rag-testing.md`.

---

## Speech / other

| Model | Overfit | Why |
|---|:--:|---|
| **Whisper tiny / base** | ✅ | whisper.cpp `ggml-*.bin` — validated (JFK clip) |
| **Whisper small / medium / large** | 🟢 | same arch (slower on CPU) |
| **LLaVA · Qwen2-VL · Pixtral · MiniCPM-V** (vision) | ❌ | no VLM stack yet |

---

## Formats & quantization

- **Files:** GGUF (memory-mapped), HuggingFace safetensors (single + sharded), Overfit `.bin`, ONNX (linear + DAG).
- **Quant:** `Q4_K_M` · `Q6_K` · `Q8_0` · `Q5_0` · `Q5_K` · `F32` · `F16` · `BF16`. Others (`Q2_K`, `Q3_K`, `IQ*`)
  throw a clear `NotSupportedException`.
- **Tokenizers:** read straight from the GGUF (`tokenizer.ggml.*`, SentencePiece + BPE) or sibling
  `tokenizer.json` / `vocab.json`+`merges.txt`. (BERT embedders use `vocab.txt` WordPiece.)

---

## Not yet supported (on the roadmap)

Architectures llama.cpp lists that Overfit doesn't load yet: **Gemma 1/2/3**, **Phi**, **DeepSeek 2/V3 (native MoE)**,
**Qwen3 / Qwen3-MoE**, **Command-R (Cohere)**, **Granite**, **Falcon**, **StableLM**, **StarCoder2**, **DBRX**,
**Grok**, **Llama-4**, **Mamba/Mamba2/Jamba**, **RWKV6/7**, **T5/FLAN**, **GLM**, plus the multilingual XLM-R
embedders and all vision-language models. Adding a new chat family is typically 3–5 days (weight-name mapping +
arch quirks); XLM-R embedders ~1.5–2 weeks (SentencePiece-Unigram tokenizer).

## How to check a specific file

```powershell
overfit pull <hf-owner/repo>      # or: overfit chat C:\path\model.gguf
```

If the architecture is unsupported, Overfit fails fast with a message naming the architecture or quant — it never
loads a model it can't run correctly.
