# Where Overfit wins — the underserved niche for 2026

**Thesis.** Almost every LLM tool in 2026 assumes Python, a GPU, a separate model server, and data leaving your
process. The fast-growing demand is the exact opposite: AI that runs **inside an existing .NET application, on a
CPU, on-premises, with nothing leaving the box** — and that you can **fine-tune, test, and audit** like normal
software. That intersection is nearly empty on the market. Overfit is built for it.

This document maps the market forces, the empty quadrant, and the concrete cases Overfit fills.

---

## Why the niche is real (and growing) in 2026

- **Regulation forces local.** GDPR, the EU AI Act (in force 2026), DORA (EU financial sector), sectoral rules in
  health / gov / defense — sensitive data can't be shipped to a hosted LLM API. "Send it to OpenAI" is a
  non-starter for a large class of buyers.
- **Most enterprises are .NET / JVM shops, but AI tooling is Python.** Banks, insurers, government, industrial,
  ISVs run on .NET. Bolting a Python sidecar onto a .NET service means a second runtime, a second container, a
  second security surface, cross-process latency, and a polyglot deploy nobody wants to own.
- **Agentic AI is moving to production** — and production needs reliability, reproducibility, audit trails, and
  regression tests, not a demo notebook.
- **GPU scarcity and cloud-inference cost** push high-volume *internal* workloads (classification, extraction,
  routing, RAG) toward CPU + on-prem.
- **"Sovereign AI" / air-gapped AI** is a 2026 procurement theme for gov and critical infrastructure.

## The empty quadrant

| | Python (PyTorch / HF / vLLM) | Ollama / llama.cpp | Cloud LLM APIs | **Overfit** |
|---|:--:|:--:|:--:|:--:|
| Runs **in-process in .NET** (NuGet, no server) | ✗ | ✗ (separate process) | ✗ | **✓** |
| **No Python**, no extra runtime to deploy | ✗ | ✓ (but C++/Go binary) | ✓ | **✓** |
| **No data egress** (fully on-prem) | ✓ | ✓ | ✗ | **✓** |
| **CPU fine-tuning** (teach it your data) | GPU only | ✗ (no training) | hosted only | **✓** |
| **Testable / regression-gated RAG** | rare | ✗ | ✗ | **✓** |
| **Single native binary** (AOT, air-gap) | ✗ | partial | n/a | **✓** |
| **Reproducible/auditable** (greedy = bit-identical) | possible | possible | ✗ | **✓** |

Overfit's edge is **not** raw tokens/sec (llama.cpp is ~1.13× faster single-stream). It's the *whole column*:
in-process .NET + on-prem + trainable + testable + auditable + air-gap-deployable.

---

## The cases (what Overfit solves that's scarce)

### 1 · The regulated .NET shop that can't use the cloud
**Who:** banks, insurers, healthcare, government, legal.
**Pain today:** they want chat/RAG/agents over internal documents, but compliance forbids sending data to a hosted
LLM, and their stack is .NET — Python tooling is a foreign body.
**Overfit delivers:** RAG, chat, tool-calling and guaranteed-JSON **inside the .NET process** (`OverfitClient` +
`VectorStore` + constrained decoding). Nothing leaves the box; the model is a NuGet dependency, not a service.
**Why scarce:** the only "local" options are separate servers (Ollama) or Python — neither is in-process .NET.

### 2 · Air-gapped / sovereign AI
**Who:** defense, critical infrastructure, classified/secure environments.
**Pain today:** no internet, no Python package supply chain, no GPU, strict provenance — most stacks can't even be
installed.
**Overfit delivers:** a **single Native-AOT binary** (`overfit`, ~7.8 MB, no .NET runtime) or one DLL; loads a
local GGUF; runs offline. Pure-managed, hand-rolled loaders (no ONNX Runtime, no native deps to vet).
**Why scarce:** a self-contained, dependency-free, no-Python LLM binary you can drop on an air-gapped host is rare.

### 3 · "Teach the model our private knowledge" — without a GPU farm
**Who:** any org with proprietary jargon, policies, product data, internal procedures.
**Pain today:** fine-tuning means CUDA + Python + a GPU box + an MLOps pipeline; most .NET teams can't run it.
**Overfit delivers:** **QLoRA fine-tuning on a CPU, in pure .NET** — frozen 4-bit GGUF base (never expanded) +
a trainable LoRA, ~3 GB RAM for a 3B model, turnkey `QLoRAFineTuner`. Validated: taught a model a made-up fact, it
then recites it.
**Why scarce:** **llama.cpp can't train at all; PyTorch-QLoRA needs CUDA + Python.** On-CPU, in-.NET training is a
genuine moat.

### 4 · Embed AI in an existing .NET product (ISV)
**Who:** software vendors adding AI features to a desktop/server/SaaS product.
**Pain today:** shipping a Python sidecar (or a cloud dependency) with your product means your customers must
deploy/secure/scale a second thing — and you've added a runtime to support.
**Overfit delivers:** add `DevOnBike.Overfit` (and `.Extensions.AI`) as a **NuGet reference**; the AI ships *inside*
your assembly. One process, one deploy, one support surface.
**Why scarce:** there is no mainstream "LLM as a NuGet you call in-proc" — that's the gap.

### 5 · Testable, auditable RAG for compliance
**Who:** regulated QA, legal, anyone who must *defend* an AI answer.
**Pain today:** RAG is a black box tested by eyeballing answers; a doc edit silently breaks retrieval and nobody
notices until a wrong answer ships.
**Overfit delivers:** the **RAG Stability Harness** — expected-source recall, paraphrase stability, false-premise
traps, corpus lint, with `RagAssert` turning retrieval quality into a **CI gate**. Plus reproducible answers
(greedy decode = bit-identical) and cited sources.
**Why scarce:** almost nobody sells "your RAG is under regression test". See [`rag-testing.md`](rag-testing.md).

### 6 · Replace a Python / ONNX / Ollama sidecar
**Who:** teams already running a polyglot AI sidecar next to a .NET service.
**Pain today:** extra container, cross-process hop, a second language/runtime/security surface, version drift.
**Overfit delivers:** delete the sidecar — same chat/embeddings/RAG in-process. OpenAI-compatible endpoints if you
still want HTTP. Maps to COMMERCIAL.md "Python / ONNX-Sidecar Replacement".
**Why scarce:** the in-process .NET replacement for the sidecar didn't exist.

### 7 · Edge / on-device / desktop AI
**Who:** industrial PCs, kiosks, desktop apps (WPF/WinForms), field devices, low-RAM hosts.
**Pain today:** no cloud, no GPU, can't install Python, tight RAM.
**Overfit delivers:** a 3B model in **~220 MB live heap, 1 byte/token**, mmap weights, runs on modest CPUs; ships
as one AOT binary. In-process speech-to-text (Whisper) too.
**Why scarce:** low-RAM, no-Python, in-app AI for the edge is underserved.

### 8 · Kill the per-token cloud bill on internal workloads
**Who:** anyone doing high-volume *internal* classification / extraction / routing / summarization.
**Pain today:** every call to a hosted API is metered and egresses data; latency is a network hop.
**Overfit delivers:** run it locally for the cost of your own CPU — structured extraction with **guaranteed schema**
(JSON-Schema constrained decoding), no parse failures, no data leaving.
**Why scarce:** local structured-output with schema *guarantees*, in .NET, is rare.

### 9 · Reproducible AI for audit / litigation
**Who:** finance, legal, insurance — anyone who may need to *prove* what the AI did.
**Pain today:** hosted models are non-deterministic and opaque; you can't reproduce or attest a past answer.
**Overfit delivers:** **greedy decode is bit-identical** (same input → same output), plus model hash, retrieved
sources, and a retrieval regression test — an auditable, reproducible pipeline.
**Why scarce:** reproducibility + provenance as a first-class property is unusual.

### 10 · Sovereign-language / domestic models in .NET
**Who:** organizations needing a strong *local-language* model on-prem (e.g. **Bielik** for Polish).
**Pain today:** the good local-language GGUFs run via Python/Ollama, not in their .NET stack — and English embedders
mis-retrieve their language.
**Overfit delivers:** loads Bielik (validated) and Qwen/Llama-class multilingual GGUFs; for RAG it can embed with
the **chat model's own hidden states** (multilingual, no separate embedder). Whole pipeline in .NET, on-prem.
**Why scarce:** in-process .NET + sovereign-language LLM + multilingual RAG is a thin slice of the market.

---

## What Overfit deliberately is *not*

Being honest sharpens the niche:
- **Not** the raw tokens/sec leader (llama.cpp is ~1.13× faster single-stream; the gap is DRAM-bandwidth-bound).
- **Not** a GPU framework, a training-from-scratch-of-giant-models platform, or a multi-tenant inference cluster.
- **Not** a vision-language / diffusion / TTS engine (yet).

If your problem is "serve 10k QPS on a GPU fleet", buy something else. If it's "put a trainable, testable,
auditable LLM *inside* my .NET app, on-prem, with no Python and no data egress" — that's the column above.

## Buyer map (→ COMMERCIAL.md)

| Scenario | Package |
|---|---|
| 1, 5, 10 (regulated RAG/agent, on-prem) | **Private .NET RAG/Agent PoC** |
| 4, 6 (embed in product / kill the sidecar) | **Python / ONNX-Sidecar Replacement** |
| 7, 9 (edge RAM/latency, reproducibility audit) | **Zero-GC Inference Audit** |
| 3 (teach-it-your-data) | QLoRA engagement (standalone commercial license + support) |

---

*One .NET process — no Python, no native binary, no model server, no data egress. Trainable, testable, auditable.*
