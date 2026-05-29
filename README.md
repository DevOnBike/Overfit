# Overfit

**Private LLMs, RAG, C# tool calling and guaranteed JSON inside your .NET app.**

<p align="center">
  <img src="docs/assets/overfit-features.png"
       alt="Overfit at a glance — run, fine-tune and build agents on real LLMs entirely in .NET: pure-managed engine (0 B/token, Native AOT), load any GGUF/ONNX model memory-mapped, run GPT-2/Qwen/Llama, in-process agentic stack (RAG, tool calling, guaranteed JSON), LoRA fine-tuning, adaptive anomaly detection, runs in-process so data never leaves, CPU-first — no Python, no native binary, no model server."
       width="560">
</p>

Overfit lets .NET teams add local AI features without Python, Ollama, a model server,
native binaries, or data leaving the process.

Use it when you need AI inside an existing ASP.NET, WPF, Blazor, desktop, on-prem
or air-gapped .NET product — especially when external LLM APIs are blocked by
security, compliance, latency, deployment, or supply-chain constraints.

```bash
dotnet add package DevOnBike.Overfit
```

---

## Start here: run the ASP.NET local agent demo

The fastest way to understand Overfit is to run the local ASP.NET agent demo.

```bash
dotnet run -c Release --project Demo/LocalAgentAspNetDemo
```

It exposes a private AI agent over HTTP:

| Endpoint | What it shows |
|---|---|
| `GET /health` | Model/service health |
| `POST /chat` | Local chat over a GGUF or safetensors model |
| `POST /documents/index` | In-process document indexing |
| `POST /rag/query` | RAG over local documents |
| `POST /agent` | C# tool calling with constrained JSON |
| `POST /chat/json` | Guaranteed-valid JSON output |
| `GET /metrics` | Prometheus-style runtime metrics |

The demo shows the full path:

```text
local model file
  -> memory-mapped load
  -> RAG over local documents
  -> C# tool call
  -> guaranteed JSON
  -> metrics
```

No Python. No Ollama. No model server. No network call. The model is a file on
disk; the agent is a singleton inside ASP.NET.

See [`Demo/LocalAgentAspNetDemo`](Demo/LocalAgentAspNetDemo/README.md).

---

## What Overfit gives you

### 1. Private local LLMs in .NET

Load Qwen, Llama, Mistral, Mixtral and related GGUF / safetensors models directly
from C#. The model runs inside your process, not behind a server.

```csharp
using DevOnBike.Overfit.LanguageModels;

using var client = OverfitClient.LoadGguf(@"C:\models\qwen2.5-3b-instruct-q4_k_m.gguf");

client.AddSystem("You are concise.");
var reply = client.Send("Explain zero-allocation decode in one sentence.");

Console.WriteLine(reply);
```

### 2. In-process RAG

Embed documents, store vectors in-process, retrieve context and answer from your
own data without an external vector database, embedding API, Python service, or
sidecar process.

Supported embedding paths include MiniLM, BGE and E5-style BERT encoders, with
vectors validated against HuggingFace / PyTorch reference outputs.

### 3. C# tool calling and guaranteed JSON

Constrained decoding forces valid JSON and valid tool-call envelopes, then
dispatches the call to your C# delegate.

No regex parsing of free text. No retry-on-bad-JSON loop. No prompt-and-pray.

```json
{
  "name": "create_ticket",
  "arguments": {
    "customerEmail": "sam@brightlabs.example",
    "subject": "Failed SSO login",
    "priority": "high"
  }
}
```

### 4. Zero-allocation inference hot paths

Overfit is built around predictable CPU inference, Native AOT compatibility,
explicit memory ownership and near-zero per-token allocations on the decode path.

The goal is not to beat hand-tuned native GPU/AVX runtimes on raw throughput.
The goal is to make local AI deployable as a normal .NET library in environments
where Python, native binaries, sidecars, external APIs and hidden allocations are
not acceptable.

---

## What you can build today

### Private AI assistant inside your app

Add a local assistant to a desktop, WPF, Blazor, ASP.NET, console or internal
enterprise app. Use a local Qwen/Llama/Mistral model and keep all prompts,
documents and outputs inside your process.

### Document Q&A and semantic search

Index support tickets, policy documents, product docs, invoices or internal
notes. Query them by meaning with in-process embeddings and vector search.

### Action-taking agents

Register C# tools such as:

- `lookup_customer`
- `create_ticket`
- `send_invoice`
- `summarize_document`
- `classify_case`
- `extract_fields`

The model chooses the tool and emits a constrained JSON call. Your C# code
executes the action.

### Structured extraction

Use guaranteed-valid JSON to extract intent, fields, summaries, routing metadata
or decision records without post-hoc repair loops.

### Audit-friendly local AI

Run deterministic greedy decoding, file-versioned weights, local decision logs,
input/output records, model hashes and timestamps. This fits teams that need
controlled deployment boundaries and explainable operational records.

---

## Why teams use Overfit

| If this is your problem... | Overfit's value |
|---|---|
| You have a .NET product and cannot send data to OpenAI/Anthropic | Run the model locally inside your process |
| You do not want to operate Python, Ollama or a model server | Ship a NuGet package and a model file |
| Your environment blocks native binaries or sidecars | Pure C# runtime, Native AOT compatible |
| You need RAG, tool calls and JSON, not just raw token generation | Built-in agentic stack |
| You care about allocations, P99 latency and GC behavior | Explicit memory ownership and zero-allocation hot paths |
| You need a commercial path for closed-source products | Dual licensing: AGPLv3 or commercial |

---

## Quick start

Install the package:

```bash
dotnet add package DevOnBike.Overfit
```

Run a local GGUF model:

```csharp
using DevOnBike.Overfit.LanguageModels;

using var client = OverfitClient.LoadGguf(@"C:\models\qwen2.5-3b-instruct-q4_k_m.gguf");

client.AddSystem("You are a concise assistant.");
var reply = client.Send("What is Overfit useful for?");

Console.WriteLine(reply);
```

Run the full ASP.NET local-agent demo:

```bash
dotnet run -c Release --project Demo/LocalAgentAspNetDemo
```

Run the console walkthrough:

```bash
dotnet run -c Release --project Demo/AgentDemo
```

More details:

- [`Demo/LocalAgentAspNetDemo`](Demo/LocalAgentAspNetDemo/README.md) — ASP.NET local agent
- [`Demo/AgentDemo`](Demo/AgentDemo/README.md) — console walkthrough: load -> RAG -> tool call -> JSON
- [`docs/TECHNICAL.md`](docs/TECHNICAL.md) — architecture, benchmarks, import pipelines
- [`ROADMAP.md`](ROADMAP.md) — current engineering priorities

---

## Benchmarks: honest headline

Test machine for current headline numbers: AMD Ryzen 9 9950X3D, Windows 11,
.NET 10, BenchmarkDotNet 0.15.8.

| Workload | Result | Allocation |
|---|---:|---:|
| Single inference `Linear(784 -> 10)` | ~7.6x faster than ONNX Runtime | 0 B |
| GPT-2 Small KV-cache decode | ~6.5x faster than naive O(N²), parity vs PyTorch | 0 B/token |
| Qwen2.5-3B Q4_K_M decode | ~19 tok/s, RAM footprint in the llama.cpp range | ~1 B/token |
| Concurrent inference, 8 threads | ~3.6x faster than ONNX Runtime | 0 B |

Honest positioning:

- llama.cpp / LLamaSharp are faster for raw CPU LLM decode.
- PyTorch CPU is faster for large-scale training.
- ONNX Runtime is mature and fast if native dependencies are acceptable.
- Overfit's axis is pure-managed .NET, in-process deployment, Native AOT,
  low allocation pressure and no native model server.

Full benchmark tables and caveats live in [`docs/TECHNICAL.md`](docs/TECHNICAL.md).

---

## Supported model families

### Language models

| Family | Verified sizes / variants | Loader | Quantization / dtype |
|---|---|---|---|
| Qwen2.5 | 0.5B / 3B / 7B / 14B / 32B | GGUF, HF safetensors, `.bin` | F32, F16, BF16, Q8_0, Q4_K_M, Q6_K |
| Llama-2 / Llama-3.x | Llama-3.2-1B onwards | GGUF, HF safetensors | F32, F16, BF16, Q8_0, Q4_K_M, Q6_K |
| Mistral 7B | 7B | GGUF | F32, F16, BF16, Q8_0, Q4_K_M |
| Qwen1.5-MoE A2.7B | 14B total / 2.7B active | GGUF | Q8_0, Q4_K_M |
| Mixtral-8x7B | 47B total / 13B active | GGUF | Q8_0, Q4_K_M |
| GPT-2 small | 124M | `.bin`, HF safetensors | F32 |
| GPT-1 | configurable | `.bin`, trained from scratch | F32 |

### Embeddings

| Model | Pooling | Validation |
|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | mean + L2 | parity vs HuggingFace/PyTorch |
| `BAAI/bge-small-en-v1.5` | CLS + L2 | parity vs HuggingFace/PyTorch |
| `intfloat/e5-small-v2` | mean + L2 | parity vs HuggingFace/PyTorch |

### Other workloads

| Area | Status |
|---|---|
| ONNX import | Linear and DAG topology, ResNet-style skip connections |
| Computer vision | MNIST CNN, Conv/BN/ReLU/Pool/FC-style networks |
| OCR | CRNN + CTC pipeline for synthetic digits / lexicon words |
| LoRA | LM head, FFN and per-head attention stages |
| Anomaly detection | Small GPT-style models for metrics and deployment-specific adaptation |

---

## Loading formats

Overfit loads models directly in managed .NET:

- GGUF with mmap-backed weights
- K-quant formats including Q4_K_M and Q6_K
- Q8_0, F32, F16 and BF16
- HuggingFace safetensors, including sharded directories
- Overfit `.bin` checkpoints
- ONNX models for supported operator sets

Tokenizers include HuggingFace ByteLevel-BPE, Qwen ChatML-aware handling,
GGUF tokenizer fallback, WordPiece and GPT-2 byte-level BPE.

---

## Why not just use...

| Tool | Use it when... | Reach for Overfit when... |
|---|---|---|
| ML.NET | You need classical ML on tabular data | You need transformer / LLM inference or deep networks inside .NET |
| ONNX Runtime | Native dependencies are acceptable | You want pure-managed, AOT-clean, low-allocation inference |
| llama.cpp / Ollama | A standalone LLM process/server is fine | You want the model inside your .NET process |
| LLamaSharp | Bundling native llama.cpp is acceptable | You cannot ship native binaries or need zero-allocation hot paths |
| PyTorch | Research, large training, GPU workflows | You want deployment inside a .NET app without Python |
| OpenAI / Anthropic APIs | Data egress is acceptable | Data must stay inside your boundary |

---

## Commercial integration

If you have a .NET system and need a private AI feature in production, the
fastest path is a fixed-scope integration.

### Private .NET RAG / Agent PoC

A local LLM + RAG + C# tool-calling proof of concept in your infrastructure.

Typical deliverables:

- ASP.NET endpoints: `/chat`, `/rag/index`, `/rag/query`, `/tools`, `/health`, `/metrics`
- local model selection and deployment
- document ingestion and vector search
- constrained JSON / tool-call flow
- benchmark report on your hardware
- deployment handover

### Python / Ollama / ONNX sidecar replacement

Move inference into the .NET process and compare P50/P99 latency, RAM, allocation
pressure and operational complexity against the existing sidecar.

### Zero-GC inference audit

Profile your current .NET inference hot path — Overfit, ML.NET, ONNX Runtime or
custom code — and identify allocation, GC, AOT and P99 latency risks.

Commercial licenses and monthly support retainers are also available.

See [`COMMERCIAL.md`](COMMERCIAL.md) or contact **devonbike@gmail.com**.

---

## Requirements

- .NET 10+
- CPU-first runtime
- no Python runtime
- no native runtime dependency for Overfit itself
- Native AOT compatible paths are guarded in CI

---

## What Overfit is not

Overfit is not a PyTorch or TensorFlow replacement.

It is not GPU-first. It is not transformer-scale-first. It is not a hosted SaaS,
not an API and not a model server.

Overfit is a .NET library that runs inside your process. If you need best-quality
frontier models, maximum GPU throughput or a hosted API, use a hosted model or a
GPU-first runtime.

If you need private local AI inside an existing .NET product, Overfit is built
for that.

---

## Roadmap

Current shipped areas include:

- GGUF Q4_K / Q6_K decode
- Qwen / Llama / Mistral / Mixtral inference
- GPT-2 / GPT-1 support
- in-process RAG
- tool calling
- guaranteed JSON
- LoRA stages
- ONNX import
- anomaly detection
- Native AOT guard

Current priorities include:

- closing part of the decode gap to llama.cpp
- batched prefill
- stronger JSON Schema constraints
- more model families and quantization formats
- broader ASP.NET / Microsoft.Extensions.AI / Aspire integration

See [`ROADMAP.md`](ROADMAP.md).

---

## Licensing

Overfit is dual-licensed.

### Open source: GNU AGPLv3

Free in production if your project is released under a compatible open-source
license. Overfit links as a library, so AGPL copyleft extends to the application.

### Commercial license

Use this for closed-source products, SaaS, regulated deployments, proprietary
internal tools, or any application that cannot be released under AGPLv3.

The simple test:

> If you cannot or will not release your application under AGPLv3, you need the
> commercial license.

Full license text: [`LICENSE.md`](LICENSE.md).  
Commercial terms and support: [`COMMERCIAL.md`](COMMERCIAL.md).  
Contact: **devonbike@gmail.com**.
