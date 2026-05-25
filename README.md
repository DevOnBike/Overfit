# Overfit

Pure C# deep-learning and LLM inference engine. Predictable CPU performance,
explicit memory ownership, zero-allocation inference hot paths.

**No native binaries. No Python runtime. No ONNX Runtime dependency.** Native-AOT compatible.

<p align="center">
  <img src="docs/assets/overfit-features.png"
       alt="Overfit at a glance — pure-C#/.NET LLM engine: load any GGUF (Q4_K/Q6_K), run and fine-tune LLMs (LoRA), in-process agentic stack (RAG, tool calling, guaranteed JSON), adaptive anomaly detection, zero-allocation, Native-AOT — no Python, no native binary, no server."
       width="520">
</p>

---

## What it does

**Run and fine-tune LLMs, and run deep networks, entirely in .NET — in-process, allocation-free.**

- **Zero-allocation CPU inference** — preallocated buffers, no per-call GC pressure. **0 bytes per token** on KV-cache decode, enforced by a build-breaking CI assertion.
- **LLM inference** — GPT-2, and Qwen / Llama / Mistral from GGUF (incl. `Q4_K_M` straight from Ollama). Qwen2.5-3B runs in **~3.2 GB RAM — matching llama.cpp's footprint** — with native K-quant kernels. **Memory-mapped weights** (default) keep the model off the managed heap: a 3B Q4_K_M loads with a **~220 MB live managed heap**, the weights paged in as shared/clean file pages.
- **In-process agentic stack** — embeddings + a built-in **vector store** (in-process cosine RAG, no external DB), **guaranteed-valid JSON output**, and **tool / function calling** (constrained decoding forces a valid call, dispatched to your C# delegate). No prompt-and-pray, no retry/repair — the grammar is enforced at the logit level so an invalid token can't be sampled. Runnable [agent demo](Demo/AgentDemo/README.md): load → RAG → tool call → JSON, one process, no Python.
- **LoRA fine-tuning** — adapt a frozen base to your own data: LM head, FFN, and per-head attention (Stages 1–3), training + inference both in pure C#.
- **Anomaly detection** — train a small GPT on *your* metrics, flag anomalies, adapt per deployment with LoRA. Nothing that just *runs* others' models can do.
- **ONNX import** — load PyTorch-exported models directly (14 operators, ResNet-style skip connections), output matches PyTorch within 1e-4.

→ **Full quick-start, benchmarks, import pipelines and architecture: [`docs/TECHNICAL.md`](docs/TECHNICAL.md).**

---

## Quick start

```bash
dotnet add package DevOnBike.Overfit
```

```csharp
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Contracts;

using var gpt2    = Gpt2.LoadSmall(@"C:\gpt2");      // or CachedLlamaInferenceEngine.Load("model.gguf")
using var session = gpt2.CreateSession();
session.Reset(gpt2.Tokenizer.Encode("The future of software development is"));

var sampling = SamplingOptions.Greedy;
for (var i = 0; i < 32; i++)                          // 0 bytes allocated per token
{
    var token = session.GenerateNextToken(in sampling);
    Console.Write(gpt2.Tokenizer.DecodeToken(token));
}
```

Already have models in Ollama? Run any GGUF in two lines:
`CachedLlamaInferenceEngine.Load(path)` → `CreateSession()`. More examples
(ONNX, training, anomaly detection) in [`docs/TECHNICAL.md`](docs/TECHNICAL.md).

---

## Benchmarks (headline)

AMD Ryzen 9 9950X3D · Windows 11 · .NET 10 · BenchmarkDotNet 0.15.8. Full tables in [`docs/TECHNICAL.md`](docs/TECHNICAL.md).

| Workload | Result | Allocation |
|----------|--------|-----------:|
| Single inference Linear(784→10) | **7.6× faster** than ONNX Runtime | **0 B** |
| GPT-2 Small KV-cache decode | 6.5× faster than naive O(N²); parity 10/10 vs PyTorch | **0 B / token** |
| Qwen2.5-3B Q4_K_M decode (same file vs llama.cpp) | ~19 tok/s @ **3.20 GB** (RAM parity); llama.cpp ~1.5× faster | **1 B / token** vs 21 KB |
| Concurrent inference (8 threads) | **3.6× faster** than ONNX Runtime | **0 B** |

**Honest positioning:** llama.cpp/LLamaSharp decodes ~1.5× faster (hand-tuned native
AVX-512/VNNI); on training, PyTorch CPU is ~2.2–3.6× faster (Intel MKL). Overfit's
axis is pure-managed, zero-allocation, AOT-compatible, no-native-dependency
execution — *not* raw matmul throughput. It matches llama.cpp on RAM and wins
~20,000× on per-token allocation.

---

## Why not just use…?

| Tool | The right choice when… | Reach for Overfit when… |
|------|------------------------|--------------------------|
| **ML.NET** | Classical ML on tabular data. | You need transformer / LLM inference or deep networks. |
| **ONNX Runtime** | You have ONNX models and accept a native dependency. | You want pure-managed, zero-allocation inference, no native binary. *(Overfit imports ONNX too.)* |
| **llama.cpp / Ollama** | A standalone CPU LLM server as a separate process. | You want the model **inside** your .NET process — no sidecar, no IPC, no exposed server. |
| **LLamaSharp** | Mature, GPU-capable default; bundling a native llama.cpp binary is fine. | You can't ship a native binary: AOT-strict, regulated/locked-down, supply-chain-audited, or zero-allocation hot paths. |
| **PyTorch** | Training and research; large models; GPU. | Deploying inference into a .NET app without the Python stack. |
| **OpenAI / Anthropic APIs** | Best quality, zero infra, data egress acceptable. | Data egress is **not** acceptable — regulated, on-prem, no third-party calls. |

---

## Requirements

.NET 10+ · no native dependencies · no Python runtime · Native AOT compatible.

---

## Roadmap

LoRA Stage 1/2/3 ✅ · GGUF Q4_K/Q6_K in-RAM decode ✅ · GPT-2 / Qwen / Llama / Mistral
inference ✅ · anomaly-detection + production base ✅. Open: closing the decode gap to
llama.cpp (full-matrix attention), batched prefill (B>1). Full
priorities and the live resume-point: [`ROADMAP.md`](ROADMAP.md).

---

## What Overfit is not

Not a PyTorch/TensorFlow replacement. Not GPU-first. Not transformer-scale-first.

**Not a hosted SaaS, and not an API.** Overfit runs as a library inside your own
process — no service to call, no API key, nothing sent anywhere during inference.
If you need AI inference where data never leaves your boundary, by construction,
that is exactly the point. See [Overfit for regulated industries](docs/scenarios/regulated-industries.md).

---

## Licensing

Dual-licensed.

- **Open source — GNU AGPLv3.** Free in production **provided your project is released under a compatible open-source licence** (Overfit links as a library, so AGPL copyleft extends to your application).
- **Commercial licence.** Closed-source product? AGPL won't work — a commercial licence removes the copyleft obligation. See [`COMMERCIAL.md`](COMMERCIAL.md) or contact **devonbike@gmail.com**.

The simple test: if you can't or won't release your application under AGPLv3, you need the commercial licence.
