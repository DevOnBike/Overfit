# Overfit — technical reference

Detailed quick-start examples, full benchmark tables, import pipelines and
architecture. The root [`README.md`](../README.md) is the short overview; this is
the deep dive. Roadmap and priorities live in [`../ROADMAP.md`](../ROADMAP.md).

---

## Quick start

### Inference — native model

```csharp
using DevOnBike.Overfit.Inference;

var model = new Sequential(
    new LinearLayer(784, 128),
    new ReluActivation(),
    new LinearLayer(128, 10));

model.Load("model.bin");
model.Eval();

using var engine = InferenceEngine.FromSequential(model, inputSize: 784, outputSize: 10);

Span<float> input  = stackalloc float[784];
Span<float> output = stackalloc float[10];
engine.Run(input, output); // zero-allocation
```

### Inference — GPT-2 Small (KV-cache)

**One-command demo** (after running `python Scripts/convert_gpt2.py --size small --out models/`):

```bash
dotnet run -c Release --project Demo/Gpt2ConsoleDemo -- \
    --prompt "The future of software development is" --tokens 64
```

Output reports the headline numbers separately from the rest of the loop:

```text
GPT-2 Small
  prompt:    "The future of software development is"
  tokens:    64
  KV-cache:  enabled

The future of software development is in the hands of the people. …

--- Inference only (GenerateNextToken) ---
  Tokens/sec:            71.4
  Managed bytes / token: 0.0  (total: 0 B)        ← the headline claim

--- Full demo loop (inference + decode + Console.Write) ---
  Tokens/sec:            71.2
  (string + console alloc dominates; not part of the 0 B / token claim)
```

**API** (what the demo wraps):

```csharp
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Contracts;

// One handle owns model + KV-cache engine + BPE tokenizer.
// Directory convention: gpt2_small.bin + vocab.json + merges.txt under modelDir.
using var gpt2    = Gpt2.LoadSmall(@"C:\gpt2");
using var session = gpt2.CreateSession();

session.Reset(gpt2.Tokenizer.Encode("The future of software development is"));

// Generate — 0 bytes allocated per token after session creation.
var sampling = SamplingOptions.Greedy;
for (var i = 0; i < 32; i++)
{
    var token = session.GenerateNextToken(in sampling);
    Console.Write(gpt2.Tokenizer.DecodeToken(token));
}
// → " in the hands of the people."
```

For non-standard layouts (shared tokenizer across sizes, custom filenames) drop
to the explicit `Gpt2.Load(modelPath, vocabPath, mergesPath, config)`, or skip
the facade entirely and compose `new GPT1Model(Gpt2Config.Small)` +
`CachedSlmInferenceEngine.FromGpt1(model)` directly.

### Inference — Qwen / Llama / Mistral from a GGUF (Ollama / HuggingFace)

```csharp
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Contracts;

// Loads F32/F16/BF16/Q8_0/Q4_K/Q6_K directly — including Q4_K_M from Ollama.
using var engine  = CachedLlamaInferenceEngine.Load(@"C:\qwen3b\qwen.q4km.gguf");
using var session = engine.CreateSession(maxContextLength: 128);

session.Reset(promptTokens);
var token = session.GenerateNextToken(in SamplingOptions.Greedy); // 0 B / token
```

Weights are **memory-mapped by default** (`mmap: true`): the verbatim K-quant weights and the
quantized token-embedding table are file-backed (shared/clean pages), so a 3B `Q4_K_M` model loads
with a **~220 MB live managed heap** instead of copying the whole model onto it. Pass `mmap: false`
to force the managed-copy path (e.g. when the file must not stay open).

### Embeddings + in-process vector store (RAG)

```csharp
using DevOnBike.Overfit.LanguageModels.Retrieval;

// Pooled, L2-normalised sentence embedding from the same model (Mean or LastToken pooling).
float[] vec = session.Embed(tokenizer.Encode("the cat sat on the mat"));

var store = new VectorStore(session.EmbeddingDimension);
store.Add("doc1", session.Embed(tokenizer.Encode(doc1Text)), payload: doc1Text);
store.Add("doc2", session.Embed(tokenizer.Encode(doc2Text)), payload: doc2Text);

VectorMatch[] hits = store.Search(session.Embed(tokenizer.Encode(query)), topK: 3); // cosine
var best = hits[0].Payload;   // hits[i].Score is the true cosine similarity
```

`VectorStore` is in-process and dependency-free — a flat dot-product scan with top-K insertion
(no external vector DB). Sized for app/document-set scale, not billion-scale ANN.

### Structured output — guaranteed well-formed JSON

```csharp
using DevOnBike.Overfit.LanguageModels.Constraints;
using DevOnBike.Overfit.LanguageModels.Chat;

var chat = new ChatSession(session, tokenizer, template);
var reply = chat.Send("Return a person object with name and age.", in options,
                      constraint: new JsonGrammarConstraint(tokenizer));

using var doc = JsonDocument.Parse(reply);   // never throws — grammar enforced at the logit level
```

The constraint (`ITokenConstraint`) masks every token that would break well-formed JSON to `-∞`
before sampling, so the model cannot emit invalid output — no prompt-engineering, no retry/repair.

### Function / tool calling

```csharp
using DevOnBike.Overfit.LanguageModels.Tools;

var tools = new[] { new ToolDefinition("get_weather", "Weather for a city.") };
var reply = chat.Send("Weather in Paris?", in options,
                      constraint: new ToolCallConstraint(tools, tokenizer));

if (ToolCall.TryParse(reply, out var call))      // always a valid {"name","arguments"} envelope
{
    using var args = JsonDocument.Parse(call.Arguments);
    var result = myHandlers[call.Name](args.RootElement);  // dispatch to your C# delegate
}
```

The tool name is constrained to the registered set and the arguments to well-formed JSON. End-to-end
runnable example (load → RAG → tool call → JSON in one process): [`Demo/AgentDemo`](../Demo/AgentDemo/README.md).

### Inference — ONNX import (linear topology)

```csharp
using DevOnBike.Overfit.Onnx;
using DevOnBike.Overfit.Inference;

var model = OnnxImporter.Load("classifier.onnx"); // .data file resolved automatically
model.Eval();

using var engine = InferenceEngine.FromSequential(model, inputSize: 784, outputSize: 10);
var prediction = engine.Predict(input); // ReadOnlySpan<float>, 0 B
```

### Inference — ONNX import (DAG topology — ResNet, skip connections)

```csharp
using DevOnBike.Overfit.Onnx;
using DevOnBike.Overfit.Inference;

// OnnxGraphImporter handles branching graphs: skip connections, residual blocks.
// OnnxImporter (above) requires linear topology and is faster for simple models.
var dagModel = OnnxGraphImporter.Load("resnet.onnx", inputSize: 784, outputSize: 10);
dagModel.Eval();

var backend = new OnnxGraphInferenceBackend(dagModel);
using var engine = InferenceEngine.FromBackend(backend);
var prediction = engine.Predict(input); // ReadOnlySpan<float>, 0 B
```

### Training

```csharp
using var conv  = new ConvLayer(1, 8, 28, 28, 3);
using var fcHid = new LinearLayer(1352, 64);
using var fcOut = new LinearLayer(64, 10);

using var optimizer = new Adam(
    conv.TrainableParameters()
        .Concat(fcHid.TrainableParameters())
        .Concat(fcOut.TrainableParameters()),
    learningRate: 0.001f) { UseAdamW = true };

using var graph = new ComputationGraph();

for (var batch = 0; batch < batches; batch++)
{
    graph.Reset();
    optimizer.ZeroGrad();

    using var h  = conv.Forward(graph, input);
    using var a  = graph.Relu(h);
    using var p  = graph.MaxPool2D(a, 8, 26, 26, 2);
    using var pF = graph.Reshape(p, batchSize, 1352);
    using var hH = fcHid.Forward(graph, pF);
    using var hA = graph.Relu(hH);
    using var lo = fcOut.Forward(graph, hA);

    using var loss = graph.SoftmaxCrossEntropy(lo, target);
    graph.Backward(loss);
    optimizer.Step();
}
```

### Anomaly detection (train a GPT on your own metrics)

```csharp
using DevOnBike.Overfit.Anomalies.Gpt;
using DevOnBike.Overfit.LanguageModels.Runtime;

using var handle   = SlmRuntimeFactory.CreateGpt1(model);   // your trained base
using var detector = new GptAnomalyDetector(handle);

var score = detector.Score(snapshot);   // MetricSnapshot from Prometheus / CSV
if (score.Score > threshold)
    Alert($"anomaly in {score.WorstMetric}");
```

Runnable end-to-end demo: `Demo/AnomalyConsoleDemo` (train a base on a metrics CSV,
then live-score a stream + an injected incident). Per-deployment LoRA adaptation
(`Gpt1LoRAFineTuner`, with optional rehearsal-lite to avoid forgetting the base
"normal") drives a benign regime's score toward zero while still catching real
incidents.

---

## Benchmark snapshot

Machine: AMD Ryzen 9 9950X3D · Windows 11 25H2 · .NET 10.0.8 · BenchmarkDotNet 0.15.8

### Qwen2.5-3B-Instruct (Q4_K_M) — single-stream CPU decode, vs llama.cpp

**Latest verified benchmark** (canonical source of truth — any number elsewhere in the repo should reconcile to this):

| Field | Value |
|---|---|
| Date | 2026-05-21 |
| Model file | `qwen.q4km.gguf` (Qwen2.5-3B-Instruct, Q4_K_M GGUF) |
| Hardware | AMD Ryzen 9 9950X3D · Windows 11 25H2 · DDR5 |
| Runtime | .NET 10.0.8 · BenchmarkDotNet 0.15.8 · single stream |
| LLamaSharp version | 0.27.0 (native llama.cpp under the hood) |

| Engine | Decode | RAM | Alloc / token |
|--------|-------:|----:|--------------:|
| **Overfit** (pure C#) | ~19 tok/s | **3.20 GB** | **1 B** |
| LLamaSharp (native llama.cpp) | ~29 tok/s | 3.20 GB | 21,220 B |

llama.cpp decodes ~1.5× faster (hand-tuned native AVX-512/VNNI) — stated honestly. Overfit matches its RAM footprint, allocates ~20,000× less per token (zero-alloc decode), and runs in-process: no native binary, no sidecar, AOT-compatible. The remaining decode gap is effective memory bandwidth (~55 % vs ~80 % of peak), rooted in per-head vs full-matrix attention — see `ROADMAP.md` option D and the "Decode throughput catch-up vs llama.cpp" section for the staged catch-up plan.

(Decode tok/s drifts with thermal state; the ~1.5× ratio is the stable figure. Pre-mmap numbers from earlier in the perf sprint — 17.2 tok/s @ 4.40 GB heap-allocated — are historical and should not be cited as current; mmap-backed K-quant weights closed the RAM gap and lifted decode to ~19 tok/s on the same hardware.)

### GPT-1 training step — parallel runtime sprint

End-to-end training step of a 4-layer GPT-1 (dModel=128, dFF=512, seqLen=128), measured by `GPT1CpuUtilizationProbeTests`:

| BatchSize | Wall/step before | Wall/step after | Cores effective | GELU bwd | LayerNorm bwd |
|---:|---:|---:|---:|---:|---:|
| 8 | 126 ms | **65 ms** (−48%) | 3.48 → **10.40** / 32 | 448 → 67 ms (7×) | 209 → 26 ms (8×) |
| 16 | 217 ms | **74 ms** (−66%) | 2.99 → **11.44** / 32 | 882 → 37 ms (24×) | 403 → 23 ms (17×) |
| 32 | 414 ms | **114 ms** (−72%) | 3.89 → **13.85** / 32 | 1774 → 42 ms (42×) | 800 → 37 ms (22×) |

Real-workload validation on TinyShakespeare (`TinyShakespeareTrainingTests`, 2-layer GPT-1, char-level, 300 steps): **~60–120 s → ~2 s (30–60× faster)**, loss drops 5.04 → 3.23 (35.9% improvement), zero numerical regression vs sequential reference (numerical gradient check passes).

Drivers:
- **`OverfitParallel`** — bulk-wake dispatcher (`SemaphoreSlim.Release(N)` instead of N × `AutoResetEvent.Set`). ~5 µs warm dispatch, 0 B/call.
- **GELU forward + backward** — re-written as `OverfitParallel.For` over chunks + `TensorPrimitives` SIMD pipeline (`Multiply` → `Add` → `Tanh` → …) on stackalloc'd 1024-element tiles.
- **LayerNorm forward + backward** — parallel-per-row with per-worker partial accumulators for `dGamma` / `dBeta` (stackalloc'd by caller, SIMD merge after parallel pass).
- **Scaled-Dot-Product Attention forward** — parallel-over-batch (symmetric to existing backward parallel path). For multi-head training the SDPA call sees effective batch `B × H` so even single-batch training parallelizes across heads.
- **`LinearKernels.BackwardInput` / `AccumulateWeightGrad`** — migrated from `Parallel.For` to `OverfitParallel`.
- **`[module: SkipLocalsInit]`** — assembly-wide; elides per-frame zero-init on 21+ `stackalloc` sites.

**Where this lands vs PyTorch CPU.** The same training step in PyTorch 2.11 (CPU,
MKL, 16 threads) takes 17.9 / 29.1 / 52.8 ms at batch 8 / 16 / 32. PyTorch is
still **~2.2–3.6× faster** — its GEMM is Intel MKL/oneDNN, decades of hand-tuned
AVX-512 assembly that a pure-C# kernel does not out-run. What the parallel sprint
did is **close the gap from ~7–8× to ~2.2–3.6×** (pre-sprint Overfit was
126 / 217 / 414 ms). Overfit's axis is pure-managed, zero-allocation,
Native-AOT-compatible execution with no native or Python dependency — not raw
GEMM throughput. Reproduce the comparison:
`python Sources/Benchmark/Helpers/benchmark_pytorch_gpt1_training.py`.

### Single inference — Overfit vs ONNX Runtime

| Method | Mean | Allocated | vs ONNX Runtime |
|--------|-----:|----------:|----------------:|
| **Overfit `InferenceEngine`** | **250.7 ns** | **0 B** | **7.6× faster** |
| ONNX Runtime (pre-allocated) | 1 899 ns | 224 B | baseline |
| ONNX Runtime (standard) | 3 388 ns | 952 B | 0.56× |

Model: Linear(784→10). Overfit is **7.6× faster** than ONNX Runtime pre-allocated path, **13.5× faster** than standard path, with zero managed allocations.

### GPT-2 Small KV-cache inference

| Method | MaxNewTokens | Mean | Allocated |
|--------|-------------|-----:|----------:|
| Legacy (full forward/token) | 64 | 6 318 ms | 62.0 MB |
| **KV-cache** | **64** | **973 ms** | **74.1 MB\*** |
| Legacy (full forward/token) | 128 | OOM | — |
| **KV-cache** | **128** | **1 916 ms** | **74.1 MB\*** |

\* The KV-cache `Allocated` is **one-time session-creation cost** (KV buffers,
sized for the full context) — it is **constant: 74.1 MB at 16, 64 and 128
tokens alike**. **Per-token decode allocation = 0 bytes** — verified on every
`dotnet test -c Release` by `Demo_Gpt2Small_KvCacheDecode_AllocatesZeroBytesPerToken`.
The legacy path's allocation instead **grows with token count** (15.9 MB at 16
→ 62.0 MB at 64 → OOM at 128).

Model: GPT-2 Small (124M params, 12 layers, 12 heads, d=768, vocab=50257).

- **6.5× faster** at 64 tokens. Legacy path OOMs at 128 tokens; KV-cache handles it cleanly.
- O(N) scaling vs O(N²) for the naive path.
- Parity: top-10 logit overlap **10/10** vs PyTorch reference, maxAbsDiff = **0.000107** (float32 noise floor).

```
"The future of software development is in the hands of the people."
"In C#, the best way to handle memory is to use the C# compiler."
"Kubernetes pod anomaly detection works by detecting the presence of a pod."
```

### DAG inference — ResNet-style model with skip connections

| Method | Mean | Allocated |
|--------|-----:|----------:|
| `OnnxGraphModel.RunInference` (direct) | ~1.0 µs | **0 B** |
| `InferenceEngine.FromBackend` (via engine) | ~0.9 µs | **0 B** |

Model: TinyResNet — Linear(8→8) + skip + Linear(8→4). Both paths: zero allocations.
Sub-µs math at this model size — timer resolution dominates, run-to-run variance is high.

### CNN training throughput (60k MNIST, batch=64)

| Epoch | Time | Alloc/epoch | Notes |
|------:|-----:|------------:|-------|
| 1 | ~1.6 s | ~32 MB | JIT warmup |
| 2–5 | **~775 ms** | **~26 MB** | steady state, post-`OverfitParallel` migration |

5-epoch run: **5551 → 4870 ms (−12% wall, −18% total CPU)** vs pre-migration baseline. Cores effective: 6.81 → 7.03 of 32 (MNIST CNN at this scale is Amdahl-limited by sequential graph/optimizer slices — the bigger win is on transformer workloads, see GPT-1 section above).

Training allocations from autograd graph temporaries — expected.
Inference path: zero allocations. Live managed memory delta per epoch: **−0.01 MB** (zero leak).

### Concurrent inference (8 threads × 1 000 calls each)

| Method | Mean | Allocated | vs ONNX Runtime |
|--------|-----:|----------:|----------------:|
| **Overfit (concurrent)** | **522.0 ms** | **0 B** | **3.6× faster** |
| ONNX Runtime (concurrent) | 1 894.0 ms | 117 MB | baseline |

Overfit scales linearly — no shared mutable state, no lock contention.
ONNX Runtime allocates 117 MB of managed memory under concurrent load (Gen0 GC pressure).

---

## GPT-2 import

```
HuggingFace openai-community/gpt2
  → Scripts/convert_gpt2.py --size small --out test_fixtures/
  → GPT1Model(Gpt2Config.Small).Load("gpt2_small.bin")
  → CachedSlmInferenceEngine.FromGpt1(model)
  → session.GenerateNextToken(in sampling)   // 0 B per token
```

Weight conversion script downloads from HuggingFace, splits the fused `c_attn` matrix into per-head Q/K/V weights (including biases), and saves in Overfit binary format.

Available configs: `Gpt2Config.Small` (124M), `Gpt2Config.Medium` (355M), `Gpt2Config.Large` (774M), `Gpt2Config.XL` (1.5B).

### KV-cache architecture

The `CachedSlmInferenceEngine` uses a zero-copy weight strategy:

- `SingleHeadWeights` — holds `TensorStorage<float>` references for Q/K/V/O weights and biases of one attention head. No data copied from the model.
- `BlockWeights` — aggregates all weights for one transformer layer (layer norms, per-head attention, FFN).
- `StackWeights` — holds `BlockWeights[]` for all layers plus final norm and LM head.

`CachedGpt1ModelAdapter` binds these structs to the live model storage at session creation. `RefreshWeightsFromModel()` is a no-op for in-place updates (e.g. LoRA) — spans already point to the updated data.

Decode call chain:

```
session.GenerateNextToken()
  → adapter.DecodeNextToken()
  → stack.Decode(hidden, weights, cache, position, logits)
    → block.Decode(input, in blockWeights, cache, layerIndex, position, output)
      → mha.Decode(hidden, in blockWeights, cache, layerIndex, position, output)
        → head.Decode(hidden, wq, wk, wv, bq, bk, bv, wo, cache, ...)
```

All weight parameters are `ReadOnlySpan<float>` obtained from `TensorStorage` at decode time — no allocations in the hot path.

---

## ONNX import

```
PyTorch model (eval mode)
  → torch.onnx.export(..., opset_version=17)
  → OnnxImporter.Load("model.onnx")     # .data file auto-resolved
  → Sequential
  → InferenceEngine.Run(input, output)  # zero-allocation
```

### Supported operators

| ONNX operator | Maps to | Notes |
|---------------|---------|-------|
| `Conv` | `ConvLayer` | 2D, NCHW, symmetric padding, any stride |
| `Gemm` | `LinearLayer` | `transB=1` handled automatically |
| `Relu` | `ReluActivation` | |
| `Tanh` | `TanhActivation` | |
| `Sigmoid` | `SigmoidActivation` | |
| `Softmax` | `SoftmaxActivation` | axis=-1 only |
| `MaxPool` | `MaxPool2DLayer` | Square kernel, stride = kernel |
| `GlobalAveragePool` | `GlobalAveragePool2DLayer` | 2D, NCHW |
| `BatchNormalization` | `BatchNorm1D` | eval mode (training_mode=0) |
| `Add` | `OnnxAddLayer` | Element-wise; used for skip connections |
| `Reshape` / `Flatten` | `FlattenLayer` | Rank reduction (4D→2D) |
| `Identity` / `Dropout` | _(no-op in eval mode)_ | |

**12 operators** (+ 2 no-ops). Unsupported operators throw a clear `NotSupportedException` naming the operator.

Two importers:
- **`OnnxImporter`** — linear topology only. Faster for simple CNNs and MLPs.
- **`OnnxGraphImporter`** — arbitrary DAG topology. Required for ResNet, DenseNet, EfficientNet (any model with skip connections or multiple inputs to a node).

External `.data` files (PyTorch ≥ 2.x default) resolved automatically.
No `Google.Protobuf` dependency.

### PyTorch export

```python
model.eval()  # IMPORTANT: folds BatchNorm into Conv weights

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=17,
    export_params=True,
)
```

---

## Architecture

```
InferenceEngine             ← zero-alloc inference facade (caller-owned buffers)
Sequential                  ← module composition
Layers                      ← Conv, Linear, ReLU, Tanh, Sigmoid, Softmax,
                               BatchNorm, MaxPool, GlobalAveragePool, Flatten, LSTM
ComputationGraph            ← autograd tape + backward
  graph.Linear(...)
  graph.Conv2D(...)
  graph.Relu(...)
  graph.SoftmaxCrossEntropy(...)
AutogradNodeOwnership       ← lifecycle metadata: Parameter / GraphTemporary /
                               GraphAuxiliary / ExternalBorrowed / View
Parameter                   ← long-lived trainable state, owns Data + Grad storage
  layer.TrainableParameters()
Kernels                     ← pure Span-based math, no AutogradNode
  LinearKernels             ← Forward, ForwardBatched, BackwardInput,
                               AccumulateWeightGrad, AccumulateBiasGrad
  PoolingKernels            ← MaxPool pool=2 SIMD fast path
OverfitParallel          ← zero-alloc bulk-wake dispatcher (Runtime/)
                               replacement for Parallel.For in zero-alloc hot paths
                               ~5 µs warm dispatch, 0 B/call, configurable via
                               OVERFIT_PARALLEL_WORKERS env var
TensorStorage<T>            ← pooled memory ownership (ArrayPool-backed)
Optimizers                  ← Adam(IEnumerable<Parameter>), SGD(IEnumerable<Parameter>)
OnnxImporter                ← PyTorch ONNX → Sequential (linear topology)
OnnxGraphImporter           ← PyTorch ONNX → OnnxGraphModel (DAG, skip connections)

GPT1Model                   ← decoder-only Transformer (GPT-1 / GPT-2 architecture)
  Gpt2Config                ← Small / Medium / Large / XL presets
CachedSlmInferenceEngine    ← KV-cache inference engine
  CachedSlmSession          ← per-session state: KV buffers, position counter
  StackWeights              ← zero-copy weight refs for full stack
    BlockWeights            ← zero-copy refs for one transformer layer
      SingleHeadWeights     ← zero-copy refs for one attention head (Q/K/V/O + biases)
  KeyValueCache             ← pre-allocated K/V storage, O(N) decode
BytePairEncoder             ← GPT-2 tokenizer (vocab.json + merges.txt)
TokenSampler                ← greedy, top-P (heap sort, zero-alloc)

OverfitOpenAiServer         ← OpenAI-compatible HTTP server (HttpListener + source-gen
                               JSON, no ASP.NET; Sources/Server/) — /v1/chat/completions
                               (SSE), /v1/embeddings, /v1/audio/speech
McpServer                   ← MCP stdio server (typed JSON-RPC 2.0 DTOs + source-gen
                               McpJsonContext, no SDK; Sources/Mcp/) — plugs local tools
                               into Claude Code / Desktop / IDEs; see docs/mcp.md
  OverfitMcpTools           ← ask / rag_query (citations, chat-model embeddings) /
                               transcribe (lazy Whisper)
```

### Autograd ownership

Every `AutogradNode` carries an `Ownership` tag set at creation:

| Ownership | Who disposes | Example |
|-----------|-------------|---------|
| `GraphTemporary` | `graph.Reset()` | ReLU output, hidden activations |
| `GraphAuxiliary` | `graph.Reset()` | MaxPool index map, Softmax probs |
| `Parameter` | Layer `Dispose()` | `LinearLayer.Weights`, `ConvLayer.Kernels` |
| `ExternalBorrowed` | Caller | Preallocated input/target batch buffers |
| `View` | Never (no storage) | `FlattenLayer` output |

`graph.Reset()` disposes by ownership — no hardcoded switch on `OpCode`.

---

## Evolutionary optimization

```csharp
var strategy = new OpenAIESStrategy(populationSize: 1024, sigma: 0.1f);
var candidates = strategy.Ask();      // 0 B allocation
strategy.Tell(fitnesses);
```

Use cases: Kubernetes tuning, game AI, industrial process search, pricing strategy.

---

## Transformer path — component status

| Component | Status | Notes |
|-----------|--------|-------|
| `LayerNorm` / `RMSNorm` | ✅ | Pre-LN, Post-LN, RMSNorm |
| `Embedding` (token + positional) | ✅ | Lookup + additive |
| `ScaledDotProductAttention` | ✅ | Causal mask, KV-cache |
| `MultiHeadAttention` | ✅ | Per-head Q/K/V/O weights + biases (GPT-2 style) |
| `GroupedQueryAttention` | ✅ | KV-head sharing (Llama / Qwen / Mistral) |
| RoPE positional encoding | ✅ | Per-layer rotation, configurable theta |
| SwiGLU FFN | ✅ | Modern SLM FFN (Llama / Qwen / Mistral) |
| Causal masking | ✅ | Auto-regressive generation |
| Transformer block | ✅ | Pre-LN + FFN + residual (GeLU and SwiGLU) |
| Tokenizer (BPE) | ✅ | GPT-2 byte-pair encoding, plus Qwen tokenizer.json |
| GPT-2 inference | ✅ | 124M params, 0 B/token, parity vs PyTorch (top-10 10/10, maxAbsDiff 0.000107) |
| Qwen / Llama inference | ✅ | 0.5B–3B FP32/F16/BF16 + Q8_0/Q4_K/Q6_K in-RAM, GQA + RoPE + SwiGLU |
| GGUF native loader | ✅ | F32, F16, BF16, Q8_0, **Q4_K**, **Q6_K** — Ollama files load directly |
| Streaming token API | ✅ | `IAsyncEnumerable<int>` with stop-tokens + cancellation |
| LoRA inference adapter | ✅ | Zero-copy weight refs; Enable/Disable/Save/Load |
| LoRA training | ✅ | GPT1 Stage 1/2/3 — LM head, FFN, per-head attention (`Gpt1LoRAFineTuner`) |
| GPT-2 training | ❌ | Gradient checkpointing required at scale |
| Transformer training (small) | 🟡 | TinyShakespeare training tests converge; quality demo runs end-to-end |
