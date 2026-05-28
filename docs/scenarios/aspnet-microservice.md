# Overfit for ASP.NET Core

**For .NET architects and backend engineers building microservices with ML inference.**

---

## The Problem

Your backend runs on .NET. Your team knows C#. Your infrastructure is built for .NET deployment. But somewhere in the pipeline, a data scientist trained a model in PyTorch and now you need to serve it in production.

The typical solutions all hurt:

- **Python sidecar** — A separate FastAPI service running in its own container, with its own dependencies, its own deployment pipeline, its own monitoring. Doubles your ops surface.
- **ONNX Runtime** — Closer, but drags in a 100+ MB native library, and adds ~3 μs of P/Invoke overhead per call. For small models, that overhead dwarfs the actual math.
- **ML.NET** — Works, but opinionated, with its own abstractions that don't always map cleanly to modern deep learning workflows.

**Overfit is different.** It runs as pure managed C# inside your existing process. No sidecar, no native binary, no serialization across process boundaries.

---

## Two integration patterns

Overfit covers two distinct deployment shapes in the same .NET process. Pick the one that matches your workload.

### Pattern A — small model / classifier (caller-owned buffers, zero allocation per request)

For tabular / vision / time-series models trained in Overfit or imported from ONNX. The `InferenceEngine` facade owns a preallocated output buffer; the caller passes in its input as a `ReadOnlySpan<float>`.

```csharp
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Inference;

var builder = WebApplication.CreateBuilder(args);

// Load once at startup. The Sequential model wraps your trained / imported layers;
// InferenceEngine owns the preallocated output buffer and is thread-safe for inference.
var model = new Sequential(new LinearLayer(784, 10));
model.Load("model.bin");
model.Eval();

var engine = InferenceEngine.FromSequential(model, inputSize: 784, outputSize: 10);
builder.Services.AddSingleton(engine);

var app = builder.Build();

app.MapPost("/predict", (float[] input, InferenceEngine engine) =>
{
    // Predict returns a ReadOnlySpan into the engine's internal buffer.
    // Materialise to an array if you need to escape the request scope.
    var output = engine.Predict(input);
    return output.ToArray();
});

app.Run();
```

No `AutogradNode`, no `model.Forward`, no per-request tensor allocations. The autograd path is for **training**; the inference path goes through `InferenceEngine.Run` / `Predict` with caller-owned (or engine-owned) buffers. See [`docs/TECHNICAL.md`](../TECHNICAL.md) for the full inference-vs-training separation.

### Pattern B — local LLM / RAG / agent (in-process chat, no external API)

For local LLM endpoints — Qwen / Llama / Mistral / Mixtral / MoE loaded directly from GGUF, with the model staying inside the .NET process. Combine with the in-process vector store for RAG, and the tool-calling constraint for agents.

```csharp
using DevOnBike.Overfit.LanguageModels;

var builder = WebApplication.CreateBuilder(args);

// Load once at startup. OverfitClient owns the engine, session, tokenizer, and chat session.
// Auto-detects chat template, stop sequences, and tokenizer from the GGUF + sibling files.
var client = OverfitClient.LoadGguf(@"/app/models/qwen2.5-3b.q4km.gguf", maxContextLength: 2048);
client.AddSystem("You are a concise assistant.");
builder.Services.AddSingleton(client);

var app = builder.Build();

app.MapPost("/chat", (ChatRequest req, OverfitClient client) =>
{
    var reply = client.Send(req.Message);
    return new { reply };
});

app.Run();

record ChatRequest(string Message);
```

For RAG: pair `OverfitClient.LoadGguf` with `SentenceEmbedder.ForBgeEnV15(...)` for embeddings + the in-process vector store. For agents: pass a tool-calling constraint to `client.Send(..., constraint: ...)` so the model is logit-forced to emit a valid call dispatched to your C# delegate.

That's it. No new containers, no new deployment pipeline, no network hop between your API and your model.

---

## What You Get

### Latency

| Scenario | Overfit | Python/FastAPI sidecar | ONNX Runtime in-process |
|----------|--------:|-----------------------:|------------------------:|
| Single inference (784→10) | **188 ns** | ~500 μs (network + serialization) | 3,331 ns |
| P99 latency (10k calls) | **0.6 μs** | variable (depends on sidecar) | 3.7 μs |
| Cold start (first request) | **291 μs** | seconds | 1,224 μs |

Inference is no longer a network call. It's a function call.

### Operational Simplicity

- **One deployment** — your ASP.NET service. That's it.
- **One runtime** — .NET. No Python environment, no Conda, no pip freeze.
- **One health check** — if your API is up, your model is up.
- **One dependency version to manage** — no more "Python 3.9 vs 3.11, which does our cluster support?"
- **One log stream, one metric source, one profiler.**

### Deployment Options

Overfit is **100% Native AOT compatible**. This means:

- **Single-file executable**: `dotnet publish -c Release -r linux-x64 /p:PublishAot=true` produces a self-contained binary (typically 20-40 MB for an ASP.NET service).
- **Container images**: base on `mcr.microsoft.com/dotnet/runtime-deps:10` instead of `aspnet:10` — saves ~150 MB per image.
- **Kubernetes friendly**: faster pod startup, less memory, cleaner rolling updates.

### GC Behavior

ONNX Runtime allocates ~900 B per inference call. At 10,000 RPS that's **9 MB/sec of allocations**, triggering Gen 0 collections and occasional Gen 1/2 promotions. Your P99 latency reflects this.

Overfit allocates **zero bytes** per inference call once your model is loaded. No GC pressure, no tail latency spikes.

---

## Recommended Architecture

### Small models (single request per call)

```csharp
// Register once. InferenceEngine wraps the model with a preallocated output
// buffer and is thread-safe for inference; share one singleton per process.
builder.Services.AddSingleton<InferenceEngine>(sp =>
{
    var model = new Sequential(/* architecture */);
    model.Load("/app/models/classifier.bin");
    model.Eval();
    return InferenceEngine.FromSequential(model, inputSize: 784, outputSize: 10);
});
```

### Batch endpoint

For higher throughput, allocate a caller-owned output buffer (e.g. from `ArrayPool<float>`) and call `engine.Run(input, output)`. The engine performs the whole batched forward pass into the supplied span with no allocations beyond the buffer you own.

```csharp
app.MapPost("/predict-batch", (float[][] inputs, InferenceEngine engine) =>
{
    var batchSize = inputs.Length;

    var flatInput = ArrayPool<float>.Shared.Rent(batchSize * 784);
    var output    = ArrayPool<float>.Shared.Rent(batchSize * 10);
    try
    {
        for (var i = 0; i < batchSize; i++)
        {
            inputs[i].AsSpan().CopyTo(flatInput.AsSpan(i * 784, 784));
        }

        engine.Run(
            flatInput.AsSpan(0, batchSize * 784),
            output.AsSpan(0, batchSize * 10));

        // Materialise into per-sample arrays for JSON serialisation.
        var result = new float[batchSize][];
        for (var i = 0; i < batchSize; i++)
        {
            result[i] = output.AsSpan(i * 10, 10).ToArray();
        }
        return result;
    }
    finally
    {
        ArrayPool<float>.Shared.Return(flatInput);
        ArrayPool<float>.Shared.Return(output);
    }
});
```

Batch ≤ 16 is where Overfit dominates ONNX Runtime. Beyond that, ONNX's MKL integration starts to win — but you can route large batches to a separate endpoint if needed.

(Note: `ArrayPool<float>.Shared` is fine in application / scenario code. The raw-`Shared` ban applies only to `Sources/Main` — see `Sources/Main/BannedSymbols.txt`. Library code uses `PooledBuffer<T>`; consumer code uses `ArrayPool` directly.)

### Thread safety

`Sequential.Forward(null, input)` is **thread-safe for inference** when the model is in `Eval()` mode. Call it concurrently from multiple request handlers without locking.

Do **not** share the same `FastTensor<float>` input buffer across threads — allocate one per request (it's cheap) or pool them via `ArrayPool<float>`.

---

## Migration from Python Sidecar

Typical migration path:

1. **Export your trained model** from PyTorch/TensorFlow to a simple binary format (weights as float arrays).
2. **Recreate the architecture in Overfit** using the matching layer types. If your model uses only Linear + Conv2D + BatchNorm + ReLU + LSTM, you're covered today.
3. **Load the weights** into the Overfit model.
4. **Delete your Python sidecar.** Remove the container from your deployment. Remove the network call from your API.
5. **Benchmark.** You'll typically see 50-100× latency improvement on small models, 2-5× on larger ones.

For more complex models, Overfit now also covers:
- **Transformer LLMs**: Qwen2.5 (0.5B-32B), Llama-2/3.x, Mistral, Mixtral, Qwen-MoE, GPT-2 — load straight from GGUF (incl. mmap K-quant) or HuggingFace safetensors via `OverfitClient.LoadGguf(path)`.
- **ONNX import**: linear topology + DAG (ResNet-style skip connections) — load PyTorch-exported models directly.
- **Sentence embeddings**: MiniLM / BGE / E5 with bit-parity vs HuggingFace, for in-process RAG.

If your model uses something we don't support yet, check the [ROADMAP](../../ROADMAP.md) — most additions are 1-3 weeks of work.

---

## When Overfit Is the Wrong Choice

Be honest with yourself:

- **You need GPU inference.** Overfit is CPU-first / pure-managed. For workloads where a GPU is mandatory (very large LLMs at high concurrency, diffusion image generation), use a GPU-accelerated runtime — Overfit isn't trying to compete on GPU throughput.
- **Raw single-stream decode tok/s is the only metric.** llama.cpp / LLamaSharp decode ~1.5× faster on the same Q4_K_M file (hand-tuned native AVX-512). Overfit matches them on RAM and wins on per-token allocation (1 B vs 21 KB) — but if pure tok/s is the only knob, use them.
- **Batch size consistently > 64 and pure throughput dominates over latency.** Overfit's advantage is latency, zero-alloc, in-process embedding, and the agentic stack — not raw BLAS throughput on huge batches.
- **You want a separate inference server / sidecar process.** Overfit is a library, not a daemon. Use Ollama / llama.cpp's server / Triton if a separate inference tier is the right architecture for you.

---

## Further Reading

- [Main README](../../README.md) — project overview and benchmarks
- [ROADMAP](../../ROADMAP.md) — what's planned, what's not
- [Edge / IoT scenario](edge-iot.md) — for field-deployed inference
- [Finance / Latency scenario](finance-latency.md) — for tail-latency-critical systems