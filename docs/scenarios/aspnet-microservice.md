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

## Integration in Three Lines

```csharp
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Tensors;

var builder = WebApplication.CreateBuilder(args);

// Load once at startup - thread-safe for inference
var model = new Sequential(new LinearLayer(784, 10));
model.Load("model.bin");
model.Eval();
builder.Services.AddSingleton(model);

var app = builder.Build();

app.MapPost("/predict", (float[] input, Sequential model) =>
{
    using var tensor = new FastTensor<float>(1, 784, clearMemory: false);
    input.AsSpan().CopyTo(tensor.GetView().AsSpan());
    using var node = new AutogradNode(tensor, requiresGrad: false);
    var result = model.Forward(null, node).DataView.AsReadOnlySpan();
    return result.ToArray();
});

app.Run();
```

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
// Register once
builder.Services.AddSingleton<Sequential>(sp =>
{
    var model = new Sequential(/* architecture */);
    model.Load("/app/models/classifier.bin");
    model.Eval();
    return model;
});
```

### Batch endpoint

Process multiple samples in a single forward pass for higher throughput:

```csharp
app.MapPost("/predict-batch", (float[][] inputs, Sequential model) =>
{
    var batchSize = inputs.Length;
    using var tensor = new FastTensor<float>(batchSize, 784, clearMemory: false);
    var span = tensor.GetView().AsSpan();
    for (var i = 0; i < batchSize; i++)
    {
        inputs[i].AsSpan().CopyTo(span.Slice(i * 784, 784));
    }
    using var node = new AutogradNode(tensor, requiresGrad: false);
    return model.Forward(null, node).DataView.AsReadOnlySpan().ToArray();
});
```

Batch ≤ 16 is where Overfit dominates ONNX Runtime. Beyond that, ONNX's MKL integration starts to win — but you can route large batches to a separate endpoint if needed.

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

If your model uses layers Overfit doesn't yet support (Transformers, advanced attention, custom ops), check the [ROADMAP](../../ROADMAP.md). ONNX import is planned for the near term, which will let you load any PyTorch-exported model directly.

---

## When Overfit Is the Wrong Choice

Be honest with yourself:

- **You need GPU inference.** Overfit is CPU-first. For models where GPU is mandatory (LLMs, diffusion), use ONNX Runtime with CUDA or a dedicated inference server.
- **Your model has layers Overfit doesn't support.** Today: no Transformers, no MultiHeadAttention. If ONNX import doesn't land in time, use ONNX Runtime.
- **Batch size consistently > 64.** ONNX with MKL wins in pure throughput. Overfit's advantage is latency and zero-alloc, not raw BLAS speed.

---

## Further Reading

- [Main README](../../README.md) — project overview and benchmarks
- [ROADMAP](../../ROADMAP.md) — what's planned, what's not
- [Edge / IoT scenario](edge-iot.md) — for field-deployed inference
- [Finance / Latency scenario](finance-latency.md) — for tail-latency-critical systems