# Overfit for Finance & Low-Latency Systems

**For engineers building trading systems, fraud detection, and risk scoring where tail latency matters more than average throughput.**

---

## The Problem

You work in a latency-sensitive environment. Maybe it's high-frequency trading where 10 microseconds of slippage costs real money. Maybe it's real-time fraud scoring where each additional millisecond increases false-decline rates. Maybe it's risk limits that must evaluate before an order crosses a matching engine.

The standard ML deployment story breaks down here:

- **Python-based inference** is a non-starter. GIL contention, variable GC behavior, unpredictable latency spikes — operationally unsuitable.
- **ONNX Runtime** is closer, but its native-to-managed transitions allocate per call, causing GC pauses that show up at P99.9 and beyond.
- **Custom C++ via P/Invoke** gives you control but creates a language boundary that fragments your observability, debugging, and testing.

**Overfit is designed for this environment.** Pure managed code, zero allocations on the inference path, predictable SIMD-backed math.

---

## Hard Numbers

Measured on Ryzen 9 9950X3D, .NET 10, Linear 784→10 model:

| Metric | Overfit | ONNX Runtime |
|--------|--------:|-------------:|
| **P50 latency** | 0.40 μs | 3.00 μs |
| **P99 latency** | 0.60 μs | 3.70 μs |
| **P99.9 latency** | 0.80 μs | 5.70 μs |
| **Max observed** | **8.40 μs** | **2,184 μs** |
| GC Gen-0 collections (100k calls) | **0** | 1 |

The **max latency** number is the important one. ONNX's 2.2 ms worst case is a GC pause. In a trading system, that single outlier could be the difference between filling an order and missing it.

Overfit's max (8.4 μs) represents normal scheduling jitter, not GC.

---

## Where This Matters

### Pre-Trade Risk Checks

Before an order leaves your system, you want to validate it against learned patterns: is this trade unusually large? Does it fit the account's historical behavior? Is it consistent with current market conditions?

```csharp
public class PreTradeRiskScorer
{
    private readonly Sequential _model;
    private readonly ThreadLocal<FastTensor<float>> _tensorPool;

    public float Score(OrderFeatures features)
    {
        var tensor = _tensorPool.Value;
        features.WriteTo(tensor.GetView().AsSpan());

        using var node = new AutogradNode(tensor, requiresGrad: false);
        return _model.Forward(null, node).DataView.AsReadOnlySpan()[0];
    }
}
```

Inference cost: **sub-microsecond**. Fits inside the hot path between order submission and venue routing without meaningfully adding to latency.

### Real-Time Fraud Detection

Card transaction arrives. Score it against a learned model of fraudulent patterns. Block or approve within the ~100ms authorization budget.

Overfit's advantage here isn't about the 100ms budget — you have plenty of time. It's about **consistency** under load. When your system is processing 50,000 transactions per second, GC pauses in the inference path cascade into increased queue depths and retry storms.

### Streaming Anomaly Detection

Financial data streams (order flow, trades, quotes, fund prices) need continuous monitoring. An autoencoder catches anomalies by measuring reconstruction error.

```csharp
// Runs in a tight consumer loop. Zero allocations per tick.
while (tickStream.TryRead(out var tick))
{
    tick.Features.CopyTo(_inputBuffer);
    var reconstruction = _autoencoder.Forward(null, _inputNode);
    var error = ComputeReconstructionError(reconstruction, _inputNode);

    if (error > _threshold)
    {
        _alertQueue.Enqueue(tick);
    }
}
```

---

## Architecture Guidance

### Thread-Per-Core Inference

In latency-critical pipelines, pin one model instance per worker thread to avoid contention:

```csharp
public class PerCorePool
{
    private readonly ThreadLocal<Sequential> _models = new(() =>
    {
        var model = new Sequential(/* architecture */);
        model.Load("/models/scorer.bin");
        model.Eval();
        return model;
    });

    public Sequential GetForCurrentThread() => _models.Value;
}
```

This eliminates false sharing on model weights (each thread has its own read-only copy cached locally) and removes the need for any locking.

### Warm-Up on Startup

JIT tiering matters. First call to `Forward` triggers JIT compilation, resulting in a slow first inference. Warm up explicitly at service startup:

```csharp
public void WarmUp(int iterations = 1000)
{
    using var dummyInput = new FastTensor<float>(1, _inputSize, clearMemory: false);
    using var node = new AutogradNode(dummyInput, requiresGrad: false);

    for (var i = 0; i < iterations; i++)
    {
        _ = _model.Forward(null, node);
    }
}
```

Run this before accepting real traffic. After ~1000 iterations, JIT has tiered up to fully optimized code and you see the benchmark numbers.

### GC Configuration

For latency-critical deployments, tune the GC:

```xml
<!-- in your .csproj -->
<PropertyGroup>
    <ServerGarbageCollection>true</ServerGarbageCollection>
    <ConcurrentGarbageCollection>true</ConcurrentGarbageCollection>
    <TieredCompilation>true</TieredCompilation>
    <TieredPGO>true</TieredPGO>
</PropertyGroup>
```

For ultra-low-latency scenarios, consider the region-based GC (available in .NET 10).

### Avoid Allocations Upstream

Overfit doesn't allocate, but your code around it might:

```csharp
// BAD: allocates per tick
var input = tick.Features.ToArray();

// GOOD: write directly into pre-allocated tensor
tick.WriteFeaturesTo(_inputBuffer.GetView().AsSpan());
```

Profile with `dotnet-counters` watching Gen 0 allocation rate. In a properly tuned inference pipeline, this should be flat.

---

## Model Architectures That Fit

### Tabular classification / regression

Linear MLPs work brilliantly at the edge of latency budgets. A 3-4 layer network with ~100 hidden units processes a feature vector in under 1 μs.

### Sequence models (LSTM)

Overfit supports LSTMLayer for rolling-window analysis (price series, order flow signatures). Stateful inference: maintain hidden state across ticks rather than re-running over the window.

### Autoencoders

For anomaly detection, a simple Linear encoder-decoder is often enough. Train on normal data, flag when reconstruction error spikes.

---

## What Overfit Doesn't Give You (Yet)

Be honest:

- **No custom CUDA kernels.** If your inference needs GPU, Overfit isn't for you today.
- **No Transformers.** Roadmap item, not available yet. For BERT-style sentiment analysis on news feeds, use ONNX Runtime until Overfit catches up.
- **No quantization.** INT8 inference is on the roadmap. For now, everything is FP32.
- **No ONNX import.** Planned, not shipped. You'll need to recreate the architecture in Overfit and load weights manually, or wait for ONNX support.

---

## Observability

Overfit's `Diagnostics` subsystem provides optional per-module timing without affecting hot-path performance:

```csharp
// Enable once
OverfitDiagnostics.Enable();

// Inference proceeds as normal
var result = model.Forward(null, input);

// Later: inspect
var events = OverfitDiagnostics.DrainEvents();
foreach (var e in events)
{
    logger.LogDebug("Module {Module} took {Duration}ms", e.ModuleType, e.DurationMs);
}
```

When disabled (default), there is zero overhead. When enabled, per-layer timing is captured without allocations.

This integrates cleanly with OpenTelemetry, Prometheus, or whatever metrics stack your trading infrastructure uses.

---

## Migration Path from Existing Solutions

### From ONNX Runtime

1. Recreate the architecture in Overfit (match layer types, dimensions).
2. Export weights from your training pipeline (PyTorch `state_dict` → binary).
3. Load into Overfit's `Sequential` using the existing serialization.
4. Benchmark with production traffic shapes — typically 5-50× latency improvement on small models.

### From a Python Microservice

1. Replace the network call with an in-process inference (see the [ASP.NET scenario](aspnet-microservice.md) for the integration pattern).
2. Delete the Python service and its deployment pipeline.
3. Latency improvement is usually 100-1000× because you're removing network serialization, not just compute.

---

## Further Reading

- [Main README](../../README.md) — project overview and full benchmark suite
- [ROADMAP](../../ROADMAP.md) — upcoming features (quantization, graph compilation)
- [ASP.NET scenario](aspnet-microservice.md) — for integration with existing .NET services
- [Edge scenario](edge-iot.md) — for co-located inference on trading appliances