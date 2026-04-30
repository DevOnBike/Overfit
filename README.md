# Overfit

Pure C# deep-learning and optimization engine. Predictable CPU performance, explicit memory ownership, zero-allocation inference hot paths.

No native binaries. No Python runtime. No ONNX Runtime dependency.

---

## What it does

**Train in PyTorch or .NET. Load or build a model. Run predictable, allocation-free inference in .NET.**

- **Zero-allocation CPU inference** — preallocated buffers, no per-call GC pressure, competitive with ONNX Runtime.
- **ONNX import** — load PyTorch-exported models directly. 11 operators, branching DAGs (ResNet skip connections), output matches PyTorch within 1e-4.
- **Evolutionary optimization** — allocation-free `Ask`/`AskThenTell` loops for black-box parameter search.

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

// Optimizers accept Parameter directly (Etap 6 API)
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

---

## Benchmark snapshot

Machine: AMD Ryzen 9 9950X3D · Windows 11 25H2 · .NET 10.0.7 · BenchmarkDotNet 0.15.8

### Single inference — Overfit vs ONNX Runtime

| Method | Mean | Allocated | vs ONNX Runtime |
|--------|-----:|----------:|----------------:|
| **Overfit `InferenceEngine`** | **201.6 ns** | **0 B** | **9.2× faster** |
| ONNX Runtime (pre-allocated) | 1 853 ns | 224 B | baseline |
| ONNX Runtime (standard) | 3 369 ns | 952 B | 0.55× |

Model: Linear(784→10). Overfit is **9.2× faster** than ONNX Runtime pre-allocated path, **16.7× faster** than standard path, with zero managed allocations.

### DAG inference — ResNet-style model with skip connections

| Method | Mean | Allocated |
|--------|-----:|----------:|
| `OnnxGraphModel.RunInference` (direct) | ~0.9–2.1 µs | **0 B** |
| `InferenceEngine.FromBackend` (via engine) | ~0.9–2.1 µs | **0 B** |

Model: TinyResNet — Linear(8→8) + skip + Linear(8→4). Both paths: zero allocations.
Bimodal distribution due to model size (sub-µs math, timer resolution dominates).

### CNN training throughput (60k MNIST, batch=64)

| Epoch | Time | Alloc/epoch | Notes |
|------:|-----:|------------:|-------|
| 1 | ~1.7 s | ~32 MB | JIT warmup |
| 2–5 | **~800 ms** | **~26 MB** | steady state |

Training allocations from autograd graph temporaries — expected.
Inference path: zero allocations. Live managed memory delta per epoch: **−0.01 MB** (zero leak).

### Concurrent inference (8 threads × 1 000 calls each)

| Method | Mean | Allocated | vs ONNX Runtime |
|--------|-----:|----------:|----------------:|
| **Overfit (concurrent)** | **637.8 ms** | **0 B** | **3.0× faster** |
| ONNX Runtime (concurrent) | 1 916.7 ms | 117 MB | baseline |

Overfit scales linearly — no shared mutable state, no lock contention.
ONNX Runtime allocates 117 MB of managed memory under concurrent load (Gen0 GC pressure).

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
InferenceEngine          ← zero-alloc inference facade (caller-owned buffers)
Sequential               ← module composition
Layers                   ← Conv, Linear, ReLU, Tanh, Sigmoid, Softmax,
                           BatchNorm, MaxPool, GlobalAveragePool, Flatten, LSTM
ComputationGraph         ← autograd tape + backward
  graph.Linear(...)      ← operation facade (PR5 Etap 1)
  graph.Conv2D(...)
  graph.Relu(...)
  graph.SoftmaxCrossEntropy(...)
AutogradNodeOwnership    ← lifecycle metadata: Parameter / GraphTemporary /
                           GraphAuxiliary / ExternalBorrowed / View
Parameter                ← long-lived trainable state, owns Data + Grad storage
  layer.TrainableParameters()  ← preferred API for optimizers (Etap 6)
Kernels                  ← pure Span-based math, no AutogradNode
  LinearKernels          ← Forward, ForwardBatched, BackwardInput,
                           AccumulateWeightGrad, AccumulateBiasGrad
  PoolingKernels         ← MaxPool pool=2 SIMD fast path
TensorStorage<T>         ← unmanaged memory ownership
Optimizers               ← Adam(IEnumerable<Parameter>), SGD(IEnumerable<Parameter>)
OnnxImporter             ← PyTorch ONNX → Sequential (linear topology)
OnnxGraphImporter        ← PyTorch ONNX → OnnxGraphModel (DAG topology, skip connections)
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

## Requirements

- .NET 10+
- No native dependencies
- No Python runtime
- Native AOT compatible

---

## Roadmap

### Recently completed

- ✅ **ONNX import — 12 operators** (Conv w/padding+stride, Gemm, ReLU, Tanh, Sigmoid, Softmax, MaxPool, GlobalAveragePool, BatchNormalization, Add, Reshape, Flatten)
- ✅ **ONNX DAG runtime** — `OnnxGraphImporter` supports branching topology (skip connections, residual blocks). Enables ResNet-style models. Zero-allocation inference via `OnnxGraphInferenceBackend`.
- ✅ **AveragePool + ReduceMean** — windowed average pooling with padding/stride; `ReduceMean` mapped to GlobalAveragePool (PyTorch `AdaptiveAvgPool2d` export pattern). Total: **14 ONNX operators**.
- ✅ **PR5 Autograd ownership cleanup** — `Parameter` type, `AutogradNodeOwnership` enum, `graph.Reset()` by ownership, all layers migrated (LinearLayer, ConvLayer, BatchNorm1D)
- ✅ **Optimizers on `Parameter`** — `Adam(IEnumerable<Parameter>)`, `SGD(IEnumerable<Parameter>)`
- ✅ **PERF-1: Linear backward kernels** — hybrid threshold eliminates `Parallel.For` overhead for small matrices; backward alloc −43% (23 MB → 13 MB per epoch)
- ✅ **BATCHED-1: `LinearKernels.ForwardBatched`** — weight-stationary outer product for small matrices, no zero-skipping branch mispredictions
- ✅ **MaxPool pool=2 SIMD** — `TensorPrimitives.Max` fast path, 659 µs vs 815 µs baseline

### Near-term

- **PR5-7d/e** — Move Conv2D/MaxPool2D into `ComputationGraph.*` (architectural cleanup, zero user impact)
- **ONNX: LSTM/GRU operators** — enables recurrent model import
- **Depthwise Conv** (group=channels) — MobileNet-style models

### Transformer path (toward GPT-1)

Building GPT-1 (117M params, decoder-only Transformer) requires these components in order:

| Component | Status | Notes |
|-----------|--------|-------|
| `LayerNorm` | ❌ | Different from BatchNorm — normalises over features, not batch |
| `Embedding` | ❌ | Token lookup + positional encoding |
| `ScaledDotProductAttention` | ❌ | Core of every Transformer |
| `MultiHeadAttention` | ❌ | Parallel attention heads |
| Causal masking | ❌ | Upper-triangular mask for autoregressive generation |
| Transformer block | ❌ | Attention + FFN + LayerNorm + residual (DAG ✅ already works) |
| Tokenizer (BPE) | ❌ | Pre/post processing |
| Gradient checkpointing | ❌ | Required for 117M params on single machine |

**Distance estimate:** 5–7 new layer types + memory management changes. Estimated 3–4 months of focused work at current pace. The DAG runtime (skip connections) already handles residual blocks — that part is done.

Overfit could run GPT-1 inference once the operator set is complete. Training at scale requires gradient checkpointing.

### Long-term

- Graph compilation / kernel fusion for fixed-shape models
- Batched GEMM parallel path (unsafe fixed-pointer `Parallel.For`)
- AOT compilation target

---

## What Overfit is not

Not a PyTorch/TensorFlow replacement. Not GPU-first. Not transformer-scale first.

The differentiator: pure C#, predictable allocation behaviour, competitive CPU inference for small/medium models where managed zero-allocation matters.