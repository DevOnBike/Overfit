# Overfit

Pure C# deep-learning and optimization engine. Predictable CPU performance, explicit memory ownership, zero-allocation inference hot paths.

No native binaries. No Python runtime. No ONNX Runtime dependency.

---

## What it does

**Train in PyTorch or .NET. Load or build a model. Run predictable, allocation-free inference in .NET.**

- **Zero-allocation CPU inference** — preallocated buffers, no per-call GC pressure, competitive with ONNX Runtime.
- **ONNX import** — load PyTorch-exported models directly. 10 operators supported, output matches PyTorch within 1e-4.
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

### Inference — ONNX import

```csharp
using DevOnBike.Overfit.Onnx;
using DevOnBike.Overfit.Inference;

var model = OnnxImporter.Load("classifier.onnx"); // .data file resolved automatically
model.Eval();

using var engine = InferenceEngine.FromSequential(model, inputSize: 784, outputSize: 10);
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

| Model | Overfit | ONNX Runtime | Overfit alloc |
|-------|--------:|-------------:|--------------:|
| Linear(784→10) | 252 ns | 1989 ns | **0 B** |
| MLP 784→128→10 | ~3.7 µs | ~5.2 µs | **0 B** |
| Small CNN (MNIST) | 5.4 µs | 6.5 µs | **0 B** |

### ONNX-imported model vs ONNX Runtime

Same weights, same graph, PyTorch export:

| Method | Mean | Allocated |
|--------|-----:|----------:|
| **Overfit (imported ONNX)** | **5.4 µs** | **0 B** |
| ONNX Runtime | 6.5 µs | 224 B |

Overfit runs the PyTorch-exported model **1.2× faster with zero allocations**.
For small models (Linear 784→10): **7.9× faster** (252 ns vs 1989 ns).

### CNN training throughput (60k MNIST, batch=64)

| Epoch | Time | Alloc/epoch |
|------:|-----:|------------:|
| 1 | ~1.7 s | ~33 MB |
| 2–5 | **~856 ms** | **~26 MB** |

Training allocations come from autograd graph temporaries — expected for the current architecture.
Inference path: zero allocations.

### Concurrent inference (8 threads × 1 000 calls)

| Method | Mean | Allocated |
|--------|-----:|----------:|
| **Overfit** | ~516 ms | **0 B** |
| ONNX Runtime | ~1811 ms | ~117 MB |

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
| `Conv` | `ConvLayer` | 2D, NCHW, padding=0, stride=1 |
| `Gemm` | `LinearLayer` | `transB=1` handled automatically |
| `Relu` | `ReluActivation` | |
| `Tanh` | `TanhActivation` | |
| `Sigmoid` | `SigmoidActivation` | |
| `Softmax` | `SoftmaxActivation` | axis=-1 only |
| `MaxPool` | `MaxPool2DLayer` | Square kernel, stride = kernel |
| `GlobalAveragePool` | `GlobalAveragePool2DLayer` | 2D, NCHW |
| `Reshape` / `Flatten` | `FlattenLayer` | Rank reduction (4D→2D) |
| `Identity` / `Dropout` | _(no-op in eval mode)_ | |

10 operators total. Unsupported operators throw a clear `NotSupportedException` naming the operator.

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
OnnxImporter             ← PyTorch ONNX → Sequential (no Google.Protobuf)
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

- ✅ **ONNX import — 10 operators** (Conv, Gemm, ReLU, Tanh, Sigmoid, Softmax, MaxPool, GlobalAveragePool, Reshape, Flatten)
- ✅ **PR5 Autograd ownership cleanup** — `Parameter` type, `AutogradNodeOwnership` enum, `graph.Reset()` by ownership, all layers migrated (LinearLayer, ConvLayer, BatchNorm1D)
- ✅ **Optimizers on `Parameter`** — `Adam(IEnumerable<Parameter>)`, `SGD(IEnumerable<Parameter>)`
- ✅ **PERF-1: Linear backward kernels** — hybrid threshold eliminates `Parallel.For` overhead for small matrices; backward alloc −43% (23 MB → 13 MB per epoch)
- ✅ **BATCHED-1: `LinearKernels.ForwardBatched`** — weight-stationary outer product for small matrices, no zero-skipping branch mispredictions
- ✅ **MaxPool pool=2 SIMD** — `TensorPrimitives.Max` fast path, 659 µs vs 815 µs baseline

### Near-term

- **ONNX-4** — Conv with padding and stride (enables ResNet-style models)
- **PR5-7** — Move graph-aware TensorMath into `ComputationGraph.*` (op-by-op)
- **ONNX-3** — `BatchNormalization` standalone operator (train-mode exports)

### Medium-term

- Skip connections / branching DAG (ResNet support)
- Batched GEMM parallel path (unsafe fixed-pointer `Parallel.For`)
- Graph compilation for fixed-shape models

---

## What Overfit is not

Not a PyTorch/TensorFlow replacement. Not GPU-first. Not transformer-scale first.

The differentiator: pure C#, predictable allocation behaviour, competitive CPU inference for small/medium models where managed zero-allocation matters.