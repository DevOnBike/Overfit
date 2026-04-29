# Overfit

Overfit is a pure C# deep-learning and optimization engine focused on predictable CPU performance, explicit memory ownership, and zero-allocation inference hot paths.

No native binaries. No Python runtime. No ONNX Runtime dependency.

---

## What it does

**Train in PyTorch or .NET. Load or build a model. Run predictable, allocation-free inference in .NET.**

Three connected capabilities ship today:

1. **Zero-allocation CPU inference** — preallocated buffers, no per-call managed allocation, competitive with ONNX Runtime on small/medium models.
2. **ONNX import MVP** — load a PyTorch-exported MNIST CNN, run it through the same zero-allocation inference path. Output matches PyTorch within 1e-4.
3. **Evolutionary optimization** — allocation-free `Ask` / `AskThenTell` loops for black-box parameter search.

---

## Quick start

### Inference (native model)

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

engine.Run(input, output); // zero-allocation hot path
```

### ONNX import

```csharp
using DevOnBike.Overfit.Onnx;
using DevOnBike.Overfit.Inference;

// PyTorch eval-mode export → OnnxImporter.Load → Sequential → InferenceEngine
var model = OnnxImporter.Load("mnist_cnn.onnx"); // .data file resolved automatically
model.Eval();

using var engine = InferenceEngine.FromSequential(model, inputSize: 784, outputSize: 10);

var prediction = engine.Predict(input); // ReadOnlySpan<float>, 0 B
```

### Training

```csharp
using var conv  = new ConvLayer(1, 8, 28, 28, 3);
using var fcHid = new LinearLayer(1352, 64);
using var fcOut = new LinearLayer(64, 10);

using var optimizer = new Adam(parameters, learningRate: 0.001f) { UseAdamW = true };
using var graph     = new ComputationGraph();

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
| Linear(64, 10) | ~80 ns | ~1.39 µs | **0 B** |
| Linear(784, 10) | 252 ns | 1989 ns (7.9×) | **0 B** |
| Linear(4096, 10) | ~1.08 µs | ~3.74 µs | **0 B** |
| MLP 784→128→10 | ~3.7 µs | ~5.2 µs | **0 B** |
| Small CNN (MNIST) | 5.4 µs | 6.5 µs | **0 B** |

### ONNX-imported model vs ONNX Runtime

Same weights, same graph, PyTorch export:

| Method | Mean | Allocated |
|--------|-----:|----------:|
| **Overfit (imported ONNX)** | **5.4 µs** | **0 B** |
| ONNX Runtime | 6.5 µs | 224 B |

Overfit runs the PyTorch-exported model **1.2× faster with zero allocations**.

For small linear inference (784→10): **7.9× faster** than ONNX Runtime pre-allocated (252 ns vs 1989 ns).

### Concurrent inference (8 threads × 1 000 calls)

| Method | Mean | Allocated |
|--------|-----:|----------:|
| **Overfit** | ~516 ms | **0 B** |
| ONNX Runtime | ~1811 ms | ~117 MB |

---

## Architecture

```
InferenceEngine        ← zero-alloc inference facade (caller-owned buffers)
Sequential             ← module composition
Layers                 ← Conv, Linear, ReLU, BatchNorm, MaxPool, Flatten, LSTM
ComputationGraph       ← autograd tape + backward
  graph.Linear(...)   ← operation facade (Etap 1 / PR5)
  graph.Conv2D(...)
  graph.Relu(...)
  graph.SoftmaxCrossEntropy(...)
AutogradNodeOwnership  ← lifecycle metadata (Etap 2 / PR5)
Kernels                ← pure Span-based math, no autograd
TensorStorage<T>       ← unmanaged memory ownership
TensorShape            ← shape metadata (readonly struct)
Optimizers             ← Adam, AdamW, SGD
OnnxImporter           ← PyTorch ONNX → Sequential (no Google.Protobuf)
```

---

## ONNX import

```
PyTorch model (eval mode)
  → torch.onnx.export(...)
  → OnnxImporter.Load("model.onnx")
  → Sequential
  → InferenceEngine.Run(input, output)
```

Supported operators: `Conv`, `Gemm` (→ Linear), `Relu`, `MaxPool`, `Reshape`/`Flatten`.

External `.data` files (PyTorch ≥ 2.x) resolved automatically. No `Google.Protobuf` dependency.

---

## Evolutionary optimization

Allocation-free `Ask` / `AskThenTell` loops for black-box problems:

```csharp
var strategy = new OpenAIESStrategy(populationSize: 1024, sigma: 0.1f);
var candidates = strategy.Ask();            // 0 B
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

### Active

Autograd ownership cleanup (PR5) — in progress:

- ✅ Etap 1: Graph operation facade (`graph.Linear`, `graph.Conv2D`, `graph.Relu`, `graph.SoftmaxCrossEntropy`)
- ✅ Etap 2: `AutogradNodeOwnership` enum + metadata on `AutogradNode`
- ⏳ Etap 3: Graph factory methods (`graph.CreateTemporary`, `graph.CreateExternalBorrowed`)
- ⏳ Etap 4: `Parameter` type — long-lived model state separate from graph temporaries
- ⏳ Etap 5: Migrate `LinearLayer` to `Parameter`
- ⏳ Etap 6: Migrate optimizers to `IEnumerable<Parameter>`
- ⏳ Etap 7: Move graph-aware TensorMath into `ComputationGraph.*`
- ⏳ Etap 8: Graph reset/disposal by ownership

### Near-term (after PR5)

- ONNX operator coverage: `Sigmoid`, `Tanh`, `Softmax`, `GlobalAveragePool`, `BatchNormalization`, Conv with padding
- Backward kernel cleanup (span-only paths for Linear/Conv backward)
- Batched GEMM kernel for batch ≥ 64

### Long-term

- Graph compilation for fixed-shape graphs
- Mixed precision / quantization for inference
- Optional GPU backend

---

## What Overfit is not

Not a PyTorch/TensorFlow replacement. Not GPU-first. Not transformer-scale first. Not a Python shim.

The differentiator: pure C#, predictable allocation behavior, competitive CPU inference for small/medium models where managed zero-allocation matters.