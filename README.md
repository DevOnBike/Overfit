# Overfit

Overfit is a C# deep-learning and optimization engine focused on predictable CPU performance, explicit memory ownership and zero-allocation inference hot paths.

The current branch adds an ONNX import MVP: PyTorch-exported models can be loaded into Overfit's `Sequential` pipeline and executed through the allocation-free `InferenceEngine.Run(...)` path.

## Current milestone

The current work validates three connected capabilities:

1. **Zero-allocation CPU inference** for core Overfit inference paths.
2. **ONNX import MVP** for a PyTorch-exported MNIST CNN.
3. **Evolutionary optimization** paths with allocation-free strategy execution in the core `Ask` / `AskThenTell` loops.

The product direction is simple:

> Train in Python or .NET. Import or build the model. Run predictable inference in .NET with explicit buffers and no per-call managed allocation.

## What works now

### Inference

Overfit has a prepared inference path:

```csharp
engine.Run(input, output);
```

The caller owns input and output buffers. The engine owns reusable intermediate buffers. The inference hot path avoids `model.Forward(...)`, `AutogradNode`, `ComputationGraph` and per-call tensor allocation.

### ONNX import MVP

The ONNX importer currently targets a focused PyTorch export path:

```text
PyTorch eval-mode model
  -> ONNX export
  -> OnnxImporter.Load(...)
  -> Sequential
  -> InferenceEngine.Run(...)
```

Current MVP scope:

```text
Conv
Relu
MaxPool
Reshape / Flatten
Gemm
FP32
NCHW
Linear Sequential topology
Opset 11-20
```

Out of scope for the MVP:

```text
Branching DAGs / skip connections
Grouped/depthwise convolutions
Conv padding/stride beyond the supported fixture path
BatchNormalization as a standalone training-mode op
FP16 / INT8 / quantized models
LSTM / GRU ONNX operators
Full ONNX runtime compatibility
```

The goal is not to become a full ONNX Runtime replacement. The goal is a practical bridge:

> Train in PyTorch. Export ONNX. Deploy through Overfit's predictable .NET inference path.

### Evolutionary optimization

Overfit also includes evolutionary optimization for black-box problems where gradients are unavailable, noisy or not the right abstraction.

Typical use cases:

```text
Kubernetes resource/autoscaling tuning
pricing or bidding strategy search
inventory and replenishment optimization
simulation/game AI parameter tuning
industrial process parameter search
```

The strategy benchmarks validate allocation-free `Ask` and `AskThenTell` paths for the core population-search loop.

## Verified benchmark environment

Current benchmark results were collected on:

```text
AMD Ryzen 9 9950X3D
Windows 11 25H2
.NET 10.0.7
BenchmarkDotNet 0.15.8
```

Performance numbers are hardware-specific. Treat them as a snapshot, not a universal guarantee.

## Current benchmark snapshot

### Single linear inference

| Benchmark | Overfit | Comparison | Allocations |
|---|---:|---:|---:|
| Linear(784,10) single inference | ~250-300 ns/op | ONNX Runtime ~2.1-2.3 us/op | Overfit 0 B, ONNX 224 B |
| Linear(784,10) throughput | ~253 ns/op | ONNX Runtime ~1.87 us/op | Overfit 0 B, ONNX 224 B |

### Scaling

| Model | Overfit | ONNX Runtime | Allocations |
|---|---:|---:|---:|
| Linear(64,10) | ~80 ns | ~1.39 us | Overfit 0 B, ONNX 224 B |
| Linear(784,10) | ~210 ns | ~1.87 us | Overfit 0 B, ONNX 224 B |
| Linear(4096,10) | ~1.08 us | ~3.74 us | Overfit 0 B, ONNX 224 B |

### MLP and CNN inference

| Benchmark | Overfit | ONNX Runtime / ML.NET | Allocations |
|---|---:|---:|---:|
| MLP 784->128->10 | ~3.7 us | ONNX ~5.2 us | Overfit 0 B, ONNX 224 B |
| MLP 784->256->128->10 | ~10-12 us | ONNX ~10-11 us | Overfit 0 B, ONNX 224 B |
| Small CNN | ~5-6.5 us | ONNX ~6-7.7 us | Overfit 0 B, ONNX 224 B |
| ML.NET API-level 3-layer MLP | Overfit ~12.1 us | ML.NET ~10.1 us, ONNX ~11.2 us | Overfit 0 B, ML.NET ~4.5 KB, ONNX 224 B |

### Imported ONNX MNIST CNN

| Runtime | Result | Notes |
|---|---:|---|
| Overfit imported ONNX | ~7.5 us/op | BenchmarkDotNet, `InferenceEngine.Run(...)`, 0 B/op |
| ONNX Runtime preallocated | ~7.5 us/op | BenchmarkDotNet, 224 B/op |
| PyTorch eager CPU | ~27.3 us/op | Python reference script, 1 thread |

Interpretation:

> Overfit matches ONNX Runtime on the imported PyTorch MNIST CNN while keeping the .NET inference path allocation-free. The PyTorch number is an external reference script, not a BenchmarkDotNet result.

### Concurrent inference

| Benchmark | Overfit | ONNX Runtime |
|---|---:|---:|
| Concurrent single-sample inference | ~516 ms, 0 B | ~1811 ms, ~117 MB |

This benchmark validates the zero-contention scenario where each worker owns its model, inference engine and buffers.

### Batch scaling

| Batch | Overfit | ONNX Runtime | Result |
|---:|---:|---:|---|
| 1 | ~286 ns, 0 B | ~1.9 us, 224 B | Overfit wins |
| 16 | ~3.0 us, 0 B | ~3.7 us, 224 B | Overfit close / wins in this run |
| 64 | ~14.4 us, 0 B | ~7.2 us, 224 B | ONNX wins |
| 256 | ~47.4 us, 0 B | ~23.4 us, 224 B | ONNX wins |

The current Overfit batch path is allocation-free but still behaves like repeated sample inference. ONNX wins at larger batches through batched GEMM. The next performance target is a batched linear kernel.

### Training

| Benchmark | Mean | Allocations | Notes |
|---|---:|---:|---|
| TrainingEngine MLP TrainBatch | ~468 us | ~26.8 KB | Training allocations are allowed; trend benchmark only |

Training is not currently documented as zero-allocation. Training benchmarks are used for performance trends, not allocation gates.

## ONNX importer usage

```csharp
using DevOnBike.Overfit.Onnx;
using DevOnBike.Overfit.Inference;

var model = OnnxImporter.Load("mnist_cnn.onnx");
model.Eval();

using var engine = InferenceEngine.FromSequential(
    model,
    inputSize: 1 * 28 * 28,
    outputSize: 10);

var input = new float[1 * 28 * 28];
var output = new float[10];

engine.Run(input, output);
```

## Benchmark commands

```bash
dotnet run -c Release --project Sources/Benchmark --filter "*SingleInferenceBenchmark*"
dotnet run -c Release --project Sources/Benchmark --filter "*ScalingBenchmark*"
dotnet run -c Release --project Sources/Benchmark --filter "*ThroughputBenchmark*"
dotnet run -c Release --project Sources/Benchmark --filter "*BatchScalingBenchmark*"
dotnet run -c Release --project Sources/Benchmark --filter "*OnnxLinearInferenceBenchmarks*"
dotnet run -c Release --project Sources/Benchmark --filter "*OnnxMlpInferenceBenchmarks*"
dotnet run -c Release --project Sources/Benchmark --filter "*MultiLayerInferenceBenchmark*"
dotnet run -c Release --project Sources/Benchmark --filter "*OnnxCnnInferenceBenchmarks*"
dotnet run -c Release --project Sources/Benchmark --filter "*ImportedOnnxMnistCnnBenchmark*"
dotnet run -c Release --project Sources/Benchmark --filter "*MLNetSingleInferenceBenchmark*"
dotnet run -c Release --project Sources/Benchmark --filter "*ConcurrentInferenceBenchmark*"
dotnet run -c Release --project Sources/Benchmark --filter "*TrainingEngineBenchmarks*"
```

PyTorch reference benchmark:

```bash
python Tests/benchmark_pytorch_mnist_cnn.py --fixture-dir Tests/test_fixtures --threads 1
```

## Benchmark policy

- Keep zero-allocation claims scoped to measured `InferenceEngine.Run(...)` paths.
- Do not use `model.Forward(...)`, `AutogradNode`, `TensorStorage` or `ComputationGraph` inside inference benchmark methods.
- Use `OperationsPerInvoke` for nanosecond/microsecond benchmark methods.
- Treat `MinIterationTime` as a benchmark defect unless explicitly justified.
- Name ONNX methods `PreAllocated` when ONNX still reports wrapper allocation.
- Use ranges when jitter or multimodal distributions are visible.
- Do not claim Overfit is always faster. The current positioning is predictable zero-allocation inference.

## Current roadmap

1. Finalize ONNX import MVP docs and tests.
2. Keep inference benchmark suite clean and tied to `InferenceEngine.Run(...)`.
3. Implement batched linear kernels for batch 64/256 workloads.
4. Continue graph/autograd ownership cleanup: separate parameters, temporaries, storage and views.
5. Keep training benchmarks as trend benchmarks.
6. Expand ONNX support only after MVP is stable.

## Business-level positioning

Fast is useful. Predictable is more useful.

Overfit is aimed at .NET systems where inference is part of the request path and latency spikes, GC pressure and memory churn matter. The practical story is:

```text
Train in PyTorch or .NET.
Deploy in .NET.
Run with explicit buffers and predictable allocation behavior.
```
