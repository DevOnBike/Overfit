# Overfit

![NuGet version](https://img.shields.io/nuget/v/DevOnBike.Overfit.svg)
![Build Status](https://github.com/DevOnBike/Overfit/actions/workflows/dotnet.yml/badge.svg)
![License: Dual](https://img.shields.io/badge/license-AGPLv3%20%2F%20Commercial-blue.svg)

A high-performance, zero-allocation machine-learning engine in pure C#.

Overfit is a ground-up deep-learning, inference, evolutionary optimization and data-preprocessing framework built for modern .NET. It targets small and medium CPU inference workloads where predictable latency, no native runtime dependency, no Python sidecar and zero managed allocations matter.

```text
Training path:   AutogradNode -> ComputationGraph -> TensorMath/Ops -> Optimizer
Inference path:  InferenceEngine -> Sequential -> layer ForwardInference -> Kernels
Hot math:        Kernels/LinearKernels.cs, Conv2DKernels.cs, ActivationKernels.cs, PoolingKernels.cs
```

## Which guide is right for you?

| If you are... | Start here |
|---|---|
| .NET architect building ML microservices | `docs/scenarios/aspnet-microservice.md` |
| Game developer running neural nets at frame rate | `docs/scenarios/game-ai.md` |
| Embedded / IoT engineer deploying to edge devices | `docs/scenarios/edge-iot.md` |
| Finance / low-latency engineer | `docs/scenarios/finance-latency.md` |
| ML engineer coming from PyTorch | `docs/scenarios/for-pytorch-users.md` |

See `ROADMAP.md` for planned features and current priorities.

---

What is intentionally not claimed:

- Training is not zero-allocation yet. Training benchmarks are performance trend checks.
- Large batched GEMM workloads are not the strongest Overfit case yet.
- CNN and larger MLP benchmarks are treated as roughly tied with ONNX Runtime, with Overfit's main advantage being `0 B/op`.
- Thread-scaling and kernel/training benchmarks are diagnostic unless their individual benchmark output has no setup warnings.

## Current benchmark snapshot

Hardware and runtime:

```text
CPU: AMD Ryzen 9 9950X3D, 16 physical / 32 logical cores
OS: Windows 11 25H2
Runtime: .NET 10.0.7
JIT: RyuJIT x86-64-v4
BenchmarkDotNet: 0.15.8
```

All inference results below use the current `InferenceEngine.Run(...)` hot path with preallocated input/output buffers. ONNX Runtime results use pre-created `OrtValue` buffers where applicable. Remaining ONNX allocation is runtime wrapper/managed overhead reported by BenchmarkDotNet.

### Linear inference: strongest zero-allocation path

| Benchmark | Overfit | ONNX Runtime | Allocations |
|---|---:|---:|---:|
| `SingleInferenceBenchmark`, Linear(784,10) | ~252 ns | ~2.14 us | Overfit 0 B, ONNX 224 B |
| `OnnxLinearInferenceBenchmarks`, Linear(784,10) | ~272 ns | ~2.18 us | Overfit 0 B, ONNX 224 B |
| `ThroughputBenchmark`, Linear(784,10) | ~253 ns/op | ~1.87 us/op | Overfit 0 B, ONNX 224 B |
| `TailLatencyBenchmark`, Linear(784,10) | ~299 ns/op | ~2.26 us/op | Overfit 0 B, ONNX 224 B |

Observed result: Overfit is roughly 7-9x faster than the preallocated ONNX Runtime path for repeated single-sample Linear(784,10) inference while remaining allocation-free.

### Scaling by input size

| Model | Overfit | ONNX Runtime | Allocations |
|---|---:|---:|---:|
| Linear(64,10) | ~80 ns | ~1.39 us | Overfit 0 B, ONNX 224 B |
| Linear(784,10) | ~210 ns | ~1.87 us | Overfit 0 B, ONNX 224 B |
| Linear(4096,10) | ~1.08 us | ~3.74 us | Overfit 0 B, ONNX 224 B |

Observed result: the smaller the model, the more ONNX Runtime fixed overhead dominates. Overfit keeps the path in managed code with no hot-path allocation.

### MLP and CNN inference

| Benchmark | Model | Overfit | Comparison | Allocations |
|---|---|---:|---:|---:|
| `OnnxMlpInferenceBenchmarks` | 784 -> 128 -> 10 | ~3.7 us | ONNX ~5.2 us | Overfit 0 B, ONNX 224 B |
| `MultiLayerInferenceBenchmark` | 784 -> 256 -> 128 -> 10 | ~10-12 us | ONNX ~10-11 us | Overfit 0 B, ONNX 224 B |
| `OnnxCnnInferenceBenchmarks` | Conv/ReLU/Pool/GAP/Linear | ~5-6.5 us | ONNX ~6-7.7 us | Overfit 0 B, ONNX 224 B |

Observed result: Overfit wins the smaller MLP benchmark and roughly matches ONNX Runtime on larger 3-layer MLP and small CNN workloads while staying allocation-free.

### ML.NET API-level single inference

| Runtime | Mean | Allocated |
|---|---:|---:|
| ML.NET `PredictionEngine`, fresh input | ~10.1 us | ~4.6 KB |
| ML.NET `PredictionEngine`, reused input | ~10.1 us | ~4.5 KB |
| ONNX Runtime preallocated | ~11.2 us | 224 B |
| Overfit `InferenceEngine.Run(...)` | ~12.1 us | 0 B |

Observed result: ML.NET can be slightly faster in this API-level 3-layer MLP run, but it allocates several KB/op. Overfit is the only 0 B/op path in this comparison.

### Batch scaling

| Batch | Overfit | ONNX Runtime | Result |
|---:|---:|---:|---|
| 1 | ~286 ns, 0 B | ~1.90 us, 224 B | Overfit wins |
| 16 | ~3.0 us, 0 B | ~3.7 us, 224 B | Overfit close / wins in this run |
| 64 | ~14.4 us, 0 B | ~7.2 us, 224 B | ONNX wins |
| 256 | ~47.4 us, 0 B | ~23.4 us, 224 B | ONNX wins |

Observed result: Overfit is excellent for batch-1 and small batches. Larger batches favor ONNX Runtime because it uses batched GEMM-style execution. The next Overfit performance target is `LinearKernels.ForwardBatched(...)`.

### Concurrent inference

| Benchmark | Overfit | ONNX Runtime |
|---|---:|---:|
| Concurrent single-sample inference | ~516 ms, 0 B | ~1811 ms, ~117 MB |

Observed result: Overfit performs well in a zero-contention concurrent scenario where each worker owns its own model, engine and buffers.

### Manual baseline comparison

| Method | Mean | Allocated |
|---|---:|---:|
| Manual single linear | ~225-228 ns | 0 B |
| Overfit single linear | ~228-232 ns | 0 B |
| Manual multilayer baseline | ~3.61 us | 0 B |
| Overfit multilayer baseline | ~3.61 us | 0 B |

Observed result: Overfit inference overhead is effectively at manual-code level in these internal baseline benchmarks.

### Training and strategy benchmarks

Training is not expected to be zero-allocation in the same sense as inference.

| Benchmark | Mean | Allocated | Notes |
|---|---:|---:|---|
| `TrainingEngine_Mlp_TrainBatch` | ~468 us | ~26.8 KB | trend benchmark, allocations allowed |
| OpenAI-ES tiny `Ask` | ~1.8 us | 0 B | zero-allocation strategy buffer path |
| OpenAI-ES tiny `AskThenTell` | ~5-6 us | 0 B | zero-allocation full strategy cycle |

### Thread-scaling diagnostics

Thread-scaling benchmarks are directional, not zero-allocation claims. Current useful signals:

| Workload | 1 thread | 8 threads | 16 threads | Notes |
|---|---:|---:|---:|---|
| MatMul MSE backward | ~3.23 ms | ~1.21 ms | ~1.22 ms | scales well to ~8 threads |
| ResidualBlock MSE backward | ~7.97 ms | ~2.27 ms | ~2.10 ms | scales well to 16 threads |
| ResidualBlock forward inference-style | ~2.60 ms | ~1.11 ms | ~1.34 ms | best around 8 threads |

---

## Installation

```bash
dotnet add package DevOnBike.Overfit
```

Requires .NET 10 or later. No Python runtime, model server or native ML runtime is required for Overfit inference.

---

## Quick start: zero-allocation inference

Use `InferenceEngine` for production inference. It hides `Eval()`, `PrepareInference(...)`, workspace allocation and warmup.

```csharp
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Inference;

using var model = new Sequential(
    new LinearLayer(784, 128),
    new ReluActivation(),
    new LinearLayer(128, 10));

using var engine = InferenceEngine.FromSequential(
    model,
    inputSize: 784,
    outputSize: 10,
    new InferenceEngineOptions
    {
        WarmupIterations = 16,
        MaxIntermediateElements = 64 * 1024
    });

var input = new float[784];
var output = new float[10];

engine.Run(input, output);
```

For convenience, `Predict` returns an internal preallocated output span. The span is overwritten on the next call.

```csharp
ReadOnlySpan<float> prediction = engine.Predict(input);
```

For batched input:

```csharp
var batchInput = new float[784 * 32];
var batchOutput = new float[10 * 32];

engine.Run(batchInput, batchOutput);
```

---

## Quick start: training facade

The training facade hides repeated training-loop boilerplate while leaving optimizer and loss selection extensible.

```csharp
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Training;

const int batchSize = 64;
const int inputSize = 784;
const int classCount = 10;

using var model = new Sequential(
    new LinearLayer(inputSize, 128),
    new ReluActivation(),
    new LinearLayer(128, classCount));

using var adam = new Adam(
    model.Parameters(),
    learningRate: 0.001f);

var optimizer = new DelegateTrainingOptimizer(
    zeroGrad: adam.ZeroGrad,
    step: adam.Step);

var loss = new DelegateTrainingLoss(
    forward: (graph, prediction, target) =>
        TensorMath.SoftmaxCrossEntropy(graph, prediction, target),
    backward: (graph, lossNode) => graph.Backward(lossNode));

using var trainer = TrainingEngine.FromBackend(
    new SequentialTrainingBackend(
        model,
        optimizer,
        loss,
        batchSize,
        inputSize,
        classCount));

TrainingStepResult result = trainer.TrainBatch(batchInput, batchTarget);
Console.WriteLine(result.Loss);
```

Training constructs and traverses the autograd graph. It is measured as a performance trend, not as a zero-allocation gate.

---

## CNN inference example

```csharp
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Inference;

const int inputChannels = 1;
const int inputH = 28;
const int inputW = 28;
const int convOutChannels = 8;
const int kernel = 3;
const int pool = 2;
const int outputClasses = 10;

const int convOutH = inputH - kernel + 1;
const int convOutW = inputW - kernel + 1;
const int poolOutH = convOutH / pool;
const int poolOutW = convOutW / pool;

using var model = new Sequential(
    new ConvLayer(inputChannels, convOutChannels, inputH, inputW, kernel),
    new ReluActivation(),
    new MaxPool2DLayer(convOutChannels, convOutH, convOutW, pool),
    new GlobalAveragePool2DLayer(convOutChannels, poolOutH, poolOutW),
    new LinearLayer(convOutChannels, outputClasses));

using var engine = InferenceEngine.FromSequential(
    model,
    inputSize: inputChannels * inputH * inputW,
    outputSize: outputClasses);

var input = new float[inputChannels * inputH * inputW];
var output = new float[outputClasses];

engine.Run(input, output);
```

---

## Architecture overview

| Area | Responsibility |
|---|---|
| `Tensors` | storage, views, pooling and span-backed tensor primitives |
| `Autograd` | reverse-mode automatic differentiation via `ComputationGraph` and `AutogradNode` |
| `Ops` | training-path tensor math and graph-aware operations |
| `Kernels` | low-level inference math and hot loops, including SIMD paths |
| `DeepLearning` | high-level modules: `Sequential`, layers, pooling, activations, recurrent layers |
| `Inference` | user-facing inference facade and backend abstraction |
| `Training` | user-facing training facade and backend/loss/optimizer abstractions |
| `Optimizers` | Adam, AdamW, SGD and parameter update logic |
| `Evolutionary` | gradient-free optimization and population evaluation |
| `Diagnostics` | optional timing/allocation tracing |

Current inference design rule:

```text
Layers own shape, parameters, save/load and train/eval state.
Kernels own hot math.
TensorMath/Ops own autograd/training graph math.
InferenceEngine owns prepared execution and workspace setup.
```

---

## Evolutionary training engine

Overfit includes gradient-free optimization for black-box training and simulation-based tasks.

- `GenerationalGeneticAlgorithm` — elitist GA with truncation selection and Gaussian mutation.
- `OpenAiEsStrategy` — Natural Evolution Strategies with shared `PrecomputedNoiseTable` and antithetic sampling.
- `ParallelPopulationEvaluator` — thread-pool fan-out with lazy per-worker context reuse.

Typical flow:

```csharp
using var evaluator = new ParallelPopulationEvaluator(
    evaluator: new MyFitnessFunction(),
    contextFactory: () => BuildContext(),
    contextDispose: ctx => ctx.Dispose());

using var strategy = new OpenAiEsStrategy(
    populationSize: 256,
    parameterCount: adapter.ParameterCount,
    sigma: 0.1f,
    learningRate: 0.01f,
    noiseTable: new PrecomputedNoiseTable(length: 1 << 24, seed: 42));

strategy.Initialize();

var genomes = new float[256 * strategy.ParameterCount];
var fitness = new float[256];

for (var generation = 0; generation < generations; generation++)
{
    strategy.Ask(genomes);
    evaluator.Evaluate(genomes, fitness, 256, strategy.ParameterCount);
    strategy.Tell(fitness);
}
```

---

## When to choose Overfit

Overfit is a good fit when:

- inference is small or medium sized and latency-sensitive;
- you need zero managed allocations in the inference hot path;
- you want in-process inference in a .NET service;
- you want Native AOT-friendly deployment;
- a Python sidecar or native model runtime is operationally undesirable;
- you need simple, controllable CPU inference kernels.

ONNX Runtime or other native runtimes may be the better fit when:

- workloads are large and heavily batched;
- framework interchange with PyTorch, TensorFlow or external model export is required;
- GPU execution is required;
- transformer-scale kernels dominate the workload.

---

## Licensing

Overfit uses a dual-license model.

1. Open source: GNU AGPLv3.
2. Commercial license: for proprietary, closed-source or enterprise use.

For commercial licensing, contact: devonbike@gmail.com.

---

## Contributing

For performance-sensitive changes, include:

- a correctness test;
- before/after BenchmarkDotNet output;
- allocation measurements where relevant;
- documentation updates when public behavior changes.

Suggested local validation:

```bash
dotnet test -c Release
dotnet run -c Release --project Sources/Benchmark --filter "*SingleInferenceBenchmark*"
dotnet run -c Release --project Sources/Benchmark --filter "*ScalingBenchmark*"
dotnet run -c Release --project Sources/Benchmark --filter "*ThroughputBenchmark*"
dotnet run -c Release --project Sources/Benchmark --filter "*OnnxMlpInferenceBenchmarks*"
dotnet run -c Release --project Sources/Benchmark --filter "*OnnxCnnInferenceBenchmarks*"
```
