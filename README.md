# Overfit

![NuGet version](https://img.shields.io/nuget/v/DevOnBike.Overfit.svg)
![Build Status](https://github.com/DevOnBike/Overfit/actions/workflows/dotnet.yml/badge.svg)
![License: Dual](https://img.shields.io/badge/license-AGPLv3%20%2F%20Commercial-blue.svg)

A high-performance, zero-allocation machine-learning engine in pure C#.

Overfit is a ground-up deep-learning, inference, evolutionary optimization and data-preprocessing framework built for modern .NET. It targets small and medium CPU inference workloads where predictable latency, no native runtime dependency, no Python sidecar and zero managed allocations matter.

The current architecture separates training, inference and low-level math:

```text
Training path:
  AutogradNode -> ComputationGraph -> TensorMath -> Optimizer

Inference path:
  InferenceEngine -> Sequential -> layer ForwardInference -> Kernels

Hot math:
  Kernels/LinearKernels.cs
  Kernels/Conv2DKernels.cs
  Kernels/ActivationKernels.cs
  Kernels/PoolingKernels.cs
```

## Which guide is right for you?

| If you are... | Start here |
|---|---|
| .NET architect building ML microservices | `docs/scenarios/aspnet-core.md` |
| Game developer running NNs at frame rate | `docs/scenarios/game-ai.md` |
| Embedded / IoT engineer deploying to edge devices | `docs/scenarios/edge-iot.md` |
| Finance / low-latency engineer | `docs/scenarios/finance-low-latency.md` |
| ML engineer coming from PyTorch | `docs/scenarios/pytorch-users.md` |

See `ROADMAP.md` for planned features and current priorities.

---

## Current benchmark snapshot

Hardware and runtime:

```text
CPU: AMD Ryzen 9 9950X3D, 16 physical / 32 logical cores
OS: Windows 11 25H2
Runtime: .NET 10.0.7
JIT: RyuJIT x86-64-v4
BenchmarkDotNet: 0.15.8
```

All results below use preallocated input/output buffers and exclude model construction. ONNX Runtime results use pre-created `OrtValue` buffers. The measured remaining allocation is ONNX Runtime wrapper/managed overhead.

### MLP inference: Overfit vs ONNX Runtime

Model:

```text
Linear(784, 128)
ReLU
Linear(128, 10)
```

| Method | Mean | Allocated |
|---|---:|---:|
| Overfit_Mlp_ZeroAlloc | 3.63 us | 0 B |
| OnnxRuntime_Mlp_TrueZeroAlloc | 4.65-5.05 us | 224 B |

Observed result: Overfit is roughly 25-31% faster for this small MLP and remains zero-allocation.

### CNN inference: Overfit vs ONNX Runtime

Model:

```text
Conv2D(1 -> 8, 3x3)
ReLU
MaxPool2D(2x2)
GlobalAveragePool2D
Linear(8, 10)
```

| Method | Mean | Allocated |
|---|---:|---:|
| Overfit_Cnn_ZeroAlloc | 4.73-5.35 us | 0 B |
| OnnxRuntime_Cnn_TrueZeroAlloc | 6.03-6.20 us | 224 B |

Observed result: Overfit is faster for this small CNN and remains zero-allocation.

### Manual baseline comparison

The manual baseline runs equivalent direct loops using the same weights. This is used to detect framework overhead.

| Method | Mean | Allocated |
|---|---:|---:|
| Manual_SingleLinear_TrueZeroAlloc | 225.5 ns | 0 B |
| Overfit_SingleLinear_ZeroAlloc | 227.5 ns | 0 B |
| Overfit_MultiLayer_ZeroAlloc | 3,622.8 ns | 0 B |
| Manual_MultiLayer_TrueZeroAlloc | 3,658.0 ns | 0 B |

Observed result: Overfit inference is effectively at manual-code performance for these workloads.

### Notes on benchmark interpretation

- These are small CPU inference workloads. Larger batched workloads can favor ONNX Runtime or other native kernels.
- Benchmark numbers vary with CPU boost state, scheduler state, thermals and benchmark configuration.
- Performance-sensitive PRs should include before/after BenchmarkDotNet output.

---

## Installation

```bash
dotnet add package DevOnBike.Overfit
```

Requires .NET 10 or later. No Python runtime is required. No model server is required.

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

The training facade hides the repeated training-loop boilerplate while leaving optimizer and loss selection extensible.

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
    backward: (graph, lossNode) =>
        graph.Backward(lossNode));

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

Training is not expected to be zero-allocation in the same sense as inference; it constructs and traverses the autograd graph.

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
| `DeepLearning` | high-level modules: `Sequential`, `LinearLayer`, `ConvLayer`, pooling, activations, recurrent layers |
| `Inference` | user-facing inference facade and backend abstraction |
| `Training` | user-facing training facade and backend/loss/optimizer abstractions |
| `Optimizers` | Adam, AdamW, SGD and parameter update logic |
| `Data` | tabular preprocessing and feature engineering pipelines |
| `Evolutionary` | gradient-free optimization and population evaluation |
| `Diagnostics` | optional timing/allocation tracing |

Current inference design rule:

```text
Layers own shape, parameters, save/load and train/eval state.
Kernels own hot math.
TensorMath owns autograd/training graph math.
InferenceEngine owns prepared execution and workspace setup.
```

---

## Evolutionary training engine

Overfit includes gradient-free optimization for black-box training and simulation-based tasks.

- `GenerationalGeneticAlgorithm` — elitist GA with truncation selection and Gaussian mutation.
- `OpenAiEsStrategy` — Natural Evolution Strategies with shared `PrecomputedNoiseTable` and antithetic sampling.
- `ParallelPopulationEvaluator<TContext>` — thread-pool fan-out with lazy per-worker context reuse.

Typical flow:

```csharp
using var evaluator = new ParallelPopulationEvaluator<MyContext>(
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

- you need framework interchange with PyTorch, TensorFlow or external model export;
- workloads are large and heavily batched;
- you need transformer-scale kernels;
- GPU execution is required.

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
dotnet run -c Release --project Sources/Benchmark --filter "*InferenceZeroAllocBenchmarks*"
dotnet run -c Release --project Sources/Benchmark --filter "*OnnxMlpInferenceBenchmarks*"
dotnet run -c Release --project Sources/Benchmark --filter "*OnnxCnnInferenceBenchmarks*"
```
