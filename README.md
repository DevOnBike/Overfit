# Overfit 🚀

[![NuGet version](https://img.shields.io/nuget/v/DevOnBike.Overfit.svg)](https://www.nuget.org/packages/DevOnBike.Overfit)
[![Build Status](https://img.shields.io/github/actions/workflow/status/DevOnBike/Overfit/ci.yml?branch=main)](https://github.com/DevOnBike/Overfit/actions)
[![License: Dual (AGPLv3 / Commercial)](https://img.shields.io/badge/License-Dual-blue.svg)](LICENSE.md)

**A High-Performance, Zero-Allocation Machine Learning Engine in Pure C#.**

Overfit is a ground-up Deep Learning and Data Preprocessing framework built specifically for modern .NET. It brings the power of neural networks, advanced feature selection, and tabular data pipelines directly to C# **without relying on Python wrappers or heavy external C++ binaries**.

Designed for maximum CPU inference speed, Overfit embraces aggressive memory management, SIMD potential, and full Native AOT compatibility.

---

## 👥 Which Guide Is Right For You?

Overfit serves different needs depending on your role. Jump to the scenario that matches your situation:

| If you are... | Read this |
|--------------|-----------|
| 🏗️ **.NET Architect** building microservices with ML | [ASP.NET Core integration guide](docs/scenarios/aspnet-microservice.md) |
| 🎮 **Game developer** running NNs at frame-rate | [Game AI guide](docs/scenarios/game-ai.md) |
| 📡 **Embedded / IoT engineer** deploying to edge devices | [Edge & IoT guide](docs/scenarios/edge-iot.md) |
| 💰 **Finance / HFT engineer** with tail-latency requirements | [Finance & Low-Latency guide](docs/scenarios/finance-latency.md) |
| 🐍 **ML engineer** coming from PyTorch | [For PyTorch users](docs/scenarios/for-pytorch-users.md) |

See [ROADMAP.md](ROADMAP.md) for planned features and current priorities.

---

## ⚡ Benchmarks: Overfit vs ONNX Runtime

Single-layer MLP (784→10), measured on AMD Ryzen 9 9950X3D, .NET 10, Windows 11.
All benchmarks use identical weights exported from the same PyTorch model.

### Single Inference Latency

| Method | Mean | Allocated | Ratio |
|--------|-----:|----------:|------:|
| **Overfit** | **188 ns** | **0 B** | **17.7×** |
| ONNX Runtime | 3,331 ns | 912 B | 1.0× |

### Batch Scaling

| Batch Size | Overfit | ONNX Runtime | Winner | Overfit Alloc |
|-----------:|--------:|-------------:|:------:|--------------:|
| 1          | 188 ns   | 3,331 ns    | **Overfit 17.7×** | 0 B |
| 16         | 2,909 ns | 4,541 ns    | **Overfit 1.56×** | 0 B |
| 64         | 14,538 ns | 5,564 ns   | ONNX 2.61× | 0 B |
| 256        | 18,852 ns | 12,242 ns  | ONNX 1.54× | 6.8 KB |

### Throughput (10,000 calls)

| Method | Mean | Gen0 | Allocated | Ratio |
|--------|-----:|-----:|----------:|------:|
| **Overfit** | **3.88 ms** | **0** | **0 B** | **9.2×** |
| ONNX Runtime | 35.60 ms | 154 | 9.12 MB | 1.0× |

### Concurrent (8 threads × 1,000 calls)

| Method | Mean | Gen0 | Allocated | Ratio |
|--------|-----:|-----:|----------:|------:|
| **Overfit** | **824 μs** | **0** | **3.3 KB** | **13.4×** |
| ONNX Runtime | 11,034 μs | 141 | 7.1 MB | 1.0× |

### Cold Start (load + first prediction)

| Method | Mean | Ratio |
|--------|-----:|------:|
| **Overfit** | **291 μs** | **4.2×** |
| ONNX Runtime | 1,224 μs | 1.0× |

### Tail Latency (100,000 calls)

| Metric | Overfit | ONNX Runtime |
|--------|--------:|-------------:|
| P50 | 0.40 μs | 3.00 μs |
| P99 | 0.60 μs | 3.70 μs |
| P99.9 | 0.80 μs | 5.70 μs |
| **Max** | **8.40 μs** | **2,184 μs** |
| GC Gen-0 | 0 | 1 |

### Scaling by Model Size

| Model | Overfit | ONNX Runtime | Ratio |
|-------|--------:|-------------:|------:|
| 128 → 10 | 74 ns | 3,116 ns | **42.1×** |
| 784 → 10 | 188 ns | 3,331 ns | **17.7×** |
| 4096 → 10 | 1,972 ns | 5,461 ns | **2.8×** |

> **Key takeaway:** ONNX Runtime carries ~3 μs of fixed overhead per call (P/Invoke marshalling, OrtValue allocation, managed↔native transitions). The smaller the model, the more this overhead dominates. At 128→10, the overhead is **42× larger than the actual math**. Overfit eliminates this entirely — pure managed SIMD with zero allocations.

**When to choose Overfit:** single-sample inference, edge/IoT, real-time streaming, game engines, microservices with tight latency budgets, batches ≤ 16.

**When to choose ONNX Runtime:** server-side batch inference (≥ 64), large transformer models, framework interop with PyTorch/TensorFlow.

---

## 🎬 Showcase: Unity Swarm Demo

**100,000 neural networks running at 60 FPS in Unity.** Each bot makes independent decisions using its own evolved brain (10 parameters via Linear 4→2). Genetic algorithm trains the swarm to orbit a target while avoiding a predator — orbital behavior emerges in under 60 seconds of training.

- Pure C#, zero GC allocations in the hot path
- 100k inferences per frame (~1.7 ms at 60 FPS)
- Offline training: 400 generations in **36 seconds** on Ryzen 9 9950X3D

See [`Demo/Unity/`](Demo/Unity) for the full code, or jump to the [Game AI scenario](docs/scenarios/game-ai.md) for integration details.

---

## 🧬 Evolutionary Training Engine

Overfit includes a production-grade evolutionary optimization module for gradient-free training — useful when your fitness function is non-differentiable, when the environment is a black box (game engine, simulator, trading backtester), or when you just need a second optimizer to contrast against Adam.

Two algorithms ship out of the box, both behind a unified **Ask/Tell API** and with **zero per-generation allocations**:

- **`GenerationalGeneticAlgorithm`** — classic elitist GA with truncation selection, Gaussian mutation, and centered-rank fitness shaping. NaN-safe ranking via an O(n log k) partial sort.
- **`OpenAiEsStrategy`** — Natural Evolution Strategies in the style of Salimans et al. 2017. Maintains a single mean parameter vector, uses a shared `PrecomputedNoiseTable` plus antithetic sampling to estimate the search gradient from population fitness.

### Per-generation cost (Ryzen 9 9950X3D, .NET 10)

Synthetic random fitness, same centered-rank shaper for both algorithms. Real training spends its wallclock in fitness evaluation — these numbers answer one narrow question: *which algorithm has a cheaper per-generation step?*

| Population × Params | GA Ask+Tell | ES Ask+Tell | ES speedup |
|---|---:|---:|---:|
| 256 × 64    | 250 μs | 42 μs | **6.0×** |
| 256 × 256   | 842 μs | 68 μs | **12.5×** |
| 1024 × 64   | 984 μs | 135 μs | **7.3×** |
| 1024 × 256  | 3,345 μs | 236 μs | **14.2×** |

### When to pick which

| Genome size | Problem type | Recommendation |
|---|---|---|
| < 50 params | Anything | GA works fine; speedup from ES is negligible at this scale |
| 50–500 params | Smooth control (RL, robotics) | **ES** — 3–10× faster per generation, lower gradient variance thanks to antithetic sampling |
| > 1000 params | Neural-network weights, large policies | **ES** — 10–50× faster per generation; the noise table pays for itself on the first generation |
| Any | Discrete / combinatorial / multi-modal | **GA** — population diversity escapes local optima that ES's single μ gets stuck in |

### Parallel fitness evaluation

Evolutionary training is bottlenecked by fitness evaluation, not by the algorithm itself. `ParallelPopulationEvaluator<TContext>` fans candidate evaluation across the thread pool with a lazy per-thread context pool, so each worker reuses its own neural network across generations instead of paying construction cost per candidate.

```csharp
// One network per worker thread, reused across every generation.
using var evaluator = new ParallelPopulationEvaluator<(IModule Net, NeuralNetworkParameterAdapter Adapter)>(
    evaluator:       new MyFitnessFunction(env),
    contextFactory:  () => BuildNetworkAndAdapter(),
    contextDispose:  ctx => ctx.Net.Dispose());

using var strategy = new OpenAiEsStrategy(
    populationSize: 256, parameterCount: adapter.ParameterCount,
    sigma: 0.1f, learningRate: 0.01f,
    noiseTable: new PrecomputedNoiseTable(length: 1 << 24, seed: 42));

strategy.Initialize();

var genomes = new float[256 * strategy.ParameterCount];
var fitness = new float[256];

for (var gen = 0; gen < generations; gen++)
{
    strategy.Ask(genomes);
    evaluator.Evaluate(genomes, fitness, 256, strategy.ParameterCount);
    strategy.Tell(fitness);

    if (gen % 100 == 0)
    {
        using var fs = File.Create($"checkpoint-{gen}.bin");
        using var bw = new BinaryWriter(fs);
        strategy.Save(bw);
    }
}
```

Both strategies implement `IEvolutionCheckpoint`, so long-running jobs can checkpoint to disk and resume across process restarts.

---

## 🔥 Key Features

* **Zero-Allocation Hot Paths:** Built heavily on `Span<T>` and custom memory pooling. Training loops and inference passes avoid triggering the Garbage Collector (GC), ensuring flat-line CPU usage.
* **100% Native AOT Compatible:** Free from runtime reflection (`System.Reflection`), `dynamic` typing, and `Reflection.Emit`. Overfit compiles perfectly into tiny, standalone native binaries with instant cold-starts.
* **Dynamic Autograd Engine:** Features a scratch-built `ComputationGraph` for automatic differentiation (Reverse-mode AutoDiff).
* **Deep Learning Toolkit:** Out-of-the-box support for MLPs, Convolutional Neural Networks (Conv2D, MaxPool, GlobalAveragePool), Residual Blocks, Batch Normalization, and LSTMs.
* **AVX-512 Optimized Optimizers:** Adam and AdamW with vectorized parameter updates (16 floats per cycle on supported CPUs).
* **Advanced Data Pipelines:** Production-ready `DataPipeline` including Boruta Feature Selection, Correlation Filters, Robust Scaling, and Outlier Clipping.
* **Reinforcement Learning:** Easily adaptable for RL scenarios (e.g., Q-Learning for game agents).
* **MNIST Benchmark:** Full 60k training set, ResNet-style architecture, ~26 seconds per epoch reaching 99% test accuracy.

---

## 📦 Installation

Install via NuGet Package Manager:
```bash
dotnet add package DevOnBike.Overfit
```

Requires .NET 10 or later. No native dependencies.

---

## ⚡ Quick Start

### 1. Building a Neural Network

Overfit makes it easy to build, train, and run inference on deep neural networks.

```csharp
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Optimizers;

// Define a ResNet-style architecture or MLP
var model = new Sequential(
    new ConvLayer(inChannels: 1, outChannels: 8, h: 28, w: 28, kSize: 3),
    new ReluActivation(),
    new MaxPool2D(poolSize: 2),
    // ... flattening ...
    new LinearLayer(1352, 10)
);

// High-performance optimizers out of the box
using var optimizer = new Adam(model.Parameters(), learningRate: 0.001f);

// Null ComputationGraph disables the tape for ultra-fast Zero-Allocation Inference
var prediction = model.Forward(null, inputTensor);
```

### 2. Robust Data Preprocessing

Taming messy tabular data before feeding it to a model:

```csharp
using DevOnBike.Overfit.Data.Prepare;

var pipeline = new DataPipeline()
    .AddLayer(new TechnicalSanityLayer(maxCorruptedRatio: 0.3f))
    .AddLayer(new ConstantColumnFilterLayer())
    .AddLayer(new OutlierClipLayer(lowerPercentile: 0.01f, upperPercentile: 0.99f))
    .AddLayer(new RobustScalingLayer(columnIndices: numericColumns));

using var cleanData = pipeline.Execute(rawFeatures, rawTargets);
```

### 3. ASP.NET Core Integration

Deploy a trained model as a REST microservice in three lines:

```csharp
using DevOnBike.Overfit.DeepLearning;

var builder = WebApplication.CreateBuilder(args);

// Load the model once at startup - thread-safe for inference
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

No Python sidecar. No inter-process serialization. Model lives in the same process as your business logic, with sub-microsecond latency per request. See the [full ASP.NET guide](docs/scenarios/aspnet-microservice.md) for deployment patterns.

---

## 💡 Why Pure C# and Native AOT?

Most .NET ML libraries act as bridges to PyTorch, TensorFlow, or ONNX Runtime. While powerful, they drag along massive dependencies (often gigabytes of CUDA libraries and Python environments).

**Overfit is different.**
By writing the math and the autograd engine entirely in modern C# (utilizing SIMD and memory-safe structures), Overfit allows you to deploy intelligent applications as **single-file native executables**.
Whether you're building a microservice, a high-frequency trading bot, or an embedded IoT application, Overfit runs with predictable latency and a tiny memory footprint.

### The "Python Tax"

Many .NET shops maintain a parallel Python stack (FastAPI, conda environments, separate containers) just to serve ML models — while the rest of their backend runs on .NET. This duplication adds operational overhead, increases the attack surface, and introduces cross-process serialization latency.

Overfit eliminates this by keeping inference in-process, in the same language as the rest of the application.

---

## 🧱 Architecture Overview

Overfit is organized into clear layers:

- **Tensors** (`DevOnBike.Overfit.Tensors`) — `FastTensor<T>`, `TensorView<T>`, `ReadOnlyTensorView<T>`. Zero-allocation memory primitives.
- **Autograd** (`DevOnBike.Overfit.Autograd`) — Reverse-mode automatic differentiation via `ComputationGraph` and `AutogradNode`.
- **Ops** (`DevOnBike.Overfit.Ops`) — Tensor mathematics: `TensorMath.Linear`, `TensorMath.Conv2D`, `TensorMath.BatchNorm1D`, and friends.
- **DeepLearning** (`DevOnBike.Overfit.DeepLearning`) — High-level modules: `Sequential`, `LinearLayer`, `ConvLayer`, `ResidualBlock`, `LSTMLayer`.
- **Optimizers** (`DevOnBike.Overfit.Optimizers`) — `Adam`, `AdamW`, `SGD` with AVX-512 optimized update paths.
- **Data** (`DevOnBike.Overfit.Data`) — Tabular preprocessing pipelines (`DataPipeline`, feature selection, scaling, outlier handling).
- **Evolutionary** (`DevOnBike.Overfit.Evolutionary`) — Gradient-free optimization via `GenerationalGeneticAlgorithm` and `OpenAiEsStrategy` behind a shared Ask/Tell API; parallel fitness fan-out through `ParallelPopulationEvaluator<TContext>`.
- **Diagnostics** (`DevOnBike.Overfit.Diagnostics`) — Optional per-module timing and allocation tracing for training inspection.

See [ROADMAP.md](ROADMAP.md) for what's planned next.

---

## 🎯 Use Cases

Overfit shines in scenarios where ONNX Runtime's per-call overhead dominates or where a Python sidecar is operationally painful:

- **ASP.NET Core microservices** — inference as a first-class citizen of your API, not a separate service. [→ Full guide](docs/scenarios/aspnet-microservice.md)
- **Game engines** — per-frame AI inference for large populations. [→ Full guide](docs/scenarios/game-ai.md)
- **Edge and IoT** — single-file AOT executables for field-deployed inference. [→ Full guide](docs/scenarios/edge-iot.md)
- **High-frequency trading / fraud detection** — predictable sub-microsecond latency with zero GC pauses. [→ Full guide](docs/scenarios/finance-latency.md)
- **Coming from PyTorch** — bridging models between Python research and .NET production. [→ Full guide](docs/scenarios/for-pytorch-users.md)

---

## ⚖️ Dual Licensing

This software is released under a **Dual License model**:

1. **Open Source (GNU AGPLv3):** Free for open-source projects, personal use, and academic research. *Note: If you use this engine in your application (even over a network/API), your entire application must also be open-sourced under the AGPLv3.*
2. **Commercial License:** For businesses building proprietary, closed-source applications or enterprise environments. Purchasing a commercial license frees you from the requirements of the AGPLv3.

**To purchase a commercial license or discuss enterprise support, please contact:** 👉 **devonbike@gmail.com**

---

## 🤝 Contributing

Contributions are welcome! Whether it's adding new activation functions, optimizing tensor math with `System.Numerics.Vectors`, or improving the documentation.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Each PR should include:
- A benchmark comparison (before/after) for performance-sensitive changes
- A correctness test for new functionality
- Documentation updates where relevant

See [ROADMAP.md](ROADMAP.md) for areas where contributions are especially welcome.