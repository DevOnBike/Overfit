# Overfit 🚀

[![NuGet version](https://img.shields.io/nuget/v/DevOnBike.Overfit.svg)](https://www.nuget.org/packages/DevOnBike.Overfit)
[![Build Status](https://img.shields.io/github/actions/workflow/status/DevOnBike/Overfit/ci.yml?branch=main)](https://github.com/DevOnBike/Overfit/actions)
[![License: Dual (AGPLv3 / Commercial)](https://img.shields.io/badge/License-Dual-blue.svg)](LICENSE.md)

**A High-Performance, Zero-Allocation Machine Learning Engine in Pure C#.**

Overfit is a ground-up Deep Learning and Data Preprocessing framework built specifically for modern .NET. It brings the power of neural networks, advanced feature selection, and tabular data pipelines directly to C# **without relying on Python wrappers or heavy external C++ binaries**. 

Designed for maximum CPU inference speed, Overfit embraces aggressive memory management, SIMD potential, and full Native AOT compatibility.

---

## ⚡ Benchmarks: Overfit vs ONNX Runtime

Single-layer MLP (784→10), measured on AMD Ryzen 9 9950X3D, .NET 10, Windows 11.
All benchmarks use identical weights exported from the same PyTorch model.

### Single Inference Latency

| Method | Mean | Allocated | Ratio |
|--------|-----:|----------:|------:|
| **Overfit** | **382 ns** | **0 B** | **9.4×** |
| ONNX Runtime | 3,573 ns | 912 B | 1.0× |

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
| 784 → 10 | 420 ns | 3,534 ns | **8.4×** |
| 4096 → 10 | 1,972 ns | 5,461 ns | **2.8×** |

> **Key takeaway:** ONNX Runtime carries ~3 μs of fixed overhead per call (P/Invoke marshalling, OrtValue allocation, managed↔native transitions). The smaller the model, the more this overhead dominates. At 128→10, the overhead is **42× larger than the actual math**. Overfit eliminates this entirely — pure managed SIMD with zero allocations.

---

## 🔥 Key Features

* **Zero-Allocation Hot Paths:** Built heavily on `Span<T>` and custom memory pooling. Training loops and inference passes avoid triggering the Garbage Collector (GC), ensuring flat-line CPU usage.
* **100% Native AOT Compatible:** Free from runtime reflection (`System.Reflection`), `dynamic` typing, and `Reflection.Emit`. Overfit compiles perfectly into tiny, standalone native binaries with instant cold-starts.
* **Dynamic Autograd Engine:** Features a scratch-built `ComputationGraph` for automatic differentiation (Reverse-mode AutoDiff).
* **Deep Learning Toolkit:** Out-of-the-box support for MLPs, Convolutional Neural Networks (Conv2D, MaxPool, GlobalAveragePool), Residual Blocks, and Batch Normalization.
* **Advanced Data Pipelines:** Production-ready `DataPipeline` including Boruta Feature Selection, Correlation Filters, Robust Scaling, and Outlier Clipping.
* **Reinforcement Learning:** Easily adaptable for RL scenarios (e.g., Q-Learning for game agents).

## 📦 Installation

Install via NuGet Package Manager:
```bash
dotnet add package DevOnBike.Overfit
```

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

---

## 💡 Why Pure C# and Native AOT?

Most .NET ML libraries act as bridges to PyTorch, TensorFlow, or ONNX Runtime. While powerful, they drag along massive dependencies (often gigabytes of CUDA libraries and Python environments). 

**Overfit is different.**
By writing the math and the autograd engine entirely in modern C# (utilizing SIMD and memory-safe structures), Overfit allows you to deploy intelligent applications as **single-file native executables**. 
Whether you're building a microservice, a high-frequency trading bot, or an embedded IoT application, Overfit runs with predictable latency and a tiny memory footprint.

---

## ⚖️ Dual Licensing

This software is released under a **Dual License model**:

1. **Open Source (GNU AGPLv3):** Free for open-source projects, personal use, and academic research. *Note: If you use this engine in your application (even over a network/API), your entire application must also be open-sourced under the AGPLv3.*
2. **Commercial License:** For businesses building proprietary, closed-source applications or enterprise environments. Purchasing a commercial license frees you from the requirements of the AGPLv3.

**To purchase a commercial license or discuss enterprise support, please contact:** 👉 **[Wpisz Twój Email / Link do Strony]**

---

## 🤝 Contributing
Contributions are welcome! Whether it's adding new activation functions, optimizing tensor math with `System.Numerics.Vectors`, or improving the documentation.
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request