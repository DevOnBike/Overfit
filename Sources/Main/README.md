# Overfit 🚀

[![NuGet version](https://img.shields.io/nuget/v/DevOnBike.Overfit.svg)](https://www.nuget.org/packages/DevOnBike.Overfit)
[![Build Status](https://img.shields.io/github/actions/workflow/status/DevOnBike/Overfit/ci.yml?branch=main)](https://github.com/DevOnBike/Overfit/actions)
[![License: Dual (AGPLv3 / Commercial)](https://img.shields.io/badge/License-Dual-blue.svg)](LICENSE.md)

**A High-Performance, Zero-Allocation Machine Learning Engine in Pure C#.**

Overfit is a ground-up Deep Learning and Data Preprocessing framework built specifically for modern .NET. It brings the power of neural networks, advanced feature selection, and tabular data pipelines directly to C# **without relying on Python wrappers or heavy external C++ binaries**. 

Designed for maximum CPU inference speed, Overfit embraces aggressive memory management, SIMD potential, and full Native AOT compatibility.

---

## 🚀 Key Features

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

## 🛠️ Quick Start

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

**To purchase a commercial license or discuss enterprise support, please contact:** 👉 **devonbike@gmail.com**

---

## 🤝 Contributing
Contributions are welcome! Whether it's adding new activation functions, optimizing tensor math with `System.Numerics.Vectors`, or improving the documentation.
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
