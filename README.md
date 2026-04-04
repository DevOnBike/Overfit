# Overfit 🚀

[![NuGet version](https://img.shields.io/nuget/v/DevOnBike.Overfit.svg)](https://www.nuget.org/packages/DevOnBike.Overfit)
[![Build Status](https://img.shields.io/github/actions/workflow/status/DevOnBike/Overfit/ci.yml?branch=main)](https://github.com/DevOnBike/Overfit/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**A High-Performance, Zero-Allocation Machine Learning Engine in Pure C#.**

Overfit is a ground-up Deep Learning and Data Preprocessing framework built specifically for modern .NET. It brings the power of neural networks, advanced feature selection, and tabular data pipelines directly to C# **without relying on Python wrappers or heavy external C++ binaries**. 

Designed for maximum CPU inference speed, Overfit embraces aggressive memory management, SIMD potential, and full Native AOT compatibility.

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