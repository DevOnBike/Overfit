# Overfit documentation

This directory contains scenario guides, benchmark notes and architecture documentation for Overfit.

## Current focus

The current branch focuses on predictable CPU inference, ONNX import and cleaner separation of responsibilities:

```text
DeepLearning layers: public API, shape, parameters, save/load, train/eval state
Kernels: hot math and SIMD-specialized inference loops
Ops/TensorMath: current graph-aware training math, pending graph facade cleanup
Inference: prepared zero-allocation inference facade
Training: training facade and backend/loss/optimizer abstractions
Evolutionary: gradient-free strategies and population evaluation
ONNX: PyTorch-exported model import into Sequential for inference
```

## Recommended reading order

1. `../README.md` — project overview and current benchmark snapshot.
2. `InferenceBenchmarkSummary.md` — detailed inference benchmark summary.
3. `../ONNX_IMPLEMENTATION_PLAN.md` — ONNX import MVP scope and implementation notes.
4. `OverfitArchitectureRefactorPlan.md` — planned autograd/graph ownership cleanup.
5. `TrainingEngineFacade.md` — current training facade design.
6. `scenarios/` — role-specific usage guides.
7. `../ROADMAP.md` — planned work and priorities.

## Benchmark policy

Performance numbers are hardware-specific. Keep benchmark claims tied to:

- CPU model;
- OS version;
- .NET SDK/runtime;
- BenchmarkDotNet version;
- benchmark class and filter;
- allocation result;
- exact model shape.

Current verified machine:

```text
AMD Ryzen 9 9950X3D
Windows 11 25H2
.NET 10.0.7
BenchmarkDotNet 0.15.8
```

## Current verified inference results

| Benchmark | Overfit | Comparison | Allocation |
|---|---:|---:|---:|
| Linear(784,10) single inference | ~250-300 ns | ONNX ~2.1-2.3 us | Overfit 0 B |
| Linear(64/784/4096,10) scaling | ~80 ns / ~210 ns / ~1.08 us | ONNX slower at all three sizes | Overfit 0 B |
| MLP 784->128->10 | ~3.7 us | ONNX ~5.2 us | Overfit 0 B |
| MLP 784->256->128->10 | ~10-12 us | roughly tied with ONNX | Overfit 0 B |
| Small CNN | ~5-6.5 us | roughly tied with ONNX | Overfit 0 B |
| Imported PyTorch ONNX MNIST CNN | ~7.5 us | ONNX ~7.5 us, PyTorch eager ~27.3 us | Overfit 0 B |
| Concurrent inference | ~516 ms | ONNX ~1811 ms | Overfit 0 B |

## ONNX import status

The ONNX MVP loads a focused PyTorch-exported CNN into `Sequential` and runs it through `InferenceEngine.Run(...)`.

Supported MVP operators:

```text
Conv
Relu
MaxPool
Reshape / Flatten
Gemm
```

Current imported ONNX benchmark:

| Runtime | Result | Notes |
|---|---:|---|
| Overfit imported ONNX | ~7.5 us/op | BenchmarkDotNet, 0 B/op |
| ONNX Runtime preallocated | ~7.5 us/op | BenchmarkDotNet, 224 B/op |
| PyTorch eager CPU | ~27.3 us/op | Python reference script, 1 thread |

## Recommended benchmark commands

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

## Documentation rules

- Prefer explicit code samples over marketing claims.
- Keep zero-allocation claims scoped to exact measured inference hot paths.
- Do not describe training as zero-allocation unless a dedicated test proves that exact path.
- Mention ONNX Runtime, ML.NET and PyTorch comparisons only with exact benchmark context and allocation result.
- Use ranges when jitter or multimodal distributions are visible.
