# Overfit documentation

This directory contains scenario guides, benchmark notes and architecture documentation for Overfit.

## Current focus

The current branch focuses on zero-allocation CPU inference and clean separation of responsibilities:

```text
DeepLearning layers: public API, shape, parameters, save/load
Kernels: hot math and SIMD-specialized inference loops
TensorMath/Ops: autograd-aware training math
Inference: simple prepared inference facade
Training: simple training facade and backend abstraction
```

## Recommended reading order

1. `../README.md` — project overview and current benchmark snapshot.
2. `InferenceBenchmarkSummary.md` — detailed inference benchmark summary.
3. `scenarios/` — role-specific usage guides.
4. `../ROADMAP.md` — planned work and priorities.

## Benchmark policy

Performance numbers should be treated as hardware-specific. Keep benchmark claims tied to:

- CPU model;
- OS version;
- .NET SDK/runtime;
- BenchmarkDotNet version;
- benchmark class and filter;
- allocation result.

Current benchmark commands:

```bash
dotnet run -c Release --project Sources/Benchmark --filter "*InferenceZeroAllocBenchmarks*"
dotnet run -c Release --project Sources/Benchmark --filter "*OnnxMlpInferenceBenchmarks*"
dotnet run -c Release --project Sources/Benchmark --filter "*OnnxCnnInferenceBenchmarks*"
```

## Current verified inference results

Machine:

```text
AMD Ryzen 9 9950X3D
Windows 11 25H2
.NET 10.0.7
BenchmarkDotNet 0.15.8
```

| Benchmark | Overfit | Comparison | Allocation |
|---|---:|---:|---:|
| MLP 784->128->10 | ~3.6 us | faster than ONNX Runtime | 0 B |
| CNN 1x28x28 -> 10 | ~4.7-5.3 us | faster than ONNX Runtime | 0 B |
| Single Linear 784->10 | ~227 ns | effectively equal to manual baseline | 0 B |
| MLP manual baseline | ~3.62 us | effectively equal to manual baseline | 0 B |

## Documentation rules

- Prefer explicit code samples over marketing claims.
- Keep zero-allocation claims scoped to inference hot paths.
- Do not describe training as zero-allocation unless a dedicated test proves that exact path.
- Mention ONNX Runtime comparisons only with the exact benchmark context.
