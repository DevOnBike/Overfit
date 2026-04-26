# Sources/Benchmark

BenchmarkDotNet project for Overfit performance and allocation validation.

## Primary benchmark commands

```bash
dotnet run -c Release --project Sources/Benchmark --filter "*InferenceZeroAllocBenchmarks*"
dotnet run -c Release --project Sources/Benchmark --filter "*OnnxMlpInferenceBenchmarks*"
dotnet run -c Release --project Sources/Benchmark --filter "*OnnxCnnInferenceBenchmarks*"
dotnet run -c Release --project Sources/Benchmark --filter "*LinearKernelBenchmarks*"
```

## Main benchmark classes

### `InferenceZeroAllocBenchmarks`

Compares Overfit against direct manual loops using identical model weights.

Current stable results on Ryzen 9 9950X3D / .NET 10:

| Method | Mean | Allocated |
|---|---:|---:|
| Manual_SingleLinear_TrueZeroAlloc | ~225 ns | 0 B |
| Overfit_SingleLinear_ZeroAlloc | ~227 ns | 0 B |
| Overfit_MultiLayer_ZeroAlloc | ~3.62 us | 0 B |
| Manual_MultiLayer_TrueZeroAlloc | ~3.66 us | 0 B |

This benchmark is the main framework-overhead test. Overfit should stay near the manual baseline.

### `OnnxMlpInferenceBenchmarks`

Compares Overfit and ONNX Runtime for:

```text
Linear(784, 128)
ReLU
Linear(128, 10)
```

Current stable result:

| Method | Mean | Allocated |
|---|---:|---:|
| Overfit_Mlp_ZeroAlloc | ~3.6 us | 0 B |
| OnnxRuntime_Mlp_TrueZeroAlloc | ~4.6-5.0 us | 224 B |

### `OnnxCnnInferenceBenchmarks`

Compares Overfit and ONNX Runtime for:

```text
Conv2D(1 -> 8, 3x3)
ReLU
MaxPool2D(2x2)
GlobalAveragePool2D
Linear(8, 10)
```

Current stable result:

| Method | Mean | Allocated |
|---|---:|---:|
| Overfit_Cnn_ZeroAlloc | ~4.7-5.3 us | 0 B |
| OnnxRuntime_Cnn_TrueZeroAlloc | ~6.0-6.2 us | 224 B |

### `LinearKernelBenchmarks`

Used for kernel selection before changing production code.

Known result:

```text
Small output, e.g. 128 -> 10:
  OutputMajorDot is best.

Large output, e.g. 784 -> 128:
  InputMajorVector4 is best.

Vector8 and fused output10 were tested and rejected.
```

## Benchmark rules

- Use `OperationsPerInvoke` for microsecond/nanosecond operations.
- Treat `MinIterationTime` warnings as a benchmark defect unless intentionally ignored.
- Do not change production kernels based on a single noisy run.
- Always check `Allocated`; zero-allocation inference should report `-` or `0 B`.
- Keep benchmark classes self-contained and deterministic.

## Current rejected experiments

These were tested and should not be reintroduced without new data:

- `Vector8` linear output blocking: caused register pressure / regression.
- Fused `outputSize == 10` input-major kernel: slower than `TensorPrimitives.Dot` path.
- Prepared interface dispatch for Conv/ReLU/Pooling/GAP: did not improve CNN path.
