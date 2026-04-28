# Sources/Benchmark

BenchmarkDotNet project for Overfit performance and allocation validation.

## Benchmark policy

- Use central `BenchmarkConfig` for normal benchmark runs.
- Use a separate disassembly config only for JIT/kernel inspection.
- Benchmark classes must not be `sealed`; BenchmarkDotNet requires benchmark classes referenced by the runner to be unsealed.
- Inference benchmarks must use `InferenceEngine.Run(...)` and preallocated buffers.
- Avoid `model.Forward(...)`, `AutogradNode`, `TensorStorage` and `ComputationGraph` in inference benchmark methods.
- Avoid `Predict(...).ToArray()` in zero-allocation inference benchmarks.
- Use `OperationsPerInvoke` for nanosecond/microsecond operations.
- Treat `MinIterationTime` as a benchmark defect unless explicitly justified.
- Name ONNX Runtime methods `PreAllocated` when they still allocate wrapper memory.
- Training/graph benchmarks are trend benchmarks; allocations are allowed unless the benchmark explicitly claims zero allocation.

## Primary benchmark commands

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

PyTorch reference script:

```bash
python Tests/benchmark_pytorch_mnist_cnn.py --fixture-dir Tests/test_fixtures --threads 1
```

## Current stable results

Machine:

```text
AMD Ryzen 9 9950X3D
Windows 11 25H2
.NET 10.0.7
BenchmarkDotNet 0.15.8
```

### `SingleInferenceBenchmark`

Linear(784,10), batch 1.

| Method | Mean | Allocated |
|---|---:|---:|
| Overfit_InferenceEngine_ZeroAlloc | ~252 ns | 0 B |
| OnnxRuntime_PreAllocated | ~2.14 us | 224 B |
| OnnxRuntime_StandardNamedValue | ~3.37 us | 952 B |

### `OnnxLinearInferenceBenchmarks`

Linear(784,10), generated from same Overfit weights.

| Method | Mean | Allocated |
|---|---:|---:|
| Overfit_Linear_InferenceEngine_ZeroAlloc | ~272 ns | 0 B |
| OnnxRuntime_Linear_PreAllocated | ~2.18 us | 224 B |

### `ScalingBenchmark`

| Method | Mean | Allocated |
|---|---:|---:|
| Overfit_64 | ~80 ns | 0 B |
| Overfit_784 | ~210 ns | 0 B |
| Overfit_4096 | ~1.08 us | 0 B |
| Onnx_64 | ~1.39 us | 224 B |
| Onnx_784 | ~1.87 us | 224 B |
| Onnx_4096 | ~3.74 us | 224 B |

### `ThroughputBenchmark`

Repeated Linear(784,10) inference.

| Method | Mean | Allocated |
|---|---:|---:|
| Overfit_10k_InferenceEngine_ZeroAlloc | ~253 ns/op | 0 B |
| OnnxRuntime_10k_PreAllocated | ~1.87 us/op | 224 B |

### `TailLatencyBenchmark`

Linear(784,10) latency profile.

| Method | Mean | Allocated |
|---|---:|---:|
| Overfit_LatencyProfile | ~299 ns | 0 B |
| OnnxRuntime_LatencyProfile | ~2.26 us | 224 B |

### `InferenceZeroAllocBenchmarks`

Manual baseline comparison.

| Method | Mean | Allocated |
|---|---:|---:|
| Manual_SingleLinear_TrueZeroAlloc | ~225-228 ns | 0 B |
| Overfit_SingleLinear_ZeroAlloc | ~228-232 ns | 0 B |
| Manual_MultiLayer_TrueZeroAlloc | ~3.61 us | 0 B |
| Overfit_MultiLayer_ZeroAlloc | ~3.61 us | 0 B |

### `OnnxMlpInferenceBenchmarks`

Model:

```text
Linear(784,128)
ReLU
Linear(128,10)
```

| Method | Mean | Allocated |
|---|---:|---:|
| Overfit_Mlp_ZeroAlloc | ~3.7 us | 0 B |
| OnnxRuntime_Mlp_PreAllocated | ~5.2 us | 224 B |

### `MultiLayerInferenceBenchmark`

Model:

```text
Linear(784,256)
ReLU
Linear(256,128)
ReLU
Linear(128,10)
```

| Method | Mean | Allocated |
|---|---:|---:|
| Overfit_3Layer_InferenceEngine_ZeroAlloc | ~10-12 us | 0 B |
| OnnxRuntime_3Layer_PreAllocated | ~10-11 us | 224 B |

### `OnnxCnnInferenceBenchmarks`

Model:

```text
Conv2D(1 -> 8, 3x3)
ReLU
MaxPool2D(2x2)
GlobalAveragePool2D
Linear(8,10)
```

| Method | Mean | Allocated |
|---|---:|---:|
| Overfit_Cnn_InferenceEngine_ZeroAlloc | ~5-6.5 us | 0 B |
| OnnxRuntime_Cnn_PreAllocated | ~6-7.7 us | 224 B |

### `ImportedOnnxMnistCnnBenchmark`

Pipeline:

```text
PyTorch ONNX fixture
-> OnnxImporter.Load(...)
-> Sequential
-> InferenceEngine.Run(...)
```

| Runtime | Mean | Allocated | Notes |
|---|---:|---:|---|
| Overfit_ImportedOnnxMnistCnn_ZeroAlloc | ~7.5 us | 0 B | BenchmarkDotNet |
| OnnxRuntime_MnistCnn_PreAllocated | ~7.5 us | 224 B | BenchmarkDotNet |
| PyTorch eager CPU | ~27.3 us | n/a | Python reference script, 1 thread |

### `MLNetSingleInferenceBenchmark`

API-level 3-layer MLP comparison.

| Method | Mean | Allocated |
|---|---:|---:|
| MLNet_PredictionEngine_FreshInput | ~10.1 us | ~4568 B |
| MLNet_PredictionEngine_ReusedInput | ~10.1 us | ~4544 B |
| OnnxRuntime_PreAllocated | ~11.2 us | 224 B |
| Overfit_InferenceEngine_ZeroAlloc | ~12.1 us | 0 B |

### `ConcurrentInferenceBenchmark`

| Method | Mean | Allocated |
|---|---:|---:|
| Overfit_Concurrent_ZeroContention | ~516 ms | 0 B |
| OnnxRuntime_Concurrent | ~1811 ms | ~117 MB |

### `BatchScalingBenchmark`

| Batch | Overfit | ONNX Runtime | Allocated |
|---:|---:|---:|---:|
| 1 | ~286 ns | ~1.9 us | Overfit 0 B, ONNX 224 B |
| 16 | ~3.0 us | ~3.7 us | Overfit 0 B, ONNX 224 B |
| 64 | ~14.4 us | ~7.2 us | Overfit 0 B, ONNX 224 B |
| 256 | ~47.4 us | ~23.4 us | Overfit 0 B, ONNX 224 B |

### `TrainingEngineBenchmarks`

| Method | Mean | Allocated |
|---|---:|---:|
| TrainingEngine_Mlp_TrainBatch | ~468 us | ~26.8 KB |

Training allocations are allowed.

## Interpretation

- Overfit is strongest on small single-sample inference and concurrent zero-contention inference.
- Overfit is roughly tied with ONNX Runtime on larger small-model CNN/MLP cases while keeping 0 B/op.
- ONNX Runtime wins larger batch sizes because the current Overfit batch path does not yet use batched GEMM.
- ML.NET can be competitive or slightly faster in API-level MLP runs, but allocates about 4.5 KB/op.
