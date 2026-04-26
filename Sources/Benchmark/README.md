# Sources/Benchmark

BenchmarkDotNet project for Overfit performance and allocation validation.

## Benchmark policy

- Use central `BenchmarkConfig` for normal benchmark runs.
- Use a separate disassembly config only for JIT/kernel inspection.
- Benchmark classes must not be `sealed`; BenchmarkDotNet requires benchmark classes referenced by runner to be unsealed.
- Inference benchmarks must use `InferenceEngine.Run(...)` and preallocated buffers.
- Avoid `model.Forward(...)`, `AutogradNode`, `TensorStorage` and `ComputationGraph` in inference benchmark methods.
- Use `OperationsPerInvoke` for nanosecond/microsecond operations.
- Treat `MinIterationTime` as a benchmark defect unless explicitly justified.
- Name ONNX methods `PreAllocated` when they still allocate wrapper memory.
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
dotnet run -c Release --project Sources/Benchmark --filter "*MLNetSingleInferenceBenchmark*"
dotnet run -c Release --project Sources/Benchmark --filter "*ConcurrentInferenceBenchmark*"
dotnet run -c Release --project Sources/Benchmark --filter "*TrainingEngineBenchmarks*"
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

### `MLNetSingleInferenceBenchmark`

API-level 3-layer MLP comparison.

| Method | Mean | Allocated |
|---|---:|---:|
| MLNet_PredictionEngine_FreshInput | ~10.1 us | ~4.6 KB |
| MLNet_PredictionEngine_ReusedInput | ~10.1 us | ~4.5 KB |
| OnnxRuntime_PreAllocated | ~11.2 us | 224 B |
| Overfit_InferenceEngine_ZeroAlloc | ~12.1 us | 0 B |

### `BatchScalingBenchmark`

| Batch | Overfit | ONNX Runtime | Result |
|---:|---:|---:|---|
| 1 | ~286 ns, 0 B | ~1.90 us, 224 B | Overfit wins |
| 16 | ~3.0 us, 0 B | ~3.7 us, 224 B | Overfit close / wins in this run |
| 64 | ~14.4 us, 0 B | ~7.2 us, 224 B | ONNX wins |
| 256 | ~47.4 us, 0 B | ~23.4 us, 224 B | ONNX wins |

Next target: `LinearKernels.ForwardBatched(...)`.

### `ConcurrentInferenceBenchmark`

| Method | Mean | Allocated |
|---|---:|---:|
| Overfit_Concurrent_ZeroContention | ~516 ms | 0 B |
| OnnxRuntime_Concurrent | ~1811 ms | ~117 MB |

### `TrainingEngineBenchmarks`

| Method | Mean | Allocated | Notes |
|---|---:|---:|---|
| TrainingEngine_Mlp_TrainBatch | ~468 us | ~26.8 KB | allocations allowed |

## Diagnostic benchmarks

### `ThreadScalingBenchmarks`

Directional only; not zero-allocation claims.

Useful current signals:

| Workload | 1 thread | 8 threads | 16 threads | Notes |
|---|---:|---:|---:|---|
| MatMul MSE backward | ~3.23 ms | ~1.21 ms | ~1.22 ms | scales well to 8 threads |
| ResidualBlock MSE backward | ~7.97 ms | ~2.27 ms | ~2.10 ms | scales well to 16 threads |
| ResidualBlock forward inference-style | ~2.60 ms | ~1.11 ms | ~1.34 ms | best around 8 threads |

### `OpenAiEsStrategyBenchmarks`

Strategy benchmark, not inference. Current cleaned-up path removes standalone `Tell` because `Tell` requires prior `Ask` state.

| Class | Case | Result |
|---|---|---|
| Tiny | P256_N64 | Ask ~1.8 us, AskThenTell ~5-6 us, 0 B |
| Medium | P256_N256 / P1024_N64 / P256_N1024 | Ask and AskThenTell 0 B |
| Large | P1024_N256 / P1024_N1024 | Ask and AskThenTell 0 B after OperationsPerInvoke cleanup |

## Rejected or constrained experiments

- `Vector8` linear output blocking caused register pressure/regression.
- Fused `outputSize == 10` input-major kernel was slower than the output-major `TensorPrimitives.Dot` path.
- Prepared interface dispatch for Conv/ReLU/Pooling/GAP did not improve the CNN hot path.
- Old `model.Forward(...)` / `AutogradNode` inference benchmarks are invalid for zero-allocation claims.
