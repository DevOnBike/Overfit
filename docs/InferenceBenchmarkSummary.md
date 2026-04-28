# Inference benchmark summary

This document summarizes the current verified inference benchmark results for Overfit.

## Environment

```text
AMD Ryzen 9 9950X3D
Windows 11 25H2
.NET 10.0.7
BenchmarkDotNet 0.15.8
```

## Policy

Only benchmarks that use `InferenceEngine.Run(...)` with preallocated input/output buffers are considered valid zero-allocation inference claims.

Invalid for zero-allocation inference claims:

```text
model.Forward(...)
AutogradNode
ComputationGraph
TensorStorage created inside benchmark methods
Predict(...).ToArray()
new float[] inside benchmark methods
```

## Linear inference

| Benchmark | Overfit | ONNX Runtime | Allocation |
|---|---:|---:|---:|
| SingleInferenceBenchmark Linear(784,10) | ~252 ns | ~2.14 us | Overfit 0 B, ONNX 224 B |
| OnnxLinearInferenceBenchmarks Linear(784,10) | ~272 ns | ~2.18 us | Overfit 0 B, ONNX 224 B |
| ThroughputBenchmark Linear(784,10) | ~253 ns/op | ~1.87 us/op | Overfit 0 B, ONNX 224 B |
| TailLatencyBenchmark Linear(784,10) | ~299 ns | ~2.26 us | Overfit 0 B, ONNX 224 B |

Interpretation:

Overfit is significantly faster than ONNX Runtime for small single-sample linear inference and maintains 0 B/op.

## Scaling benchmark

| Model | Overfit | ONNX Runtime | Allocation |
|---|---:|---:|---:|
| Linear(64,10) | ~80 ns | ~1.39 us | Overfit 0 B, ONNX 224 B |
| Linear(784,10) | ~210 ns | ~1.87 us | Overfit 0 B, ONNX 224 B |
| Linear(4096,10) | ~1.08 us | ~3.74 us | Overfit 0 B, ONNX 224 B |

Interpretation:

The prepared Overfit single-layer path scales cleanly across input sizes while remaining allocation-free.

## MLP inference

| Benchmark | Model | Overfit | Comparison | Allocation |
|---|---|---:|---:|---:|
| OnnxMlpInferenceBenchmarks | 784->128->10 | ~3.7 us | ONNX ~5.2 us | Overfit 0 B, ONNX 224 B |
| MultiLayerInferenceBenchmark | 784->256->128->10 | ~10-12 us | ONNX ~10-11 us | Overfit 0 B, ONNX 224 B |
| MLNetSingleInferenceBenchmark | API-level 3-layer MLP | ~12.1 us | ML.NET ~10.1 us, ONNX ~11.2 us | Overfit 0 B, ML.NET ~4.5 KB, ONNX 224 B |

Interpretation:

Overfit is competitive with ONNX Runtime on larger MLPs while remaining allocation-free. ML.NET can be slightly faster in the API-level benchmark, but allocates about 4.5 KB/op.

## CNN inference

| Benchmark | Overfit | ONNX Runtime | Allocation |
|---|---:|---:|---:|
| Small CNN | ~5-6.5 us | ~6-7.7 us | Overfit 0 B, ONNX 224 B |
| Imported PyTorch ONNX MNIST CNN | ~7.5 us | ~7.5 us | Overfit 0 B, ONNX 224 B |

Interpretation:

Overfit roughly matches ONNX Runtime on small CNN workloads while keeping the inference path allocation-free.

## Imported ONNX benchmark

The imported ONNX benchmark validates the full bridge:

```text
PyTorch export -> ONNX file -> OnnxImporter.Load(...) -> Sequential -> InferenceEngine.Run(...)
```

| Runtime | Result | Notes |
|---|---:|---|
| Overfit imported ONNX | ~7.5 us/op | BenchmarkDotNet, 0 B/op |
| ONNX Runtime preallocated | ~7.5 us/op | BenchmarkDotNet, 224 B/op |
| PyTorch eager CPU | ~27.3 us/op | Python reference script, 1 thread |

The PyTorch result is an external reference script, not a BenchmarkDotNet result. It is used to position deployment overhead, not as a strict apples-to-apples benchmark.

## Concurrent inference

| Benchmark | Overfit | ONNX Runtime |
|---|---:|---:|
| Concurrent single-sample inference | ~516 ms, 0 B | ~1811 ms, ~117 MB |

Interpretation:

Overfit performs well in a zero-contention concurrent scenario where each worker owns its model, engine and buffers.

## Batch scaling

| Batch | Overfit | ONNX Runtime | Result |
|---:|---:|---:|---|
| 1 | ~286 ns, 0 B | ~1.9 us, 224 B | Overfit wins |
| 16 | ~3.0 us, 0 B | ~3.7 us, 224 B | Overfit close / wins in this run |
| 64 | ~14.4 us, 0 B | ~7.2 us, 224 B | ONNX wins |
| 256 | ~47.4 us, 0 B | ~23.4 us, 224 B | ONNX wins |

Interpretation:

The current Overfit batch path is allocation-free but not yet a batched GEMM path. ONNX wins larger batches through optimized batched execution. Next target: batched linear kernels.

## Training benchmark

| Benchmark | Mean | Allocations | Notes |
|---|---:|---:|---|
| TrainingEngine MLP TrainBatch | ~468 us | ~26.8 KB | Trend benchmark only |

Training allocations are currently allowed.

## Recommended public claim

Use:

```text
Overfit provides predictable zero-allocation CPU inference for .NET workloads and can now import a focused PyTorch ONNX CNN into the same inference path.
```

Avoid:

```text
Overfit is always faster than ONNX Runtime or ML.NET.
```
