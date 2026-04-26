# Overfit zero-allocation inference summary

## Environment

Benchmark machine:

- CPU: AMD Ryzen 9 9950X3D
- Cores: 16 physical / 32 logical
- OS: Windows 11 25H2
- Runtime: .NET 10.0.7
- JIT: RyuJIT x86-64-v4
- BenchmarkDotNet: 0.15.8

## Current inference architecture

Overfit has two separate execution paths.

### Training path

Used for learning/backpropagation.

```text
AutogradNode
ComputationGraph
TensorMath/Ops
Optimizer
```

This path may allocate graph/tensor state. Training benchmarks are performance trend benchmarks, not zero-allocation gates.

### Inference path

Used for production prediction.

```text
InferenceEngine.Run(...)
Sequential.ForwardInference(...)
layer.ForwardInference(...)
Kernels.*(...)
```

This path is designed to allocate `0 B/op` after engine/model preparation.

## Confirmed zero-allocation inference components

The following paths are verified through current benchmark coverage:

```text
InferenceEngine.Run
Sequential.ForwardInference
LinearLayer.ForwardInference
ReluActivation.ForwardInference
ConvLayer.ForwardInference
MaxPool2DLayer.ForwardInference
GlobalAveragePool2DLayer.ForwardInference
```

## Benchmark results

### Single Linear(784,10)

| Benchmark | Overfit | ONNX Runtime | Allocations |
|---|---:|---:|---:|
| `SingleInferenceBenchmark` | ~252 ns | ~2.14 us | Overfit 0 B, ONNX 224 B |
| `OnnxLinearInferenceBenchmarks` | ~272 ns | ~2.18 us | Overfit 0 B, ONNX 224 B |
| `TailLatencyBenchmark` | ~299 ns | ~2.26 us | Overfit 0 B, ONNX 224 B |
| `ThroughputBenchmark` | ~253 ns/op | ~1.87 us/op | Overfit 0 B, ONNX 224 B |

Conclusion:

```text
Overfit is roughly 7-9x faster than the preallocated ONNX Runtime path for repeated single-sample Linear(784,10) inference and remains zero-allocation.
```

### Scaling by input size

| Model | Overfit | ONNX Runtime | Allocation |
|---|---:|---:|---:|
| Linear(64,10) | ~80 ns | ~1.39 us | Overfit 0 B, ONNX 224 B |
| Linear(784,10) | ~210 ns | ~1.87 us | Overfit 0 B, ONNX 224 B |
| Linear(4096,10) | ~1.08 us | ~3.74 us | Overfit 0 B, ONNX 224 B |

Conclusion:

```text
Overfit keeps the Linear inference path allocation-free across tested input sizes. ONNX Runtime fixed call overhead dominates at small model sizes.
```

### MLP: 784 -> 128 -> 10

| Runtime | Mean | Allocated |
|---|---:|---:|
| Overfit | ~3.7 us | 0 B |
| ONNX Runtime preallocated | ~5.2 us | 224 B |

Conclusion:

```text
Overfit is faster for this small MLP workload and keeps the hot path at 0 B/op.
```

### MLP: 784 -> 256 -> 128 -> 10

| Runtime | Mean | Allocated |
|---|---:|---:|
| Overfit | ~10-12 us | 0 B |
| ONNX Runtime preallocated | ~10-11 us | 224 B |

Conclusion:

```text
Overfit roughly matches ONNX Runtime for this larger 3-layer MLP while staying allocation-free.
```

### CNN inference

Model:

```text
Conv2D(1 -> 8, 3x3)
ReLU
MaxPool2D(2x2)
GlobalAveragePool2D
Linear(8, 10)
```

| Runtime | Mean | Allocated |
|---|---:|---:|
| Overfit | ~5-6.5 us | 0 B |
| ONNX Runtime preallocated | ~6-7.7 us | 224 B |

Conclusion:

```text
Overfit is roughly tied with ONNX Runtime on the small CNN benchmark and remains zero-allocation.
```

### ML.NET API-level benchmark

Model: 3-layer MLP, API-level single inference.

| Runtime | Mean | Allocated |
|---|---:|---:|
| ML.NET PredictionEngine, fresh input | ~10.1 us | ~4.6 KB |
| ML.NET PredictionEngine, reused input | ~10.1 us | ~4.5 KB |
| ONNX Runtime preallocated | ~11.2 us | 224 B |
| Overfit InferenceEngine | ~12.1 us | 0 B |

Conclusion:

```text
ML.NET is slightly faster in this API-level run but allocates several KB/op. Overfit is the only zero-allocation path.
```

### Batch scaling

| Batch | Overfit | ONNX Runtime | Result |
|---:|---:|---:|---|
| 1 | ~286 ns, 0 B | ~1.90 us, 224 B | Overfit wins |
| 16 | ~3.0 us, 0 B | ~3.7 us, 224 B | Overfit close / wins in this run |
| 64 | ~14.4 us, 0 B | ~7.2 us, 224 B | ONNX wins |
| 256 | ~47.4 us, 0 B | ~23.4 us, 224 B | ONNX wins |

Conclusion:

```text
Overfit's current batch path remains allocation-free but processes batches as repeated sample inference. ONNX Runtime wins at larger batch sizes because it uses batched GEMM-style execution. Next target: LinearKernels.ForwardBatched(...).
```

### Concurrent inference

| Runtime | Mean | Allocated |
|---|---:|---:|
| Overfit | ~516 ms | 0 B |
| ONNX Runtime | ~1811 ms | ~117 MB |

Conclusion:

```text
Overfit performs well in a zero-contention concurrent single-sample inference scenario where each worker owns its own model, engine and buffers.
```

## Benchmark interpretation rules

- `Allocated = -` or `0 B` is required for zero-allocation inference claims.
- ONNX methods should be named `PreAllocated`, not `TrueZeroAlloc`, when allocations remain.
- `MinIterationTime` warnings should be fixed with `OperationsPerInvoke` before documenting numbers.
- Training and graph benchmarks can allocate and should be documented separately.
- Thread-scaling benchmarks are directional unless explicitly stabilized.

## Current recommendation

Use this public API shape for production inference:

```csharp
using var engine = InferenceEngine.FromSequential(
    model,
    inputSize: 784,
    outputSize: 10,
    new InferenceEngineOptions
    {
        WarmupIterations = 16,
        MaxIntermediateElements = 64 * 1024
    });

engine.Run(input, output);
```

For convenience:

```csharp
ReadOnlySpan<float> prediction = engine.Predict(input);
```

The prediction span is internal and overwritten on the next call.
