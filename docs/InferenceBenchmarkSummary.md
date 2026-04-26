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

Overfit now has two separate execution paths:

### Training path

Used for learning/backpropagation.

```text
AutogradNode
ComputationGraph
TensorMath
Optimizer
```

This path is allowed to allocate internal graph/tensor state.

### Inference path

Used for production prediction.

```text
Sequential.ForwardInference(...)
Span<float>
preallocated workspace
layer-specific inference kernels
```

This path is designed to allocate `0 B/op` after engine/model preparation.

## Confirmed zero-allocation components

The following inference paths have been verified as zero-allocation:

```text
LinearLayer.ForwardInference
ReluActivation.ForwardInference
BatchNorm1D.ForwardInference
ConvLayer.ForwardInference
MaxPool2DLayer.ForwardInference
GlobalAveragePool2DLayer.ForwardInference
Sequential.ForwardInference
```

## MLP inference benchmark

Model:

```text
Linear(784, 128)
ReLU
Linear(128, 10)
```

Result:

| Runtime | Mean | Allocated |
|---|---:|---:|
| Overfit | 3.607 us | 0 B |
| ONNX Runtime | 4.760 us | 224 B |

Conclusion:

```text
Overfit is faster than ONNX Runtime for this small MLP workload.
Overfit also keeps the hot path at 0 B/op.
```

## CNN inference benchmark

Model:

```text
Conv2D(1 -> 8, 3x3)
ReLU
MaxPool2D(2x2)
GlobalAveragePool2D
Linear(8, 10)
```

Result:

| Runtime | Mean | Allocated |
|---|---:|---:|
| Manual CNN fast path | 5.147 us | 0 B |
| Overfit | 5.534–5.592 us | 0 B |
| ONNX Runtime | 6.049 us | 224 B |

Conclusion:

```text
Overfit CNN inference is close to manual specialized code.
Overfit is faster than ONNX Runtime in this small CNN workload.
ONNX Runtime still allocates wrapper memory in the measured path.
```

## Key optimizations added

### Linear inference

For small output sizes, Overfit uses output-major dot-product:

```text
Linear(784, 10)
Linear(128, 10)
```

For larger output sizes, Overfit uses input-major SIMD Vector4 kernel:

```text
Linear(784, 128)
```

This avoids the slow scalar input-major path and avoids the register pressure regression observed with Vector8.

### Conv inference

Specialized fast path added for:

```text
inChannels = 1
kernel = 3x3
```

This targets the MNIST-style CNN case and uses SIMD over output width.

## Current recommendation

Use this public API shape for inference:

```csharp
using var engine = InferenceEngine.FromSequential(model);

ReadOnlySpan<float> prediction = engine.Predict(input);
```

For explicit control:

```csharp
using var engine = InferenceEngine.FromSequential(
    model,
    inputSize: 784,
    outputSize: 10,
    new InferenceEngineOptions
    {
        WarmupIterations = 8,
        MaxIntermediateElements = 64 * 1024
    });

engine.Run(input, output);
```

## Design direction

Inference should stay separated from training:

```text
Training:
    flexible graph/autograd system

Inference:
    prepared immutable-ish engine
    preallocated buffers
    no AutogradNode
    no graph construction
    no temporary storage allocation
```

The next architectural step is a small `InferenceEngine` facade that hides:

```text
model.Eval()
model.PrepareInference(...)
workspace sizing
output buffer ownership
warmup
input/output validation
```

while still allowing future backends:

```text
SequentialInferenceBackend
OnnxInferenceBackend
CompiledInferenceBackend
GpuInferenceBackend
CustomUserBackend
```
