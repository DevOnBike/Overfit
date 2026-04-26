# Sources/Main

This project contains the core Overfit runtime.

## Responsibility split

```text
Autograd/
  ComputationGraph and AutogradNode.
  Current training runtime. Planned cleanup: graph operation facade + ownership metadata.

DeepLearning/
  User-facing model/layer API.
  Layers own shape, parameters, save/load and train/eval state.

Kernels/
  Low-level hot math.
  SIMD and specialized inference loops live here, not in layers.

Ops/
  Current TensorMath and graph-aware training operations.
  Long-term direction: graph-aware operations move to ComputationGraph.*; TensorMath/Kernels become pure math.

Inference/
  InferenceEngine facade and backend abstraction.

Training/
  TrainingEngine facade and training backend/loss/optimizer abstractions.

Evolutionary/
  Gradient-free strategies, population storage and fitness evaluation.
```

## Inference design rule

The inference path should avoid graph construction and avoid managed allocations after preparation.

Preferred call stack:

```text
InferenceEngine.Run(...)
  -> Sequential.ForwardInference(...)
    -> layer.ForwardInference(...)
      -> Kernels.*(...)
```

Prepared path:

```text
Sequential.PrepareInference(...)
  -> layer.PrepareInference()
  -> preallocated intermediate buffers
  -> prepared dispatch for selected modules
```

Current prepared module:

```text
LinearLayer implements IPreparedInferenceModule
```

Do not add `IPreparedInferenceModule` to a layer unless a benchmark shows a win. It was tested for Conv/ReLU/Pooling/GAP and did not improve the hot path.

## Kernel ownership

### Keep in layers

- constructor validation;
- parameter initialization;
- `Weights`, `Bias`, `Kernels` nodes;
- shape metadata;
- `Train()`, `Eval()`, `PrepareInference()`;
- save/load;
- cache invalidation.

### Move to kernels

- SIMD loops;
- pooling loops;
- activation loops;
- convolution loops;
- linear algebra hot loops;
- batch-aware span loops.

## Current kernel files

```text
Kernels/LinearKernels.cs
Kernels/Conv2DKernels.cs
Kernels/ActivationKernels.cs
Kernels/PoolingKernels.cs
```

## Current inference baseline

The current `InferenceEngine.Run(...)` path is verified as zero-allocation in the benchmark suite.

Representative results on AMD Ryzen 9 9950X3D / .NET 10:

| Workload | Overfit | Allocation |
|---|---:|---:|
| Linear(784,10) single inference | ~250-300 ns | 0 B |
| Linear(4096,10) | ~1.08 us | 0 B |
| MLP 784->128->10 | ~3.7 us | 0 B |
| MLP 784->256->128->10 | ~10-12 us | 0 B |
| Small CNN | ~5-6.5 us | 0 B |

## Next performance target

`BatchScalingBenchmark` shows Overfit winning batch 1/16 while ONNX Runtime wins batch 64/256.

Current reason:

```text
Overfit: repeated sample inference
ONNX: likely batched GEMM-style execution
```

Next kernel target:

```text
LinearKernels.ForwardBatched(
    ReadOnlySpan<float> inputBatch,
    ReadOnlySpan<float> weights,
    ReadOnlySpan<float> bias,
    Span<float> outputBatch,
    int batchSize,
    int inputSize,
    int outputSize)
```

## Training path

Training remains graph-based:

```text
Layer.Forward(graph, input)
  -> TensorMath/Ops today
  -> ComputationGraph
  -> graph.Backward(...)
  -> Optimizer.Step()
```

Training code should prioritize correctness and explicit graph behavior. Inference kernels can be reused by training primitives where useful, but inference must not depend on autograd state.

Current `TrainingEngineBenchmarks` baseline:

```text
TrainingEngine_Mlp_TrainBatch: ~468 us, ~26.8 KB allocated
```

Allocations are allowed in training benchmarks. They are tracked as performance trend data.

## Planned graph architecture cleanup

Target model:

```text
TrainingEngine = workflow facade
ComputationGraph = autograd brain / operation facade
Parameter = long-lived trainable model state
AutogradNode = graph-visible value handle
Kernels = pure Span-based math
InferenceEngine = separate zero-allocation inference workflow
```

Near-term order:

1. Add `ComputationGraph.*` operation facade wrappers.
2. Add `AutogradNodeOwnership` metadata.
3. Add graph factory methods for temporary/external/parameter-view nodes.
4. Introduce `Parameter` as a separate type.
5. Migrate `LinearLayer` first.
6. Migrate optimizers to `IEnumerable<Parameter>`.
7. Clean up graph reset/disposal by ownership.
