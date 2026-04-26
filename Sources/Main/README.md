# Sources/Main

This project contains the core Overfit runtime.

## Responsibility split

```text
Autograd/
  ComputationGraph and AutogradNode.

DeepLearning/
  User-facing model/layer API.
  Layers own shape, parameters, save/load and train/eval state.

Kernels/
  Low-level hot math.
  SIMD and specialized inference loops live here, not in layers.

Ops/
  TensorMath and graph-aware training operations.

Inference/
  InferenceEngine facade and backend abstraction.

Training/
  TrainingEngine facade and training backend/loss/optimizer abstractions.
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

## Training path

Training remains graph-based:

```text
Layer.Forward(graph, input)
  -> TensorMath.*(...)
  -> ComputationGraph
  -> graph.Backward(...)
  -> Optimizer.Step()
```

Training code should prioritize correctness and explicit graph behavior. Inference kernels can be reused by training primitives where useful, but inference must not depend on autograd state.
