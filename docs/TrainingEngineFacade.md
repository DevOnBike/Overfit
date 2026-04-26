# TrainingEngine facade

The training facade hides common training boilerplate while keeping the training system extensible.

## Purpose

The training path is intentionally different from the inference path.

Inference:

```text
InferenceEngine
preallocated buffers
Span-based hot path
zero allocations after preparation
```

Training:

```text
TrainingEngine
ComputationGraph
AutogradNode
optimizer
loss
backward
step
```

Training is not promised to be zero-allocation. It owns graph/autograd state and may allocate depending on active operators.

## Current baseline

Current `TrainingEngineBenchmarks` result on AMD Ryzen 9 9950X3D / .NET 10:

| Benchmark | Mean | Allocated | Purpose |
|---|---:|---:|---|
| `TrainingEngine_Mlp_TrainBatch` | ~468 us | ~26.8 KB | performance trend tracking |

This is not a zero-allocation gate. Use it to detect major regressions after graph, optimizer or layer changes.

## Developer-facing API

```csharp
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Training;

const int batchSize = 64;
const int inputSize = 784;
const int classCount = 10;

using var model = new Sequential(
    new LinearLayer(inputSize, 128),
    new ReluActivation(),
    new LinearLayer(128, classCount));

using var adam = new Adam(
    model.Parameters(),
    learningRate: 0.001f);

var optimizer = new DelegateTrainingOptimizer(
    zeroGrad: adam.ZeroGrad,
    step: adam.Step);

var loss = new DelegateTrainingLoss(
    forward: (graph, prediction, target) =>
        TensorMath.SoftmaxCrossEntropy(graph, prediction, target),
    backward: (graph, lossNode) => graph.Backward(lossNode));

using var trainer = TrainingEngine.FromBackend(
    new SequentialTrainingBackend(
        model,
        optimizer,
        loss,
        batchSize,
        inputSize,
        classCount,
        new TrainingEngineOptions
        {
            ResetGraphAfterStep = true,
            ValidateFiniteInput = false,
            ValidateFiniteTarget = false
        }));

TrainingStepResult result = trainer.TrainBatch(batchInput, batchTarget);
Console.WriteLine(result.Loss);
```

## Extension points

```text
ITrainingOptimizer
ITrainingLoss
ITrainingBackend
TrainingEngine
```

Potential future backends:

```text
CompiledTrainingBackend
GradientAccumulationBackend
MixedPrecisionTrainingBackend
DistributedTrainingBackend
CustomReinforcementLearningBackend
```

## Architecture direction

`TrainingEngine` should remain the workflow facade. It should not absorb the responsibilities of `ComputationGraph`.

Target split:

```text
TrainingEngine: batch workflow
ComputationGraph: autograd operation runtime
Parameter: long-lived model state
Optimizer: parameter update
Kernels: pure math
```
