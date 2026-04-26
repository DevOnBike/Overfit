# TrainingEngine façade

This patch adds a small training façade that hides common training boilerplate while keeping the training system extensible.

## Purpose

The training path is intentionally different from the inference path.

Inference:

```text
InferenceEngine
preallocated buffers
Span<float>
zero allocations on hot path
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

Training is not promised to be zero-allocation. It owns graph/autograd state and may allocate depending on the active operators.

## Added files

```text
Sources/Main/Training/ITrainingOptimizer.cs
Sources/Main/Training/DelegateTrainingOptimizer.cs
Sources/Main/Training/ITrainingLoss.cs
Sources/Main/Training/DelegateTrainingLoss.cs
Sources/Main/Training/TrainingEngineOptions.cs
Sources/Main/Training/TrainingStepResult.cs
Sources/Main/Training/ITrainingBackend.cs
Sources/Main/Training/TrainingEngine.cs
Sources/Main/Training/SequentialTrainingBackend.cs
```

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

    backward: (graph, lossNode) =>
        graph.Backward(lossNode));

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
```

Training loop:

```csharp
TrainingStepResult result = trainer.TrainBatch(
    batchInput,
    batchTarget);

Console.WriteLine(result.Loss);
```

## Extension points

```text
ITrainingOptimizer
ITrainingLoss
ITrainingBackend
TrainingEngine
```

Future backends can be added without changing public `TrainingEngine`:

```text
CompiledTrainingBackend
GradientAccumulationBackend
MixedPrecisionTrainingBackend
DistributedTrainingBackend
CustomReinforcementLearningBackend
```
