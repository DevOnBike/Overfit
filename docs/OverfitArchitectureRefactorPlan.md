# Overfit Architecture Refactor Plan

## Goal of the refactor

The goal of the refactor is not a one-time rewrite of the library. The goal is to separate concerns so that further development of training, inference, optimizers, and kernels is simpler, safer, and easier to measure.

The most important architectural decision:

```text
TrainingEngine   = training workflow facade
ComputationGraph = autograd / training runtime brain
Layers           = parameters + graph operation composition
Kernels          = pure math on Spans
InferenceEngine  = separate inference facade
```

The current biggest architectural problem is role mixing:

```text
AutogradNode = graph value + parameter + temporary + memory owner + gradient owner
ComputationGraph = tape + allocator + backward executor + op facade + workspace manager
TensorMath = name suggests pure math, but currently contains graph-aware operations
```

The target is for `ComputationGraph` to be the central training API. Methods that record the tape should be methods on the graph:

```csharp
var prediction = model.Forward(graph, input);
var loss = graph.SoftmaxCrossEntropy(prediction, target);

graph.Backward(loss);
```

Instead of the current style:

```csharp
TensorMath.SoftmaxCrossEntropy(graph, prediction, target);
```

`TensorMath` should mean pure math or be replaced by `Kernels`.

---

## Target responsibility breakdown

### 1. `TensorStorage<T>`

**Responsibility:** physical memory.

Does:

```text
- owns the buffer
- returns Span<T> / ReadOnlySpan<T>
- knows Length / Size
- Dispose / ReturnToPool
```

Does not:

```text
- does not know shape
- does not know gradients
- does not know autograd
- does not know graph
- does not know layers
- does not know optimizers
```

This is the lowest memory level.

---

### 2. `TensorShape`

**Responsibility:** description of tensor dimensions.

Does:

```text
- D0/D1/D2/D3
- Rank
- ElementCount
- shape validation
```

Does not:

```text
- does not know storage
- does not know data
- does not know gradients
- does not know graph
```

---

### 3. `TensorView<T>` / `TensorSpan<T>`

**Responsibility:** view over memory + shape, without ownership.

Example direction:

```csharp
public readonly struct TensorView<T>
{
    public TensorStorage<T> Storage { get; }
    public TensorShape Shape { get; }
    public int Offset { get; }
}
```

Or for the hot path:

```csharp
public readonly ref struct TensorSpan<T>
{
    public Span<T> Data { get; }
    public TensorShape Shape { get; }
}
```

Does:

```text
- interprets memory as a tensor
- may have offset/stride in the future
```

Does not:

```text
- does not own memory
- does not dispose
- does not know gradients
- does not know the graph
```

---

### 4. `Parameter`

**Responsibility:** long-lived trainable model tensor.

A parameter should not be a plain `AutogradNode`, because it has a different lifecycle than a temporary node in the graph.

Example abstraction:

```csharp
public sealed class Parameter : IDisposable
{
    public TensorStorage<float> Data { get; }
    public TensorStorage<float> Grad { get; }
    public TensorShape Shape { get; }
    public bool RequiresGrad { get; }

    public void ZeroGrad()
    {
        Grad.AsSpan().Clear();
    }

    public AutogradNode AsNode()
    {
        // Lightweight graph-visible view over parameter.
        // Initially may return a wrapper compatible with the old API.
    }
}
```

Does:

```text
- owns Data
- owns Grad
- lives together with the layer/model
- is visible to optimizers
- can zero Grad
- may be able to Save/Load data
```

Does not:

```text
- is not a tape op
- is not a temporary node
- does not know ComputationGraph
- does not execute backward
```

Ultimately, optimizers should accept:

```csharp
IEnumerable<Parameter>
```

instead of:

```csharp
IEnumerable<AutogradNode>
```

---

### 5. `AutogradNode`

**Responsibility:** handle for a value participating in the autograd graph.

Ultimately it should be a graph value handle rather than an owner of everything.

Does:

```text
- points to DataView
- points to GradView, if it exists
- has Shape
- has RequiresGrad
- has Ownership metadata
- may have NodeId/generation for debugging
```

Does not:

```text
- should not decide on its own where to allocate memory
- should not create grad storage on its own without context
- should not be both a parameter and a temporary simultaneously
- does not know optimizers
- does not execute backward
```

Minimal ownership metadata:

```csharp
public enum AutogradNodeOwnership
{
    Unknown = 0,
    Parameter = 1,
    GraphTemporary = 2,
    GraphAuxiliary = 3,
    ExternalBorrowed = 4,
    View = 5
}
```

Target properties:

```csharp
public AutogradNodeOwnership Ownership { get; }
public bool OwnsDataStorage { get; }
public bool OwnsGradStorage { get; }
public bool IsDisposed { get; }
```

This allows debugging and enforcing the lifecycle.

---

### 6. `ComputationGraph`

**Responsibility:** training/autograd brain.

The graph should tie training logic together at the autograd operation level.

Does:

```text
- creates temporary nodes
- creates external/input nodes
- creates parameter views
- records TapeOp
- executes Backward
- resets graph temporaries
- exposes graph-aware ops: Linear, Conv2D, ReLU, Losses
```

Does not:

```text
- is not a TrainingEngine
- is not an optimizer
- does not contain manual SIMD loops
- should not be a general-purpose storage pool
- does not own model parameters
```

Example API:

```csharp
public partial class ComputationGraph
{
    public AutogradNode Linear(
        AutogradNode input,
        AutogradNode weights,
        AutogradNode bias)
    {
        return TensorMath.Linear(this, input, weights, bias); // transitional stage
    }

    public AutogradNode Relu(AutogradNode input)
    {
        return TensorMath.ReLU(this, input); // transitional stage
    }

    public AutogradNode SoftmaxCrossEntropy(
        AutogradNode prediction,
        AutogradNode target)
    {
        return TensorMath.SoftmaxCrossEntropy(this, prediction, target);
    }
}
```

Ultimately, the implementation of these methods should look like this:

```text
graph.Linear(...)
    -> validate shape
    -> graph.CreateTemporary(...)
    -> LinearKernels.Forward(...)
    -> graph.Record(TapeOp.Linear)
    -> return output
```

Important: `ComputationGraph` may have operations as methods, but the implementation must be split across `partial` files so the main file does not become a monolith.

Proposed folder:

```text
Autograd/
  ComputationGraph.cs
  ComputationGraph.Linear.cs
  ComputationGraph.Conv2D.cs
  ComputationGraph.Activation.cs
  ComputationGraph.Pooling.cs
  ComputationGraph.Losses.cs
  ComputationGraph.Backward.cs
  TapeOp.cs
  OpCode.cs
  AutogradNode.cs
  AutogradNodeOwnership.cs
```

---

### 7. Graph allocator

**Responsibility:** creating nodes with the correct lifecycle.

May be internal.

Example abstraction:

```csharp
internal interface IGraphTensorAllocator
{
    AutogradNode CreateTemporary(
        TensorShape shape,
        bool requiresGrad,
        string? debugName = null);

    AutogradNode CreateExternalBorrowed(
        TensorStorage<float> data,
        TensorShape shape,
        bool requiresGrad,
        string? debugName = null);

    AutogradNode CreateParameterView(
        Parameter parameter,
        string? debugName = null);
}
```

This removes allocation decisions from `AutogradNode` and moves them to the graph.

---

### 8. `TapeOp`

**Responsibility:** recording a single operation to the backward tape.

Does:

```text
- OpCode
- references to input/output nodes
- small context slots
- shape/context ints
```

Does not:

```text
- does not execute forward
- does not allocate
- does not dispose itself
```

For performance, the better direction is:

```text
struct TapeOp + enum OpCode + switch in BackwardExecutor
```

Using a basic `IOpBackward` per-op system is not recommended, as it may add dispatch overhead and complexity. Custom ops can be added later as a separate extension point.

---

### 9. `BackwardExecutor`

**Responsibility:** executing backward over the tape.

May be a separate class or part of `ComputationGraph.Backward.cs`.

Does:

```text
- iterates the tape from the end
- switch on OpCode
- calls backward kernels
- accumulates gradients
```

Does not:

```text
- does not build the forward graph
- does not know optimizers
- does not know layers
```

Example:

```csharp
internal sealed class BackwardExecutor
{
    public void Execute(
        ReadOnlySpan<TapeOp> tape,
        AutogradNode loss)
    {
        for (var i = tape.Length - 1; i >= 0; i--)
        {
            var op = tape[i];

            switch (op.OpCode)
            {
                case OpCode.Linear:
                    // Linear backward
                    break;

                case OpCode.Relu:
                    // ReLU backward
                    break;
            }
        }
    }
}
```

---

### 10. `TensorMath` and `Kernels`

Naming decision:

```text
TensorMath = pure math or facade over kernels
Kernels    = low-level optimized Span-only implementations
```

The current graph-aware `TensorMath.*` should ultimately disappear from layers.

Target clean API:

```csharp
TensorMath.LinearForward(...);
TensorMath.Conv2DValidNchw(...);
TensorMath.Relu(...);
TensorMath.SoftmaxCrossEntropyForward(...);
```

Or without the intermediary:

```csharp
LinearKernels.Forward(...);
Conv2DKernels.ForwardValidNchw(...);
ActivationKernels.Relu(...);
LossKernels.SoftmaxCrossEntropyForward(...);
```

`Kernels` do:

```text
- Span<float> in/out
- int shape params
- SIMD/scalar dispatch
- zero knowledge of AutogradNode
- zero knowledge of ComputationGraph
- no allocations
```

Do not:

```text
- do not record tape
- do not manage ownership
- do not know parameters
```

---

### 11. Layers

**Responsibility:** parameters + operation composition.

A layer should:

```text
- own Parameters
- know input/output shape
- Forward for training via graph ops
- ForwardInference via Kernels/TensorMath
- Train/Eval
- PrepareInference cache
- Save/Load parameters
```

A layer should not:

```text
- implement long SIMD loops
- record TapeOp manually
- manage graph temporary lifecycle
- know optimizers
```

Target style:

```csharp
public sealed class LinearLayer : IModule
{
    public Parameter Weights { get; }
    public Parameter Bias { get; }

    public AutogradNode Forward(
        ComputationGraph graph,
        AutogradNode input)
    {
        return graph.Linear(input, Weights, Bias);
    }

    public void ForwardInference(
        ReadOnlySpan<float> input,
        Span<float> output)
    {
        LinearKernels.Forward(
            input,
            Weights.Data.AsReadOnlySpan(),
            _weightsTransposed.AsReadOnlySpan(),
            Bias.Data.AsReadOnlySpan(),
            output,
            _inputSize,
            _outputSize);
    }

    public IEnumerable<Parameter> Parameters()
    {
        yield return Weights;
        yield return Bias;
    }
}
```

---

### 12. Module interfaces

The current `IModule` can be split into roles, with `IModule` kept as a composite facade.

Example interfaces:

```csharp
public interface IParameterProvider
{
    IEnumerable<Parameter> Parameters();
}

public interface ITrainableModule
{
    AutogradNode Forward(
        ComputationGraph graph,
        AutogradNode input);
}

public interface IInferenceModule
{
    void ForwardInference(
        ReadOnlySpan<float> input,
        Span<float> output);
}

public interface IModelModeAware
{
    bool IsTraining { get; }

    void Train();

    void Eval();

    void PrepareInference();
}

public interface IModule :
    IParameterProvider,
    ITrainableModule,
    IInferenceModule,
    IModelModeAware,
    IDisposable
{
}
```

In practice, this does not need to be done immediately. This is the target direction, for when the large surface area of `IModule` starts to get in the way.

---

### 13. `Sequential`

**Responsibility:** module composition.

Does:

```text
- stores a list of IModule
- Forward through modules
- ForwardInference through modules
- manages inference workspace between layers
- propagates Train/Eval
- aggregates Parameters()
- Save/Load through modules
```

Does not:

```text
- does not know Linear/Conv/ReLU math
- does not know optimizers
- does not execute backward
```

---

### 14. Losses

Loss can be part of graph ops, because loss records the tape:

```csharp
var loss = graph.SoftmaxCrossEntropy(prediction, target);
```

A public abstraction for `TrainingEngine` can also be provided:

```csharp
public interface ITrainingLoss
{
    AutogradNode Forward(
        ComputationGraph graph,
        AutogradNode prediction,
        AutogradNode target);

    void Backward(
        ComputationGraph graph,
        AutogradNode loss);

    float ReadScalar(AutogradNode loss);
}
```

Target concrete implementation:

```csharp
public sealed class SoftmaxCrossEntropyLoss : ITrainingLoss
{
    public AutogradNode Forward(
        ComputationGraph graph,
        AutogradNode prediction,
        AutogradNode target)
    {
        return graph.SoftmaxCrossEntropy(prediction, target);
    }

    public void Backward(
        ComputationGraph graph,
        AutogradNode loss)
    {
        graph.Backward(loss);
    }

    public float ReadScalar(AutogradNode loss)
    {
        return loss.DataView.AsReadOnlySpan()[0];
    }
}
```

---

### 15. Optimizers

**Responsibility:** parameter updates.

An optimizer does:

```text
- holds optimizer state
- ZeroGrad
- Step
- updates Parameter.Data based on Parameter.Grad
```

Does not:

```text
- does not know ComputationGraph
- does not do forward
- does not compute loss
- does not know about the batch
```

Ultimately:

```csharp
public interface IOptimizer
{
    void ZeroGrad();
    void Step();
}
```

But underneath it should be built from:

```csharp
IEnumerable<Parameter>
```

not from temporary autograd nodes.

---

### 16. `TrainingEngine`

**Responsibility:** workflow for a single training step.

Does:

```text
- accepts batch input/target
- creates/fills input/target nodes
- optimizer.ZeroGrad()
- model.Forward(graph, input)
- loss.Forward(graph, prediction, target)
- graph.Backward(loss)
- optimizer.Step()
- graph.Reset()
```

Does not:

```text
- does not implement math
- does not know SIMD
- is not ComputationGraph
- is not an optimizer
```

You already have this as a facade. In this model, `TrainingEngine` remains the orchestrator and `ComputationGraph` is the autograd brain.

---

### 17. `InferenceEngine`

**Responsibility:** inference workflow.

Does:

```text
- model.Eval()
- model.PrepareInference()
- warmup
- input/output validation
- preallocated output buffer
- Run/Predict
```

Does not:

```text
- does not know AutogradNode
- does not know ComputationGraph
- does not know gradients
- does not know optimizers
```

The inference path should be completely separated from autograd.

---

## Target training flow

```text
TrainingEngine.TrainBatch(input, target)
    -> inputNode = graph.CreateExternalBorrowed(...) or preallocated input node
    -> targetNode = graph.CreateExternalBorrowed(...) or preallocated target node
    -> optimizer.ZeroGrad()
    -> prediction = model.Forward(graph, inputNode)
        -> LinearLayer.Forward
            -> graph.Linear(input, Weights, Bias)
                -> graph.CreateTemporary(...)
                -> LinearKernels.Forward(...)
                -> graph.Record(TapeOp.Linear)
                -> return outputNode
    -> lossNode = graph.SoftmaxCrossEntropy(prediction, targetNode)
    -> graph.Backward(lossNode)
        -> BackwardExecutor
            -> Linear backward kernels
            -> Conv backward kernels
    -> optimizer.Step()
    -> graph.Reset()
```

---

## Target inference flow

```text
InferenceEngine.Predict(input)
    -> Sequential.ForwardInference(input, output)
        -> LinearLayer.ForwardInference
            -> LinearKernels.Forward
        -> ReluActivation.ForwardInference
            -> ActivationKernels.Relu
        -> ConvLayer.ForwardInference
            -> Conv2DKernels.ForwardValidNchw
```

Inference should not touch:

```text
AutogradNode
ComputationGraph
TapeOp
Grad
Optimizer
Loss
```

---

## Refactor plan by stages

### Stage 0 — freeze the current inference milestone

Status:

```text
- zero-alloc inference works
- MLP/CNN faster than ONNX Runtime on small workloads
- kernels partially extracted from layers
- TrainingEngine exists as a training facade
```

Action:

```text
- commit the current stable state
- do not mix the next refactor with inference optimization
```

---

### Stage 1 — Graph operation facade

Goal: layers should call `graph.*`, not `TensorMath.*`.

Scope:

```text
- add partial ComputationGraph.* wrappers
- do not move TensorMath implementations yet
- change layers to call graph.Linear / graph.Conv2D / graph.Relu etc.
```

Files:

```text
Sources/Main/Autograd/ComputationGraph.Linear.cs
Sources/Main/Autograd/ComputationGraph.Conv2D.cs
Sources/Main/Autograd/ComputationGraph.Activation.cs
Sources/Main/Autograd/ComputationGraph.Pooling.cs
Sources/Main/Autograd/ComputationGraph.Losses.cs
```

Example:

```csharp
public partial class ComputationGraph
{
    public AutogradNode Linear(
        AutogradNode input,
        AutogradNode weights,
        AutogradNode bias)
    {
        return TensorMath.Linear(this, input, weights, bias);
    }
}
```

Layer after the change:

```csharp
public AutogradNode Forward(
    ComputationGraph graph,
    AutogradNode input)
{
    return graph.Linear(input, Weights, Bias);
}
```

Risk: low. This is mainly an API/call-site change.

---

### Stage 2 — `AutogradNodeOwnership`

Goal: name node lifecycles without changing behavior.

Scope:

```text
- add AutogradNodeOwnership enum
- add Ownership to AutogradNode
- add debug/tests for ownership
- do not change Reset/Dispose semantics yet
```

Files:

```text
Sources/Main/Autograd/AutogradNodeOwnership.cs
Sources/Main/Autograd/AutogradNode.cs
Tests/AutogradOwnershipTests.cs
```

Example:

```csharp
public enum AutogradNodeOwnership
{
    Unknown = 0,
    Parameter = 1,
    GraphTemporary = 2,
    GraphAuxiliary = 3,
    ExternalBorrowed = 4,
    View = 5
}
```

Risk: low, if we are only adding metadata.

---

### Stage 3 — Graph factory methods

Goal: the graph starts explicitly creating nodes by ownership.

Scope:

```text
- graph.CreateTemporary(...)
- graph.CreateExternalBorrowed(...)
- graph.CreateAuxiliary(...)
- graph.CreateParameterView(...)
```

Example:

```csharp
public AutogradNode CreateTemporary(
    TensorShape shape,
    bool requiresGrad,
    string? debugName = null)
{
    // allocate storage from graph arena/pool
    // create node with Ownership = GraphTemporary
}
```

Risk: medium. Not yet rewiring all of TensorMath.

---

### Stage 4 — `Parameter` as a separate type

Goal: separate long-lived model parameters from temporary nodes.

Scope:

```text
- add Parameter
- add ParameterCollection
- add ParameterFactory optionally
- do not rewire all layers at once yet
```

Files:

```text
Sources/Main/Parameters/Parameter.cs
Sources/Main/Parameters/ParameterCollection.cs
Sources/Main/Parameters/ParameterFactory.cs
Tests/ParameterTests.cs
```

Example:

```csharp
public sealed class Parameter : IDisposable
{
    public TensorStorage<float> Data { get; }
    public TensorStorage<float> Grad { get; }
    public TensorShape Shape { get; }
    public bool RequiresGrad { get; }

    public void ZeroGrad()
    {
        Grad.AsSpan().Clear();
    }
}
```

Risk: low, if we are only adding the type.

---

### Stage 5 — rewire `LinearLayer` to `Parameter`

Goal: the first layer uses the new parameter model.

Scope:

```text
- LinearLayer.Weights: Parameter
- LinearLayer.Bias: Parameter
- graph.Linear overload for Parameter
- Parameters() returns Parameter or a transitional adapter
```

Example:

```csharp
public AutogradNode Forward(
    ComputationGraph graph,
    AutogradNode input)
{
    return graph.Linear(input, Weights, Bias);
}
```

Risk: medium. Only one layer.

---

### Stage 6 — optimizer adapter / Adam on `Parameter`

Goal: the optimizer does not see temporary `AutogradNode`.

Scope:

```text
- Adam(IEnumerable<Parameter>)
- optionally an adapter for the old API
- ZeroGrad/Step over Parameter.Data/Grad
```

Example:

```csharp
public sealed class Adam : IOptimizer
{
    private readonly Parameter[] _parameters;

    public Adam(IEnumerable<Parameter> parameters, float learningRate)
    {
        _parameters = parameters.ToArray();
    }

    public void ZeroGrad()
    {
        foreach (var p in _parameters)
        {
            p.ZeroGrad();
        }
    }

    public void Step()
    {
        // update p.Data using p.Grad
    }
}
```

Risk: medium/high, because it touches training.

---

### Stage 7 — move graph-aware implementations from `TensorMath` to `ComputationGraph.*`

Goal: `TensorMath` stops being graph-aware.

Scope:

```text
- graph.Linear contains the autograd wrapper
- graph.Conv2D contains the autograd wrapper
- graph.Relu contains the autograd wrapper
- graph.SoftmaxCrossEntropy contains the autograd wrapper
- TensorMath remains as pure math or is removed
```

Target example:

```csharp
public AutogradNode Linear(
    AutogradNode input,
    Parameter weights,
    Parameter bias)
{
    var output = CreateTemporary(
        new TensorShape(input.Shape.D0, weights.Shape.D1),
        requiresGrad: input.RequiresGrad || weights.RequiresGrad || bias.RequiresGrad);

    LinearKernels.Forward(
        input.DataView.AsReadOnlySpan(),
        weights.Data.AsReadOnlySpan(),
        weights.TransposedCache.AsReadOnlySpan(),
        bias.Data.AsReadOnlySpan(),
        output.DataView.AsSpan(),
        weights.Shape.D0,
        weights.Shape.D1);

    RecordLinear(input, weights, bias, output);

    return output;
}
```

Risk: medium/high. Do one operation at a time.

---

### Stage 8 — Graph Reset cleanup

Goal: the graph cleans up by ownership, not by manual exceptions.

Scope:

```text
- graph.Reset disposes only GraphTemporary / GraphAuxiliary
- does not touch Parameter / ExternalBorrowed
- debug assert: node disposed exactly once
```

Example:

```csharp
private static void DisposeIfGraphOwned(AutogradNode? node)
{
    if (node is null)
    {
        return;
    }

    if (node.Ownership is AutogradNodeOwnership.GraphTemporary or
        AutogradNodeOwnership.GraphAuxiliary)
    {
        node.Dispose();
    }
}
```

Risk: high. Only do this once ownership is certain.

---

### Stage 9 — Backward kernels cleanup

Goal: backward uses clean kernels, analogously to inference forward.

Scope:

```text
- LinearKernels.BackwardInput
- LinearKernels.AccumulateWeightGrad
- Conv2DKernels.BackwardInput
- Conv2DKernels.AccumulateKernelGrad
- LossKernels backward helpers
```

Risk: medium/high. This is the performance stage.

---

### Stage 10 — Training performance work

Only after ownership cleanup.

Scope:

```text
- profiling ZeroGrad / Forward / Loss / Backward / Step / Reset
- allocation reduction, if it matters
- optimizer kernels
- graph arena improvements
- custom scheduler, if still needed
```

At this stage, training allocations may exist. Correctness of the lifecycle and performance are the primary concerns.

---

## Refactor principles

### 1. Do not mix architectural refactoring with optimization

One PR = one goal.

Good:

```text
PR: Add graph operation facade
PR: Add AutogradNodeOwnership metadata
PR: Introduce Parameter type
```

Bad:

```text
PR: Add Parameter + rewrite Adam + optimize Conv backward + change benchmarks
```

---

### 2. Adapters first, then migration

Add the new API as a wrapper over the old one first.

Only then rewire layers and implementations.

---

### 3. Keep inference separated from the graph

The inference path is already good and zero-alloc. Do not connect inference back to `ComputationGraph`.

---

### 4. Kernels do not know autograd

If a method accepts an `AutogradNode`, it is not a kernel.

---

### 5. A method with `ComputationGraph` in its signature should be a method on the graph

Ultimately we do not want:

```csharp
TensorMath.Linear(graph, input, weights, bias);
```

We want:

```csharp
graph.Linear(input, weights, bias);
```

---

## Minimal first PR

The safest first step:

```text
PR: Add graph operation facade
```

Scope:

```text
1. Add partial ComputationGraph.* files.
2. Each method delegates to the current TensorMath.
3. Rewire layers to graph.*.
4. Tests.
5. Zero changes to algorithms.
```

Example files:

```text
Sources/Main/Autograd/ComputationGraph.Linear.cs
Sources/Main/Autograd/ComputationGraph.Conv2D.cs
Sources/Main/Autograd/ComputationGraph.Activation.cs
Sources/Main/Autograd/ComputationGraph.Pooling.cs
Sources/Main/Autograd/ComputationGraph.Losses.cs
```

Example wrappers:

```csharp
public partial class ComputationGraph
{
    public AutogradNode Linear(
        AutogradNode input,
        AutogradNode weights,
        AutogradNode bias)
    {
        return TensorMath.Linear(this, input, weights, bias);
    }

    public AutogradNode Conv2D(
        AutogradNode input,
        AutogradNode kernels,
        int inChannels,
        int outChannels,
        int inputH,
        int inputW,
        int kernelSize)
    {
        return TensorMath.Conv2D(
            this,
            input,
            kernels,
            inChannels,
            outChannels,
            inputH,
            inputW,
            kernelSize);
    }

    public AutogradNode Relu(AutogradNode input)
    {
        return TensorMath.ReLU(this, input);
    }

    public AutogradNode SoftmaxCrossEntropy(
        AutogradNode prediction,
        AutogradNode target)
    {
        return TensorMath.SoftmaxCrossEntropy(this, prediction, target);
    }
}
```

Then the layer:

```csharp
public AutogradNode Forward(
    ComputationGraph graph,
    AutogradNode input)
{
    return graph.Linear(input, Weights, Bias);
}
```

---

## Target state after the refactor

After full migration:

```text
TrainingEngine
    orchestrates workflow

ComputationGraph
    owns autograd runtime and graph-aware operations

Parameter
    owns model trainable state

AutogradNode
    is graph-visible value handle

Kernels/TensorMath
    pure math only

Layers
    compose graph ops and own parameters

Optimizers
    update Parameters

InferenceEngine
    separate zero-alloc inference workflow
```

Most important outcome:

```text
AutogradNode stops being everything at once.
ComputationGraph is the training brain, but not the muscle of the math.
TensorMath/Kernels are pure math.
Layer is simple and readable.
Optimizer does not see temporary graph nodes.
```
