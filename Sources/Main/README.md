# Sources/Main

Core Overfit runtime and library code.

## Current runtime areas

```text
Autograd      Reverse-mode graph and training operations
DeepLearning  Layers, Sequential, train/eval state, save/load
Inference     Prepared zero-allocation inference facade
Ops           TensorMath and graph-aware training math
Tensors       Tensor storage, views and memory abstractions
Evolutionary  Population-based gradient-free optimization
Onnx          Focused ONNX import MVP for PyTorch-exported inference models
```

## Inference path

The preferred production-style inference API is:

```csharp
engine.Run(input, output);
```

The hot path should avoid:

```text
model.Forward(...)
AutogradNode
ComputationGraph
new arrays per call
ToArray()
LINQ in runtime code
```

`InferenceEngine.Run(...)` uses caller-owned input/output buffers and prepared reusable internal buffers.

## ONNX import

The ONNX importer is a focused load-time feature. It is not a full ONNX runtime.

Current MVP goal:

```text
PyTorch-exported eval-mode ONNX CNN
-> OnnxImporter.Load(path)
-> Sequential
-> InferenceEngine.Run(...)
```

Supported MVP operators:

```text
Conv
Relu
MaxPool
Reshape / Flatten
Gemm
```

Constraints:

```text
FP32 only
NCHW layout
Linear topology only
Concrete shapes required
No branching / skip connections
No grouped/depthwise conv
No quantized or FP16 tensors
```

Example:

```csharp
using DevOnBike.Overfit.Onnx;
using DevOnBike.Overfit.Inference;

var model = OnnxImporter.Load("mnist_cnn.onnx");
model.Eval();

using var engine = InferenceEngine.FromSequential(
    model,
    inputSize: 1 * 28 * 28,
    outputSize: 10);

var input = new float[1 * 28 * 28];
var output = new float[10];

engine.Run(input, output);
```

## Main-project coding rules

Runtime code in `Sources/Main` should stay conservative:

- Avoid LINQ in hot/runtime paths.
- Prefer explicit loops and spans.
- Avoid hidden allocations.
- Keep import-time allocations out of inference hot paths.
- Keep training/graph allocation policy separate from inference policy.
- Do not add dependencies unless there is a clear architectural reason.

## Benchmark status

Current verified hot-path inference results include:

```text
Linear(784,10): ~250-300 ns/op, 0 B
Linear(4096,10): ~1.08 us, 0 B
MLP 784->256->128->10: ~10-12 us, 0 B
Small CNN: ~5-6.5 us, 0 B
Imported ONNX MNIST CNN: ~7.5 us, 0 B
```

See `Sources/Benchmark/README.md` and `docs/InferenceBenchmarkSummary.md` for full benchmark tables.
