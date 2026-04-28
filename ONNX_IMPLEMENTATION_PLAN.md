# ONNX Import — Implementation Plan

**Target:** Load PyTorch-exported ONNX models into Overfit's `Sequential` for inference.

**MVP scope:** MNIST CNN (`Conv + ReLU + MaxPool + Flatten + Linear`). PyTorch 2.x exporter, opset 11-20, NCHW layout, FP32.

## Current status

The ONNX import MVP is now implemented far enough to validate the full deployment bridge:

```text
PyTorch ONNX fixture
-> OnnxImporter.Load(...)
-> Sequential
-> InferenceEngine.Run(...)
```

Current imported ONNX benchmark:

| Runtime | Result | Notes |
|---|---:|---|
| Overfit imported ONNX | ~7.5 us/op | BenchmarkDotNet, 0 B/op |
| ONNX Runtime preallocated | ~7.5 us/op | BenchmarkDotNet, 224 B/op |
| PyTorch eager CPU | ~27.3 us/op | Python reference script, 1 thread |

## Done

| Area | Status |
|---|---|
| Minimal protobuf reader | Done |
| ONNX domain model | Done |
| ONNX protobuf parser | Done |
| External data resolution | Done |
| Shape context | Done |
| Operator mapper | Done |
| Conv / Gemm / Relu / MaxPool / Reshape-Flatten handlers | Done for MVP |
| FlattenLayer / MaxPool2DLayer / GlobalAveragePool2DLayer | Done |
| Fixture generation script | Done |
| InferenceEngine smoke test | Done |
| Imported ONNX BenchmarkDotNet benchmark | Done |
| PyTorch reference script | Done |

## Supported operators in MVP

```text
Conv       valid convolution only, group=1, dilation=1, stride=1 for current fixture path
Relu       element-wise activation
MaxPool    2D square pool, no padding, ceil_mode=0
Reshape    structural flatten when rank decreases to 2
Flatten    explicit flatten layer
Gemm       LinearLayer, with PyTorch [out,in] -> Overfit [in,out] transpose at import time
```

## Non-goals for MVP

```text
Full ONNX runtime compatibility
Branching DAGs / skip connections
ResNet import
Grouped/depthwise convolution
FP16 / INT8 / quantized models
BatchNormalization operator from train-mode exports
LSTM / GRU ONNX operators
Symbolic shape inference beyond concrete model shapes
```

## Public usage

Prefer documenting imported ONNX models through `InferenceEngine.Run(...)`, not `model.Forward(...)`:

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

## Safety and correctness notes

- External data paths must be resolved relative to the model directory.
- Absolute external data paths should be rejected.
- Path traversal outside the model directory should be rejected.
- Import-time allocations are acceptable.
- Inference-time allocations are not acceptable for zero-allocation claims.
- `Predict(...).ToArray()` must not be used in zero-allocation tests or benchmarks.
- `Sources/Main` should avoid LINQ and prefer explicit loops.

## Benchmark commands

```bash
dotnet test -c Release --filter "OnnxImporterTests"
dotnet run -c Release --project Sources/Benchmark --filter "*ImportedOnnxMnistCnnBenchmark*"
python Tests/benchmark_pytorch_mnist_cnn.py --fixture-dir Tests/test_fixtures --threads 1
```

## Next steps

1. Add README scenario doc: `docs/scenarios/onnx-import.md`.
2. Add one or two negative tests for unsupported Conv attributes.
3. Keep `OnnxImporter` and operators LINQ-free in `Sources/Main`.
4. Add optional `OnnxImporter.LoadInferenceEngine(...)` convenience API later.
5. Defer Conv padding/stride until the MVP is stable on `main`.
