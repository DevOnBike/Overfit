# ONNX Import — Implementation Plan

**Target:** Load PyTorch-exported ONNX models into Overfit's `Sequential` for inference via `InferenceEngine`.  
**MVP scope:** MNIST CNN (Conv + ReLU + MaxPool + Flatten + Linear). PyTorch 2.x, opset 11-20, NCHW, FP32.

---

## Status: ~85% complete

### ✅ Done

| File | Path | Notes |
|------|------|-------|
| `ProtoReader.cs` | `Sources/Main/Onnx/Protobuf/` | Minimal protobuf wire reader, zero deps |
| `OnnxModel.cs` | `Sources/Main/Onnx/Schema/` | 1 class |
| `OnnxGraph.cs` | `Sources/Main/Onnx/Schema/` | 1 class |
| `OnnxNode.cs` | `Sources/Main/Onnx/Schema/` | 1 class |
| `OnnxAttribute.cs` | `Sources/Main/Onnx/Schema/` | 1 class |
| `OnnxAttributeType.cs` | `Sources/Main/Onnx/Schema/` | 1 enum |
| `OnnxDataType.cs` | `Sources/Main/Onnx/Schema/` | 1 enum |
| `OnnxTensor.cs` | `Sources/Main/Onnx/Schema/` | 1 class, includes ExternalData support |
| `OnnxExternalDataInfo.cs` | `Sources/Main/Onnx/Schema/` | 1 class |
| `OnnxValueInfo.cs` | `Sources/Main/Onnx/Schema/` | 1 class |
| `OnnxOpsetImport.cs` | `Sources/Main/Onnx/Schema/` | 1 class |
| `OnnxProtoParser.cs` | `Sources/Main/Onnx/` | protobuf → OnnxModel, handles external_data fields 13/14 |
| `OnnxShapeContext.cs` | `Sources/Main/Onnx/` | Shape propagation through graph |
| `OnnxImporter.cs` | `Sources/Main/Onnx/` | Entry point, external data resolution, `DecodeFloatTensor` |
| `OnnxOperatorMapper.cs` | `Sources/Main/Onnx/Operators/` | Dispatch switch, all 5 ops wired |
| `GemmOperator.cs` | `Sources/Main/Onnx/Operators/` | → LinearLayer, handles transB=1 transpose |
| `ReluOperator.cs` | `Sources/Main/Onnx/Operators/` | → ReluActivation |
| `MaxPoolOperator.cs` | `Sources/Main/Onnx/Operators/` | → MaxPool2DLayer |
| `ConvOperator.cs` | `Sources/Main/Onnx/Operators/` | → ConvLayer, VALID only |
| `ReshapeOperator.cs` | `Sources/Main/Onnx/Operators/` | → FlattenLayer (rank 4→2), null (no-op if rank unchanged) |
| `MaxPool2DLayer.cs` | `Sources/Main/DeepLearning/` | IModule wrapper, new TensorStorage API |
| `GlobalAveragePool2DLayer.cs` | `Sources/Main/DeepLearning/` | IModule wrapper, new TensorStorage API |
| `FlattenLayer.cs` | `Sources/Main/DeepLearning/` | Zero-copy via AutogradNode.ViewOf |
| `LinearLayer.cs` | `Sources/Main/DeepLearning/` | +`LoadParameters(ReadOnlySpan, ReadOnlySpan)` |
| `ConvLayer.cs` | `Sources/Main/DeepLearning/` | +`LoadParameters(ReadOnlySpan)` for kernels |

### ⏳ Remaining

| Item | Effort |
|------|--------|
| 2 integration tests | ~1h |
| Test fixture files in `Tests/test_fixtures/` | 5 min (copy from generated files) |
| Update README / docs | ~30 min |

**Total remaining: ~2 hours.**

---

## Quick API (after completion)

```csharp
// Load PyTorch-exported model
var model = OnnxImporter.Load("mnist_cnn.onnx"); // .data file resolved automatically
model.Eval();

// Zero-allocation inference via InferenceEngine
using var engine = InferenceEngine.FromSequential(
    model,
    inputSize: 1 * 28 * 28,
    outputSize: 10);

var output = engine.Predict(inputFloats); // ReadOnlySpan<float>
```

---

## Test fixtures

Run once with Python to generate:
```bash
pip install torch
python Tests/test_fixtures/generate_fixture.py
# Produces: mnist_cnn.onnx, mnist_cnn.onnx.data, mnist_input.bin, mnist_output.bin
```

**Files must be in `Tests/test_fixtures/`** (or wherever the test project's working directory resolves).

---

## Integration tests to write

**Test 1 — structure:**
```csharp
[Test]
public void Load_MnistCnn_BuildsExpectedSequential()
{
    var model = OnnxImporter.Load("test_fixtures/mnist_cnn.onnx");

    var modules = model.Modules.ToArray(); // assumes Sequential exposes Modules
    Assert.That(modules[0], Is.InstanceOf<ConvLayer>());
    Assert.That(modules[1], Is.InstanceOf<ReluActivation>());
    Assert.That(modules[2], Is.InstanceOf<MaxPool2DLayer>());
    Assert.That(modules[3], Is.InstanceOf<FlattenLayer>());
    Assert.That(modules[4], Is.InstanceOf<LinearLayer>());
}
```

**Test 2 — numerical parity:**
```csharp
[Test]
public void Load_MnistCnn_OutputMatchesPyTorchReference()
{
    var model = OnnxImporter.Load("test_fixtures/mnist_cnn.onnx");
    model.Eval();

    using var engine = InferenceEngine.FromSequential(model, inputSize: 784, outputSize: 10);

    var inputBytes = File.ReadAllBytes("test_fixtures/mnist_input.bin");
    var input = MemoryMarshal.Cast<byte, float>(inputBytes).ToArray();

    var output = engine.Predict(input).ToArray();

    var expectedBytes = File.ReadAllBytes("test_fixtures/mnist_output.bin");
    var expected = MemoryMarshal.Cast<byte, float>(expectedBytes).ToArray();

    Assert.That(output.Length, Is.EqualTo(expected.Length));
    for (var i = 0; i < output.Length; i++)
    {
        Assert.That(output[i], Is.EqualTo(expected[i]).Within(1e-4f),
            $"Mismatch at output[{i}]: got {output[i]}, expected {expected[i]}");
    }
}
```

---

## File structure (complete)

```
Sources/Main/Onnx/
├── OnnxImporter.cs             ← public entry point
├── OnnxProtoParser.cs          ← byte[] → OnnxModel
├── OnnxShapeContext.cs         ← shape propagation
├── Protobuf/
│   └── ProtoReader.cs          ← wire format reader
├── Schema/                     ← 1 file per type
│   ├── OnnxModel.cs
│   ├── OnnxGraph.cs
│   ├── OnnxNode.cs
│   ├── OnnxAttribute.cs
│   ├── OnnxAttributeType.cs
│   ├── OnnxDataType.cs
│   ├── OnnxTensor.cs
│   ├── OnnxExternalDataInfo.cs
│   ├── OnnxValueInfo.cs
│   └── OnnxOpsetImport.cs
└── Operators/                  ← 1 file per operator
    ├── OnnxOperatorMapper.cs
    ├── GemmOperator.cs
    ├── ReluOperator.cs
    ├── MaxPoolOperator.cs
    ├── ConvOperator.cs
    └── ReshapeOperator.cs

Sources/Main/DeepLearning/
├── LinearLayer.cs              ← +LoadParameters(ReadOnlySpan, ReadOnlySpan)
├── ConvLayer.cs                ← +LoadParameters(ReadOnlySpan) for kernels
├── MaxPool2DLayer.cs           ← new IModule, TensorStorage API
├── GlobalAveragePool2DLayer.cs ← new IModule, TensorStorage API
└── FlattenLayer.cs             ← zero-copy via AutogradNode.ViewOf

Tests/test_fixtures/
├── mnist_cnn.onnx
├── mnist_cnn.onnx.data
├── mnist_input.bin
├── mnist_output.bin
└── generate_fixture.py
```

---

## Architectural decisions

1. **No Google.Protobuf** — custom ProtoReader, zero added dependencies.
2. **Linear topology only** — branching DAGs rejected with clear error. Residual/skip connection support deferred.
3. **External data resolved at load time** — cached per-file. Handles PyTorch ≥ 2.x default behaviour.
4. **Reshape → FlattenLayer** (never a no-op when rank changes) — explicit is safer than implicit.
5. **Weight transpose at import** — Gemm transB=1 → transpose once, zero cost at inference.
6. **`LoadParameters(ReadOnlySpan, ReadOnlySpan)`** — not `float[]`, not reflection, not BinaryWriter round-trip.
7. **`InferenceEngine.Run`** in docs, not `model.Forward(null, ...)` — promotes zero-alloc path.
8. **Conv padding deferred** — MNIST fixture uses padding=0; adding padding support touches ConvLayer + TensorMath and risks benchmark regression.

## Out of scope (defer)

- Skip connections / branching DAG
- Conv padding != 0, stride != 1, dilation != 1, grouped conv
- BatchNormalization op (folded by PyTorch eval-mode export)
- FP16 / INT8 quantized models
- ONNX export (Sequential → .onnx)
- Transformer ops (MultiHeadAttention, LayerNorm)
- LSTM/GRU ONNX mapping

---

## ONNX schema field numbers (parser reference)

| Proto | Fields |
|-------|--------|
| ModelProto | ir_version=1, producer_name=2, producer_version=3, graph=7, opset_import=8 |
| GraphProto | node=1, name=2, initializer=5, input=11, output=12 |
| NodeProto | input=1, output=2, name=3, op_type=4, attribute=5 |
| AttributeProto | name=1, f=2, i=3, s=4, t=5, floats=7, ints=8, type=20 |
| TensorProto | dims=1, data_type=2, float_data=4, int64_data=7, name=8, raw_data=9, external_data=13, data_location=14 |
| ValueInfoProto | name=1, type=2 |

Source of truth: [onnx.proto3](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3)
