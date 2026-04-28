# ONNX Import — Implementation Plan

**Target:** Load PyTorch-exported ONNX models into Overfit's `Sequential` for inference.
**MVP scope:** MNIST CNN (Conv + Pool + Linear). PyTorch 2.x exporter, opset 11-20, NCHW layout, FP32.

---

## Status: ~50% complete

### ✅ Done

| File | Path | Purpose |
|------|------|---------|
| `ProtoReader.cs` | `Sources/Main/Onnx/Protobuf/` | Minimal Protocol Buffers wire format reader (varint, fixed32/64, length-delimited). Zero dependencies. |
| `OnnxModel.cs` | `Sources/Main/Onnx/` | Domain types: `OnnxModel`, `OnnxGraph`, `OnnxNode`, `OnnxTensor`, `OnnxAttribute`, `OnnxValueInfo`, `OnnxExternalDataInfo`. Mirrors a subset of the official `onnx.proto3` schema. |
| `OnnxProtoParser.cs` | `Sources/Main/Onnx/` | Parser `byte[]` → `OnnxModel`. Supports embedded weights (raw_data, float_data) and **external data references** (PyTorch ≥ 2.x default). Field numbers per official schema, ignores unknown fields. |
| `OnnxImporter.cs` | `Sources/Main/Onnx/` | High-level entry point: `OnnxImporter.Load(path)` returns `Sequential`. Validation (opset version, no branching), external data resolution, helpers `DecodeFloatTensor`/`LoadIntoFastTensor`. Defines `OnnxShapeContext` for shape propagation through graph. |
| `MaxPool2DLayer.cs` | `Sources/Main/DeepLearning/` | `IModule` wrapper around `TensorMath.MaxPool2D` — required because `Sequential` composes `IModule`s. |
| `GlobalAveragePool2DLayer.cs` | `Sources/Main/DeepLearning/` | `IModule` wrapper around `TensorMath.GlobalAveragePool2D`. |
| `FlattenLayer.cs` | `Sources/Main/DeepLearning/` | Flatten `[batch, ...]` → `[batch, prod(...)]`. Element-identity, used between Conv and Linear in CNNs. |
| `generate_fixture.py` | (test harness) | Python script: trains tiny MnistCnn, exports to ONNX, saves reference input/output as raw float32 bin files. |

### ⏳ Remaining

| Item | Owner | Estimated effort |
|------|-------|------------------|
| `OnnxOperatorMapper.cs` (dispatch by op_type) | Next session | ~1 hour |
| 5 operator handlers (Conv, Gemm, Relu, MaxPool, Reshape/Flatten) | Next session | ~3 hours |
| Conv padding support in `ConvLayer` (currently VALID-only) | Next session | ~1 hour |
| Test fixture wiring + integration test | Next session | ~1 hour |
| README docs section + scenario doc update | After it works | ~30 min |

**Total remaining: ~6 hours of focused work.**

---

## What the test target looks like

The test fixture (PyTorch `nn.Conv2d(1,8,3) → ReLU → MaxPool2d(2) → Flatten → Linear(8*13*13, 10)`) exports as ONNX with these operators:

```
Node 1: Conv     in=(input, conv.weight, conv.bias)  out=(getitem)
                 attrs: group=1, auto_pad=NOTSET, dilations=[1,1], strides=[1,1], pads=[0,0,0,0]
Node 2: Relu     in=(getitem) out=(relu)
Node 3: MaxPool  in=(relu) out=(max_pool2d)
                 attrs: kernel_shape=[2,2], strides=[2,2], pads=[0,0,0,0], ceil_mode=0
Node 4: Reshape  in=(max_pool2d, val_12) out=(view)        ← treat as no-op
                 attrs: allowzero=1
Node 5: Gemm     in=(view, fc.weight, fc.bias) out=(output)
                 attrs: transA=0, transB=1, alpha=1.0, beta=1.0
```

**Key observations:**

- **No BatchNorm** — PyTorch `model.eval()` + new exporter folds BN into Conv weights.
- **External data** — `conv.weight` and `fc.weight` (the large initializers) are stored in `mnist_cnn.onnx.data`. Already supported.
- **Weight transpose** — `fc.weight` is `[10, 1352]` (out, in) but Overfit `LinearLayer.Weights` is `[in, out]`. Use `LoadIntoFastTensor(data, dims, transpose2D: true)`.
- **Reshape with constant operand** — `val_12` is an `int64[]` initializer (the target shape), but the reshape happens between `[1, 8, 13, 13]` and `[1, 1352]` which is just flatten. Handler can treat as no-op since `LinearLayer` accepts the implicit flatten.

---

## Implementation guide for remaining work

### 1. OnnxOperatorMapper.cs

**Location:** `Sources/Main/Onnx/OnnxOperatorMapper.cs`

**Purpose:** Dispatch table mapping ONNX `op_type` strings to operator handlers. One method, switch-based, returns `IModule?` (null = no-op skip).

**Skeleton:**
```csharp
internal static class OnnxOperatorMapper
{
    public static IModule? MapNode(
        OnnxNode node,
        Dictionary<string, OnnxTensor> initializers,
        OnnxShapeContext shapeContext)
    {
        return node.OpType switch
        {
            "Conv"     => ConvOperator.Build(node, initializers, shapeContext),
            "Gemm"     => GemmOperator.Build(node, initializers, shapeContext),
            "Relu"     => ReluOperator.Build(node, initializers, shapeContext),
            "MaxPool"  => MaxPoolOperator.Build(node, initializers, shapeContext),

            // Structural no-ops in linear pipeline
            "Reshape"  => ReshapeOperator.Build(node, initializers, shapeContext),
            "Flatten"  => FlattenOperator.Build(node, initializers, shapeContext),
            "Identity" => null,

            _ => throw new NotSupportedException(
                $"Unsupported ONNX operator: '{node.OpType}'. Supported: Conv, Gemm, Relu, MaxPool, Reshape, Flatten.")
        };
    }
}
```

Each handler:
1. Reads attributes from `node.Attributes`
2. Looks up weights in `initializers`
3. Reads input shape from `shapeContext`
4. Constructs the appropriate Overfit layer
5. Writes output shape to `shapeContext` for the next node

### 2. Operator handlers (placement: `Sources/Main/Onnx/Operators/`)

#### ConvOperator.cs

```csharp
internal static class ConvOperator
{
    public static IModule Build(OnnxNode node, Dictionary<string, OnnxTensor> initializers, OnnxShapeContext shapes)
    {
        // 1. Validate attributes
        var pads = node.Attributes.GetValueOrDefault("pads")?.IntArray ?? new long[] {0,0,0,0};
        var strides = node.Attributes.GetValueOrDefault("strides")?.IntArray ?? new long[] {1,1};
        var dilations = node.Attributes.GetValueOrDefault("dilations")?.IntArray ?? new long[] {1,1};
        var group = node.Attributes.GetValueOrDefault("group")?.IntValue ?? 1;
        var kernelShape = node.Attributes.GetValueOrDefault("kernel_shape")?.IntArray;

        // MVP: assert that all unsupported features are at default values
        if (pads.Any(p => p != 0))
            throw new NotSupportedException($"Conv padding not yet supported (pads={string.Join(",", pads)}).");
        if (strides.Any(s => s != 1))
            throw new NotSupportedException("Conv stride != 1 not yet supported.");
        if (dilations.Any(d => d != 1))
            throw new NotSupportedException("Dilated conv not yet supported.");
        if (group != 1)
            throw new NotSupportedException("Grouped conv not yet supported.");

        // 2. Extract weights (weight shape: [outC, inC, kH, kW])
        var weightTensor = initializers[node.Inputs[1]];
        var dims = weightTensor.Dims;
        if (dims.Length != 4)
            throw new InvalidDataException($"Conv weight rank should be 4, got {dims.Length}.");
        int outC = (int)dims[0], inC = (int)dims[1], kH = (int)dims[2], kW = (int)dims[3];
        if (kH != kW)
            throw new NotSupportedException($"Non-square kernels not supported ({kH}x{kW}).");

        var weightData = OnnxImporter.DecodeFloatTensor(weightTensor);

        // 3. Optional bias
        float[]? biasData = null;
        if (node.Inputs.Count >= 3 && !string.IsNullOrEmpty(node.Inputs[2]))
        {
            biasData = OnnxImporter.DecodeFloatTensor(initializers[node.Inputs[2]]);
        }

        // 4. Get input shape (must be [batch, inC, h, w])
        var inputShape = shapes.GetShape(node.Inputs[0])
            ?? throw new InvalidDataException($"Conv input '{node.Inputs[0]}' has no known shape.");
        if (inputShape.Length != 4)
            throw new InvalidDataException($"Conv input rank should be 4, got {inputShape.Length}.");
        int h = inputShape[2], w = inputShape[3];

        // 5. Construct ConvLayer
        var layer = new ConvLayer(inC, outC, h, w, kH);
        // TODO: load weights via reflection on Weights property (it's an AutogradNode)
        //       see how Sequential.Load handles this — same pattern

        // 6. Compute output shape (VALID convolution)
        int outH = h - kH + 1, outW = w - kW + 1;
        shapes.SetShape(node.Outputs[0], new[] { inputShape[0], outC, outH, outW });

        return layer;
    }
}
```

**Tricky bit:** loading weights into `ConvLayer.Weights` (which is `AutogradNode` wrapping `FastTensor`). Two options:

**A)** Add `ConvLayer.LoadWeights(float[] weights, float[]? bias)` method — clean API.
**B)** Use `BinaryWriter`/`Reader` round-trip — write to memory stream, call existing `Load`.

Option A is cleaner. Add to ConvLayer:
```csharp
public void LoadWeights(float[] weights, float[]? bias)
{
    if (weights.Length != Weights.DataView.AsReadOnlySpan().Length)
        throw new ArgumentException($"Weight size mismatch: expected {Weights.DataView.AsReadOnlySpan().Length}, got {weights.Length}.");
    weights.AsSpan().CopyTo(Weights.DataView.AsSpan());
    if (bias != null && Biases != null)
    {
        if (bias.Length != Biases.DataView.AsReadOnlySpan().Length)
            throw new ArgumentException($"Bias size mismatch.");
        bias.AsSpan().CopyTo(Biases.DataView.AsSpan());
    }
}
```

Same pattern for `LinearLayer.LoadWeights`.

#### GemmOperator.cs

```csharp
internal static class GemmOperator
{
    public static IModule Build(OnnxNode node, Dictionary<string, OnnxTensor> initializers, OnnxShapeContext shapes)
    {
        // PyTorch Linear exports as Gemm with: alpha=1, beta=1, transA=0, transB=1
        // Formula: Y = alpha*A * (transB ? B^T : B) + beta*C
        var transA = node.Attributes.GetValueOrDefault("transA")?.IntValue ?? 0;
        var transB = node.Attributes.GetValueOrDefault("transB")?.IntValue ?? 0;
        var alpha = node.Attributes.GetValueOrDefault("alpha")?.FloatValue ?? 1f;
        var beta = node.Attributes.GetValueOrDefault("beta")?.FloatValue ?? 1f;

        if (transA != 0)
            throw new NotSupportedException("Gemm transA=1 not supported.");
        if (alpha != 1f || beta != 1f)
            throw new NotSupportedException($"Gemm alpha/beta != 1 not supported (alpha={alpha}, beta={beta}).");

        // Weight tensor (B in Gemm formula)
        var weightTensor = initializers[node.Inputs[1]];
        var dims = weightTensor.Dims;
        if (dims.Length != 2)
            throw new InvalidDataException($"Gemm weight rank should be 2, got {dims.Length}.");

        // PyTorch exports with transB=1 and weight shape [out, in].
        // Overfit LinearLayer.Weights expects [in, out], so we transpose.
        // If transB=0, weight is already [in, out] — no transpose needed.
        bool needTranspose = (transB == 1);
        int inFeatures, outFeatures;
        if (transB == 1)
        {
            outFeatures = (int)dims[0];
            inFeatures = (int)dims[1];
        }
        else
        {
            inFeatures = (int)dims[0];
            outFeatures = (int)dims[1];
        }

        var weightData = OnnxImporter.DecodeFloatTensor(weightTensor);

        // Bias (optional)
        float[]? biasData = null;
        if (node.Inputs.Count >= 3 && !string.IsNullOrEmpty(node.Inputs[2]))
        {
            biasData = OnnxImporter.DecodeFloatTensor(initializers[node.Inputs[2]]);
        }

        // Build LinearLayer
        var layer = new LinearLayer(inFeatures, outFeatures);

        // If transposing needed: pre-transpose weight data before loading
        if (needTranspose)
        {
            var transposed = new float[weightData.Length];
            for (var r = 0; r < outFeatures; r++)
                for (var c = 0; c < inFeatures; c++)
                    transposed[c * outFeatures + r] = weightData[r * inFeatures + c];
            weightData = transposed;
        }

        layer.LoadWeights(weightData, biasData); // need to add this method

        // Output shape: [batch, outFeatures]
        var inputShape = shapes.GetShape(node.Inputs[0]);
        var batch = inputShape != null ? inputShape[0] : 1;
        shapes.SetShape(node.Outputs[0], new[] { batch, outFeatures });

        return layer;
    }
}
```

#### ReluOperator.cs

Trivial:
```csharp
internal static class ReluOperator
{
    public static IModule Build(OnnxNode node, Dictionary<string, OnnxTensor> initializers, OnnxShapeContext shapes)
    {
        // Output shape = input shape (element-wise activation)
        var inputShape = shapes.GetShape(node.Inputs[0]);
        if (inputShape != null) shapes.SetShape(node.Outputs[0], inputShape);
        return new ReluActivation();
    }
}
```

#### MaxPoolOperator.cs

```csharp
internal static class MaxPoolOperator
{
    public static IModule Build(OnnxNode node, Dictionary<string, OnnxTensor> initializers, OnnxShapeContext shapes)
    {
        var kernelShape = node.Attributes.GetValueOrDefault("kernel_shape")?.IntArray
            ?? throw new InvalidDataException("MaxPool missing kernel_shape attribute.");
        var strides = node.Attributes.GetValueOrDefault("strides")?.IntArray ?? new long[] {1,1};
        var pads = node.Attributes.GetValueOrDefault("pads")?.IntArray ?? new long[] {0,0,0,0};
        var ceilMode = node.Attributes.GetValueOrDefault("ceil_mode")?.IntValue ?? 0;

        // Validate constraints
        if (kernelShape.Length != 2 || kernelShape[0] != kernelShape[1])
            throw new NotSupportedException($"Non-square pool kernel not supported.");
        if (strides[0] != strides[1] || strides[0] != kernelShape[0])
            throw new NotSupportedException($"Stride must equal kernel size (was {strides[0]} vs {kernelShape[0]}).");
        if (pads.Any(p => p != 0))
            throw new NotSupportedException("MaxPool padding not supported.");
        if (ceilMode != 0)
            throw new NotSupportedException("MaxPool ceil_mode=1 not supported.");

        var poolSize = (int)kernelShape[0];

        var inputShape = shapes.GetShape(node.Inputs[0])
            ?? throw new InvalidDataException($"MaxPool input '{node.Inputs[0]}' has no known shape.");
        if (inputShape.Length != 4)
            throw new InvalidDataException($"MaxPool input rank should be 4, got {inputShape.Length}.");

        int channels = inputShape[1], h = inputShape[2], w = inputShape[3];

        var layer = new MaxPool2DLayer(channels, h, w, poolSize);

        // Output shape
        shapes.SetShape(node.Outputs[0], new[] { inputShape[0], channels, h / poolSize, w / poolSize });

        return layer;
    }
}
```

#### ReshapeOperator.cs / FlattenOperator.cs

```csharp
internal static class ReshapeOperator
{
    public static IModule? Build(OnnxNode node, Dictionary<string, OnnxTensor> initializers, OnnxShapeContext shapes)
    {
        // Reshape between Conv output [batch, C, H, W] and Linear input [batch, C*H*W]
        // is a flatten — Overfit's LinearLayer accepts unflattened input and flattens internally.
        // Treat as no-op, but propagate shape.

        var inputShape = shapes.GetShape(node.Inputs[0]);
        if (inputShape != null)
        {
            // Try to read the target shape from the second input (which is an int64[] initializer).
            // Common case: [-1, flat_dim] or [batch, flat_dim].
            if (node.Inputs.Count >= 2 && initializers.TryGetValue(node.Inputs[1], out var shapeTensor)
                && shapeTensor.Int64Data != null)
            {
                var newShape = shapeTensor.Int64Data.Select(d => (int)d).ToArray();
                // Resolve -1 dimension
                if (newShape.Contains(-1))
                {
                    var product = inputShape.Aggregate(1, (a, b) => a * b);
                    var knownProduct = newShape.Where(d => d != -1).Aggregate(1, (a, b) => a * b);
                    for (var i = 0; i < newShape.Length; i++)
                        if (newShape[i] == -1) newShape[i] = product / knownProduct;
                }
                shapes.SetShape(node.Outputs[0], newShape);
            }
            else
            {
                // Fallback: assume flatten to [batch, -1]
                var batch = inputShape[0];
                var rest = inputShape.Skip(1).Aggregate(1, (a, b) => a * b);
                shapes.SetShape(node.Outputs[0], new[] { batch, rest });
            }
        }

        // Return null = skip this node, no-op in the Sequential pipeline.
        // LinearLayer's forward path handles non-2D inputs by treating storage as flat.
        return null;
    }
}
```

**Caveat:** Verify that Overfit's `LinearLayer.Forward` accepts a 4D tensor and flattens internally. If it strictly requires 2D, insert a `FlattenLayer` here instead of returning null.

### 3. Conv padding support in ConvLayer

Currently `ConvLayer` performs valid (no-padding) convolution. PyTorch models often use `padding=1` for 3x3 convs. To unblock more models:

```csharp
// In ConvLayer.cs:
public ConvLayer(int inChannels, int outChannels, int h, int w, int kSize, int padding = 0) { ... }
```

Then update `TensorMath.Conv2D` to accept padding parameter and apply zero-padding to input before convolution. Output shape becomes `(h + 2*padding - kSize + 1)` instead of `(h - kSize + 1)`.

**Defer:** MNIST CNN test fixture uses `padding=0` so this can wait until a second test fixture (e.g., CIFAR-10 ResNet) is added.

### 4. Integration test

**Location:** `Tests/Onnx/OnnxImporterTests.cs`

```csharp
public class OnnxImporterTests
{
    private const string FixtureDir = "test_fixtures";
    private const float Tolerance = 1e-4f;

    [Test]
    public void Load_MnistCnn_ProducesEquivalentOutput()
    {
        // 1. Load model
        var modelPath = Path.Combine(FixtureDir, "mnist_cnn.onnx");
        var model = OnnxImporter.Load(modelPath);
        model.Eval();

        // 2. Load reference input
        var inputBytes = File.ReadAllBytes(Path.Combine(FixtureDir, "mnist_input.bin"));
        var inputFloats = new float[inputBytes.Length / 4];
        for (var i = 0; i < inputFloats.Length; i++)
            inputFloats[i] = BitConverter.ToSingle(inputBytes, i * 4);

        using var inputTensor = new FastTensor<float>(1, 1, 28, 28, clearMemory: false);
        inputFloats.AsSpan().CopyTo(inputTensor.GetView().AsSpan());
        using var inputNode = new AutogradNode(inputTensor, requiresGrad: false);

        // 3. Forward pass
        var output = model.Forward(null, inputNode).DataView.AsReadOnlySpan().ToArray();

        // 4. Load expected output
        var expectedBytes = File.ReadAllBytes(Path.Combine(FixtureDir, "mnist_output.bin"));
        var expected = new float[expectedBytes.Length / 4];
        for (var i = 0; i < expected.Length; i++)
            expected[i] = BitConverter.ToSingle(expectedBytes, i * 4);

        // 5. Compare element-wise
        Assert.That(output.Length, Is.EqualTo(expected.Length));
        for (var i = 0; i < output.Length; i++)
        {
            Assert.That(output[i], Is.EqualTo(expected[i]).Within(Tolerance),
                $"Output[{i}] mismatch: got {output[i]}, expected {expected[i]}");
        }
    }
}
```

**Test fixture files** (place in `Tests/test_fixtures/`):
- `mnist_cnn.onnx` (1.4 KB) — generated graph
- `mnist_cnn.onnx.data` (54 KB) — external weights file
- `mnist_input.bin` (3136 bytes = 1×1×28×28×4) — random input (seed=42)
- `mnist_output.bin` (40 bytes = 1×10×4) — reference output from PyTorch

**Generated by:** `generate_fixture.py` (already in repo).

### 5. Build / project file updates

In `Sources/Main/Main.csproj` (or wherever it lives), no changes needed — files in subdirectories are auto-included.

If using `dotnet pack`, ensure `Sources/Main/Onnx/**` is included in the package.

---

## Architectural decisions made

1. **No Google.Protobuf dependency** — we implemented a minimal ProtoReader (~250 lines). Keeps Overfit's "zero dependencies" promise.

2. **Linear topology only for MVP** — branching graphs (skip connections, residual blocks) explicitly rejected with clear error. Support requires a graph runtime, which is a larger architectural change. ResNet support deferred.

3. **PyTorch eval-mode export expected** — BatchNorm is folded into Conv weights at export time, so we don't need a `BatchNormOperator`. If users export in train-mode, they'll see `BatchNormalization` op_type and get a clear NotSupportedException.

4. **External data resolved at load time** — cached per-file (multiple initializers from same `.data` file are read once). Required because PyTorch 2.x default behavior is to externalize weights even for tiny models.

5. **Reshape as structural no-op** — between Conv output and Linear input, Reshape is a flatten. Overfit's LinearLayer handles unflattened input implicitly (treats storage as flat). If this assumption breaks for some models, add explicit `FlattenLayer` insertion in `ReshapeOperator`.

6. **Weight transpose handled in importer** — PyTorch Linear exports weight as `[out, in]`, Overfit stores as `[in, out]`. Transpose happens once at import, not per-inference.

---

## Out of scope for MVP (defer to v2)

- **Skip connections / branching DAG** — needs graph runtime in `Sequential` or new `OnnxGraphRuntime` class
- **Conv with padding != 0** — requires extending `ConvLayer` and `TensorMath.Conv2D`
- **Conv with stride != 1** — same as above
- **Grouped/depthwise convolutions** — major change to Conv kernel
- **BatchNormalization op** — only relevant if model is exported in train-mode
- **Dropout** — only meaningful in training, dropped at eval-time
- **Sigmoid, Tanh, Softmax as standalone ops** — they exist in Overfit, just need handler files
- **LSTM/GRU operators** — Overfit has LSTMLayer but mapping is non-trivial (multiple tensors, hidden state)
- **FP16 / quantized models** — INT8/FP16 require type expansion in Overfit (currently float32 only)
- **Model save/export** — only Load is in scope; Save would write `.onnx` from a `Sequential`
- **Symbolic shape inference** — currently we require concrete shapes; dynamic batch is the only dynamic dim supported

---

## Test fixture: how to regenerate

`generate_fixture.py` (already in the repo / outputs):
```bash
pip install torch
python generate_fixture.py
# Produces:
#   test_fixtures/mnist_cnn.onnx
#   test_fixtures/mnist_cnn.onnx.data
#   test_fixtures/mnist_input.bin
#   test_fixtures/mnist_output.bin
#   test_fixtures/mnist_shapes.txt
```

If switching to a different model, update the architecture in the script and re-run. Keep test fixtures small (< 100 KB total) for fast CI runs.

---

## Order of implementation for next session

1. **Add `LoadWeights(float[], float[]?)` methods** to `LinearLayer` and `ConvLayer` — 15 min, blocks everything else.
2. **Implement 5 operator handlers** in `Sources/Main/Onnx/Operators/` — 2-3 hours, can be parallelized.
3. **Implement `OnnxOperatorMapper.cs`** — 30 min, just the dispatch switch.
4. **Wire up `Sources/Main/Onnx/` namespace** if not already (ensure `OnnxOperatorMapper` is referenced from `OnnxImporter.LoadFromBytes`).
5. **Add fixture files** to `Tests/test_fixtures/`.
6. **Implement integration test** in `Tests/Onnx/OnnxImporterTests.cs` — 30 min.
7. **Run, debug shape/transpose mismatches** — typically 1-2 iterations.
8. **Add scenario doc** `docs/scenarios/onnx-import.md` — link from main README.

**Total:** ~6 hours of focused work, plus debug time.

---

## Quick API preview (after completion)

```csharp
using DevOnBike.Overfit.Onnx;
using DevOnBike.Overfit.DeepLearning;

// Load PyTorch model
var model = OnnxImporter.Load("path/to/model.onnx");
model.Eval();

// Run inference
using var input = new FastTensor<float>(1, 1, 28, 28, clearMemory: false);
// ... fill input ...
using var inputNode = new AutogradNode(input, requiresGrad: false);
var output = model.Forward(null, inputNode);

// Output is a regular Sequential — use it like any other model
```

This gives Overfit users a one-line bridge from PyTorch training to .NET deployment.

---

## Reference: ONNX schema field numbers used

For anyone extending the parser (`OnnxProtoParser.cs`):

**ModelProto:** ir_version=1, producer_name=2, producer_version=3, graph=7, opset_import=8

**GraphProto:** node=1, name=2, initializer=5, input=11, output=12

**NodeProto:** input=1, output=2, name=3, op_type=4, attribute=5, domain=7

**AttributeProto:** name=1, f=2, i=3, s=4, t=5, floats=7, ints=8, type=20

**TensorProto:** dims=1, data_type=2, float_data=4, int64_data=7, name=8, raw_data=9, external_data=13, data_location=14

**ValueInfoProto:** name=1, type=2

**TypeProto:** tensor_type=1

**TypeProto.Tensor:** elem_type=1, shape=2

**TensorShapeProto:** dim=1

**Dimension:** dim_value=1, dim_param=2

**Source of truth:** [onnx.proto3 in the official onnx repository](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3)
