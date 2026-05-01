// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Onnx.Operators;
using DevOnBike.Overfit.Onnx.Schema;

namespace DevOnBike.Overfit.Onnx
{
    /// <summary>
    /// Loads ONNX models with arbitrary DAG topology (including skip connections)
    /// into an <see cref="OnnxGraphModel"/>.
    ///
    /// Use this instead of <see cref="OnnxImporter"/> when the model has branching
    /// tensors (ResNet, DenseNet, EfficientNet).
    ///
    /// Entry points:
    ///   <see cref="Load(string, int, int)"/>               — from file path.
    ///   <see cref="LoadFromBytes(byte[], int, int, string?)"/> — from byte array.
    ///
    /// The difference from OnnxImporter (Sequential):
    ///   - Accepts branching graphs (tensor with multiple consumers).
    ///   - Returns OnnxGraphModel (not Sequential) — no IModule interface.
    ///   - Buffers are pre-allocated per tensor (slot-based execution).
    /// </summary>
    public static class OnnxGraphImporter
    {
        private const int MinSupportedOpset = 11;
        private const int MaxSupportedOpset = 20;

        public static OnnxGraphModel Load(string path, int inputSize, int outputSize)
        {
            var fullPath = Path.GetFullPath(path);
            var modelBytes = File.ReadAllBytes(fullPath);
            var modelDir = Path.GetDirectoryName(fullPath) ?? string.Empty;

            return LoadFromBytes(modelBytes, inputSize, outputSize, modelDir);
        }

        public static OnnxGraphModel LoadFromBytes(
            byte[] modelBytes,
            int inputSize,
            int outputSize,
            string? externalDataDir = null)
        {
            var model = OnnxProtoParser.ParseModel(modelBytes);

            ValidateOpsets(model);

            // Re-use OnnxImporter's external data resolution and initializer lookup.
            // We access them via the internal API (same assembly).
            ResolveExternalData(model, externalDataDir);

            var initializers = BuildInitializerLookup(model.Graph.Initializers);
            var shapeContext = new OnnxShapeContext();

            SeedInputShapes(model, initializers, shapeContext);
            SeedInitializerShapes(model, shapeContext);

            // ── Build tensor → slot mapping ─────────────────────────────────
            // Each unique intermediate tensor gets a buffer slot.
            // Slot 0 = model input (written by caller before RunInference).
            // Slots 1..N = intermediate activations.

            var slotMap = new Dictionary<string, int>(StringComparer.Ordinal);
            var bufferSizes = new List<int> { inputSize }; // slot 0 = input

            // Seed slot 0 with the graph's primary input tensor name.
            var graphInputName = GetPrimaryInputName(model, initializers);
            slotMap[graphInputName] = 0;

            var execNodes = new List<OnnxGraphNode>();

            foreach (var onnxNode in model.Graph.Nodes)
            {
                // Skip no-ops
                if (IsNoOp(onnxNode.OpType))
                {
                    // Propagate slot: output = input
                    if (onnxNode.Inputs.Count > 0 && onnxNode.Outputs.Count > 0)
                    {
                        var inName = onnxNode.Inputs[0];
                        var outName = onnxNode.Outputs[0];

                        if (slotMap.TryGetValue(inName, out var slot))
                        {
                            slotMap[outName] = slot;
                        }
                    }

                    continue;
                }

                var module = OnnxOperatorMapper.MapNode(onnxNode, initializers, shapeContext);

                if (module == null)
                {
                    continue;
                }

                // Resolve input slots
                var inputSlots = ResolveInputSlots(
                    onnxNode, slotMap, initializers, shapeContext,
                    bufferSizes);

                // Allocate output slot
                var outputName = onnxNode.Outputs[0];
                var outputShape = shapeContext.GetShape(outputName);
                var outputSize2 = outputShape != null
                    ? ComputeSize(outputShape)
                    : bufferSizes[inputSlots[0]]; // fallback: same as input

                var outputSlot = bufferSizes.Count;
                bufferSizes.Add(outputSize2);
                slotMap[outputName] = outputSlot;

                execNodes.Add(new OnnxGraphNode(module, inputSlots, outputSlot, outputSize2));
            }

            // ── Allocate buffers ────────────────────────────────────────────
            var buffers = new float[bufferSizes.Count][];

            for (var i = 0; i < bufferSizes.Count; i++)
            {
                buffers[i] = new float[bufferSizes[i]];
            }

            return new OnnxGraphModel(
                execNodes.ToArray(),
                buffers,
                inputSize,
                outputSize);
        }

        // ─────────────────────────────────────────────────────────────────────
        // Private helpers
        // ─────────────────────────────────────────────────────────────────────

        private static int[] ResolveInputSlots(
            OnnxNode node,
            Dictionary<string, int> slotMap,
            Dictionary<string, OnnxTensor> initializers,
            OnnxShapeContext shapeContext,
            List<int> bufferSizes)
        {
            // For Add: two non-initializer inputs.
            if (node.OpType == "Add")
            {
                var slots = new int[2];

                for (var i = 0; i < 2; i++)
                {
                    var name = node.Inputs[i];

                    if (!slotMap.TryGetValue(name, out var slot))
                    {
                        throw new InvalidOperationException(
                            $"Add node input '{name}' not yet computed. " +
                            "Check that the ONNX graph is topologically sorted.");
                    }

                    slots[i] = slot;
                }

                return slots;
            }

            // Standard single-activation-input node:
            // Find the first non-initializer input (the activation).
            foreach (var inputName in node.Inputs)
            {
                if (string.IsNullOrEmpty(inputName))
                {
                    continue;
                }
                if (initializers.ContainsKey(inputName))
                {
                    continue;
                }

                if (slotMap.TryGetValue(inputName, out var slot))
                {
                    return [slot];
                }

                throw new InvalidOperationException(
                    $"Node '{node.Name}' (op={node.OpType}) input '{inputName}' " +
                    "not yet computed. Check that nodes are topologically sorted.");
            }

            // Fallback: use slot 0 (model input) — should not happen in valid models.
            return [0];
        }

        private static string GetPrimaryInputName(
            OnnxModel model,
            Dictionary<string, OnnxTensor> initializers)
        {
            foreach (var input in model.Graph.Inputs)
            {
                if (!initializers.ContainsKey(input.Name))
                {
                    return input.Name;
                }
            }

            throw new InvalidDataException(
                "ONNX graph has no non-initializer inputs (cannot determine model input tensor).");
        }

        private static bool IsNoOp(string opType)
            => opType is "Identity" or "Dropout";

        private static int ComputeSize(int[] shape)
        {
            var size = 1;
            foreach (var dim in shape)
            {
                size *= dim;
            }
            return size;
        }

        private static void ValidateOpsets(OnnxModel model)
        {
            foreach (var opset in model.OpsetImports)
            {
                if (!string.IsNullOrEmpty(opset.Domain) && opset.Domain != "ai.onnx")
                {
                    continue;
                }

                if (opset.Version < MinSupportedOpset || opset.Version > MaxSupportedOpset)
                {
                    throw new NotSupportedException(
                        $"ONNX opset version {opset.Version} not supported. " +
                        $"Tested range: {MinSupportedOpset}-{MaxSupportedOpset}.");
                }
            }
        }

        private static void ResolveExternalData(OnnxModel model, string? externalDataDir)
        {
            // Delegate to OnnxImporter's implementation (same assembly, internal access).
            // Reflection workaround: call Load which resolves internally, or duplicate.
            // To avoid duplication we just re-implement the minimal version here.
            if (externalDataDir == null)
            {
                return;
            }

            var fileCache = new Dictionary<string, byte[]>(StringComparer.OrdinalIgnoreCase);

            for (var i = 0; i < model.Graph.Initializers.Count; i++)
            {
                var init = model.Graph.Initializers[i];
                if (init.ExternalData == null)
                {
                    continue;
                }

                var fullPath = Path.GetFullPath(
                    Path.Combine(externalDataDir, init.ExternalData.Location));

                if (!fileCache.TryGetValue(fullPath, out var fileBytes))
                {
                    fileBytes = File.ReadAllBytes(fullPath);
                    fileCache[fullPath] = fileBytes;
                }

                var offset = (int)init.ExternalData.Offset;
                var length = init.ExternalData.Length > 0
                    ? (int)init.ExternalData.Length
                    : fileBytes.Length - offset;

                var raw = new byte[length];
                Buffer.BlockCopy(fileBytes, offset, raw, 0, length);

                model.Graph.Initializers[i] = new OnnxTensor
                {
                    Name = init.Name,
                    DataType = init.DataType,
                    Dims = init.Dims,
                    RawData = raw,
                    FloatData = init.FloatData,
                    Int64Data = init.Int64Data,
                    ExternalData = null,
                };
            }
        }

        private static Dictionary<string, OnnxTensor> BuildInitializerLookup(
            List<OnnxTensor> initializers)
        {
            var lookup = new Dictionary<string, OnnxTensor>(
                initializers.Count, StringComparer.Ordinal);

            foreach (var init in initializers)
            {
                lookup[init.Name] = init;
            }

            return lookup;
        }

        private static void SeedInputShapes(
            OnnxModel model,
            Dictionary<string, OnnxTensor> initializers,
            OnnxShapeContext ctx)
        {
            foreach (var input in model.Graph.Inputs)
            {
                if (initializers.ContainsKey(input.Name))
                {
                    continue;
                }

                if (input.Shape.Length > 0)
                {
                    var shape = new int[input.Shape.Length];
                    var valid = true;

                    for (var i = 0; i < input.Shape.Length; i++)
                    {
                        var v = input.Shape[i];
                        if (!v.HasValue || v.Value <= 0) { valid = false; break; }
                        shape[i] = (int)v.Value;
                    }

                    if (valid)
                    {
                        ctx.SetShape(input.Name, shape);
                    }
                }
            }
        }

        private static void SeedInitializerShapes(OnnxModel model, OnnxShapeContext ctx)
        {
            foreach (var init in model.Graph.Initializers)
            {
                var dims = new int[init.Dims.Length];
                for (var d = 0; d < init.Dims.Length; d++)
                {
                    dims[d] = (int)init.Dims[d];
                }
                ctx.SetShape(init.Name, dims);
            }
        }
    }
}