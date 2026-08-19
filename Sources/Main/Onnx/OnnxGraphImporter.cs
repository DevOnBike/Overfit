// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Onnx.Operators;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Onnx.Schema;
using DevOnBike.Overfit.Tensors.Core;

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

        // OVERFIT040 for `Load` — same constraint as `OnnxImporter.Load`, which this is the DAG counterpart
        // of: a whole `.onnx` file slurped once, at model-construction time, on the caller's own thread, with
        // no pool thread behind it; and `LoadFromBytes` immediately below already serves a caller who
        // obtained the bytes some other way.
        //
        // WHAT IS GIVEN UP: `Load` is public API of the shipped `DevOnBike.Overfit` package.
#pragma warning disable OVERFIT040
        public static OnnxGraphModel Load(string path, int inputSize, int outputSize)
        {
            var fullPath = Path.GetFullPath(path);
            var modelBytes = File.ReadAllBytes(fullPath);
            var modelDir = Path.GetDirectoryName(fullPath) ?? string.Empty;

            return LoadFromBytes(modelBytes, inputSize, outputSize, modelDir);
        }

        /// <summary>
        /// The same import with <c>Conv</c>-<c>Relu</c> fusion suppressed, for the arm that has to run beside
        /// the fused one.
        ///
        /// <para><see cref="FuseConvReluEnabled"/> is a <c>static readonly</c> read once per process, so a
        /// test cannot produce an unfused model by setting the environment variable and importing again —
        /// it would import the fused path a second time and compare it with itself. The parameter is the
        /// only honest way to hold both shapes in one process.</para>
        /// </summary>
        internal static OnnxGraphModel LoadUnfused(string path, int inputSize, int outputSize)
        {
            var fullPath = Path.GetFullPath(path);
            var modelBytes = File.ReadAllBytes(fullPath);
            var modelDir = Path.GetDirectoryName(fullPath) ?? string.Empty;

            return LoadFromBytes(modelBytes, inputSize, outputSize, modelDir, fuseConvRelu: false);
        }

        /// <summary>
        /// The fused counterpart of <see cref="LoadUnfused"/>, ignoring the environment switch.
        ///
        /// <para>A test that reached fusion through <see cref="Load"/> would assert on
        /// <see cref="FuseConvReluEnabled"/> rather than on the fusion pass, and would fail in the arm that
        /// turns the switch off — which it did, on the run that produced this method. The switch decides the
        /// default; these two entry points decide what the tests are about.</para>
        /// </summary>
        internal static OnnxGraphModel LoadFused(string path, int inputSize, int outputSize)
        {
            var fullPath = Path.GetFullPath(path);
            var modelBytes = File.ReadAllBytes(fullPath);
            var modelDir = Path.GetDirectoryName(fullPath) ?? string.Empty;

            return LoadFromBytes(modelBytes, inputSize, outputSize, modelDir, fuseConvRelu: true);
        }
#pragma warning restore OVERFIT040

        public static OnnxGraphModel LoadFromBytes(
            byte[] modelBytes,
            int inputSize,
            int outputSize,
            string? externalDataDir = null)
        {
            return LoadFromBytes(modelBytes, inputSize, outputSize, externalDataDir, FuseConvReluEnabled);
        }

        private static OnnxGraphModel LoadFromBytes(
            byte[] modelBytes,
            int inputSize,
            int outputSize,
            string? externalDataDir,
            bool fuseConvRelu)
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
                    if (onnxNode.Inputs.Count > 0 && onnxNode.Outputs.Count > 0)
                    {
                        var inName = onnxNode.Inputs[0];
                        var outName = onnxNode.Outputs[0];

                        var hasSlot = slotMap.TryGetValue(inName, out var slot);
                        var hasInitializer = initializers.TryGetValue(inName, out var initTensor);

                        if (hasSlot)
                        {
                            // Relabels an activation tensor: output reads from the same slot.
                            slotMap[outName] = slot;
                        }

                        // Test the out-value rather than the `hasInitializer` bool: routed through a separate
                        // variable the compiler loses the link to initTensor's null state (CS8601), which the
                        // AOT guard promotes to an error. Same condition, analysable.
                        if (!hasSlot && initTensor != null)
                        {
                            // Relabels a CONSTANT: a folded/deduplicated weight or bias routed to its
                            // consumer under a new name (e.g. torch's constant-folding aliases equal biases
                            // via Identity → "features.28.bias"). Alias the output to the same initializer so
                            // the downstream Conv/Gemm resolves it — otherwise the consumer throws KeyNotFound.
                            initializers[outName] = initTensor;
                            if (shapeContext.GetShape(outName) == null && shapeContext.GetShape(inName) is { } initShape)
                            {
                                shapeContext.SetShape(outName, initShape);
                            }
                        }
                    }

                    continue;
                }

                var module = OnnxOperatorMapper.MapNode(onnxNode, initializers, shapeContext);

                if (module == null)
                {
                    // A mapped operator can legitimately return null when it is a pure buffer relabel — e.g. a
                    // Reshape/Flatten whose output rank equals its input rank, so no data moves. Unlike the
                    // structural no-ops (Identity/Dropout) handled above, the slot must still be propagated so
                    // the downstream consumer can find this tensor; dropping it silently breaks the topology
                    // (a GlobalAveragePool → Flatten → Gemm head fails with "input not yet computed").
                    PropagateSlot(onnxNode, slotMap, initializers);
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

            // ── Fuse Conv → Relu ────────────────────────────────────────────
            if (fuseConvRelu)
            {
                FuseConvRelu(execNodes);
            }

            // ── Allocate buffers ────────────────────────────────────────────
            // TensorStorage rents from ArrayPool<T>.Shared (via its PooledBuffer<T> field)
            // — buffers are returned to the pool on OnnxGraphModel.Dispose(), avoiding GC pressure.
            var buffers = new TensorStorage<float>[bufferSizes.Count];

            for (var i = 0; i < bufferSizes.Count; i++)
            {
                buffers[i] = new TensorStorage<float>(bufferSizes[i]);
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

        /// <summary>
        /// Folds every <c>Relu</c> that consumes a convolution's output, and nothing else, into that
        /// convolution's epilogue — removing the node and rewiring its readers onto the convolution's slot.
        ///
        /// <para><b>Why, in the numbers that produced it.</b> Measured on VGG-16 on 2026-08-19: the fifteen
        /// <c>Relu</c> nodes cost 1.72 ms at 32 threads against 1.85 ms at 1. A 1.08x speedup invites the
        /// conclusion that they were never parallelised — <b>they are at 65.9 GB/s against this box's
        /// measured 90.8 GB/s DRAM ceiling</b>, so they are at 73% of the memory limit and threads cannot
        /// help. The operator is not slow; the second traversal of the tensor is the cost. Removing the
        /// traversal is the only lever that addresses it, and the convolution's bias epilogue already walks
        /// the same memory, so the clamp is free where it lands.</para>
        ///
        /// <para><b>The safety condition is that the convolution's output has exactly one reader.</b> A
        /// pre-activation tensor consumed by a skip connection as well as by the <c>Relu</c> must not be
        /// clamped in place — the other consumer would silently read activated values. Readers are counted
        /// across every node's input slots, <c>Add</c>'s two included, rather than assumed from the graph's
        /// shape.</para>
        /// </summary>
        internal static readonly bool FuseConvReluEnabled =
            Environment.GetEnvironmentVariable(OverfitEnvironment.FuseConvRelu) != "0";

        internal static void FuseConvRelu(List<OnnxGraphNode> nodes)
        {
            var index = 0;

            while (index < nodes.Count - 1)
            {
                if (!CanFuseConvRelu(nodes, index))
                {
                    index++;

                    continue;
                }

                var conv = (ConvLayer)nodes[index].Module;
                var convSlot = nodes[index].OutputSlot;
                var reluSlot = nodes[index + 1].OutputSlot;

                conv.FusedRelu = true;
                nodes.RemoveAt(index + 1);

                // Everything that read the Relu's output now reads the convolution's, which holds the
                // activated values. Rewritten in place because InputSlots is the array the run loop indexes.
                foreach (var node in nodes)
                {
                    for (var i = 0; i < node.InputSlots.Length; i++)
                    {
                        if (node.InputSlots[i] == reluSlot)
                        {
                            node.InputSlots[i] = convSlot;
                        }
                    }
                }

                // Deliberately no index++: the node that followed the Relu is now at index + 1, and a
                // Conv → Relu → Relu chain would otherwise be half-fused.
            }
        }

        /// <summary>
        /// Whether the node at <paramref name="index"/> is a convolution whose output is consumed by the
        /// immediately following <c>Relu</c> and by nothing else.
        /// </summary>
        internal static bool CanFuseConvRelu(List<OnnxGraphNode> nodes, int index)
        {
            if (nodes[index].Module is not ConvLayer conv || conv.FusedRelu)
            {
                return false;
            }

            var relu = nodes[index + 1];

            if (relu.Module is not ReluActivation || relu.InputSlots.Length != 1)
            {
                return false;
            }

            var convSlot = nodes[index].OutputSlot;

            if (relu.InputSlots[0] != convSlot)
            {
                return false;
            }

            var readers = 0;

            foreach (var node in nodes)
            {
                foreach (var slot in node.InputSlots)
                {
                    if (slot == convSlot)
                    {
                        readers++;
                    }
                }
            }

            // The Relu is one of them; a second reader means the pre-activation tensor is live elsewhere.
            return readers == 1;
        }

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
                        throw new OverfitRuntimeException(
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

                throw new OverfitRuntimeException(
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

            throw new OverfitFormatException(
                "ONNX graph has no non-initializer inputs (cannot determine model input tensor).");
        }

        private static bool IsNoOp(string opType)
            => opType is "Identity" or "Dropout";

        // Aliases a node's output tensor to its activation input's slot, used when the operator carries no
        // data movement (returned a null module). Picks the first non-initializer input that already has a
        // slot — the same activation-input rule ResolveInputSlots uses.
        private static void PropagateSlot(
            OnnxNode node,
            Dictionary<string, int> slotMap,
            Dictionary<string, OnnxTensor> initializers)
        {
            if (node.Outputs.Count == 0)
            {
                return;
            }

            foreach (var inputName in node.Inputs)
            {
                if (string.IsNullOrEmpty(inputName) || initializers.ContainsKey(inputName))
                {
                    continue;
                }

                if (slotMap.TryGetValue(inputName, out var slot))
                {
                    slotMap[node.Outputs[0]] = slot;
                    return;
                }
            }
        }

        /// <summary>
        /// Elements a shape describes, refusing a product that cannot be one.
        ///
        /// <para>The shape comes from the model file. Multiplied unchecked it wraps to something small,
        /// positive and plausible, which then sizes a buffer that every later read overruns — or passes a
        /// shape comparison the real layout fails. The same unchecked product existed in
        /// <c>OnnxTensor.ElementCount</c> and <c>GgufTensorInfo.ElementCount</c>.</para>
        /// </summary>
        private static int ComputeSize(int[] shape)
        {
            long size = 1;

            foreach (var dim in shape)
            {
                if (dim < 0)
                {
                    throw new OverfitFormatException($"Shape declares a negative dimension ({dim}).");
                }

                size *= dim;

                if (size > int.MaxValue)
                {
                    throw new OverfitFormatException(
                        $"Shape declares {size} or more elements, which exceeds the {int.MaxValue} this "
                        + "runtime can address.");
                }
            }

            return (int)size;
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
                    throw new OverfitRuntimeException(
                        $"ONNX opset version {opset.Version} not supported. " +
                        $"Tested range: {MinSupportedOpset}-{MaxSupportedOpset}.");
                }
            }
        }

        /// <summary>
        /// Shared with <see cref="OnnxImporter"/>, and the sharing is the fix.
        ///
        /// <para>This method used to hold a second implementation whose own comment said it was
        /// "the minimal version" — minimal meaning without the path validation, the checked narrowing or
        /// the bounds test its sibling has. See <see cref="OnnxExternalData"/> for what each of those
        /// closes.</para>
        /// </summary>
        private static void ResolveExternalData(OnnxModel model, string? externalDataDir)
        {
            OnnxExternalData.Resolve(model, externalDataDir);
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
                        if (!v.HasValue || v.Value <= 0)
                        {
                            valid = false;
                            break;
                        }
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