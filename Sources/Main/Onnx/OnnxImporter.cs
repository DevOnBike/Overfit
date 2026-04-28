// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using System.Linq;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Onnx.Operators;
using DevOnBike.Overfit.Onnx.Schema;

namespace DevOnBike.Overfit.Onnx
{
    /// <summary>
    /// Loads PyTorch-exported ONNX models into Overfit's <see cref="Sequential"/>.
    ///
    /// Supported operators (MVP):
    ///   - Gemm  → LinearLayer
    ///   - Conv  → ConvLayer  (padding=0, stride=1, dilation=1, group=1 only)
    ///   - Relu  → ReluActivation
    ///   - MaxPool → MaxPool2DLayer
    ///   - Reshape / Flatten → FlattenLayer (when rank decreases to 2)
    ///
    /// Limitations:
    ///   - Linear topology only (no skip connections, no branching DAG)
    ///   - Float32 weights only
    ///   - Model must be exported in eval() mode so BatchNorm is folded into Conv
    /// </summary>
    public static class OnnxImporter
    {
        /// <summary>
        /// Loads an ONNX model from disk.
        /// External data files (PyTorch ≥ 2.x default) are resolved automatically
        /// relative to the .onnx file's directory.
        /// </summary>
        public static Sequential Load(string path)
        {
            var modelBytes = File.ReadAllBytes(path);
            var modelDir = Path.GetDirectoryName(Path.GetFullPath(path)) ?? "";
            return LoadFromBytes(modelBytes, modelDir);
        }

        /// <summary>
        /// Loads an ONNX model from a raw byte array.
        /// Pass <paramref name="externalDataDir"/> when initializers reference external files.
        /// </summary>
        public static Sequential LoadFromBytes(byte[] modelBytes, string? externalDataDir = null)
        {
            var model = OnnxProtoParser.ParseModel(modelBytes);
            ValidateModel(model);
            ResolveExternalData(model, externalDataDir);

            var initializerLookup = BuildInitializerLookup(model.Graph.Initializers);
            var shapeContext = new OnnxShapeContext();
            var modules = new List<IModule>();

            // Seed shape context from graph inputs with fully static shapes.
            foreach (var graphInput in model.Graph.Inputs)
            {
                if (initializerLookup.ContainsKey(graphInput.Name))
                {
                    continue; // PyTorch exports weights as both graph inputs and initializers
                }

                if (graphInput.Shape.All(d => d.HasValue && d.Value > 0))
                {
                    var shape = graphInput.Shape.Select(d => (int)d!.Value).ToArray();
                    shapeContext.SetShape(graphInput.Name, shape);
                }
            }

            // Seed shape context from initializers so operators can look up weight dims.
            foreach (var init in model.Graph.Initializers)
            {
                shapeContext.SetShape(init.Name, init.Dims.Select(d => (int)d).ToArray());
            }

            foreach (var node in model.Graph.Nodes)
            {
                var module = OnnxOperatorMapper.MapNode(node, initializerLookup, shapeContext);

                if (module != null)
                {
                    modules.Add(module);
                }
                // null means structural no-op (e.g., Reshape that doesn't change element layout)
            }

            if (modules.Count == 0)
            {
                throw new InvalidOperationException(
                    "ONNX import produced no modules. Check that the model contains supported operators.");
            }

            return new Sequential(modules.ToArray());
        }

        private static void ValidateModel(OnnxModel model)
        {
            if (model.Graph.Nodes.Count == 0)
            {
                throw new InvalidOperationException("ONNX model has no nodes.");
            }

            if (model.Graph.Initializers.Count == 0)
            {
                throw new InvalidOperationException(
                    "ONNX model has no initializers (weights). " +
                    "Ensure the model was exported with export_params=True.");
            }

            foreach (var opset in model.OpsetImports)
            {
                if (string.IsNullOrEmpty(opset.Domain) || opset.Domain == "ai.onnx")
                {
                    if (opset.Version is < 11 or > 20)
                    {
                        throw new NotSupportedException(
                            $"ONNX opset version {opset.Version} not supported. Tested range: 11-20.");
                    }
                }
            }

            // MVP requires linear graph — reject branching DAGs.
            var consumerCount = new Dictionary<string, int>();
            foreach (var node in model.Graph.Nodes)
            {
                foreach (var input in node.Inputs)
                {
                    if (string.IsNullOrEmpty(input)) continue;
                    consumerCount.TryGetValue(input, out var c);
                    consumerCount[input] = c + 1;
                }
            }

            var initializerNames = new HashSet<string>(
                model.Graph.Initializers.Select(i => i.Name));

            foreach (var (output, count) in consumerCount)
            {
                if (count > 1 && !initializerNames.Contains(output))
                {
                    throw new NotSupportedException(
                        $"Branching topology detected: tensor '{output}' has {count} consumers. " +
                        "MVP supports linear graphs only. Skip connections not yet supported.");
                }
            }
        }

        private static void ResolveExternalData(OnnxModel model, string? externalDataDir)
        {
            var fileCache = new Dictionary<string, byte[]>();

            for (var i = 0; i < model.Graph.Initializers.Count; i++)
            {
                var init = model.Graph.Initializers[i];
                if (init.ExternalData == null) continue;

                if (string.IsNullOrEmpty(externalDataDir))
                {
                    throw new InvalidOperationException(
                        $"Initializer '{init.Name}' references external data " +
                        $"'{init.ExternalData.Location}', but no externalDataDir was provided. " +
                        "Use OnnxImporter.Load(path) which resolves it automatically.");
                }

                var fullPath = Path.Combine(externalDataDir, init.ExternalData.Location);

                if (!fileCache.TryGetValue(fullPath, out var fileBytes))
                {
                    if (!File.Exists(fullPath))
                    {
                        throw new FileNotFoundException(
                            $"External data file not found: {fullPath} " +
                            $"(referenced by initializer '{init.Name}').");
                    }

                    fileBytes = File.ReadAllBytes(fullPath);
                    fileCache[fullPath] = fileBytes;
                }

                var offset = (int)init.ExternalData.Offset;
                var length = init.ExternalData.Length > 0
                    ? (int)init.ExternalData.Length
                    : fileBytes.Length - offset;

                if (offset + length > fileBytes.Length)
                {
                    throw new InvalidDataException(
                        $"External data for '{init.Name}' requests bytes [{offset}, {offset + length}) " +
                        $"but '{Path.GetFileName(fullPath)}' is only {fileBytes.Length} bytes.");
                }

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
            var lookup = new Dictionary<string, OnnxTensor>(initializers.Count);
            foreach (var init in initializers)
            {
                lookup[init.Name] = init;
            }
            return lookup;
        }

        /// <summary>
        /// Decodes raw float32 bytes from an ONNX tensor initializer.
        /// Handles both raw_data (packed bytes, most common) and float_data (legacy varint).
        /// </summary>
        internal static float[] DecodeFloatTensor(OnnxTensor tensor)
        {
            if (tensor.DataType != OnnxDataType.Float)
            {
                throw new NotSupportedException(
                    $"Tensor '{tensor.Name}' has type {tensor.DataType}. " +
                    "Only Float32 is supported; FP16/INT8 quantization not yet implemented.");
            }

            if (tensor.FloatData is { Length: > 0 })
            {
                return tensor.FloatData;
            }

            if (tensor.RawData.Length == 0)
            {
                throw new InvalidDataException(
                    $"Tensor '{tensor.Name}' has no data. Was external_data resolved?");
            }

            if (tensor.RawData.Length % 4 != 0)
            {
                throw new InvalidDataException(
                    $"Tensor '{tensor.Name}' raw data length {tensor.RawData.Length} " +
                    "is not divisible by 4.");
            }

            var count = tensor.RawData.Length / 4;
            var result = new float[count];
            var span = tensor.RawData.AsSpan();

            for (var i = 0; i < count; i++)
            {
                result[i] = BitConverter.UInt32BitsToSingle(
                    BinaryPrimitives.ReadUInt32LittleEndian(span.Slice(i * 4, 4)));
            }

            return result;
        }
    }
}
