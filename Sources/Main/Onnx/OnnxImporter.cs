// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Onnx.Operators;
using DevOnBike.Overfit.Onnx.Schema;

namespace DevOnBike.Overfit.Onnx
{
    /// <summary>
    /// Loads PyTorch-exported ONNX models into Overfit's Sequential.
    ///
    /// Supported operators (MVP):
    /// - Gemm -> LinearLayer
    /// - Conv -> ConvLayer (padding=0, stride=1, dilation=1, group=1 only)
    /// - Relu -> ReluActivation
    /// - MaxPool -> MaxPool2DLayer
    /// - Reshape / Flatten -> FlattenLayer when rank decreases to 2
    ///
    /// Limitations:
    /// - Linear topology only. No skip connections, no branching DAG.
    /// - Float32 weights only.
    /// - Model should be exported in eval() mode so BatchNorm is folded into Conv.
    /// </summary>
    public static class OnnxImporter
    {
        private const int MinSupportedOpset = 11;
        private const int MaxSupportedOpset = 20;

        /// <summary>
        /// Loads an ONNX model from disk.
        /// External data files are resolved relative to the .onnx file directory.
        /// </summary>
        public static Sequential Load(
            string path)
        {
            if (string.IsNullOrWhiteSpace(path))
            {
                throw new ArgumentException(
                    "ONNX model path cannot be null or empty.",
                    nameof(path));
            }

            var fullPath = Path.GetFullPath(path);
            var modelBytes = File.ReadAllBytes(fullPath);
            var modelDir = Path.GetDirectoryName(fullPath) ?? string.Empty;

            return LoadFromBytes(
                modelBytes,
                modelDir);
        }

        /// <summary>
        /// Loads an ONNX model from a raw byte array.
        /// Pass externalDataDir when initializers reference external files.
        /// </summary>
        public static Sequential LoadFromBytes(
            byte[] modelBytes,
            string? externalDataDir = null)
        {
            if (modelBytes == null)
            {
                throw new ArgumentNullException(nameof(modelBytes));
            }

            if (modelBytes.Length == 0)
            {
                throw new ArgumentException(
                    "ONNX model byte array cannot be empty.",
                    nameof(modelBytes));
            }

            var model = OnnxProtoParser.ParseModel(modelBytes);

            ValidateModel(model);
            ResolveExternalData(
                model,
                externalDataDir);

            var initializerLookup = BuildInitializerLookup(
                model.Graph.Initializers);

            var shapeContext = new OnnxShapeContext();
            var modules = new List<IModule>();

            SeedGraphInputShapes(
                model,
                initializerLookup,
                shapeContext);

            SeedInitializerShapes(
                model,
                shapeContext);

            for (var i = 0; i < model.Graph.Nodes.Count; i++)
            {
                var node = model.Graph.Nodes[i];

                var module = OnnxOperatorMapper.MapNode(
                    node,
                    initializerLookup,
                    shapeContext);

                if (module != null)
                {
                    modules.Add(module);
                }
            }

            if (modules.Count == 0)
            {
                throw new InvalidOperationException(
                    "ONNX import produced no modules. " +
                    "Check that the model contains supported operators.");
            }

            return new Sequential(
                modules.ToArray());
        }

        private static void SeedGraphInputShapes(
            OnnxModel model,
            Dictionary<string, OnnxTensor> initializerLookup,
            OnnxShapeContext shapeContext)
        {
            for (var i = 0; i < model.Graph.Inputs.Count; i++)
            {
                var graphInput = model.Graph.Inputs[i];

                // PyTorch often exports weights as both graph inputs and initializers.
                if (initializerLookup.ContainsKey(graphInput.Name))
                {
                    continue;
                }

                if (!TryCreateStaticShape(
                    graphInput.Shape,
                    out var shape))
                {
                    continue;
                }

                shapeContext.SetShape(
                    graphInput.Name,
                    shape);
            }
        }

        private static void SeedInitializerShapes(
            OnnxModel model,
            OnnxShapeContext shapeContext)
        {
            for (var i = 0; i < model.Graph.Initializers.Count; i++)
            {
                var initializer = model.Graph.Initializers[i];

                var dims = new int[initializer.Dims.Length];

                for (var d = 0; d < initializer.Dims.Length; d++)
                {
                    dims[d] = CheckedToInt32(
                        initializer.Dims[d],
                        $"Initializer '{initializer.Name}' dimension {d}");
                }

                shapeContext.SetShape(
                    initializer.Name,
                    dims);
            }
        }

        private static bool TryCreateStaticShape(
            IReadOnlyList<long?> source,
            out int[] shape)
        {
            shape = Array.Empty<int>();

            if (source.Count == 0)
            {
                return false;
            }

            var result = new int[source.Count];

            for (var i = 0; i < source.Count; i++)
            {
                var value = source[i];

                if (!value.HasValue || value.Value <= 0)
                {
                    return false;
                }

                result[i] = CheckedToInt32(
                    value.Value,
                    $"Shape dimension {i}");
            }

            shape = result;
            return true;
        }

        private static void ValidateModel(
            OnnxModel model)
        {
            if (model.Graph.Nodes.Count == 0)
            {
                throw new InvalidOperationException(
                    "ONNX model has no nodes.");
            }

            if (model.Graph.Initializers.Count == 0)
            {
                throw new InvalidOperationException(
                    "ONNX model has no initializers (weights). " +
                    "Ensure the model was exported with export_params=True.");
            }

            ValidateOpsets(model);
            ValidateLinearTopology(model);
        }

        private static void ValidateOpsets(
            OnnxModel model)
        {
            for (var i = 0; i < model.OpsetImports.Count; i++)
            {
                var opset = model.OpsetImports[i];

                if (!string.IsNullOrEmpty(opset.Domain) &&
                    opset.Domain != "ai.onnx")
                {
                    continue;
                }

                if (opset.Version < MinSupportedOpset ||
                    opset.Version > MaxSupportedOpset)
                {
                    throw new NotSupportedException(
                        $"ONNX opset version {opset.Version} not supported. " +
                        $"Tested range: {MinSupportedOpset}-{MaxSupportedOpset}.");
                }
            }
        }

        private static void ValidateLinearTopology(
            OnnxModel model)
        {
            var consumerCount = new Dictionary<string, int>();

            for (var nodeIndex = 0; nodeIndex < model.Graph.Nodes.Count; nodeIndex++)
            {
                var node = model.Graph.Nodes[nodeIndex];

                for (var inputIndex = 0; inputIndex < node.Inputs.Count; inputIndex++)
                {
                    var input = node.Inputs[inputIndex];

                    if (string.IsNullOrEmpty(input))
                    {
                        continue;
                    }

                    consumerCount.TryGetValue(
                        input,
                        out var count);

                    consumerCount[input] = count + 1;
                }
            }

            var initializerNames = new HashSet<string>(
                StringComparer.Ordinal);

            for (var i = 0; i < model.Graph.Initializers.Count; i++)
            {
                initializerNames.Add(
                    model.Graph.Initializers[i].Name);
            }

            foreach (var pair in consumerCount)
            {
                var tensorName = pair.Key;
                var count = pair.Value;

                if (count <= 1)
                {
                    continue;
                }

                if (initializerNames.Contains(tensorName))
                {
                    continue;
                }

                throw new NotSupportedException(
                    $"Branching topology detected: tensor '{tensorName}' has {count} consumers. " +
                    "MVP supports linear graphs only. Skip connections are not yet supported.");
            }
        }

        private static void ResolveExternalData(
            OnnxModel model,
            string? externalDataDir)
        {
            var fileCache = new Dictionary<string, byte[]>(
                StringComparer.OrdinalIgnoreCase);

            for (var i = 0; i < model.Graph.Initializers.Count; i++)
            {
                var initializer = model.Graph.Initializers[i];

                if (initializer.ExternalData == null)
                {
                    continue;
                }

                if (string.IsNullOrEmpty(externalDataDir))
                {
                    throw new InvalidOperationException(
                        $"Initializer '{initializer.Name}' references external data " +
                        $"'{initializer.ExternalData.Location}', but no externalDataDir was provided. " +
                        "Use OnnxImporter.Load(path), which resolves it automatically.");
                }

                var fullPath = ResolveExternalDataPath(
                    externalDataDir,
                    initializer.ExternalData.Location,
                    initializer.Name);

                if (!fileCache.TryGetValue(
                        fullPath,
                        out var fileBytes))
                {
                    if (!File.Exists(fullPath))
                    {
                        throw new FileNotFoundException(
                            $"External data file not found: {fullPath} " +
                            $"(referenced by initializer '{initializer.Name}').",
                            fullPath);
                    }

                    fileBytes = File.ReadAllBytes(fullPath);
                    fileCache[fullPath] = fileBytes;
                }

                var offset = CheckedToInt32(
                    initializer.ExternalData.Offset,
                    $"External data offset for initializer '{initializer.Name}'");

                var length = GetExternalDataLength(
                    initializer,
                    fileBytes.Length,
                    offset);

                if (offset < 0 ||
                    length < 0 ||
                    offset > fileBytes.Length ||
                    length > fileBytes.Length - offset)
                {
                    throw new InvalidDataException(
                        $"External data for '{initializer.Name}' requests bytes " +
                        $"[{offset}, {offset + length}) but '{Path.GetFileName(fullPath)}' " +
                        $"is only {fileBytes.Length} bytes.");
                }

                var raw = new byte[length];

                Buffer.BlockCopy(
                    fileBytes,
                    offset,
                    raw,
                    0,
                    length);

                model.Graph.Initializers[i] = new OnnxTensor
                {
                    Name = initializer.Name,
                    DataType = initializer.DataType,
                    Dims = initializer.Dims,
                    RawData = raw,
                    FloatData = initializer.FloatData,
                    Int64Data = initializer.Int64Data,
                    ExternalData = null,
                };
            }
        }

        private static string ResolveExternalDataPath(
            string externalDataDir,
            string location,
            string initializerName)
        {
            if (string.IsNullOrWhiteSpace(location))
            {
                throw new InvalidDataException(
                    $"Initializer '{initializerName}' has an empty external data location.");
            }

            if (Path.IsPathRooted(location))
            {
                throw new InvalidDataException(
                    $"Initializer '{initializerName}' references an absolute external data path: '{location}'.");
            }

            var baseDir = Path.GetFullPath(externalDataDir);
            var fullPath = Path.GetFullPath(
                Path.Combine(
                    baseDir,
                    location));

            var comparison = OperatingSystem.IsWindows()
                ? StringComparison.OrdinalIgnoreCase
                : StringComparison.Ordinal;

            if (!IsPathInsideDirectory(
                    fullPath,
                    baseDir,
                    comparison))
            {
                throw new InvalidDataException(
                    $"External data path for initializer '{initializerName}' escapes the model directory: '{location}'.");
            }

            return fullPath;
        }

        private static bool IsPathInsideDirectory(
            string path,
            string directory,
            StringComparison comparison)
        {
            var normalizedDirectory = directory;

            if (!normalizedDirectory.EndsWith(
                    Path.DirectorySeparatorChar.ToString(),
                    comparison))
            {
                normalizedDirectory += Path.DirectorySeparatorChar;
            }

            return path.StartsWith(
                normalizedDirectory,
                comparison);
        }

        private static int GetExternalDataLength(
            OnnxTensor initializer,
            int fileLength,
            int offset)
        {
            if (initializer.ExternalData == null)
            {
                throw new InvalidOperationException(
                    "ExternalData must be present.");
            }

            if (initializer.ExternalData.Length == 0)
            {
                return fileLength - offset;
            }

            return CheckedToInt32(
                initializer.ExternalData.Length,
                $"External data length for initializer '{initializer.Name}'");
        }

        private static Dictionary<string, OnnxTensor> BuildInitializerLookup(
            List<OnnxTensor> initializers)
        {
            var lookup = new Dictionary<string, OnnxTensor>(
                initializers.Count,
                StringComparer.Ordinal);

            for (var i = 0; i < initializers.Count; i++)
            {
                var initializer = initializers[i];
                lookup[initializer.Name] = initializer;
            }

            return lookup;
        }

        /// <summary>
        /// Decodes raw float32 bytes from an ONNX tensor initializer.
        /// Handles both raw_data and float_data.
        /// </summary>
        internal static float[] DecodeFloatTensor(
            OnnxTensor tensor)
        {
            if (tensor.DataType != OnnxDataType.Float)
            {
                throw new NotSupportedException(
                    $"Tensor '{tensor.Name}' has type {tensor.DataType}. " +
                    "Only Float32 is supported; FP16/INT8 quantization is not yet implemented.");
            }

            if (tensor.FloatData != null &&
                tensor.FloatData.Length > 0)
            {
                return tensor.FloatData;
            }

            if (tensor.RawData.Length == 0)
            {
                throw new InvalidDataException(
                    $"Tensor '{tensor.Name}' has no data. Was external_data resolved?");
            }

            if (tensor.RawData.Length % sizeof(float) != 0)
            {
                throw new InvalidDataException(
                    $"Tensor '{tensor.Name}' raw data length {tensor.RawData.Length} " +
                    "is not divisible by 4.");
            }

            var count = tensor.RawData.Length / sizeof(float);
            var result = new float[count];
            var span = tensor.RawData.AsSpan();

            for (var i = 0; i < count; i++)
            {
                result[i] = BitConverter.UInt32BitsToSingle(
                    BinaryPrimitives.ReadUInt32LittleEndian(
                        span.Slice(
                            i * sizeof(float),
                            sizeof(float))));
            }

            return result;
        }

        private static int CheckedToInt32(
            long value,
            string name)
        {
            if (value < int.MinValue ||
                value > int.MaxValue)
            {
                throw new NotSupportedException(
                    $"{name} value {value} is outside Int32 range.");
            }

            return (int)value;
        }
    }
}