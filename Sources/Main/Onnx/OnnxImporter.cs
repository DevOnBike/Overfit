// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Onnx.Operators;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Onnx
{
    /// <summary>
    /// Loads PyTorch-exported ONNX models into Overfit's <see cref="Sequential"/>.
    ///
    /// Supported operators (MVP):
    ///   - Conv (2D, NCHW, no padding for now)
    ///   - Gemm (general matrix multiply, used as Linear by PyTorch)
    ///   - Relu, MaxPool, GlobalAveragePool
    ///   - Reshape, Flatten (no-ops in our linear pipeline)
    ///
    /// Limitations:
    ///   - Linear topology only (no skip connections)
    ///   - Float32 only
    ///   - Conv padding must be zero (PyTorch nn.Conv2d(padding=0))
    ///   - PyTorch model must be exported in eval() mode (BatchNorm folded into Conv)
    /// </summary>
    public static class OnnxImporter
    {
        /// <summary>
        /// Loads an ONNX model from disk. If the model uses external data (PyTorch ≥ 2.x default),
        /// the data file is automatically resolved relative to the .onnx file's directory.
        /// </summary>
        public static Sequential Load(string path)
        {
            var modelBytes = File.ReadAllBytes(path);
            var modelDir = Path.GetDirectoryName(Path.GetFullPath(path)) ?? "";
            return LoadFromBytes(modelBytes, modelDir);
        }

        /// <summary>
        /// Loads an ONNX model from a byte array. If <paramref name="externalDataDir"/> is provided,
        /// initializers with external_data references will be resolved from that directory.
        /// </summary>
        public static Sequential LoadFromBytes(byte[] modelBytes, string? externalDataDir = null)
        {
            var model = OnnxProtoParser.ParseModel(modelBytes);
            ValidateModel(model);
            ResolveExternalData(model, externalDataDir);

            var initializerLookup = BuildInitializerLookup(model.Graph.Initializers);
            var modules = new List<IModule>();

            // Build a lookup of input shapes propagating through the graph.
            // For MVP we trust the graph's declared input shape and let operators infer
            // their output shapes (Conv from kernel_shape, Pool from strides etc.)
            var shapeContext = new OnnxShapeContext();

            // Seed with graph inputs that have static shapes
            foreach (var graphInput in model.Graph.Inputs)
            {
                if (initializerLookup.ContainsKey(graphInput.Name))
                {
                    continue; // initializers are not real inputs in PyTorch exports
                }

                if (graphInput.Shape.All(d => d.HasValue && d.Value > 0))
                {
                    var shape = graphInput.Shape.Select(d => (int)d!.Value).ToArray();
                    shapeContext.SetShape(graphInput.Name, shape);
                }
            }

            foreach (var node in model.Graph.Nodes)
            {
                var module = OnnxOperatorMapper.MapNode(node, initializerLookup, shapeContext);

                if (module != null)
                {
                    modules.Add(module);
                }
                // null = no-op (Reshape, Flatten that's already handled by next layer)
            }

            if (modules.Count == 0)
            {
                throw new InvalidOperationException("ONNX import produced no modules. Check supported operator list.");
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
                    "ONNX model has no initializers (weights). Ensure the model was exported with export_params=True.");
            }

            // Opset version check - we target 11-20 (PyTorch ~2.x range)
            foreach (var opset in model.OpsetImports)
            {
                if (string.IsNullOrEmpty(opset.Domain) || opset.Domain == "ai.onnx")
                {
                    if (opset.Version < 11 || opset.Version > 20)
                    {
                        throw new NotSupportedException(
                            $"ONNX opset version {opset.Version} not supported. Tested range: 11-20.");
                    }
                }
            }

            // Detect branching topology - MVP requires linear graph
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

            var initializerNames = new HashSet<string>();
            foreach (var init in model.Graph.Initializers)
            {
                initializerNames.Add(init.Name);
            }

            foreach (var (output, count) in consumerCount)
            {
                if (count > 1 && !initializerNames.Contains(output))
                {
                    throw new NotSupportedException(
                        $"Branching topology detected: '{output}' has {count} consumers. " +
                        $"MVP supports linear graphs only. Skip connections / residual blocks not yet supported.");
                }
            }
        }

        /// <summary>
        /// Loads bytes referenced by external_data into in-memory RawData on each tensor.
        /// </summary>
        private static void ResolveExternalData(OnnxModel model, string? externalDataDir)
        {
            // Cache opened external files (often all initializers reference the same .data file)
            var fileCache = new Dictionary<string, byte[]>();

            for (var i = 0; i < model.Graph.Initializers.Count; i++)
            {
                var init = model.Graph.Initializers[i];
                if (init.ExternalData == null) continue;

                if (string.IsNullOrEmpty(externalDataDir))
                {
                    throw new InvalidOperationException(
                        $"Initializer '{init.Name}' references external data '{init.ExternalData.Location}', " +
                        $"but no externalDataDir was provided. Use OnnxImporter.Load(path) which auto-resolves.");
                }

                var fullPath = Path.Combine(externalDataDir, init.ExternalData.Location);

                if (!fileCache.TryGetValue(fullPath, out var fileBytes))
                {
                    if (!File.Exists(fullPath))
                    {
                        throw new FileNotFoundException(
                            $"External data file not found: {fullPath} (referenced by initializer '{init.Name}').");
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
                        $"External data for '{init.Name}' references bytes [{offset}, {offset + length}) " +
                        $"but file '{fullPath}' is only {fileBytes.Length} bytes.");
                }

                var raw = new byte[length];
                Buffer.BlockCopy(fileBytes, offset, raw, 0, length);

                // Replace the initializer with one containing inline raw data (records are immutable)
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

        private static Dictionary<string, OnnxTensor> BuildInitializerLookup(List<OnnxTensor> initializers)
        {
            var lookup = new Dictionary<string, OnnxTensor>(initializers.Count);
            foreach (var init in initializers)
            {
                lookup[init.Name] = init;
            }
            return lookup;
        }

        /// <summary>
        /// Decodes raw float32 bytes from an ONNX tensor.
        /// Handles both raw_data (most common) and float_data (legacy unpacked).
        /// </summary>
        internal static float[] DecodeFloatTensor(OnnxTensor tensor)
        {
            if (tensor.DataType != OnnxDataType.Float)
            {
                throw new NotSupportedException(
                    $"Tensor '{tensor.Name}' has type {tensor.DataType}, expected Float. " +
                    $"FP16/FP64/INT quantization not yet supported.");
            }

            if (tensor.FloatData != null && tensor.FloatData.Length > 0)
            {
                return tensor.FloatData;
            }

            if (tensor.RawData.Length == 0)
            {
                throw new InvalidDataException(
                    $"Tensor '{tensor.Name}' has no data (was external_data resolved?).");
            }

            if (tensor.RawData.Length % 4 != 0)
            {
                throw new InvalidDataException(
                    $"Tensor '{tensor.Name}' raw data length {tensor.RawData.Length} not divisible by 4.");
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

        /// <summary>
        /// Loads a float[] into a FastTensor with given shape.
        /// </summary>
        internal static FastTensor<float> LoadIntoFastTensor(float[] data, long[] dims, bool transpose2D = false)
        {
            if (dims.Length == 1)
            {
                var t = new FastTensor<float>((int)dims[0], clearMemory: false);
                data.AsSpan().CopyTo(t.GetView().AsSpan());
                return t;
            }

            if (dims.Length == 2)
            {
                if (transpose2D)
                {
                    int rows = (int)dims[0], cols = (int)dims[1];
                    var t = new FastTensor<float>(cols, rows, clearMemory: false);
                    var dst = t.GetView().AsSpan();
                    for (var r = 0; r < rows; r++)
                    {
                        for (var c = 0; c < cols; c++)
                        {
                            dst[c * rows + r] = data[r * cols + c];
                        }
                    }
                    return t;
                }
                else
                {
                    var t = new FastTensor<float>((int)dims[0], (int)dims[1], clearMemory: false);
                    data.AsSpan().CopyTo(t.GetView().AsSpan());
                    return t;
                }
            }

            if (dims.Length == 4)
            {
                var t = new FastTensor<float>((int)dims[0], (int)dims[1], (int)dims[2], (int)dims[3], clearMemory: false);
                data.AsSpan().CopyTo(t.GetView().AsSpan());
                return t;
            }

            throw new NotSupportedException($"Tensor rank {dims.Length} not supported (only 1D, 2D, 4D).");
        }
    }

    /// <summary>
    /// Tracks tensor shapes propagating through the ONNX graph during import.
    /// Operators read input shapes here and write output shapes for downstream nodes.
    /// </summary>
    internal sealed class OnnxShapeContext
    {
        private readonly Dictionary<string, int[]> _shapes = new();

        public void SetShape(string tensorName, int[] shape)
        {
            _shapes[tensorName] = shape;
        }

        public int[]? GetShape(string tensorName)
        {
            return _shapes.TryGetValue(tensorName, out var shape) ? shape : null;
        }
    }
}
