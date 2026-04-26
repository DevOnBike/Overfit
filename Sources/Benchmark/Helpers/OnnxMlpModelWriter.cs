// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using System.Text;

namespace Benchmarks.Helpers
{
    internal static class OnnxMlpModelWriter
    {
        private const int TensorProtoDataTypeFloat = 1;

        public static void WriteLinearReluLinearModel(
            string path,
            int inputSize,
            int hiddenSize,
            int outputSize,
            float[] w1InputHidden,
            float[] b1,
            float[] w2HiddenOutput,
            float[] b2)
        {
            if (w1InputHidden.Length != inputSize * hiddenSize)
            {
                throw new ArgumentException("W1 length mismatch.", nameof(w1InputHidden));
            }

            if (b1.Length != hiddenSize)
            {
                throw new ArgumentException("B1 length mismatch.", nameof(b1));
            }

            if (w2HiddenOutput.Length != hiddenSize * outputSize)
            {
                throw new ArgumentException("W2 length mismatch.", nameof(w2HiddenOutput));
            }

            if (b2.Length != outputSize)
            {
                throw new ArgumentException("B2 length mismatch.", nameof(b2));
            }

            using var file = File.Create(path);

            WriteModel(
                file,
                inputSize,
                hiddenSize,
                outputSize,
                w1InputHidden,
                b1,
                w2HiddenOutput,
                b2);
        }

        private static void WriteModel(
            Stream stream,
            int inputSize,
            int hiddenSize,
            int outputSize,
            float[] w1InputHidden,
            float[] b1,
            float[] w2HiddenOutput,
            float[] b2)
        {
            // ModelProto
            Proto.WriteInt64(stream, 1, 9); // ir_version
            Proto.WriteString(stream, 2, "DevOnBike.Overfit.Benchmark"); // producer_name

            Proto.WriteMessage(stream, 7, graph =>
            {
                WriteGraph(
                    graph,
                    inputSize,
                    hiddenSize,
                    outputSize,
                    w1InputHidden,
                    b1,
                    w2HiddenOutput,
                    b2);
            });

            Proto.WriteMessage(stream, 8, opset =>
            {
                // default ONNX domain, opset 13
                Proto.WriteInt64(opset, 2, 13);
            });
        }

        private static void WriteGraph(
            Stream stream,
            int inputSize,
            int hiddenSize,
            int outputSize,
            float[] w1InputHidden,
            float[] b1,
            float[] w2HiddenOutput,
            float[] b2)
        {
            // Node 1: hidden_linear = Gemm(input, W1, B1)
            Proto.WriteMessage(stream, 1, node =>
            {
                Proto.WriteString(node, 1, "input");
                Proto.WriteString(node, 1, "W1");
                Proto.WriteString(node, 1, "B1");

                Proto.WriteString(node, 2, "hidden_linear");

                Proto.WriteString(node, 3, "linear1_gemm");
                Proto.WriteString(node, 4, "Gemm");
            });

            // Node 2: hidden_relu = Relu(hidden_linear)
            Proto.WriteMessage(stream, 1, node =>
            {
                Proto.WriteString(node, 1, "hidden_linear");

                Proto.WriteString(node, 2, "hidden_relu");

                Proto.WriteString(node, 3, "relu");
                Proto.WriteString(node, 4, "Relu");
            });

            // Node 3: output = Gemm(hidden_relu, W2, B2)
            Proto.WriteMessage(stream, 1, node =>
            {
                Proto.WriteString(node, 1, "hidden_relu");
                Proto.WriteString(node, 1, "W2");
                Proto.WriteString(node, 1, "B2");

                Proto.WriteString(node, 2, "output");

                Proto.WriteString(node, 3, "linear2_gemm");
                Proto.WriteString(node, 4, "Gemm");
            });

            Proto.WriteString(stream, 2, "overfit_mlp_graph");

            // Initializers
            WriteFloatTensor(
                stream,
                fieldNumber: 5,
                name: "W1",
                dims: [inputSize, hiddenSize],
                data: w1InputHidden);

            WriteFloatTensor(
                stream,
                fieldNumber: 5,
                name: "B1",
                dims: [hiddenSize],
                data: b1);

            WriteFloatTensor(
                stream,
                fieldNumber: 5,
                name: "W2",
                dims: [hiddenSize, outputSize],
                data: w2HiddenOutput);

            WriteFloatTensor(
                stream,
                fieldNumber: 5,
                name: "B2",
                dims: [outputSize],
                data: b2);

            // Graph input/output
            WriteValueInfo(
                stream,
                fieldNumber: 11,
                name: "input",
                dims: [1, inputSize]);

            WriteValueInfo(
                stream,
                fieldNumber: 12,
                name: "output",
                dims: [1, outputSize]);
        }

        private static void WriteFloatTensor(
            Stream stream,
            int fieldNumber,
            string name,
            long[] dims,
            float[] data)
        {
            Proto.WriteMessage(stream, fieldNumber, tensor =>
            {
                for (var i = 0; i < dims.Length; i++)
                {
                    Proto.WriteInt64(tensor, 1, dims[i]);
                }

                Proto.WriteInt32(tensor, 2, TensorProtoDataTypeFloat);
                Proto.WriteString(tensor, 8, name);

                var raw = new byte[data.Length * sizeof(float)];

                for (var i = 0; i < data.Length; i++)
                {
                    BinaryPrimitives.WriteUInt32LittleEndian(
                        raw.AsSpan(i * sizeof(float), sizeof(float)),
                        BitConverter.SingleToUInt32Bits(data[i]));
                }

                Proto.WriteBytes(tensor, 9, raw);
            });
        }

        private static void WriteValueInfo(
            Stream stream,
            int fieldNumber,
            string name,
            long[] dims)
        {
            Proto.WriteMessage(stream, fieldNumber, valueInfo =>
            {
                Proto.WriteString(valueInfo, 1, name);

                // TypeProto
                Proto.WriteMessage(valueInfo, 2, typeProto =>
                {
                    // TypeProto.tensor_type
                    Proto.WriteMessage(typeProto, 1, tensorType =>
                    {
                        Proto.WriteInt32(tensorType, 1, TensorProtoDataTypeFloat);

                        // TensorShapeProto
                        Proto.WriteMessage(tensorType, 2, shape =>
                        {
                            for (var i = 0; i < dims.Length; i++)
                            {
                                var dimValue = dims[i];

                                // TensorShapeProto.Dimension
                                Proto.WriteMessage(shape, 1, dim =>
                                {
                                    Proto.WriteInt64(dim, 1, dimValue);
                                });
                            }
                        });
                    });
                });
            });
        }

        private static class Proto
        {
            private const int WireTypeVarint = 0;
            private const int WireTypeLengthDelimited = 2;

            public static void WriteInt32(Stream stream, int fieldNumber, int value)
            {
                WriteTag(stream, fieldNumber, WireTypeVarint);
                WriteVarUInt64(stream, unchecked((ulong)value));
            }

            public static void WriteInt64(Stream stream, int fieldNumber, long value)
            {
                WriteTag(stream, fieldNumber, WireTypeVarint);
                WriteVarUInt64(stream, unchecked((ulong)value));
            }

            public static void WriteString(Stream stream, int fieldNumber, string value)
            {
                WriteBytes(stream, fieldNumber, Encoding.UTF8.GetBytes(value));
            }

            public static void WriteBytes(Stream stream, int fieldNumber, byte[] value)
            {
                WriteBytes(stream, fieldNumber, value.AsSpan());
            }

            public static void WriteBytes(Stream stream, int fieldNumber, ReadOnlySpan<byte> value)
            {
                WriteTag(stream, fieldNumber, WireTypeLengthDelimited);
                WriteVarUInt64(stream, (ulong)value.Length);
                stream.Write(value);
            }

            public static void WriteMessage(Stream stream, int fieldNumber, Action<Stream> writeMessage)
            {
                using var buffer = new MemoryStream();

                writeMessage(buffer);

                WriteTag(stream, fieldNumber, WireTypeLengthDelimited);
                WriteVarUInt64(stream, (ulong)buffer.Length);

                buffer.Position = 0;
                buffer.CopyTo(stream);
            }

            private static void WriteTag(Stream stream, int fieldNumber, int wireType)
            {
                WriteVarUInt64(stream, (ulong)fieldNumber << 3 | (uint)wireType);
            }

            private static void WriteVarUInt64(Stream stream, ulong value)
            {
                Span<byte> buffer = stackalloc byte[10];
                var index = 0;

                while (value >= 0x80)
                {
                    buffer[index++] = (byte)(value | 0x80);
                    value >>= 7;
                }

                buffer[index++] = (byte)value;
                stream.Write(buffer.Slice(0, index));
            }
        }
    }
}