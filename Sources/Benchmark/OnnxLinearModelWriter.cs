using System.Buffers.Binary;
using System.Text;

namespace Benchmarks
{
    internal static class OnnxLinearModelWriter
    {
        private const int TensorProtoDataTypeFloat = 1;

        public static void WriteLinearGemmModel(
            string path,
            int inputSize,
            int outputSize,
            float[] weightsInputOutput,
            float[] bias)
        {
            if (weightsInputOutput.Length != inputSize * outputSize)
            {
                throw new ArgumentException("Weights length mismatch.", nameof(weightsInputOutput));
            }

            if (bias.Length != outputSize)
            {
                throw new ArgumentException("Bias length mismatch.", nameof(bias));
            }

            using var file = File.Create(path);

            WriteModel(
                file,
                inputSize,
                outputSize,
                weightsInputOutput,
                bias);
        }

        private static void WriteModel(
            Stream stream,
            int inputSize,
            int outputSize,
            float[] weightsInputOutput,
            float[] bias)
        {
            Proto.WriteInt64(stream, 1, 9);
            Proto.WriteString(stream, 2, "DevOnBike.Overfit.Benchmark");

            Proto.WriteMessage(stream, 7, graph =>
            {
                WriteGraph(
                    graph,
                    inputSize,
                    outputSize,
                    weightsInputOutput,
                    bias);
            });

            Proto.WriteMessage(stream, 8, opset =>
            {
                Proto.WriteInt64(opset, 2, 13);
            });
        }

        private static void WriteGraph(
            Stream stream,
            int inputSize,
            int outputSize,
            float[] weightsInputOutput,
            float[] bias)
        {
            Proto.WriteMessage(stream, 1, node =>
            {
                Proto.WriteString(node, 1, "input");
                Proto.WriteString(node, 1, "W");
                Proto.WriteString(node, 1, "B");

                Proto.WriteString(node, 2, "output");

                Proto.WriteString(node, 3, "linear_gemm");
                Proto.WriteString(node, 4, "Gemm");
            });

            Proto.WriteString(stream, 2, "overfit_linear_graph");

            WriteFloatTensor(
                stream,
                fieldNumber: 5,
                name: "W",
                dims: [inputSize, outputSize],
                data: weightsInputOutput);

            WriteFloatTensor(
                stream,
                fieldNumber: 5,
                name: "B",
                dims: [outputSize],
                data: bias);

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

                Proto.WriteMessage(valueInfo, 2, typeProto =>
                {
                    Proto.WriteMessage(typeProto, 1, tensorType =>
                    {
                        Proto.WriteInt32(tensorType, 1, TensorProtoDataTypeFloat);

                        Proto.WriteMessage(tensorType, 2, shape =>
                        {
                            for (var i = 0; i < dims.Length; i++)
                            {
                                var dimValue = dims[i];

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
                WriteVarUInt64(stream, ((ulong)fieldNumber << 3) | (uint)wireType);
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