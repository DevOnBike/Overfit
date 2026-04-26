// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using System.Text;

namespace Benchmarks.Helpers
{
    internal static class OnnxCnnModelWriter
    {
        private const int TensorProtoDataTypeFloat = 1;

        private const int AttributeTypeInt = 2;
        private const int AttributeTypeInts = 7;

        public static void WriteConvReluPoolGapLinearModel(
            string path,
            int inputChannels,
            int inputH,
            int inputW,
            int convOutChannels,
            int kernel,
            int pool,
            int outputClasses,
            float[] convKernels,
            float[] linearWeightsInputOutput,
            float[] linearBias)
        {
            var convOutH = inputH - kernel + 1;
            var convOutW = inputW - kernel + 1;

            var poolOutH = convOutH / pool;
            var poolOutW = convOutW / pool;

            var expectedConvKernelLength = convOutChannels * inputChannels * kernel * kernel;
            var expectedLinearWeightLength = convOutChannels * outputClasses;

            if (convKernels.Length != expectedConvKernelLength)
            {
                throw new ArgumentException(
                    $"Conv kernel length mismatch. Expected {expectedConvKernelLength}, got {convKernels.Length}.",
                    nameof(convKernels));
            }

            if (linearWeightsInputOutput.Length != expectedLinearWeightLength)
            {
                throw new ArgumentException(
                    $"Linear weight length mismatch. Expected {expectedLinearWeightLength}, got {linearWeightsInputOutput.Length}.",
                    nameof(linearWeightsInputOutput));
            }

            if (linearBias.Length != outputClasses)
            {
                throw new ArgumentException(
                    $"Linear bias length mismatch. Expected {outputClasses}, got {linearBias.Length}.",
                    nameof(linearBias));
            }

            if (convOutH % pool != 0 || convOutW % pool != 0)
            {
                throw new ArgumentException("Conv output spatial dimensions must be divisible by pool.");
            }

            using var file = File.Create(path);

            WriteModel(
                file,
                inputChannels,
                inputH,
                inputW,
                convOutChannels,
                kernel,
                pool,
                poolOutH,
                poolOutW,
                outputClasses,
                convKernels,
                linearWeightsInputOutput,
                linearBias);
        }

        private static void WriteModel(
            Stream stream,
            int inputChannels,
            int inputH,
            int inputW,
            int convOutChannels,
            int kernel,
            int pool,
            int poolOutH,
            int poolOutW,
            int outputClasses,
            float[] convKernels,
            float[] linearWeightsInputOutput,
            float[] linearBias)
        {
            Proto.WriteInt64(stream, 1, 9); // ir_version
            Proto.WriteString(stream, 2, "DevOnBike.Overfit.Benchmark");

            Proto.WriteMessage(stream, 7, graph =>
            {
                WriteGraph(
                    graph,
                    inputChannels,
                    inputH,
                    inputW,
                    convOutChannels,
                    kernel,
                    pool,
                    poolOutH,
                    poolOutW,
                    outputClasses,
                    convKernels,
                    linearWeightsInputOutput,
                    linearBias);
            });

            Proto.WriteMessage(stream, 8, opset =>
            {
                Proto.WriteInt64(opset, 2, 13);
            });
        }

        private static void WriteGraph(
            Stream stream,
            int inputChannels,
            int inputH,
            int inputW,
            int convOutChannels,
            int kernel,
            int pool,
            int poolOutH,
            int poolOutW,
            int outputClasses,
            float[] convKernels,
            float[] linearWeightsInputOutput,
            float[] linearBias)
        {
            // conv_out = Conv(input, conv_W)
            Proto.WriteMessage(stream, 1, node =>
            {
                Proto.WriteString(node, 1, "input");
                Proto.WriteString(node, 1, "conv_W");

                Proto.WriteString(node, 2, "conv_out");

                Proto.WriteString(node, 3, "conv");
                Proto.WriteString(node, 4, "Conv");

                WriteIntsAttribute(node, "kernel_shape", [kernel, kernel]);
                WriteIntsAttribute(node, "strides", [1, 1]);
            });

            // relu_out = Relu(conv_out)
            Proto.WriteMessage(stream, 1, node =>
            {
                Proto.WriteString(node, 1, "conv_out");

                Proto.WriteString(node, 2, "relu_out");

                Proto.WriteString(node, 3, "relu");
                Proto.WriteString(node, 4, "Relu");
            });

            // pool_out = MaxPool(relu_out)
            Proto.WriteMessage(stream, 1, node =>
            {
                Proto.WriteString(node, 1, "relu_out");

                Proto.WriteString(node, 2, "pool_out");

                Proto.WriteString(node, 3, "maxpool");
                Proto.WriteString(node, 4, "MaxPool");

                WriteIntsAttribute(node, "kernel_shape", [pool, pool]);
                WriteIntsAttribute(node, "strides", [pool, pool]);
            });

            // gap_out = GlobalAveragePool(pool_out)
            Proto.WriteMessage(stream, 1, node =>
            {
                Proto.WriteString(node, 1, "pool_out");

                Proto.WriteString(node, 2, "gap_out");

                Proto.WriteString(node, 3, "gap");
                Proto.WriteString(node, 4, "GlobalAveragePool");
            });

            // flat_out = Flatten(gap_out), axis=1 => [N, C]
            Proto.WriteMessage(stream, 1, node =>
            {
                Proto.WriteString(node, 1, "gap_out");

                Proto.WriteString(node, 2, "flat_out");

                Proto.WriteString(node, 3, "flatten");
                Proto.WriteString(node, 4, "Flatten");

                WriteIntAttribute(node, "axis", 1);
            });

            // output = Gemm(flat_out, fc_W, fc_B)
            Proto.WriteMessage(stream, 1, node =>
            {
                Proto.WriteString(node, 1, "flat_out");
                Proto.WriteString(node, 1, "fc_W");
                Proto.WriteString(node, 1, "fc_B");

                Proto.WriteString(node, 2, "output");

                Proto.WriteString(node, 3, "fc");
                Proto.WriteString(node, 4, "Gemm");
            });

            Proto.WriteString(stream, 2, "overfit_cnn_graph");

            WriteFloatTensor(
                stream,
                fieldNumber: 5,
                name: "conv_W",
                dims:
                [
                    convOutChannels,
                    inputChannels,
                    kernel,
                    kernel
                ],
                data: convKernels);

            WriteFloatTensor(
                stream,
                fieldNumber: 5,
                name: "fc_W",
                dims:
                [
                    convOutChannels,
                    outputClasses
                ],
                data: linearWeightsInputOutput);

            WriteFloatTensor(
                stream,
                fieldNumber: 5,
                name: "fc_B",
                dims:
                [
                    outputClasses
                ],
                data: linearBias);

            WriteValueInfo(
                stream,
                fieldNumber: 11,
                name: "input",
                dims:
                [
                    1,
                    inputChannels,
                    inputH,
                    inputW
                ]);

            WriteValueInfo(
                stream,
                fieldNumber: 12,
                name: "output",
                dims:
                [
                    1,
                    outputClasses
                ]);

            // Optional value info; helps debug malformed shape graphs.
            WriteValueInfo(
                stream,
                fieldNumber: 13,
                name: "pool_out",
                dims:
                [
                    1,
                    convOutChannels,
                    poolOutH,
                    poolOutW
                ]);

            WriteValueInfo(
                stream,
                fieldNumber: 13,
                name: "gap_out",
                dims:
                [
                    1,
                    convOutChannels,
                    1,
                    1
                ]);
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

        private static void WriteIntAttribute(
            Stream node,
            string name,
            long value)
        {
            Proto.WriteMessage(node, 5, attr =>
            {
                Proto.WriteString(attr, 1, name);
                Proto.WriteInt64(attr, 3, value);
                Proto.WriteInt32(attr, 20, AttributeTypeInt);
            });
        }

        private static void WriteIntsAttribute(
            Stream node,
            string name,
            long[] values)
        {
            Proto.WriteMessage(node, 5, attr =>
            {
                Proto.WriteString(attr, 1, name);

                for (var i = 0; i < values.Length; i++)
                {
                    Proto.WriteInt64(attr, 8, values[i]);
                }

                Proto.WriteInt32(attr, 20, AttributeTypeInts);
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