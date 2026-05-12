// Copyright (c) 2026 DevOnBike. AGPLv3.

using System.Buffers.Binary;
using DevOnBike.Overfit.LanguageModels.Loading;
using Xunit;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Tests for quantized GGUF tensor dequantization.
    /// Each block is built byte-by-byte to match the ggml-quants.c reference layout.
    /// </summary>
    [Trait("Category", "Gguf")]
    [Trait("Category", "Quantization")]
    public sealed class GgmlDequantTests
    {
        [Fact]
        public void Q8_0_DequantizesSingleBlock_RoundTrip()
        {
            // Build one Q8_0 block: scale=2.0, quants = [-128, -1, 0, 1, ..., 31, ..., 127, 50] (32 ints)
            // Expected: dst[i] = scale * quant[i]
            const float scale = 2.0f;
            var quants = new sbyte[32];
            quants[0] = -128;
            quants[1] = -1;
            quants[2] = 0;
            quants[3] = 1;
            quants[4] = 127;
            for (var i = 5; i < 32; i++) { quants[i] = (sbyte)(i - 16); }  // mix of pos/neg

            var ms = BuildSingleBlockGguf(GgmlType.Q8_0, 32, scale, quants);
            using var reader = new GgufReader(ms);

            var dst = new float[32];
            reader.LoadTensorAsF32(reader.Tensors["test"], dst);

            // FP16 scale 2.0 is exactly representable
            for (var i = 0; i < 32; i++)
            {
                Assert.Equal(scale * quants[i], dst[i], precision: 4);
            }
        }

        [Fact]
        public void Q8_0_DequantizesMultipleBlocks_DifferentScales()
        {
            // 3 blocks (96 elements) with different scales — verify each block uses its own scale
            const int nBlocks = 3;
            const int totalElements = nBlocks * 32;
            var scales = new float[] { 0.5f, 1.0f, 0.125f };
            var allQuants = new sbyte[totalElements];
            for (var b = 0; b < nBlocks; b++)
            {
                for (var i = 0; i < 32; i++)
                {
                    allQuants[b * 32 + i] = (sbyte)(b * 10 + i - 5);
                }
            }

            var ms = BuildMultiBlockGguf(GgmlType.Q8_0, totalElements, scales, allQuants);
            using var reader = new GgufReader(ms);

            var dst = new float[totalElements];
            reader.LoadTensorAsF32(reader.Tensors["test"], dst);

            for (var b = 0; b < nBlocks; b++)
            {
                for (var i = 0; i < 32; i++)
                {
                    var expected = scales[b] * allQuants[b * 32 + i];
                    Assert.Equal(expected, dst[b * 32 + i], precision: 4);
                }
            }
        }

        [Fact]
        public void Q8_0_ThrowsWhenElementCountNotMultipleOf32()
        {
            // 31 elements is invalid for Q8_0 (block size = 32)
            // Build minimal raw GGUF with dim=31
            var ms = BuildRawGgufWithDim(GgmlType.Q8_0, 31, dataBytes: new byte[34]);
            using var reader = new GgufReader(ms);

            var dst = new float[31];
            var ex = Assert.Throws<InvalidDataException>(() =>
                reader.LoadTensorAsF32(reader.Tensors["test"], dst));
            Assert.Contains("not divisible by 32", ex.Message);
        }

        // ── Helpers ─────────────────────────────────────────────────────────

        private static MemoryStream BuildSingleBlockGguf(GgmlType type, int elementCount, float scale, sbyte[] quants)
        {
            return BuildMultiBlockGguf(type, elementCount, new[] { scale }, quants);
        }

        private static MemoryStream BuildMultiBlockGguf(GgmlType type, int elementCount, float[] scales, sbyte[] quants)
        {
            // Build raw byte payload for the data section
            const int blockBytes = 34;  // Q8_0: 2 bytes scale + 32 quants
            var nBlocks = scales.Length;
            var payload = new byte[nBlocks * blockBytes];

            for (var b = 0; b < nBlocks; b++)
            {
                // FP16 scale (2 bytes LE)
                var halfBits = BitConverter.HalfToUInt16Bits((Half)scales[b]);
                BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(b * blockBytes, 2), halfBits);
                // 32 int8 quants
                for (var i = 0; i < 32; i++)
                {
                    payload[b * blockBytes + 2 + i] = (byte)quants[b * 32 + i];
                }
            }

            return BuildRawGgufWithDim(type, elementCount, payload);
        }

        private static MemoryStream BuildRawGgufWithDim(GgmlType type, int elementCount, byte[] dataBytes)
        {
            var ms = new MemoryStream();
            using (var bw = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
            {
                bw.Write(GgufFormat.Magic);
                bw.Write((uint)3);  // version
                bw.Write((ulong)1);  // tensorCount
                bw.Write((ulong)0);  // metaCount

                // Tensor info: name="test", 1 dim = elementCount, type, offset=0
                var nameBytes = System.Text.Encoding.UTF8.GetBytes("test");
                bw.Write((ulong)nameBytes.Length);
                bw.Write(nameBytes);
                bw.Write((uint)1);  // nDims
                bw.Write((ulong)elementCount);
                bw.Write((uint)type);
                bw.Write((ulong)0);  // offset

                // Align to 32
                var pos = ms.Position;
                var aligned = (pos + 31) & ~31L;
                for (var p = pos; p < aligned; p++) { bw.Write((byte)0); }

                // Data
                bw.Write(dataBytes);
            }
            ms.Position = 0;
            return ms;
        }
    }
}
