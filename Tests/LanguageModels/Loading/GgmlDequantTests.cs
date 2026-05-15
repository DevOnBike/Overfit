// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using System.Text;
using DevOnBike.Overfit.LanguageModels.Loading;

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
            var scales = new[] { 0.5f, 1.0f, 0.125f };
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

        // ── UnpackQ4_KScalesMins (pure 6-bit unpack) ────────────────────────

        [Fact]
        public void UnpackQ4_KScalesMins_LowerBranch_Jbelow4_Trivial()
        {
            // packed[0..3] → scales[0..3] (low 6 bits), packed[4..7] → mins[0..3]
            // packed[8..11] left zero so scales[4..7]=mins[4..7]=0.
            Span<byte> packed = stackalloc byte[12];
            packed[0] = 10; packed[1] = 20; packed[2] = 30; packed[3] = 40;
            packed[4] =  5; packed[5] = 15; packed[6] = 21; packed[7] = 27;

            Span<byte> scales = stackalloc byte[8];
            Span<byte> mins   = stackalloc byte[8];
            GgmlDequant.UnpackQ4_KScalesMins(packed, scales, mins);

            Assert.Equal(10, scales[0]); Assert.Equal(20, scales[1]);
            Assert.Equal(30, scales[2]); Assert.Equal(40, scales[3]);
            Assert.Equal( 5, mins[0]);   Assert.Equal(15, mins[1]);
            Assert.Equal(21, mins[2]);   Assert.Equal(27, mins[3]);
            for (var j = 4; j < 8; j++) { Assert.Equal(0, scales[j]); Assert.Equal(0, mins[j]); }
        }

        [Fact]
        public void UnpackQ4_KScalesMins_UpperBranch_Jat4Plus_HighBitsContribute()
        {
            // Drive the j>=4 branch: high 2 bits of packed[0..3]/packed[4..7] shift into
            // the high nibble of scales[4..7]/mins[4..7]; low nibble of those comes from
            // packed[8..11] low/high nibbles respectively.
            //   packed[0..3] = 0xC0 (high 2 bits = 0b11 → 0x30 contribution)
            //   packed[4..7] = 0x80 (high 2 bits = 0b10 → 0x20 contribution)
            //   packed[8..11] = 0x35 (low nibble 0x05, high nibble 0x03)
            // → scales[4..7] = 0x30 | 0x05 = 0x35, mins[4..7] = 0x20 | 0x03 = 0x23.
            Span<byte> packed = stackalloc byte[12];
            for (var i = 0; i < 4; i++)  { packed[i]     = 0xC0; }
            for (var i = 4; i < 8; i++)  { packed[i]     = 0x80; }
            for (var i = 8; i < 12; i++) { packed[i]     = 0x35; }

            Span<byte> scales = stackalloc byte[8];
            Span<byte> mins   = stackalloc byte[8];
            GgmlDequant.UnpackQ4_KScalesMins(packed, scales, mins);

            for (var j = 0; j < 4; j++) { Assert.Equal(0, scales[j]); Assert.Equal(0, mins[j]); }
            for (var j = 4; j < 8; j++) { Assert.Equal(0x35, scales[j]); Assert.Equal(0x23, mins[j]); }
        }

        [Fact]
        public void UnpackQ4_KScalesMins_RejectsWrongSizes()
        {
            var p12 = new byte[12]; var s8 = new byte[8]; var m8 = new byte[8];
            Assert.Throws<ArgumentException>(() =>
                GgmlDequant.UnpackQ4_KScalesMins(new byte[11], s8, m8));
            Assert.Throws<ArgumentException>(() =>
                GgmlDequant.UnpackQ4_KScalesMins(p12, new byte[7], m8));
            Assert.Throws<ArgumentException>(() =>
                GgmlDequant.UnpackQ4_KScalesMins(p12, s8, new byte[9]));
        }

        // ── DecodeQ4_KBlock ─────────────────────────────────────────────────

        [Fact]
        public void DecodeQ4_KBlock_AllSubBlocksUseScale1Min0_NibbleEqualsValue()
        {
            // d = 1, dmin = 0, all 8 scales = 1, all 8 mins = 0, all nibble-bytes = 0x21.
            // Per sub-pair: low nibble = 1, high nibble = 2.
            // Expected pattern: 32×1, 32×2, repeating 4× → covers all 256 elements.
            var block = BuildQ4_KBlock(
                d: 1.0f, dmin: 0.0f,
                scalesMinsPacked: ScalesAllOnesMinsAllZeros(),
                nibbleFill: 0x21);

            Span<float> dst = stackalloc float[256];
            GgmlDequant.DecodeQ4_KBlock(block, dst);

            for (var p = 0; p < 4; p++)
            {
                for (var i = 0; i < 32; i++)
                {
                    Assert.Equal(1.0f, dst[64 * p + i],      precision: 6);
                    Assert.Equal(2.0f, dst[64 * p + 32 + i], precision: 6);
                }
            }
        }

        [Fact]
        public void DecodeQ4_KBlock_ScaleAndMinApplyPerSubBlock()
        {
            // d = 2, dmin = 1; scales = 1..8, mins = 0..7; nibbles = 0x53 (low=3, high=5).
            // sub-pair p uses scales[2p], scales[2p+1], mins[2p], mins[2p+1].
            var packed = new byte[12];
            for (var i = 0; i < 4; i++) { packed[i]     = (byte)(i + 1); } // scales[0..3] = 1..4
            for (var i = 0; i < 4; i++) { packed[i + 4] = (byte)i; }       // mins[0..3]   = 0..3
            for (var j = 4; j < 8; j++)
            {
                var sc = (byte)(j + 1);  // scales[4..7] = 5..8
                var mn = (byte)j;        // mins[4..7]   = 4..7
                packed[j + 4] = (byte)((mn << 4) | (sc & 0x0F));
            }

            var block = BuildQ4_KBlock(
                d: 2.0f, dmin: 1.0f, scalesMinsPacked: packed, nibbleFill: 0x53);

            Span<float> dst = stackalloc float[256];
            GgmlDequant.DecodeQ4_KBlock(block, dst);

            for (var p = 0; p < 4; p++)
            {
                var expectedLow  = 2.0f * (2 * p + 1) * 3 - 1.0f * (2 * p);
                var expectedHigh = 2.0f * (2 * p + 2) * 5 - 1.0f * (2 * p + 1);
                for (var i = 0; i < 32; i++)
                {
                    Assert.Equal(expectedLow,  dst[64 * p + i],      precision: 4);
                    Assert.Equal(expectedHigh, dst[64 * p + 32 + i], precision: 4);
                }
            }
        }

        [Fact]
        public void DecodeQ4_KBlock_RejectsWrongSizes()
        {
            Assert.Throws<ArgumentException>(() =>
                GgmlDequant.DecodeQ4_KBlock(new byte[GgmlDequant.Q4_K_BlockBytes - 1], new float[256]));
            Assert.Throws<ArgumentException>(() =>
                GgmlDequant.DecodeQ4_KBlock(new byte[GgmlDequant.Q4_K_BlockBytes], new float[255]));
        }

        // ── DecodeQ6_KBlock ─────────────────────────────────────────────────

        [Fact]
        public void DecodeQ6_KBlock_AllZerosWithUnitScale_GivesBiasMinus32()
        {
            // q = 0 everywhere; biased = 0 − 32 = −32. d=1, scales=1 → dst[i] = −32.
            var block = BuildQ6_KBlock(
                d: 1.0f, scales: ConstantScales(1), qlFill: 0, qhFill: 0);

            Span<float> dst = stackalloc float[256];
            GgmlDequant.DecodeQ6_KBlock(block, dst);

            for (var i = 0; i < 256; i++) { Assert.Equal(-32f, dst[i], precision: 4); }
        }

        [Fact]
        public void DecodeQ6_KBlock_HighBitsAndSignedScalesApply()
        {
            // ql = 0xFF, qh = 0xFF → each q = 0xF | (0b11 << 4) = 0x3F = 63; biased = 31.
            // d = 0.5, scales = -1 (signed!) → dst[i] = 0.5 * -1 * 31 = -15.5.
            var block = BuildQ6_KBlock(
                d: 0.5f, scales: ConstantScales(-1), qlFill: 0xFF, qhFill: 0xFF);

            Span<float> dst = stackalloc float[256];
            GgmlDequant.DecodeQ6_KBlock(block, dst);

            for (var i = 0; i < 256; i++) { Assert.Equal(-15.5f, dst[i], precision: 4); }
        }

        [Fact]
        public void DecodeQ6_KBlock_LowAndHighNibblesDecodeIndependently()
        {
            // ql[0] = 0x10 (low nibble = 0, high nibble = 1); rest of ql = 0; qh = 0.
            // For l=0 in half-block 0:
            //   q3 = ql[0] >> 4 = 1, biased = 1 − 32 = −31 → dst[64] = -31
            //   q1 = ql[0]   & 0xF = 0 → dst[0]  = -32
            //   q2 = ql[32]  & 0xF = 0 → dst[32] = -32
            //   q4 = ql[32]  >> 4   = 0 → dst[96] = -32
            var ql = new byte[128];
            ql[0] = 0x10;
            var block = BuildQ6_KBlock(
                d: 1.0f, scales: ConstantScales(1), ql: ql, qhFill: 0);

            Span<float> dst = stackalloc float[256];
            GgmlDequant.DecodeQ6_KBlock(block, dst);

            Assert.Equal(-31f, dst[64], precision: 4);
            Assert.Equal(-32f, dst[0],  precision: 4);
            Assert.Equal(-32f, dst[32], precision: 4);
            Assert.Equal(-32f, dst[96], precision: 4);
            // Random other elements stay at the all-zero baseline.
            Assert.Equal(-32f, dst[128], precision: 4);
            Assert.Equal(-32f, dst[255], precision: 4);
        }

        [Fact]
        public void DecodeQ6_KBlock_RejectsWrongSizes()
        {
            Assert.Throws<ArgumentException>(() =>
                GgmlDequant.DecodeQ6_KBlock(new byte[GgmlDequant.Q6_K_BlockBytes - 1], new float[256]));
            Assert.Throws<ArgumentException>(() =>
                GgmlDequant.DecodeQ6_KBlock(new byte[GgmlDequant.Q6_K_BlockBytes], new float[255]));
        }

        // ── Q4_K / Q6_K helpers ──────────────────────────────────────────────

        private static byte[] BuildQ4_KBlock(
            float d, float dmin, byte[] scalesMinsPacked, byte nibbleFill)
        {
            var block = new byte[GgmlDequant.Q4_K_BlockBytes];
            WriteFp16Le(block.AsSpan(0, 2), d);
            WriteFp16Le(block.AsSpan(2, 2), dmin);
            scalesMinsPacked.AsSpan().CopyTo(block.AsSpan(4, 12));
            block.AsSpan(16, 128).Fill(nibbleFill);
            return block;
        }

        private static byte[] BuildQ6_KBlock(float d, sbyte[] scales, byte qlFill, byte qhFill)
        {
            var ql = new byte[128];
            ql.AsSpan().Fill(qlFill);
            return BuildQ6_KBlock(d, scales, ql, qhFill);
        }

        private static byte[] BuildQ6_KBlock(float d, sbyte[] scales, byte[] ql, byte qhFill)
        {
            var block = new byte[GgmlDequant.Q6_K_BlockBytes];
            ql.AsSpan().CopyTo(block.AsSpan(0, 128));
            block.AsSpan(128, 64).Fill(qhFill);
            for (var i = 0; i < 16; i++) { block[192 + i] = (byte)scales[i]; }
            WriteFp16Le(block.AsSpan(208, 2), d);
            return block;
        }

        private static byte[] ScalesAllOnesMinsAllZeros()
        {
            // For UnpackQ4_KScalesMins to yield scales[0..7]=1, mins[0..7]=0:
            //   packed[0..3] = 0x01 (low 6 bits = 1, high 2 bits = 0)
            //   packed[4..7] = 0x00
            //   packed[8..11]: low nibble = 1 (→ scales[4..7]=1), high nibble = 0 (→ mins[4..7]=0)
            var packed = new byte[12];
            for (var i = 0; i < 4; i++)  { packed[i]     = 0x01; }
            for (var i = 8; i < 12; i++) { packed[i]     = 0x01; }
            return packed;
        }

        private static sbyte[] ConstantScales(sbyte v)
        {
            var s = new sbyte[16];
            for (var i = 0; i < 16; i++) { s[i] = v; }
            return s;
        }

        private static void WriteFp16Le(Span<byte> dst2, float value)
        {
            BinaryPrimitives.WriteUInt16LittleEndian(
                dst2, BitConverter.HalfToUInt16Bits((Half)value));
        }

        // ── Existing Q8_0 helpers ───────────────────────────────────────────

        private static MemoryStream BuildSingleBlockGguf(GgmlType type, int elementCount, float scale, sbyte[] quants)
        {
            return BuildMultiBlockGguf(type, elementCount, [scale], quants);
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
            using (var bw = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
            {
                bw.Write(GgufFormat.Magic);
                bw.Write((uint)3);  // version
                bw.Write((ulong)1);  // tensorCount
                bw.Write((ulong)0);  // metaCount

                // Tensor info: name="test", 1 dim = elementCount, type, offset=0
                var nameBytes = Encoding.UTF8.GetBytes("test");
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
