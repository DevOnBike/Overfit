// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// The streaming Q5_0 to Q8 conversion must equal the route through the full F32 table, exactly.
    ///
    /// <para><b>The oracle is an equality, not a tolerance, and that is the point.</b>
    /// <see cref="Q8Weight.QuantizeQ5_0Rows"/> decodes each Q5_0 block with
    /// <see cref="GgmlDequant.DecodeQ5_0Block"/> and then quantizes the row with
    /// <c>Q8DotKernel.Quantize</c> — the same two operations, in the same order, that
    /// <see cref="Q8Weight.QuantizeRows"/> performs after somebody else has decoded the whole tensor. Only
    /// the buffering differs. So the two must agree bit for bit, and a tolerance here would hide exactly the
    /// kind of off-by-one-block indexing error the streaming version can make and the batch version cannot.
    /// </para>
    ///
    /// <para><b>The subject and the oracle do not share a variable.</b> The reference decodes the whole
    /// tensor into its own F32 array first; the subject never allocates one. Their only common input is the
    /// raw byte array. That matters — `XC-59` records a self-consistency assertion in this repository that
    /// was blind by construction because one local fed both sides.</para>
    ///
    /// <para>Written for `XC-97`: a Q5_0 embedding used to take the F32 fallback, measured at 519 MB for a
    /// 469 MB model.</para>
    /// </summary>
    public sealed class Q5ToQ8ConversionTests
    {
        /// <summary>Deterministic Q5_0 blocks. Every byte pattern is exercised, including the qh bits.</summary>
        private static byte[] SyntheticQ5_0(int rowCount, int rowLength, int seed)
        {
            var blocksPerRow = rowLength / GgmlDequant.Q5_0_BlockElements;
            var bytes = new byte[rowCount * blocksPerRow * GgmlDequant.Q5_0_BlockBytes];
            var rng = new Random(seed);

            for (var i = 0; i < bytes.Length; i++)
            {
                bytes[i] = (byte)rng.Next(256);
            }

            // Byte 0-1 of each block is an FP16 scale. Random bits there can land on NaN or infinity, which
            // would make every downstream comparison vacuously equal. Pin them to a finite, varied value.
            for (var b = 0; b < bytes.Length / GgmlDequant.Q5_0_BlockBytes; b++)
            {
                var half = (Half)(0.001f + (b % 97) * 0.01f);
                var raw = BitConverter.HalfToUInt16Bits(half);

                bytes[b * GgmlDequant.Q5_0_BlockBytes] = (byte)(raw & 0xFF);
                bytes[(b * GgmlDequant.Q5_0_BlockBytes) + 1] = (byte)(raw >> 8);
            }

            return bytes;
        }

        private static Q8Weight ViaFullF32Table(byte[] raw, int rowCount, int rowLength)
        {
            var blocksPerRow = rowLength / GgmlDequant.Q5_0_BlockElements;
            var table = new float[rowCount * rowLength];

            for (var r = 0; r < rowCount; r++)
            {
                for (var b = 0; b < blocksPerRow; b++)
                {
                    var blockOffset = ((r * blocksPerRow) + b) * GgmlDequant.Q5_0_BlockBytes;

                    GgmlDequant.DecodeQ5_0Block(
                        raw.AsSpan(blockOffset, GgmlDequant.Q5_0_BlockBytes),
                        table.AsSpan((r * rowLength) + (b * GgmlDequant.Q5_0_BlockElements),
                            GgmlDequant.Q5_0_BlockElements));
                }
            }

            return Q8Weight.QuantizeRows(table, rowCount, rowLength);
        }

        [Theory]
        // 896 is Qwen2.5-0.5B's hidden size — the model that produced the finding. 64 and 2048 bracket it.
        [InlineData(7, 64)]
        [InlineData(31, 896)]
        [InlineData(3, 2048)]
        public void StreamingConversionIsBitIdenticalToTheFullTableRoute(int rowCount, int rowLength)
        {
            var raw = SyntheticQ5_0(rowCount, rowLength, seed: 20260820 + rowLength);

            var expected = ViaFullF32Table(raw, rowCount, rowLength);
            var actual = Q8Weight.QuantizeQ5_0Rows(raw, rowCount, rowLength);

            Assert.Equal(expected.InputSize, actual.InputSize);
            Assert.Equal(expected.OutputSize, actual.OutputSize);
            Assert.Equal(expected.Quants.Length, actual.Quants.Length);
            Assert.Equal(expected.Scales.Length, actual.Scales.Length);

            for (var i = 0; i < expected.Quants.Length; i++)
            {
                if (expected.Quants[i] != actual.Quants[i])
                {
                    Assert.Fail($"quant[{i}] differs: expected {expected.Quants[i]}, got {actual.Quants[i]} "
                                + $"(row {i / rowLength}, column {i % rowLength})");
                }
            }

            for (var i = 0; i < expected.Scales.Length; i++)
            {
                if (!expected.Scales[i].Equals(actual.Scales[i]))
                {
                    Assert.Fail($"scale[{i}] differs: expected {expected.Scales[i]:R}, "
                                + $"got {actual.Scales[i]:R}");
                }
            }
        }

        [Fact]
        public void ADecodedRowRoundTripsCloseToTheQ5_0ValuesItCameFrom()
        {
            // The equality above pins the conversion against its own reference, which cannot catch both
            // routes being wrong the same way. This one checks the ACCURACY claim the change rests on: a
            // Q5_0 block carries 32 distinct levels and a Q8 block carries 256, so re-quantizing should cost
            // far less than the source quantization already did.
            const int RowLength = 896;

            var raw = SyntheticQ5_0(rowCount: 4, RowLength, seed: 1);
            var weight = Q8Weight.QuantizeQ5_0Rows(raw, rowCount: 4, RowLength);

            var q5 = new float[RowLength];
            var blocksPerRow = RowLength / GgmlDequant.Q5_0_BlockElements;

            for (var b = 0; b < blocksPerRow; b++)
            {
                GgmlDequant.DecodeQ5_0Block(
                    raw.AsSpan(b * GgmlDequant.Q5_0_BlockBytes, GgmlDequant.Q5_0_BlockBytes),
                    q5.AsSpan(b * GgmlDequant.Q5_0_BlockElements, GgmlDequant.Q5_0_BlockElements));
            }

            var q8 = new float[RowLength];

            weight.DecodeRow(0, q8);

            var worst = 0f;
            var magnitude = 0f;

            for (var i = 0; i < RowLength; i++)
            {
                worst = MathF.Max(worst, MathF.Abs(q8[i] - q5[i]));
                magnitude = MathF.Max(magnitude, MathF.Abs(q5[i]));
            }

            Assert.True(magnitude > 0f, "the synthetic row is all zeros — the comparison would be vacuous");

            // A Q8 block's step is max|x| / 127, so the worst re-quantization error is half of that. Allowing
            // one full step is loose enough to survive rounding and tight enough that a wrong scale or a
            // mis-indexed block fails.
            var step = magnitude / 127f;

            Assert.True(worst <= step,
                $"re-quantization error {worst:E3} exceeds one Q8 step {step:E3} (row magnitude {magnitude:E3})");
        }
    }
}
