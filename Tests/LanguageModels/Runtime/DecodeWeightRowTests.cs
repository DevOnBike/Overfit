// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// Fast unit tests for the per-row dequant primitive (<see cref="DecodeWeight.DequantizeRow"/> /
    /// <see cref="Q8Weight.DecodeRow"/>) that backs the quantized token-embedding lookup: row r of a
    /// [rows × cols] table must come back as the r-th cols-length slice. Real Q4_K/Q6_K row decode is
    /// covered end-to-end by the Q4_K_M decode-parity `[LongFact]` (bit-identical to full-tensor dequant).
    /// </summary>
    public sealed class DecodeWeightRowTests
    {
        [Fact]
        public void F32Backing_DequantizeRow_ReturnsExactRow()
        {
            const int rows = 5;
            const int cols = 8;
            using var storage = TensorStorage<float>.Unpooled(rows * cols);
            var span = storage.AsSpan();
            for (var r = 0; r < rows; r++)
            {
                for (var c = 0; c < cols; c++) { span[r * cols + c] = r * 100 + c; }
            }

            DecodeWeight w = storage;
            Span<float> dst = stackalloc float[cols];
            for (var r = 0; r < rows; r++)
            {
                w.DequantizeRow(r, dst);
                for (var c = 0; c < cols; c++)
                {
                    Assert.Equal(r * 100 + c, dst[c]);
                }
            }
        }

        [Fact]
        public void Q8Backing_DecodeRow_MatchesQuantizedRoundTrip()
        {
            // Two rows of 64 (= 2 Q8 blocks of 32). Symmetric Q8 (scale = absmax/127) is lossy,
            // so compare against the round-tripped value, not the original — the row decode must
            // reproduce exactly what the dot kernel would see.
            const int rows = 2;
            const int cols = 64;
            var src = new float[rows * cols];
            for (var r = 0; r < rows; r++)
            {
                for (var c = 0; c < cols; c++) { src[r * cols + c] = MathF.Sin((r + 1) * 0.3f * c) * (r + 1); }
            }

            var q8 = Q8Weight.QuantizeRows(src, rows, cols);
            DecodeWeight w = q8;

            Span<float> dst = stackalloc float[cols];
            for (var r = 0; r < rows; r++)
            {
                w.DequantizeRow(r, dst);
                for (var c = 0; c < cols; c++)
                {
                    // Q8 step is absmax/127; within one step of the source is correct round-trip.
                    var rowAbsMax = 0f;
                    for (var k = 0; k < cols; k++) { rowAbsMax = MathF.Max(rowAbsMax, MathF.Abs(src[r * cols + k])); }
                    var tol = rowAbsMax / 127f + 1e-4f;
                    Assert.True(MathF.Abs(dst[c] - src[r * cols + c]) <= tol,
                        $"row {r} col {c}: {dst[c]} vs {src[r * cols + c]} (tol {tol}).");
                }
            }
        }

        [Fact]
        public void DequantizeRow_OutOfRange_Throws()
        {
            var q8 = Q8Weight.QuantizeRows(new float[64], rowCount: 1, rowLength: 64);
            DecodeWeight w = q8;
            Assert.Throws<ArgumentOutOfRangeException>(() =>
            {
                Span<float> dst = new float[64];
                w.DequantizeRow(1, dst);   // only row 0 exists
            });
        }
    }
}
