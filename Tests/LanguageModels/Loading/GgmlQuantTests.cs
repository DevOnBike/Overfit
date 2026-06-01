// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Round-trip validation for the F32 → Q4_K encoder (<see cref="GgmlQuant"/>): quantizing then
    /// decoding (via the independent <see cref="GgmlDequant"/> / <see cref="Q4KWeight.DecodeRow"/>)
    /// must reproduce the original within Q4_K precision. This also proves the 6-bit scale/min
    /// packing is the exact inverse of the decoder's unpack.
    /// </summary>
    public sealed class GgmlQuantTests
    {
        private readonly ITestOutputHelper _out;
        public GgmlQuantTests(ITestOutputHelper output) => _out = output;

        [Theory]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(3)]
        public void QuantizeQ4K_RoundTrips_WithinPrecision(int seed)
        {
            const int inputSize = 512;   // 2 super-blocks / row
            const int outputSize = 8;
            var rng = new Random(seed);

            // Smooth-ish weights (a few scales) so per-sub-block min/max is representative.
            var f32 = new float[outputSize * inputSize];
            for (var i = 0; i < f32.Length; i++)
            {
                f32[i] = (float)(rng.NextDouble() * 2 - 1) * (0.2f + (i % 7) * 0.1f);
            }

            var bytes = GgmlQuant.QuantizeQ4_K(f32, inputSize, outputSize);
            var weight = new Q4KWeight(bytes, inputSize, outputSize);

            var row = new float[inputSize];
            double dot = 0, na = 0, nb = 0, maxAbs = 0, sumSq = 0, sumRange = 0;
            for (var o = 0; o < outputSize; o++)
            {
                weight.DecodeRow(o, row);
                var lo = float.PositiveInfinity;
                var hi = float.NegativeInfinity;
                for (var i = 0; i < inputSize; i++)
                {
                    var a = f32[o * inputSize + i];
                    var b = row[i];
                    dot += a * b; na += a * (double)a; nb += b * (double)b;
                    var e = Math.Abs(a - b);
                    if (e > maxAbs) { maxAbs = e; }
                    sumSq += e * (double)e;
                    if (a < lo) { lo = a; }
                    if (a > hi) { hi = a; }
                }
                sumRange += hi - lo;
            }

            var cos = dot / (Math.Sqrt(na) * Math.Sqrt(nb));
            var rmse = Math.Sqrt(sumSq / f32.Length);
            var avgRange = sumRange / outputSize;
            _out.WriteLine($"cosine {cos:F6} | rmse {rmse:E3} | maxAbs {maxAbs:E3} | avg sub range ~{avgRange:F2}");

            // Q4_K precision (4-bit + 6-bit sub-block scales): ~0.015·range RMSE, cosine ~0.997 — this
            // is the inherent floor of plain min/max Q4_K (ggml's iterative-RMSE refinement gains only a
            // hair). The LoRA adapter corrects residual base error anyway, so this is ample for a frozen
            // QLoRA base. A quantizer BUG (wrong packing/scales) collapses cosine well below 0.99.
            Assert.True(cos > 0.995, $"cosine too low: {cos:F6}");
            Assert.True(rmse < 0.02 * avgRange, $"rmse {rmse:E3} too high vs range {avgRange:F2}");
        }

        [Fact]
        public void QuantizeQ4K_ConstantRow_IsExact()
        {
            const int inputSize = 256;
            var f32 = new float[inputSize];
            Array.Fill(f32, 0.37f);

            var bytes = GgmlQuant.QuantizeQ4_K(f32, inputSize, 1);
            var weight = new Q4KWeight(bytes, inputSize, 1);
            var row = new float[inputSize];
            weight.DecodeRow(0, row);

            double maxAbs = 0;
            for (var i = 0; i < inputSize; i++) { maxAbs = Math.Max(maxAbs, Math.Abs(0.37f - row[i])); }
            _out.WriteLine($"constant-row maxAbs {maxAbs:E3}");
            // A constant sub-block has zero range → quantizes to a single min; near-exact.
            Assert.True(maxAbs < 1e-3, $"constant row not reproduced: {maxAbs:E3}");
        }
    }
}
