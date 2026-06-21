// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Numerics.Tensors;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Measures the FFN SiLU activation after the TensorPrimitives sweep: scalar <c>x / (1 + MathF.Exp(-x))</c> vs
    /// the SIMD <c>Sigmoid → Multiply</c> path now in <c>CachedFeedForwardBlock.ApplySiLU</c>, over a realistic dFF
    /// span. Activation-level only — the FFN as a whole is matmul-dominated, so the E2E decode share is smaller.
    /// </summary>
    public sealed class SiluSimdPerfTests
    {
        private readonly ITestOutputHelper _output;

        public SiluSimdPerfTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact]
        public void SiLU_Simd_BeatsScalar()
        {
            const int dff = 11008; // Qwen2.5-3B intermediate size
            const int iters = 50_000;

            var input = new float[dff];
            var rng = new Random(1);
            for (var i = 0; i < dff; i++)
            {
                input[i] = (float)(rng.NextDouble() * 8.0 - 4.0);
            }

            var work = new float[dff];
            var scratch = new float[dff];

            var scalarNs = BestOf(() =>
            {
                input.CopyTo(work.AsSpan());
                for (var i = 0; i < work.Length; i++)
                {
                    var x = work[i];
                    work[i] = x / (1f + MathF.Exp(-x));
                }
            }, iters);

            var simdNs = BestOf(() =>
            {
                input.CopyTo(work.AsSpan());
                TensorPrimitives.Sigmoid(work, scratch);
                TensorPrimitives.Multiply(work, scratch, work);
            }, iters);

            _output.WriteLine($"SiLU over dFF={dff}:");
            _output.WriteLine($"  scalar (MathF.Exp)        : {scalarNs:F0} ns/call");
            _output.WriteLine($"  SIMD (Sigmoid→Multiply)   : {simdNs:F0} ns/call   ({scalarNs / simdNs:F1}x faster)");

            Assert.True(simdNs < scalarNs, $"SIMD {simdNs:F0} ns not faster than scalar {scalarNs:F0} ns");
        }

        private static double BestOf(Action body, int iters)
        {
            for (var w = 0; w < 3; w++)
            {
                body();
            }

            var best = double.MaxValue;
            for (var rep = 0; rep < 7; rep++)
            {
                var sw = Stopwatch.StartNew();
                for (var i = 0; i < iters; i++)
                {
                    body();
                }
                sw.Stop();
                best = Math.Min(best, sw.Elapsed.TotalMilliseconds * 1e6 / iters);
            }
            return best;
        }
    }
}
