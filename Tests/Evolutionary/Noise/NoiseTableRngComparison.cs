// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests
{
    /// <summary>
    ///     Direct comparison: for seed=1 and length=1024, what's the per-dimension bias in
    ///     the first 8×4 slices when filled with a classic Random vs a fresh VectorizedRandom
    ///     backed table? If both RNGs exhibit comparable biases, the ES-reset test was
    ///     over-fitted to specific values from Random and needs a different seed (or a more
    ///     robust assertion). If only the VectorizedRandom path shows a large bias, there's
    ///     something wrong with VectorizedRandom itself — or with its interaction with
    ///     Box-Muller.
    /// </summary>
    public sealed class NoiseTableRngComparison
    {
        private readonly ITestOutputHelper _output;

        public NoiseTableRngComparison(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void CompareFirst8SlicesBias_RandomVsVectorized()
        {
            const int length = 1024;
            const int masterSeed = 1;
            const int paramCount = 4;
            const int pairs = 8;

            // Reproduce the old Random-backed fill on the same partitioning scheme that
            // PrecomputedNoiseTable used before the switch: one partition because length
            // (1024) is below MinPartitionSize (4096), so partitionRng seed is
            // HashCombine(masterSeed, 0).
            var partitionSeed = HashCombine(masterSeed, 0);
            var randomBuffer = new float[length];
            FillRangeWithRandom(randomBuffer.AsSpan(), new Random(partitionSeed));

            ReportBias("Random fill", randomBuffer, pairs, paramCount);

            // Reproduce the new VectorizedRandom-backed fill. Same partitionSeed semantics
            // except cast from int to uint preserves the bit pattern.
            var vecBuffer = new float[length];
            FillRangeWithVectorized(vecBuffer.AsSpan(), new Randomization.VectorizedRandom(unchecked((uint)partitionSeed)));

            ReportBias("VectorizedRandom fill", vecBuffer, pairs, paramCount);
        }

        // Verbatim copy of the pre-change PrecomputedNoiseTable.FillRange body, just to
        // avoid relying on any state we might have lost.
        private static void FillRangeWithRandom(Span<float> target, Random rng)
        {
            var i = 0;
            while (i + 1 < target.Length)
            {
                float u1;
                do { u1 = rng.NextSingle(); } while (u1 == 0f);
                var u2 = rng.NextSingle();
                var mag = MathF.Sqrt(-2f * MathF.Log(u1));
                var (sin, cos) = MathF.SinCos(2f * MathF.PI * u2);
                target[i] = mag * cos;
                target[i + 1] = mag * sin;
                i += 2;
            }
            if (i < target.Length)
            {
                float u1;
                do { u1 = rng.NextSingle(); } while (u1 == 0f);
                var u2 = rng.NextSingle();
                target[i] = MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * u2);
            }
        }

        private static void FillRangeWithVectorized(Span<float> target, Randomization.VectorizedRandom rng)
        {
            // Same algorithm, just with VectorizedRandom as the uniform source. This is
            // literally what PrecomputedNoiseTable does after the switch.
            var i = 0;
            while (i + 1 < target.Length)
            {
                float u1;
                do { u1 = rng.NextSingle(); } while (u1 == 0f);
                var u2 = rng.NextSingle();
                var mag = MathF.Sqrt(-2f * MathF.Log(u1));
                var (sin, cos) = MathF.SinCos(2f * MathF.PI * u2);
                target[i] = mag * cos;
                target[i + 1] = mag * sin;
                i += 2;
            }
            if (i < target.Length)
            {
                float u1;
                do { u1 = rng.NextSingle(); } while (u1 == 0f);
                var u2 = rng.NextSingle();
                target[i] = MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * u2);
            }
        }

        private void ReportBias(string label, float[] buffer, int pairs, int paramCount)
        {
            var sumsPerDim = new float[paramCount];
            for (var p = 0; p < pairs; p++)
            {
                for (var j = 0; j < paramCount; j++)
                {
                    sumsPerDim[j] += buffer[(p * paramCount) + j];
                }
            }

            var overallMean = 0.0;
            for (var i = 0; i < 128; i++)
            {
                overallMean += buffer[i];
            }
            overallMean /= 128;

            _output.WriteLine($"--- {label} ---");
            _output.WriteLine($"  mean over first 128 samples: {overallMean:F4}");
            for (var j = 0; j < paramCount; j++)
            {
                _output.WriteLine($"  dim[{j}] sum over first 8 4-slices = {sumsPerDim[j]:F4}");
            }
        }

        private static int HashCombine(int a, int b)
        {
            var h = (uint)a;
            h ^= (uint)b + 0x9E3779B9u + (h << 6) + (h >> 2);
            return (int)h;
        }
    }
}