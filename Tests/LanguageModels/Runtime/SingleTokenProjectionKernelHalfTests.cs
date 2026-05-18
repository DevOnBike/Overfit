// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// Parity tests for the FP16-resident projection kernel (Slot 2c).
    ///
    /// <see cref="SingleTokenProjectionKernel.ProjectHalf"/> must produce results
    /// bit-identical to <see cref="SingleTokenProjectionKernel.Project"/> fed the
    /// same <see cref="Half"/> values widened back to F32: F16 → F32 widening is
    /// exact (single represents every half value), and both paths feed the result
    /// into the same elementwise multiply-add — so the only allowed difference is
    /// none. The cases below straddle the 1024-float widen tile boundary.
    /// </summary>
    public sealed class SingleTokenProjectionKernelHalfTests
    {
        [Theory]
        [InlineData(8, 128)]     // tiny — single sub-tile
        [InlineData(64, 2048)]   // exact multiple of the 1024 widen tile
        [InlineData(33, 3000)]   // crosses tile boundary, non-multiple
        [InlineData(2048, 152)]  // many input rows, small output
        public void ProjectHalf_MatchesProject_OnHalfWidenedWeights(int inputSize, int outputSize)
        {
            var rng = new Random(1234);
            var input = NewRandom(rng, inputSize);
            var bias = NewRandom(rng, outputSize);

            var half = new Half[inputSize * outputSize];
            var f32 = new float[half.Length];
            for (var i = 0; i < half.Length; i++)
            {
                half[i] = (Half)((rng.NextSingle() - 0.5f) * 2f);
                f32[i] = (float)half[i];
            }

            var expected = new float[outputSize];
            var actual = new float[outputSize];

            SingleTokenProjectionKernel.Project(input, f32, bias, expected, inputSize, outputSize);
            SingleTokenProjectionKernel.ProjectHalf(input, half, bias, actual, inputSize, outputSize);

            Assert.Equal(expected, actual);
        }

        [Theory]
        [InlineData(16, 1024)]
        [InlineData(40, 4097)]
        public void AccumulateHalf_MatchesAccumulate_OntoExistingContent(int inputSize, int outputSize)
        {
            var rng = new Random(99);
            var input = NewRandom(rng, inputSize);

            var half = new Half[inputSize * outputSize];
            var f32 = new float[half.Length];
            for (var i = 0; i < half.Length; i++)
            {
                half[i] = (Half)((rng.NextSingle() - 0.5f) * 2f);
                f32[i] = (float)half[i];
            }

            // Accumulate adds onto existing content — seed both buffers identically.
            var expected = new float[outputSize];
            var actual = new float[outputSize];
            for (var i = 0; i < outputSize; i++)
            {
                expected[i] = actual[i] = rng.NextSingle();
            }

            SingleTokenProjectionKernel.Accumulate(input, f32, expected, inputSize, outputSize);
            SingleTokenProjectionKernel.AccumulateHalf(input, half, actual, inputSize, outputSize);

            Assert.Equal(expected, actual);
        }

        [Fact]
        public void ProjectHalf_WithoutBias_ZeroesThenAccumulates()
        {
            const int inputSize = 12;
            const int outputSize = 1500;

            var rng = new Random(7);
            var input = NewRandom(rng, inputSize);

            var half = new Half[inputSize * outputSize];
            var f32 = new float[half.Length];
            for (var i = 0; i < half.Length; i++)
            {
                half[i] = (Half)((rng.NextSingle() - 0.5f) * 2f);
                f32[i] = (float)half[i];
            }

            var expected = new float[outputSize];
            var actual = new float[outputSize];

            SingleTokenProjectionKernel.Project(input, f32, [], expected, inputSize, outputSize);
            SingleTokenProjectionKernel.ProjectHalf(input, half, [], actual, inputSize, outputSize);

            Assert.Equal(expected, actual);
        }

        private static float[] NewRandom(Random rng, int n)
        {
            var a = new float[n];
            for (var i = 0; i < n; i++)
            {
                a[i] = (rng.NextSingle() - 0.5f) * 2f;
            }
            return a;
        }
    }
}
