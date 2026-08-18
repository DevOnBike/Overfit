// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Kernels;

namespace DevOnBike.Overfit.Tests.Core.Kernels
{
    /// <summary>
    /// The row-split dense forward matches a naive reference, and the policy that selects it fires exactly
    /// where it was measured to pay.
    ///
    /// <para><b>What this is for (`XC-78`).</b> VGG-16's `Linear(25088 -> 4096)` reads 392 MiB of weights per
    /// inference and took 9.99 ms = 41.1 GB/s, against this box's measured 90.8 GB/s read ceiling. The column
    /// split gives each worker a range of output columns, so every worker walks every input row and reads
    /// 256 B before skipping 16 KB — a different 4 KB page on every read. Splitting by input row instead makes
    /// each worker read one contiguous slab. Measured: <b>the SINGLE-THREADED row split beats the 32-worker
    /// column split</b>.</para>
    ///
    /// <para><b>The kernel is tested directly rather than through the gate.</b> Firing the gate needs a 96 MiB
    /// weight matrix, which does not belong in a suite that must stay fast. So the sweep is called at small
    /// shapes, where its arithmetic is identical, and the gate is tested separately as a pure predicate at the
    /// byte counts the benchmark actually measured.</para>
    /// </summary>
    public sealed class LinearRowSplitTests
    {
        public static TheoryData<int, int> Shapes()
        {
            return new TheoryData<int, int>
            {
                // Wider than one 64-column block, with a remainder, so the scalar tail runs too.
                { 512, 100 },

                // An exact multiple of the 64-column block, so the tail runs zero times.
                { 300, 128 },

                // Fewer input rows than there are workers on any machine, so some workers get an empty range
                // and the reduction must still be correct.
                { 3, 96 },

                // One input row: every worker but one is empty.
                { 1, 64 },

                // Narrow output, below one 512-bit vector, so only the scalar tail executes.
                { 257, 7 },
            };
        }

        [Theory]
        [MemberData(nameof(Shapes))]
        public void RowSplit_MatchesANaiveReference(int inputSize, int outputSize)
        {
            var input = Deterministic(inputSize, seed: 5);
            var weights = Deterministic(inputSize * outputSize, seed: 19);
            var bias = Deterministic(outputSize, seed: 41);
            var actual = new float[outputSize];

            LinearKernels.ForwardRowsParallel(input, weights, bias, actual, inputSize, outputSize);

            for (var j = 0; j < outputSize; j++)
            {
                var expected = bias[j];
                var absSum = MathF.Abs(bias[j]);

                for (var i = 0; i < inputSize; i++)
                {
                    var product = input[i] * weights[(i * outputSize) + j];

                    expected += product;
                    absSum += MathF.Abs(product);
                }

                // The two orders sum the same products differently, so the bound is the textbook one for a
                // K-term dot product — K * u * sum|a_i b_i|, u = 2^-24 — accumulated rather than guessed. A
                // relative test against |expected| would measure cancellation instead of the kernel.
                var bound = MathF.Max(1e-6f, 2f * inputSize * 5.9604645e-8f * absSum);

                Assert.True(
                    MathF.Abs(actual[j] - expected) <= bound,
                    $"inputSize={inputSize} outputSize={outputSize}, column {j}: kernel {actual[j]}, "
                    + $"reference {expected}, difference {MathF.Abs(actual[j] - expected)} over bound {bound}");
            }
        }

        /// <summary>
        /// Every output is written. A worker range computed wrongly leaves columns holding whatever the caller
        /// left there, and a stale plausible number reads as a correct answer.
        /// </summary>
        [Fact]
        public void RowSplit_WritesEveryOutput()
        {
            const int InputSize = 512;
            const int OutputSize = 100;
            const float Sentinel = -98765.5f;

            var input = Deterministic(InputSize, seed: 5);
            var weights = Deterministic(InputSize * OutputSize, seed: 19);
            var bias = Deterministic(OutputSize, seed: 41);
            var output = new float[OutputSize];

            for (var j = 0; j < OutputSize; j++)
            {
                output[j] = Sentinel;
            }

            LinearKernels.ForwardRowsParallel(input, weights, bias, output, InputSize, OutputSize);

            for (var j = 0; j < OutputSize; j++)
            {
                Assert.True(output[j] != Sentinel, $"column {j} was never written");
            }
        }

        /// <summary>
        /// The policy fires where the benchmark measured a win and nowhere else. The byte counts below are the
        /// ones actually measured in <c>LinearGemvRowSplitBenchmark</c>, not round numbers.
        /// </summary>
        [Theory]
        // 392 MiB — row split measured 1.49x faster.
        [InlineData(1, 25_088, 4_096, true)]
        // 192 MiB — 1.43x faster.
        [InlineData(1, 12_288, 4_096, true)]
        // 96 MiB — 1.15x faster and clearly separated. This is the first point at or above the threshold.
        [InlineData(1, 6_144, 4_096, true)]
        // 88 MiB — row was 1.05x ahead but inside the error bars, so it stays on the measured-good path.
        [InlineData(1, 5_632, 4_096, false)]
        // 80 MiB — column split measured 1.16x faster.
        [InlineData(1, 5_120, 4_096, false)]
        // 64 MiB — column split measured 1.14x faster.
        [InlineData(1, 4_096, 4_096, false)]
        // 0.77 MiB — column split measured 3.36x faster.
        [InlineData(1, 784, 256, false)]
        // Batch 2 at the largest shape: excluded on purpose, because the partial accumulators would cost
        // workers * batch * outputSize and the real win with a batch is a GEMM, not this.
        [InlineData(2, 25_088, 4_096, false)]
        public void Policy_SelectsTheRowSplitWhereItWasMeasuredToPay(
            int batchSize,
            int inputSize,
            int outputSize,
            bool expected)
        {
            var weightMebibytes = (long)inputSize * outputSize * sizeof(float) / (1024.0 * 1024.0);

            // On hardware without AVX-512 the row split is never selected, because its sweep is written at
            // 512 bits and nobody has measured the alternative. Assert that instead of the size rule there.
            var actual = LinearKernels.ShouldSplitByRow(batchSize, inputSize, outputSize);

            if (!LinearKernels.UseAvx512Linear)
            {
                Assert.False(actual, "without AVX-512 the row split must never be selected");
                return;
            }

            Assert.True(
                actual == expected,
                $"batch {batchSize}, {inputSize}x{outputSize} = {weightMebibytes:F1} MiB of weights: "
                + $"expected rowSplit={expected}, got {actual}");
        }

        private static float[] Deterministic(int length, int seed)
        {
            var values = new float[length];
            var state = (uint)(0x9E3779B9 + seed);

            for (var i = 0; i < length; i++)
            {
                state = (state * 1664525u) + 1013904223u;
                values[i] = (((state & 0x00FFFFFF) / 16777216f) * 2f) - 1f;
            }

            return values;
        }
    }
}
