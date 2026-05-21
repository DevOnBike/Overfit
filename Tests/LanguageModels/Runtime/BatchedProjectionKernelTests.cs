// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// Prefill Phase 1: <see cref="BatchedProjectionKernel"/> must be bit-identical
    /// to running <see cref="SingleTokenProjectionKernel.Project"/> on each row
    /// independently — for N=1, N&gt;1, with/without bias, with output sizes above
    /// the L1 tile width, and with zeros in the input (the <c>x == 0</c> skip path).
    /// Both the sequential and parallel batched paths are checked.
    /// </summary>
    public sealed class BatchedProjectionKernelTests
    {
        [Theory]
        [InlineData(1, 64, 32, true)]
        [InlineData(1, 64, 32, false)]
        [InlineData(7, 128, 96, true)]
        [InlineData(13, 257, 50, false)]
        [InlineData(4, 96, 5000, true)]   // outputSize > OutputTile (2048) — exercises tiling
        [InlineData(8, 2048, 2048, true)] // GPT-2-scale-ish
        public void Batched_IsBitIdentical_To_PerRowSingleToken(
            int rows, int inputSize, int outputSize, bool withBias)
        {
            var rng = new Random(1234 + rows * 31 + inputSize * 7 + outputSize);

            var input = new float[rows * inputSize];
            for (var i = 0; i < input.Length; i++)
            {
                // Inject ~10% exact zeros so the x==0 skip path is exercised identically.
                input[i] = rng.NextSingle() < 0.1f ? 0f : (rng.NextSingle() * 2f - 1f);
            }

            var weights = new float[inputSize * outputSize];
            for (var i = 0; i < weights.Length; i++)
            {
                weights[i] = (rng.NextSingle() * 2f - 1f) * 0.05f;
            }

            var bias = withBias ? new float[outputSize] : [];
            for (var i = 0; i < bias.Length; i++)
            {
                bias[i] = rng.NextSingle() * 2f - 1f;
            }

            // Reference: per-row single-token projection.
            var expected = new float[rows * outputSize];
            for (var n = 0; n < rows; n++)
            {
                SingleTokenProjectionKernel.Project(
                    input.AsSpan(n * inputSize, inputSize),
                    weights,
                    bias,
                    expected.AsSpan(n * outputSize, outputSize),
                    inputSize,
                    outputSize);
            }

            // Sequential batched.
            var seq = new float[rows * outputSize];
            BatchedProjectionKernel.Project(input, rows, weights, bias, seq, inputSize, outputSize);

            // Parallel batched.
            var par = new float[rows * outputSize];
            BatchedProjectionKernel.ProjectParallel(input, rows, weights, bias, par, inputSize, outputSize);

            for (var i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], seq[i]);   // exact — same accumulation order
                Assert.Equal(expected[i], par[i]);   // exact — disjoint output columns
            }
        }
    }
}
