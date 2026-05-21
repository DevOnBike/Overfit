// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// Prefill Phase 2: <see cref="BatchedAttentionKernel"/> must be bit-identical to
    /// running <see cref="CachedAttentionKernel.ComputeSingleHead"/> per query under
    /// the causal mask — query <c>i</c> attends over <c>[0 .. basePos + i]</c>. Covers
    /// fresh prefill (basePos=0), prefill onto an existing prefix (basePos&gt;0), and a
    /// Qwen-scale head. Both sequential and parallel batched paths are checked.
    /// </summary>
    public sealed class BatchedAttentionKernelTests
    {
        [Theory]
        [InlineData(1, 1, 8)]
        [InlineData(4, 4, 16)]    // fresh prefill, basePos=0
        [InlineData(4, 10, 16)]   // prefill onto a 6-token prefix, basePos=6
        [InlineData(8, 8, 64)]
        [InlineData(16, 64, 128)] // Qwen-scale head, basePos=48
        public void Batched_IsBitIdentical_To_PerQuerySingleHead(
            int rows, int cacheLength, int headDim)
        {
            var rng = new Random(99 + rows * 17 + cacheLength * 5 + headDim);
            var scale = 1f / MathF.Sqrt(headDim);
            var basePos = cacheLength - rows;

            var query = Random(rng, rows * headDim);
            var keys = Random(rng, cacheLength * headDim);
            var values = Random(rng, cacheLength * headDim);

            // Reference: per-query single-head, visible length = basePos + i + 1.
            var expected = new float[rows * headDim];
            var refScratch = new float[cacheLength];
            for (var i = 0; i < rows; i++)
            {
                CachedAttentionKernel.ComputeSingleHead(
                    query.AsSpan(i * headDim, headDim),
                    keys,
                    values,
                    expected.AsSpan(i * headDim, headDim),
                    refScratch,
                    sequenceLength: basePos + i + 1,
                    headDim,
                    scale);
            }

            // Sequential batched.
            var seq = new float[rows * headDim];
            var seqScratch = new float[rows * cacheLength];
            BatchedAttentionKernel.Compute(query, keys, values, seq, seqScratch, rows, cacheLength, headDim, scale);

            // Parallel batched.
            var par = new float[rows * headDim];
            var parScratch = new float[rows * cacheLength];
            BatchedAttentionKernel.ComputeParallel(query, keys, values, par, parScratch, rows, cacheLength, headDim, scale);

            for (var i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], seq[i]);
                Assert.Equal(expected[i], par[i]);
            }
        }

        private static float[] Random(Random rng, int n)
        {
            var a = new float[n];
            for (var i = 0; i < n; i++)
            {
                a[i] = rng.NextSingle() * 2f - 1f;
            }
            return a;
        }
    }
}
