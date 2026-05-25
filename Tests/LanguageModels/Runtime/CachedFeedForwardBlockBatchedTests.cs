// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// Prefill Phase 3 (FFN slice): <see cref="CachedFeedForwardBlock.DecodeBatched"/>
    /// must be bit-identical to running <see cref="CachedFeedForwardBlock.Decode"/> on
    /// each row — same projections (via <see cref="BatchedProjectionKernel"/>) and the
    /// same element-wise activation across the whole <c>[rows × dFF]</c> intermediate.
    /// </summary>
    public sealed class CachedFeedForwardBlockBatchedTests
    {
        [Theory]
        [InlineData(1, 32, 128, FeedForwardActivation.GeLU)]
        [InlineData(7, 64, 256, FeedForwardActivation.GeLU)]
        [InlineData(13, 96, 384, FeedForwardActivation.ReLU)]
        [InlineData(8, 256, 1024, FeedForwardActivation.GeLU)]
        public void DecodeBatched_IsBitIdentical_To_PerRowDecode(
            int rows, int dModel, int dFF, FeedForwardActivation activation)
        {
            var rng = new Random(7 + rows * 13 + dModel + dFF);

            var hidden = Random(rng, rows * dModel, 1f);
            var w1 = Random(rng, dModel * dFF, 0.05f);
            var b1 = Random(rng, dFF, 0.1f);
            var w2 = Random(rng, dFF * dModel, 0.05f);
            var b2 = Random(rng, dModel, 0.1f);

            var block = new CachedFeedForwardBlock(dModel, dFF, activation);

            // Reference: per-row single-token decode.
            var expected = new float[rows * dModel];
            for (var n = 0; n < rows; n++)
            {
                block.Decode(
                    hidden.AsSpan(n * dModel, dModel),
                    w1, b1, w2, b2,
                    expected.AsSpan(n * dModel, dModel));
            }

            // Batched.
            var batched = new float[rows * dModel];
            block.DecodeBatched(hidden, rows, w1, b1, w2, b2, batched);

            for (var i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], batched[i]);
            }
        }

        private static float[] Random(Random rng, int n, float scale)
        {
            var a = new float[n];
            for (var i = 0; i < n; i++)
            {
                a[i] = (rng.NextSingle() * 2f - 1f) * scale;
            }
            return a;
        }
    }
}
