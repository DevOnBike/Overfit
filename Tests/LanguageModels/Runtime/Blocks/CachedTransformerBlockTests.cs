// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Blocks
{
    public class CachedTransformerBlockTests
    {
        private static float[] Identity(int n)
        {
            var m = new float[n * n];
            for (var i = 0; i < n; i++)
            {
                m[i * n + i] = 1f;
            }
            return m;
        }

        private static SingleHeadWeights ZeroHead(int dModel)
        {
            var zero = new float[dModel * dModel];
            return new SingleHeadWeights(wq: zero, wk: zero, wv: zero, wo: zero);
        }

        private static SingleHeadWeights IdentityHead(int dModel)
        {
            var id = Identity(dModel);
            return new SingleHeadWeights(wq: id, wk: id, wv: id, wo: id);
        }

        private static void AssertClose(float expected, float actual, float eps = 1e-4f)
            => Assert.True(Math.Abs(expected - actual) < eps, $"Expected {expected}, got {actual}");

        [Fact]
        public void Constructor_ExposesShape()
        {
            var block = new CachedTransformerBlock(4, 2, 8, 16, feedForwardActivation: FeedForwardActivation.GeLU);
            Assert.Equal(4, block.DModel);
            Assert.Equal(2, block.HeadCount);
        }

        [Fact]
        public void Decode_WithZeroAttentionAndZeroFfn_ReturnsInput()
        {
            using var cache = KeyValueCache.Create(1, 1, 4, 2);
            var block = new CachedTransformerBlock(2, 1, 2, 4, feedForwardActivation: FeedForwardActivation.None);

            var bw     = new BlockWeights(heads: new[] { ZeroHead(2) }, ffnW1: new float[4], ffnW2: new float[4]);
            var output = new float[2];

            cache.Advance();
            block.Decode(new float[] { 3f, 4f }, in bw, cache, 0, 0, output);

            AssertClose(3f, output[0]);
            AssertClose(4f, output[1]);
        }

        [Fact]
        public void Decode_WithIdentityAttentionAndZeroFfn_AddsLayerNorm1Output()
        {
            using var cache = KeyValueCache.Create(1, 1, 4, 2);
            var block = new CachedTransformerBlock(2, 1, 2, 4, feedForwardActivation: FeedForwardActivation.None);

            var bw     = new BlockWeights(heads: new[] { IdentityHead(2) }, ffnW1: new float[4], ffnW2: new float[4]);
            var output = new float[2];

            cache.Advance();
            block.Decode(new float[] { 1f, -1f }, in bw, cache, 0, 0, output);

            AssertClose(2f, output[0], eps: 0.01f);
            AssertClose(-2f, output[1], eps: 0.01f);
        }

        [Fact]
        public void Decode_WritesKeysAndValuesFromLayerNorm1Output()
        {
            using var cache = KeyValueCache.Create(1, 1, 4, 2);
            var block = new CachedTransformerBlock(2, 1, 2, 4, feedForwardActivation: FeedForwardActivation.None);

            var bw = new BlockWeights(heads: new[] { IdentityHead(2) }, ffnW1: new float[4], ffnW2: new float[4]);

            cache.Advance();
            block.Decode(new float[] { 3f, 4f }, in bw, cache, 0, 0, new float[2]);

            var key = cache.GetKeyReadSpan(0, 0, 0, 1).ToArray();
            Assert.Equal(2, key.Length);
            Assert.All(key, v => Assert.False(float.IsNaN(v)));
        }
    }
}
