// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Blocks
{
    public class CachedGptStackTests
    {
        private static SingleHeadWeights[] MakeZeroHeads(int dModel, int count)
        {
            var arr = new SingleHeadWeights[count];
            for (var i = 0; i < count; i++)
            {
                arr[i] = ZeroHead(dModel);
            }
            return arr;
        }

        private static SingleHeadWeights ZeroHead(int dModel)
        {
            var zero = new float[dModel * dModel];
            return new SingleHeadWeights(wq: zero, wk: zero, wv: zero, wo: zero);
        }

        private static StackWeights ZeroStack(int layers, int dModel, int dFF, int heads, float[] lmHead)
            => StackWeights.ForTest(
                layers, heads,
                _ => new BlockWeights(
                    heads:  MakeZeroHeads(dModel, heads),
                    ffnW1:  new float[dModel * dFF],
                    ffnW2:  new float[dFF * dModel]),
                finalNormGamma: Array.Empty<float>(),
                finalNormBeta:  Array.Empty<float>(),
                lmHead:         lmHead);

        private static void AssertClose(float expected, float actual, float eps = 1e-4f)
            => Assert.True(Math.Abs(expected - actual) < eps, $"Expected {expected}, got {actual}");

        [Fact]
        public void Constructor_ExposesShape()
        {
            var stack = new CachedGptStack(2, 4, 2, 8, 10, 16, 1e-5f, FeedForwardActivation.GeLU);
            Assert.Equal(2,  stack.LayerCount);
            Assert.Equal(4,  stack.DModel);
            Assert.Equal(10, stack.VocabSize);
        }

        [Fact]
        public void Decode_ZeroBlocksAndIdentityLmHead_ReturnsLayerNormLogits()
        {
            using var cache = KeyValueCache.Create(1, 1, 4, 2);
            var stack = new CachedGptStack(1, 2, 1, 2, 2, 4, 1e-5f, FeedForwardActivation.None);

            var lmHead  = new float[] { 1f, 0f,  0f, 1f };
            var weights = ZeroStack(1, 2, 2, 1, lmHead);
            var logits  = new float[2];

            cache.Advance();
            stack.Decode(new float[] { 3f, -3f }, weights, cache, 0, logits);

            AssertClose(1f,  logits[0], eps: 0.01f);
            AssertClose(-1f, logits[1], eps: 0.01f);
        }

        [Fact]
        public void Decode_StoresLastFinalHiddenAndLogits()
        {
            using var cache = KeyValueCache.Create(1, 1, 4, 2);
            var stack = new CachedGptStack(1, 2, 1, 2, 2, 4, 1e-5f, FeedForwardActivation.None);

            var weights = ZeroStack(1, 2, 2, 1, new float[] { 1f, 0f, 0f, 1f });
            var logits  = new float[2];

            cache.Advance();
            stack.Decode(new float[] { 1f, -1f }, weights, cache, 0, logits);

            var finalHidden = new float[2];
            var lastLogits  = new float[2];
            stack.GetLastFinalHidden(finalHidden);
            stack.GetLastLogits(lastLogits);

            Assert.Equal(logits[0], lastLogits[0]);
            Assert.Equal(logits[1], lastLogits[1]);
            Assert.All(finalHidden, v => Assert.False(float.IsNaN(v)));
        }

        [Fact]
        public void Blocks_ExposesBlocksAndThrowsOnOutOfRange()
        {
            var stack = new CachedGptStack(2, 2, 1, 2, 4, 8, 1e-5f, FeedForwardActivation.None);
            Assert.NotNull(stack.Blocks[0]);
            Assert.NotNull(stack.Blocks[1]);
            Assert.Throws<IndexOutOfRangeException>(() => stack.Blocks[2]);
        }
    }
}
