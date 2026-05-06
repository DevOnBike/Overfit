// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Blocks
{
    public class CachedMultiHeadAttentionTests
    {
        private static float[] Identity(int n)
        {
            var m = new float[n * n];
            for (var i = 0; i < n; i++) m[i * n + i] = 1f;
            return m;
        }

        private static void AssertClose(float expected, float actual, float eps = 1e-4f)
            => Assert.True(Math.Abs(expected - actual) < eps, $"Expected {expected}, got {actual}");

        [Fact]
        public void Constructor_ExposesShape()
        {
            var decoder = new CachedMultiHeadAttention(dModel: 4, headCount: 2, maxSequenceLength: 8);
            Assert.Equal(4, decoder.DModel);
            Assert.Equal(2, decoder.HeadCount);
        }

        [Fact]
        public void Decode_OneHead_Identity_MatchesHidden()
        {
            using var cache   = KeyValueCache.Create(1, 1, 4, 2);
            var decoder = new CachedMultiHeadAttention(2, 1, 4);

            var id     = Identity(2);
            var head   = new SingleHeadWeights(wq: id, wk: id, wv: id, wo: id);
            var bw     = new BlockWeights(heads: new[] { head });
            var output = new float[2];

            cache.Advance();
            decoder.Decode(new float[] { 3f, 4f }, in bw, cache, 0, 0, output);

            AssertClose(3f, output[0]);
            AssertClose(4f, output[1]);
        }

        [Fact]
        public void Decode_TwoHeads_SumsOutputsAndAddsBias()
        {
            using var cache   = KeyValueCache.Create(1, 2, 4, 2);
            var decoder = new CachedMultiHeadAttention(4, 2, 4);

            var head0In  = new float[] { 1f, 0f,  0f, 1f,  0f, 0f,  0f, 0f };  // picks hidden dims 0,1
            var head1In  = new float[] { 0f, 0f,  0f, 0f,  1f, 0f,  0f, 1f };  // picks hidden dims 2,3
            // wo: headDim=2 → dModel=4, laid out [headDim rows × dModel cols]
            // head0Out maps head space → output dims 0,1
            var head0Out = new float[] { 1f, 0f, 0f, 0f,  0f, 1f, 0f, 0f };  // head→output dims 0,1
            // head1Out maps head space → output dims 2,3
            var head1Out = new float[] { 0f, 0f, 1f, 0f,  0f, 0f, 0f, 1f };  // head→output dims 2,3
            var bias     = new float[] { 10f, 20f, 30f, 40f };

            var bw = new BlockWeights(
                heads: new[]
                {
                    new SingleHeadWeights(wq: head0In, wk: head0In, wv: head0In, wo: head0Out),
                    new SingleHeadWeights(wq: head1In, wk: head1In, wv: head1In, wo: head1Out),
                },
                attentionBias: bias);

            var output = new float[4];
            cache.Advance();
            decoder.Decode(new float[] { 1f, 2f, 3f, 4f }, in bw, cache, 0, 0, output);

            // head0: q=k=v = head0In^T × [1,2,3,4] = [1,2], wo → [1,2,0,0]
            // head1: q=k=v = head1In^T × [1,2,3,4] = [3,4], wo → [0,0,3,4]
            // output = bias + head0 + head1 = [11,22,33,44]
            AssertClose(11f, output[0]);
            AssertClose(22f, output[1]);
            AssertClose(33f, output[2]);
            AssertClose(44f, output[3]);
        }

        [Fact]
        public void Decode_WritesKeysAndValuesForAllHeads()
        {
            using var cache   = KeyValueCache.Create(1, 2, 4, 2);
            var decoder = new CachedMultiHeadAttention(4, 2, 4);

            var head0In = new float[] { 1f, 0f,  0f, 1f,  0f, 0f,  0f, 0f };  // picks dims 0,1
            var head1In = new float[] { 0f, 0f,  0f, 0f,  1f, 0f,  0f, 1f };  // picks dims 2,3
            var wo      = new float[] { 1f, 0f, 0f, 0f,  0f, 1f, 0f, 0f };  // headDim=2 → dModel=4

            var bw = new BlockWeights(heads: new[]
            {
                new SingleHeadWeights(wq: head0In, wk: head0In, wv: head0In, wo: wo),
                new SingleHeadWeights(wq: head1In, wk: head1In, wv: head1In, wo: wo),
            });

            cache.Advance();
            decoder.Decode(new float[] { 7f, 8f, 9f, 10f }, in bw, cache, 0, 0, new float[4]);

            Assert.Equal(new float[] { 7f, 8f  }, cache.GetKeyReadSpan(0, 0, 0, 1).ToArray());
            Assert.Equal(new float[] { 9f, 10f }, cache.GetKeyReadSpan(0, 1, 0, 1).ToArray());
        }

        [Fact]
        public void GetHeadDecoder_ReturnsRequestedHead()
        {
            var decoder = new CachedMultiHeadAttention(4, 2, 8);
            Assert.NotNull(decoder.GetHeadDecoder(0));
            Assert.NotNull(decoder.GetHeadDecoder(1));
            Assert.Throws<ArgumentOutOfRangeException>(() => decoder.GetHeadDecoder(2));
        }
    }
}
