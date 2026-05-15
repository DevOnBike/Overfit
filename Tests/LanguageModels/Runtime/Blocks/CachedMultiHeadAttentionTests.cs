// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Blocks
{
    public sealed class CachedMultiHeadAttentionTests
    {
        [Fact]
        public void Constructor_ExposesShape()
        {
            var decoder = new CachedMultiHeadAttention(dModel: 4, headCount: 2, maxSequenceLength: 8);

            Assert.Equal(4, decoder.DModel);
            Assert.Equal(2, decoder.HeadCount);
            Assert.Equal(2, decoder.HeadDimension);
            Assert.Equal(8, decoder.MaxSequenceLength);
        }

        [Fact]
        public void Constructor_InvalidArguments_Throw()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedMultiHeadAttention(dModel: 0, headCount: 2, maxSequenceLength: 8));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedMultiHeadAttention(dModel: 4, headCount: 0, maxSequenceLength: 8));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedMultiHeadAttention(dModel: 4, headCount: 2, maxSequenceLength: 0));

            Assert.Throws<ArgumentException>(() =>
                new CachedMultiHeadAttention(dModel: 5, headCount: 2, maxSequenceLength: 8));
        }

        [Fact]
        public void Decode_OneHead_Identity_MatchesHidden()
        {
            // dModel=2, headDim=2, one head — identity Q/K/V/O → output equals input
            using var cache = KeyValueCache.Create(
                layerCount: 1, kvHeadCount: 1, maxSequenceLength: 4, headDimension: 2);

            var decoder = new CachedMultiHeadAttention(dModel: 2, headCount: 1, maxSequenceLength: 4);

            var identity = new[] { 1f, 0f, 0f, 1f };
            var bw = MakeBlockWeights(
                wq: [identity], wk: [identity],
                wv: [identity], wo: [identity]);

            var output = new float[2];
            cache.Advance();
            decoder.Decode([3f, 4f], in bw, cache, 0, 0, output);

            AssertClose(3f, output[0]);
            AssertClose(4f, output[1]);
        }

        [Fact]
        public void Decode_TwoHeads_SumsHeadOutputsAndAddsBias()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1, kvHeadCount: 2, maxSequenceLength: 4, headDimension: 2);

            var decoder = new CachedMultiHeadAttention(dModel: 4, headCount: 2, maxSequenceLength: 4);

            // head0: picks dims 0,1  head1: picks dims 2,3
            var head0In  = new[] { 1f, 0f,  0f, 1f,  0f, 0f,  0f, 0f };
            var head1In  = new[] { 0f, 0f,  0f, 0f,  1f, 0f,  0f, 1f };
            var head0Out = new[] { 1f, 0f, 0f, 0f,  0f, 1f, 0f, 0f };
            var head1Out = new[] { 0f, 0f, 1f, 0f,  0f, 0f, 0f, 1f };
            var outputBias = new[] { 10f, 20f, 30f, 40f };

            var bw = MakeBlockWeights(
                wq: [head0In, head1In], wk: [head0In, head1In],
                wv: [head0In, head1In], wo: [head0Out, head1Out],
                attentionBias: outputBias);

            var output = new float[4];
            cache.Advance();
            decoder.Decode([1f, 2f, 3f, 4f], in bw, cache, 0, 0, output);

            AssertClose(11f, output[0]);
            AssertClose(22f, output[1]);
            AssertClose(33f, output[2]);
            AssertClose(44f, output[3]);
        }

        [Fact]
        public void Decode_TwoTokens_UsesCacheHistoryPerHead()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1, kvHeadCount: 2, maxSequenceLength: 4, headDimension: 2);

            var decoder = new CachedMultiHeadAttention(dModel: 4, headCount: 2, maxSequenceLength: 4);

            var head0In  = new[] { 1f, 0f,  0f, 1f,  0f, 0f,  0f, 0f };
            var head1In  = new[] { 0f, 0f,  0f, 0f,  1f, 0f,  0f, 1f };
            var head0Out = new[] { 1f, 0f, 0f, 0f,  0f, 1f, 0f, 0f };
            var head1Out = new[] { 0f, 0f, 1f, 0f,  0f, 0f, 0f, 1f };
            var bw = MakeBlockWeights(
                wq: [head0In, head1In], wk: [head0In, head1In],
                wv: [head0In, head1In], wo: [head0Out, head1Out]);

            var output = new float[4];
            cache.Advance();
            decoder.Decode([1f, 0f, 1f, 0f], in bw, cache, 0, 0, output);
            cache.Advance();
            decoder.Decode([0f, 1f, 0f, 1f], in bw, cache, 0, 1, output);

            var scale  = 1f / MathF.Sqrt(2f);
            var score0 = 0f * scale;
            var score1 = 1f * scale;
            var maxS   = MathF.Max(score0, score1);
            var e0     = MathF.Exp(score0 - maxS);
            var e1     = MathF.Exp(score1 - maxS);
            var p0     = e0 / (e0 + e1);
            var p1     = e1 / (e0 + e1);

            AssertClose(p0, output[0]);
            AssertClose(p1, output[1]);
            AssertClose(p0, output[2]);
            AssertClose(p1, output[3]);
        }

        [Fact]
        public void Decode_WritesKeysAndValuesForAllHeads()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1, kvHeadCount: 2, maxSequenceLength: 2, headDimension: 2);

            var decoder = new CachedMultiHeadAttention(dModel: 4, headCount: 2, maxSequenceLength: 2);

            var head0In  = new[] { 1f, 0f,  0f, 1f,  0f, 0f,  0f, 0f };
            var head1In  = new[] { 0f, 0f,  0f, 0f,  1f, 0f,  0f, 1f };
            var head0Out = new[] { 1f, 0f, 0f, 0f,  0f, 1f, 0f, 0f };
            var head1Out = new[] { 0f, 0f, 1f, 0f,  0f, 0f, 0f, 1f };
            var bw = MakeBlockWeights(
                wq: [head0In, head1In], wk: [head0In, head1In],
                wv: [head0In, head1In], wo: [head0Out, head1Out]);

            cache.Advance();
            decoder.Decode([7f, 8f, 9f, 10f], in bw, cache, 0, 0, new float[4]);

            Assert.Equal(new[] { 7f, 8f }, cache.GetKeyReadSpan(0, 0, 0, 1).ToArray());
            Assert.Equal(new[] { 7f, 8f }, cache.GetValueReadSpan(0, 0, 0, 1).ToArray());
            Assert.Equal(new[] { 9f, 10f }, cache.GetKeyReadSpan(0, 1, 0, 1).ToArray());
            Assert.Equal(new[] { 9f, 10f }, cache.GetValueReadSpan(0, 1, 0, 1).ToArray());
        }

        [Fact]
        public void Decode_PositionNotVisible_Throws()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1, kvHeadCount: 1, maxSequenceLength: 2, headDimension: 2);

            var decoder = new CachedMultiHeadAttention(dModel: 2, headCount: 1, maxSequenceLength: 2);
            var identity = new[] { 1f, 0f, 0f, 1f };
            var bw = MakeBlockWeights(
                wq: [identity], wk: [identity],
                wv: [identity], wo: [identity]);

            // Cache not advanced — position 0 not visible
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                decoder.Decode([1f, 2f], in bw, cache, 0, 0, new float[2]));
        }

        [Fact]
        public void GetHeadDecoder_ReturnsRequestedHead()
        {
            var decoder = new CachedMultiHeadAttention(dModel: 4, headCount: 2, maxSequenceLength: 2);

            Assert.NotNull(decoder.GetHeadDecoder(0));
            Assert.NotNull(decoder.GetHeadDecoder(1));
            Assert.Throws<ArgumentOutOfRangeException>(() => decoder.GetHeadDecoder(2));
        }

        // ── helpers ─────────────────────────────────────────────────────────

        private static BlockWeights MakeBlockWeights(
            float[][] wq, float[][] wk, float[][] wv, float[][] wo,
            float[]? attentionBias = null)
        {
            var heads = new SingleHeadWeights[wq.Length];
            for (var h = 0; h < wq.Length; h++)
            {
                heads[h] = new SingleHeadWeights(wq: wq[h], wk: wk[h], wv: wv[h], wo: wo[h]);
            }
            return new BlockWeights(heads: heads, attentionBias: attentionBias);
        }

        private static void AssertClose(float expected, float actual, float eps = 1e-5f) =>
            Assert.True(MathF.Abs(expected - actual) <= eps,
                $"Expected {expected}, got {actual}");
    }
}
