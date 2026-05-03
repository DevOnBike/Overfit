// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
//
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit;

namespace DevOnBike.Overfit.Tests.LanguageModels
{
    public sealed class CachedMultiHeadAttentionTests
    {
        [Fact]
        public void Constructor_ExposesShape()
        {
            var decoder = new CachedMultiHeadAttention(
                dModel: 4,
                headCount: 2,
                maxSequenceLength: 8);

            Assert.Equal(4, decoder.DModel);
            Assert.Equal(2, decoder.HeadCount);
            Assert.Equal(2, decoder.HeadDimension);
            Assert.Equal(8, decoder.MaxSequenceLength);
        }

        [Fact]
        public void Constructor_InvalidArguments_Throw()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedMultiHeadAttention(
                    dModel: 0,
                    headCount: 2,
                    maxSequenceLength: 8));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedMultiHeadAttention(
                    dModel: 4,
                    headCount: 0,
                    maxSequenceLength: 8));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedMultiHeadAttention(
                    dModel: 4,
                    headCount: 2,
                    maxSequenceLength: 0));

            Assert.Throws<ArgumentException>(() =>
                new CachedMultiHeadAttention(
                    dModel: 5,
                    headCount: 2,
                    maxSequenceLength: 8));
        }

        [Fact]
        public void Decode_OneHead_Identity_MatchesHidden()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 1,
                maxSequenceLength: 4,
                headDimension: 2);

            var decoder = new CachedMultiHeadAttention(
                dModel: 2,
                headCount: 1,
                maxSequenceLength: 4);

            var identity = new float[]
            {
                1f, 0f,
                0f, 1f
            };

            var heads = new[] { identity };
            var output = new float[2];

            cache.Advance();

            decoder.DecodeWithoutOutputBias(
                hidden: new float[] { 3f, 4f },
                wqHeads: heads,
                wkHeads: heads,
                wvHeads: heads,
                woHeads: heads,
                cache,
                layerIndex: 0,
                position: 0,
                output);

            AssertClose(3f, output[0]);
            AssertClose(4f, output[1]);
        }

        [Fact]
        public void Decode_TwoHeads_SumsHeadOutputsAndAddsBias()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 2,
                maxSequenceLength: 4,
                headDimension: 2);

            var decoder = new CachedMultiHeadAttention(
                dModel: 4,
                headCount: 2,
                maxSequenceLength: 4);

            var head0In = new float[]
            {
                1f, 0f,
                0f, 1f,
                0f, 0f,
                0f, 0f
            };

            var head1In = new float[]
            {
                0f, 0f,
                0f, 0f,
                1f, 0f,
                0f, 1f
            };

            var head0Out = new float[]
            {
                1f, 0f, 0f, 0f,
                0f, 1f, 0f, 0f
            };

            var head1Out = new float[]
            {
                0f, 0f, 1f, 0f,
                0f, 0f, 0f, 1f
            };

            var wq = new[] { head0In, head1In };
            var wk = new[] { head0In, head1In };
            var wv = new[] { head0In, head1In };
            var wo = new[] { head0Out, head1Out };

            var outputBias = new float[] { 10f, 20f, 30f, 40f };
            var output = new float[4];

            cache.Advance();

            decoder.Decode(
                hidden: new float[] { 1f, 2f, 3f, 4f },
                wqHeads: wq,
                wkHeads: wk,
                wvHeads: wv,
                woHeads: wo,
                outputBias,
                cache,
                layerIndex: 0,
                position: 0,
                output);

            AssertClose(11f, output[0]);
            AssertClose(22f, output[1]);
            AssertClose(33f, output[2]);
            AssertClose(44f, output[3]);
        }

        [Fact]
        public void Decode_TwoTokens_UsesCacheHistoryPerHead()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 2,
                maxSequenceLength: 4,
                headDimension: 2);

            var decoder = new CachedMultiHeadAttention(
                dModel: 4,
                headCount: 2,
                maxSequenceLength: 4);

            var head0In = new float[]
            {
                1f, 0f,
                0f, 1f,
                0f, 0f,
                0f, 0f
            };

            var head1In = new float[]
            {
                0f, 0f,
                0f, 0f,
                1f, 0f,
                0f, 1f
            };

            var head0Out = new float[]
            {
                1f, 0f, 0f, 0f,
                0f, 1f, 0f, 0f
            };

            var head1Out = new float[]
            {
                0f, 0f, 1f, 0f,
                0f, 0f, 0f, 1f
            };

            var wq = new[] { head0In, head1In };
            var wk = new[] { head0In, head1In };
            var wv = new[] { head0In, head1In };
            var wo = new[] { head0Out, head1Out };

            var output = new float[4];

            cache.Advance();

            decoder.DecodeWithoutOutputBias(
                hidden: new float[] { 1f, 0f, 1f, 0f },
                wq,
                wk,
                wv,
                wo,
                cache,
                layerIndex: 0,
                position: 0,
                output);

            cache.Advance();

            decoder.DecodeWithoutOutputBias(
                hidden: new float[] { 0f, 1f, 0f, 1f },
                wq,
                wk,
                wv,
                wo,
                cache,
                layerIndex: 0,
                position: 1,
                output);

            var scale = 1f / MathF.Sqrt(2f);
            var score0 = 0f * scale;
            var score1 = 1f * scale;

            var maxScore = MathF.Max(score0, score1);
            var e0 = MathF.Exp(score0 - maxScore);
            var e1 = MathF.Exp(score1 - maxScore);
            var p0 = e0 / (e0 + e1);
            var p1 = e1 / (e0 + e1);

            AssertClose(p0, output[0]);
            AssertClose(p1, output[1]);
            AssertClose(p0, output[2]);
            AssertClose(p1, output[3]);
        }

        [Fact]
        public void Decode_WritesKeysAndValuesForAllHeads()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 2,
                maxSequenceLength: 2,
                headDimension: 2);

            var decoder = new CachedMultiHeadAttention(
                dModel: 4,
                headCount: 2,
                maxSequenceLength: 2);

            var head0In = new float[]
            {
                1f, 0f,
                0f, 1f,
                0f, 0f,
                0f, 0f
            };

            var head1In = new float[]
            {
                0f, 0f,
                0f, 0f,
                1f, 0f,
                0f, 1f
            };

            var head0Out = new float[]
            {
                1f, 0f, 0f, 0f,
                0f, 1f, 0f, 0f
            };

            var head1Out = new float[]
            {
                0f, 0f, 1f, 0f,
                0f, 0f, 0f, 1f
            };

            var wq = new[] { head0In, head1In };
            var wk = new[] { head0In, head1In };
            var wv = new[] { head0In, head1In };
            var wo = new[] { head0Out, head1Out };

            cache.Advance();

            decoder.DecodeWithoutOutputBias(
                hidden: new float[] { 7f, 8f, 9f, 10f },
                wq,
                wk,
                wv,
                wo,
                cache,
                layerIndex: 0,
                position: 0,
                output: new float[4]);

            Assert.Equal(new float[] { 7f, 8f }, cache.GetKeyReadSpan(0, 0, 0, 1).ToArray());
            Assert.Equal(new float[] { 7f, 8f }, cache.GetValueReadSpan(0, 0, 0, 1).ToArray());
            Assert.Equal(new float[] { 9f, 10f }, cache.GetKeyReadSpan(0, 1, 0, 1).ToArray());
            Assert.Equal(new float[] { 9f, 10f }, cache.GetValueReadSpan(0, 1, 0, 1).ToArray());
        }

        [Fact]
        public void Decode_PositionNotVisible_Throws()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 1,
                maxSequenceLength: 2,
                headDimension: 2);

            var decoder = new CachedMultiHeadAttention(
                dModel: 2,
                headCount: 1,
                maxSequenceLength: 2);

            var identity = new float[]
            {
                1f, 0f,
                0f, 1f
            };

            var heads = new[] { identity };

            Assert.Throws<InvalidOperationException>(() =>
                decoder.DecodeWithoutOutputBias(
                    hidden: new float[] { 1f, 2f },
                    wqHeads: heads,
                    wkHeads: heads,
                    wvHeads: heads,
                    woHeads: heads,
                    cache,
                    layerIndex: 0,
                    position: 0,
                    output: new float[2]));
        }

        [Fact]
        public void Decode_InvalidHeadCollections_Throw()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 2,
                maxSequenceLength: 2,
                headDimension: 2);

            cache.Advance();

            var decoder = new CachedMultiHeadAttention(
                dModel: 4,
                headCount: 2,
                maxSequenceLength: 2);

            var oneHead = new[]
            {
                new float[]
                {
                    1f, 0f,
                    0f, 1f,
                    0f, 0f,
                    0f, 0f
                }
            };

            Assert.Throws<ArgumentException>(() =>
                decoder.DecodeWithoutOutputBias(
                    hidden: new float[] { 1f, 2f, 3f, 4f },
                    wqHeads: oneHead,
                    wkHeads: oneHead,
                    wvHeads: oneHead,
                    woHeads: oneHead,
                    cache,
                    layerIndex: 0,
                    position: 0,
                    output: new float[4]));
        }

        [Fact]
        public void GetHeadDecoder_ReturnsRequestedHead()
        {
            var decoder = new CachedMultiHeadAttention(
                dModel: 4,
                headCount: 2,
                maxSequenceLength: 2);

            Assert.NotNull(decoder.GetHeadDecoder(0));
            Assert.NotNull(decoder.GetHeadDecoder(1));
            Assert.Throws<ArgumentOutOfRangeException>(() => decoder.GetHeadDecoder(2));
        }

        private static void AssertClose(float expected, float actual)
        {
            Assert.True(
                MathF.Abs(expected - actual) <= 1e-5f,
                $"Expected {expected}, actual {actual}.");
        }
    }
}
