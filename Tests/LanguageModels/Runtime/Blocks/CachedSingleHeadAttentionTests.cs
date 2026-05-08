// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Blocks
{
    public sealed class CachedSingleHeadAttentionTests
    {
        [Fact]
        public void Decode_SingleToken_WithIdentityWeights_ReturnsHidden()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                kvHeadCount: 1,
                maxSequenceLength: 4,
                headDimension: 2);

            var decoder = new CachedSingleHeadAttention(
                dModel: 2,
                headDimension: 2,
                maxSequenceLength: 4);

            var identity = new float[]
            {
                1f, 0f,
                0f, 1f
            };

            var hidden = new float[] { 3f, 4f };
            var output = new float[2];

            cache.Advance();

            decoder.Decode(
                hidden,
                identity,
                identity,
                identity,
                [],  // bq
                [],  // bk
                [],  // bv
                identity,
                cache,
                0,  // layerIndex
                0,  // headIndex
                0,  // position
                output);

            AssertClose(3f, output[0]);
            AssertClose(4f, output[1]);

            Assert.Equal(new float[] { 3f, 4f }, cache.GetKeyReadSpan(0, 0, 0, 1).ToArray());
            Assert.Equal(new float[] { 3f, 4f }, cache.GetValueReadSpan(0, 0, 0, 1).ToArray());
        }

        [Fact]
        public void Decode_TwoTokens_WithIdentityWeights_MatchesManualCachedAttention()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                kvHeadCount: 1,
                maxSequenceLength: 4,
                headDimension: 2);

            var decoder = new CachedSingleHeadAttention(
                dModel: 2,
                headDimension: 2,
                maxSequenceLength: 4);

            var identity = new float[]
            {
                1f, 0f,
                0f, 1f
            };

            var output = new float[2];

            cache.Advance();

            decoder.Decode(
                hidden: new float[] { 1f, 0f },
                wq: identity,
                wk: identity,
                wv: identity,
                bq: [],
                bk: [],
                bv: [],
                wo: identity,
                cache,
                0,  // layerIndex
                0,  // headIndex
                0,  // position
                output);

            cache.Advance();

            decoder.Decode(
                hidden: new float[] { 0f, 1f },
                wq: identity,
                wk: identity,
                wv: identity,
                bq: [],
                bk: [],
                bv: [],
                wo: identity,
                cache,
                0,  // layerIndex
                0,  // headIndex
                1,  // position
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
        }

        [Fact]
        public void Decode_AppliesOutputProjectionAndBias()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                kvHeadCount: 1,
                maxSequenceLength: 2,
                headDimension: 2);

            var decoder = new CachedSingleHeadAttention(
                dModel: 3,
                headDimension: 2,
                maxSequenceLength: 2);

            // hidden [3] -> q/k/v [2], use first two dimensions.
            var wIn = new float[]
            {
                1f, 0f,
                0f, 1f,
                0f, 0f
            };

            // attention [2] -> output [3]
            // [a,b] * wo =
            // out0 = a*1 + b*4
            // out1 = a*2 + b*5
            // out2 = a*3 + b*6
            var wo = new float[]
            {
                1f, 2f, 3f,
                4f, 5f, 6f
            };

            var bias = new float[] { 10f, 20f, 30f };
            var output = new float[3];

            cache.Advance();

            decoder.Decode(
                hidden: new float[] { 2f, 3f, 99f },
                wq: wIn,
                wk: wIn,
                wv: wIn,
                bq: [],
                bk: [],
                bv: [],
                wo,
                cache,
                0,  // layerIndex
                0,  // headIndex
                0,  // position
                output);

            // Single token attention output == V == [2,3].
            AssertClose(14f, output[0]); // 2*1 + 3*4 (no outputBias in new API)
            AssertClose(19f, output[1]); // 2*2 + 3*5
            AssertClose(24f, output[2]); // 2*3 + 3*6
        }

        [Fact]
        public void Decode_StoresLastProjectionBuffers()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                kvHeadCount: 1,
                maxSequenceLength: 2,
                headDimension: 2);

            var decoder = new CachedSingleHeadAttention(
                dModel: 2,
                headDimension: 2,
                maxSequenceLength: 2);

            var identity = new float[]
            {
                1f, 0f,
                0f, 1f
            };

            var output = new float[2];

            cache.Advance();

            decoder.Decode(
                hidden: new float[] { 5f, 6f },
                wq: identity,
                wk: identity,
                wv: identity,
                bq: [],
                bk: [],
                bv: [],
                wo: identity,
                cache,
                0,  // layerIndex
                0,  // headIndex
                0,  // position
                output);

            var query = new float[2];
            var key = new float[2];
            var value = new float[2];
            var attention = new float[2];

            decoder.GetLastQuery(query);
            decoder.GetLastKey(key);
            decoder.GetLastValue(value);
            decoder.GetLastAttentionOutput(attention);

            Assert.Equal(new float[] { 5f, 6f }, query);
            Assert.Equal(new float[] { 5f, 6f }, key);
            Assert.Equal(new float[] { 5f, 6f }, value);
            Assert.Equal(new float[] { 5f, 6f }, attention);
        }

        [Fact]
        public void Decode_PositionNotVisible_Throws()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                kvHeadCount: 1,
                maxSequenceLength: 2,
                headDimension: 2);

            var decoder = new CachedSingleHeadAttention(
                dModel: 2,
                headDimension: 2,
                maxSequenceLength: 2);

            var identity = new float[]
            {
                1f, 0f,
                0f, 1f
            };

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                decoder.Decode(
                    hidden: new float[] { 1f, 2f },
                    wq: identity,
                    wk: identity,
                    wv: identity,
                bq: [],
                bk: [],
                bv: [],
                    wo: identity,
                    cache,
                    0,  // layerIndex
                    0,  // headIndex
                    0,  // position
                    output: new float[2]));
        }

        [Fact]
        public void Decode_InvalidShapes_Throw()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                kvHeadCount: 1,
                maxSequenceLength: 2,
                headDimension: 2);

            cache.Advance();

            var decoder = new CachedSingleHeadAttention(
                dModel: 2,
                headDimension: 2,
                maxSequenceLength: 2);

            var identity = new float[]
            {
                1f, 0f,
                0f, 1f
            };

            Assert.Throws<ArgumentException>(() =>
                decoder.Decode(
                    hidden: new float[1],
                    wq: identity,
                    wk: identity,
                    wv: identity,
                bq: [],
                bk: [],
                bv: [],
                    wo: identity,
                    cache,
                    0,  // layerIndex
                    0,  // headIndex
                    0,  // position
                    output: new float[2]));

            Assert.Throws<ArgumentException>(() =>
                decoder.Decode(
                    hidden: new float[2],
                    wq: new float[3],
                    wk: identity,
                    wv: identity,
                bq: [],
                bk: [],
                bv: [],
                    wo: identity,
                    cache,
                    0,  // layerIndex
                    0,  // headIndex
                    0,  // position
                    output: new float[2]));

            Assert.Throws<ArgumentException>(() =>
                decoder.Decode(
                    hidden: new float[2],
                    wq: identity,
                    wk: identity,
                    wv: identity,
                bq: [],
                bk: [],
                bv: [],
                    wo: identity,
                    cache,
                    0,  // layerIndex
                    0,  // headIndex
                    0,  // position
                    output: new float[1]));
        }

        [Fact]
        public void Constructor_InvalidArguments_Throw()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedSingleHeadAttention(
                    dModel: 0,
                    headDimension: 2,
                    maxSequenceLength: 2));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedSingleHeadAttention(
                    dModel: 2,
                    headDimension: 0,
                    maxSequenceLength: 2));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedSingleHeadAttention(
                    dModel: 2,
                    headDimension: 2,
                    maxSequenceLength: 0));
        }

        [Fact]
        public void GetLastBuffers_DestinationTooSmall_Throws()
        {
            var decoder = new CachedSingleHeadAttention(
                dModel: 2,
                headDimension: 2,
                maxSequenceLength: 2);

            Assert.Throws<ArgumentException>(() =>
                decoder.GetLastQuery(new float[1]));

            Assert.Throws<ArgumentException>(() =>
                decoder.GetLastKey(new float[1]));

            Assert.Throws<ArgumentException>(() =>
                decoder.GetLastValue(new float[1]));

            Assert.Throws<ArgumentException>(() =>
                decoder.GetLastAttentionOutput(new float[1]));
        }

        private static void AssertClose(float expected, float actual)
        {
            Assert.True(
                MathF.Abs(expected - actual) <= 1e-5f,
                $"Expected {expected}, actual {actual}.");
        }
    }
}
