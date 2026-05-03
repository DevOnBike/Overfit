// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
//
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit;

namespace DevOnBike.Overfit.Tests.LanguageModels
{
    public sealed class CachedAttentionKernelTests
    {
        [Fact]
        public void ComputeSingleHead_ZeroLength_ClearsOutput()
        {
            var query = new float[] { 1, 2 };
            var keys = Array.Empty<float>();
            var values = Array.Empty<float>();
            var output = new float[] { 9, 9 };
            var scratch = Array.Empty<float>();

            CachedAttentionKernel.ComputeSingleHead(
                query,
                keys,
                values,
                output,
                scratch,
                sequenceLength: 0,
                headDimension: 2,
                scale: 1f);

            Assert.Equal(new float[] { 0, 0 }, output);
        }

        [Fact]
        public void ComputeSingleHead_SingleToken_ReturnsValueVector()
        {
            var query = new float[] { 1, 0, 0 };
            var keys = new float[] { 0.5f, 0.25f, -1f };
            var values = new float[] { 10f, 20f, 30f };
            var output = new float[3];
            var scratch = new float[1];

            CachedAttentionKernel.ComputeSingleHead(
                query,
                keys,
                values,
                output,
                scratch,
                sequenceLength: 1,
                headDimension: 3,
                scale: 1f);

            AssertClose(10f, output[0]);
            AssertClose(20f, output[1]);
            AssertClose(30f, output[2]);
        }

        [Fact]
        public void ComputeSingleHead_TwoTokens_MatchesManualSoftmaxWeightedSum()
        {
            var query = new float[] { 1f, 0f };
            var keys = new float[]
            {
                1f, 0f,
                0f, 1f
            };
            var values = new float[]
            {
                10f, 0f,
                0f, 20f
            };
            var output = new float[2];
            var scratch = new float[2];

            CachedAttentionKernel.ComputeSingleHead(
                query,
                keys,
                values,
                output,
                scratch,
                sequenceLength: 2,
                headDimension: 2,
                scale: 1f);

            var e1 = MathF.Exp(1f);
            var e0 = MathF.Exp(0f);
            var p0 = e1 / (e1 + e0);
            var p1 = e0 / (e1 + e0);

            AssertClose(10f * p0, output[0]);
            AssertClose(20f * p1, output[1]);
        }

        [Fact]
        public void ComputeSingleHead_IsNumericallyStableForLargeScores()
        {
            var query = new float[] { 1000f, 1000f };
            var keys = new float[]
            {
                1000f, 1000f,
                999f, 999f
            };
            var values = new float[]
            {
                1f, 2f,
                3f, 4f
            };
            var output = new float[2];
            var scratch = new float[2];

            CachedAttentionKernel.ComputeSingleHead(
                query,
                keys,
                values,
                output,
                scratch,
                sequenceLength: 2,
                headDimension: 2,
                scale: 1f);

            Assert.False(float.IsNaN(output[0]));
            Assert.False(float.IsNaN(output[1]));
            Assert.False(float.IsInfinity(output[0]));
            Assert.False(float.IsInfinity(output[1]));

            AssertClose(1f, output[0]);
            AssertClose(2f, output[1]);
        }

        [Fact]
        public void ComputeSingleHead_DoesNotUseStaleOutputValues()
        {
            var query = new float[] { 1f };
            var keys = new float[] { 1f };
            var values = new float[] { 7f };
            var output = new float[] { 123f };
            var scratch = new float[1];

            CachedAttentionKernel.ComputeSingleHead(
                query,
                keys,
                values,
                output,
                scratch,
                sequenceLength: 1,
                headDimension: 1,
                scale: 1f);

            AssertClose(7f, output[0]);
        }

        [Fact]
        public void ComputeSingleHead_InvalidArguments_Throw()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                CachedAttentionKernel.ComputeSingleHead(
                    query: new float[1],
                    keys: new float[1],
                    values: new float[1],
                    output: new float[1],
                    scoreScratch: new float[1],
                    sequenceLength: -1,
                    headDimension: 1,
                    scale: 1f));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                CachedAttentionKernel.ComputeSingleHead(
                    query: new float[1],
                    keys: new float[1],
                    values: new float[1],
                    output: new float[1],
                    scoreScratch: new float[1],
                    sequenceLength: 1,
                    headDimension: 0,
                    scale: 1f));

            Assert.Throws<ArgumentException>(() =>
                CachedAttentionKernel.ComputeSingleHead(
                    query: new float[0],
                    keys: new float[1],
                    values: new float[1],
                    output: new float[1],
                    scoreScratch: new float[1],
                    sequenceLength: 1,
                    headDimension: 1,
                    scale: 1f));

            Assert.Throws<ArgumentException>(() =>
                CachedAttentionKernel.ComputeSingleHead(
                    query: new float[1],
                    keys: new float[0],
                    values: new float[1],
                    output: new float[1],
                    scoreScratch: new float[1],
                    sequenceLength: 1,
                    headDimension: 1,
                    scale: 1f));

            Assert.Throws<ArgumentException>(() =>
                CachedAttentionKernel.ComputeSingleHead(
                    query: new float[1],
                    keys: new float[1],
                    values: new float[0],
                    output: new float[1],
                    scoreScratch: new float[1],
                    sequenceLength: 1,
                    headDimension: 1,
                    scale: 1f));

            Assert.Throws<ArgumentException>(() =>
                CachedAttentionKernel.ComputeSingleHead(
                    query: new float[1],
                    keys: new float[1],
                    values: new float[1],
                    output: new float[0],
                    scoreScratch: new float[1],
                    sequenceLength: 1,
                    headDimension: 1,
                    scale: 1f));

            Assert.Throws<ArgumentException>(() =>
                CachedAttentionKernel.ComputeSingleHead(
                    query: new float[1],
                    keys: new float[1],
                    values: new float[1],
                    output: new float[1],
                    scoreScratch: new float[0],
                    sequenceLength: 1,
                    headDimension: 1,
                    scale: 1f));
        }

        [Fact]
        public void ComputeSingleHeadFromCache_ReadsKeysAndValuesFromCache()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 1,
                maxSequenceLength: 2,
                headDimension: 2);

            var k0 = cache.GetKeyWriteSpan(0, 0, 0);
            k0[0] = 1f;
            k0[1] = 0f;

            var v0 = cache.GetValueWriteSpan(0, 0, 0);
            v0[0] = 10f;
            v0[1] = 0f;

            var k1 = cache.GetKeyWriteSpan(0, 0, 1);
            k1[0] = 0f;
            k1[1] = 1f;

            var v1 = cache.GetValueWriteSpan(0, 0, 1);
            v1[0] = 0f;
            v1[1] = 20f;

            cache.Advance(2);

            var output = new float[2];
            var scratch = new float[2];

            CachedAttentionKernel.ComputeSingleHeadFromCache(
                cache.AsReader(),
                layerIndex: 0,
                headIndex: 0,
                query: new float[] { 1f, 0f },
                output,
                scratch,
                scale: 1f);

            var e1 = MathF.Exp(1f);
            var e0 = MathF.Exp(0f);
            var p0 = e1 / (e1 + e0);
            var p1 = e0 / (e1 + e0);

            AssertClose(10f * p0, output[0]);
            AssertClose(20f * p1, output[1]);
        }

        private static void AssertClose(float expected, float actual)
        {
            Assert.True(
                MathF.Abs(expected - actual) <= 1e-4f,
                $"Expected {expected}, actual {actual}.");
        }
    }
}
