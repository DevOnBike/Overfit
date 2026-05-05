// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Blocks
{
    public class CachedTransformerBlockTests
    {
        [Fact]
        public void Constructor_ExposesShape()
        {
            var block = new CachedTransformerBlock(
                dModel: 4,
                headCount: 2,
                dFF: 8,
                maxSequenceLength: 16,
                layerNormEpsilon: 1e-5f,
                feedForwardActivation: FeedForwardActivation.ReLU);

            Assert.Equal(4, block.DModel);
            Assert.Equal(2, block.HeadCount);
            Assert.Equal(2, block.HeadDimension);
            Assert.Equal(8, block.DFF);
            Assert.Equal(16, block.MaxSequenceLength);
            Assert.Equal(1e-5f, block.LayerNormEpsilon);
            Assert.Equal(FeedForwardActivation.ReLU, block.FeedForwardActivation);
        }

        [Fact]
        public void Constructor_InvalidArguments_Throw()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedTransformerBlock(
                    dModel: 0,
                    headCount: 1,
                    dFF: 4,
                    maxSequenceLength: 4));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedTransformerBlock(
                    dModel: 4,
                    headCount: 0,
                    dFF: 4,
                    maxSequenceLength: 4));

            Assert.Throws<ArgumentException>(() =>
                new CachedTransformerBlock(
                    dModel: 5,
                    headCount: 2,
                    dFF: 4,
                    maxSequenceLength: 4));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedTransformerBlock(
                    dModel: 4,
                    headCount: 2,
                    dFF: 0,
                    maxSequenceLength: 4));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedTransformerBlock(
                    dModel: 4,
                    headCount: 2,
                    dFF: 4,
                    maxSequenceLength: 0));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedTransformerBlock(
                    dModel: 4,
                    headCount: 2,
                    dFF: 4,
                    maxSequenceLength: 4,
                    layerNormEpsilon: 0f));
        }

        [Fact]
        public void Decode_WithZeroAttentionAndZeroFfn_ReturnsInput()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 1,
                maxSequenceLength: 4,
                headDimension: 2);

            var block = new CachedTransformerBlock(
                dModel: 2,
                headCount: 1,
                dFF: 2,
                maxSequenceLength: 4,
                feedForwardActivation: FeedForwardActivation.None);

            var zeroHeadWeights = CreateZeroHeadWeights();
            var zeroHeadBiases = CreateZeroHeadBiases();
            var ffnW1 = new float[2 * 2];
            var ffnW2 = new float[2 * 2];
            var output = new float[2];

            cache.Advance();

            block.DecodeWithoutLayerNormAffine(
                input: new float[] { 3f, 4f },
                wqHeads: zeroHeadWeights,
                wkHeads: zeroHeadWeights,
                wvHeads: zeroHeadWeights,
                bqHeads: zeroHeadBiases,
                bkHeads: zeroHeadBiases,
                bvHeads: zeroHeadBiases,
                woHeads: zeroHeadWeights,
                attentionOutputBias: [],
                ffnW1: ffnW1,
                ffnB1: [],
                ffnW2: ffnW2,
                ffnB2: [],
                cache: cache,
                layerIndex: 0,
                position: 0,
                output: output);

            AssertClose(3f, output[0]);
            AssertClose(4f, output[1]);
        }

        [Fact]
        public void Decode_WithIdentityAttentionAndZeroFfn_AddsLayerNorm1Output()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 1,
                maxSequenceLength: 4,
                headDimension: 2);

            var block = new CachedTransformerBlock(
                dModel: 2,
                headCount: 1,
                dFF: 2,
                maxSequenceLength: 4,
                feedForwardActivation: FeedForwardActivation.None);

            var identity = new float[]
            {
                1f, 0f,
                0f, 1f
            };

            var heads = new[] { identity };
            var biases = CreateZeroHeadBiases();
            var ffnW1 = new float[2 * 2];
            var ffnW2 = new float[2 * 2];
            var input = new float[] { 1f, -1f };
            var output = new float[2];

            cache.Advance();

            block.DecodeWithoutLayerNormAffine(
                input: input,
                wqHeads: heads,
                wkHeads: heads,
                wvHeads: heads,
                bqHeads: biases,
                bkHeads: biases,
                bvHeads: biases,
                woHeads: heads,
                attentionOutputBias: [],
                ffnW1: ffnW1,
                ffnB1: [],
                ffnW2: ffnW2,
                ffnB2: [],
                cache: cache,
                layerIndex: 0,
                position: 0,
                output: output);

            var ln1 = LayerNorm(input, epsilon: 1e-5f);

            AssertClose(input[0] + ln1[0], output[0]);
            AssertClose(input[1] + ln1[1], output[1]);
        }

        [Fact]
        public void Decode_WithZeroAttentionAndIdentityFfn_AddsLayerNorm2Output()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 1,
                maxSequenceLength: 4,
                headDimension: 2);

            var block = new CachedTransformerBlock(
                dModel: 2,
                headCount: 1,
                dFF: 2,
                maxSequenceLength: 4,
                feedForwardActivation: FeedForwardActivation.None);

            var zeroHeadWeights = CreateZeroHeadWeights();
            var zeroHeadBiases = CreateZeroHeadBiases();
            var identity = new float[]
            {
                1f, 0f,
                0f, 1f
            };
            var input = new float[] { 1f, -1f };
            var output = new float[2];

            cache.Advance();

            block.DecodeWithoutLayerNormAffine(
                input: input,
                wqHeads: zeroHeadWeights,
                wkHeads: zeroHeadWeights,
                wvHeads: zeroHeadWeights,
                bqHeads: zeroHeadBiases,
                bkHeads: zeroHeadBiases,
                bvHeads: zeroHeadBiases,
                woHeads: zeroHeadWeights,
                attentionOutputBias: [],
                ffnW1: identity,
                ffnB1: [],
                ffnW2: identity,
                ffnB2: [],
                cache: cache,
                layerIndex: 0,
                position: 0,
                output: output);

            var ln2 = LayerNorm(input, epsilon: 1e-5f);

            AssertClose(input[0] + ln2[0], output[0]);
            AssertClose(input[1] + ln2[1], output[1]);
        }

        [Fact]
        public void Decode_WritesKeysAndValuesFromLayerNorm1Output()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 1,
                maxSequenceLength: 4,
                headDimension: 2);

            var block = new CachedTransformerBlock(
                dModel: 2,
                headCount: 1,
                dFF: 2,
                maxSequenceLength: 4,
                feedForwardActivation: FeedForwardActivation.None);

            var identity = new float[]
            {
                1f, 0f,
                0f, 1f
            };

            var heads = new[] { identity };
            var biases = CreateZeroHeadBiases();
            var ffnW1 = new float[2 * 2];
            var ffnW2 = new float[2 * 2];
            var input = new float[] { 1f, -1f };

            cache.Advance();

            block.DecodeWithoutLayerNormAffine(
                input: input,
                wqHeads: heads,
                wkHeads: heads,
                wvHeads: heads,
                bqHeads: biases,
                bkHeads: biases,
                bvHeads: biases,
                woHeads: heads,
                attentionOutputBias: [],
                ffnW1: ffnW1,
                ffnB1: [],
                ffnW2: ffnW2,
                ffnB2: [],
                cache: cache,
                layerIndex: 0,
                position: 0,
                output: new float[2]);

            var ln1 = LayerNorm(input, epsilon: 1e-5f);
            var key = cache.GetKeyReadSpan(0, 0, 0, 1);
            var value = cache.GetValueReadSpan(0, 0, 0, 1);

            AssertClose(ln1[0], key[0]);
            AssertClose(ln1[1], key[1]);
            AssertClose(ln1[0], value[0]);
            AssertClose(ln1[1], value[1]);
        }

        [Fact]
        public void Decode_StoresLastIntermediateBuffers()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 1,
                maxSequenceLength: 4,
                headDimension: 2);

            var block = new CachedTransformerBlock(
                dModel: 2,
                headCount: 1,
                dFF: 2,
                maxSequenceLength: 4,
                feedForwardActivation: FeedForwardActivation.None);

            var zeroHeadWeights = CreateZeroHeadWeights();
            var zeroHeadBiases = CreateZeroHeadBiases();
            var ffnW1 = new float[2 * 2];
            var ffnW2 = new float[2 * 2];

            cache.Advance();

            block.DecodeWithoutLayerNormAffine(
                input: new float[] { 1f, -1f },
                wqHeads: zeroHeadWeights,
                wkHeads: zeroHeadWeights,
                wvHeads: zeroHeadWeights,
                bqHeads: zeroHeadBiases,
                bkHeads: zeroHeadBiases,
                bvHeads: zeroHeadBiases,
                woHeads: zeroHeadWeights,
                attentionOutputBias: [],
                ffnW1: ffnW1,
                ffnB1: [],
                ffnW2: ffnW2,
                ffnB2: [],
                cache: cache,
                layerIndex: 0,
                position: 0,
                output: new float[2]);

            var ln1 = new float[2];
            var attn = new float[2];
            var residual = new float[2];
            var ln2 = new float[2];
            var ffn = new float[2];

            block.GetLastLayerNorm1Output(ln1);
            block.GetLastAttentionOutput(attn);
            block.GetLastAfterAttentionResidual(residual);
            block.GetLastLayerNorm2Output(ln2);
            block.GetLastFeedForwardOutput(ffn);

            Assert.DoesNotContain(ln1, float.IsNaN);
            Assert.Equal(new float[] { 0f, 0f }, attn);
            Assert.Equal(new float[] { 1f, -1f }, residual);
            Assert.DoesNotContain(ln2, float.IsNaN);
            Assert.Equal(new float[] { 0f, 0f }, ffn);
        }

        [Fact]
        public void Decode_PositionNotVisible_Throws()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 1,
                maxSequenceLength: 4,
                headDimension: 2);

            var block = new CachedTransformerBlock(
                dModel: 2,
                headCount: 1,
                dFF: 2,
                maxSequenceLength: 4);

            var zeroHeadWeights = CreateZeroHeadWeights();
            var zeroHeadBiases = CreateZeroHeadBiases();

            Assert.Throws<InvalidOperationException>(() =>
                block.DecodeWithoutLayerNormAffine(
                    input: new float[] { 1f, -1f },
                    wqHeads: zeroHeadWeights,
                    wkHeads: zeroHeadWeights,
                    wvHeads: zeroHeadWeights,
                    bqHeads: zeroHeadBiases,
                    bkHeads: zeroHeadBiases,
                    bvHeads: zeroHeadBiases,
                    woHeads: zeroHeadWeights,
                    attentionOutputBias: [],
                    ffnW1: new float[2 * 2],
                    ffnB1: [],
                    ffnW2: new float[2 * 2],
                    ffnB2: [],
                    cache: cache,
                    layerIndex: 0,
                    position: 0,
                    output: new float[2]));
        }

        [Fact]
        public void Decode_InvalidArguments_Throw()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 1,
                maxSequenceLength: 4,
                headDimension: 2);

            cache.Advance();

            var block = new CachedTransformerBlock(
                dModel: 2,
                headCount: 1,
                dFF: 2,
                maxSequenceLength: 4);

            var zeroHeadWeights = CreateZeroHeadWeights();
            var zeroHeadBiases = CreateZeroHeadBiases();

            Assert.Throws<ArgumentException>(() =>
                block.DecodeWithoutLayerNormAffine(
                    input: new float[1],
                    wqHeads: zeroHeadWeights,
                    wkHeads: zeroHeadWeights,
                    wvHeads: zeroHeadWeights,
                    bqHeads: zeroHeadBiases,
                    bkHeads: zeroHeadBiases,
                    bvHeads: zeroHeadBiases,
                    woHeads: zeroHeadWeights,
                    attentionOutputBias: [],
                    ffnW1: new float[2 * 2],
                    ffnB1: [],
                    ffnW2: new float[2 * 2],
                    ffnB2: [],
                    cache: cache,
                    layerIndex: 0,
                    position: 0,
                    output: new float[2]));

            Assert.Throws<ArgumentException>(() =>
                block.DecodeWithoutLayerNormAffine(
                    input: new float[2],
                    wqHeads: zeroHeadWeights,
                    wkHeads: zeroHeadWeights,
                    wvHeads: zeroHeadWeights,
                    bqHeads: zeroHeadBiases,
                    bkHeads: zeroHeadBiases,
                    bvHeads: zeroHeadBiases,
                    woHeads: zeroHeadWeights,
                    attentionOutputBias: [],
                    ffnW1: new float[1],
                    ffnB1: [],
                    ffnW2: new float[2 * 2],
                    ffnB2: [],
                    cache: cache,
                    layerIndex: 0,
                    position: 0,
                    output: new float[2]));

            Assert.Throws<ArgumentException>(() =>
                block.DecodeWithoutLayerNormAffine(
                    input: new float[2],
                    wqHeads: zeroHeadWeights,
                    wkHeads: zeroHeadWeights,
                    wvHeads: zeroHeadWeights,
                    bqHeads: zeroHeadBiases,
                    bkHeads: zeroHeadBiases,
                    bvHeads: zeroHeadBiases,
                    woHeads: zeroHeadWeights,
                    attentionOutputBias: [],
                    ffnW1: new float[2 * 2],
                    ffnB1: [],
                    ffnW2: new float[1],
                    ffnB2: [],
                    cache: cache,
                    layerIndex: 0,
                    position: 0,
                    output: new float[2]));

            Assert.Throws<ArgumentException>(() =>
                block.DecodeWithoutLayerNormAffine(
                    input: new float[2],
                    wqHeads: zeroHeadWeights,
                    wkHeads: zeroHeadWeights,
                    wvHeads: zeroHeadWeights,
                    bqHeads: zeroHeadBiases,
                    bkHeads: zeroHeadBiases,
                    bvHeads: zeroHeadBiases,
                    woHeads: zeroHeadWeights,
                    attentionOutputBias: [],
                    ffnW1: new float[2 * 2],
                    ffnB1: [],
                    ffnW2: new float[2 * 2],
                    ffnB2: [],
                    cache: cache,
                    layerIndex: 0,
                    position: 0,
                    output: new float[1]));
        }

        private static float[][] CreateZeroHeadWeights()
        {
            return new[]
            {
                new float[2 * 2]
            };
        }

        private static float[][] CreateZeroHeadBiases()
        {
            return new[]
            {
                new float[2]
            };
        }

        private static float[] LayerNorm(
            ReadOnlySpan<float> input,
            float epsilon)
        {
            var mean = 0f;

            for (var i = 0; i < input.Length; i++)
            {
                mean += input[i];
            }

            mean /= input.Length;

            var variance = 0f;

            for (var i = 0; i < input.Length; i++)
            {
                var centered = input[i] - mean;
                variance += centered * centered;
            }

            variance /= input.Length;

            var invStd = 1f / MathF.Sqrt(variance + epsilon);
            var output = new float[input.Length];

            for (var i = 0; i < input.Length; i++)
            {
                output[i] = (input[i] - mean) * invStd;
            }

            return output;
        }

        private static void AssertClose(float expected, float actual)
        {
            Assert.True(
                MathF.Abs(expected - actual) <= 1e-5f,
                $"Expected {expected}, actual {actual}.");
        }
    }
}
