// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit;

namespace DevOnBike.Overfit.Tests.LanguageModels
{
    public class CachedGptStackTests
    {
        [Fact]
        public void Constructor_ExposesShape()
        {
            var stack = new CachedGptStack(
                layerCount: 2,
                dModel: 4,
                headCount: 2,
                dFF: 8,
                vocabSize: 16,
                maxSequenceLength: 32,
                layerNormEpsilon: 1e-5f,
                feedForwardActivation: FeedForwardActivation.ReLU);

            Assert.Equal(2, stack.LayerCount);
            Assert.Equal(4, stack.DModel);
            Assert.Equal(2, stack.HeadCount);
            Assert.Equal(2, stack.HeadDimension);
            Assert.Equal(8, stack.DFF);
            Assert.Equal(16, stack.VocabSize);
            Assert.Equal(32, stack.MaxSequenceLength);
            Assert.Equal(1e-5f, stack.LayerNormEpsilon);
            Assert.Equal(FeedForwardActivation.ReLU, stack.FeedForwardActivation);
        }

        [Fact]
        public void Constructor_InvalidArguments_Throw()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedGptStack(0, 2, 1, 2, 2, 2));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedGptStack(1, 0, 1, 2, 2, 2));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedGptStack(1, 2, 0, 2, 2, 2));

            Assert.Throws<ArgumentException>(() =>
                new CachedGptStack(1, 3, 2, 2, 2, 2));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedGptStack(1, 2, 1, 0, 2, 2));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedGptStack(1, 2, 1, 2, 0, 2));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedGptStack(1, 2, 1, 2, 2, 0));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedGptStack(1, 2, 1, 2, 2, 2, layerNormEpsilon: 0f));
        }

        [Fact]
        public void Decode_ZeroBlocksAndIdentityLmHead_ReturnsFinalLayerNormLogits()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 1,
                maxSequenceLength: 4,
                headDimension: 2);

            var stack = new CachedGptStack(
                layerCount: 1,
                dModel: 2,
                headCount: 1,
                dFF: 2,
                vocabSize: 2,
                maxSequenceLength: 4,
                feedForwardActivation: FeedForwardActivation.None);

            var zeroHeads = new[]
            {
                new[]
                {
                    new float[2 * 2]
                }
            };

            var attentionBiases = new[]
            {
                Array.Empty<float>()
            };

            var ffnW1 = new[]
            {
                new float[2 * 2]
            };

            var ffnB1 = new[]
            {
                Array.Empty<float>()
            };

            var ffnW2 = new[]
            {
                new float[2 * 2]
            };

            var ffnB2 = new[]
            {
                Array.Empty<float>()
            };

            var lmHeadIdentity = new float[]
            {
                1f, 0f,
                0f, 1f
            };

            var logits = new float[2];

            cache.Advance();

            stack.DecodeWithoutLayerNormAffine(
                inputHidden: new float[] { 1f, -1f },
                wqHeadsByLayer: zeroHeads,
                wkHeadsByLayer: zeroHeads,
                wvHeadsByLayer: zeroHeads,
                woHeadsByLayer: zeroHeads,
                attentionOutputBiases: attentionBiases,
                ffnW1ByLayer: ffnW1,
                ffnB1ByLayer: ffnB1,
                ffnW2ByLayer: ffnW2,
                ffnB2ByLayer: ffnB2,
                lmHeadWeights: lmHeadIdentity,
                lmHeadBias: ReadOnlySpan<float>.Empty,
                cache,
                position: 0,
                logits);

            var expected = LayerNorm(new float[] { 1f, -1f }, 1e-5f);

            AssertClose(expected[0], logits[0]);
            AssertClose(expected[1], logits[1]);
        }

        [Fact]
        public void Decode_MultipleZeroLayers_PreservesHiddenUntilFinalNorm()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 2,
                headCount: 1,
                maxSequenceLength: 4,
                headDimension: 2);

            var stack = new CachedGptStack(
                layerCount: 2,
                dModel: 2,
                headCount: 1,
                dFF: 2,
                vocabSize: 2,
                maxSequenceLength: 4,
                feedForwardActivation: FeedForwardActivation.None);

            var zeroHeads = new[]
            {
                new[]
                {
                    new float[2 * 2]
                },
                new[]
                {
                    new float[2 * 2]
                }
            };

            var attentionBiases = new[]
            {
                Array.Empty<float>(),
                Array.Empty<float>()
            };

            var ffnW1 = new[]
            {
                new float[2 * 2],
                new float[2 * 2]
            };

            var ffnB1 = new[]
            {
                Array.Empty<float>(),
                Array.Empty<float>()
            };

            var ffnW2 = new[]
            {
                new float[2 * 2],
                new float[2 * 2]
            };

            var ffnB2 = new[]
            {
                Array.Empty<float>(),
                Array.Empty<float>()
            };

            var lmHeadIdentity = new float[]
            {
                1f, 0f,
                0f, 1f
            };

            var logits = new float[2];

            cache.Advance();

            stack.DecodeWithoutLayerNormAffine(
                inputHidden: new float[] { 2f, -2f },
                wqHeadsByLayer: zeroHeads,
                wkHeadsByLayer: zeroHeads,
                wvHeadsByLayer: zeroHeads,
                woHeadsByLayer: zeroHeads,
                attentionOutputBiases: attentionBiases,
                ffnW1ByLayer: ffnW1,
                ffnB1ByLayer: ffnB1,
                ffnW2ByLayer: ffnW2,
                ffnB2ByLayer: ffnB2,
                lmHeadWeights: lmHeadIdentity,
                lmHeadBias: ReadOnlySpan<float>.Empty,
                cache,
                position: 0,
                logits);

            var expected = LayerNorm(new float[] { 2f, -2f }, 1e-5f);

            AssertClose(expected[0], logits[0]);
            AssertClose(expected[1], logits[1]);
        }

        [Fact]
        public void Decode_AppliesLmHeadBias()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 1,
                maxSequenceLength: 4,
                headDimension: 2);

            var stack = new CachedGptStack(
                layerCount: 1,
                dModel: 2,
                headCount: 1,
                dFF: 2,
                vocabSize: 2,
                maxSequenceLength: 4,
                feedForwardActivation: FeedForwardActivation.None);

            var zeroHeads = new[]
            {
                new[]
                {
                    new float[2 * 2]
                }
            };

            var attentionBiases = new[]
            {
                Array.Empty<float>()
            };

            var ffnW1 = new[]
            {
                new float[2 * 2]
            };

            var ffnB1 = new[]
            {
                Array.Empty<float>()
            };

            var ffnW2 = new[]
            {
                new float[2 * 2]
            };

            var ffnB2 = new[]
            {
                Array.Empty<float>()
            };

            var lmHeadIdentity = new float[]
            {
                1f, 0f,
                0f, 1f
            };

            var lmHeadBias = new float[] { 10f, 20f };
            var logits = new float[2];

            cache.Advance();

            stack.DecodeWithoutLayerNormAffine(
                inputHidden: new float[] { 1f, -1f },
                wqHeadsByLayer: zeroHeads,
                wkHeadsByLayer: zeroHeads,
                wvHeadsByLayer: zeroHeads,
                woHeadsByLayer: zeroHeads,
                attentionOutputBiases: attentionBiases,
                ffnW1ByLayer: ffnW1,
                ffnB1ByLayer: ffnB1,
                ffnW2ByLayer: ffnW2,
                ffnB2ByLayer: ffnB2,
                lmHeadWeights: lmHeadIdentity,
                lmHeadBias,
                cache,
                position: 0,
                logits);

            var expected = LayerNorm(new float[] { 1f, -1f }, 1e-5f);

            AssertClose(expected[0] + 10f, logits[0]);
            AssertClose(expected[1] + 20f, logits[1]);
        }

        [Fact]
        public void Decode_StoresLastFinalHiddenAndLogits()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 1,
                maxSequenceLength: 4,
                headDimension: 2);

            var stack = new CachedGptStack(
                layerCount: 1,
                dModel: 2,
                headCount: 1,
                dFF: 2,
                vocabSize: 2,
                maxSequenceLength: 4,
                feedForwardActivation: FeedForwardActivation.None);

            var zeroHeads = new[]
            {
                new[]
                {
                    new float[2 * 2]
                }
            };

            var attentionBiases = new[]
            {
                Array.Empty<float>()
            };

            var ffnW1 = new[]
            {
                new float[2 * 2]
            };

            var ffnB1 = new[]
            {
                Array.Empty<float>()
            };

            var ffnW2 = new[]
            {
                new float[2 * 2]
            };

            var ffnB2 = new[]
            {
                Array.Empty<float>()
            };

            var lmHeadIdentity = new float[]
            {
                1f, 0f,
                0f, 1f
            };

            var logits = new float[2];

            cache.Advance();

            stack.DecodeWithoutLayerNormAffine(
                inputHidden: new float[] { 1f, -1f },
                wqHeadsByLayer: zeroHeads,
                wkHeadsByLayer: zeroHeads,
                wvHeadsByLayer: zeroHeads,
                woHeadsByLayer: zeroHeads,
                attentionOutputBiases: attentionBiases,
                ffnW1ByLayer: ffnW1,
                ffnB1ByLayer: ffnB1,
                ffnW2ByLayer: ffnW2,
                ffnB2ByLayer: ffnB2,
                lmHeadWeights: lmHeadIdentity,
                lmHeadBias: ReadOnlySpan<float>.Empty,
                cache,
                position: 0,
                logits);

            var finalHidden = new float[2];
            var lastLogits = new float[2];

            stack.GetLastFinalHidden(finalHidden);
            stack.GetLastLogits(lastLogits);

            Assert.Equal(logits, finalHidden);
            Assert.Equal(logits, lastLogits);
        }

        [Fact]
        public void Decode_PositionNotVisible_Throws()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 1,
                maxSequenceLength: 4,
                headDimension: 2);

            var stack = new CachedGptStack(
                layerCount: 1,
                dModel: 2,
                headCount: 1,
                dFF: 2,
                vocabSize: 2,
                maxSequenceLength: 4);

            var zeroHeads = new[]
            {
                new[]
                {
                    new float[2 * 2]
                }
            };

            Assert.Throws<InvalidOperationException>(() =>
                stack.DecodeWithoutLayerNormAffine(
                    inputHidden: new float[] { 1f, -1f },
                    wqHeadsByLayer: zeroHeads,
                    wkHeadsByLayer: zeroHeads,
                    wvHeadsByLayer: zeroHeads,
                    woHeadsByLayer: zeroHeads,
                    attentionOutputBiases: new[] { Array.Empty<float>() },
                    ffnW1ByLayer: new[] { new float[2 * 2] },
                    ffnB1ByLayer: new[] { Array.Empty<float>() },
                    ffnW2ByLayer: new[] { new float[2 * 2] },
                    ffnB2ByLayer: new[] { Array.Empty<float>() },
                    lmHeadWeights: new float[2 * 2],
                    lmHeadBias: ReadOnlySpan<float>.Empty,
                    cache,
                    position: 0,
                    logits: new float[2]));
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

            var stack = new CachedGptStack(
                layerCount: 1,
                dModel: 2,
                headCount: 1,
                dFF: 2,
                vocabSize: 2,
                maxSequenceLength: 4);

            var zeroHeads = new[]
            {
                new[]
                {
                    new float[2 * 2]
                }
            };

            Assert.Throws<ArgumentException>(() =>
                stack.DecodeWithoutLayerNormAffine(
                    inputHidden: new float[1],
                    wqHeadsByLayer: zeroHeads,
                    wkHeadsByLayer: zeroHeads,
                    wvHeadsByLayer: zeroHeads,
                    woHeadsByLayer: zeroHeads,
                    attentionOutputBiases: new[] { Array.Empty<float>() },
                    ffnW1ByLayer: new[] { new float[2 * 2] },
                    ffnB1ByLayer: new[] { Array.Empty<float>() },
                    ffnW2ByLayer: new[] { new float[2 * 2] },
                    ffnB2ByLayer: new[] { Array.Empty<float>() },
                    lmHeadWeights: new float[2 * 2],
                    lmHeadBias: ReadOnlySpan<float>.Empty,
                    cache,
                    position: 0,
                    logits: new float[2]));

            Assert.Throws<ArgumentException>(() =>
                stack.DecodeWithoutLayerNormAffine(
                    inputHidden: new float[2],
                    wqHeadsByLayer: zeroHeads,
                    wkHeadsByLayer: zeroHeads,
                    wvHeadsByLayer: zeroHeads,
                    woHeadsByLayer: zeroHeads,
                    attentionOutputBiases: new[] { Array.Empty<float>() },
                    ffnW1ByLayer: new[] { new float[1] },
                    ffnB1ByLayer: new[] { Array.Empty<float>() },
                    ffnW2ByLayer: new[] { new float[2 * 2] },
                    ffnB2ByLayer: new[] { Array.Empty<float>() },
                    lmHeadWeights: new float[2 * 2],
                    lmHeadBias: ReadOnlySpan<float>.Empty,
                    cache,
                    position: 0,
                    logits: new float[2]));

            Assert.Throws<ArgumentException>(() =>
                stack.DecodeWithoutLayerNormAffine(
                    inputHidden: new float[2],
                    wqHeadsByLayer: zeroHeads,
                    wkHeadsByLayer: zeroHeads,
                    wvHeadsByLayer: zeroHeads,
                    woHeadsByLayer: zeroHeads,
                    attentionOutputBiases: new[] { Array.Empty<float>() },
                    ffnW1ByLayer: new[] { new float[2 * 2] },
                    ffnB1ByLayer: new[] { Array.Empty<float>() },
                    ffnW2ByLayer: new[] { new float[2 * 2] },
                    ffnB2ByLayer: new[] { Array.Empty<float>() },
                    lmHeadWeights: new float[1],
                    lmHeadBias: ReadOnlySpan<float>.Empty,
                    cache,
                    position: 0,
                    logits: new float[2]));

            Assert.Throws<ArgumentException>(() =>
                stack.DecodeWithoutLayerNormAffine(
                    inputHidden: new float[2],
                    wqHeadsByLayer: zeroHeads,
                    wkHeadsByLayer: zeroHeads,
                    wvHeadsByLayer: zeroHeads,
                    woHeadsByLayer: zeroHeads,
                    attentionOutputBiases: new[] { Array.Empty<float>() },
                    ffnW1ByLayer: new[] { new float[2 * 2] },
                    ffnB1ByLayer: new[] { Array.Empty<float>() },
                    ffnW2ByLayer: new[] { new float[2 * 2] },
                    ffnB2ByLayer: new[] { Array.Empty<float>() },
                    lmHeadWeights: new float[2 * 2],
                    lmHeadBias: ReadOnlySpan<float>.Empty,
                    cache,
                    position: 0,
                    logits: new float[1]));
        }

        [Fact]
        public void GetBlock_ReturnsRequestedBlock()
        {
            var stack = new CachedGptStack(
                layerCount: 2,
                dModel: 2,
                headCount: 1,
                dFF: 2,
                vocabSize: 2,
                maxSequenceLength: 4);

            Assert.NotNull(stack.GetBlock(0));
            Assert.NotNull(stack.GetBlock(1));
            Assert.Throws<ArgumentOutOfRangeException>(() => stack.GetBlock(2));
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
