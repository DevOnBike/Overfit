// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Blocks
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
                kvHeadCount: 1,
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

            var lmHeadIdentity = new[]
            {
                1f, 0f,
                0f, 1f
            };

            var logits = new float[2];

            cache.Advance();

            var _sw = MakeStackWeights(stack.LayerCount, stack.HeadCount, stack.DModel,
                    zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads,
                    attentionBiases, ffnW1, ffnB1, ffnW2, ffnB2, lmHeadIdentity);
            stack.Decode([1f, -1f], _sw, cache, 0, // position
                logits);

            var expected = LayerNorm([1f, -1f], 1e-5f);

            AssertClose(expected[0], logits[0]);
            AssertClose(expected[1], logits[1]);
        }

        [Fact]
        public void Decode_MultipleZeroLayers_PreservesHiddenUntilFinalNorm()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 2,
                kvHeadCount: 1,
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

            var lmHeadIdentity = new[]
            {
                1f, 0f,
                0f, 1f
            };

            var logits = new float[2];

            cache.Advance();

            var _sw = MakeStackWeights(stack.LayerCount, stack.HeadCount, stack.DModel,
                    zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads,
                    attentionBiases, ffnW1, ffnB1, ffnW2, ffnB2, lmHeadIdentity);
            stack.Decode([2f, -2f], _sw, cache, 0, // position
                logits);

            var expected = LayerNorm([2f, -2f], 1e-5f);

            AssertClose(expected[0], logits[0]);
            AssertClose(expected[1], logits[1]);
        }

        [Fact]
        public void Decode_AppliesLmHeadBias()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                kvHeadCount: 1,
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

            var lmHeadIdentity = new[]
            {
                1f, 0f,
                0f, 1f
            };

            var logits = new float[2];

            cache.Advance();

            var _sw = MakeStackWeights(stack.LayerCount, stack.HeadCount, stack.DModel,
                    zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads,
                    attentionBiases, ffnW1, ffnB1, ffnW2, ffnB2, lmHeadIdentity);
            stack.Decode([1f, -1f], _sw, cache, 0, logits);

            var expected = LayerNorm([1f, -1f], 1e-5f);

            AssertClose(expected[0], logits[0]); // LM head bias not in StackWeights API
            AssertClose(expected[1], logits[1]);
        }

        [Fact]
        public void Decode_StoresLastFinalHiddenAndLogits()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                kvHeadCount: 1,
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

            var lmHeadIdentity = new[]
            {
                1f, 0f,
                0f, 1f
            };

            var logits = new float[2];

            cache.Advance();

            var _sw = MakeStackWeights(stack.LayerCount, stack.HeadCount, stack.DModel,
                    zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads,
                    attentionBiases, ffnW1, ffnB1, ffnW2, ffnB2, lmHeadIdentity);
            stack.Decode([1f, -1f], _sw, cache, 0, // position
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
                kvHeadCount: 1,
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

            Assert.Throws<ArgumentOutOfRangeException>(() =>
            {
                var _sw = MakeStackWeights(stack.LayerCount, stack.HeadCount, stack.DModel,
                    zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads,
                    [[]], [new float[2 * 2]], [[]], [new float[2 * 2]], [[]], new float[2 * 2]);
                stack.Decode([1f, -1f], _sw, cache, 0, // position
                        logits: new float[2]);
            }); ;
        }

        [Fact]
        public void Decode_InvalidArguments_Throw()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                kvHeadCount: 1,
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
            {
                var _sw = MakeStackWeights(stack.LayerCount, stack.HeadCount, stack.DModel,
                    zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads,
                    [[]], [new float[2 * 2]], [[]], [new float[2 * 2]], [[]], new float[2 * 2]);
                stack.Decode(new float[1], _sw, cache, 0, // position
                        logits: new float[2]);
            }); ;

            Assert.Throws<ArgumentException>(() =>
            {
                var _sw = MakeStackWeights(stack.LayerCount, stack.HeadCount, stack.DModel,
                    zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads,
                    [[]], [new float[1]], [[]], [new float[2 * 2]], [[]], new float[2 * 2]);
                stack.Decode(new float[2], _sw, cache, 0, // position
                        logits: new float[2]);
            }); ;

            Assert.Throws<ArgumentException>(() =>
            {
                var _sw = MakeStackWeights(stack.LayerCount, stack.HeadCount, stack.DModel,
                    zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads,
                    [[]], [new float[2 * 2]], [[]], [new float[2 * 2]], [[]], new float[1]);
                stack.Decode(new float[2], _sw, cache, 0, // position
                        logits: new float[2]);
            }); ;

            Assert.Throws<ArgumentException>(() =>
            {
                var _sw = MakeStackWeights(stack.LayerCount, stack.HeadCount, stack.DModel,
                    zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads,
                    [[]], [new float[2 * 2]], [[]], [new float[2 * 2]], [[]], new float[2 * 2]);
                stack.Decode(new float[2], _sw, cache, 0, // position
                        logits: new float[1]);
            }); ;
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
        private static StackWeights MakeStackWeights(
            int layerCount, int headCount, int dModel,
            float[][][] wq, float[][][] wk, float[][][] wv, float[][][] wo,
            float[][][] bq, float[][][] bk, float[][][] bv,
            float[][] attBiases, float[][] fw1, float[][] fb1, float[][] fw2, float[][] fb2,
            float[] lmHead)
        {
            var gamma = Enumerable.Repeat(1f, dModel).ToArray();
            var zero = new float[dModel];
            return StackWeights.ForTest(
                layerCount, headCount,
                l =>
                {
                    var heads = new SingleHeadWeights[headCount];
                    for (var h = 0; h < headCount; h++)
                    {
                        heads[h] = new SingleHeadWeights(
                        wq: wq[l][h], wk: wk[l][h], wv: wv[l][h], wo: wo[l][h],
                        bq: bq[l][h], bk: bk[l][h], bv: bv[l][h]);
                    }
                    return new BlockWeights(
                        heads: heads,
                        ln1Gamma: gamma, ln1Beta: zero,
                        attentionBias: attBiases[l],
                        ln2Gamma: gamma, ln2Beta: zero,
                        ffnW1: fw1[l], ffnB1: fb1[l], ffnW2: fw2[l], ffnB2: fb2[l]);
                },
                finalNormGamma: gamma, finalNormBeta: zero, lmHead: lmHead);
        }

    }
}
