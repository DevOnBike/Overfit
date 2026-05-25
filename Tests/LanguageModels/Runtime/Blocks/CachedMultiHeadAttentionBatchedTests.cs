// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Blocks
{
    /// <summary>
    /// Prefill Phase 3 (attention slice): <see cref="CachedMultiHeadAttention.DecodeBatched"/>
    /// must be bit-identical to running <see cref="CachedMultiHeadAttention.Decode"/> on
    /// each prompt token in sequence (advancing the cache per token). Covers the
    /// F32 / GPT-2 path: standard MHA, no RoPE, with an attention output bias.
    /// </summary>
    public sealed class CachedMultiHeadAttentionBatchedTests
    {
        [Theory]
        [InlineData(1, 4, 2)]
        [InlineData(5, 8, 2)]
        [InlineData(7, 32, 4)]
        [InlineData(12, 128, 8)]
        public void DecodeBatched_IsBitIdentical_To_PerTokenDecode(int rows, int dModel, int headCount)
        {
            var rng = new Random(41 + rows * 9 + dModel + headCount);
            var dHead = dModel / headCount;

            var hidden = Random(rng, rows * dModel, 1f);
            var wq = PerHead(rng, headCount, dModel * dHead, 0.1f);
            var wk = PerHead(rng, headCount, dModel * dHead, 0.1f);
            var wv = PerHead(rng, headCount, dModel * dHead, 0.1f);
            var wo = PerHead(rng, headCount, dHead * dModel, 0.1f);
            var attentionBias = Random(rng, dModel, 0.2f);

            var weights = MakeBlockWeights(wq, wk, wv, wo, attentionBias);
            var mha = new CachedMultiHeadAttention(dModel, headCount, maxSequenceLength: rows);

            // Reference: single-token decode, advancing the cache per token.
            var expected = new float[rows * dModel];
            using (var cacheRef = KeyValueCache.Create(
                layerCount: 1, kvHeadCount: headCount, maxSequenceLength: rows, headDimension: dHead))
            {
                for (var i = 0; i < rows; i++)
                {
                    cacheRef.Advance();
                    mha.Decode(
                        hidden.AsSpan(i * dModel, dModel),
                        in weights,
                        cacheRef,
                        layerIndex: 0,
                        position: i,
                        expected.AsSpan(i * dModel, dModel));
                }
            }

            // Batched: cache advanced to N up front, all N processed in one call.
            var batched = new float[rows * dModel];
            using (var cacheBat = KeyValueCache.Create(
                layerCount: 1, kvHeadCount: headCount, maxSequenceLength: rows, headDimension: dHead))
            {
                for (var i = 0; i < rows; i++)
                {
                    cacheBat.Advance();
                }

                mha.DecodeBatched(hidden, rows, in weights, cacheBat, layerIndex: 0, basePosition: 0, batched);
            }

            for (var i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], batched[i]);
            }
        }

        private static float[][] PerHead(Random rng, int heads, int len, float scale)
        {
            var a = new float[heads][];
            for (var h = 0; h < heads; h++)
            {
                a[h] = Random(rng, len, scale);
            }
            return a;
        }

        private static float[] Random(Random rng, int n, float scale)
        {
            var a = new float[n];
            for (var i = 0; i < n; i++)
            {
                a[i] = (rng.NextSingle() * 2f - 1f) * scale;
            }
            return a;
        }

        private static BlockWeights MakeBlockWeights(
            float[][] wq, float[][] wk, float[][] wv, float[][] wo, float[]? attentionBias)
        {
            var heads = new SingleHeadWeights[wq.Length];
            for (var h = 0; h < wq.Length; h++)
            {
                heads[h] = new SingleHeadWeights(wq: wq[h], wk: wk[h], wv: wv[h], wo: wo[h]);
            }
            return new BlockWeights(heads: heads, attentionBias: attentionBias);
        }
    }
}
