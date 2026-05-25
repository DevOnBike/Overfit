// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// Synthetic tests for <see cref="MoeFeedForwardBlock"/> (F32 experts): the routed/combined
    /// output must equal the independent reference — the selected experts' SwiGLU FFNs (via the
    /// well-tested <see cref="CachedFeedForwardBlock.DecodeSwiGlu"/>) combined by the router weights.
    /// Validates the new MoE plumbing (top-k selection, per-expert dispatch, weighted accumulate)
    /// without a real MoE GGUF.
    /// </summary>
    public sealed class MoeFeedForwardBlockTests
    {
        private const int DModel = 4;
        private const int DFF = 6;
        private const int ExpertCount = 3;

        [Fact]
        public void TopK1_OutputEqualsSelectedExpertFfn()
        {
            var (gate, up, down) = BuildExperts();
            var hidden = Vector(DModel, seed: 7);
            // Router logits forced so expert 1 dominates (≈ weight 1).
            var router = RouterFavouring(expert: 1);

            var moe = new MoeFeedForwardBlock(DModel, DFF, ExpertCount, expertUsedCount: 1);
            var actual = new float[DModel];
            moe.Decode(hidden, router, gate, up, down, actual);

            // Reference: just expert 1's SwiGLU FFN.
            var reference = new CachedFeedForwardBlock(DModel, DFF, FeedForwardActivation.SwiGLU);
            var expected = new float[DModel];
            reference.DecodeSwiGlu(hidden, gate[1].F32, up[1].F32, down[1].F32, expected);

            for (var d = 0; d < DModel; d++) { Assert.Equal(expected[d], actual[d], 4); }
        }

        [Fact]
        public void TopK2_OutputEqualsWeightedSumOfSelectedExperts()
        {
            var (gate, up, down) = BuildExperts();
            var hidden = Vector(DModel, seed: 3);
            var router = RouterColumns(col0: 1.0f, col1: 2.0f, col2: 0.5f);   // experts 1,0 selected

            var moe = new MoeFeedForwardBlock(DModel, DFF, ExpertCount, expertUsedCount: 2);
            var actual = new float[DModel];
            moe.Decode(hidden, router, gate, up, down, actual);

            // Independent reference: recompute logits → top-2 → per-expert FFN → weighted sum.
            var logits = new float[ExpertCount];
            SingleTokenProjectionKernel.ProjectParallel(hidden, router, [], logits, DModel, ExpertCount);
            var idx = new int[2];
            var w = new float[2];
            MoeRouter.SelectTopK(logits, 2, idx, w);

            var reference = new CachedFeedForwardBlock(DModel, DFF, FeedForwardActivation.SwiGLU);
            var expected = new float[DModel];
            var scratch = new float[DModel];
            for (var i = 0; i < 2; i++)
            {
                reference.DecodeSwiGlu(hidden, gate[idx[i]].F32, up[idx[i]].F32, down[idx[i]].F32, scratch);
                for (var d = 0; d < DModel; d++) { expected[d] += w[i] * scratch[d]; }
            }

            // The two experts must actually differ, or the test proves nothing about selection.
            Assert.NotEqual(idx[0], idx[1]);
            for (var d = 0; d < DModel; d++) { Assert.Equal(expected[d], actual[d], 4); }
        }

        // ── helpers ───────────────────────────────────────────────────────────

        private static (DecodeWeight[] gate, DecodeWeight[] up, DecodeWeight[] down) BuildExperts()
        {
            var gate = new DecodeWeight[ExpertCount];
            var up = new DecodeWeight[ExpertCount];
            var down = new DecodeWeight[ExpertCount];
            for (var e = 0; e < ExpertCount; e++)
            {
                gate[e] = Weight(DModel * DFF, seed: 100 + e);
                up[e] = Weight(DModel * DFF, seed: 200 + e);
                down[e] = Weight(DFF * DModel, seed: 300 + e);
            }
            return (gate, up, down);
        }

        private static DecodeWeight Weight(int n, int seed)
        {
            var storage = TensorStorage<float>.Unpooled(n);
            var span = storage.AsSpan();
            for (var i = 0; i < n; i++) { span[i] = MathF.Sin(seed * 0.1f + i * 0.37f) * 0.2f; }
            return storage;
        }

        private static float[] Vector(int n, int seed)
        {
            var v = new float[n];
            for (var i = 0; i < n; i++) { v[i] = MathF.Cos(seed * 0.2f + i * 0.5f) * 0.5f; }
            return v;
        }

        // Router [dModel × ExpertCount] (input-major) — one column big so that expert wins.
        private static float[] RouterFavouring(int expert)
        {
            var w = new float[DModel * ExpertCount];
            for (var d = 0; d < DModel; d++) { w[d * ExpertCount + expert] = 10f; }
            return w;
        }

        private static float[] RouterColumns(float col0, float col1, float col2)
        {
            var cols = new[] { col0, col1, col2 };
            var w = new float[DModel * ExpertCount];
            for (var d = 0; d < DModel; d++)
            {
                for (var e = 0; e < ExpertCount; e++) { w[d * ExpertCount + e] = cols[e]; }
            }
            return w;
        }
    }
}
