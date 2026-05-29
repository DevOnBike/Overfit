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
    /// Synthetic test for the Qwen2-MoE FFN composition: output must equal the independent reference
    /// <c>σ(w·x)·shared(x) + routed(x)</c>, where routed(x) is the (already-tested)
    /// <see cref="MoeFeedForwardBlock"/> and shared(x) is a SwiGLU FFN via the tested
    /// <see cref="CachedFeedForwardBlock.DecodeSwiGlu"/>. Validates the shared-expert sigmoid gate
    /// and the combine — the Qwen-specific part on top of the routed core.
    /// </summary>
    public sealed class Qwen2MoeFeedForwardBlockTests
    {
        private const int DModel = 4;
        private const int ExpertDff = 3;
        private const int SharedDff = 5;
        private const int ExpertCount = 3;
        private const int TopK = 2;

        [Fact]
        public void Output_EqualsGatedShared_Plus_RoutedReference()
        {
            var (gate, up, down) = BuildExperts();
            var sGate = Weight(DModel * SharedDff, 500);
            var sUp = Weight(DModel * SharedDff, 600);
            var sDown = Weight(SharedDff * DModel, 700);
            var sharedGateInp = Vector(DModel, 9);
            var router = RouterColumns(1.0f, 2.0f, 0.5f);   // experts 1,0 selected
            var hidden = Vector(DModel, 3);

            var block = new Qwen2MoeFeedForwardBlock(DModel, ExpertDff, SharedDff, ExpertCount, TopK);
            var actual = new float[DModel];
            block.Decode(hidden, router, gate, up, down, sharedGateInp, sGate, sUp, sDown, actual);

            // ── Reference ──────────────────────────────────────────────────
            // routed(x): reuse the routed block directly (it has its own test).
            var routed = new MoeFeedForwardBlock(DModel, ExpertDff, ExpertCount, TopK);
            var routedOut = new float[DModel];
            routed.Decode(hidden, router, gate, up, down, routedOut);

            // shared(x): SwiGLU FFN, scaled by sigmoid(sharedGateInp · hidden).
            var logit = 0f;
            for (var d = 0; d < DModel; d++) { logit += hidden[d] * sharedGateInp[d]; }
            var g = 1f / (1f + MathF.Exp(-logit));

            var sharedFfn = new CachedFeedForwardBlock(DModel, SharedDff, FeedForwardActivation.SwiGLU);
            var sharedOut = new float[DModel];
            sharedFfn.DecodeSwiGlu(hidden, sGate.F32, sUp.F32, sDown.F32, sharedOut);

            for (var d = 0; d < DModel; d++)
            {
                var expected = (g * sharedOut[d]) + routedOut[d];
                Assert.Equal(expected, actual[d], 4);
            }
        }

        [Fact]
        public void NoSharedExpert_Mixtral_OutputEqualsRoutedSumAlone()
        {
            // sharedFeedForwardLength = 0 ⇒ Mixtral shape (routed-only). The shared-expert weights /
            // gate are absent (default / empty) and must be ignored: output == routed reference, and
            // top-k renormalisation is on (Mixtral norm_topk_prob=true).
            var (gate, up, down) = BuildExperts();
            var router = RouterColumns(1.0f, 2.0f, 0.5f);
            var hidden = Vector(DModel, 3);

            var block = new Qwen2MoeFeedForwardBlock(
                DModel, ExpertDff, sharedFeedForwardLength: 0, ExpertCount, TopK, normalizeExpertWeights: true);
            Assert.False(block.HasSharedExpert);

            var actual = new float[DModel];
            // Shared weights are default/empty — the routed-only path must not touch them.
            block.Decode(hidden, router, gate, up, down, [], default, default, default, actual);

            var routed = new MoeFeedForwardBlock(DModel, ExpertDff, ExpertCount, TopK, normalizeWeights: true);
            var routedOut = new float[DModel];
            routed.Decode(hidden, router, gate, up, down, routedOut);

            for (var d = 0; d < DModel; d++)
            {
                Assert.Equal(routedOut[d], actual[d], 5);
            }
        }

        [Theory]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(5)]
        public void DecodeBatched_MatchesPerRowDecode(int rows)
        {
            var (gate, up, down) = BuildExperts();
            var sGate = Weight(DModel * SharedDff, 500);
            var sUp = Weight(DModel * SharedDff, 600);
            var sDown = Weight(SharedDff * DModel, 700);
            var sharedGateInp = Vector(DModel, 9);
            var router = RandomRouter(13);

            var block = new Qwen2MoeFeedForwardBlock(DModel, ExpertDff, SharedDff, ExpertCount, TopK);

            // Build N distinct hidden rows so different rows route to different experts.
            var hidden = new float[rows * DModel];
            for (var n = 0; n < rows; n++)
            {
                for (var d = 0; d < DModel; d++) { hidden[n * DModel + d] = MathF.Sin((n + 1) * 0.7f + d * 0.9f) * 0.6f; }
            }

            // Reference: per-row single-token Decode.
            var expected = new float[rows * DModel];
            for (var n = 0; n < rows; n++)
            {
                block.Decode(
                    hidden.AsSpan(n * DModel, DModel), router, gate, up, down,
                    sharedGateInp, sGate, sUp, sDown, expected.AsSpan(n * DModel, DModel));
            }

            // Batched.
            var batched = new float[rows * DModel];
            block.DecodeBatched(
                hidden, rows, router, gate, up, down,
                sharedGateInp, sGate, sUp, sDown, batched);

            // Per-row weighted sum runs in top-k slot order = single-token order, and the expert SwiGLU
            // is bit-identical batched-vs-single — so the block is bit-identical, not merely FP-close.
            for (var i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], batched[i]);
            }
        }

        private static float[] RandomRouter(int seed)
        {
            var rng = new Random(seed);
            var w = new float[DModel * ExpertCount];
            for (var i = 0; i < w.Length; i++) { w[i] = (float)(rng.NextDouble() * 2.0 - 1.0); }
            return w;
        }

        private static (DecodeWeight[] gate, DecodeWeight[] up, DecodeWeight[] down) BuildExperts()
        {
            var gate = new DecodeWeight[ExpertCount];
            var up = new DecodeWeight[ExpertCount];
            var down = new DecodeWeight[ExpertCount];
            for (var e = 0; e < ExpertCount; e++)
            {
                gate[e] = Weight(DModel * ExpertDff, 100 + e);
                up[e] = Weight(DModel * ExpertDff, 200 + e);
                down[e] = Weight(ExpertDff * DModel, 300 + e);
            }
            return (gate, up, down);
        }

        private static DecodeWeight Weight(int n, int seed)
        {
            var s = TensorStorage<float>.Unpooled(n);
            var span = s.AsSpan();
            for (var i = 0; i < n; i++) { span[i] = MathF.Sin(seed * 0.1f + i * 0.37f) * 0.2f; }
            return s;
        }

        private static float[] Vector(int n, int seed)
        {
            var v = new float[n];
            for (var i = 0; i < n; i++) { v[i] = MathF.Cos(seed * 0.2f + i * 0.5f) * 0.5f; }
            return v;
        }

        private static float[] RouterColumns(float c0, float c1, float c2)
        {
            var cols = new[] { c0, c1, c2 };
            var w = new float[DModel * ExpertCount];
            for (var d = 0; d < DModel; d++)
            {
                for (var e = 0; e < ExpertCount; e++) { w[d * ExpertCount + e] = cols[e]; }
            }
            return w;
        }
    }
}
