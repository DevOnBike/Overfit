// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// Unit tests for the MoE gating core (<see cref="MoeRouter.SelectTopK"/>): correct top-k
    /// selection, renormalised weights summing to 1, the Mixtral identity (top-k softmax) and edge
    /// cases. This is the only novel MoE math; the per-expert FFN reuses existing kernels.
    /// </summary>
    public sealed class MoeRouterTests
    {
        [Fact]
        public void TopK1_PicksArgmax_WithWeightOne()
        {
            float[] logits = [0.1f, 2.0f, -1.0f, 0.5f];
            Span<int> idx = stackalloc int[1];
            Span<float> w = stackalloc float[1];

            var k = MoeRouter.SelectTopK(logits, topK: 1, idx, w);

            Assert.Equal(1, k);
            Assert.Equal(1, idx[0]);          // expert 1 has the highest logit
            Assert.Equal(1f, w[0], 5);        // single expert ⇒ weight 1
        }

        [Fact]
        public void TopK2_SelectsTwoHighest_DescendingByLogit()
        {
            float[] logits = [0.1f, 2.0f, -1.0f, 1.5f];
            Span<int> idx = stackalloc int[2];
            Span<float> w = stackalloc float[2];

            var k = MoeRouter.SelectTopK(logits, topK: 2, idx, w);

            Assert.Equal(2, k);
            Assert.Equal(1, idx[0]);          // logit 2.0
            Assert.Equal(3, idx[1]);          // logit 1.5
            Assert.True(w[0] > w[1]);         // higher logit ⇒ higher weight
        }

        [Fact]
        public void Weights_SumToOne()
        {
            float[] logits = [0.3f, 1.1f, 0.7f, -0.4f, 2.2f];
            Span<int> idx = stackalloc int[3];
            Span<float> w = stackalloc float[3];

            var k = MoeRouter.SelectTopK(logits, topK: 3, idx, w);

            var sum = 0f;
            for (var i = 0; i < k; i++) { sum += w[i]; }
            Assert.Equal(1f, sum, 5);
        }

        [Fact]
        public void MatchesRenormalisedFullSoftmax_OverTopK()
        {
            // Mixtral semantics: softmax over ALL, take top-k, renormalise. Must equal SelectTopK.
            float[] logits = [0.5f, 2.0f, 1.0f, 1.8f];
            const int topK = 2;

            // Reference: full softmax → top-k → renormalise.
            var max = float.NegativeInfinity;
            foreach (var l in logits) { max = MathF.Max(max, l); }
            var full = new float[logits.Length];
            var z = 0f;
            for (var i = 0; i < logits.Length; i++) { full[i] = MathF.Exp(logits[i] - max); z += full[i]; }
            for (var i = 0; i < logits.Length; i++) { full[i] /= z; }
            // top-2 = experts 1 (2.0) and 3 (1.8); renormalise their probs
            var pair = full[1] + full[3];
            var refW1 = full[1] / pair;
            var refW3 = full[3] / pair;

            Span<int> idx = stackalloc int[topK];
            Span<float> w = stackalloc float[topK];
            MoeRouter.SelectTopK(logits, topK, idx, w);

            Assert.Equal(1, idx[0]);
            Assert.Equal(3, idx[1]);
            Assert.Equal(refW1, w[0], 5);
            Assert.Equal(refW3, w[1], 5);
        }

        [Fact]
        public void TopKExceedingExpertCount_SelectsAll()
        {
            float[] logits = [1.0f, 0.0f];
            Span<int> idx = stackalloc int[5];
            Span<float> w = stackalloc float[5];

            var k = MoeRouter.SelectTopK(logits, topK: 5, idx, w);

            Assert.Equal(2, k);   // clamped to expert count
            var sum = w[0] + w[1];
            Assert.Equal(1f, sum, 5);
            Assert.Equal(0, idx[0]);   // higher logit first
        }

        [Fact]
        public void Empty_OrNonPositiveK_Throws()
        {
            Assert.Throws<ArgumentException>(() =>
            {
                Span<int> idx = stackalloc int[1];
                Span<float> w = stackalloc float[1];
                MoeRouter.SelectTopK([], topK: 1, idx, w);
            });
            Assert.Throws<ArgumentOutOfRangeException>(() =>
            {
                Span<int> idx = stackalloc int[1];
                Span<float> w = stackalloc float[1];
                MoeRouter.SelectTopK([1f, 2f], topK: 0, idx, w);
            });
        }
    }
}
