// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Sampling
{
    [Trait("Category", "Sampling")]
    public sealed class RepetitionPenaltyTests
    {
        [Fact]
        public void NoOp_WhenPenaltyIsOne()
        {
            float[] logits = { 1.0f, 2.0f, -3.0f, 4.0f };
            int[] recent = { 0, 1, 2 };
            float[] expected = { 1.0f, 2.0f, -3.0f, 4.0f };

            TokenSampler.ApplyRepetitionPenalty(logits, recent, penalty: 1.0f);

            Assert.Equal(expected, logits);
        }

        [Fact]
        public void NoOp_WhenRecentTokensEmpty()
        {
            float[] logits = { 1.0f, 2.0f, 3.0f };
            float[] expected = { 1.0f, 2.0f, 3.0f };

            TokenSampler.ApplyRepetitionPenalty(logits, Array.Empty<int>(), penalty: 2.0f);

            Assert.Equal(expected, logits);
        }

        [Fact]
        public void PositiveLogit_DividedByPenalty()
        {
            float[] logits = { 4.0f, 8.0f };
            int[] recent = { 0 };

            TokenSampler.ApplyRepetitionPenalty(logits, recent, penalty: 2.0f);

            Assert.Equal(2.0f, logits[0]);  // 4.0 / 2.0
            Assert.Equal(8.0f, logits[1]);  // unchanged
        }

        [Fact]
        public void NegativeLogit_MultipliedByPenalty()
        {
            float[] logits = { -3.0f, 5.0f };
            int[] recent = { 0 };

            TokenSampler.ApplyRepetitionPenalty(logits, recent, penalty: 2.0f);

            Assert.Equal(-6.0f, logits[0]);  // -3.0 * 2.0
            Assert.Equal(5.0f, logits[1]);   // unchanged
        }

        [Fact]
        public void DuplicateTokens_AppliedMultipleTimes()
        {
            float[] logits = { 8.0f };
            int[] recent = { 0, 0, 0 };  // 3 times

            TokenSampler.ApplyRepetitionPenalty(logits, recent, penalty: 2.0f);

            // 8 / 2 / 2 / 2 = 1.0
            Assert.Equal(1.0f, logits[0]);
        }

        [Fact]
        public void OutOfRangeTokens_SilentlyIgnored()
        {
            float[] logits = { 1.0f, 2.0f };
            int[] recent = { -5, 0, 100, 1, 1000 };

            TokenSampler.ApplyRepetitionPenalty(logits, recent, penalty: 2.0f);

            Assert.Equal(0.5f, logits[0]);  // 1.0 / 2.0
            Assert.Equal(1.0f, logits[1]);  // 2.0 / 2.0
        }

        [Fact]
        public void Penalty_ChangesArgMax()
        {
            // Without penalty: ArgMax = 1 (highest logit 10.0)
            // With penalty: token 1 logit becomes 10/1.5 ≈ 6.67, so ArgMax = 2 (logit 8.0)
            float[] logits = { 5.0f, 10.0f, 8.0f, 3.0f };
            int[] recent = { 1 };

            TokenSampler.ApplyRepetitionPenalty(logits, recent, penalty: 1.5f);

            var argMax = TokenSampler.ArgMax(logits);
            Assert.Equal(2, argMax);
        }

        [Fact]
        public void GreedyWithPenalty_FactoryProducesExpectedDefaults()
        {
            var opts = SamplingOptions.GreedyWithPenalty();

            Assert.Equal(SamplingStrategy.Greedy, opts.Strategy);
            Assert.Equal(1.1f, opts.RepetitionPenalty);
            Assert.Equal(64, opts.RepetitionPenaltyContextSize);
        }

        [Fact]
        public void GreedyDefault_HasNoPenalty()
        {
            var opts = SamplingOptions.Greedy;

            Assert.Equal(1.0f, opts.RepetitionPenalty);
            Assert.Equal(0, opts.RepetitionPenaltyContextSize);
        }
    }
}
