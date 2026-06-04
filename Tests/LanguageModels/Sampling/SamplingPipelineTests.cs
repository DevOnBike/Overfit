// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Sampling;

namespace DevOnBike.Overfit.Tests.LanguageModels.Sampling
{
    /// <summary>
    /// Fast tests for the composable <see cref="SamplingPipeline"/> and its built-in steps/processors:
    /// top-k / top-p / min-p survivor selection, the repetition-penalty processor, and the terminal draw.
    /// </summary>
    public sealed class SamplingPipelineTests
    {
        [Fact]
        public void TopK1_AlwaysSelectsTheArgmax()
        {
            var pipeline = new SamplingPipeline().Use(new SamplingPipeline.TopK(1));
            var rng = new Random(1);
            for (var trial = 0; trial < 20; trial++)
            {
                var logits = new[] { 0.1f, 0.5f, 0.2f, 0.9f, 0.3f };
                Assert.Equal(3, pipeline.Sample(logits, ReadOnlySpan<int>.Empty, rng));   // 0.9 is the max
            }
        }

        [Fact]
        public void TopK_SamplesOnlyAmongTheKHighest()
        {
            var pipeline = new SamplingPipeline().Use(new SamplingPipeline.TopK(2));
            var rng = new Random(7);
            for (var trial = 0; trial < 50; trial++)
            {
                var logits = new[] { 0.1f, 0.5f, 0.2f, 0.9f };   // top-2 are indices 3 (0.9) and 1 (0.5)
                var token = pipeline.Sample(logits, ReadOnlySpan<int>.Empty, rng);
                Assert.Contains(token, new[] { 1, 3 });
            }
        }

        [Fact]
        public void TopP_KeepsOnlyTheNucleus()
        {
            // A peaked distribution: token 0 carries almost all the mass, so top-p 0.5 keeps only it.
            var step = new SamplingPipeline.TopP(0.5f);
            var logits = new[] { 10f, 0f, 0f, 0f };
            step.Apply(logits);

            Assert.False(float.IsNegativeInfinity(logits[0]));
            Assert.True(float.IsNegativeInfinity(logits[1]));
            Assert.True(float.IsNegativeInfinity(logits[2]));
            Assert.True(float.IsNegativeInfinity(logits[3]));
        }

        [Fact]
        public void MinP_MasksTokensBelowTheRelativeThreshold()
        {
            var step = new SamplingPipeline.MinP(0.5f);
            // exp(0)=1 vs exp(-5)≈0.0067 — the far-below token is < 0.5×P(top), so it is masked.
            var logits = new[] { 0f, -0.1f, -5f };
            step.Apply(logits);

            Assert.False(float.IsNegativeInfinity(logits[0]));
            Assert.False(float.IsNegativeInfinity(logits[1]));   // close to the top — survives
            Assert.True(float.IsNegativeInfinity(logits[2]));    // far below — masked
        }

        [Fact]
        public void RepetitionPenalty_LowersLogitsOfRecentTokens()
        {
            var processor = new SamplingPipeline.RepetitionPenalty(2.0f);
            var logits = new[] { 4f, -4f, 1f };
            int[] history = [0, 1];   // tokens 0 and 1 were recently generated
            processor.Process(logits, history);

            Assert.Equal(2f, logits[0]);    // positive logit divided by penalty (4/2)
            Assert.Equal(-8f, logits[1]);   // negative logit multiplied by penalty (-4*2)
            Assert.Equal(1f, logits[2]);    // untouched
        }

        [Fact]
        public void Composed_ProcessorThenSteps_SamplesAValidToken()
        {
            var pipeline = new SamplingPipeline()
                .Use(new SamplingPipeline.RepetitionPenalty(1.3f, contextSize: 8))
                .Use(new SamplingPipeline.Temperature(0.8f))
                .Use(new SamplingPipeline.TopP(0.95f));

            var logits = new[] { 1.0f, 2.0f, 0.5f, 3.0f, 0.2f };
            int[] history = [3];
            var token = pipeline.Sample(logits, history, new Random(123));

            Assert.InRange(token, 0, 4);
        }
    }
}
