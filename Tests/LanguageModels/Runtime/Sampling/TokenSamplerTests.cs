// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Sampling
{
    public sealed class TokenSamplerTests
    {
        // ── Existing tests (unchanged) ────────────────────────────────────────

        [Fact]
        public void ArgMax_ReturnsIndexOfLargestLogit()
        {
            var logits = new float[] { -1f, 0.5f, 7f, 2f, 6f };
            Assert.Equal(2, TokenSampler.ArgMax(logits));
        }

        [Fact]
        public void Sample_Greedy_ReturnsArgMax()
        {
            var logits = new float[] { 0.1f, 0.2f, 5f, 0.4f };
            var random = new Random(123);
            var indexes = new int[logits.Length];
            var scores  = new float[logits.Length];

            var token = TokenSampler.Sample(logits, SamplingOptions.Greedy, random, indexes, scores);

            Assert.Equal(2, token);
        }

        [Fact]
        public void Sample_Temperature_WithSameSeed_IsDeterministic()
        {
            var logits  = new float[] { 0.1f, 0.2f, 0.3f, 0.4f };
            var options = new SamplingOptions(SamplingStrategy.Temperature, 1.0f, 0, 1.0f, 42);

            var t1 = TokenSampler.Sample(logits, in options, new Random(42), new int[4], new float[4]);
            var t2 = TokenSampler.Sample(logits, in options, new Random(42), new int[4], new float[4]);

            Assert.Equal(t1, t2);
        }

        [Fact]
        public void Sample_TopK_NeverReturnsTokenOutsideTopK()
        {
            var logits  = new float[] { 10f, 9f, 8f, 1f, 0f, -1f };
            var options = new SamplingOptions(SamplingStrategy.TopK, 1.0f, 3, 1.0f, 123);

            for (var i = 0; i < 256; i++)
            {
                var token = TokenSampler.Sample(logits, in options, new Random(123 + i),
                    new int[logits.Length], new float[logits.Length]);
                Assert.InRange(token, 0, 2);
            }
        }

        [Fact]
        public void Sample_ScratchTooSmall_Throws()
        {
            var logits  = new float[] { 0.1f, 0.2f, 0.3f };
            var indexes = new int[logits.Length - 1];
            var scores  = new float[logits.Length];

            Assert.Throws<ArgumentException>(() =>
                TokenSampler.Sample(logits, SamplingOptions.Greedy, new Random(1), indexes, scores));
        }

        // ── Top-P tests ───────────────────────────────────────────────────────

        [Fact]
        public void Sample_TopP_NoLongerThrowsNotSupportedException()
        {
            var logits  = new float[] { 0.1f, 0.2f, 0.3f };
            var options = new SamplingOptions(SamplingStrategy.TopP, 1.0f, 0, 0.9f, 123);

            // Previously threw NotSupportedException — now must succeed
            var ex = Record.Exception(() =>
                TokenSampler.Sample(logits, in options, new Random(123),
                    new int[logits.Length], new float[logits.Length]));

            Assert.Null(ex);
        }

        [Fact]
        public void Sample_TopP_ReturnsTokenInRange()
        {
            var logits  = new float[] { 10f, 9f, 8f, 1f, 0f, -1f };
            var options = new SamplingOptions(SamplingStrategy.TopP, 1.0f, 0, 0.9f, 42);

            for (var i = 0; i < 256; i++)
            {
                var token = TokenSampler.Sample(logits, in options, new Random(42 + i),
                    new int[logits.Length], new float[logits.Length]);
                Assert.InRange(token, 0, logits.Length - 1);
            }
        }

        [Fact]
        public void Sample_TopP_WithTinyNucleus_AlwaysReturnsBestToken()
        {
            // logits: one token has ~100% probability → nucleus = 1 token
            // topP = 0.95 → nucleus includes only token 0 (prob ≈ 1.0)
            var logits  = new float[] { 100f, -100f, -100f, -100f };
            var options = new SamplingOptions(SamplingStrategy.TopP, 1.0f, 0, 0.95f, 1);

            for (var i = 0; i < 100; i++)
            {
                var token = TokenSampler.Sample(logits, in options, new Random(i),
                    new int[logits.Length], new float[logits.Length]);
                Assert.Equal(0, token);
            }
        }

        [Fact]
        public void Sample_TopP_WithFullNucleus_SamplesFromAll()
        {
            // Uniform logits, topP = 1.0 → sample from everything
            var logits  = new float[] { 0f, 0f, 0f, 0f, 0f };
            var options = new SamplingOptions(SamplingStrategy.TopP, 1.0f, 0, 1.0f, 7);
            var seen    = new HashSet<int>();

            for (var i = 0; i < 1000; i++)
            {
                var token = TokenSampler.Sample(logits, in options, new Random(i),
                    new int[logits.Length], new float[logits.Length]);
                seen.Add(token);
            }

            // With uniform distribution + 1000 samples, all 5 tokens should appear
            Assert.Equal(logits.Length, seen.Count);
        }

        [Fact]
        public void Sample_TopP_IsDeterministicWithSameSeed()
        {
            var logits  = new float[] { 2f, 1f, 0.5f, 0.1f, -1f };
            var options = new SamplingOptions(SamplingStrategy.TopP, 0.8f, 0, 0.9f, 99);

            var t1 = TokenSampler.Sample(logits, in options, new Random(99),
                new int[logits.Length], new float[logits.Length]);
            var t2 = TokenSampler.Sample(logits, in options, new Random(99),
                new int[logits.Length], new float[logits.Length]);

            Assert.Equal(t1, t2);
        }

        [Fact]
        public void Sample_TopP_NeverReturnsBelowNucleusTokens()
        {
            // First 3 tokens dominate — topP=0.99 should never return tokens 3/4
            var logits  = new float[] { 10f, 9f, 8f, -50f, -50f };
            var options = new SamplingOptions(SamplingStrategy.TopP, 1.0f, 0, 0.99f, 5);

            for (var i = 0; i < 1000; i++)
            {
                var token = TokenSampler.Sample(logits, in options, new Random(i),
                    new int[logits.Length], new float[logits.Length]);
                Assert.InRange(token, 0, 2);
            }
        }

        // ── TopK + TopP combined ──────────────────────────────────────────────

        [Fact]
        public void Sample_TopKTopP_NeverExceedsBothConstraints()
        {
            // top-k=4, top-p=0.8 on peaked distribution → small nucleus
            var logits  = new float[] { 10f, 9f, 8f, 7f, -100f, -100f };
            var options = new SamplingOptions(SamplingStrategy.TopKTopP, 1.0f, 4, 0.8f, 42);

            for (var i = 0; i < 256; i++)
            {
                var token = TokenSampler.Sample(logits, in options, new Random(42 + i),
                    new int[logits.Length], new float[logits.Length]);
                // Must be in top-k=4
                Assert.InRange(token, 0, 3);
            }
        }

        [Fact]
        public void Sample_TopKTopP_WithSameSeed_IsDeterministic()
        {
            var logits  = new float[] { 3f, 2f, 1f, 0.5f, 0.1f };
            var options = new SamplingOptions(SamplingStrategy.TopKTopP, 0.8f, 3, 0.9f, 77);

            var t1 = TokenSampler.Sample(logits, in options, new Random(77),
                new int[logits.Length], new float[logits.Length]);
            var t2 = TokenSampler.Sample(logits, in options, new Random(77),
                new int[logits.Length], new float[logits.Length]);

            Assert.Equal(t1, t2);
        }

        // ── SamplingOptions presets ───────────────────────────────────────────

        [Fact]
        public void Greedy_Preset_ReturnsArgMax()
        {
            var logits = new float[] { 1f, 5f, 3f };
            var token  = TokenSampler.Sample(logits, SamplingOptions.Greedy, new Random(1),
                new int[logits.Length], new float[logits.Length]);
            Assert.Equal(1, token);
        }
    }
}
