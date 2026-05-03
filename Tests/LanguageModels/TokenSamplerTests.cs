// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
//
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit;

namespace DevOnBike.Overfit.Tests.LanguageModels
{
    public sealed class TokenSamplerTests
    {
        [Fact]
        public void ArgMax_ReturnsIndexOfLargestLogit()
        {
            var logits = new float[] { -1f, 0.5f, 7f, 2f, 6f };

            var actual = TokenSampler.ArgMax(logits);

            Assert.Equal(2, actual);
        }

        [Fact]
        public void Sample_Greedy_ReturnsArgMax()
        {
            var logits = new float[] { 0.1f, 0.2f, 5f, 0.4f };
            var random = new Random(123);
            var indexes = new int[logits.Length];
            var scores = new float[logits.Length];

            var token = TokenSampler.Sample(
                logits,
                SamplingOptions.Greedy,
                random,
                indexes,
                scores);

            Assert.Equal(2, token);
        }

        [Fact]
        public void Sample_Temperature_WithSameSeed_IsDeterministic()
        {
            var logits = new float[] { 0.1f, 0.2f, 0.3f, 0.4f };
            var indexes1 = new int[logits.Length];
            var scores1 = new float[logits.Length];
            var indexes2 = new int[logits.Length];
            var scores2 = new float[logits.Length];

            var options = new SamplingOptions(
                strategy: SamplingStrategy.Temperature,
                temperature: 1.0f,
                topK: 0,
                topP: 1.0f,
                seed: 42);

            var token1 = TokenSampler.Sample(
                logits,
                in options,
                new Random(options.Seed),
                indexes1,
                scores1);

            var token2 = TokenSampler.Sample(
                logits,
                in options,
                new Random(options.Seed),
                indexes2,
                scores2);

            Assert.Equal(token1, token2);
        }

        [Fact]
        public void Sample_TopK_NeverReturnsTokenOutsideTopK()
        {
            var logits = new float[] { 10f, 9f, 8f, 1f, 0f, -1f };
            var indexes = new int[logits.Length];
            var scores = new float[logits.Length];

            var options = new SamplingOptions(
                strategy: SamplingStrategy.TopK,
                temperature: 1.0f,
                topK: 3,
                topP: 1.0f,
                seed: 123);

            for (var i = 0; i < 256; i++)
            {
                var token = TokenSampler.Sample(
                    logits,
                    in options,
                    new Random(123 + i),
                    indexes,
                    scores);

                Assert.InRange(token, 0, 2);
            }
        }

        [Fact]
        public void Sample_TopP_ThrowsUntilImplemented()
        {
            var logits = new float[] { 0.1f, 0.2f, 0.3f };
            var random = new Random(123);
            var indexes = new int[logits.Length];
            var scores = new float[logits.Length];

            var options = new SamplingOptions(
                strategy: SamplingStrategy.TopP,
                temperature: 1.0f,
                topK: 0,
                topP: 0.9f,
                seed: 123);

            Assert.Throws<NotSupportedException>(() =>
                TokenSampler.Sample(
                    logits,
                    in options,
                    random,
                    indexes,
                    scores));
        }

        [Fact]
        public void Sample_ScratchTooSmall_Throws()
        {
            var logits = new float[] { 0.1f, 0.2f, 0.3f };
            var random = new Random(123);
            var indexes = new int[logits.Length - 1];
            var scores = new float[logits.Length];

            Assert.Throws<ArgumentException>(() =>
                TokenSampler.Sample(
                    logits,
                    SamplingOptions.Greedy,
                    random,
                    indexes,
                    scores));
        }
    }
}
