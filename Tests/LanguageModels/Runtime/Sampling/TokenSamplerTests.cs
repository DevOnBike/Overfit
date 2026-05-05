// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
//
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Sampling
{
    public sealed class TokenSamplerTests
    {
        [Fact]
        public void ArgMax_ReturnsIndexOfLargestLogit()
        {
            var logits = new float[] { -1f, 0.5f, 7f, 2f, 6f };

            Assert.Equal(
                2,
                TokenSampler.ArgMax(logits));
        }

        [Fact]
        public void ArgMax_ReturnsFirstMax_WhenTied()
        {
            var logits = new float[] { 1f, 5f, 5f, 4f };

            Assert.Equal(
                1,
                TokenSampler.ArgMax(logits));
        }

        [Fact]
        public void ArgMax_Empty_Throws()
        {
            Assert.Throws<ArgumentException>(
                () => TokenSampler.ArgMax([]));
        }

        [Fact]
        public void Sample_Greedy_ReturnsArgMax()
        {
            var logits = new float[] { 0.1f, 0.2f, 5f, 0.4f };

            var token = TokenSampler.Sample(
                logits,
                SamplingOptions.Greedy,
                new Random(123),
                new int[logits.Length],
                new float[logits.Length]);

            Assert.Equal(
                2,
                token);
        }

        [Fact]
        public void Sample_Greedy_DoesNotRequireRandomOrScratchBuffers()
        {
            var logits = new float[] { 0.1f, 0.2f, 5f, 0.4f };

            var token = TokenSampler.Sample(
                logits,
                SamplingOptions.Greedy,
                null!,
                [],
                []);

            Assert.Equal(
                2,
                token);
        }

        [Fact]
        public void Sample_Temperature_WithSameSeed_IsDeterministic()
        {
            var logits = new float[] { 0.1f, 0.2f, 0.3f, 0.4f };
            var options = new SamplingOptions(
                SamplingStrategy.Temperature,
                1.0f,
                0,
                1.0f,
                42);

            var t1 = TokenSampler.Sample(
                logits,
                in options,
                new Random(42),
                new int[4],
                new float[4]);

            var t2 = TokenSampler.Sample(
                logits,
                in options,
                new Random(42),
                new int[4],
                new float[4]);

            Assert.Equal(
                t1,
                t2);
        }

        [Fact]
        public void Sample_Temperature_ScratchTooSmall_Throws()
        {
            var logits = new float[] { 0.1f, 0.2f, 0.3f };
            var options = new SamplingOptions(
                SamplingStrategy.Temperature,
                1.0f,
                0,
                1.0f,
                1);

            Assert.Throws<ArgumentException>(
                () => TokenSampler.Sample(
                    logits,
                    in options,
                    new Random(1),
                    new int[logits.Length - 1],
                    new float[logits.Length]));

            Assert.Throws<ArgumentException>(
                () => TokenSampler.Sample(
                    logits,
                    in options,
                    new Random(1),
                    new int[logits.Length],
                    new float[logits.Length - 1]));
        }

        [Fact]
        public void Sample_Temperature_NullRandom_Throws()
        {
            var logits = new float[] { 0.1f, 0.2f, 0.3f };
            var options = new SamplingOptions(
                SamplingStrategy.Temperature,
                1.0f,
                0,
                1.0f,
                1);

            Assert.Throws<ArgumentNullException>(
                () => TokenSampler.Sample(
                    logits,
                    in options,
                    null!,
                    new int[logits.Length],
                    new float[logits.Length]));
        }

        [Fact]
        public void Sample_TopK_NeverReturnsTokenOutsideTopK()
        {
            var logits = new float[] { 10f, 9f, 8f, 1f, 0f, -1f };
            var options = new SamplingOptions(
                SamplingStrategy.TopK,
                1.0f,
                3,
                1.0f,
                123);

            for (var i = 0; i < 256; i++)
            {
                var token = TokenSampler.Sample(
                    logits,
                    in options,
                    new Random(123 + i),
                    new int[logits.Length],
                    new float[logits.Length]);

                Assert.InRange(
                    token,
                    0,
                    2);
            }
        }

        [Fact]
        public void Sample_TopK_WithTemperature_NeverReturnsTokenOutsideTopK()
        {
            var logits = new float[] { 10f, 9f, 8f, 1f, 0f, -1f };
            var options = new SamplingOptions(
                SamplingStrategy.TopK,
                0.5f,
                3,
                1.0f,
                123);

            for (var i = 0; i < 256; i++)
            {
                var token = TokenSampler.Sample(
                    logits,
                    in options,
                    new Random(123 + i),
                    new int[logits.Length],
                    new float[logits.Length]);

                Assert.InRange(
                    token,
                    0,
                    2);
            }
        }

        [Fact]
        public void Sample_TopP_NoLongerThrowsNotSupportedException()
        {
            var logits = new float[] { 0.1f, 0.2f, 0.3f };
            var options = new SamplingOptions(
                SamplingStrategy.TopP,
                1.0f,
                0,
                0.9f,
                123);

            var ex = Record.Exception(
                () => TokenSampler.Sample(
                    logits,
                    in options,
                    new Random(123),
                    new int[logits.Length],
                    new float[logits.Length]));

            Assert.Null(ex);
        }

        [Fact]
        public void Sample_TopP_ReturnsTokenInRange()
        {
            var logits = new float[] { 10f, 9f, 8f, 1f, 0f, -1f };
            var options = new SamplingOptions(
                SamplingStrategy.TopP,
                1.0f,
                0,
                0.9f,
                42);

            for (var i = 0; i < 256; i++)
            {
                var token = TokenSampler.Sample(
                    logits,
                    in options,
                    new Random(42 + i),
                    new int[logits.Length],
                    new float[logits.Length]);

                Assert.InRange(
                    token,
                    0,
                    logits.Length - 1);
            }
        }

        [Fact]
        public void Sample_TopP_WithTinyNucleus_AlwaysReturnsBestToken()
        {
            var logits = new float[] { 100f, -100f, -100f, -100f };
            var options = new SamplingOptions(
                SamplingStrategy.TopP,
                1.0f,
                0,
                0.95f,
                1);

            for (var i = 0; i < 100; i++)
            {
                var token = TokenSampler.Sample(
                    logits,
                    in options,
                    new Random(i),
                    new int[logits.Length],
                    new float[logits.Length]);

                Assert.Equal(
                    0,
                    token);
            }
        }

        [Fact]
        public void Sample_TopP_WithFullNucleus_SamplesFromAll()
        {
            var logits = new float[] { 0f, 0f, 0f, 0f, 0f };
            var options = new SamplingOptions(
                SamplingStrategy.TopP,
                1.0f,
                0,
                1.0f,
                7);

            var seen = new HashSet<int>();

            for (var i = 0; i < 1000; i++)
            {
                var token = TokenSampler.Sample(
                    logits,
                    in options,
                    new Random(i),
                    new int[logits.Length],
                    new float[logits.Length]);

                seen.Add(token);
            }

            Assert.Equal(
                logits.Length,
                seen.Count);
        }

        [Fact]
        public void Sample_TopP_IsDeterministicWithSameSeed()
        {
            var logits = new float[] { 2f, 1f, 0.5f, 0.1f, -1f };
            var options = new SamplingOptions(
                SamplingStrategy.TopP,
                0.8f,
                0,
                0.9f,
                99);

            var t1 = TokenSampler.Sample(
                logits,
                in options,
                new Random(99),
                new int[logits.Length],
                new float[logits.Length]);

            var t2 = TokenSampler.Sample(
                logits,
                in options,
                new Random(99),
                new int[logits.Length],
                new float[logits.Length]);

            Assert.Equal(
                t1,
                t2);
        }

        [Fact]
        public void Sample_TopP_NeverReturnsBelowNucleusTokens()
        {
            var logits = new float[] { 10f, 9f, 8f, -50f, -50f };
            var options = new SamplingOptions(
                SamplingStrategy.TopP,
                1.0f,
                0,
                0.99f,
                5);

            for (var i = 0; i < 1000; i++)
            {
                var token = TokenSampler.Sample(
                    logits,
                    in options,
                    new Random(i),
                    new int[logits.Length],
                    new float[logits.Length]);

                Assert.InRange(
                    token,
                    0,
                    2);
            }
        }

        [Fact]
        public void Sample_TopKTopP_NeverExceedsBothConstraints()
        {
            var logits = new float[] { 10f, 9f, 8f, 7f, -100f, -100f };
            var options = new SamplingOptions(
                SamplingStrategy.TopKTopP,
                1.0f,
                4,
                0.8f,
                42);

            for (var i = 0; i < 256; i++)
            {
                var token = TokenSampler.Sample(
                    logits,
                    in options,
                    new Random(42 + i),
                    new int[logits.Length],
                    new float[logits.Length]);

                Assert.InRange(
                    token,
                    0,
                    3);
            }
        }

        [Fact]
        public void Sample_TopKTopP_WithSameSeed_IsDeterministic()
        {
            var logits = new float[] { 3f, 2f, 1f, 0.5f, 0.1f };
            var options = new SamplingOptions(
                SamplingStrategy.TopKTopP,
                0.8f,
                3,
                0.9f,
                77);

            var t1 = TokenSampler.Sample(
                logits,
                in options,
                new Random(77),
                new int[logits.Length],
                new float[logits.Length]);

            var t2 = TokenSampler.Sample(
                logits,
                in options,
                new Random(77),
                new int[logits.Length],
                new float[logits.Length]);

            Assert.Equal(
                t1,
                t2);
        }

        [Fact]
        public void Greedy_Preset_ReturnsArgMax()
        {
            var logits = new float[] { 1f, 5f, 3f };

            var token = TokenSampler.Sample(
                logits,
                SamplingOptions.Greedy,
                new Random(1),
                new int[logits.Length],
                new float[logits.Length]);

            Assert.Equal(
                1,
                token);
        }
    }
}
