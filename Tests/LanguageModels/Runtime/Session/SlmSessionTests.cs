// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit;

namespace DevOnBike.Overfit.Tests.LanguageModels
{
    public sealed class SlmSessionTests
    {
        [Fact]
        public void Constructor_ExposesModelShape()
        {
            using var model = CreateModel();
            using var session = new SlmSession(model, maxContextLength: 8);

            Assert.Equal(0, session.CurrentPosition);
            Assert.Equal(8, session.MaxContextLength);
            Assert.Equal(model.Config.VocabSize, session.VocabularySize);
            Assert.False(session.HasKeyValueCache);
        }

        [Fact]
        public void Constructor_ContextLongerThanModel_Throws()
        {
            using var model = CreateModel();

            Assert.Throws<ArgumentException>(() =>
                new SlmSession(model, model.Config.ContextLength + 1));
        }

        [Fact]
        public void Reset_WithPrompt_SetsCurrentPosition()
        {
            using var model = CreateModel();
            using var session = new SlmSession(model, maxContextLength: 8);

            session.Reset([1, 2, 3, 4]);

            Assert.Equal(4, session.CurrentPosition);
        }

        [Fact]
        public void Reset_WithLongPrompt_KeepsTotalPosition()
        {
            using var model = CreateModel();
            using var session = new SlmSession(model, maxContextLength: 4);

            session.Reset([1, 2, 3, 4, 5, 6]);

            Assert.Equal(6, session.CurrentPosition);
        }

        [Fact]
        public void GenerateNextToken_ReturnsTokenInVocabularyRange()
        {
            using var model = CreateModel();
            using var session = new SlmSession(model, maxContextLength: 8);

            session.Reset([1, 2, 3]);

            var token = session.GenerateNextToken(SamplingOptions.Greedy);

            Assert.InRange(token, 0, model.Config.VocabSize - 1);
            Assert.Equal(4, session.CurrentPosition);
        }

        [Fact]
        public void GenerateNextToken_WithoutPrompt_Throws()
        {
            using var model = CreateModel();
            using var session = new SlmSession(model, maxContextLength: 8);

            Assert.Throws<InvalidOperationException>(() =>
                session.GenerateNextToken(SamplingOptions.Greedy));
        }

        [Fact]
        public void GetLastLogits_AfterGenerateNextToken_ReturnsVocabularySizedLogits()
        {
            using var model = CreateModel();
            using var session = new SlmSession(model, maxContextLength: 8);
            var logits = new float[model.Config.VocabSize];

            session.Reset([1, 2, 3]);
            _ = session.GenerateNextToken(SamplingOptions.Greedy);
            session.GetLastLogits(logits);

            Assert.Equal(model.Config.VocabSize, logits.Length);
            Assert.DoesNotContain(logits, float.IsNaN);
            Assert.DoesNotContain(logits, float.IsInfinity);
        }

        [Fact]
        public void GetLastLogits_DestinationTooSmall_Throws()
        {
            using var model = CreateModel();
            using var session = new SlmSession(model, maxContextLength: 8);
            var logits = new float[model.Config.VocabSize - 1];

            Assert.Throws<ArgumentException>(() =>
                session.GetLastLogits(logits));
        }

        [Fact]
        public void Generate_WritesOutputTokensAndReturnsCount()
        {
            using var model = CreateModel();
            using var session = new SlmSession(model, maxContextLength: 8);
            var output = new int[2];

            var options = new GenerationOptions(
                maxNewTokens: 2,
                maxContextLength: 8,
                sampling: SamplingOptions.Greedy);

            var generated = session.Generate(
                [1, 2, 3],
                output,
                in options);

            Assert.Equal(2, generated);
            Assert.All(output, token => Assert.InRange(token, 0, model.Config.VocabSize - 1));
            Assert.Equal(5, session.CurrentPosition);
        }

        [Fact]
        public void Generate_StopsAtOutputSpanLength()
        {
            using var model = CreateModel();
            using var session = new SlmSession(model, maxContextLength: 8);
            var output = new int[1];

            var options = new GenerationOptions(
                maxNewTokens: 4,
                maxContextLength: 8,
                sampling: SamplingOptions.Greedy);

            var generated = session.Generate(
                [1, 2, 3],
                output,
                in options);

            Assert.Equal(1, generated);
        }

        [Fact]
        public void Dispose_ThenUse_Throws()
        {
            using var model = CreateModel();
            var session = new SlmSession(model, maxContextLength: 8);

            session.Dispose();

            Assert.Throws<ObjectDisposedException>(() =>
                session.Reset([1, 2, 3]));
        }

        private static GPT1Model CreateModel()
        {
            var config = new GPT1Config
            {
                VocabSize = 32,
                ContextLength = 8,
                DModel = 16,
                NHeads = 2,
                NLayers = 1,
                DFF = 32,
                TieWeights = true,
                PreLayerNorm = true
            };

            var model = new GPT1Model(config);
            model.Eval();

            return model;
        }
    }
}
