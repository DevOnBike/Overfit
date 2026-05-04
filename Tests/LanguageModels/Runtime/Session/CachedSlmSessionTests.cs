// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Session
{
    public class CachedSlmSessionTests
    {
        [Fact]
        public void Constructor_ExposesShapeAndCacheSupport()
        {
            using var model = CreateSmallModel();
            using var session = new CachedSlmSession(model);

            Assert.Equal(model.Config.ContextLength, session.MaxContextLength);
            Assert.Equal(model.Config.VocabSize, session.VocabularySize);
            Assert.True(session.HasKeyValueCache);
            Assert.Equal(0, session.CurrentPosition);
        }

        [Fact]
        public void Reset_WithPrompt_PreFillsCacheAndLogits()
        {
            using var model = CreateSmallModel();
            using var session = new CachedSlmSession(model);

            session.Reset([1, 2, 3]);

            Assert.Equal(3, session.CurrentPosition);

            var logits = new float[model.Config.VocabSize];
            session.GetLastLogits(logits);

            Assert.DoesNotContain(logits, float.IsNaN);
            Assert.DoesNotContain(logits, float.IsInfinity);
        }

        [Fact]
        public void Reset_EmptyPrompt_ClearsPositionAndLogits()
        {
            using var model = CreateSmallModel();
            using var session = new CachedSlmSession(model);

            session.Reset([1, 2]);
            Assert.Equal(2, session.CurrentPosition);

            session.Reset();

            Assert.Equal(0, session.CurrentPosition);

            var logits = new float[model.Config.VocabSize];
            session.GetLastLogits(logits);

            Assert.All(logits, value => Assert.Equal(0f, value));
        }

        [Fact]
        public void Reset_PromptLongerThanContext_Throws()
        {
            using var model = CreateSmallModel();
            using var session = new CachedSlmSession(model);

            var prompt = Enumerable
                .Range(0, model.Config.ContextLength + 1)
                .Select(i => i % model.Config.VocabSize)
                .ToArray();

            Assert.Throws<ArgumentException>(() =>
                session.Reset(prompt));
        }

        [Fact]
        public void GenerateNextToken_FromPrompt_ReturnsTokenAndAdvancesPosition()
        {
            using var model = CreateSmallModel();
            using var session = new CachedSlmSession(model);

            var sampling = SamplingOptions.Greedy;

            session.Reset([1, 2]);

            var token = session.GenerateNextToken(in sampling);

            Assert.InRange(token, 0, model.Config.VocabSize - 1);
            Assert.Equal(3, session.CurrentPosition);
        }

        [Fact]
        public void GenerateNextToken_WithoutPrompt_Throws()
        {
            using var model = CreateSmallModel();
            using var session = new CachedSlmSession(model);

            var sampling = SamplingOptions.Greedy;

            Assert.Throws<InvalidOperationException>(() =>
                session.GenerateNextToken(in sampling));
        }

        [Fact]
        public void Generate_WritesRequestedTokensAndAdvancesPosition()
        {
            using var model = CreateSmallModel();
            using var session = new CachedSlmSession(model);

            var output = new int[3];

            var options = new GenerationOptions(
                maxNewTokens: 3,
                maxContextLength: model.Config.ContextLength,
                sampling: SamplingOptions.Greedy,
                stopOnEndOfTextToken: false);

            var generated = session.Generate(
                promptTokens: [1, 2],
                outputTokens: output,
                in options);

            Assert.Equal(3, generated);
            Assert.Equal(5, session.CurrentPosition);
            Assert.All(output, token => Assert.InRange(token, 0, model.Config.VocabSize - 1));
        }

        [Fact]
        public void Generate_StopsAtOutputBufferLength()
        {
            using var model = CreateSmallModel();
            using var session = new CachedSlmSession(model);

            var output = new int[2];

            var options = new GenerationOptions(
                maxNewTokens: 5,
                maxContextLength: model.Config.ContextLength,
                sampling: SamplingOptions.Greedy,
                stopOnEndOfTextToken: false);

            var generated = session.Generate(
                promptTokens: [1, 2],
                outputTokens: output,
                in options);

            Assert.Equal(2, generated);
            Assert.Equal(4, session.CurrentPosition);
        }

        [Fact]
        public void Generate_WhenCacheWouldOverflow_Throws()
        {
            var config = new GPT1Config
            {
                VocabSize = 16,
                ContextLength = 4,
                DModel = 8,
                NHeads = 2,
                NLayers = 1,
                DFF = 16,
                TieWeights = true,
                PreLayerNorm = true
            };

            using var model = new GPT1Model(config);
            using var session = new CachedSlmSession(model);

            var output = new int[2];

            var options = new GenerationOptions(
                maxNewTokens: 2,
                maxContextLength: config.ContextLength,
                sampling: SamplingOptions.Greedy,
                stopOnEndOfTextToken: false);

            Assert.Throws<InvalidOperationException>(() =>
                session.Generate(
                    promptTokens: [1, 2, 3, 4],
                    outputTokens: output,
                    in options));
        }

        [Fact]
        public void GetLastLogits_DestinationTooSmall_Throws()
        {
            using var model = CreateSmallModel();
            using var session = new CachedSlmSession(model);

            Assert.Throws<ArgumentException>(() =>
                session.GetLastLogits(new float[model.Config.VocabSize - 1]));
        }

        [Fact]
        public void RefreshWeightsFromModel_DoesNotThrow()
        {
            using var model = CreateSmallModel();
            using var session = new CachedSlmSession(model);

            session.RefreshWeightsFromModel();
        }

        [Fact]
        public void Dispose_ThenUse_Throws()
        {
            using var model = CreateSmallModel();
            var session = new CachedSlmSession(model);

            session.Dispose();

            Assert.Throws<ObjectDisposedException>(() =>
                session.Reset());

            var sampling = SamplingOptions.Greedy;

            Assert.Throws<ObjectDisposedException>(() =>
                session.GenerateNextToken(in sampling));
        }

        private static GPT1Model CreateSmallModel()
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
