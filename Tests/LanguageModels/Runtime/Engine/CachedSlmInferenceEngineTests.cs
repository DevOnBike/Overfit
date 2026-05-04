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
    public class CachedSlmInferenceEngineTests
    {
        [Fact]
        public void FromGpt1_ExposesModelShape()
        {
            using var model = CreateSmallModel();
            using var engine = CachedSlmInferenceEngine.FromGpt1(model);

            Assert.Equal(model.Config.VocabSize, engine.VocabularySize);
            Assert.Equal(model.Config.ContextLength, engine.MaxContextLength);
            Assert.Equal(model.Config.DModel, engine.DModel);
            Assert.Equal(model.Config.NLayers, engine.LayerCount);
            Assert.Equal(model.Config.NHeads, engine.HeadCount);
            Assert.True(engine.HasKeyValueCache);
        }

        [Fact]
        public void FromGpt1_NullModel_Throws()
        {
            Assert.Throws<ArgumentNullException>(() =>
                CachedSlmInferenceEngine.FromGpt1(null!));
        }

        [Fact]
        public void CreateSession_ReturnsCachedSession()
        {
            using var model = CreateSmallModel();
            using var engine = CachedSlmInferenceEngine.FromGpt1(model);

            using var session = engine.CreateSession();

            Assert.True(session.HasKeyValueCache);
            Assert.Equal(model.Config.ContextLength, session.MaxContextLength);
            Assert.Equal(model.Config.VocabSize, session.VocabularySize);
        }

        [Fact]
        public void Generate_ProducesRequestedTokens()
        {
            using var model = CreateSmallModel();
            using var engine = CachedSlmInferenceEngine.FromGpt1(model);

            var output = new int[3];

            var options = new GenerationOptions(
                maxNewTokens: 3,
                maxContextLength: model.Config.ContextLength,
                sampling: SamplingOptions.Greedy,
                stopOnEndOfTextToken: false);

            var generated = engine.Generate(
                promptTokens: [1, 2],
                outputTokens: output,
                in options);

            Assert.Equal(3, generated);
            Assert.All(output, token => Assert.InRange(token, 0, model.Config.VocabSize - 1));
        }

        [Fact]
        public void GenerateGreedy_ProducesRequestedTokens()
        {
            using var model = CreateSmallModel();
            using var engine = CachedSlmInferenceEngine.FromGpt1(model);

            var output = new int[2];

            var generated = engine.GenerateGreedy(
                promptTokens: [1, 2],
                outputTokens: output,
                maxNewTokens: 2);

            Assert.Equal(2, generated);
            Assert.All(output, token => Assert.InRange(token, 0, model.Config.VocabSize - 1));
        }

        [Fact]
        public void Generate_StopsAtOutputBufferLength()
        {
            using var model = CreateSmallModel();
            using var engine = CachedSlmInferenceEngine.FromGpt1(model);

            var output = new int[2];

            var options = new GenerationOptions(
                maxNewTokens: 5,
                maxContextLength: model.Config.ContextLength,
                sampling: SamplingOptions.Greedy,
                stopOnEndOfTextToken: false);

            var generated = engine.Generate(
                promptTokens: [1, 2],
                outputTokens: output,
                in options);

            Assert.Equal(2, generated);
        }

        [Fact]
        public void Dispose_ThenUse_Throws()
        {
            using var model = CreateSmallModel();
            var engine = CachedSlmInferenceEngine.FromGpt1(model);

            engine.Dispose();

            Assert.Throws<ObjectDisposedException>(() =>
                engine.CreateSession());

            Assert.Throws<ObjectDisposedException>(() =>
                engine.GenerateGreedy(
                    promptTokens: [1, 2],
                    outputTokens: new int[1],
                    maxNewTokens: 1));
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
