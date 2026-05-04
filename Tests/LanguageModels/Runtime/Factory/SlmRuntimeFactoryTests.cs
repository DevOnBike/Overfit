// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Factory
{
    public class SlmRuntimeFactoryTests
    {
        [Fact]
        public void CreateGpt1_DefaultMode_ReturnsCachedRuntime()
        {
            using var model = CreateSmallModel();
            using var runtime = SlmRuntimeFactory.CreateGpt1(model);

            Assert.Equal(SlmRuntimeMode.Cached, runtime.Mode);
            Assert.True(runtime.HasKeyValueCache);
            Assert.Equal(model.Config.ContextLength, runtime.MaxContextLength);
            Assert.Equal(model.Config.VocabSize, runtime.VocabularySize);
        }

        [Fact]
        public void CreateGpt1_LegacyMode_ReturnsLegacyRuntime()
        {
            using var model = CreateSmallModel();
            using var runtime = SlmRuntimeFactory.CreateGpt1(
                model,
                SlmRuntimeMode.Legacy);

            Assert.Equal(SlmRuntimeMode.Legacy, runtime.Mode);
            Assert.False(runtime.HasKeyValueCache);
            Assert.Equal(model.Config.ContextLength, runtime.MaxContextLength);
            Assert.Equal(model.Config.VocabSize, runtime.VocabularySize);
        }

        [Fact]
        public void CreateGpt1_CachedMode_ReturnsCachedRuntime()
        {
            using var model = CreateSmallModel();
            using var runtime = SlmRuntimeFactory.CreateGpt1(
                model,
                SlmRuntimeMode.Cached);

            Assert.Equal(SlmRuntimeMode.Cached, runtime.Mode);
            Assert.True(runtime.HasKeyValueCache);
        }

        [Fact]
        public void CreateGpt1_NullModel_Throws()
        {
            Assert.Throws<ArgumentNullException>(() =>
                SlmRuntimeFactory.CreateGpt1(null!));
        }

        [Fact]
        public void CreateLegacyGpt1_NullModel_Throws()
        {
            Assert.Throws<ArgumentNullException>(() =>
                SlmRuntimeFactory.CreateLegacyGpt1(null!));
        }

        [Fact]
        public void CreateCachedGpt1_NullModel_Throws()
        {
            Assert.Throws<ArgumentNullException>(() =>
                SlmRuntimeFactory.CreateCachedGpt1(null!));
        }

        [Fact]
        public void RuntimeHandle_GenerateGreedy_WritesTokens()
        {
            using var model = CreateSmallModel();
            using var runtime = SlmRuntimeFactory.CreateCachedGpt1(model);

            var output = new int[2];

            var generated = runtime.GenerateGreedy(
                promptTokens: [1, 2],
                outputTokens: output,
                maxNewTokens: 2);

            Assert.Equal(2, generated);
            Assert.All(output, token => Assert.InRange(token, 0, model.Config.VocabSize - 1));
        }

        [Fact]
        public void RuntimeHandle_Generate_WritesTokens()
        {
            using var model = CreateSmallModel();
            using var runtime = SlmRuntimeFactory.CreateCachedGpt1(model);

            var output = new int[3];

            var options = new GenerationOptions(
                maxNewTokens: 3,
                maxContextLength: model.Config.ContextLength,
                sampling: SamplingOptions.Greedy,
                stopOnEndOfTextToken: false);

            var generated = runtime.Generate(
                promptTokens: [1, 2],
                outputTokens: output,
                in options);

            Assert.Equal(3, generated);
            Assert.All(output, token => Assert.InRange(token, 0, model.Config.VocabSize - 1));
        }

        [Fact]
        public void RuntimeHandle_Dispose_ThenUse_Throws()
        {
            using var model = CreateSmallModel();
            var runtime = SlmRuntimeFactory.CreateCachedGpt1(model);

            runtime.Dispose();

            Assert.Throws<ObjectDisposedException>(() =>
            {
                _ = runtime.Session;
            });

            Assert.Throws<ObjectDisposedException>(() =>
            {
                _ = runtime.HasKeyValueCache;
            });

            Assert.Throws<ObjectDisposedException>(() =>
                runtime.GenerateGreedy(
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
