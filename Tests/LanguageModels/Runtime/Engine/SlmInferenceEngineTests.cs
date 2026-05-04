// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Engine
{
    public sealed class SlmInferenceEngineTests
    {
        [Fact]
        public void FromGpt1_ExposesModelMetadata()
        {
            using var model = CreateModel();
            using var engine = SlmInferenceEngine.FromGpt1(model);

            Assert.Equal(model.Config.VocabSize, engine.VocabularySize);
            Assert.Equal(model.Config.ContextLength, engine.MaxContextLength);
            Assert.False(engine.SupportsKeyValueCache);
            Assert.True(engine.SupportsStreaming);
            Assert.Equal(model.Config.VocabSize, engine.Model.VocabularySize);
            Assert.Equal(model.Config.ContextLength, engine.Model.ContextLength);
        }

        [Fact]
        public void CreateSession_ReturnsSessionForFullContext()
        {
            using var model = CreateModel();
            using var engine = SlmInferenceEngine.FromGpt1(model);

            using var session = engine.CreateSession();

            Assert.Equal(model.Config.ContextLength, session.MaxContextLength);
            Assert.Equal(model.Config.VocabSize, session.VocabularySize);
        }

        [Fact]
        public void CreateSession_WithCustomContext_ReturnsSessionForRequestedContext()
        {
            using var model = CreateModel();
            using var engine = SlmInferenceEngine.FromGpt1(model);

            using var session = engine.CreateSession(4);

            Assert.Equal(4, session.MaxContextLength);
        }

        [Fact]
        public void Generate_WritesTokensAndUpdatesStats()
        {
            using var model = CreateModel();
            using var engine = SlmInferenceEngine.FromGpt1(model);
            var output = new int[2];

            var options = new GenerationOptions(
                maxNewTokens: 2,
                maxContextLength: 8,
                sampling: SamplingOptions.Greedy,
                stopOnEndOfTextToken: false);

            var generated = engine.Generate(
                [1, 2, 3],
                output,
                in options);

            var stats = engine.GetLastGenerationStats();

            Assert.Equal(2, generated);
            Assert.Equal(3, stats.PromptTokens);
            Assert.Equal(2, stats.GeneratedTokens);
            Assert.False(stats.UsedKeyValueCache);
            Assert.True(stats.ElapsedNanoseconds > 0);
            Assert.True(stats.AllocatedBytes >= 0);
            Assert.All(output, token => Assert.InRange(token, 0, model.Config.VocabSize - 1));
        }

        [Fact]
        public void GenerateStreaming_InvokesCallbackForGeneratedToken()
        {
            using var model = CreateModel();
            using var engine = SlmInferenceEngine.FromGpt1(model);

            var options = new GenerationOptions(
                maxNewTokens: 1,
                maxContextLength: 8,
                sampling: SamplingOptions.Greedy,
                stopOnEndOfTextToken: false);

            var callbackCount = 0;
            var observedPosition = 0;

            var stats = engine.GenerateStreaming(
                [1, 2],
                in options,
                (tokenId, position, logits) =>
                {
                    Assert.InRange(tokenId, 0, model.Config.VocabSize - 1);
                    Assert.Equal(model.Config.VocabSize, logits.Length);
                    Assert.True(position > 2);

                    observedPosition = position;
                    callbackCount++;

                    return true;
                });

            Assert.Equal(1, callbackCount);
            Assert.Equal(1, stats.GeneratedTokens);
            Assert.Equal(1, engine.GetLastGenerationStats().GeneratedTokens);
            Assert.True(observedPosition > 2);
        }

        [Fact]
        public void GenerateStreaming_CanEmitMultipleTokens()
        {
            using var model = CreateModel();
            using var engine = SlmInferenceEngine.FromGpt1(model);

            var options = new GenerationOptions(
                maxNewTokens: 3,
                maxContextLength: 8,
                sampling: SamplingOptions.Greedy,
                stopOnEndOfTextToken: false);

            var callbackCount = 0;
            var previousPosition = 0;

            var stats = engine.GenerateStreaming(
                [1, 2],
                in options,
                (tokenId, position, logits) =>
                {
                    Assert.InRange(tokenId, 0, model.Config.VocabSize - 1);
                    Assert.Equal(model.Config.VocabSize, logits.Length);
                    Assert.True(position > previousPosition);

                    previousPosition = position;
                    callbackCount++;

                    return callbackCount < 3;
                });

            Assert.InRange(callbackCount, 1, 3);
            Assert.Equal(callbackCount, stats.GeneratedTokens);
            Assert.Equal(stats.GeneratedTokens, engine.GetLastGenerationStats().GeneratedTokens);
        }

        [Fact]
        public void GenerateStreaming_CanStopFromCallback()
        {
            using var model = CreateModel();
            using var engine = SlmInferenceEngine.FromGpt1(model);

            var options = new GenerationOptions(
                maxNewTokens: 5,
                maxContextLength: 8,
                sampling: SamplingOptions.Greedy,
                stopOnEndOfTextToken: false);

            var callbackCount = 0;

            var stats = engine.GenerateStreaming(
                [1, 2],
                in options,
                (_, _, _) =>
                {
                    callbackCount++;
                    return false;
                });

            Assert.Equal(1, callbackCount);
            Assert.Equal(1, stats.GeneratedTokens);
        }

        [Fact]
        public void ResetMetrics_ClearsLastStats()
        {
            using var model = CreateModel();
            using var engine = SlmInferenceEngine.FromGpt1(model);
            var output = new int[1];

            var options = new GenerationOptions(
                maxNewTokens: 1,
                maxContextLength: 8,
                sampling: SamplingOptions.Greedy,
                stopOnEndOfTextToken: false);

            _ = engine.Generate(
                [1, 2, 3],
                output,
                in options);

            Assert.True(engine.GetLastGenerationStats().GeneratedTokens > 0);

            engine.ResetMetrics();

            Assert.Equal(0, engine.GetLastGenerationStats().GeneratedTokens);
            Assert.Equal(0, engine.GetLastGenerationStats().ElapsedNanoseconds);
        }

        [Fact]
        public void Dispose_ThenUse_Throws()
        {
            using var model = CreateModel();
            var engine = SlmInferenceEngine.FromGpt1(model);

            engine.Dispose();

            Assert.Throws<ObjectDisposedException>(() =>
                engine.CreateSession());
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
