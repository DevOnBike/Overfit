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
    /// <summary>
    /// API-level parity tests for legacy vs cached SLM generation.
    ///
    /// The lower-level logits parity tests prove that cached logits match
    /// GPT1Model.GenerateLogits(...). These tests verify the public generation
    /// behavior:
    ///
    /// legacy session:
    ///   SlmRuntimeFactory.CreateGpt1(model, SlmRuntimeMode.Legacy)
    ///
    /// cached session:
    ///   SlmRuntimeFactory.CreateGpt1(model, SlmRuntimeMode.Cached)
    ///
    /// With greedy sampling, identical logits should produce identical tokens.
    /// </summary>
    public class CachedGenerationParityTests
    {
        [Fact]
        public void CachedAndLegacy_GreedyGenerate_OneToken_ProduceSameToken()
        {
            using var model = CreateTinyModel(seed: 123);

            AssertGeneratedTokensMatch(
                model,
                promptTokens: [1, 2],
                maxNewTokens: 1);
        }

        [Fact]
        public void CachedAndLegacy_GreedyGenerate_MultipleTokens_ProduceSameTokens()
        {
            using var model = CreateTinyModel(seed: 456);

            AssertGeneratedTokensMatch(
                model,
                promptTokens: [1, 2, 3],
                maxNewTokens: 4);
        }

        [Fact]
        public void CachedAndLegacy_GreedyGenerate_WithDifferentPrompts_ProduceSameTokens()
        {
            using var model = CreateTinyModel(seed: 789);

            var prompts = new[]
            {
                new[] { 0 },
                new[] { 1, 1 },
                new[] { 2, 3, 4 },
                new[] { 7, 6, 5, 4 }
            };

            foreach (var prompt in prompts)
            {
                AssertGeneratedTokensMatch(
                    model,
                    prompt,
                    maxNewTokens: 3);
            }
        }

        [Fact]
        public void CachedAndLegacy_SessionGenerateNextToken_ProduceSameTokensStepByStep()
        {
            using var model = CreateTinyModel(seed: 321);
            using var legacy = SlmRuntimeFactory.CreateGpt1(
                model,
                SlmRuntimeMode.Legacy);

            using var cached = SlmRuntimeFactory.CreateGpt1(
                model,
                SlmRuntimeMode.Cached);

            var prompt = new[] { 1, 2, 3 };
            var sampling = SamplingOptions.Greedy;

            legacy.Session.Reset(prompt);
            cached.Session.Reset(prompt);

            for (var i = 0; i < 4; i++)
            {
                var legacyToken = legacy.Session.GenerateNextToken(in sampling);
                var cachedToken = cached.Session.GenerateNextToken(in sampling);

                Assert.Equal(legacyToken, cachedToken);
            }
        }

        [Fact]
        public void CachedAndLegacy_GenerationOptions_StopAtOutputLength_ProduceSameGeneratedCount()
        {
            using var model = CreateTinyModel(seed: 654);

            var prompt = new[] { 1, 2 };
            var legacyOutput = new int[2];
            var cachedOutput = new int[2];

            var options = new GenerationOptions(
                maxNewTokens: 5,
                maxContextLength: model.Config.ContextLength,
                sampling: SamplingOptions.Greedy,
                stopOnEndOfTextToken: false);

            using var legacy = SlmRuntimeFactory.CreateGpt1(
                model,
                SlmRuntimeMode.Legacy);

            using var cached = SlmRuntimeFactory.CreateGpt1(
                model,
                SlmRuntimeMode.Cached);

            var legacyGenerated = legacy.Generate(
                prompt,
                legacyOutput,
                in options);

            var cachedGenerated = cached.Generate(
                prompt,
                cachedOutput,
                in options);

            Assert.Equal(legacyGenerated, cachedGenerated);
            Assert.Equal(legacyOutput, cachedOutput);
        }

        [Fact]
        public void CachedRuntimeFactory_DefaultMode_MatchesExplicitCachedMode()
        {
            using var model = CreateTinyModel(seed: 987);

            var prompt = new[] { 1, 2 };
            var defaultOutput = new int[3];
            var cachedOutput = new int[3];

            using var defaultRuntime = SlmRuntimeFactory.CreateGpt1(model);
            using var cachedRuntime = SlmRuntimeFactory.CreateGpt1(
                model,
                SlmRuntimeMode.Cached);

            var defaultGenerated = defaultRuntime.GenerateGreedy(
                prompt,
                defaultOutput,
                maxNewTokens: 3);

            var cachedGenerated = cachedRuntime.GenerateGreedy(
                prompt,
                cachedOutput,
                maxNewTokens: 3);

            Assert.Equal(defaultGenerated, cachedGenerated);
            Assert.Equal(defaultOutput, cachedOutput);
        }

        private static void AssertGeneratedTokensMatch(
            GPT1Model model,
            int[] promptTokens,
            int maxNewTokens)
        {
            var legacyOutput = new int[maxNewTokens];
            var cachedOutput = new int[maxNewTokens];

            using var legacy = SlmRuntimeFactory.CreateGpt1(
                model,
                SlmRuntimeMode.Legacy);

            using var cached = SlmRuntimeFactory.CreateGpt1(
                model,
                SlmRuntimeMode.Cached);

            var legacyGenerated = legacy.GenerateGreedy(
                promptTokens,
                legacyOutput,
                maxNewTokens);

            var cachedGenerated = cached.GenerateGreedy(
                promptTokens,
                cachedOutput,
                maxNewTokens);

            Assert.Equal(legacyGenerated, cachedGenerated);
            Assert.Equal(legacyOutput, cachedOutput);
        }

        private static GPT1Model CreateTinyModel(int seed)
        {
            var config = new GPT1Config
            {
                VocabSize = 8,
                ContextLength = 8,
                DModel = 8,
                NHeads = 2,
                NLayers = 2,
                DFF = 16,
                TieWeights = true,
                PreLayerNorm = true
            };

            var model = new GPT1Model(config);
            model.Eval();

            InitializeDeterministicWeights(
                model,
                seed);

            return model;
        }

        private static void InitializeDeterministicWeights(
            GPT1Model model,
            int seed)
        {
            var rng = new Random(seed);

            Fill(model.TokenEmbedding.Weight.DataSpan, rng, scale: 0.02f);
            Fill(model.PositionEmbedding.Weight.DataSpan, rng, scale: 0.02f);

            foreach (var block in model.Blocks)
            {
                Fill(block.Norm1.Gamma.DataSpan, value: 1f);
                Fill(block.Norm1.Beta.DataSpan, value: 0f);

                for (var head = 0; head < model.Config.NHeads; head++)
                {
                    Fill(block.Attention.WqHeads[head].DataSpan, rng, scale: 0.02f);
                    Fill(block.Attention.WkHeads[head].DataSpan, rng, scale: 0.02f);
                    Fill(block.Attention.WvHeads[head].DataSpan, rng, scale: 0.02f);
                    Fill(block.Attention.WoHeads[head].DataSpan, rng, scale: 0.02f);
                }

                Fill(block.Attention.Bo.DataSpan, value: 0f);

                Fill(block.Norm2.Gamma.DataSpan, value: 1f);
                Fill(block.Norm2.Beta.DataSpan, value: 0f);

                Fill(block.FFN.W1.DataSpan, rng, scale: 0.02f);
                Fill(block.FFN.B1.DataSpan, value: 0f);
                Fill(block.FFN.W2.DataSpan, rng, scale: 0.02f);
                Fill(block.FFN.B2.DataSpan, value: 0f);
            }

            Fill(model.FinalNorm.Gamma.DataSpan, value: 1f);
            Fill(model.FinalNorm.Beta.DataSpan, value: 0f);

            if (!model.Config.TieWeights)
            {
                Fill(model.LMHead.DataSpan, rng, scale: 0.02f);
            }
        }

        private static void Fill(
            Span<float> values,
            Random rng,
            float scale)
        {
            for (var i = 0; i < values.Length; i++)
            {
                values[i] = ((float)rng.NextDouble() * 2f - 1f) * scale;
            }
        }

        private static void Fill(
            Span<float> values,
            float value)
        {
            for (var i = 0; i < values.Length; i++)
            {
                values[i] = value;
            }
        }
    }
}
