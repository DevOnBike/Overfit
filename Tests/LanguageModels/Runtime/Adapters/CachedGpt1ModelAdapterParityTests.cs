// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit;

namespace DevOnBike.Overfit.Tests.LanguageModels
{
    /// <summary>
    /// Parity tests between the legacy full-context GPT1Model path and the
    /// cached one-token adapter path.
    ///
    /// These tests intentionally compare:
    ///
    /// legacy:
    ///   model.GenerateLogits(prefixTokens)
    ///
    /// cached:
    ///   adapter.Reset()
    ///   adapter.DecodeNextToken(prefixTokens[0])
    ///   adapter.DecodeNextToken(prefixTokens[1])
    ///   ...
    ///   adapter.GetLastLogits(...)
    ///
    /// If these pass, the KV-cache runtime is numerically aligned with the old
    /// implementation for the tested tiny shapes.
    /// </summary>
    public class CachedGpt1ModelAdapterParityTests
    {
        [Fact]
        public void CachedAdapter_LogitsMatchLegacyGenerateLogits_ForSingleTokenPrefix()
        {
            using var model = CreateTinyModel(seed: 123);
            using var adapter = new CachedGpt1ModelAdapter(model);

            AssertLogitsMatch(
                model,
                adapter,
                prefixTokens: [1],
                tolerance: 1e-3f);
        }

        [Fact]
        public void CachedAdapter_LogitsMatchLegacyGenerateLogits_ForGrowingPrefix()
        {
            using var model = CreateTinyModel(seed: 456);
            using var adapter = new CachedGpt1ModelAdapter(model);

            var prompt = new[] { 1, 2, 3, 4 };

            for (var length = 1; length <= prompt.Length; length++)
            {
                AssertLogitsMatch(
                    model,
                    adapter,
                    prefixTokens: prompt.AsSpan(0, length).ToArray(),
                    tolerance: 1e-3f);
            }
        }

        [Fact]
        public void CachedAdapter_LogitsMatchLegacyGenerateLogits_ForMultiplePrompts()
        {
            using var model = CreateTinyModel(seed: 789);
            using var adapter = new CachedGpt1ModelAdapter(model);

            var prompts = new[]
            {
                new[] { 0 },
                new[] { 2, 5 },
                new[] { 7, 1, 3 },
                new[] { 4, 4, 4, 4 }
            };

            foreach (var prompt in prompts)
            {
                AssertLogitsMatch(
                    model,
                    adapter,
                    prompt,
                    tolerance: 1e-3f);
            }
        }

        [Fact]
        public void CachedAdapter_ResetMakesRepeatedParityDeterministic()
        {
            using var model = CreateTinyModel(seed: 321);
            using var adapter = new CachedGpt1ModelAdapter(model);

            var prompt = new[] { 3, 1, 4 };

            var first = DecodeCachedLogits(
                adapter,
                prompt,
                model.Config.VocabSize);

            var second = DecodeCachedLogits(
                adapter,
                prompt,
                model.Config.VocabSize);

            AssertEqual(
                first,
                second,
                tolerance: 0f);
        }

        [Fact]
        public void CachedAdapter_ContinuationLogitsMatchLegacyAfterAppendingGeneratedToken()
        {
            using var model = CreateTinyModel(seed: 654);
            using var adapter = new CachedGpt1ModelAdapter(model);

            var prompt = new[] { 1, 2, 3 };
            var generatedToken = 4;

            var legacyPrefix = new[] { 1, 2, 3, generatedToken };
            var legacyLogits = model.GenerateLogits(legacyPrefix);

            adapter.Reset();

            foreach (var token in prompt)
            {
                adapter.DecodeNextToken(
                    token,
                    new float[model.Config.VocabSize]);
            }

            var cachedLogits = new float[model.Config.VocabSize];

            adapter.DecodeNextToken(
                generatedToken,
                cachedLogits);

            AssertEqual(
                legacyLogits,
                cachedLogits,
                tolerance: 1e-3f);
        }

        private static void AssertLogitsMatch(
            GPT1Model model,
            CachedGpt1ModelAdapter adapter,
            int[] prefixTokens,
            float tolerance)
        {
            var legacyLogits = model.GenerateLogits(prefixTokens);
            var cachedLogits = DecodeCachedLogits(
                adapter,
                prefixTokens,
                model.Config.VocabSize);

            AssertEqual(
                legacyLogits,
                cachedLogits,
                tolerance);
        }

        private static float[] DecodeCachedLogits(
            CachedGpt1ModelAdapter adapter,
            int[] prefixTokens,
            int vocabSize)
        {
            adapter.Reset();

            var logits = new float[vocabSize];

            foreach (var token in prefixTokens)
            {
                adapter.DecodeNextToken(
                    token,
                    logits);
            }

            return logits;
        }

        private static void AssertEqual(
            IReadOnlyList<float> expected,
            IReadOnlyList<float> actual,
            float tolerance)
        {
            Assert.Equal(expected.Count, actual.Count);

            for (var i = 0; i < expected.Count; i++)
            {
                var diff = MathF.Abs(expected[i] - actual[i]);

                Assert.True(
                    diff <= tolerance,
                    $"Logit mismatch at index {i}: expected={expected[i]}, actual={actual[i]}, diff={diff}, tolerance={tolerance}.");
            }
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
