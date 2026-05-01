// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using Xunit;

namespace DevOnBike.Overfit.Tests
{
    /// <summary>
    /// Tests for GPT1Model and GPT1Config.
    /// Uses GPT1Config.Small to keep tests fast.
    /// </summary>
    public class GPT1ModelTests
    {
        // Small config: vocab=256, ctx=16, d=64, h=4, L=2, ff=256
        private static GPT1Config SmallConfig => GPT1Config.Small;

        // ─────────────────────────────────────────────────────────────────────
        // Config
        // ─────────────────────────────────────────────────────────────────────

        [Fact]
        public void GPT1Config_ParameterCount_IsReasonable()
        {
            var params1 = GPT1Config.GPT1.ParameterCount;
            // GPT-1 should be ~117M params
            Assert.True(params1 > 100_000_000, $"GPT-1 params too low: {params1:N0}");
            Assert.True(params1 < 140_000_000, $"GPT-1 params too high: {params1:N0}");
        }

        [Fact]
        public void GPT1Config_Small_ToString_ContainsInfo()
        {
            var s = SmallConfig.ToString();
            Assert.Contains("GPT", s);
            Assert.Contains("vocab=256", s);
        }

        // ─────────────────────────────────────────────────────────────────────
        // Forward pass
        // ─────────────────────────────────────────────────────────────────────

        [Fact]
        public void GPT1Model_Forward_OutputShape_IsCorrect()
        {
            using var model = new GPT1Model(SmallConfig);
            model.Eval();

            int[] tokens = [1, 2, 3, 4, 5];
            using var graph  = new ComputationGraph();
            using var logits = model.Forward(graph, tokens, batchSize: 1, seqLen: 5);

            Assert.Equal(1,                    logits.Shape.D0);
            Assert.Equal(5,                    logits.Shape.D1);
            Assert.Equal(SmallConfig.VocabSize, logits.Shape.D2);
        }

        [Fact]
        public void GPT1Model_Forward_NoNaNOrInf()
        {
            using var model = new GPT1Model(SmallConfig);
            model.Eval();

            int[] tokens = [0, 10, 20, 30];
            using var graph  = new ComputationGraph();
            using var logits = model.Forward(graph, tokens, batchSize: 1, seqLen: 4);

            foreach (var v in logits.DataView.AsReadOnlySpan())
            {
                Assert.False(float.IsNaN(v),      "NaN in GPT1 logits");
                Assert.False(float.IsInfinity(v), "Inf in GPT1 logits");
            }
        }

        [Fact]
        public void GPT1Model_SeqLenExceedsContext_Throws()
        {
            using var model = new GPT1Model(SmallConfig);
            model.Eval();

            var tooLong = new int[SmallConfig.ContextLength + 1];
            using var graph = new ComputationGraph();

            Assert.Throws<ArgumentException>(() =>
                model.Forward(graph, tooLong, batchSize: 1, seqLen: tooLong.Length));
        }

        // ─────────────────────────────────────────────────────────────────────
        // Weight tying
        // ─────────────────────────────────────────────────────────────────────

        [Fact]
        public void GPT1Model_WeightTied_LMHeadIsTransposeOfTokenEmb()
        {
            var config = new GPT1Config
            {
                VocabSize     = 16,
                ContextLength = 8,
                DModel        = 8,
                NHeads        = 2,
                NLayers       = 1,
                DFF           = 16,
                TieWeights    = true,
            };

            using var model = new GPT1Model(config);

            // LMHead[c, r] should equal TokenEmbedding.Weight[r, c]
            var tokEmb = model.TokenEmbedding.Weight.DataReadOnlySpan;
            var lmHead = model.LMHead.DataReadOnlySpan;

            for (var r = 0; r < config.VocabSize; r++)
            {
                for (var c = 0; c < config.DModel; c++)
                {
                    var expected = tokEmb[r * config.DModel + c];
                    var actual   = lmHead[c * config.VocabSize + r];
                    Assert.True(MathF.Abs(expected - actual) < 1e-6f,
                        $"Weight tie mismatch at [{r},{c}]: emb={expected}, lmhead={actual}");
                }
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Parameter count
        // ─────────────────────────────────────────────────────────────────────

        [Fact]
        public void GPT1Model_ParameterCount_MatchesConfig()
        {
            using var model = new GPT1Model(SmallConfig);

            var actual   = model.TrainableParameters().Sum(p => (long)p.Shape.Size);
            var expected = SmallConfig.ParameterCount;

            Assert.Equal(expected, actual);
        }

        // ─────────────────────────────────────────────────────────────────────
        // GenerateLogits
        // ─────────────────────────────────────────────────────────────────────

        [Fact]
        public void GPT1Model_GenerateLogits_LengthIsVocabSize()
        {
            using var model  = new GPT1Model(SmallConfig);
            model.Eval();

            var logits = model.GenerateLogits([1, 2, 3]);

            Assert.Equal(SmallConfig.VocabSize, logits.Length);
        }

        [Fact]
        public void GPT1Model_GenerateLogits_NoNaNOrInf()
        {
            using var model  = new GPT1Model(SmallConfig);
            model.Eval();

            var logits = model.GenerateLogits([0, 1, 2, 3, 4]);

            Assert.DoesNotContain(logits, float.IsNaN);
            Assert.DoesNotContain(logits, float.IsInfinity);
        }

        // ─────────────────────────────────────────────────────────────────────
        // Greedy generation
        // ─────────────────────────────────────────────────────────────────────

        [Fact]
        public void GPT1Model_Generate_ReturnsCorrectNumberOfTokens()
        {
            using var model = new GPT1Model(SmallConfig);
            model.Eval();

            var generated = model.Generate([1, 2, 3], maxNewTokens: 5);

            Assert.Equal(5, generated.Length);
            Assert.All(generated, t => Assert.InRange(t, 0, SmallConfig.VocabSize - 1));
        }

        [Fact]
        public void GPT1Model_Generate_IsDeterministic()
        {
            using var model = new GPT1Model(SmallConfig);
            model.Eval();

            int[] prompt = [1, 2, 3];
            var gen1 = model.Generate(prompt, maxNewTokens: 4);
            var gen2 = model.Generate(prompt, maxNewTokens: 4);

            Assert.Equal(gen1, gen2);
        }

        // ─────────────────────────────────────────────────────────────────────
        // Save / Load
        // ─────────────────────────────────────────────────────────────────────

        [Fact]
        public void GPT1Model_SaveLoad_SameLogits()
        {
            using var model1 = new GPT1Model(SmallConfig);
            model1.Eval();

            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms);
            model1.Save(bw);

            ms.Position = 0;
            using var br     = new BinaryReader(ms);
            using var model2 = new GPT1Model(SmallConfig);
            model2.Load(br);
            model2.Eval();

            int[] tokens = [1, 5, 10, 15];

            var logits1 = model1.GenerateLogits(tokens);
            var logits2 = model2.GenerateLogits(tokens);

            for (var i = 0; i < logits1.Length; i++)
            {
                Assert.True(MathF.Abs(logits1[i] - logits2[i]) < 1e-4f,
                    $"Logit mismatch at [{i}]: {logits1[i]:F6} vs {logits2[i]:F6}");
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Train / Eval mode
        // ─────────────────────────────────────────────────────────────────────

        [Fact]
        public void GPT1Model_TrainEval_DoesNotThrow()
        {
            using var model = new GPT1Model(SmallConfig);
            model.Train();
            model.Eval();
            model.Train();
            Assert.True(model.IsTraining);
        }
    }
}
