// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.LoRA;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.LoRA
{
    /// <summary>
    /// End-to-end QLoRA on a real <see cref="GPT1Model"/>: the LM head is frozen as Q4_K
    /// (<c>quantizeBase: true</c>) and only the LoRA adapter trains, via the new
    /// <c>FrozenQuantizedLinear</c> output-level head hook. Proves the full path —
    /// quantize → frozen-quant forward/backward → adapter learns — works in the fine-tuner.
    /// </summary>
    public sealed class Gpt1QLoRATests
    {
        private readonly ITestOutputHelper _out;
        public Gpt1QLoRATests(ITestOutputHelper output) => _out = output;

        [Fact]
        public void QLoRA_LmHead_FrozenQ4KBase_AdapterReducesLoss()
        {
            var config = new GPT1Config
            {
                VocabSize = 64,
                ContextLength = 16,
                DModel = 256,   // multiple of 256 → Q4_K head
                NHeads = 4,
                NLayers = 2,
                DFF = 512,
                TieWeights = false, // LM-head LoRA needs an untied head
            };

            using var model = new GPT1Model(config);

            // Repeating, learnable corpus (period 5) so next-token loss can drop.
            var corpus = new int[256];
            for (var i = 0; i < corpus.Length; i++) { corpus[i] = (i * 7 + (i / 5)) % 64; }

            using var tuner = new Gpt1LoRAFineTuner(
                model, rank: 8, LoRATargetModules.LanguageModelHead, seed: 7, quantizeBase: true);

            Assert.True(model.LMHeadOutputProvider is not null, "QLoRA output hook not attached");

            var history = tuner.FineTune(corpus, steps: 150, contextLength: 16, learningRate: 0.01f, seed: 99);

            var first = AvgFirst(history, 5);
            var last = AvgLast(history, 5);
            _out.WriteLine($"QLoRA LM-head loss: {first:F3} -> {last:F3}  ({(1 - last / first) * 100:F0}% drop)");

            Assert.True(last < first, $"loss did not decrease: {first:F3} -> {last:F3}");
            Assert.True(last < 0.85f * first, $"loss drop too small for a trainable corpus: {first:F3} -> {last:F3}");
        }

        private static float AvgFirst(IReadOnlyList<float> h, int n)
        {
            float s = 0; var c = Math.Min(n, h.Count);
            for (var i = 0; i < c; i++) { s += h[i]; }
            return s / c;
        }

        private static float AvgLast(IReadOnlyList<float> h, int n)
        {
            float s = 0; var c = Math.Min(n, h.Count);
            for (var i = 0; i < c; i++) { s += h[h.Count - 1 - i]; }
            return s / c;
        }
    }
}
