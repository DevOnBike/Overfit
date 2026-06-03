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

        [Fact]
        public void QLoRA_HeadAndFFN_FrozenQ4KBases_ReduceLoss()
        {
            var config = new GPT1Config
            {
                VocabSize = 64,
                ContextLength = 16,
                DModel = 256,   // %256 → Q4_K head + FFN-up input
                NHeads = 4,
                NLayers = 2,
                DFF = 512,      // %256 → Q4_K FFN-down input
                TieWeights = false,
            };

            using var model = new GPT1Model(config);

            var corpus = new int[256];
            for (var i = 0; i < corpus.Length; i++) { corpus[i] = (i * 7 + (i / 5)) % 64; }

            using var tuner = new Gpt1LoRAFineTuner(
                model, rank: 8,
                LoRATargetModules.LanguageModelHead | LoRATargetModules.FeedForward,
                seed: 7, quantizeBase: true);

            // Both the head AND every block's FFN are now frozen Q4_K with output-level QLoRA hooks.
            Assert.True(model.LMHeadOutputProvider is not null, "LM-head QLoRA hook missing");
            Assert.True(model.Blocks[0].FFN.W1OutputProvider is not null, "FFN-up QLoRA hook missing");
            Assert.True(model.Blocks[0].FFN.W2OutputProvider is not null, "FFN-down QLoRA hook missing");

            var history = tuner.FineTune(corpus, steps: 150, contextLength: 16, learningRate: 0.01f, seed: 99);

            var first = AvgFirst(history, 5);
            var last = AvgLast(history, 5);
            _out.WriteLine($"QLoRA head+FFN loss: {first:F3} -> {last:F3}  ({(1 - last / first) * 100:F0}% drop)");

            Assert.True(last < first, $"loss did not decrease: {first:F3} -> {last:F3}");
            Assert.True(last < 0.85f * first, $"loss drop too small: {first:F3} -> {last:F3}");
        }

        [Fact]
        public void QLoRA_Q8Base_HeadAndFFN_ReduceLoss()
        {
            var config = new GPT1Config
            {
                VocabSize = 64,
                ContextLength = 16,
                DModel = 256,
                NHeads = 4,
                NLayers = 2,
                DFF = 512,
                TieWeights = false,
            };

            using var model = new GPT1Model(config);
            var corpus = new int[256];
            for (var i = 0; i < corpus.Length; i++) { corpus[i] = (i * 7 + (i / 5)) % 64; }

            // Q8 frozen base (higher fidelity than Q4_K; 32-element blocks).
            using var tuner = new Gpt1LoRAFineTuner(
                model, rank: 8,
                LoRATargetModules.LanguageModelHead | LoRATargetModules.FeedForward,
                seed: 7, quantizeBase: true, baseFormat: QLoRABaseFormat.Q8);

            Assert.True(model.LMHeadOutputProvider is not null, "LM-head QLoRA hook missing");
            Assert.True(model.Blocks[0].FFN.W1OutputProvider is not null, "FFN-up QLoRA hook missing");

            var history = tuner.FineTune(corpus, steps: 150, contextLength: 16, learningRate: 0.01f, seed: 99);
            var first = AvgFirst(history, 5);
            var last = AvgLast(history, 5);
            _out.WriteLine($"QLoRA Q8 head+FFN loss: {first:F3} -> {last:F3}  ({(1 - last / first) * 100:F0}% drop)");

            Assert.True(last < 0.85f * first, $"Q8-base loss drop too small: {first:F3} -> {last:F3}");
        }

        [Fact]
        public void QLoRA_Q8Base_AllLinear_HeadFFNAttention_ReduceLoss()
        {
            var config = new GPT1Config
            {
                VocabSize = 64,
                ContextLength = 16,
                DModel = 256,   // %32 (Q8); dHead = 256/4 = 64, %32 ✓
                NHeads = 4,
                NLayers = 2,
                DFF = 512,
                TieWeights = false,
            };

            using var model = new GPT1Model(config);
            var corpus = new int[256];
            for (var i = 0; i < corpus.Length; i++) { corpus[i] = (i * 7 + (i / 5)) % 64; }

            // The WHOLE quantizable base: LM head + FFN + per-head attention Q/K/V/O, all frozen Q8.
            var targets = LoRATargetModules.LanguageModelHead | LoRATargetModules.FeedForward | LoRATargetModules.Attention;
            using var tuner = new Gpt1LoRAFineTuner(
                model, rank: 8, targets, seed: 7, quantizeBase: true, baseFormat: QLoRABaseFormat.Q8);

            Assert.True(model.LMHeadOutputProvider is not null, "LM-head QLoRA hook missing");
            Assert.True(model.Blocks[0].FFN.W1OutputProvider is not null, "FFN QLoRA hook missing");
            Assert.True(model.Blocks[0].Attention.GetQueryOutputProvider(0) is not null, "attention-Q QLoRA hook missing");
            Assert.True(model.Blocks[0].Attention.GetOutputOutputProvider(0) is not null, "attention-O QLoRA hook missing");

            // 300 steps: with the whole base (head+FFN+attention) frozen-quantized, there are many
            // small adapters to coordinate, so attention-inclusive convergence is slower than head+FFN.
            var history = tuner.FineTune(corpus, steps: 300, contextLength: 16, learningRate: 0.01f, seed: 99);
            var first = AvgFirst(history, 5);
            var last = AvgLast(history, 5);
            _out.WriteLine($"QLoRA Q8 all-linear loss: {first:F3} -> {last:F3}  ({(1 - last / first) * 100:F0}% drop)");

            // The point is that the full path TRAINS through the frozen-quant attention/FFN/head;
            // a robust margin (not a tight ratio) avoids flakiness on this tiny, many-adapter task.
            Assert.True(last < first - 0.5f, $"all-linear QLoRA did not train meaningfully: {first:F3} -> {last:F3}");
        }

        [Fact]
        public void QLoRA_FreeQuantizedBase_StillTrains_AndDisposesF32()
        {
            var config = new GPT1Config
            {
                VocabSize = 64,
                ContextLength = 16,
                DModel = 256,
                NHeads = 4,
                NLayers = 2,
                DFF = 512,
                TieWeights = false,
            };
            using var model = new GPT1Model(config);
            var corpus = new int[256];
            for (var i = 0; i < corpus.Length; i++) { corpus[i] = (i * 7 + (i / 5)) % 64; }

            using var tuner = new Gpt1LoRAFineTuner(
                model, rank: 8,
                LoRATargetModules.LanguageModelHead | LoRATargetModules.FeedForward,
                seed: 7, quantizeBase: true, baseFormat: QLoRABaseFormat.Q4K, freeQuantizedBase: true);

            // The F32 weights for the quantized projections were released (model is now QLoRA-only).
            Assert.Throws<ObjectDisposedException>(() => model.LMHead.DataReadOnlySpan.Length);
            Assert.Throws<ObjectDisposedException>(() => model.Blocks[0].FFN.W1.DataReadOnlySpan.Length);

            // ...yet the QLoRA forward (which uses the quantized copies) still trains.
            var history = tuner.FineTune(corpus, steps: 150, contextLength: 16, learningRate: 0.01f, seed: 99);
            var first = AvgFirst(history, 5);
            var last = AvgLast(history, 5);
            _out.WriteLine($"QLoRA free-base loss: {first:F3} -> {last:F3}");
            Assert.True(last < first - 0.5f, $"did not train with freed F32 base: {first:F3} -> {last:F3}");
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
