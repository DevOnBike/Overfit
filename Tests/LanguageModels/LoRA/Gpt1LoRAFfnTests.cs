// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.LoRA;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.LoRA
{
    /// <summary>
    /// Stage 2 of the GPT1 LoRA path: LoRA adapters on the per-block feed-forward
    /// matrices (W1 up-projection, W2 down-projection), selected via
    /// <see cref="LoRATargetModules"/>.
    ///
    /// Proves a multi-entry FFN adapter trains, saves, merges into every block,
    /// is observed by KV-cached decode (the runtime GptAnomalyDetector uses), and
    /// that Disable() restores the base model — and that FFN and LM-head adapters
    /// compose when targeted together.
    /// </summary>
    public sealed class Gpt1LoRAFfnTests
    {
        private readonly ITestOutputHelper _output;

        public Gpt1LoRAFfnTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void FfnLoRA_MergesIntoEveryBlock_AndIsReversible()
        {
            using var model = NewModel();
            var corpus = PeriodicCorpus();
            var window = corpus[0..20];
            var path = TempPath();

            try
            {
                using (var tuner = new Gpt1LoRAFineTuner(model, rank: 8, targets: LoRATargetModules.FeedForward))
                {
                    // 2 blocks x {W1, W2} = 4 adapter pairs.
                    Assert.Equal(4, tuner.AdapterCount);

                    var history = tuner.FineTune(corpus, steps: 400, contextLength: 16, learningRate: 1e-2f);
                    _output.WriteLine($"FFN LoRA fine-tune loss: {history[0]:F4} -> {history[^1]:F4}");

                    Assert.True(history[^1] < history[0], "FFN LoRA training loss did not decrease.");

                    tuner.Save(path);
                }

                var baseLoss = MeasureCachedLoss(model, window);

                using var merge = Gpt1LoRAMergeAdapter.Load(model, path);
                Assert.Equal(4, merge.TargetCount);

                merge.Enable();
                var mergedLoss = MeasureCachedLoss(model, window);

                merge.Disable();
                var restoredLoss = MeasureCachedLoss(model, window);

                _output.WriteLine($"base loss (cached):     {baseLoss:F4}");
                _output.WriteLine($"merged loss (cached):   {mergedLoss:F4}");
                _output.WriteLine($"restored loss (cached): {restoredLoss:F4}");

                Assert.False(float.IsNaN(mergedLoss) || float.IsInfinity(mergedLoss), "Merged loss is not finite.");

                // FFN LoRA reaches cached decode and improves next-token loss.
                Assert.True(
                    mergedLoss < baseLoss,
                    $"FFN LoRA merge did not improve cached decode: base={baseLoss:F4}, merged={mergedLoss:F4}.");

                // Disable() un-merges every block — the base model is restored.
                Assert.True(
                    MathF.Abs(restoredLoss - baseLoss) < 5e-2f,
                    $"Disable() did not restore the base model: base={baseLoss:F4}, restored={restoredLoss:F4}.");
            }
            finally
            {
                if (File.Exists(path))
                {
                    File.Delete(path);
                }
            }
        }

        [Fact]
        public void HeadAndFfnLoRA_ComposeWhenTargetedTogether()
        {
            using var model = NewModel();
            var corpus = PeriodicCorpus();
            var window = corpus[0..20];
            var path = TempPath();

            try
            {
                const LoRATargetModules targets =
                    LoRATargetModules.LanguageModelHead | LoRATargetModules.FeedForward;

                using (var tuner = new Gpt1LoRAFineTuner(model, rank: 8, targets: targets))
                {
                    // LM head + 2 blocks x {W1, W2} = 5 adapter pairs.
                    Assert.Equal(5, tuner.AdapterCount);

                    tuner.FineTune(corpus, steps: 300, contextLength: 16, learningRate: 1e-2f);
                    tuner.Save(path);
                }

                var baseLoss = MeasureCachedLoss(model, window);

                using var merge = Gpt1LoRAMergeAdapter.Load(model, path);
                Assert.Equal(5, merge.TargetCount);

                merge.Enable();
                var mergedLoss = MeasureCachedLoss(model, window);

                merge.Disable();
                var restoredLoss = MeasureCachedLoss(model, window);

                _output.WriteLine($"base loss (cached):     {baseLoss:F4}");
                _output.WriteLine($"merged loss (cached):   {mergedLoss:F4}");
                _output.WriteLine($"restored loss (cached): {restoredLoss:F4}");

                // Head + FFN adapters compose — at least as strong as head-only Stage 1.
                Assert.True(
                    mergedLoss < baseLoss * 0.7f,
                    $"Combined head+FFN LoRA did not improve cached decode: base={baseLoss:F4}, merged={mergedLoss:F4}.");

                Assert.True(
                    MathF.Abs(restoredLoss - baseLoss) < 5e-2f,
                    $"Disable() did not restore the base model: base={baseLoss:F4}, restored={restoredLoss:F4}.");
            }
            finally
            {
                if (File.Exists(path))
                {
                    File.Delete(path);
                }
            }
        }

        private static GPT1Model NewModel() => new(new GPT1Config
        {
            VocabSize = 16,
            ContextLength = 32,
            DModel = 32,
            NHeads = 2,
            NLayers = 2,
            DFF = 64,
            TieWeights = false,
            PreLayerNorm = true,
        });

        private static int[] PeriodicCorpus()
        {
            var corpus = new int[256];
            for (var i = 0; i < corpus.Length; i++)
            {
                corpus[i] = i % 4;
            }

            return corpus;
        }

        private static string TempPath()
            => Path.Combine(Path.GetTempPath(), $"overfit_ffn_lora_{Guid.NewGuid():N}.bin");

        /// <summary>Mean next-token cross-entropy of a window via a fresh KV-cached decode.</summary>
        private static float MeasureCachedLoss(GPT1Model model, int[] window)
        {
            using var cached = new CachedGpt1ModelAdapter(model);
            cached.Reset();

            var logits = new float[cached.VocabSize];
            var total = 0f;

            for (var i = 0; i < window.Length - 1; i++)
            {
                cached.DecodeNextToken(window[i], logits);
                total += NegLogProb(logits, window[i + 1]);
            }

            return total / (window.Length - 1);
        }

        private static float NegLogProb(float[] logits, int target)
        {
            var max = logits[0];
            for (var i = 1; i < logits.Length; i++)
            {
                if (logits[i] > max)
                {
                    max = logits[i];
                }
            }

            var sumExp = 0f;
            for (var i = 0; i < logits.Length; i++)
            {
                sumExp += MathF.Exp(logits[i] - max);
            }

            return -(logits[target] - max - MathF.Log(sumExp));
        }
    }
}
