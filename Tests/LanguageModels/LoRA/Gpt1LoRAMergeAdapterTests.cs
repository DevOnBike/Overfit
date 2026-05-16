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
    /// Second half of the GPT1 LoRA path: a LoRA adapter trained by
    /// <see cref="Gpt1LoRAFineTuner"/> is applied at inference time via
    /// <see cref="Gpt1LoRAMergeAdapter"/> and observed through the KV-cached
    /// decode path (<see cref="CachedGpt1ModelAdapter"/>) — i.e. exactly the
    /// runtime GptAnomalyDetector uses.
    ///
    /// Proves: Enable() merges the trained delta into the LM head, cached decode
    /// sees it (zero-copy StackWeights ref), and Disable() exactly restores the
    /// base model.
    /// </summary>
    public sealed class Gpt1LoRAMergeAdapterTests
    {
        private readonly ITestOutputHelper _output;

        public Gpt1LoRAMergeAdapterTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void MergedLoRA_IsVisibleToCachedDecode_AndReversible()
        {
            using var model = new GPT1Model(new GPT1Config
            {
                VocabSize = 16,
                ContextLength = 32,
                DModel = 32,
                NHeads = 2,
                NLayers = 1,
                DFF = 64,
                TieWeights = false,
                PreLayerNorm = true,
            });

            var corpus = new int[256];
            for (var i = 0; i < corpus.Length; i++)
            {
                corpus[i] = i % 4;
            }

            var window = corpus[0..20];
            var tempPath = Path.Combine(Path.GetTempPath(), $"overfit_merge_lora_{Guid.NewGuid():N}.bin");

            try
            {
                // --- Train phase: produce a LoRA adapter, then detach the fine-tuner ---
                using (var lora = new Gpt1LoRAFineTuner(model, rank: 8))
                {
                    lora.FineTune(corpus, steps: 300, contextLength: 16, learningRate: 1e-2f);
                    lora.Save(tempPath);
                }

                // --- Inference phase: KV-cached decode (the GptAnomalyDetector runtime) ---
                using var cached = new CachedGpt1ModelAdapter(model);

                var baseLoss = CachedLoss(cached, window);

                using var merge = Gpt1LoRAMergeAdapter.Load(model, tempPath);
                merge.Enable();
                Assert.True(merge.IsEnabled);

                var mergedLoss = CachedLoss(cached, window);

                merge.Disable();
                Assert.False(merge.IsEnabled);

                var restoredLoss = CachedLoss(cached, window);

                _output.WriteLine($"base loss (cached):     {baseLoss:F4}");
                _output.WriteLine($"merged loss (cached):   {mergedLoss:F4}");
                _output.WriteLine($"restored loss (cached): {restoredLoss:F4}");

                Assert.False(float.IsNaN(mergedLoss) || float.IsInfinity(mergedLoss), "Merged loss is not finite.");

                // Enable(): the trained adapter is visible to cached decode.
                Assert.True(
                    mergedLoss < baseLoss * 0.7f,
                    $"Merged LoRA did not improve cached decode: base={baseLoss:F4}, merged={mergedLoss:F4}.");

                // Disable(): the merge is exactly reversible — base is restored.
                Assert.True(
                    MathF.Abs(restoredLoss - baseLoss) < 1e-2f,
                    $"Disable() did not restore the base model: base={baseLoss:F4}, restored={restoredLoss:F4}.");
            }
            finally
            {
                if (File.Exists(tempPath))
                {
                    File.Delete(tempPath);
                }
            }
        }

        /// <summary>Mean next-token cross-entropy of a window via KV-cached decode.</summary>
        private static float CachedLoss(CachedGpt1ModelAdapter cached, int[] window)
        {
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
