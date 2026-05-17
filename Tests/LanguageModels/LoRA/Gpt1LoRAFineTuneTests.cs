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
    /// Stage 1 of the GPT1 LoRA fine-tune design: LM-head-only LoRA.
    ///
    /// Proves the path end-to-end: the LoRA factors fit a learnable corpus
    /// (next-token loss drops), the frozen base model is left bit-identical,
    /// and the trained adapter round-trips through the .bin format.
    /// </summary>
    public sealed class Gpt1LoRAFineTuneTests
    {
        private readonly ITestOutputHelper _output;

        public Gpt1LoRAFineTuneTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void LMHeadLoRA_FineTune_ReducesLoss_KeepsBaseFrozen_RoundTripsAdapter()
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

            // Period-4 corpus — next token is fully determined by position phase,
            // so an LM-head adapter alone has enough signal to fit it.
            var corpus = new int[256];
            for (var i = 0; i < corpus.Length; i++)
            {
                corpus[i] = i % 4;
            }

            const int rank = 8;
            const int contextLength = 16;

            // Snapshot every base parameter to prove the base is never updated.
            var baseSnapshot = new List<float[]>();
            foreach (var p in model.TrainableParameters())
            {
                baseSnapshot.Add(p.DataReadOnlySpan.ToArray());
            }

            float finalLoss;
            var tempPath = Path.Combine(Path.GetTempPath(), $"overfit_lmhead_lora_{Guid.NewGuid():N}.bin");

            try
            {
                using (var lora = new Gpt1LoRAFineTuner(model, rank))
                {
                    // B is zero-initialised, so before training the adapter is the
                    // identity — this measures the frozen base loss.
                    var initialLoss = lora.EvaluateLoss(corpus, contextLength, start: 0);

                    var history = lora.FineTune(corpus, steps: 300, contextLength, learningRate: 1e-2f);
                    finalLoss = lora.EvaluateLoss(corpus, contextLength, start: 0);

                    _output.WriteLine($"LoRA params:  {lora.TrainableParameterCount}");
                    _output.WriteLine($"initial loss: {initialLoss:F4}");
                    _output.WriteLine($"step 1 loss:  {history[0]:F4}");
                    _output.WriteLine($"final loss:   {finalLoss:F4}");

                    Assert.False(float.IsNaN(finalLoss) || float.IsInfinity(finalLoss), "Final loss is not finite.");
                    Assert.True(
                        finalLoss < initialLoss * 0.85f,
                        $"LoRA fine-tune did not reduce loss enough: {initialLoss:F4} -> {finalLoss:F4}.");

                    lora.Save(tempPath);
                }

                // Base model must be bit-identical after fine-tuning.
                var idx = 0;
                foreach (var p in model.TrainableParameters())
                {
                    Assert.Equal(baseSnapshot[idx], p.DataReadOnlySpan.ToArray());
                    idx++;
                }

                // A fresh fine-tuner starts as identity; Load must restore the
                // trained factors so EvaluateLoss matches the trained run.
                using var reloaded = new Gpt1LoRAFineTuner(model, rank);
                reloaded.Load(tempPath);
                var reloadedLoss = reloaded.EvaluateLoss(corpus, contextLength, start: 0);

                _output.WriteLine($"reloaded loss: {reloadedLoss:F4}");
                Assert.Equal(finalLoss, reloadedLoss, precision: 3);
            }
            finally
            {
                if (File.Exists(tempPath))
                {
                    File.Delete(tempPath);
                }
            }
        }
    }
}
