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
    /// Rehearsal-lite LoRA fine-tuning: mixing a fraction of the base regime's data
    /// into a per-task adapter's training reduces forgetting of that base regime,
    /// at a small cost to peak specialisation. The cheapest form of replay — one
    /// mixing fraction, no extra machinery (cf. rehearsal-free continual-learning
    /// research, arXiv:2406.09384 / NeurIPS 2024 continual-pretraining guide).
    /// </summary>
    public sealed class Gpt1LoRARehearsalTests
    {
        private readonly ITestOutputHelper _output;

        public Gpt1LoRARehearsalTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void Rehearsal_ReducesForgettingOfBaseRegime()
        {
            using var model = NewModel();

            // Two disjoint token regimes. "New task" = tokens 0..3; "base" = 8..11.
            // An adapter trained only on the new task should predict the base poorly;
            // mixing base windows back in should preserve it.
            var newTask = Periodic(baseToken: 0);
            var baseRegime = Periodic(baseToken: 8);

            const int ctx = 16;
            const int steps = 400;

            float forgottenBaseLoss;
            using (var tuner = new Gpt1LoRAFineTuner(model, rank: 8))
            {
                tuner.FineTune(newTask, steps, ctx, learningRate: 1e-2f);
                forgottenBaseLoss = tuner.EvaluateLoss(baseRegime, ctx, start: 0);
            }
            // adapter detached on dispose — model back to pure base.

            float rehearsedBaseLoss;
            float rehearsedNewLoss;
            using (var tuner = new Gpt1LoRAFineTuner(model, rank: 8))
            {
                // Same new-task training, but 30 % of windows drawn from the base regime.
                tuner.FineTune(
                    newTask, steps, ctx, learningRate: 1e-2f,
                    rehearsalCorpus: baseRegime, rehearsalFraction: 0.3f);
                rehearsedBaseLoss = tuner.EvaluateLoss(baseRegime, ctx, start: 0);
                rehearsedNewLoss = tuner.EvaluateLoss(newTask, ctx, start: 0);
            }

            _output.WriteLine($"base-regime loss — no rehearsal: {forgottenBaseLoss:F4}");
            _output.WriteLine($"base-regime loss — rehearsal:    {rehearsedBaseLoss:F4}");
            _output.WriteLine($"new-task loss   — rehearsal:    {rehearsedNewLoss:F4}");

            // Rehearsal preserves the base regime markedly better.
            Assert.True(
                rehearsedBaseLoss < forgottenBaseLoss,
                $"Rehearsal did not reduce base-regime forgetting: " +
                $"no-rehearsal={forgottenBaseLoss:F4}, rehearsal={rehearsedBaseLoss:F4}.");

            // …while still learning the new task (sanity: it didn't just ignore it).
            Assert.True(
                rehearsedNewLoss < forgottenBaseLoss,
                $"Rehearsal adapter failed to learn the new task: new-task loss {rehearsedNewLoss:F4}.");
        }

        // A period-4 sequence over four consecutive tokens starting at baseToken.
        private static int[] Periodic(int baseToken)
        {
            var corpus = new int[256];
            for (var i = 0; i < corpus.Length; i++)
            {
                corpus[i] = baseToken + (i % 4);
            }
            return corpus;
        }

        private static GPT1Model NewModel() => new(new GPT1Config
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
    }
}
