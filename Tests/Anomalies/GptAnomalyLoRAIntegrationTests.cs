// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Gpt;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.LoRA;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// End-to-end integration test for the K8s anomaly-detection LoRA path —
    /// the wiring that previously existed only as separately-tested building
    /// blocks (LoRA fine-tuner, merge adapter, anomaly detector).
    ///
    /// Chain under test:
    ///   MetricSnapshot[] --MetricTokenizer.EncodeSequence--> int[] corpus
    ///     --Gpt1LoRAFineTuner.FineTune--> LM-head LoRA on the production regime
    ///     --Gpt1LoRAFineTuner.Save--> .bin
    ///     --Gpt1LoRAMergeAdapter.Load + Enable--> delta merged into GPT1Model.LMHead
    ///     --SlmRuntimeFactory.CreateGpt1--> SlmRuntimeHandle
    ///     --GptAnomalyDetector.Score--> anomaly score
    ///
    /// The integration claim: a LoRA adapter trained on a production metric
    /// regime, once merged, actually reaches the anomaly detector and makes that
    /// same regime score as <i>less</i> anomalous — and Disable() restores the
    /// base behaviour. Each measurement builds a fresh detector over the current
    /// model state, so the test holds whether the runtime is zero-copy or snapshots.
    /// </summary>
    public sealed class GptAnomalyLoRAIntegrationTests
    {
        private readonly ITestOutputHelper _output;

        public GptAnomalyLoRAIntegrationTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        [Trait("Category", "Integration")]
        public void LoRA_FineTunedOnProductionRegime_LowersAnomalyScore_ThroughDetector()
        {
            const int contextSnapshots = 6;
            var tps = MetricTokenizer.TokensPerSnapshot;

            using var model = new GPT1Model(new GPT1Config
            {
                VocabSize = MetricTokenizer.VocabSize,
                ContextLength = 16 * tps,
                DModel = 32,
                NHeads = 2,
                NLayers = 1,
                DFF = 64,
                TieWeights = false,   // LoRA targets the untied LM head
                PreLayerNorm = true,
            });

            // A stable "production normal" regime — 48 snapshots of the same pod.
            // Tokenised, this is a periodic sequence the LM-head LoRA can learn.
            var regime = new MetricSnapshot[48];
            for (var i = 0; i < regime.Length; i++)
            {
                regime[i] = MakeNormalSnapshot("payments-api");
            }

            var corpus = new MetricTokenizer().EncodeSequence(regime);

            var loraPath = Path.Combine(
                Path.GetTempPath(), $"overfit_anomaly_lora_{Guid.NewGuid():N}.bin");

            try
            {
                // ── 1. Fine-tune an LM-head LoRA adapter on the production regime ──
                using (var tuner = new Gpt1LoRAFineTuner(model, rank: 16))
                {
                    // The detector primes a (contextSnapshots-1)-snapshot context and
                    // decodes one more snapshot, exercising positions up to
                    // contextSnapshots*tps-1. The LoRA head must be fine-tuned over that
                    // exact position range — a shorter window leaves the detector's
                    // positions untrained and the merge would mis-fire.
                    var history = tuner.FineTune(
                        corpus, steps: 300, contextLength: contextSnapshots * tps, learningRate: 1e-2f);

                    _output.WriteLine($"LoRA fine-tune loss: {history[0]:F4} -> {history[^1]:F4}");

                    tuner.Save(loraPath);
                }
                // tuner disposed -> LMHeadWeightProvider detached; model is plain again.

                // ── 2. Anomaly score with the base (un-adapted) model ──
                var baseScore = MeasureRegime(model, regime, contextSnapshots);

                // ── 3. Merge the LoRA adapter, then score again ──
                using var merge = Gpt1LoRAMergeAdapter.Load(model, loraPath);
                merge.Enable();
                var mergedScore = MeasureRegime(model, regime, contextSnapshots);

                // ── 4. Disable — the merge must be reversible end-to-end ──
                merge.Disable();
                var restoredScore = MeasureRegime(model, regime, contextSnapshots);

                _output.WriteLine($"anomaly score - base:     {baseScore:F4}");
                _output.WriteLine($"anomaly score - LoRA:     {mergedScore:F4}");
                _output.WriteLine($"anomaly score - restored: {restoredScore:F4}");

                Assert.False(
                    float.IsNaN(mergedScore) || float.IsInfinity(mergedScore),
                    "Merged anomaly score is not finite.");

                // The integration claim: the LoRA adapter trained on the production
                // regime reaches the detector and makes that regime less anomalous.
                Assert.True(
                    mergedScore < baseScore,
                    "LoRA merge did not lower the anomaly score through the detector: "
                    + $"base={baseScore:F4}, merged={mergedScore:F4}.");

                // Disable() is reversible — the detector is back to base behaviour.
                Assert.True(
                    MathF.Abs(restoredScore - baseScore) < 5e-2f,
                    "Disable() did not restore base detector behaviour: "
                    + $"base={baseScore:F4}, restored={restoredScore:F4}.");
            }
            finally
            {
                if (File.Exists(loraPath))
                {
                    File.Delete(loraPath);
                }
            }
        }

        /// <summary>
        /// Builds a fresh detector over the model's current weights, feeds the
        /// steady regime through it, and returns the last post-warmup score.
        /// </summary>
        private static float MeasureRegime(
            GPT1Model model, MetricSnapshot[] regime, int contextSnapshots)
        {
            using var handle = SlmRuntimeFactory.CreateGpt1(model);
            using var detector = new GptAnomalyDetector(handle, contextSnapshots);

            var score = 0f;
            var feedCount = Math.Min(regime.Length, contextSnapshots * 2);

            for (var i = 0; i < feedCount; i++)
            {
                var result = detector.Score(regime[i]);
                if (!result.IsWarmup)
                {
                    score = result.Score;
                }
            }

            return score;
        }

        private static MetricSnapshot MakeNormalSnapshot(string pod) => new()
        {
            Timestamp = DateTime.UtcNow,
            PodName = pod,
            CpuUsageRatio = 0.22f,
            CpuThrottleRatio = 0.02f,
            MemoryWorkingSetBytes = 360_000_000f,
            OomEventsRate = 0f,
            LatencyP50Ms = 13f,
            LatencyP95Ms = 38f,
            LatencyP99Ms = 78f,
            RequestsPerSecond = 270f,
            ErrorRate = 0.003f,
            GcGen2HeapBytes = 52_000_000f,
            GcPauseRatio = 0.004f,
            ThreadPoolQueueLength = 9f,
        };
    }
}
