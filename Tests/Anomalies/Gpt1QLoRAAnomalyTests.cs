// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Gpt;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.LoRA;
using DevOnBike.Overfit.Maths;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// Validates QLoRA on the REAL anomaly task (not a synthetic corpus): a per-deployment LoRA
    /// adapter trained over a frozen <b>Q4_K</b> base on a pod's benign metric regime flattens that
    /// regime (loss drops) WITHOUT blinding the detector to an injected incident (anomaly loss stays
    /// high). Train and eval both go through the quantized base + adapter (the QLoRA output hooks),
    /// so it is self-consistent — unlike merging the adapter back onto an F32 base.
    /// </summary>
    public sealed class Gpt1QLoRAAnomalyTests
    {
        private readonly ITestOutputHelper _out;
        public Gpt1QLoRAAnomalyTests(ITestOutputHelper output) => _out = output;

        [Fact]
        public void QLoRA_AdaptsBenignRegime_OnFrozenQ4KBase_StillFlagsIncident()
        {
            var tps = MetricTokenizer.TokensPerSnapshot;
            const int contextSnapshots = 6;
            var contextLength = contextSnapshots * tps;

            MathUtils.SetSeed(100);
            using var model = new GPT1Model(new GPT1Config
            {
                VocabSize = MetricTokenizer.VocabSize,
                ContextLength = 16 * tps,
                DModel = 256,   // %256 → Q4_K frozen base
                NHeads = 4,
                NLayers = 2,
                DFF = 512,
                TieWeights = false,
                PreLayerNorm = true,
            });

            var tokenizer = new MetricTokenizer();

            // Benign regime (one pod, stable) + a benign-primed incident sequence.
            var benign = new MetricSnapshot[48];
            for (var i = 0; i < benign.Length; i++) { benign[i] = MakeNormalSnapshot("payments-api"); }
            var benignCorpus = tokenizer.EncodeSequence(benign);

            var incident = new MetricSnapshot[contextSnapshots + 8];
            for (var i = 0; i < contextSnapshots; i++) { incident[i] = MakeNormalSnapshot("payments-api"); }
            for (var i = contextSnapshots; i < incident.Length; i++) { incident[i] = MakeAnomalySnapshot("payments-api"); }
            var incidentCorpus = tokenizer.EncodeSequence(incident);

            using var tuner = new Gpt1LoRAFineTuner(
                model, rank: 8, LoRATargetModules.LanguageModelHead, seed: 7, quantizeBase: true);

            Assert.True(model.LMHeadOutputProvider is not null, "QLoRA hook not attached");

            var benignBefore = tuner.EvaluateLoss(benignCorpus, contextLength, start: 0);

            tuner.FineTune(benignCorpus, steps: 200, contextLength, learningRate: 0.01f, seed: 99);

            var benignAfter = tuner.EvaluateLoss(benignCorpus, contextLength, start: 0);
            var incidentAfter = tuner.EvaluateLoss(incidentCorpus, contextLength, start: 0);

            _out.WriteLine($"QLoRA anomaly: benign {benignBefore:F3} -> {benignAfter:F3} | incident (post-adapt) {incidentAfter:F3}");

            // 1. The adapter flattened the benign regime on the frozen Q4_K base.
            Assert.True(benignAfter < benignBefore * 0.6f, $"benign not flattened: {benignBefore:F3} -> {benignAfter:F3}");
            // 2. Discrimination preserved: the incident is still clearly more surprising than benign.
            Assert.True(incidentAfter > benignAfter * 1.5f, $"incident not separated: incident {incidentAfter:F3} vs benign {benignAfter:F3}");
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

        private static MetricSnapshot MakeAnomalySnapshot(string pod) => new()
        {
            Timestamp = DateTime.UtcNow,
            PodName = pod,
            CpuUsageRatio = 0.99f,
            CpuThrottleRatio = 0.65f,
            MemoryWorkingSetBytes = 1_900_000_000f,
            OomEventsRate = 3f,
            LatencyP50Ms = 220f,
            LatencyP95Ms = 1_400f,
            LatencyP99Ms = 2_900f,
            RequestsPerSecond = 12f,
            ErrorRate = 0.42f,
            GcGen2HeapBytes = 1_400_000_000f,
            GcPauseRatio = 0.38f,
            ThreadPoolQueueLength = 240f,
        };
    }
}
