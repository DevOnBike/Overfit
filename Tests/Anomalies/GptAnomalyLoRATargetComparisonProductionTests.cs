// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Gpt;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.Anomalies.Training;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.LoRA;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// The LoRA target A/B (<see cref="GptAnomalyLoRATargetComparisonTests"/>) repeated on
    /// the REAL trained production base (256d/6L) instead of a tiny random model — the
    /// counterpart that confirms (or refutes) the tiny-base finding on the deployable
    /// artifact. The open question: on a properly TRAINED base, is single-stage LoRA
    /// (especially LM-head, used by the earlier production validation) reliable, or does
    /// the tiny-base instability persist? Reports benign/incident/separation/params per
    /// target; asserts the robust invariants. [LongFact] — loads a ~20 MB checkpoint and
    /// fine-tunes several adapters (minutes); skipped unless the production base is present.
    /// </summary>
    public sealed class GptAnomalyLoRATargetComparisonProductionTests
    {
        private const int ContextSnapshots = 8;
        private const int Rank = 8;
        private const int Steps = 300;

        private readonly ITestOutputHelper _out;
        public GptAnomalyLoRATargetComparisonProductionTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void LoRATargetStages_OnProductionBase_FlattenBenign_AndKeepDetection()
        {
            var path = ResolveProductionBase();
            if (path is null)
            {
                _out.WriteLine("Production base not found (k8s_anomaly_production.bin) — skipping.");
                return;
            }

            var config = DetectConfigFromCheckpoint(path);
            _out.WriteLine($"Loaded base: {config.DModel}d / {config.NLayers}L from {path}");

            using var model = new GPT1Model(new GPT1Config
            {
                VocabSize = MetricTokenizer.VocabSize,
                ContextLength = config.ContextLength,
                DModel = config.DModel,
                NHeads = config.NHeads,
                NLayers = config.NLayers,
                DFF = config.DModel * 4,
                TieWeights = false,
                PreLayerNorm = true,
            });
            model.Eval();
            using (var br = new BinaryReader(File.OpenRead(path))) { model.Load(br); }

            var regime = new MetricSnapshot[48];
            for (var i = 0; i < regime.Length; i++) { regime[i] = MakeNormalSnapshot("payments-api"); }
            var incident = MakeAnomalySnapshot("payments-api");
            var corpus = new MetricTokenizer().EncodeSequence(regime);

            var (baseBenign, baseIncident) = Measure(model, regime, incident);
            var baseSeparation = baseIncident / MathF.Max(baseBenign, 1e-3f);
            _out.WriteLine($"BASE                     benign={baseBenign:F4}  incident={baseIncident:F4}  sep={baseSeparation:F1}x");
            _out.WriteLine("stage                    benign     incident   adapters  params");

            (string Name, LoRATargetModules Targets)[] stages =
            [
                ("Stage1 LMHead",      LoRATargetModules.LanguageModelHead),
                ("Stage2 FFN",         LoRATargetModules.FeedForward),
                ("Stage3 Attention",   LoRATargetModules.Attention),
                ("All  LMHead+FFN+Att", LoRATargetModules.AllLinear),
            ];

            var results = new Dictionary<string, (float Benign, float Incident)>();
            foreach (var (name, targets) in stages)
            {
                var (benign, incidentScore, adapters, parameters) = RunStage(model, corpus, regime, incident, targets);
                results[name] = (benign, incidentScore);
                _out.WriteLine(
                    $"{name,-24} {benign,8:F4}  {incidentScore,9:F4}  {adapters,8}  {parameters,7}  sep={incidentScore / MathF.Max(benign, 1e-3f),8:F1}x");

                Assert.False(float.IsNaN(benign) || float.IsInfinity(benign), $"{name}: benign not finite.");
                // Robust invariant: per-pod adaptation lowers the benign regime's score
                // below the (false-positive-prone) cross-pod base.
                Assert.True(benign < baseBenign,
                    $"{name}: did not flatten benign ({benign:F4} >= base {baseBenign:F4}).");
            }

            // The union is the reliable deployable choice (matches the tiny-base finding);
            // assert it flattens hard and keeps the incident firing.
            var all = results["All  LMHead+FFN+Att"];
            Assert.True(all.Benign < 1.0f, $"Union benign not flattened ({all.Benign:F4}).");
            Assert.True(all.Incident > 5.0f, $"Union blinded the detector (incident {all.Incident:F4}).");

            // Whether LM-head ALONE is now reliable (the trained-base nuance) is REPORTED,
            // not asserted — the comparison is the finding.
            var lm = results["Stage1 LMHead"];
            _out.WriteLine($"\nLM-head-only on trained base: benign={lm.Benign:F4} (tiny base was high-variance 0.07–6.9).");
        }

        private (float Benign, float Incident, int Adapters, long Params) RunStage(
            GPT1Model model, int[] corpus, MetricSnapshot[] regime, MetricSnapshot incident, LoRATargetModules targets)
        {
            var loraPath = Path.Combine(Path.GetTempPath(), $"overfit_prod_stage_{Guid.NewGuid():N}.bin");
            int adapters;
            long parameters;
            try
            {
                using (var tuner = new Gpt1LoRAFineTuner(model, Rank, targets))
                {
                    adapters = tuner.AdapterCount;
                    parameters = tuner.TrainableParameterCount;
                    tuner.FineTune(corpus, steps: Steps,
                        contextLength: ContextSnapshots * MetricTokenizer.TokensPerSnapshot, learningRate: 1e-2f);
                    tuner.Save(loraPath);
                }

                using var merge = Gpt1LoRAMergeAdapter.Load(model, loraPath);
                merge.Enable();
                var (benign, incidentScore) = Measure(model, regime, incident);
                merge.Disable();
                return (benign, incidentScore, adapters, parameters);
            }
            finally
            {
                if (File.Exists(loraPath)) { File.Delete(loraPath); }
            }
        }

        private static (float normal, float incident) Measure(
            GPT1Model model, MetricSnapshot[] regime, MetricSnapshot incident)
        {
            using var handle = SlmRuntimeFactory.CreateGpt1(model);
            using var detector = new GptAnomalyDetector(handle, ContextSnapshots);
            var normal = 0f;
            for (var i = 0; i < ContextSnapshots * 2; i++)
            {
                var r = detector.Score(regime[i % regime.Length]);
                if (!r.IsWarmup) { normal = r.Score; }
            }
            return (normal, detector.Score(incident).Score);
        }

        private static string? ResolveProductionBase()
        {
            var dir = Environment.GetEnvironmentVariable("OVERFIT_MODEL_DIR");
            var candidates = new[]
            {
                dir is null ? null : Path.Combine(dir, "k8s_anomaly_production.bin"),
                Path.Combine("test_fixtures", "k8s_anomaly_production.bin"),
                @"D:\k8s_anomaly_production.bin",
            };
            foreach (var c in candidates)
            {
                if (c is not null && File.Exists(c)) { return c; }
            }
            return null;
        }

        private static GptTrainingConfig DetectConfigFromCheckpoint(string path)
        {
            using var fs = File.OpenRead(path);
            using var br = new BinaryReader(fs);
            var length = br.ReadInt32();
            var dModel = length / MetricTokenizer.VocabSize;
            return dModel switch
            {
                64 => GptTrainingConfig.Quick,
                128 => GptTrainingConfig.Medium,
                256 => GptTrainingConfig.Production,
                _ => new GptTrainingConfig { DModel = dModel, NHeads = dModel / 32, NLayers = 4, ContextLength = 120 },
            };
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
