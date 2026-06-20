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
    /// End-to-end validation of the per-deployment LoRA story on the REAL trained
    /// production base (256d/6L, val loss 0.856), not a random-init toy model. This
    /// is the deployable-artifact counterpart to <see cref="GptAnomalyLoRAIntegrationTests"/>
    /// (which proves the same loop on a tiny model fast, in CI).
    ///
    /// The claim under test — the deployment value proposition: a base trained across
    /// all pods carries residual surprise on any single pod's benign regime (it can
    /// even cross the anomaly threshold = a false positive), and a cheap per-pod
    /// LM-head LoRA drives that benign score toward zero WITHOUT blinding the detector
    /// to a real incident on the same pod.
    ///
    /// [LongFact]: loads a ~20 MB checkpoint + fine-tunes 300 LoRA steps on a 256d
    /// model — seconds, but real-model-dependent. Skipped by default; flip to [Fact]
    /// on a box where the production base is present to run it. Resolves the base from
    /// $OVERFIT_MODEL_DIR, test_fixtures/, or D:\ (dev box); no-ops with a log if none
    /// found so it never fails for a missing artifact.
    /// </summary>
    public sealed class GptAnomalyProductionLoRATests
    {
        private const int ContextSnapshots = 8;

        private readonly ITestOutputHelper _output;

        public GptAnomalyProductionLoRATests(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact]
        public void ProductionBase_PerPodLoRA_FlattensBenignRegime_StillFlagsIncident()
        {
            var path = ResolveProductionBase();
            if (path is null)
            {
                _output.WriteLine(
                    "Production base not found ($OVERFIT_MODEL_DIR / test_fixtures / D:\\ " +
                    "k8s_anomaly_production.bin) — skipping.");
                return;
            }

            var config = DetectConfigFromCheckpoint(path);
            _output.WriteLine($"Loaded base: {config.DModel}d / {config.NLayers}L from {path}");

            using var model = new GPT1Model(new GPT1Config
            {
                VocabSize = MetricTokenizer.VocabSize,
                ContextLength = config.ContextLength,
                DModel = config.DModel,
                NHeads = config.NHeads,
                NLayers = config.NLayers,
                DFF = config.DModel * 4,
                TieWeights = false,   // LoRA targets the untied LM head
                PreLayerNorm = true,
            });
            model.Eval();
            using (var fs = File.OpenRead(path))
            using (var br = new BinaryReader(fs))
            {
                model.Load(br);
            }

            var tps = MetricTokenizer.TokensPerSnapshot;

            // A stable benign regime for one pod, and the incident we must keep catching.
            var regime = new MetricSnapshot[48];
            for (var i = 0; i < regime.Length; i++)
            {
                regime[i] = MakeNormalSnapshot("payments-api");
            }
            var incident = MakeAnomalySnapshot("payments-api");
            var corpus = new MetricTokenizer().EncodeSequence(regime);

            // ── Base behaviour ──────────────────────────────────────────────────
            var (baseNormal, baseIncident) = Measure(model, regime, incident);

            // ── Per-pod LM-head LoRA, merged in place ───────────────────────────
            var loraPath = Path.Combine(
                Path.GetTempPath(), $"overfit_prod_lora_{Guid.NewGuid():N}.bin");
            try
            {
                using (var tuner = new Gpt1LoRAFineTuner(model, rank: 16))
                {
                    var history = tuner.FineTune(
                        corpus, steps: 300, contextLength: ContextSnapshots * tps, learningRate: 1e-2f);
                    _output.WriteLine($"LoRA loss {history[0]:F3} -> {history[^1]:F3}");
                    tuner.Save(loraPath);
                }

                using var merge = Gpt1LoRAMergeAdapter.Load(model, loraPath);
                merge.Enable();
                var (loraNormal, loraIncident) = Measure(model, regime, incident);

                _output.WriteLine($"benign  base={baseNormal:F2}  -> LoRA={loraNormal:F2}");
                _output.WriteLine($"incident base={baseIncident:F2} -> LoRA={loraIncident:F2}");

                // (1) LoRA lowered the benign regime's score, and flattened it near zero.
                Assert.True(loraNormal < baseNormal,
                    $"LoRA did not lower benign score: base={baseNormal:F3}, lora={loraNormal:F3}.");
                Assert.True(loraNormal < 1.0f,
                    $"LoRA did not flatten the benign regime: lora-normal={loraNormal:F3}.");

                // (2) The detector still fires on a real incident, far above benign.
                Assert.False(float.IsNaN(loraIncident) || float.IsInfinity(loraIncident),
                    "Adapted incident score is not finite.");
                Assert.True(loraIncident > 5.0f,
                    $"LoRA blinded the detector to a real incident: incident={loraIncident:F3}.");
                Assert.True(loraIncident > MathF.Max(loraNormal * 3f, 5f),
                    $"Adapted incident not clearly separated: normal={loraNormal:F3}, incident={loraIncident:F3}.");
            }
            finally
            {
                if (File.Exists(loraPath))
                {
                    File.Delete(loraPath);
                }
            }
        }

        // Builds a fresh detector over the model's current weights, warms it on the
        // benign regime, then scores one incident against that primed context.
        private static (float normal, float incident) Measure(
            GPT1Model model, MetricSnapshot[] regime, MetricSnapshot incident)
        {
            using var handle = SlmRuntimeFactory.CreateGpt1(model);
            using var detector = new GptAnomalyDetector(handle, ContextSnapshots);

            var normal = 0f;
            for (var i = 0; i < ContextSnapshots * 2; i++)
            {
                var r = detector.Score(regime[i % regime.Length]);
                if (!r.IsWarmup)
                {
                    normal = r.Score;
                }
            }
            var inc = detector.Score(incident);
            return (normal, inc.Score);
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
                if (c is not null && File.Exists(c))
                {
                    return c;
                }
            }
            return null;
        }

        // Mirrors GptAnomalyDetectorTests: dModel = first-parameter length / VocabSize.
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
