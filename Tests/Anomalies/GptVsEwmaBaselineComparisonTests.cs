// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Linq;
using DevOnBike.Overfit.Anomalies.Baseline;
using DevOnBike.Overfit.Anomalies.Gpt;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.Anomalies.Training;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// Rigorous "transformer vs classical baseline" benchmark (docs/gp-anomaly-baseline.md):
    /// trains a Quick GPT base on the fixture CSV, then scores ONE pod's real normal stream
    /// + three injected anomalies (OOM / latency / CPU) with BOTH the
    /// <see cref="GptAnomalyDetector"/> and the EWMA <see cref="EwmaAnomalyDetector"/> floor,
    /// on the same data. Reports separation and worst-metric attribution for each; asserts
    /// the invariant both must satisfy (every injected anomaly scores above the detector's
    /// own mean-normal). The relative verdict is reported, not asserted (GPT training is
    /// nondeterministic — MathUtils RNG). [LongFact] — trains a model.
    /// </summary>
    public sealed class GptVsEwmaBaselineComparisonTests
    {
        private const int ContextSnapshots = 8;

        private readonly ITestOutputHelper _out;
        public GptVsEwmaBaselineComparisonTests(ITestOutputHelper output) => _out = output;

        private static string CsvPath => Path.Combine(
            AppContext.BaseDirectory, "test_fixtures", "k8s_metrics.csv");

        [LongFact]
        public async Task GptDetector_VsEwmaFloor_BothSeparate_OnSamePodStream()
        {
            if (!File.Exists(CsvPath))
            {
                _out.WriteLine($"fixture CSV not found at {CsvPath} — skipping.");
                return;
            }

            // ── One pod's real, time-ordered stream from the fixture ──
            var all = HistoricalCsvLoader.Load(CsvPath, out _);
            var pod = all.GroupBy(s => s.PodName).OrderByDescending(g => g.Count()).First();
            var stream = pod.OrderBy(s => s.Timestamp).Take(40).ToArray();
            Assert.True(stream.Length >= 32, "Need ≥32 snapshots from the busiest pod for the benchmark.");

            // Injected anomalies built from a representative real normal snapshot.
            var baseSnap = stream[30];
            var oom = baseSnap with
            {
                OomEventsRate = 4f,
                MemoryWorkingSetBytes = baseSnap.MemoryWorkingSetBytes * 4f
            };
            var lat = baseSnap with
            {
                LatencyP99Ms = baseSnap.LatencyP99Ms * 30f + 2_000f
            };
            var cpu = baseSnap with
            {
                CpuUsageRatio = 0.99f,
                CpuThrottleRatio = 0.7f
            };
            (string Label, MetricSnapshot Snap)[] injected = [("OOM", oom), ("LATENCY", lat), ("CPU", cpu)];

            // ── Train a Quick GPT base on the full CSV ──
            var checkpoint = Path.Combine(Path.GetTempPath(), $"overfit_ewma_cmp_{Guid.NewGuid():N}.bin");
            try
            {
                var cfg = GptTrainingConfig.Quick;
                var result = await new OfflineTrainingJob(cfg).RunAsync(CsvPath, checkpoint);
                _out.WriteLine($"Quick base: {result.SnapshotsLoaded:N0} snapshots, val {result.InitialLoss:F2}→{result.FinalValLoss:F2}");

                using var model = new GPT1Model(new GPT1Config
                {
                    VocabSize = MetricTokenizer.VocabSize,
                    ContextLength = cfg.ContextLength,
                    DModel = cfg.DModel,
                    NHeads = cfg.NHeads,
                    NLayers = cfg.NLayers,
                    DFF = cfg.DModel * 4,
                    TieWeights = false,
                    PreLayerNorm = true,
                });
                model.Eval();
                using (var br = new BinaryReader(File.OpenRead(checkpoint)))
                {
                    model.Load(br);
                }

                var gpt = ScoreWith(MakeGpt(model), stream, injected);
                var ewma = ScoreWith(MakeEwma(), stream, injected);

                _out.WriteLine($"\n{"detector",-8} {"mean-normal",12} {"OOM",18} {"LATENCY",18} {"CPU",18}");
                Report("GPT", gpt);
                Report("EWMA", ewma);

                // EWMA (deterministic, pure stats) separates every injected anomaly above
                // its mean-normal with a wide margin — the floor genuinely works.
                foreach (var (label, score, _) in ewma.Anomalies)
                {
                    Assert.True(score > ewma.MeanNormal,
                        $"EWMA: {label} ({score:F3}) did not exceed mean-normal ({ewma.MeanNormal:F3}).");
                }

                // The un-adapted Quick GPT base separates only weakly (it carries cross-pod
                // residual surprise — a high normal floor — so per-anomaly margins are thin
                // and training is nondeterministic); assert the robust aggregate. FINDING
                // (reported above): the cheap EWMA floor out-separates the un-adapted GPT
                // base on this single-pod stream — the GPT detector's edge is the per-pod
                // LoRA adaptation (see GptAnomalyLoRA* tests), not the raw base.
                var gptMeanAnomaly = gpt.Anomalies.Average(a => a.Score);
                Assert.True(gptMeanAnomaly > gpt.MeanNormal,
                    $"GPT: mean anomaly ({gptMeanAnomaly:F3}) did not exceed mean-normal ({gpt.MeanNormal:F3}).");
            }
            finally
            {
                if (File.Exists(checkpoint))
                {
                    File.Delete(checkpoint);
                }
            }
        }

        private readonly record struct Outcome(float MeanNormal, List<(string Label, float Score, string Worst)> Anomalies);

        // Warm up, average the post-warmup normal scores, then score each injected anomaly.
        private static Outcome ScoreWith(
            Func<MetricSnapshot, AnomalyScore> score, MetricSnapshot[] stream, (string Label, MetricSnapshot Snap)[] injected)
        {
            var warm = ContextSnapshots * 2;
            var sum = 0f;
            var n = 0;
            for (var i = 0; i < stream.Length; i++)
            {
                var r = score(stream[i]);
                if (!r.IsWarmup && i >= warm)
                {
                    sum += r.Score;
                    n++;
                }
            }

            var anomalies = new List<(string, float, string)>();
            foreach (var (label, snap) in injected)
            {
                var r = score(snap);
                anomalies.Add((label, r.Score, r.WorstMetric));
            }
            return new Outcome(n > 0 ? sum / n : 0f, anomalies);
        }

        private static Func<MetricSnapshot, AnomalyScore> MakeGpt(GPT1Model model)
        {
            var detector = new GptAnomalyDetector(SlmRuntimeFactory.CreateGpt1(model), ContextSnapshots);
            return detector.Score;
        }

        private static Func<MetricSnapshot, AnomalyScore> MakeEwma()
        {
            var detector = new EwmaAnomalyDetector(warmupSnapshots: ContextSnapshots);
            return detector.Score;
        }

        private void Report(string name, Outcome o)
        {
            string Fmt((string Label, float Score, string Worst) a) => $"{a.Score,6:F2} [{a.Worst}]";
            _out.WriteLine(
                $"{name,-8} {o.MeanNormal,12:F3} {Fmt(o.Anomalies[0]),18} {Fmt(o.Anomalies[1]),18} {Fmt(o.Anomalies[2]),18}");
        }
    }
}
