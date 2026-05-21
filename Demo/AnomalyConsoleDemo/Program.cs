// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Gpt;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.Anomalies.Training;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Demo.AnomalyConsole
{
    /// <summary>
    /// User-facing console demo of the GPT anomaly detector — train a small GPT on
    /// your own metrics, then score a live stream and flag anomalies, all pure C#.
    ///
    /// Flow:
    ///   1. Train a Quick base on a metrics CSV (or load --checkpoint).
    ///   2. Build a GptAnomalyDetector over it.
    ///   3. Stream a steady "normal" regime, then inject one incident snapshot.
    ///   4. Print each step's anomaly score + the worst-explaining metric.
    ///
    /// Example:
    ///   dotnet run -c Release --project Demo/AnomalyConsoleDemo -- \
    ///     --csv Tests/test_fixtures/k8s_metrics.csv
    ///
    /// Path resolution for --csv: explicit arg → $OVERFIT_MODEL_DIR/k8s_metrics.csv
    /// → Tests/test_fixtures/k8s_metrics.csv → ./k8s_metrics.csv.
    /// </summary>
    internal static class Program
    {
        private const int ContextSnapshots = 8;

        private static async Task<int> Main(string[] args)
        {
            try
            {
                var csv = ResolveCsv(GetArg(args, "--csv"));
                var checkpoint = GetArg(args, "--checkpoint") ?? "anomaly_demo.bin";

                var (model, config) = await GetModelAsync(csv, checkpoint);
                using (model)
                {
                    RunScenario(model, config);
                }

                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"error: {ex.Message}");
                return 1;
            }
        }

        // ── Model: load a checkpoint if present, else train a Quick base ─────────
        private static async Task<(GPT1Model model, GptTrainingConfig config)> GetModelAsync(
            string? csv, string checkpoint)
        {
            var config = GptTrainingConfig.Quick;

            if (!File.Exists(checkpoint))
            {
                if (csv is null || !File.Exists(csv))
                {
                    throw new FileNotFoundException(
                        "No checkpoint and no metrics CSV found. Pass --csv <path> " +
                        "(e.g. Tests/test_fixtures/k8s_metrics.csv) or --checkpoint <path>.");
                }

                Console.WriteLine($"No checkpoint — training a Quick base on {csv} ...");
                var job = new OfflineTrainingJob(config);
                var progress = new Progress<TrainingProgress>(p =>
                {
                    if (p.Step > 0 && p.TrainLoss > 0)
                    {
                        Console.WriteLine($"  step {p.Step}/{p.TotalSteps}  train={p.TrainLoss:F3}  val={p.ValLoss:F3}");
                    }
                });
                var result = await job.RunAsync(csv, checkpoint, progress);
                Console.WriteLine($"Trained: {result.SnapshotsLoaded:N0} snapshots, " +
                    $"val loss {result.InitialLoss:F2} → {result.FinalValLoss:F2}, {result.TrainingTime:mm\\:ss}.");
            }
            else
            {
                Console.WriteLine($"Loading checkpoint {checkpoint} ...");
            }

            var model = new GPT1Model(new GPT1Config
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

            using (var fs = File.OpenRead(checkpoint))
            using (var br = new BinaryReader(fs))
            {
                model.Load(br);
            }

            return (model, config);
        }

        // ── Scenario: steady normal stream, then one injected incident ──────────
        private static void RunScenario(GPT1Model model, GptTrainingConfig config)
        {
            using var handle = SlmRuntimeFactory.CreateGpt1(model);
            using var detector = new GptAnomalyDetector(handle, ContextSnapshots);

            Console.WriteLine();
            Console.WriteLine("Streaming 'payments-api' metrics (1 snapshot ≈ 15 s)  —  score = surprise (nats/token):");
            Console.WriteLine();

            // Steady normal regime — fills the window, then settles to a low score.
            for (var i = 0; i < ContextSnapshots * 2; i++)
            {
                Report(i, detector.Score(NormalSnapshot()));
            }

            // Injected incident: CPU saturated + throttled, tail latency + errors blown out.
            Console.WriteLine("  ── injecting incident ──");
            Report(ContextSnapshots * 2, detector.Score(AnomalySnapshot()));

            Console.WriteLine();
            Console.WriteLine("The base scores its trained 'normal' low and the incident high, naming the");
            Console.WriteLine("worst metric. Per-deployment LoRA fine-tuning (see Gpt1LoRAFineTuner) drives a");
            Console.WriteLine("benign regime's score toward zero while still catching real incidents.");
        }

        private static void Report(int step, AnomalyScore s)
        {
            if (s.IsWarmup)
            {
                Console.WriteLine($"  t={step,2}  (warmup — filling {ContextSnapshots}-snapshot window)");
                return;
            }

            var bar = new string('█', Math.Min(40, (int)MathF.Round(s.Score * 2f)));
            var flag = s.Score > 5f ? "  ⚠ ANOMALY" : string.Empty;
            Console.WriteLine($"  t={step,2}  score={s.Score,6:F2}  worst={s.WorstMetric,-22} {bar}{flag}");
        }

        // ── Synthetic snapshots (mirror the integration tests) ──────────────────
        private static MetricSnapshot NormalSnapshot() => new()
        {
            Timestamp = DateTime.UtcNow,
            PodName = "payments-api",
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

        private static MetricSnapshot AnomalySnapshot() => new()
        {
            Timestamp = DateTime.UtcNow,
            PodName = "payments-api",
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

        // ── CLI helpers ─────────────────────────────────────────────────────────
        private static string? GetArg(string[] args, string name)
        {
            for (var i = 0; i < args.Length - 1; i++)
            {
                if (args[i] == name) { return args[i + 1]; }
            }
            return null;
        }

        private static string? ResolveCsv(string? explicitPath)
        {
            if (!string.IsNullOrEmpty(explicitPath)) { return explicitPath; }

            var dir = Environment.GetEnvironmentVariable("OVERFIT_MODEL_DIR");
            var candidates = new[]
            {
                dir is null ? null : Path.Combine(dir, "k8s_metrics.csv"),
                Path.Combine("Tests", "test_fixtures", "k8s_metrics.csv"),
                "k8s_metrics.csv",
            };

            foreach (var c in candidates)
            {
                if (c is not null && File.Exists(c)) { return c; }
            }

            return null;
        }
    }
}
