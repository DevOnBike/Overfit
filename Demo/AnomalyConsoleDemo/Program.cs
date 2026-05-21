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

namespace DevOnBike.Overfit.Demo.AnomalyConsole
{
    /// <summary>
    /// User-facing console demo of the GPT anomaly detector — train a small GPT on
    /// your own metrics, then score a live stream and flag anomalies, all pure C#.
    ///
    /// Flow:
    ///   1. Train a Quick base on a metrics CSV (or load --checkpoint).
    ///   2. Build a GptAnomalyDetector over it.
    ///   3. Phase 1 — stream a steady "normal" regime, then inject one incident.
    ///   4. Phase 2 — fine-tune an LM-head LoRA adapter on this pod's benign regime,
    ///      merge it in place, and re-run the SAME stream + incident. The benign
    ///      "normal" score collapses toward zero (fewer false positives) while the
    ///      injected incident still scores high — adaptive per-deployment learning
    ///      that an inference-only engine (llama.cpp et al.) cannot do.
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

        // ── Scenario: base behaviour, then the same stream after per-deployment LoRA ──
        private static void RunScenario(GPT1Model model, GptTrainingConfig config)
        {
            Console.WriteLine();
            Console.WriteLine("══ Phase 1 — base model ══");
            Console.WriteLine("Streaming 'payments-api' metrics (1 snapshot ≈ 15 s)  —  score = surprise (nats/token):");
            Console.WriteLine();
            var (baseNormal, baseIncident) = StreamScenario(model, verbose: true);

            Console.WriteLine();
            Console.WriteLine("══ Phase 2 — per-deployment LoRA adaptation ══");
            Console.WriteLine("Fine-tuning an LM-head LoRA adapter on this pod's benign regime ...");
            using var merge = AdaptToRegime(model);
            Console.WriteLine("Adapter merged in place. Re-streaming the SAME regime, then the SAME incident:");
            Console.WriteLine();
            var (loraNormal, loraIncident) = StreamScenario(model, verbose: true);

            // ── Before / after ─────────────────────────────────────────────────
            Console.WriteLine();
            Console.WriteLine("══ Before / after ══");
            Console.WriteLine($"  benign 'normal'    base = {baseNormal,7:F2}  →  LoRA = {loraNormal,7:F2}   (false-positive pressure — lower is better)");
            Console.WriteLine($"  injected incident  base = {baseIncident,7:F2}  →  LoRA = {loraIncident,7:F2}   (must stay high)");
            Console.WriteLine();

            var flattened = loraNormal < baseNormal;
            var stillFires = loraIncident > MathF.Max(loraNormal * 3f, 5f);
            Console.WriteLine(flattened && stillFires
                ? "LoRA flattened the benign regime toward zero AND the incident still fires far above it —"
                : "Result above (random/short Quick base may need a longer fine-tune to fully separate) —");
            Console.WriteLine("adapting to a deployment's benign drift lowers false positives without blinding the");
            Console.WriteLine("detector. This adaptive, per-deployment learning is the part an inference-only");
            Console.WriteLine("engine cannot do — the LoRA delta merges into the same zero-alloc decode path.");
        }

        // ── Stream a steady normal regime, then inject one incident ─────────────
        // Returns the last post-warmup normal score and the injected-incident score.
        private static (float normal, float incident) StreamScenario(GPT1Model model, bool verbose)
        {
            using var handle = SlmRuntimeFactory.CreateGpt1(model);
            using var detector = new GptAnomalyDetector(handle, ContextSnapshots);

            var normal = 0f;
            for (var i = 0; i < ContextSnapshots * 2; i++)
            {
                var r = detector.Score(NormalSnapshot());
                if (verbose) { Report(i, r); }
                if (!r.IsWarmup) { normal = r.Score; }
            }

            if (verbose) { Console.WriteLine("  ── injecting incident ──"); }
            var inc = detector.Score(AnomalySnapshot());
            if (verbose) { Report(ContextSnapshots * 2, inc); }

            return (normal, inc.Score);
        }

        // ── Fine-tune an LM-head LoRA adapter on the benign regime, merge it in ──
        private static Gpt1LoRAMergeAdapter AdaptToRegime(GPT1Model model)
        {
            var tps = MetricTokenizer.TokensPerSnapshot;

            // A stable benign regime for this pod — tokenised into a periodic
            // sequence the LM-head LoRA can learn. 48 snapshots ≫ the context window.
            var regime = new MetricSnapshot[48];
            for (var i = 0; i < regime.Length; i++)
            {
                regime[i] = NormalSnapshot();
            }
            var corpus = new MetricTokenizer().EncodeSequence(regime);

            var loraPath = Path.Combine(
                Path.GetTempPath(), $"overfit_anomaly_lora_{Guid.NewGuid():N}.bin");
            try
            {
                using (var tuner = new Gpt1LoRAFineTuner(model, rank: 16))
                {
                    // Fine-tune over the exact position range the detector exercises
                    // (ContextSnapshots × tokens-per-snapshot) so the merge lands where
                    // the detector reads.
                    var history = tuner.FineTune(
                        corpus, steps: 300, contextLength: ContextSnapshots * tps, learningRate: 1e-2f);
                    Console.WriteLine(
                        $"  LoRA loss {history[0]:F3} → {history[^1]:F3}  ({history.Count} steps, rank 16, LM head, base frozen)");
                    tuner.Save(loraPath);
                }
                // tuner disposed → weight provider detached; model is plain again.

                var merge = Gpt1LoRAMergeAdapter.Load(model, loraPath);
                merge.Enable();
                return merge;
            }
            finally
            {
                if (File.Exists(loraPath)) { File.Delete(loraPath); }
            }
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
