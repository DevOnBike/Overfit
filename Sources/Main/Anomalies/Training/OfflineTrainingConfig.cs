// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Training
{
    /// <summary>
    /// Configuration for OfflineTrainingJob.
    ///
    /// Model size guidance:
    ///   Small (fast, ~30 min, ~2M params):  DModel=128, NHeads=4, NLayers=4
    ///   Medium (~2h, ~10M params):           DModel=256, NHeads=8, NLayers=6
    ///   Large (~8h, ~30M params):            DModel=512, NHeads=8, NLayers=8
    ///
    /// Context window:
    ///   ContextLength tokens / MetricTokenizer.TokensPerSnapshot (12) = snapshots in window
    ///   Default 256 / 12 = 21 snapshots = ~5 minutes at 15s scrape interval
    ///   For longer patterns (hourly jobs), use ContextLength=768 (64 snapshots = 16 min)
    /// </summary>
    public sealed class GptTrainingConfig
    {
        // ── Model architecture ────────────────────────────────────────────────
        public int DModel { get; set; } = 128;
        public int NHeads { get; set; } = 4;
        public int NLayers { get; set; } = 4;

        /// <summary>
        /// Token context window. Must be divisible by MetricTokenizer.TokensPerSnapshot (12).
        /// Default 252 = 21 snapshots = 5.25 minutes.
        /// </summary>
        public int ContextLength { get; set; } = 252;

        // ── Training ──────────────────────────────────────────────────────────
        public int Steps { get; set; } = 5_000;
        public int ReportEvery { get; set; } = 500;
        public int ValSteps { get; set; } = 50;
        public float LearningRateMax { get; set; } = 3e-4f;
        public float LearningRateMin { get; set; } = 3e-5f;
        public float WeightDecay { get; set; } = 0.1f;
        public float MaxGradNorm { get; set; } = 1.0f;
        public int Seed { get; set; } = 42;

        /// <summary>
        /// Data parallel workers. Each worker runs a separate forward/backward,
        /// gradients are averaged to master. Effective batch = WorkerCount × 1.
        /// LR is scaled by sqrt(WorkerCount) automatically.
        /// Default: 8 workers on Ryzen 9 9950X3D (32 cores).
        /// </summary>
        public int WorkerCount { get; set; } = 8;

        /// <summary>
        /// ComputationGraph arena in floats. Default 100M = 400MB.
        /// Increase if you get NativeBuffer exhausted during training.
        /// </summary>
        public int ArenaSize { get; set; } = 100_000_000;

        /// <summary>Preset for quick experiments (~5 min training).</summary>
        public static GptTrainingConfig Quick => new()
        {
            DModel = 64,
            NHeads = 2,
            NLayers = 2,
            ContextLength = 120,
            Steps = 1_000,
            ReportEvery = 200,
        };

        /// <summary>
        /// Preset for quick validation (~5-10 min training).
        /// Enough to verify the pipeline works end-to-end before committing to Production.
        /// Expected val loss: ~3.0-3.5 after 2000 steps.
        /// </summary>
        public static GptTrainingConfig Medium => new GptTrainingConfig
        {
            DModel = 128,
            NHeads = 4,
            NLayers = 4,
            ContextLength = 120,   // 10 snapshots = ~2.5 min context
            Steps = 2_000,
            ReportEvery = 200,
            ValSteps = 30,
            LearningRateMax = 3e-4f,
            LearningRateMin = 3e-5f,
            MaxGradNorm = 1.0f,
            WeightDecay = 0.1f,
            ArenaSize = 80_000_000, // 320MB
            WorkerCount = 8,   // 8 parallel workers
        };

        /// <summary>Preset for production quality (~2h training).</summary>
        public static GptTrainingConfig Production => new GptTrainingConfig
        {
            DModel = 256,
            NHeads = 8,
            NLayers = 6,
            ContextLength = 252,   // 21 snapshots = ~5 min context
            Steps = 10_000,
            ReportEvery = 1_000,
            ValSteps = 50,
            LearningRateMax = 3e-4f,
            LearningRateMin = 3e-5f,
            MaxGradNorm = 1.0f,
            WeightDecay = 0.1f,
            ArenaSize = 400_000_000, // 1.6GB
            WorkerCount = 8,   // 8 parallel workers
        };
    }

    /// <summary>Result returned after OfflineTrainingJob.RunAsync completes.</summary>
    public sealed class OfflineTrainingResult
    {
        public int SnapshotsLoaded { get; init; }
        public int SkippedCsvRows { get; init; }
        public float InitialLoss { get; init; }
        public float FinalValLoss { get; init; }
        public string CheckpointPath { get; init; } = string.Empty;
        public TimeSpan TrainingTime { get; init; }

        public override string ToString() =>
            $"Snapshots: {SnapshotsLoaded:N0} | ValLoss: {FinalValLoss:F4} | " +
            $"Time: {TrainingTime:mm\\:ss} | Checkpoint: {CheckpointPath}";
    }
}
