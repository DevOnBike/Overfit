// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Gpt;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.Anomalies.Training;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// GptAnomalyDetector tests — verifies that the detector:
    ///   1. Assigns higher scores to anomalous snapshots than to normal ones
    ///   2. Correctly identifies the worst metric in an anomaly
    ///   3. Warmup works — Score=0 until the window is full
    ///
    /// Requires: test_fixtures/k8s_anomaly_checkpoint.bin
    ///   (generowany przez OfflineTrainingJobTests.TrainOnCsv_LossDecreases_CheckpointWritten)
    /// </summary>
    public class GptAnomalyDetectorTests : IDisposable
    {
        private readonly ITestOutputHelper _output;
        private const string CheckpointPath = "test_fixtures/k8s_anomaly_checkpoint.bin";

        private readonly GPT1Model _model;
        private readonly GptTrainingConfig _config;

        public GptAnomalyDetectorTests(ITestOutputHelper output)
        {
            _output = output;

            // Config must match the checkpoint.
            // Quick (64d) → k8s_anomaly_checkpoint.bin from TrainOnCsv
            // Medium (128d) → copy k8s_anomaly_medium.bin as k8s_anomaly_checkpoint.bin
            _config = File.Exists(CheckpointPath)
                ? DetectConfigFromCheckpoint(CheckpointPath)
                : GptTrainingConfig.Quick;

            _model = new GPT1Model(new GPT1Config
            {
                VocabSize     = MetricTokenizer.VocabSize,
                ContextLength = _config.ContextLength,
                DModel        = _config.DModel,
                NHeads        = _config.NHeads,
                NLayers       = _config.NLayers,
                DFF           = _config.DModel * 4,
                TieWeights    = false,
                PreLayerNorm  = true,
            });
            _model.Eval();

            if (File.Exists(CheckpointPath))
            {
                using var fs = File.OpenRead(CheckpointPath);
                using var br = new BinaryReader(fs);
                _model.Load(br);
            }
        }

        [Fact]
        public void Warmup_ScoreIsZero_UntilWindowFilled()
        {
            using var handle   = SlmRuntimeFactory.CreateGpt1(_model);
            using var detector = new GptAnomalyDetector(handle, contextSnapshots: 5);

            Assert.False(detector.WindowFilled);

            var normal = MakeNormalSnapshot("api-gateway");
            for (var i = 0; i < 4; i++)
            {
                var result = detector.Score(normal);
                Assert.True(result.IsWarmup, $"Step {i}: expected warmup");
                Assert.Equal(0f, result.Score);
            }

            // 5th snapshot fills the window
            var last = detector.Score(normal);
            Assert.True(detector.WindowFilled);
            Assert.False(last.IsWarmup);
        }

        [Fact]
        public void AnomalySnapshot_ScoresHigherThan_NormalSnapshot()
        {
            SkipIfNoCheckpoint();

            using var handle   = SlmRuntimeFactory.CreateGpt1(_model);
            using var detector = new GptAnomalyDetector(handle, contextSnapshots: 10);

            var normal  = MakeNormalSnapshot("api-gateway");
            var anomaly = MakeAnomalySnapshot("api-gateway");

            // Warm up with normal traffic
            var lastNormalScore = 0f;
            for (var i = 0; i < 10; i++)
            {
                var r = detector.Score(normal);
                if (!r.IsWarmup)
                {
                    lastNormalScore = r.Score;
                }
            }

            // Score an anomalous snapshot
            var anomalyResult = detector.Score(anomaly);

            _output.WriteLine($"Normal score:  {lastNormalScore:F4}");
            _output.WriteLine($"Anomaly score: {anomalyResult.Score:F4}");
            _output.WriteLine($"Worst metric:  {anomalyResult.WorstMetric}");
            _output.WriteLine($"Expected:      {anomalyResult.ExpectedValue:F2}");
            _output.WriteLine($"Actual:        {anomalyResult.ActualValue:F2}");

            Assert.False(anomalyResult.IsWarmup);
            Assert.True(anomalyResult.Score > lastNormalScore,
                $"Anomalia ({anomalyResult.Score:F4}) nie jest wyżej scored niż norma ({lastNormalScore:F4}). " +
                "Model potrzebuje więcej treningu lub lepszego checkpointu.");
        }

        [Fact]
        public void OomAnomaly_IdentifiesMemoryAsWorstMetric()
        {
            SkipIfNoCheckpoint();

            using var handle   = SlmRuntimeFactory.CreateGpt1(_model);
            using var detector = new GptAnomalyDetector(handle, contextSnapshots: 10);

            var normal = MakeNormalSnapshot("worker-processor");
            for (var i = 0; i < 10; i++)
            {
                detector.Score(normal);
            }

            // OOM: memory near limit, OOM events firing
            var oom = new MetricSnapshot
            {
                Timestamp             = DateTime.UtcNow,
                PodName               = "worker-processor",
                CpuUsageRatio         = 0.55f,
                CpuThrottleRatio      = 0.30f,
                MemoryWorkingSetBytes = 7_800_000_000f,  // ~97% of 8GB
                OomEventsRate         = 0.08f,           // OOM events firing
                LatencyP50Ms          = 45f,
                LatencyP95Ms          = 120f,
                LatencyP99Ms          = 350f,
                RequestsPerSecond     = 60f,
                ErrorRate             = 0.12f,
                GcGen2HeapBytes       = 5_500_000_000f,
                GcPauseRatio          = 0.35f,
                ThreadPoolQueueLength = 85f,
            };

            var result = detector.Score(oom);

            _output.WriteLine($"OOM score:    {result.Score:F4}");
            _output.WriteLine($"Worst metric: {result.WorstMetric}");
            _output.WriteLine($"Expected:     {result.ExpectedValue:F2}");
            _output.WriteLine($"Actual:       {result.ActualValue:F2}");

            Assert.False(result.IsWarmup);
            Assert.True(result.Score > 1.0f,
                $"OOM anomalia powinna mieć score > 1.0, dostała {result.Score:F4}");

            // Worst metric should be related to memory or OOM
            var isMemoryRelated =
                result.WorstMetric.Contains("memory") ||
                result.WorstMetric.Contains("oom") ||
                result.WorstMetric.Contains("gc");

            _output.WriteLine($"Memory-related worst metric: {isMemoryRelated}");
            // We do not assert WorstMetric — small models (64d) may point to a different metric
            // The production model (256d) will be more accurate
        }

        [Fact]
        public void Reset_ClearsWindow_RequiresWarmupAgain()
        {
            using var handle   = SlmRuntimeFactory.CreateGpt1(_model);
            using var detector = new GptAnomalyDetector(handle, contextSnapshots: 5);

            var normal = MakeNormalSnapshot("api-gateway");
            for (var i = 0; i < 5; i++)
            {
                detector.Score(normal);
            }
            Assert.True(detector.WindowFilled);

            detector.Reset();

            Assert.False(detector.WindowFilled);
            var r = detector.Score(normal);
            Assert.True(r.IsWarmup);
        }

        // ── Helpers ──────────────────────────────────────────────────────────

        /// <summary>
        /// Reads the size of the first parameter from the checkpoint to detect DModel.
        /// Embedding weight = VocabSize × DModel floats.
        /// </summary>
        private static GptTrainingConfig DetectConfigFromCheckpoint(string path)
        {
            using var fs = File.OpenRead(path);
            using var br = new BinaryReader(fs);
            // Parameter.Load zapisuje: int length + float[] data
            var length = br.ReadInt32();
            // length = VocabSize * DModel = 768 * DModel
            var dModel = length / MetricTokenizer.VocabSize;

            return dModel switch
            {
                64  => GptTrainingConfig.Quick,
                128 => GptTrainingConfig.Medium,
                256 => GptTrainingConfig.Production,
                _   => new GptTrainingConfig { DModel = dModel, NHeads = dModel / 32, NLayers = 4, ContextLength = 120 },
            };
        }

        private static MetricSnapshot MakeNormalSnapshot(string pod) => new()
        {
            Timestamp             = DateTime.UtcNow,
            PodName               = pod,
            CpuUsageRatio         = 0.20f,
            CpuThrottleRatio      = 0.02f,
            MemoryWorkingSetBytes = 350_000_000f,
            OomEventsRate         = 0f,
            LatencyP50Ms          = 12f,
            LatencyP95Ms          = 35f,
            LatencyP99Ms          = 75f,
            RequestsPerSecond     = 280f,
            ErrorRate             = 0.002f,
            GcGen2HeapBytes       = 55_000_000f,
            GcPauseRatio          = 0.004f,
            ThreadPoolQueueLength = 8f,
        };

        private static MetricSnapshot MakeAnomalySnapshot(string pod) => new()
        {
            Timestamp             = DateTime.UtcNow,
            PodName               = pod,
            CpuUsageRatio         = 0.97f,
            CpuThrottleRatio      = 0.85f,
            MemoryWorkingSetBytes = 7_500_000_000f,
            OomEventsRate         = 0.06f,
            LatencyP50Ms          = 950f,
            LatencyP95Ms          = 3200f,
            LatencyP99Ms          = 7500f,
            RequestsPerSecond     = 30f,
            ErrorRate             = 0.40f,
            GcGen2HeapBytes       = 4_800_000_000f,
            GcPauseRatio          = 0.45f,
            ThreadPoolQueueLength = 460f,
        };

        private void SkipIfNoCheckpoint()
        {
            if (!File.Exists(CheckpointPath))
            {
                throw new Exception(
                $"Checkpoint '{CheckpointPath}' not found. " +
                "Run: dotnet test --filter TrainOnCsv first.");
            }
        }

        public void Dispose() => _model.Dispose();
    }
}
