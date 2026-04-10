// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Monitoring;
using DevOnBike.Overfit.Monitoring.Abstractions;
using DevOnBike.Overfit.Monitoring.Contracts;

namespace DevOnBike.Overfit.Tests.Monitoring
{
    public sealed class AnomalyDetectionPipelineTests
    {
        // -------------------------------------------------------------------------
        // Test doubles
        // -------------------------------------------------------------------------

        /// <summary>
        /// Produces exactly <c>count</c> snapshots then blocks until cancellation.
        /// scrapeIntervalMs=0 → instant ValueTask.FromResult, no Task allocation.
        /// </summary>
        private sealed class FiniteMetricSource(
            int count,
            string podName = "test-pod",
            float cpuUsage = 0.3f) : IMetricSource
        {
            private int _remaining = count;

            public string PodName => podName;

            public ValueTask<MetricSnapshot> ReadAsync(CancellationToken ct = default)
            {
                if (Interlocked.Decrement(ref _remaining) >= 0)
                {
                    return ValueTask.FromResult(new MetricSnapshot
                    {
                        Timestamp = DateTime.UtcNow,
                        PodName = podName,
                        CpuUsageRatio = cpuUsage,
                        CpuThrottleRatio = 0.02f,
                        MemoryWorkingSetBytes = 400_000_000f,
                        OomEventsRate = 0f,
                        LatencyP50Ms = 60f,
                        LatencyP95Ms = 120f,
                        LatencyP99Ms = 200f,
                        RequestsPerSecond = 100f,
                        ErrorRate = 0.01f,
                        GcGen2HeapBytes = 50_000_000f,
                        GcPauseRatio = 0.02f,
                        ThreadPoolQueueLength = 10f
                    });
                }

                // All snapshots consumed — block until cancellation
                return new ValueTask<MetricSnapshot>(
                Task.Delay(Timeout.Infinite, ct)
                    .ContinueWith<MetricSnapshot>(_ =>
                        throw new OperationCanceledException(ct), ct));
            }

            public void Dispose() { }
        }

        private sealed class CapturingAlertSink : IAlertSink
        {
            private readonly List<AlertEvent> _events = [];
            public IReadOnlyList<AlertEvent> Events => _events;

            public Task SendAsync(AlertEvent alert, CancellationToken ct = default)
            {
                lock (_events) { _events.Add(alert); }
                return Task.CompletedTask;
            }
        }

        // -------------------------------------------------------------------------
        // Helpers
        // -------------------------------------------------------------------------

        private static AnomalyAutoencoder MakeAutoencoder(int inputSize = 48)
        {
            var m = new AnomalyAutoencoder(inputSize, hidden1: 12, hidden2: 6, bottleneckDim: 3);
            m.Eval();
            return m;
        }

        private static ReconstructionScorer MakeCalibratedScorer(float threshold = 1.0f)
        {
            var scorer = new ReconstructionScorer();
            scorer.Calibrate([threshold]);
            return scorer;
        }

        private static SlidingWindowBuffer MakeBuffer(int windowSize = 6)
            => new(windowSize, 1, MetricSnapshot.FeatureCount);

        // featureCount × StatsPerFeature = 12 × 4 = 48 → matches default autoencoder inputSize
        private const int InputSize = MetricSnapshot.FeatureCount * FeatureExtractor.StatsPerFeature; // 48

        // -------------------------------------------------------------------------
        // Constructor — argument validation
        // -------------------------------------------------------------------------

        [Fact]
        public void Constructor_WhenSourceIsNull_ThenThrowsArgumentNullException()
        {
            using var buffer = MakeBuffer();
            using var autoencoder = MakeAutoencoder();
            var scorer = MakeCalibratedScorer();

            Assert.Throws<ArgumentNullException>(
            () => new AnomalyDetectionPipeline(null!, buffer, autoencoder, scorer));
        }

        [Fact]
        public void Constructor_WhenBufferIsNull_ThenThrowsArgumentNullException()
        {
            using var autoencoder = MakeAutoencoder();
            var source = new FiniteMetricSource(1);
            var scorer = MakeCalibratedScorer();

            Assert.Throws<ArgumentNullException>(
            () => new AnomalyDetectionPipeline(source, null!, autoencoder, scorer));
        }

        [Fact]
        public void Constructor_WhenAutoencoderIsNull_ThenThrowsArgumentNullException()
        {
            using var buffer = MakeBuffer();
            var source = new FiniteMetricSource(1);
            var scorer = MakeCalibratedScorer();

            Assert.Throws<ArgumentNullException>(
            () => new AnomalyDetectionPipeline(source, buffer, null!, scorer));
        }

        [Fact]
        public void Constructor_WhenScorerIsNull_ThenThrowsArgumentNullException()
        {
            using var buffer = MakeBuffer();
            using var autoencoder = MakeAutoencoder();
            var source = new FiniteMetricSource(1);

            Assert.Throws<ArgumentNullException>(
            () => new AnomalyDetectionPipeline(source, buffer, autoencoder, null!));
        }

        // -------------------------------------------------------------------------
        // Create factory
        // -------------------------------------------------------------------------

        [Fact]
        public void Create_WhenConfigIsNull_ThenThrowsArgumentNullException()
        {
            using var autoencoder = MakeAutoencoder();
            var source = new FiniteMetricSource(1);
            var scorer = MakeCalibratedScorer();

            Assert.Throws<ArgumentNullException>(
            () => AnomalyDetectionPipeline.Create(source, null!, autoencoder, scorer));
        }

        [Fact]
        public async Task Create_WhenCalled_ThenPipelineDisposesOwnedBuffer()
        {
            using var autoencoder = MakeAutoencoder();
            var source = new FiniteMetricSource(0);
            var scorer = MakeCalibratedScorer();
            var config = new AnomalyDetectionPipelineConfig
            {
                WindowSize = 3
            };

            var pipeline = AnomalyDetectionPipeline.Create(source, config, autoencoder, scorer);

            using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(100));
            await pipeline.RunAsync(cts.Token);

            await pipeline.DisposeAsync(); // must not throw — buffer is owned
        }

        // -------------------------------------------------------------------------
        // RunAsync — completion
        // -------------------------------------------------------------------------

        [Fact]
        public async Task RunAsync_WhenCancelled_ThenCompletesWithoutThrowingOce()
        {
            var source = new FiniteMetricSource(0); // no snapshots — immediately waits
            using var buffer = MakeBuffer();
            using var autoencoder = MakeAutoencoder();
            var scorer = MakeCalibratedScorer();
            var pipeline = new AnomalyDetectionPipeline(source, buffer, autoencoder, scorer);

            using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(50));
            await pipeline.RunAsync(cts.Token); // must return normally, not throw
        }

        [Fact]
        public async Task RunAsync_WhenSourceExhausted_ThenCompletesAfterCancellation()
        {
            var source = new FiniteMetricSource(count: 3);
            using var buffer = MakeBuffer(windowSize: 6);
            using var autoencoder = MakeAutoencoder();
            var scorer = MakeCalibratedScorer();
            var pipeline = new AnomalyDetectionPipeline(source, buffer, autoencoder, scorer);

            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(2));
            await pipeline.RunAsync(cts.Token);
            // Cancels after source delivers 3 samples then blocks — test should not hang
        }

        // -------------------------------------------------------------------------
        // RunAsync — WindowsProcessed counter
        // -------------------------------------------------------------------------

        [Fact]
        public async Task RunAsync_WhenFewerSamplesThanWindowSize_ThenWindowsProcessedIsZero()
        {
            const int windowSize = 6;
            var source = new FiniteMetricSource(count: windowSize - 1); // not enough
            using var buffer = MakeBuffer(windowSize);
            using var autoencoder = MakeAutoencoder();
            var scorer = MakeCalibratedScorer();
            var pipeline = new AnomalyDetectionPipeline(source, buffer, autoencoder, scorer);

            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(2));
            await pipeline.RunAsync(cts.Token);

            Assert.Equal(0L, pipeline.WindowsProcessed);
        }

        [Fact]
        public async Task RunAsync_WhenExactlyWindowSizeSamples_ThenWindowsProcessedIsOne()
        {
            const int windowSize = 6;
            var source = new FiniteMetricSource(count: windowSize);
            using var buffer = MakeBuffer(windowSize);
            using var autoencoder = MakeAutoencoder();
            var scorer = MakeCalibratedScorer();
            var pipeline = new AnomalyDetectionPipeline(source, buffer, autoencoder, scorer);

            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(2));
            await pipeline.RunAsync(cts.Token);

            Assert.Equal(1L, pipeline.WindowsProcessed);
        }

        [Fact]
        public async Task RunAsync_WhenNSamples_ThenWindowsProcessedMatchesExpected()
        {
            const int windowSize = 4;
            const int totalSamples = 10;
            var expectedWindows = (long)(totalSamples - (windowSize - 1)); // = 7

            var source = new FiniteMetricSource(count: totalSamples);
            using var buffer = new SlidingWindowBuffer(windowSize, 1, MetricSnapshot.FeatureCount);
            using var autoencoder = MakeAutoencoder();
            var scorer = MakeCalibratedScorer();
            var pipeline = new AnomalyDetectionPipeline(source, buffer, autoencoder, scorer);

            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(2));
            await pipeline.RunAsync(cts.Token);

            Assert.Equal(expectedWindows, pipeline.WindowsProcessed);
        }

        // -------------------------------------------------------------------------
        // RunAsync — onInference callback
        // -------------------------------------------------------------------------

        [Fact]
        public async Task RunAsync_WhenWindowReady_ThenOnInferenceIsInvoked()
        {
            const int windowSize = 3;
            var source = new FiniteMetricSource(count: windowSize);
            using var buffer = MakeBuffer(windowSize);
            using var autoencoder = MakeAutoencoder();
            var scorer = MakeCalibratedScorer();
            var results = new List<InferenceResult>();
            var pipeline = new AnomalyDetectionPipeline(source, buffer, autoencoder, scorer);

            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(2));
            await pipeline.RunAsync(cts.Token, onInference: r => {
                lock (results) results.Add(r);
            });

            Assert.Single(results);
        }

        [Fact]
        public async Task RunAsync_WhenWindowReady_ThenInferenceResultHasCorrectPodName()
        {
            const int windowSize = 3;
            const string podName = "my-special-pod";
            var source = new FiniteMetricSource(count: windowSize, podName: podName);
            using var buffer = MakeBuffer(windowSize);
            using var autoencoder = MakeAutoencoder();
            var scorer = MakeCalibratedScorer();
            InferenceResult? captured = null;
            var pipeline = new AnomalyDetectionPipeline(source, buffer, autoencoder, scorer);

            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(2));
            await pipeline.RunAsync(cts.Token, onInference: r => captured = r);

            Assert.NotNull(captured);
            Assert.Equal(podName, captured.PodName);
        }

        [Fact]
        public async Task RunAsync_WhenWindowReady_ThenInferenceResultHasFiniteScore()
        {
            const int windowSize = 3;
            var source = new FiniteMetricSource(count: windowSize);
            using var buffer = MakeBuffer(windowSize);
            using var autoencoder = MakeAutoencoder();
            var scorer = MakeCalibratedScorer();
            InferenceResult? captured = null;
            var pipeline = new AnomalyDetectionPipeline(source, buffer, autoencoder, scorer);

            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(2));
            await pipeline.RunAsync(cts.Token, onInference: r => captured = r);

            Assert.NotNull(captured);
            Assert.True(float.IsFinite(captured.AnomalyScore), $"Score={captured.AnomalyScore}");
            Assert.True(float.IsFinite(captured.ReconstructionMse), $"MSE={captured.ReconstructionMse}");
            Assert.True(captured.AnomalyScore >= 0f && captured.AnomalyScore <= 1f);
        }

        [Fact]
        public async Task RunAsync_WhenWindowReady_ThenWindowEndIsRecent()
        {
            const int windowSize = 3;
            var before = DateTime.UtcNow;
            var source = new FiniteMetricSource(count: windowSize);
            using var buffer = MakeBuffer(windowSize);
            using var autoencoder = MakeAutoencoder();
            var scorer = MakeCalibratedScorer();
            InferenceResult? captured = null;
            var pipeline = new AnomalyDetectionPipeline(source, buffer, autoencoder, scorer);

            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(2));
            await pipeline.RunAsync(cts.Token, onInference: r => captured = r);
            var after = DateTime.UtcNow;

            Assert.NotNull(captured);
            Assert.True(captured.WindowEnd >= before && captured.WindowEnd <= after);
        }

        // -------------------------------------------------------------------------
        // RunAsync — metrics integration
        // -------------------------------------------------------------------------

        [Fact]
        public async Task RunAsync_WhenWindowReady_ThenMetricsRecordsInference()
        {
            const int windowSize = 3;
            var source = new FiniteMetricSource(count: windowSize, podName: "pod-x");
            using var buffer = MakeBuffer(windowSize);
            using var autoencoder = MakeAutoencoder();
            var scorer = MakeCalibratedScorer();
            var metrics = new AnomalyMetricsCollector();
            var pipeline = new AnomalyDetectionPipeline(source, buffer, autoencoder, scorer,
            metrics: metrics);

            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(2));
            await pipeline.RunAsync(cts.Token);

            var snap = metrics.GetSnapshot("pod-x");
            Assert.NotNull(snap);
            Assert.Equal(1L, snap.WindowsProcessed);
        }

        // -------------------------------------------------------------------------
        // RunAsync — alert integration
        // -------------------------------------------------------------------------

        [Fact]
        public async Task RunAsync_WhenScoreExceedsThreshold_ThenAlertFiredCounterIncrements()
        {
            const int windowSize = 3;
            // Calibrate scorer with tiny threshold so every MSE triggers an alert
            var scorer = new ReconstructionScorer();
            scorer.Calibrate([float.Epsilon]); // threshold ≈ 0 → any MSE → score=1

            var alertSink = new CapturingAlertSink();
            var config = new AlertEngineConfig
            {
                AlertThreshold = 0.01f, // very low — will trigger
                CriticalThreshold = 0.99f,
                CooldownDuration = TimeSpan.Zero
            };

            await using var alertEngine = new AlertEngine(config, alertSink);

            var source = new FiniteMetricSource(count: windowSize);
            using var buffer = MakeBuffer(windowSize);
            using var autoencoder = MakeAutoencoder();
            var pipeline = new AnomalyDetectionPipeline(
            source, buffer, autoencoder, scorer, alertEngine: alertEngine);

            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(2));
            await pipeline.RunAsync(cts.Token);

            Assert.Equal(1L, pipeline.AlertsFired);
        }

        [Fact]
        public async Task RunAsync_WhenScoreBelowThreshold_ThenAlertsFiredIsZero()
        {
            const int windowSize = 3;
            // Calibrate scorer with very high threshold so no alert fires
            var scorer = new ReconstructionScorer();
            scorer.Calibrate([float.MaxValue]); // threshold = MaxValue → score always near 0

            var config = new AlertEngineConfig
            {
                AlertThreshold = 0.99f, // very high — won't trigger
                CriticalThreshold = 1.0f,
                CooldownDuration = TimeSpan.Zero
            };

            await using var alertEngine = new AlertEngine(config, new CapturingAlertSink());

            var source = new FiniteMetricSource(count: windowSize);
            using var buffer = MakeBuffer(windowSize);
            using var autoencoder = MakeAutoencoder();
            var pipeline = new AnomalyDetectionPipeline(
            source, buffer, autoencoder, scorer, alertEngine: alertEngine);

            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(2));
            await pipeline.RunAsync(cts.Token);

            Assert.Equal(0L, pipeline.AlertsFired);
        }

        // -------------------------------------------------------------------------
        // InferenceResult record
        // -------------------------------------------------------------------------

        [Fact]
        public void InferenceResult_WhenCreated_ThenPropertiesAreAccessible()
        {
            var ts = DateTime.UtcNow;
            var r = new InferenceResult
            {
                PodName = "pod-1",
                AnomalyScore = 0.5f,
                ReconstructionMse = 0.02f,
                WindowEnd = ts
            };

            Assert.Equal("pod-1", r.PodName);
            Assert.Equal(0.5f, r.AnomalyScore);
            Assert.Equal(0.02f, r.ReconstructionMse);
            Assert.Equal(ts, r.WindowEnd);
        }

        // -------------------------------------------------------------------------
        // AnomalyDetectionPipelineConfig defaults
        // -------------------------------------------------------------------------

        [Fact]
        public void Config_WhenDefaultConfig_ThenWindowSizeIs6()
            => Assert.Equal(6, new AnomalyDetectionPipelineConfig().WindowSize);

        [Fact]
        public void Config_WhenDefaultConfig_ThenStepSizeIs1()
            => Assert.Equal(1, new AnomalyDetectionPipelineConfig().StepSize);

        [Fact]
        public void Config_WhenDefaultConfig_ThenFeatureCountEqualsMetricSnapshotFeatureCount()
            => Assert.Equal(MetricSnapshot.FeatureCount, new AnomalyDetectionPipelineConfig().FeatureCount);
    }
}