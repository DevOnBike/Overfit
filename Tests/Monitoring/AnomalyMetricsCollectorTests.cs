// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Monitoring;

namespace DevOnBike.Overfit.Tests.Monitoring
{
    public sealed class AnomalyMetricsCollectorTests
    {
        [Fact]
        public void RecordInference_WhenCalled_ThenSnapshotHasCorrectScore()
        {
            var collector = new AnomalyMetricsCollector();
            collector.RecordInference("pod-1", anomalyScore: 0.42f, reconstructionMse: 0.01f);

            var snap = collector.GetSnapshot("pod-1");
            Assert.NotNull(snap);
            Assert.Equal(0.42f, snap.AnomalyScore);
        }

        [Fact]
        public void RecordInference_WhenCalled_ThenSnapshotHasCorrectMse()
        {
            var collector = new AnomalyMetricsCollector();
            collector.RecordInference("pod-1", anomalyScore: 0.42f, reconstructionMse: 0.033f);

            var snap = collector.GetSnapshot("pod-1");
            Assert.NotNull(snap);
            Assert.Equal(0.033f, snap.ReconstructionMse);
        }

        [Fact]
        public void RecordInference_WhenCalledMultipleTimes_ThenWindowsProcessedIncrements()
        {
            var collector = new AnomalyMetricsCollector();
            collector.RecordInference("pod-1", 0.1f, 0.001f);
            collector.RecordInference("pod-1", 0.2f, 0.002f);
            collector.RecordInference("pod-1", 0.3f, 0.003f);

            var snap = collector.GetSnapshot("pod-1");
            Assert.Equal(3L, snap!.WindowsProcessed);
        }

        [Fact]
        public void RecordInference_WhenCalledTwice_ThenScoreIsOverwritten()
        {
            var collector = new AnomalyMetricsCollector();
            collector.RecordInference("pod-1", 0.1f, 0.001f);
            collector.RecordInference("pod-1", 0.9f, 0.05f);

            var snap = collector.GetSnapshot("pod-1");
            Assert.Equal(0.9f, snap!.AnomalyScore);
        }

        [Fact]
        public void RecordInference_WhenMultiplePods_ThenEachTrackedIndependently()
        {
            var collector = new AnomalyMetricsCollector();
            collector.RecordInference("pod-A", 0.1f, 0.001f);
            collector.RecordInference("pod-B", 0.9f, 0.05f);

            Assert.Equal(0.1f, collector.GetSnapshot("pod-A")!.AnomalyScore);
            Assert.Equal(0.9f, collector.GetSnapshot("pod-B")!.AnomalyScore);
            Assert.Equal(2, collector.TrackedPodCount);
        }

        // -------------------------------------------------------------------------
        // RecordAlert
        // -------------------------------------------------------------------------

        [Fact]
        public void RecordAlert_WhenCalled_ThenAlertsFiredIncrements()
        {
            var collector = new AnomalyMetricsCollector();
            collector.RecordInference("pod-1", 0.9f, 0.05f);
            collector.RecordAlert("pod-1");
            collector.RecordAlert("pod-1");

            Assert.Equal(2L, collector.GetSnapshot("pod-1")!.AlertsFired);
        }

        [Fact]
        public void RecordAlert_WhenCalledWithoutPriorInference_ThenStillTrackedPod()
        {
            var collector = new AnomalyMetricsCollector();
            collector.RecordAlert("pod-1");

            Assert.Equal(1L, collector.GetSnapshot("pod-1")!.AlertsFired);
        }

        // -------------------------------------------------------------------------
        // UpdateScorerThreshold
        // -------------------------------------------------------------------------

        [Fact]
        public void UpdateScorerThreshold_WhenCalled_ThenAppearsInPrometheusOutput()
        {
            var collector = new AnomalyMetricsCollector();
            collector.UpdateScorerThreshold(0.075f);

            var output = collector.FormatPrometheus();
            Assert.Contains("overfit_scorer_threshold", output);
            Assert.Contains("0.075", output);
        }

        [Fact]
        public void UpdateScorerThreshold_WhenZero_ThenNotIncludedInOutput()
        {
            var collector = new AnomalyMetricsCollector();
            // threshold defaults to 0 — should not appear

            var output = collector.FormatPrometheus();
            Assert.DoesNotContain("overfit_scorer_threshold", output);
        }

        // -------------------------------------------------------------------------
        // GetSnapshot
        // -------------------------------------------------------------------------

        [Fact]
        public void GetSnapshot_WhenPodNotRecorded_ThenReturnsNull()
        {
            var collector = new AnomalyMetricsCollector();
            Assert.Null(collector.GetSnapshot("nonexistent-pod"));
        }

        [Fact]
        public void GetSnapshot_WhenPodRecorded_ThenReturnsNonNull()
        {
            var collector = new AnomalyMetricsCollector();
            collector.RecordInference("pod-1", 0.5f, 0.02f);

            Assert.NotNull(collector.GetSnapshot("pod-1"));
        }

        [Fact]
        public void GetSnapshot_WhenCalled_ThenLastUpdatedIsRecent()
        {
            var before = DateTime.UtcNow;
            var collector = new AnomalyMetricsCollector();
            collector.RecordInference("pod-1", 0.5f, 0.02f);
            var after = DateTime.UtcNow;

            var snap = collector.GetSnapshot("pod-1")!;
            Assert.InRange(snap.LastUpdated, before, after);
        }

        // -------------------------------------------------------------------------
        // GetAllSnapshots
        // -------------------------------------------------------------------------

        [Fact]
        public void GetAllSnapshots_WhenNoPodsRecorded_ThenReturnsEmptyList()
        {
            var collector = new AnomalyMetricsCollector();
            Assert.Empty(collector.GetAllSnapshots());
        }

        [Fact]
        public void GetAllSnapshots_WhenThreePodsRecorded_ThenReturnsThreeSnapshots()
        {
            var collector = new AnomalyMetricsCollector();
            collector.RecordInference("pod-A", 0.1f, 0.01f);
            collector.RecordInference("pod-B", 0.5f, 0.02f);
            collector.RecordInference("pod-C", 0.9f, 0.05f);

            Assert.Equal(3, collector.GetAllSnapshots().Count);
        }

        // -------------------------------------------------------------------------
        // FormatPrometheus
        // -------------------------------------------------------------------------

        [Fact]
        public void FormatPrometheus_WhenNoData_ThenContainsNoMetricLines()
        {
            var collector = new AnomalyMetricsCollector();
            var output = collector.FormatPrometheus();

            // Headers only — no metric lines with pod labels
            Assert.DoesNotContain("{pod=", output);
        }

        [Fact]
        public void FormatPrometheus_WhenDataRecorded_ThenContainsAnomalyScoreMetric()
        {
            var collector = new AnomalyMetricsCollector();
            collector.RecordInference("pod-1", 0.42f, 0.01f);

            var output = collector.FormatPrometheus();
            Assert.Contains("overfit_anomaly_score{pod=\"pod-1\"}", output);
        }

        [Fact]
        public void FormatPrometheus_WhenDataRecorded_ThenContainsAllFourMetricFamilies()
        {
            var collector = new AnomalyMetricsCollector();
            collector.RecordInference("pod-1", 0.5f, 0.02f);
            collector.RecordAlert("pod-1");

            var output = collector.FormatPrometheus();

            Assert.Contains("overfit_anomaly_score", output);
            Assert.Contains("overfit_reconstruction_mse", output);
            Assert.Contains("overfit_windows_processed_total", output);
            Assert.Contains("overfit_alerts_fired_total", output);
        }

        [Fact]
        public void FormatPrometheus_WhenDataRecorded_ThenContainsHelpAndTypeLines()
        {
            var collector = new AnomalyMetricsCollector();
            collector.RecordInference("pod-1", 0.5f, 0.02f);

            var output = collector.FormatPrometheus();

            Assert.Contains("# HELP overfit_anomaly_score", output);
            Assert.Contains("# TYPE overfit_anomaly_score gauge", output);
            Assert.Contains("# TYPE overfit_windows_processed_total counter", output);
        }

        [Fact]
        public void FormatPrometheus_WhenMultiplePods_ThenEachPodHasLabel()
        {
            var collector = new AnomalyMetricsCollector();
            collector.RecordInference("pod-A", 0.1f, 0.001f);
            collector.RecordInference("pod-B", 0.9f, 0.05f);

            var output = collector.FormatPrometheus();

            Assert.Contains("pod=\"pod-A\"", output);
            Assert.Contains("pod=\"pod-B\"", output);
        }

        [Fact]
        public void FormatPrometheus_WhenCalled_ThenOutputContainsTimestamp()
        {
            var collector = new AnomalyMetricsCollector();
            collector.RecordInference("pod-1", 0.5f, 0.02f);

            var output = collector.FormatPrometheus();

            // Each metric line should end with a Unix ms timestamp (13 digits)
            var lines = output.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            var metricLine = lines.First(l => l.StartsWith("overfit_anomaly_score{"));
            var parts = metricLine.Split(' ');
            Assert.Equal(3, parts.Length); // name{label} value timestamp
            Assert.True(long.TryParse(parts[2], out _), $"Timestamp not parseable: {parts[2]}");
        }

        [Fact]
        public void FormatPrometheus_WhenCalledTwice_ThenOutputIsConsistent()
        {
            var collector = new AnomalyMetricsCollector();
            collector.RecordInference("pod-1", 0.5f, 0.02f);

            var output1 = collector.FormatPrometheus();
            var output2 = collector.FormatPrometheus();

            // Same score value in both calls (timestamps may differ)
            Assert.Contains("overfit_anomaly_score{pod=\"pod-1\"}", output1);
            Assert.Contains("overfit_anomaly_score{pod=\"pod-1\"}", output2);
        }

        // -------------------------------------------------------------------------
        // TrackedPodCount
        // -------------------------------------------------------------------------

        [Fact]
        public void TrackedPodCount_WhenNoPods_ThenIsZero()
            => Assert.Equal(0, new AnomalyMetricsCollector().TrackedPodCount);

        [Fact]
        public void TrackedPodCount_WhenSamePodRecordedTwice_ThenIsOne()
        {
            var collector = new AnomalyMetricsCollector();
            collector.RecordInference("pod-1", 0.1f, 0.001f);
            collector.RecordInference("pod-1", 0.2f, 0.002f);
            Assert.Equal(1, collector.TrackedPodCount);
        }
    }
}