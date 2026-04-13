// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Concurrent;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;
using DevOnBike.Overfit.Anomalies.Alerting;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    ///     Collects anomaly detection metrics and exposes them in
    ///     Prometheus text exposition format (v0.0.4).
    ///     Serve the output of <see cref="FormatPrometheus" /> from any HTTP endpoint
    ///     at the path your Prometheus scrape job targets (typically /metrics).
    ///     Metrics exposed:
    ///     overfit_anomaly_score{pod}           — current score ∈ [0, 1]  (gauge)
    ///     overfit_reconstruction_mse{pod}      — raw MSE                  (gauge)
    ///     overfit_windows_processed_total{pod} — windows processed        (counter)
    ///     overfit_alerts_fired_total{pod}      — alerts dispatched        (counter)
    ///     overfit_scorer_threshold             — calibrated MSE threshold (gauge)
    ///     Integration with <see cref="AlertEngine" />:
    ///     <code>
    ///   // Wire up after scoring:
    ///   var score = scorer.Score(features, reconstruction);
    ///   metrics.RecordInference(podName, score, mse);
    ///   alertEngine.TryAlert(podName, score, mse);
    ///   metrics.RecordAlert(podName, alertEngine.TryAlert(podName, score, mse));
    /// </code>
    ///     ASP.NET Core endpoint (minimal API example):
    ///     <code>
    ///   app.MapGet("/metrics", (AnomalyMetricsCollector m) =>
    ///       Results.Content(m.FormatPrometheus(), "text/plain; version=0.0.4"));
    /// </code>
    /// </summary>
    public sealed class AnomalyMetricsCollector
    {

        private readonly ConcurrentDictionary<string, PodState> _pods = new();

        // Global scorer threshold — not per-pod (shared model)
        private float _scorerThreshold;

        /// <summary>Number of pods currently being tracked.</summary>
        public int TrackedPodCount => _pods.Count;

        // -------------------------------------------------------------------------
        // Recording — called by the inference pipeline
        // -------------------------------------------------------------------------

        /// <summary>
        ///     Records the result of one inference window.
        ///     Thread-safe. Called on the scoring hot-path after each TryGetWindow.
        /// </summary>
        /// <param name="podName">K8s pod identifier.</param>
        /// <param name="anomalyScore">Score ∈ [0, 1] from <see cref="ReconstructionScorer" />.</param>
        /// <param name="reconstructionMse">Raw MSE for diagnostic purposes.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void RecordInference(string podName, float anomalyScore, float reconstructionMse)
        {
            var state = GetOrAddPodState(podName);

            Volatile.Write(ref state.AnomalyScore, anomalyScore);
            Volatile.Write(ref state.ReconstructionMse, reconstructionMse);
            Interlocked.Increment(ref state.WindowsProcessed);
            Volatile.Write(ref state.LastUpdatedTicks, DateTime.UtcNow.Ticks);
        }

        /// <summary>
        ///     Increments the alert counter for the given pod.
        ///     Call only when an alert was actually fired (TryAlert returned true).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void RecordAlert(string podName)
        {
            Interlocked.Increment(ref GetOrAddPodState(podName).AlertsFired);
        }

        /// <summary>
        ///     Updates the global scorer threshold gauge.
        ///     Call after <see cref="ReconstructionScorer.Calibrate" /> or
        ///     <see cref="OfflineTrainingJob.Run" /> completes.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void UpdateScorerThreshold(float threshold)
        {
            Volatile.Write(ref _scorerThreshold, threshold);
        }

        // -------------------------------------------------------------------------
        // Query
        // -------------------------------------------------------------------------

        /// <summary>Returns a snapshot of current metrics for the given pod.</summary>
        /// <returns>null if no data has been recorded for this pod yet.</returns>
        public PodMetricsSnapshot GetSnapshot(string podName)
        {
            if (!_pods.TryGetValue(podName, out var state))
            {
                return null;
            }

            return BuildSnapshot(podName, state);
        }

        /// <summary>Returns snapshots for all pods that have reported metrics.</summary>
        public IReadOnlyList<PodMetricsSnapshot> GetAllSnapshots()
        {
            var result = new List<PodMetricsSnapshot>(_pods.Count);

            foreach (var (name, state) in _pods)
            {
                result.Add(BuildSnapshot(name, state));
            }

            return result;
        }

        // -------------------------------------------------------------------------
        // Prometheus exposition
        // -------------------------------------------------------------------------

        /// <summary>
        ///     Formats all current metrics in Prometheus text exposition format v0.0.4.
        ///     The output is suitable for direct return from a /metrics HTTP endpoint.
        ///     Timestamps are Unix milliseconds (optional in Prometheus format, included
        ///     here for scrape-time accuracy).
        /// </summary>
        public string FormatPrometheus()
        {
            var sb = new StringBuilder(512);
            var pods = _pods.ToArray(); // snapshot for consistent formatting

            WriteMetricFamily(sb,
            "overfit_anomaly_score",
            "gauge",
            "Current anomaly score per monitored pod (0=normal, 1=anomaly threshold reached)");

            foreach (var (name, state) in pods)
            {
                WriteMetricLine(sb, "overfit_anomaly_score", name,
                Volatile.Read(ref state.AnomalyScore),
                Volatile.Read(ref state.LastUpdatedTicks));
            }

            WriteMetricFamily(sb,
            "overfit_reconstruction_mse",
            "gauge",
            "MSE between input features and autoencoder reconstruction");

            foreach (var (name, state) in pods)
            {
                WriteMetricLine(sb, "overfit_reconstruction_mse", name,
                Volatile.Read(ref state.ReconstructionMse),
                Volatile.Read(ref state.LastUpdatedTicks));
            }

            WriteMetricFamily(sb,
            "overfit_windows_processed_total",
            "counter",
            "Total number of feature windows processed by the anomaly detector");

            foreach (var (name, state) in pods)
            {
                WriteMetricLine(sb, "overfit_windows_processed_total", name,
                Interlocked.Read(ref state.WindowsProcessed),
                Volatile.Read(ref state.LastUpdatedTicks));
            }

            WriteMetricFamily(sb,
            "overfit_alerts_fired_total",
            "counter",
            "Total number of anomaly alerts dispatched");

            foreach (var (name, state) in pods)
            {
                WriteMetricLine(sb, "overfit_alerts_fired_total", name,
                Interlocked.Read(ref state.AlertsFired),
                Volatile.Read(ref state.LastUpdatedTicks));
            }

            // Global metric — no pod label
            var threshold = Volatile.Read(ref _scorerThreshold);

            if (threshold > 0f)
            {
                WriteMetricFamily(sb,
                "overfit_scorer_threshold",
                "gauge",
                "Calibrated anomaly threshold (MSE p99 from offline training)");

                var tsMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                sb.Append("overfit_scorer_threshold ")
                    .Append(threshold.ToString(CultureInfo.InvariantCulture))
                    .Append(' ').AppendLine(tsMs.ToString());
            }

            return sb.ToString();
        }

        // -------------------------------------------------------------------------
        // Private helpers
        // -------------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private PodState GetOrAddPodState(string podName)
        {
            return _pods.GetOrAdd(podName, valueFactory: _ => new PodState());
        }

        private static PodMetricsSnapshot BuildSnapshot(string podName, PodState state)
        {
            return new PodMetricsSnapshot
            {
                PodName = podName,
                AnomalyScore = Volatile.Read(ref state.AnomalyScore),
                ReconstructionMse = Volatile.Read(ref state.ReconstructionMse),
                WindowsProcessed = Interlocked.Read(ref state.WindowsProcessed),
                AlertsFired = Interlocked.Read(ref state.AlertsFired),
                LastUpdated = new DateTime(Volatile.Read(ref state.LastUpdatedTicks), DateTimeKind.Utc)
            };
        }

        private static void WriteMetricFamily(StringBuilder sb, string name, string type, string help)
        {
            sb.Append("# HELP ").Append(name).Append(' ').AppendLine(help);
            sb.Append("# TYPE ").Append(name).Append(' ').AppendLine(type);
        }

        private static void WriteMetricLine(StringBuilder sb, string name, string pod, float value, long ticks)
        {
            var tsMs = new DateTimeOffset(ticks, TimeSpan.Zero).ToUnixTimeMilliseconds();
            sb.Append(name).Append("{pod=\"").Append(pod).Append("\"} ")
                .Append(value.ToString(CultureInfo.InvariantCulture))
                .Append(' ').AppendLine(tsMs.ToString());
        }

        private static void WriteMetricLine(StringBuilder sb, string name, string pod, long value, long ticks)
        {
            var tsMs = new DateTimeOffset(ticks, TimeSpan.Zero).ToUnixTimeMilliseconds();
            sb.Append(name).Append("{pod=\"").Append(pod).Append("\"} ")
                .Append(value).Append(' ').AppendLine(tsMs.ToString());
        }

        // Per-pod state bucket — allocated once per pod on first observation
        private sealed class PodState
        {
            public long AlertsFired;
            public float AnomalyScore;
            public long LastUpdatedTicks = DateTime.UtcNow.Ticks;
            public float ReconstructionMse;
            public long WindowsProcessed;
        }
    }
}