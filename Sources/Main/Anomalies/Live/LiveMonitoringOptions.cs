// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Adaptive;
using DevOnBike.Overfit.Anomalies.Gpt;

namespace DevOnBike.Overfit.Anomalies.Live
{
    /// <summary>Options for <see cref="LiveMonitoringPipeline"/>.</summary>
    public sealed class LiveMonitoringOptions
    {
        /// <summary>Prometheus HTTP API base URL, e.g. "http://prometheus:9090".</summary>
        public required string PrometheusBaseUrl
        {
            get; init;
        }

        /// <summary>PromQL regex matching pods to monitor, e.g. "my-service-.*".</summary>
        public required string PodRegex
        {
            get; init;
        }

        /// <summary>
        /// Per-pod adaptive policy — drives scoring window (<see cref="AdaptivePolicy.ContextSnapshots"/>),
        /// the false-positive-pressure band, and where per-pod adapters are stored. The pipeline
        /// scores every pod through an <see cref="AdaptiveAnomalyMonitor"/> built from this.
        /// NB: the FP-pressure band (<see cref="AdaptivePolicy.AlertThreshold"/> ..
        /// <see cref="AdaptivePolicy.CriticalThreshold"/>) is SEPARATE from the alert-engine
        /// thresholds below — a cross-pod base's benign can sit above the alert threshold yet
        /// still be adaptable, so the critical (real-incident) cutoff for adaptation is usually higher.
        /// </summary>
        public required AdaptivePolicy Adaptation
        {
            get; init;
        }

        public TimeSpan ScrapeInterval { get; init; } = TimeSpan.FromSeconds(15);

        /// <summary>
        /// Alerting thresholds on the RAW GPT surprise score (nats/token): a score ≥
        /// <see cref="AlertThreshold"/> fires a Warning, ≥ <see cref="CriticalThreshold"/> a Critical
        /// (per-pod cooldown applies). Independent of the adaptation band in <see cref="Adaptation"/>.
        /// </summary>
        public float AlertThreshold { get; init; } = 5.0f;
        public float CriticalThreshold { get; init; } = 10.0f;
        public TimeSpan CooldownDuration { get; init; } = TimeSpan.FromMinutes(5);

        /// <summary>
        /// When true, the pipeline auto-adapts pods the monitor recommends (unattended mode).
        /// Default false — adaptation is operator-gated (an elevated score can't be told from a
        /// moderate real incident online); use <see cref="OnAdaptationRecommended"/> + the
        /// pipeline's <c>Adapt</c> to review first.
        /// </summary>
        public bool AutoAdapt
        {
            get; init;
        }

        /// <summary>Called for every scored snapshot (logging, metrics export).</summary>
        public Action<AnomalyScore>? OnScore
        {
            get; init;
        }

        /// <summary>Called (once per scrape it newly applies) with the pod name when the monitor recommends adapting it.</summary>
        public Action<string>? OnAdaptationRecommended
        {
            get; init;
        }

        /// <summary>Called on non-cancellation errors in the scrape loop.</summary>
        public Action<Exception>? OnError
        {
            get; init;
        }
    }
}
