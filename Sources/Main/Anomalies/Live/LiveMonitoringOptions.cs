// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Gpt;

namespace DevOnBike.Overfit.Anomalies.Live
{
    /// <summary>Options for <see cref="LiveMonitoringPipeline"/>.</summary>
    public sealed class LiveMonitoringOptions
    {
        /// <summary>Prometheus HTTP API base URL, e.g. "http://prometheus:9090".</summary>
        public required string PrometheusBaseUrl { get; init; }

        /// <summary>PromQL regex matching pods to monitor, e.g. "my-service-.*".</summary>
        public required string PodRegex { get; init; }

        public TimeSpan ScrapeInterval { get; init; } = TimeSpan.FromSeconds(15);
        public float AlertThreshold { get; init; } = 3.0f;
        public float CriticalThreshold { get; init; } = 6.0f;
        public TimeSpan CooldownDuration { get; init; } = TimeSpan.FromMinutes(5);

        /// <summary>
        /// Rolling window in snapshots fed to each detector.
        /// Default 21 = ~5 minutes at 15s scrape.
        /// </summary>
        public int ContextSnapshots { get; init; } = 21;

        /// <summary>Called for every scored snapshot (logging, metrics export).</summary>
        public Action<AnomalyScore>? OnScore { get; init; }

        /// <summary>Called on non-cancellation errors in the scrape loop.</summary>
        public Action<Exception>? OnError { get; init; }
    }
}
