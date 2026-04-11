// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com


// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Monitoring.Contracts
{
    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------

    /// <summary>Immutable configuration for <see cref="PrometheusHistoricalSource"/>.</summary>
    public sealed record PrometheusHistoricalSourceConfig
    {
        /// <summary>Prometheus HTTP API base URL, e.g. "http://prometheus:9090".</summary>
        public required string PrometheusBaseUrl { get; init; }

        /// <summary>K8s pod name used as the {pod} label filter in all PromQL queries.</summary>
        public required string PodName { get; init; }

        /// <summary>K8s namespace of the pod. Default: "default".</summary>
        public string Namespace { get; init; } = "default";

        /// <summary>Start of the historical range to fetch.</summary>
        public required DateTime RangeStart { get; init; }

        /// <summary>End of the historical range to fetch.</summary>
        public required DateTime RangeEnd { get; init; }

        /// <summary>
        /// Step between samples. Should match the original scrape_interval.
        /// Default: 10 seconds.
        /// </summary>
        public TimeSpan Step { get; init; } = TimeSpan.FromSeconds(10);

        /// <summary>HTTP request timeout per Prometheus query. Default: 30 seconds.</summary>
        public TimeSpan HttpTimeout { get; init; } = TimeSpan.FromSeconds(30);
    }

    // -------------------------------------------------------------------------
    // PrometheusHistoricalSource
    // -------------------------------------------------------------------------

}