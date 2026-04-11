// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Monitoring.Contracts
{
    /// <summary>Immutable configuration for PrometheusMetricSource.</summary>
    public sealed record PrometheusMetricSourceConfig
    {
        /// <summary>Prometheus HTTP API base URL, e.g. "http://prometheus:9090".</summary>
        public required string PrometheusBaseUrl { get; init; }

        /// <summary>
        ///     PromQL regex matching all pods to monitor, e.g. "my-service-.*".
        ///     Used as pod=~"{PodRegex}" in all instant queries.
        /// </summary>
        public required string PodRegex { get; init; }

        /// <summary>Prometheus dc label value for DataCenter.West.</summary>
        public string DcWestLabel { get; init; } = "west";

        /// <summary>Prometheus dc label value for DataCenter.East.</summary>
        public string DcEastLabel { get; init; } = "east";

        /// <summary>
        ///     How often to scrape — should match Prometheus scrape_interval.
        ///     Default: 15 seconds.
        /// </summary>
        public TimeSpan ScrapeInterval { get; init; } = TimeSpan.FromSeconds(15);

        /// <summary>HTTP request timeout per query. Default: 5 seconds.</summary>
        public TimeSpan HttpTimeout { get; init; } = TimeSpan.FromSeconds(5);
    }
}