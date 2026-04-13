// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Monitoring.Contracts
{
    /// <summary>Immutable configuration for PrometheusHistoricalSource.</summary>
    public sealed record PrometheusHistoricalSourceConfig
    {
        /// <summary>Prometheus HTTP API base URL, e.g. "http://prometheus:9090".</summary>
        public required string PrometheusBaseUrl { get; init; }

        /// <summary>
        ///     PromQL regex matching all pods to monitor, e.g. "my-service-.*".
        ///     Used as pod=~"{PodRegex}" in all queries.
        /// </summary>
        public required string PodRegex { get; init; }

        /// <summary>Prometheus dc label value for DataCenter.West.</summary>
        public string DcWestLabel { get; init; } = "west";

        /// <summary>Prometheus dc label value for DataCenter.East.</summary>
        public string DcEastLabel { get; init; } = "east";

        /// <summary>Start of the Golden Window to fetch.</summary>
        public required DateTime RangeStart { get; init; }

        /// <summary>End of the Golden Window to fetch.</summary>
        public required DateTime RangeEnd { get; init; }

        /// <summary>
        ///     Step between samples — must match Prometheus scrape_interval.
        ///     Default: 15 seconds.
        /// </summary>
        public TimeSpan Step { get; init; } = TimeSpan.FromSeconds(15);

        /// <summary>HTTP request timeout per query. Default: 30 seconds.</summary>
        public TimeSpan HttpTimeout { get; init; } = TimeSpan.FromSeconds(30);
    }
}