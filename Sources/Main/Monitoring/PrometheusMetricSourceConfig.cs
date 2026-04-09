// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com
namespace DevOnBike.Overfit.Monitoring
{
    /// <summary>
    /// Immutable configuration for <see cref="PrometheusMetricSource"/>.
    /// </summary>
    public sealed record PrometheusMetricSourceConfig
    {
        /// <summary>
        /// Prometheus HTTP API base URL (e.g. "http://prometheus:9090").
        /// </summary>
        public required string PrometheusBaseUrl { get; init; }

        /// <summary>
        /// The K8s pod name to scrape metrics for.
        /// Used as the {pod} label filter in PromQL queries.
        /// </summary>
        public required string PodName { get; init; }

        /// <summary>
        /// K8s namespace of the pod. Used in PromQL label selectors.
        /// Default: "default".
        /// </summary>
        public string Namespace { get; init; } = "default";

        /// <summary>
        /// How long to wait between scrape attempts.
        /// Should align with the Prometheus scrape_interval. Default: 10 seconds.
        /// </summary>
        public TimeSpan ScrapeInterval { get; init; } = TimeSpan.FromSeconds(10);

        /// <summary>HTTP request timeout per Prometheus query. Default: 5 seconds.</summary>
        public TimeSpan HttpTimeout { get; init; } = TimeSpan.FromSeconds(5);
    }
}