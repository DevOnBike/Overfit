// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Monitoring;

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>Immutable configuration for <c>PrometheusMetricSource</c> (instant queries).</summary>
    public sealed record PrometheusMetricSourceConfig : IPrometheusQuerySelector
    {
        /// <summary>Prometheus HTTP API base URL, e.g. "http://prometheus:9090".</summary>
        public required string PrometheusBaseUrl
        {
            get; init;
        }

        /// <inheritdoc/>
        public required string PodRegex
        {
            get; init;
        }

        /// <summary>
        /// Kubernetes namespace to restrict every query to. Empty means no namespace matcher — rarely what
        /// anyone wants on a shared cluster, where a pod regex alone will happily match somebody else's
        /// workload.
        /// </summary>
        public string Namespace { get; init; } = string.Empty;

        /// <summary>
        /// Label separating data centres. <b>Empty means a single-data-centre deployment</b>: no matcher is
        /// emitted and one set of queries is issued instead of two.
        ///
        /// <para>This has to be opt-out rather than opt-in. The label is a site-specific convention that
        /// nothing adds for you — kube-state-metrics and cAdvisor certainly do not — so a hard-coded
        /// <c>dc="west"</c> matcher silently reduces every query to the empty set on any cluster that has
        /// never heard of it. That failure is invisible: Prometheus returns success with no series, and a
        /// pipeline that filled missing features with zero could not tell it from a healthy quiet system.</para>
        /// </summary>
        public string DataCenterLabel { get; init; } = "dc";

        /// <inheritdoc/>
        public string DcWestLabel { get; init; } = "west";

        /// <inheritdoc/>
        public string DcEastLabel { get; init; } = "east";

        /// <inheritdoc/>
        public IReadOnlyDictionary<MetricIndex, string>? QueryOverrides
        {
            get; init;
        }

        /// <summary>
        ///     How often to scrape — should match Prometheus scrape_interval.
        ///     Default: 15 seconds.
        /// </summary>
        public TimeSpan ScrapeInterval { get; init; } = TimeSpan.FromSeconds(15);

        /// <summary>HTTP request timeout per query. Default: 5 seconds.</summary>
        public TimeSpan HttpTimeout { get; init; } = TimeSpan.FromSeconds(5);

        /// <summary>
        /// Configuration for a cluster running this project's own ASP.NET server — the shape the
        /// repository's <c>k8s/</c> lab stands up. Query set from
        /// <see cref="PromqlCatalog.OverfitServerQueries"/>, which documents its two approximations.
        /// </summary>
        public static PrometheusMetricSourceConfig ForOverfitServer(
            string prometheusBaseUrl,
            string podRegex,
            string namespaceName,
            TimeSpan? window = null)
        {
            return new PrometheusMetricSourceConfig
            {
                PrometheusBaseUrl = prometheusBaseUrl,
                PodRegex = podRegex,
                Namespace = namespaceName,
                DataCenterLabel = string.Empty,
                QueryOverrides = PromqlCatalog.OverfitServerQueries(window)
            };
        }
    }
}
