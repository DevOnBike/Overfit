// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Monitoring.Contracts
{
    /// <summary>Immutable configuration for <c>PrometheusHistoricalSource</c> (range queries).</summary>
    public sealed record PrometheusHistoricalSourceConfig : IPrometheusQuerySelector
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

        /// <summary>Kubernetes namespace to restrict every query to; empty for no namespace matcher.</summary>
        public string Namespace { get; init; } = string.Empty;

        /// <summary>
        /// Label separating data centres. <b>Empty means a single-data-centre deployment</b>: no matcher, one
        /// pass instead of two. See <see cref="PrometheusMetricSourceConfig.DataCenterLabel"/> for why this is
        /// opt-out — the same hard-coded matcher silently emptied every query in this class too.
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

        /// <summary>Start of the Golden Window to fetch.</summary>
        public required DateTime RangeStart
        {
            get; init;
        }

        /// <summary>End of the Golden Window to fetch.</summary>
        public required DateTime RangeEnd
        {
            get; init;
        }

        /// <summary>
        ///     Step between samples — must match Prometheus scrape_interval.
        ///     Default: 15 seconds.
        /// </summary>
        public TimeSpan Step { get; init; } = TimeSpan.FromSeconds(15);

        /// <summary>HTTP request timeout per query. Default: 30 seconds.</summary>
        public TimeSpan HttpTimeout { get; init; } = TimeSpan.FromSeconds(30);

        /// <summary>
        /// Configuration for a cluster running this project's own ASP.NET server. Same query set as the
        /// instant source — one catalog, so the two cannot drift apart the way they already did once.
        /// </summary>
        /// <param name="prometheusBaseUrl">e.g. <c>http://127.0.0.1:9090</c>.</param>
        /// <param name="podRegex">e.g. <c>overfit-server-.*</c>.</param>
        /// <param name="namespaceName">Kubernetes namespace, e.g. <c>overfit</c>.</param>
        /// <param name="rangeStart">Start of the window to fetch.</param>
        /// <param name="rangeEnd">End of it.</param>
        /// <param name="step">Spacing between returned samples; should match the Prometheus scrape interval.</param>
        /// <param name="window">Range used by every <c>rate()</c> <i>inside</i> a sample. Distinct from
        /// <paramref name="step"/>, which is how far apart the samples are: a rate window shorter than a
        /// couple of steps produces gaps, and one much longer smooths away what the detectors look for.</param>
        public static PrometheusHistoricalSourceConfig ForOverfitServer(
            string prometheusBaseUrl,
            string podRegex,
            string namespaceName,
            DateTime rangeStart,
            DateTime rangeEnd,
            TimeSpan? step = null,
            TimeSpan? window = null)
        {
            var resolvedStep = step ?? TimeSpan.FromSeconds(30);

            return new PrometheusHistoricalSourceConfig
            {
                PrometheusBaseUrl = prometheusBaseUrl,
                PodRegex = podRegex,
                Namespace = namespaceName,
                DataCenterLabel = string.Empty,
                QueryOverrides = PromqlCatalog.OverfitServerQueries(window ?? TimeSpan.FromMinutes(2)),
                RangeStart = rangeStart,
                RangeEnd = rangeEnd,
                Step = resolvedStep
            };
        }
    }
}
