// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Generic;

namespace DevOnBike.Overfit.Anomalies.Monitoring.Contracts
{
    /// <summary>Immutable configuration for <c>PrometheusMetricSource</c>.</summary>
    public sealed record PrometheusMetricSourceConfig
    {
        /// <summary>
        /// Placeholder every query template expands to the label matcher set — pod regex, namespace and data
        /// centre. A token rather than <c>string.Format</c> because PromQL is full of braces and every
        /// template would otherwise need them doubled, which is exactly the kind of quoting that produces a
        /// query that looks right and matches nothing.
        /// </summary>
        public const string SelectorToken = "%selector%";

        /// <summary>Prometheus HTTP API base URL, e.g. "http://prometheus:9090".</summary>
        public required string PrometheusBaseUrl
        {
            get; init;
        }

        /// <summary>
        ///     PromQL regex matching all pods to monitor, e.g. "my-service-.*".
        ///     Expands into <see cref="SelectorToken"/> as <c>pod=~"{PodRegex}"</c>.
        /// </summary>
        public required string PodRegex
        {
            get; init;
        }

        /// <summary>
        /// Kubernetes namespace to restrict every query to. Empty means no namespace matcher — which is
        /// rarely what anyone wants on a shared cluster, where a pod regex alone will happily match somebody
        /// else's workload.
        /// </summary>
        public string Namespace { get; init; } = string.Empty;

        /// <summary>
        /// Label separating data centres. <b>Empty means a single-data-centre deployment</b>: no <c>dc</c>
        /// matcher is emitted and one set of queries is issued instead of two.
        ///
        /// <para>This has to be opt-out rather than opt-in. The label is a site-specific convention that
        /// nothing adds for you — kube-state-metrics and cAdvisor certainly do not — so a hard-coded
        /// <c>dc="west"</c> matcher silently reduces every query to the empty set on any cluster that has
        /// never heard of it. That failure is invisible: Prometheus returns success with no series, and a
        /// pipeline that fills missing features with zero cannot tell it from a healthy quiet system.</para>
        /// </summary>
        public string DataCenterLabel { get; init; } = "dc";

        /// <summary>Value of <see cref="DataCenterLabel"/> for <c>DataCenter.West</c>.</summary>
        public string DcWestLabel { get; init; } = "west";

        /// <summary>Value of <see cref="DataCenterLabel"/> for <c>DataCenter.East</c>.</summary>
        public string DcEastLabel { get; init; } = "east";

        /// <summary>
        /// Per-metric PromQL overriding the built-in templates. Each value must contain
        /// <see cref="SelectorToken"/> and must produce a result carrying a <c>pod</c> label, since that is
        /// what attributes the sample. An <b>empty or whitespace</b> value marks the metric as having no
        /// source in this deployment: no query is issued and the feature is reported as missing rather than
        /// silently read as zero.
        /// </summary>
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

        /// <summary>Whether queries should omit the data-centre matcher and run once rather than per centre.</summary>
        public bool IsSingleDataCenter => string.IsNullOrWhiteSpace(DataCenterLabel);

        /// <summary>
        /// Configuration for a cluster running this project's own ASP.NET server, scraped by
        /// kube-prometheus-stack — the shape the repository's <c>k8s/</c> lab stands up.
        ///
        /// <para><b>Every template here was checked against a live Prometheus before being written down</b>,
        /// which is how the defaults were found to match nothing: the built-ins ask for OpenTelemetry names
        /// (<c>http_server_request_duration_seconds</c>, <c>process_runtime_dotnet_*</c>) that this server
        /// does not emit, and filter on a <c>dc</c> label the lab does not have.</para>
        ///
        /// <para><b>Two features are approximations, and the deviation is stated rather than hidden.</b>
        /// <c>CpuUsageRatio</c> is documented as a fraction of the pod's CPU limit, but a limit is optional in
        /// Kubernetes and three of the lab's four pods have none — so this returns cores consumed, and a
        /// consumer comparing it against a 0…1 expectation will be wrong. <c>GcGen2HeapBytes</c> maps to the
        /// whole managed heap, because the server exposes <c>dotnet_gc_heap_size_bytes</c> without a
        /// generation label.</para>
        ///
        /// <para><c>sum by (pod)</c> wraps the cAdvisor queries deliberately: cAdvisor emits one series per
        /// container plus a pod-level aggregate, and without the aggregation a single pod arrives as several
        /// samples that overwrite each other in feature assembly.</para>
        /// </summary>
        /// <param name="prometheusBaseUrl">e.g. <c>http://127.0.0.1:9090</c>.</param>
        /// <param name="podRegex">e.g. <c>overfit-server-.*</c>.</param>
        /// <param name="namespaceName">Kubernetes namespace, e.g. <c>overfit</c>.</param>
        /// <param name="window">Range used by every <c>rate()</c>. Should be at least four scrape intervals,
        /// or a rate over a restart-truncated window reads as a spike.</param>
        public static PrometheusMetricSourceConfig ForOverfitServer(
            string prometheusBaseUrl,
            string podRegex,
            string namespaceName,
            TimeSpan? window = null)
        {
            var range = FormatRange(window ?? TimeSpan.FromMinutes(2));
            var s = SelectorToken;

            var queries = new Dictionary<MetricIndex, string>(MetricSnapshotFeatures)
            {
                [MetricIndex.CpuUsageRatio] =
                    $"sum by (pod) (rate(container_cpu_usage_seconds_total{{{s}}}[{range}]))",

                // Present only on pods that carry a CPU limit — CFS accounting does not exist without a
                // quota. Pods without a limit produce no series at all, which is not the same as zero and is
                // why the source reports per-metric coverage.
                [MetricIndex.CpuThrottleRatio] =
                    $"sum by (pod) (rate(container_cpu_cfs_throttled_periods_total{{{s}}}[{range}]))"
                    + $" / sum by (pod) (rate(container_cpu_cfs_periods_total{{{s}}}[{range}]))",

                [MetricIndex.MemoryWorkingSetBytes] =
                    $"sum by (pod) (container_memory_working_set_bytes{{{s}}})",

                [MetricIndex.OomEventsRate] =
                    $"sum by (pod) (rate(container_oom_events_total{{{s}}}[{range}]))",

                // sum by (pod, le) keeps the pod label through the quantile; without it the result carries no
                // pod and every sample is discarded during parsing.
                [MetricIndex.LatencyP50Ms] = LatencyQuery(0.50, s, range),
                [MetricIndex.LatencyP95Ms] = LatencyQuery(0.95, s, range),
                [MetricIndex.LatencyP99Ms] = LatencyQuery(0.99, s, range),

                [MetricIndex.RequestsPerSecond] =
                    $"sum by (pod) (rate(overfit_chat_requests_total{{{s}}}[{range}]))",

                // Undefined while no requests are arriving, and left undefined on purpose: 0/0 yields NaN,
                // which is the honest answer. Substituting zero would report a perfect error rate for a
                // server that is not serving anything.
                [MetricIndex.ErrorRate] =
                    $"sum by (pod) (rate(overfit_http_responses_total{{{s},status=\"5xx\"}}[{range}]))"
                    + $" / sum by (pod) (rate(overfit_http_responses_total{{{s}}}[{range}]))",

                [MetricIndex.GcGen2HeapBytes] =
                    $"sum by (pod) (dotnet_gc_heap_size_bytes{{{s}}})",

                [MetricIndex.GcPauseRatio] =
                    $"sum by (pod) (rate(dotnet_gc_pause_seconds_total{{{s}}}[{range}]))",

                [MetricIndex.ThreadPoolQueueLength] =
                    $"sum by (pod) (dotnet_threadpool_queue_length{{{s}}})"
            };

            return new PrometheusMetricSourceConfig
            {
                PrometheusBaseUrl = prometheusBaseUrl,
                PodRegex = podRegex,
                Namespace = namespaceName,
                DataCenterLabel = string.Empty,
                QueryOverrides = queries
            };
        }

        private static Dictionary<MetricIndex, string> MetricSnapshotFeatures => new((int)MetricIndex.Count);

        private static string LatencyQuery(double quantile, string selector, string range)
        {
            return $"histogram_quantile({quantile.ToString(System.Globalization.CultureInfo.InvariantCulture)},"
                   + $" sum by (pod, le) (rate(overfit_chat_response_time_seconds_bucket{{{selector}}}[{range}])))"
                   + " * 1000";
        }

        /// <summary>Whole seconds or minutes, because PromQL ranges take no fractional units.</summary>
        private static string FormatRange(TimeSpan window)
        {
            var seconds = (long)window.TotalSeconds;

            if (seconds <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(window), window, "Range must be positive.");
            }

            if (seconds % 60 == 0)
            {
                return $"{seconds / 60}m";
            }

            return $"{seconds}s";
        }
    }
}
