// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text;
using DevOnBike.Overfit.Anomalies.Contracts;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    /// The single place a <see cref="MetricIndex"/> becomes PromQL. Both
    /// <see cref="PrometheusMetricSource"/> (instant) and <see cref="PrometheusHistoricalSource"/> (range)
    /// go through here, so a metric name or a label matcher exists exactly once.
    ///
    /// <para><b>Metric names are a property of whoever exports them, not a contract.</b> The built-in
    /// templates use OpenTelemetry naming and are a starting point; any deployment whose exporter disagrees
    /// supplies <see cref="IPrometheusQuerySelector.QueryOverrides"/>. <see cref="OverfitServerQueries"/> is
    /// the worked example, and every template in it was run against a live Prometheus before being written
    /// down — which is how the built-ins were found to match nothing here.</para>
    /// </summary>
    public static class PromqlCatalog
    {
        /// <summary>
        /// Placeholder every template expands to the label matcher set. A token rather than
        /// <c>string.Format</c> because PromQL is full of braces and every template would otherwise need them
        /// doubled — exactly the kind of quoting that produces a query which looks right and matches nothing.
        /// </summary>
        public const string SelectorToken = "%selector%";

        /// <summary>
        /// The label matcher set every template expands <see cref="SelectorToken"/> to. Built once per data
        /// centre and shared by all twelve queries, so a namespace or pod-regex mistake is wrong everywhere at
        /// once rather than in eleven places out of twelve.
        /// </summary>
        public static string BuildSelector(IPrometheusQuerySelector selector, DataCenter dc)
        {
            ArgumentNullException.ThrowIfNull(selector);

            var sb = new StringBuilder(96);
            sb.Append("pod=~\"").Append(selector.PodRegex).Append('"');

            if (selector.Namespace.Length > 0)
            {
                sb.Append(",namespace=\"").Append(selector.Namespace).Append('"');
            }

            if (IsSingleDataCenter(selector))
            {
                return sb.ToString();
            }

            var value = dc == DataCenter.West ? selector.DcWestLabel : selector.DcEastLabel;
            sb.Append(',').Append(selector.DataCenterLabel).Append("=\"").Append(value).Append('"');

            return sb.ToString();
        }

        /// <summary>
        /// Label matchers for series that carry no <c>pod</c> label — ReplicaSet and Deployment metadata. Namespace
        /// and data centre still apply; the pod regex cannot.
        /// </summary>
        public static string BuildNamespaceSelector(IPrometheusQuerySelector selector)
        {
            ArgumentNullException.ThrowIfNull(selector);

            var sb = new StringBuilder(64);

            if (selector.Namespace.Length > 0)
            {
                sb.Append("namespace=\"").Append(selector.Namespace).Append('"');
            }

            if (IsSingleDataCenter(selector))
            {
                return sb.ToString();
            }

            if (sb.Length > 0)
            {
                sb.Append(',');
            }

            sb.Append(selector.DataCenterLabel).Append("=\"").Append(selector.DcWestLabel).Append('"');

            return sb.ToString();
        }

        /// <summary>
        /// <b>Pod to ReplicaSet.</b> On a Deployment the ReplicaSet is the version, so this is the query that makes
        /// a rollout visible — and it is the only thing standing between the guard and going blind through one.
        ///
        /// <para>The answer is in the series' <c>owner_name</c> <b>label</b>, not its value: kube-state-metrics
        /// emits a constant 1 and carries the relationship in labels. A reader for these has to keep the labels,
        /// which is why the metric sources cannot be reused as-is — they parse for a value.</para>
        /// </summary>
        public static string PodOwnershipQuery(IPrometheusQuerySelector selector)
        {
            var matchers = BuildSelector(selector, DataCenter.West);

            return $"kube_pod_owner{{{matchers},owner_kind=\"ReplicaSet\"}}";
        }

        /// <summary><b>ReplicaSet to Deployment</b>, closing the chain from a pod up to the workload a human names.</summary>
        public static string ReplicaSetOwnershipQuery(IPrometheusQuerySelector selector)
        {
            var matchers = BuildNamespaceSelector(selector);

            return matchers.Length > 0
                ? $"kube_replicaset_owner{{{matchers}}}"
                : "kube_replicaset_owner";
        }

        /// <summary>
        /// <b>Pod to node.</b> Fills the coordinate that catches the fault nobody looks for — a failing node
        /// degrading workloads with nothing else in common. Currently nothing populates it.
        /// </summary>
        public static string PodNodeQuery(IPrometheusQuerySelector selector)
        {
            var matchers = BuildSelector(selector, DataCenter.West);

            return $"kube_pod_info{{{matchers}}}";
        }

        /// <summary>
        /// <b>Pod labels.</b> Carries the one fact the guard cannot work out for itself: which replicas are
        /// meant to be compared against each other.
        ///
        /// <para>A rollout, a canary and an elected leader all look identical from the metrics — a minority of
        /// pods behaving unlike the majority — and each calls for a different answer. The cluster already
        /// knows which is which, because the operator that runs the workload publishes it as a label and
        /// updates it on failover. Reading that beats both guessing and asking a human to keep a list.</para>
        ///
        /// <para>kube-state-metrics exposes every pod label as <c>label_&lt;key&gt;</c> on this series, with
        /// dots and dashes flattened to underscores.</para>
        /// </summary>
        public static string PodLabelsQuery(IPrometheusQuerySelector selector)
        {
            var matchers = BuildSelector(selector, DataCenter.West);

            return $"kube_pod_labels{{{matchers}}}";
        }

        /// <summary>
        /// The Prometheus label name kube-state-metrics gives a Kubernetes pod label. Anything that is not a
        /// letter, digit or underscore becomes an underscore, which is what makes <c>app.kubernetes.io/name</c>
        /// reachable at all.
        /// </summary>
        public static string PodLabelSeriesName(string kubernetesLabel)
        {
            ArgumentNullException.ThrowIfNull(kubernetesLabel);

            var text = new StringBuilder(kubernetesLabel.Length + 6);
            text.Append("label_");

            for (var i = 0; i < kubernetesLabel.Length; i++)
            {
                var c = kubernetesLabel[i];

                text.Append(char.IsAsciiLetterOrDigit(c) || c == '_' ? c : '_');
            }

            return text.ToString();
        }

        /// <summary>
        /// <b>The rollout timestamp</b>, as a unix time in the sample's value rather than in a label.
        ///
        /// <para>An incident whose start coincides with this moving is <i>more</i> informative, not less: "this
        /// began when version N went out" is the attribution a bare recording rule cannot produce. Suppressing
        /// alerts during a rollout — the common workaround — hides exactly the faults a rollout causes.</para>
        /// </summary>
        public static string DeploymentCreatedQuery(IPrometheusQuerySelector selector)
        {
            var matchers = BuildNamespaceSelector(selector);

            return matchers.Length > 0
                ? $"kube_deployment_created{{{matchers}}}"
                : "kube_deployment_created";
        }

        /// <summary>
        /// <b>How many pods each ReplicaSet holds</b> — the old-versus-new split during a rollout, in one query.
        /// A workload with two non-zero entries is mid-rollout, which is the condition under which the peer family
        /// should be partitioned rather than trusted.
        /// </summary>
        public static string ReplicaSetPopulationQuery(IPrometheusQuerySelector selector)
        {
            return $"sum by (owner_name) ({PodOwnershipQuery(selector)})";
        }

        /// <summary>Whether queries should omit the data-centre matcher and run once rather than per centre.</summary>
        public static bool IsSingleDataCenter(IPrometheusQuerySelector selector)
        {
            ArgumentNullException.ThrowIfNull(selector);

            return string.IsNullOrWhiteSpace(selector.DataCenterLabel);
        }

        /// <summary>
        /// The PromQL for one feature with <paramref name="labelSelector"/> substituted, or <c>null</c> when
        /// this deployment has no source for it.
        /// </summary>
        public static string? Build(
            IPrometheusQuerySelector selector,
            MetricIndex metric,
            string labelSelector)
        {
            var template = ResolveTemplate(selector, metric);

            if (template.Length == 0)
            {
                return null;
            }

            return template.Replace(SelectorToken, labelSelector, StringComparison.Ordinal);
        }

        /// <summary>
        /// Configured override if one exists — <b>including an empty one</b>, which is how a deployment
        /// declares that a feature has no source — otherwise the built-in template.
        /// </summary>
        public static string ResolveTemplate(IPrometheusQuerySelector selector, MetricIndex metric)
        {
            ArgumentNullException.ThrowIfNull(selector);

            var overrides = selector.QueryOverrides;

            if (overrides is not null && overrides.TryGetValue(metric, out var configured))
            {
                return string.IsNullOrWhiteSpace(configured) ? string.Empty : configured;
            }

            return DefaultTemplate(metric);
        }

        /// <summary>
        /// Query set for a cluster running this project's own ASP.NET server, scraped by
        /// kube-prometheus-stack — the shape the repository's <c>k8s/</c> lab stands up.
        ///
        /// <para><b>Two features are approximations, and the deviation is stated rather than hidden.</b>
        /// <c>CpuUsageRatio</c> is documented as a fraction of the pod's CPU limit, but a limit is optional in
        /// Kubernetes and three of the lab's four pods have none — so this returns cores consumed, and a
        /// consumer comparing it against a 0…1 expectation will be wrong. <c>GcGen2HeapBytes</c> maps to the
        /// whole managed heap, because the server exposes <c>dotnet_gc_heap_size_bytes</c> with no generation
        /// label.</para>
        ///
        /// <para><c>sum by (pod)</c> wraps the cAdvisor queries deliberately: cAdvisor emits one series per
        /// container plus a pod-level aggregate, and without the aggregation a single pod arrives as several
        /// samples that overwrite each other during feature assembly.</para>
        /// </summary>
        /// <param name="window">Range used by every <c>rate()</c>. Should be at least four scrape intervals,
        /// or a rate over a restart-truncated window reads as a spike.</param>
        public static Dictionary<MetricIndex, string> OverfitServerQueries(TimeSpan? window = null)
        {
            var range = FormatRange(window ?? TimeSpan.FromMinutes(2));
            const string S = SelectorToken;

            return new Dictionary<MetricIndex, string>((int)MetricIndex.Count)
            {
                [MetricIndex.CpuUsageRatio] =
                    $"sum by (pod) (rate(container_cpu_usage_seconds_total{{{S}}}[{range}]))",

                // Present only on pods that carry a CPU limit — CFS accounting does not exist without a
                // quota. Pods without a limit produce no series at all, which is not the same as zero.
                [MetricIndex.CpuThrottleRatio] =
                    $"sum by (pod) (rate(container_cpu_cfs_throttled_periods_total{{{S}}}[{range}]))"
                    + $" / sum by (pod) (rate(container_cpu_cfs_periods_total{{{S}}}[{range}]))",

                [MetricIndex.MemoryWorkingSetBytes] =
                    $"sum by (pod) (container_memory_working_set_bytes{{{S}}})",

                [MetricIndex.OomEventsRate] =
                    $"sum by (pod) (rate(container_oom_events_total{{{S}}}[{range}]))",

                [MetricIndex.LatencyP50Ms] = LatencyQuery(0.50, S, range),
                [MetricIndex.LatencyP95Ms] = LatencyQuery(0.95, S, range),
                [MetricIndex.LatencyP99Ms] = LatencyQuery(0.99, S, range),

                [MetricIndex.RequestsPerSecond] =
                    $"sum by (pod) (rate(overfit_chat_requests_total{{{S}}}[{range}]))",

                // Undefined while no requests are arriving, and left undefined on purpose: 0/0 yields NaN,
                // which is the honest answer. Substituting zero would report a perfect error rate for a
                // server that is not serving anything.
                [MetricIndex.ErrorRate] =
                    $"sum by (pod) (rate(overfit_http_responses_total{{{S},status=\"5xx\"}}[{range}]))"
                    + $" / sum by (pod) (rate(overfit_http_responses_total{{{S}}}[{range}]))",

                [MetricIndex.GcGen2HeapBytes] =
                    $"sum by (pod) (dotnet_gc_heap_size_bytes{{{S}}})",

                [MetricIndex.GcPauseRatio] =
                    $"sum by (pod) (rate(dotnet_gc_pause_seconds_total{{{S}}}[{range}]))",

                [MetricIndex.ThreadPoolQueueLength] =
                    $"sum by (pod) (dotnet_threadpool_queue_length{{{S}}})",

                // increase(), not the raw counter: the absolute restart total says how old a pod is, while its
                // increase over the window says whether it restarted just now. Only the second is a signal.
                [MetricIndex.ContainerRestarts] =
                    $"sum by (pod) (increase(kube_pod_container_status_restarts_total{{{S}}}[{range}]))"
            };
        }

        /// <summary>
        /// The built-in queries, in OpenTelemetry naming. A starting point, not a promise — see the class
        /// remarks.
        /// </summary>
        public static string DefaultTemplate(MetricIndex metric)
        {
            const string S = SelectorToken;

            return metric switch
            {
                MetricIndex.CpuUsageRatio =>
                    $"sum by (pod) (rate(container_cpu_usage_seconds_total{{{S}}}[1m]))",

                MetricIndex.CpuThrottleRatio =>
                    $"sum by (pod) (rate(container_cpu_cfs_throttled_periods_total{{{S}}}[1m]))"
                    + $" / sum by (pod) (rate(container_cpu_cfs_periods_total{{{S}}}[1m]))",

                MetricIndex.MemoryWorkingSetBytes =>
                    $"sum by (pod) (container_memory_working_set_bytes{{{S}}})",

                MetricIndex.OomEventsRate =>
                    $"sum by (pod) (rate(container_oom_events_total{{{S}}}[1m]))",

                MetricIndex.LatencyP50Ms => OtelLatencyQuery(0.50, S),
                MetricIndex.LatencyP95Ms => OtelLatencyQuery(0.95, S),
                MetricIndex.LatencyP99Ms => OtelLatencyQuery(0.99, S),

                MetricIndex.RequestsPerSecond =>
                    $"sum by (pod) (rate(http_server_request_duration_seconds_count{{{S}}}[1m]))",

                MetricIndex.ErrorRate =>
                    $"sum by (pod) (rate(http_server_request_duration_seconds_count{{{S},http_response_status_code=~\"5..\"}}[1m]))"
                    + $" / sum by (pod) (rate(http_server_request_duration_seconds_count{{{S}}}[1m]))",

                MetricIndex.GcGen2HeapBytes =>
                    $"sum by (pod) (process_runtime_dotnet_gc_heap_size_bytes{{{S},generation=\"2\"}})",

                MetricIndex.GcPauseRatio =>
                    $"sum by (pod) (rate(process_runtime_dotnet_gc_pause_total_seconds_total{{{S}}}[1m]))",

                MetricIndex.ThreadPoolQueueLength =>
                    $"sum by (pod) (process_runtime_dotnet_thread_pool_queue_length{{{S}}})",

                MetricIndex.ContainerRestarts =>
                    $"sum by (pod) (increase(kube_pod_container_status_restarts_total{{{S}}}[1m]))",

                _ => throw new ArgumentOutOfRangeException(nameof(metric), metric, null)
            };
        }

        /// <summary>
        /// <c>sum by (pod, le)</c> keeps the pod label through the quantile. Without it the result carries no
        /// pod and every sample is discarded during parsing — a silent, total loss of the latency features.
        /// </summary>
        private static string LatencyQuery(double quantile, string selector, string range)
        {
            return $"histogram_quantile({Format(quantile)},"
                   + $" sum by (pod, le) (rate(overfit_chat_response_time_seconds_bucket{{{selector}}}[{range}])))"
                   + " * 1000";
        }

        private static string OtelLatencyQuery(double quantile, string selector)
        {
            return $"histogram_quantile({Format(quantile)},"
                   + $" sum by (pod, le) (rate(http_server_request_duration_seconds_bucket{{{selector}}}[1m])))"
                   + " * 1000";
        }

        private static string Format(double value) => value.ToString(CultureInfo.InvariantCulture);

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
