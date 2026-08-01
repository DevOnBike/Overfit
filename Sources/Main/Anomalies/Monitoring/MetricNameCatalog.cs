// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    /// The metric names each channel is known to appear under, across the runtimes and exporters this is
    /// likely to meet.
    ///
    /// <para><b>This is a table of conventions, not a schema.</b> Nothing here is authoritative — it is a
    /// list of names that have been observed in the wild, used to <i>propose</i> a mapping that a human then
    /// confirms. It exists because the alternative is a customer hand-writing thirteen bindings, and the
    /// evidence that this goes wrong is not hypothetical: the mapping for this project's own lab was written
    /// by the author of the system and still left <b>two channels of thirteen unbound</b>, reporting blind for
    /// hours before anyone looked.</para>
    ///
    /// <para><b>Container-level names are stack-independent</b> and cover roughly half the channels: cAdvisor
    /// and kube-state-metrics report the same series whether the pod runs Java, PHP, Go or .NET. The
    /// runtime-level ones are not, and that is the entire reason discovery has to look at what a workload
    /// actually exports rather than assume.</para>
    /// </summary>
    public static class MetricNameCatalog
    {
        /// <summary>
        /// Candidate series for <paramref name="metric"/>, most conventional first. Evidence is left at zero;
        /// <see cref="MetricDiscovery"/> fills it in from what the pods actually report.
        /// </summary>
        public static IReadOnlyList<MetricCandidate> For(MetricIndex metric)
        {
            return metric switch
            {
                MetricIndex.CpuUsageRatio =>
                [
                    Container("container_cpu_usage_seconds_total", MetricSourceKind.Counter),
                ],

                MetricIndex.CpuThrottleRatio =>
                [
                    Container("container_cpu_cfs_throttled_periods_total", MetricSourceKind.Counter),
                ],

                MetricIndex.MemoryWorkingSetBytes =>
                [
                    Container("container_memory_working_set_bytes", MetricSourceKind.Gauge),
                ],

                MetricIndex.OomEventsRate =>
                [
                    Container("container_oom_events_total", MetricSourceKind.EventCount),
                ],

                MetricIndex.ContainerRestarts =>
                [
                    Container("kube_pod_container_status_restarts_total", MetricSourceKind.EventCount),
                ],

                // Latency: three live conventions, and a cluster can carry more than one at once — an
                // application instrumented with Micrometer behind an OTel collector exports both.
                MetricIndex.LatencyP50Ms => Latency(0.50),
                MetricIndex.LatencyP95Ms => Latency(0.95),
                MetricIndex.LatencyP99Ms => Latency(0.99),

                MetricIndex.RequestsPerSecond =>
                [
                    new("http_server_request_duration_seconds_count", MetricSourceKind.Counter, "otel", 0, double.NaN),
                    new("http_requests_total", MetricSourceKind.Counter, "prometheus-client", 0, double.NaN),
                    new("http_server_requests_seconds_count", MetricSourceKind.Counter, "micrometer", 0, double.NaN),
                    new("nginx_http_requests_total", MetricSourceKind.Counter, "nginx", 0, double.NaN),
                    new("phpfpm_accepted_connections", MetricSourceKind.Counter, "php-fpm", 0, double.NaN),
                ],

                // Deliberately listed WITHOUT a status-code selector. Which codes count as an error is a
                // business decision — 4xx is a client's fault and also, sometimes, an outage — and no series
                // answers it. Discovery surfaces the counter and a human writes the selector.
                MetricIndex.ErrorRate =>
                [
                    new("http_requests_total", MetricSourceKind.Counter, "prometheus-client", 0, double.NaN),
                    new("http_server_requests_seconds_count", MetricSourceKind.Counter, "micrometer", 0, double.NaN),
                    new("nginx_http_requests_total", MetricSourceKind.Counter, "nginx", 0, double.NaN),
                ],

                MetricIndex.GcGen2HeapBytes =>
                [
                    new("dotnet_gc_heap_size_bytes", MetricSourceKind.Gauge, "dotnet", 0, double.NaN),
                    new("jvm_memory_used_bytes", MetricSourceKind.Gauge, "jvm", 0, double.NaN),
                    new("go_memstats_heap_inuse_bytes", MetricSourceKind.Gauge, "go", 0, double.NaN),
                    new("python_gc_objects_collected_total", MetricSourceKind.Counter, "python", 0, double.NaN),
                ],

                MetricIndex.GcPauseRatio =>
                [
                    new("dotnet_gc_pause_seconds_total", MetricSourceKind.Counter, "dotnet", 0, double.NaN),
                    new("jvm_gc_pause_seconds_sum", MetricSourceKind.Counter, "jvm", 0, double.NaN),
                    new("go_gc_duration_seconds_sum", MetricSourceKind.Counter, "go", 0, double.NaN),
                ],

                MetricIndex.ThreadPoolQueueLength =>
                [
                    new("dotnet_threadpool_queue_length", MetricSourceKind.Gauge, "dotnet", 0, double.NaN),
                    new("executor_queued_tasks", MetricSourceKind.Gauge, "micrometer", 0, double.NaN),
                    new("jvm_threads_states_threads", MetricSourceKind.Gauge, "jvm", 0, double.NaN),
                    new("phpfpm_listen_queue", MetricSourceKind.Gauge, "php-fpm", 0, double.NaN),
                    new("nginx_connections_waiting", MetricSourceKind.Gauge, "nginx", 0, double.NaN),
                ],

                _ => [],
            };
        }

        /// <summary>
        /// Name <b>endings</b> that identify a channel whatever the application calls itself.
        ///
        /// <para><b>This is what makes discovery work on an application nobody has met.</b> Prometheus
        /// convention governs the suffix — <c>_total</c> for a counter, <c>_seconds_bucket</c> for a latency
        /// histogram — while the prefix is just the service's own name. Matching whole names therefore finds
        /// only applications instrumented by a library this catalog happens to list, and measured on this
        /// project's own lab that meant <b>four of thirteen channels blind</b>: the workload exports
        /// <c>labapp_requests_total</c> and <c>labapp_request_duration_seconds</c>, which are perfectly
        /// conventional and matched nothing.</para>
        ///
        /// <para><b>Consulted only when no exact candidate was evidenced</b>, so a known name always wins. A
        /// suffix match is a weaker claim and is marked <see cref="MetricCandidate.Inferred"/> so a human
        /// reading the report can see which is which.</para>
        ///
        /// <para><b>Deliberately not maximally permissive.</b> A bare <c>_seconds_bucket</c> would match every
        /// histogram in a cluster and turn a useful proposal into a list nobody reads, so the latency
        /// suffixes insist on <c>duration</c> or <c>latency</c>. Where several still match, the answer is
        /// <see cref="DiscoveryOutcome.Ambiguous"/> — which is a good outcome, not a failed one.</para>
        /// </summary>
        public static IReadOnlyList<string> SuffixesFor(MetricIndex metric)
        {
            return metric switch
            {
                MetricIndex.RequestsPerSecond =>
                [
                    "_requests_total",
                    "_request_duration_seconds_count",
                    "_requests_seconds_count",
                    "_request_duration_seconds_count",
                ],

                MetricIndex.ErrorRate =>
                [
                    "_errors_total",
                    "_errors_count",
                    "_failures_total",
                    "_failed_total",
                ],

                MetricIndex.LatencyP50Ms or MetricIndex.LatencyP95Ms or MetricIndex.LatencyP99Ms =>
                [
                    "_request_duration_seconds_bucket",
                    "_duration_seconds_bucket",
                    "_latency_seconds_bucket",
                ],

                MetricIndex.GcGen2HeapBytes =>
                [
                    "_heap_size_bytes",
                    "_heap_used_bytes",
                    "_heap_inuse_bytes",
                ],

                MetricIndex.GcPauseRatio =>
                [
                    "_gc_pause_seconds_total",
                    "_gc_duration_seconds_sum",
                    "_gc_collection_seconds_sum",
                ],

                MetricIndex.ThreadPoolQueueLength =>
                [
                    "_queue_length",
                    "_queued_tasks",
                    "_listen_queue",
                    "_queue_size",
                ],

                // The container-level channels are exact by nature: cAdvisor and kube-state-metrics do not
                // vary their names per application, so a suffix rule here could only add false candidates.
                _ => [],
            };
        }

        /// <summary>
        /// Reads a series name's own suffix to decide how it must be sampled.
        ///
        /// <para>Prometheus convention again: <c>_bucket</c> is a histogram, <c>_total</c> is a counter,
        /// anything else is a gauge. Getting this wrong is not cosmetic — reading a counter as a gauge reports
        /// the monotonically rising total instead of the rate, which looks like an unstoppable upward trend on
        /// every pod for ever.</para>
        /// </summary>
        public static MetricSourceKind KindOf(string series)
        {
            ArgumentNullException.ThrowIfNull(series);

            if (series.EndsWith("_bucket", StringComparison.Ordinal))
            {
                return MetricSourceKind.HistogramSeconds;
            }

            if (series.EndsWith("_total", StringComparison.Ordinal)
                || series.EndsWith("_count", StringComparison.Ordinal)
                || series.EndsWith("_sum", StringComparison.Ordinal))
            {
                return MetricSourceKind.Counter;
            }

            return MetricSourceKind.Gauge;
        }

        /// <summary>
        /// The runtime families this can name, for the summary line. Used to tell an operator what the
        /// workload looks like — "these pods are a JVM" — before any binding is discussed.
        /// </summary>
        public static string StackOf(string metricName)
        {
            ArgumentNullException.ThrowIfNull(metricName);

            if (metricName.StartsWith("dotnet_", StringComparison.Ordinal)
                || metricName.StartsWith("process_runtime_dotnet_", StringComparison.Ordinal))
            {
                return "dotnet";
            }

            if (metricName.StartsWith("jvm_", StringComparison.Ordinal))
            {
                return "jvm";
            }

            if (metricName.StartsWith("go_", StringComparison.Ordinal))
            {
                return "go";
            }

            if (metricName.StartsWith("phpfpm_", StringComparison.Ordinal))
            {
                return "php-fpm";
            }

            if (metricName.StartsWith("nodejs_", StringComparison.Ordinal))
            {
                return "nodejs";
            }

            if (metricName.StartsWith("python_", StringComparison.Ordinal))
            {
                return "python";
            }

            if (metricName.StartsWith("nginx_", StringComparison.Ordinal))
            {
                return "nginx";
            }

            return string.Empty;
        }

        private static MetricCandidate Container(string source, MetricSourceKind kind)
            => new(source, kind, "container", 0, double.NaN);

        private static IReadOnlyList<MetricCandidate> Latency(double quantile)
        {
            return
            [
                new("http_server_request_duration_seconds", MetricSourceKind.HistogramSeconds, "otel", 0, quantile),
                new("http_request_duration_seconds", MetricSourceKind.HistogramSeconds, "prometheus-client", 0, quantile),
                new("http_server_requests_seconds", MetricSourceKind.HistogramSeconds, "micrometer", 0, quantile),
            ];
        }
    }
}
