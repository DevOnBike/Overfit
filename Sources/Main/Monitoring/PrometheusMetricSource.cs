// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Net.Http.Headers;
using DevOnBike.Overfit.Monitoring.Abstractions;
using DevOnBike.Overfit.Monitoring.Contracts;

namespace DevOnBike.Overfit.Monitoring
{
    /// <summary>
    /// Scrapes all 12 must-have metrics for one K8s .NET pod from the Prometheus
    /// HTTP API and surfaces them as a stream of <see cref="MetricSnapshot"/> values.
    ///
    /// Uses the Prometheus instant query endpoint (/api/v1/query) — one HTTP
    /// request per metric per scrape interval. All 12 features are fetched in
    /// parallel and assembled into a single zero-allocation struct.
    ///
    /// PromQL queries issued (one per ReadAsync call):
    ///   CPU usage:       rate(container_cpu_usage_seconds_total{pod,ns}[1m]) / cpu_limit
    ///   CPU throttle:    rate(cfs_throttled_periods{pod,ns}[1m]) / rate(cfs_periods{pod,ns}[1m])
    ///   Memory:          container_memory_working_set_bytes{pod,ns}
    ///   OOM rate:        rate(container_oom_events_total{pod,ns}[1m])
    ///   Latency p50:     histogram_quantile(0.50, rate(http_server_request_duration_seconds_bucket{pod}[1m])) * 1000
    ///   Latency p95:     histogram_quantile(0.95, ...) * 1000
    ///   Latency p99:     histogram_quantile(0.99, ...) * 1000
    ///   RPS:             rate(http_server_request_duration_seconds_count{pod}[1m])
    ///   Error rate:      rate(5xx_count{pod}[1m]) / (rate(total_count{pod}[1m]) or vector(1))
    ///   GC Gen2 heap:    process_runtime_dotnet_gc_heap_size_bytes{pod,generation="2"}
    ///   GC pause ratio:  rate(process_runtime_dotnet_gc_pause_total_seconds_total{pod}[1m])
    ///   ThreadPool:      process_runtime_dotnet_thread_pool_queue_length{pod}
    ///
    /// Callers own the lifetime — call Dispose() when the pipeline stops.
    /// The internal HttpClient is disposed together with this instance.
    /// </summary>
    public sealed class PrometheusMetricSource : IMetricSource
    {
        private readonly PrometheusMetricSourceConfig _config;
        private readonly HttpClient _http;
        private bool _disposed;

        public string PodName => _config.PodName;

        /// <param name="config">Scrape configuration.</param>
        /// <param name="httpClient">
        ///   Optional pre-configured HttpClient (e.g. for test injection or
        ///   shared connection pools). When null, an internal client is created
        ///   and disposed together with this source.
        /// </param>
        public PrometheusMetricSource(
            PrometheusMetricSourceConfig config,
            HttpClient? httpClient = null)
        {
            ArgumentNullException.ThrowIfNull(config);
            ArgumentException.ThrowIfNullOrEmpty(config.PrometheusBaseUrl);
            ArgumentException.ThrowIfNullOrEmpty(config.PodName);

            _config = config;
            _http = httpClient ?? BuildHttpClient(config);
        }

        // -------------------------------------------------------------------------
        // IMetricSource
        // -------------------------------------------------------------------------

        /// <summary>
        /// Waits for the configured scrape interval, then issues 12 PromQL queries
        /// in parallel and assembles the results into a <see cref="MetricSnapshot"/>.
        ///
        /// On Prometheus connectivity failure, the method throws — the pipeline
        /// will propagate this to its RunAsync caller where it can be handled.
        /// </summary>
        public async ValueTask<MetricSnapshot> ReadAsync(CancellationToken ct = default)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            await Task.Delay(_config.ScrapeInterval, ct).ConfigureAwait(false);

            var pod = _config.PodName;
            var ns = _config.Namespace;

            // Issue all 12 queries in parallel — total latency ≈ single query latency
            var cpuUsageTask = QueryScalarAsync(BuildCpuUsageQuery(pod, ns), ct);
            var cpuThrottleTask = QueryScalarAsync(BuildCpuThrottleQuery(pod, ns), ct);
            var memTask = QueryScalarAsync(BuildMemoryQuery(pod, ns), ct);
            var oomTask = QueryScalarAsync(BuildOomQuery(pod, ns), ct);
            var latP50Task = QueryScalarAsync(BuildLatencyQuery(pod, 0.50f), ct);
            var latP95Task = QueryScalarAsync(BuildLatencyQuery(pod, 0.95f), ct);
            var latP99Task = QueryScalarAsync(BuildLatencyQuery(pod, 0.99f), ct);
            var rpsTask = QueryScalarAsync(BuildRpsQuery(pod), ct);
            var errorRateTask = QueryScalarAsync(BuildErrorRateQuery(pod), ct);
            var gcGen2Task = QueryScalarAsync(BuildGcGen2Query(pod), ct);
            var gcPauseTask = QueryScalarAsync(BuildGcPauseRatioQuery(pod), ct);
            var threadPoolTask = QueryScalarAsync(BuildThreadPoolQuery(pod), ct);

            await Task.WhenAll(
                cpuUsageTask, cpuThrottleTask, memTask, oomTask,
                latP50Task, latP95Task, latP99Task, rpsTask, errorRateTask,
                gcGen2Task, gcPauseTask, threadPoolTask
            ).ConfigureAwait(false);

            return new MetricSnapshot
            {
                Timestamp = DateTime.UtcNow,
                PodName = pod,
                CpuUsageRatio = Math.Clamp(await cpuUsageTask, 0f, 1f),
                CpuThrottleRatio = Math.Clamp(await cpuThrottleTask, 0f, 1f),
                MemoryWorkingSetBytes = MathF.Max(0f, await memTask),
                OomEventsRate = MathF.Max(0f, await oomTask),
                LatencyP50Ms = MathF.Max(0f, await latP50Task),
                LatencyP95Ms = MathF.Max(0f, await latP95Task),
                LatencyP99Ms = MathF.Max(0f, await latP99Task),
                RequestsPerSecond = MathF.Max(0f, await rpsTask),
                ErrorRate = Math.Clamp(await errorRateTask, 0f, 1f),
                GcGen2HeapBytes = MathF.Max(0f, await gcGen2Task),
                GcPauseRatio = Math.Clamp(await gcPauseTask, 0f, 1f),
                ThreadPoolQueueLength = MathF.Max(0f, await threadPoolTask)
            };
        }

        public void Dispose()
        {
            if (_disposed) { return; }
            _disposed = true;
            _http.Dispose();
        }

        // -------------------------------------------------------------------------
        // PromQL query builders
        // -------------------------------------------------------------------------

        private static string BuildCpuUsageQuery(string pod, string ns)
            // CPU usage normalised to the container's CPU limit → [0, 1]
            => $"rate(container_cpu_usage_seconds_total{{pod=\"{pod}\",namespace=\"{ns}\"}}[1m])" +
               $" / on(pod) kube_pod_container_resource_limits{{pod=\"{pod}\",resource=\"cpu\"}}";

        private static string BuildCpuThrottleQuery(string pod, string ns)
            // Fraction of CFS periods in which the container was throttled → [0, 1]
            // Divide throttled by total periods — not by time.
            => $"rate(container_cpu_cfs_throttled_periods_total{{pod=\"{pod}\",namespace=\"{ns}\"}}[1m])" +
               $" / rate(container_cpu_cfs_periods_total{{pod=\"{pod}\",namespace=\"{ns}\"}}[1m])";

        private static string BuildMemoryQuery(string pod, string ns)
            // Working set excludes reclaimable page cache — what K8s uses for eviction
            => $"container_memory_working_set_bytes{{pod=\"{pod}\",namespace=\"{ns}\"}}";

        private static string BuildOomQuery(string pod, string ns)
            // OOM kill rate — normally always 0
            => $"rate(container_oom_events_total{{pod=\"{pod}\",namespace=\"{ns}\"}}[1m])";

        private static string BuildLatencyQuery(string pod, float quantile)
            // Single builder for all latency percentiles — quantile passed as parameter.
            // Uses ASP.NET Core built-in histogram metric name (net9+).
            => $"histogram_quantile({quantile.ToString("F2", CultureInfo.InvariantCulture)}," +
               $" rate(http_server_request_duration_seconds_bucket{{pod=\"{pod}\"}}[1m])) * 1000";

        private static string BuildRpsQuery(string pod)
            => $"rate(http_server_request_duration_seconds_count{{pod=\"{pod}\"}}[1m])";

        private static string BuildErrorRateQuery(string pod)
            // or vector(1) guards against division by zero when the pod has no traffic.
            // Result: 0/1 = 0 (safe) rather than 0/0 = NaN (propagates as 0 via ParseFirstScalar,
            // but vector(1) makes the intent explicit).
            => $"rate(http_server_request_duration_seconds_count{{pod=\"{pod}\",http_response_status_code=~\"5..\"}}[1m])" +
               $" / (rate(http_server_request_duration_seconds_count{{pod=\"{pod}\"}}[1m]) or vector(1))";

        private static string BuildGcGen2Query(string pod)
            // Gen2 heap is the best early signal for memory leaks
            => $"process_runtime_dotnet_gc_heap_size_bytes{{pod=\"{pod}\",generation=\"2\"}}";

        private static string BuildGcPauseRatioQuery(string pod)
            // rate() yields seconds-of-GC-per-second — a dimensionless ratio in [0, 1]
            // that stays comparable across different load levels
            => $"rate(process_runtime_dotnet_gc_pause_total_seconds_total{{pod=\"{pod}\"}}[1m])";

        private static string BuildThreadPoolQuery(string pod)
            => $"process_runtime_dotnet_thread_pool_queue_length{{pod=\"{pod}\"}}";

        // -------------------------------------------------------------------------
        // HTTP + JSON parsing
        // -------------------------------------------------------------------------

        /// <summary>
        /// Calls /api/v1/query, returns the first scalar result value.
        /// Returns 0 when the query returns no data (metric not yet available).
        /// Throws <see cref="HttpRequestException"/> on network failures.
        /// </summary>
        private async Task<float> QueryScalarAsync(string promql, CancellationToken ct)
        {
            var url = $"{_config.PrometheusBaseUrl}/api/v1/query" +
                      $"?query={Uri.EscapeDataString(promql)}";

            using var response = await _http.GetAsync(url, ct).ConfigureAwait(false);
            response.EnsureSuccessStatusCode();

            var body = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            return ParseFirstScalar(body);
        }

        /// <summary>
        /// Minimal Prometheus API response parser — no external JSON library required.
        ///
        /// Response shape (success):
        /// {
        ///   "status": "success",
        ///   "data": {
        ///     "resultType": "vector",
        ///     "result": [ { "metric": {...}, "value": [timestamp, "1.234"] } ]
        ///   }
        /// }
        ///
        /// We only need the string value at result[0].value[1].
        /// Parsing is done with simple string search rather than System.Text.Json
        /// to keep zero allocations for common fast-path single-metric responses.
        /// Falls back to 0 on any parse failure — the pipeline continues with
        /// a conservative "no data" value rather than crashing.
        /// </summary>
        public static float ParseFirstScalar(string json)
        {
            // Find "value": followed by optional whitespace and [timestamp,
            const string valueKey = "\"value\":";
            var vi = json.IndexOf(valueKey, StringComparison.Ordinal);
            if (vi < 0) { return 0f; }

            // Skip to opening bracket (handles "value":[... and "value": [...])
            var bracket = json.IndexOf('[', vi + valueKey.Length);
            if (bracket < 0) { return 0f; }

            // Skip past the timestamp and the comma
            var comma = json.IndexOf(',', bracket);
            if (comma < 0) { return 0f; }

            // Find the quoted value string
            var q1 = json.IndexOf('"', comma);
            if (q1 < 0) { return 0f; }

            var q2 = json.IndexOf('"', q1 + 1);
            if (q2 < 0) { return 0f; }

            var valueStr = json.AsSpan(q1 + 1, q2 - q1 - 1);

            // Handle Prometheus special values
            if (valueStr.Equals("NaN", StringComparison.Ordinal) ||
                valueStr.Equals("+Inf", StringComparison.Ordinal) ||
                valueStr.Equals("-Inf", StringComparison.Ordinal))
            {
                return 0f;
            }

            return float.TryParse(
                valueStr,
                NumberStyles.Float,
                CultureInfo.InvariantCulture,
                out var result) ? result : 0f;
        }

        private static HttpClient BuildHttpClient(PrometheusMetricSourceConfig config)
        {
            var client = new HttpClient
            {
                Timeout = config.HttpTimeout,
                BaseAddress = new Uri(config.PrometheusBaseUrl)
            };

            client.DefaultRequestHeaders.Accept.Add(
                new MediaTypeWithQualityHeaderValue("application/json"));

            return client;
        }
    }
}