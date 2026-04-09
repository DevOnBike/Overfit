// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Net.Http.Headers;

namespace DevOnBike.Overfit.Monitoring
{
    /// <summary>
    /// Scrapes a single K8s pod's metrics from the Prometheus HTTP API
    /// and surfaces them as a stream of <see cref="MetricSnapshot"/> values.
    ///
    /// Uses the Prometheus instant query endpoint (/api/v1/query) — one HTTP
    /// request per metric per scrape interval. All 8 features are fetched in
    /// parallel and assembled into a single zero-allocation struct.
    ///
    /// PromQL queries issued (one per ReadAsync call):
    ///   CPU:           rate(container_cpu_usage_seconds_total{pod="P",namespace="N"}[1m])
    ///                  / on(pod) kube_pod_container_resource_limits{pod="P",resource="cpu"}
    ///   Memory:        container_memory_working_set_bytes{pod="P",namespace="N"}
    ///   Latency p95:   histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{pod="P"}[1m])) * 1000
    ///   RPS:           rate(http_requests_total{pod="P"}[1m])
    ///   Error rate:    rate(http_requests_total{pod="P",status=~"5.."}[1m])
    ///                  / rate(http_requests_total{pod="P"}[1m])
    ///   GC pause:      rate(dotnet_gc_pause_total_seconds_total{pod="P"}[1m]) * 1000
    ///   ThreadPool:    dotnet_threadpool_queue_length{pod="P"}
    ///   Heap bytes:    dotnet_gc_heap_size_bytes{pod="P"}
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
        /// Waits for the configured scrape interval, then issues 8 PromQL queries
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

            // Issue all 8 queries in parallel — total latency ≈ single query latency
            var cpuTask = QueryScalarAsync(BuildCpuQuery(pod, ns), ct);
            var memTask = QueryScalarAsync(BuildMemoryQuery(pod, ns), ct);
            var latencyTask = QueryScalarAsync(BuildLatencyP95Query(pod), ct);
            var rpsTask = QueryScalarAsync(BuildRpsQuery(pod), ct);
            var errorRateTask = QueryScalarAsync(BuildErrorRateQuery(pod), ct);
            var gcPauseTask = QueryScalarAsync(BuildGcPauseQuery(pod), ct);
            var threadPoolTask = QueryScalarAsync(BuildThreadPoolQuery(pod), ct);
            var heapTask = QueryScalarAsync(BuildHeapQuery(pod), ct);

            await Task.WhenAll(
                cpuTask, memTask, latencyTask, rpsTask,
                errorRateTask, gcPauseTask, threadPoolTask, heapTask
            ).ConfigureAwait(false);

            return new MetricSnapshot
            {
                Timestamp = DateTime.UtcNow,
                PodName = pod,
                CpuUsage = Math.Clamp(await cpuTask, 0f, 1f),
                MemoryBytes = MathF.Max(0f, await memTask),
                RequestLatencyP95 = MathF.Max(0f, await latencyTask),
                RequestsPerSecond = MathF.Max(0f, await rpsTask),
                ErrorRate = Math.Clamp(await errorRateTask, 0f, 1f),
                GcPauseMs = MathF.Max(0f, await gcPauseTask),
                ThreadPoolQueue = MathF.Max(0f, await threadPoolTask),
                HeapBytes = MathF.Max(0f, await heapTask)
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

        private static string BuildCpuQuery(string pod, string ns)
            // CPU usage as fraction of the container's CPU limit (0–1)
            => $"rate(container_cpu_usage_seconds_total{{pod=\"{pod}\",namespace=\"{ns}\"}}[1m])" +
               $" / on(pod) kube_pod_container_resource_limits{{pod=\"{pod}\",resource=\"cpu\"}}";

        private static string BuildMemoryQuery(string pod, string ns)
            // Working set bytes — excludes cached pages, matches what K8s eviction uses
            => $"container_memory_working_set_bytes{{pod=\"{pod}\",namespace=\"{ns}\"}}";

        private static string BuildLatencyP95Query(string pod)
            // p95 latency in milliseconds
            => $"histogram_quantile(0.95," +
               $" rate(http_request_duration_seconds_bucket{{pod=\"{pod}\"}}[1m])) * 1000";

        private static string BuildRpsQuery(string pod)
            => $"rate(http_requests_total{{pod=\"{pod}\"}}[1m])";

        private static string BuildErrorRateQuery(string pod)
            // Fraction of 5xx responses (0–1); returns 0 when total RPS is 0
            => $"rate(http_requests_total{{pod=\"{pod}\",status=~\"5..\"}}[1m])" +
               $" / rate(http_requests_total{{pod=\"{pod}\"}}[1m])";

        private static string BuildGcPauseQuery(string pod)
            // .NET GC pause rate in milliseconds per second
            => $"rate(dotnet_gc_pause_total_seconds_total{{pod=\"{pod}\"}}[1m]) * 1000";

        private static string BuildThreadPoolQuery(string pod)
            => $"dotnet_threadpool_queue_length{{pod=\"{pod}\"}}";

        private static string BuildHeapQuery(string pod)
            => $"dotnet_gc_heap_size_bytes{{pod=\"{pod}\"}}";

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
        internal static float ParseFirstScalar(string json)
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