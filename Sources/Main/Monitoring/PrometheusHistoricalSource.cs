using System.Globalization;
using System.Net.Http.Headers;
using DevOnBike.Overfit.Monitoring.Contracts;

namespace DevOnBike.Overfit.Monitoring
{
    /// <summary>
    /// Fetches historical metric data from Prometheus using the range query API
    /// (/api/v1/query_range) and reconstructs a sequence of <see cref="MetricSnapshot"/>
    /// values suitable for <see cref="OfflineTrainingJob"/>.
    ///
    /// Each of the 12 must-have metrics is fetched with a separate range query,
    /// then aligned by timestamp to build complete snapshots.
    ///
    /// Usage:
    /// <code>
    ///   var config = new PrometheusHistoricalSourceConfig
    ///   {
    ///       PrometheusBaseUrl = "http://prometheus:9090",
    ///       PodName           = "payment-service-7f9b4c",
    ///       RangeStart        = DateTime.UtcNow.AddDays(-7),
    ///       RangeEnd          = DateTime.UtcNow,
    ///       Step              = TimeSpan.FromSeconds(10)
    ///   };
    ///
    ///   using var source  = new PrometheusHistoricalSource(config);
    ///   var snapshots     = await source.FetchAsync(ct);
    ///   var vectors       = ExtractFeatureVectors(snapshots);
    ///   var report        = new TrainingDataAnalyzer().Analyze(vectors);
    ///   var result        = new OfflineTrainingJob().Run(autoencoder, scorer, vectors);
    /// </code>
    ///
    /// Gaps in data (pod restarts, Prometheus scrape failures) produce samples with
    /// 0-values for the affected metric — the same conservative fallback used in
    /// <see cref="PrometheusMetricSource.ParseFirstScalar"/>.
    /// </summary>
    public sealed class PrometheusHistoricalSource : IDisposable
    {
        private readonly PrometheusHistoricalSourceConfig _config;
        private readonly HttpClient _http;
        private bool _disposed;

        public PrometheusHistoricalSource(
            PrometheusHistoricalSourceConfig config,
            HttpClient? httpClient = null)
        {
            ArgumentNullException.ThrowIfNull(config);
            ArgumentException.ThrowIfNullOrEmpty(config.PrometheusBaseUrl);
            ArgumentException.ThrowIfNullOrEmpty(config.PodName);

            if (config.RangeEnd <= config.RangeStart)
            {
                throw new ArgumentException(
                "RangeEnd must be after RangeStart.", nameof(config));
            }

            _config = config;
            _http = httpClient ?? BuildHttpClient(config);
        }

        // -------------------------------------------------------------------------
        // FetchAsync
        // -------------------------------------------------------------------------

        /// <summary>
        /// Fetches all 12 metrics over the configured time range and assembles
        /// them into a chronologically ordered list of <see cref="MetricSnapshot"/>.
        ///
        /// Issues 12 parallel range queries — total latency ≈ slowest single query.
        /// For a 7-day range at 10s step, each query returns ~60,480 data points.
        /// </summary>
        public async Task<IReadOnlyList<MetricSnapshot>> FetchAsync(CancellationToken ct = default)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            var pod = _config.PodName;
            var ns = _config.Namespace;

            // Issue all 12 range queries in parallel
            var cpuUsageTask = FetchSeriesAsync(BuildCpuUsageQuery(pod, ns), ct);
            var cpuThrottleTask = FetchSeriesAsync(BuildCpuThrottleQuery(pod, ns), ct);
            var memTask = FetchSeriesAsync(BuildMemoryQuery(pod, ns), ct);
            var oomTask = FetchSeriesAsync(BuildOomQuery(pod, ns), ct);
            var latP50Task = FetchSeriesAsync(BuildLatencyQuery(pod, 0.50f), ct);
            var latP95Task = FetchSeriesAsync(BuildLatencyQuery(pod, 0.95f), ct);
            var latP99Task = FetchSeriesAsync(BuildLatencyQuery(pod, 0.99f), ct);
            var rpsTask = FetchSeriesAsync(BuildRpsQuery(pod), ct);
            var errorRateTask = FetchSeriesAsync(BuildErrorRateQuery(pod), ct);
            var gcGen2Task = FetchSeriesAsync(BuildGcGen2Query(pod), ct);
            var gcPauseTask = FetchSeriesAsync(BuildGcPauseRatioQuery(pod), ct);
            var threadPoolTask = FetchSeriesAsync(BuildThreadPoolQuery(pod), ct);

            await Task.WhenAll(
            cpuUsageTask, cpuThrottleTask, memTask, oomTask,
            latP50Task, latP95Task, latP99Task, rpsTask, errorRateTask,
            gcGen2Task, gcPauseTask, threadPoolTask
            ).ConfigureAwait(false);

            // Use the CPU series timestamps as the master timeline — it is the
            // most reliably scraped metric and defines the sample grid.
            var masterSeries = await cpuUsageTask;
            if (masterSeries.Count == 0) { return []; }

            var cpuThrottle = await cpuThrottleTask;
            var mem = await memTask;
            var oom = await oomTask;
            var latP50 = await latP50Task;
            var latP95 = await latP95Task;
            var latP99 = await latP99Task;
            var rps = await rpsTask;
            var errorRate = await errorRateTask;
            var gcGen2 = await gcGen2Task;
            var gcPause = await gcPauseTask;
            var threadPool = await threadPoolTask;

            var snapshots = new List<MetricSnapshot>(masterSeries.Count);

            foreach (var (ts, cpuVal) in masterSeries)
            {
                snapshots.Add(new MetricSnapshot
                {
                    Timestamp = ts,
                    PodName = pod,
                    CpuUsageRatio = Math.Clamp(cpuVal, 0f, 1f),
                    CpuThrottleRatio = Math.Clamp(Lookup(cpuThrottle, ts), 0f, 1f),
                    MemoryWorkingSetBytes = MathF.Max(0f, Lookup(mem, ts)),
                    OomEventsRate = MathF.Max(0f, Lookup(oom, ts)),
                    LatencyP50Ms = MathF.Max(0f, Lookup(latP50, ts)),
                    LatencyP95Ms = MathF.Max(0f, Lookup(latP95, ts)),
                    LatencyP99Ms = MathF.Max(0f, Lookup(latP99, ts)),
                    RequestsPerSecond = MathF.Max(0f, Lookup(rps, ts)),
                    ErrorRate = Math.Clamp(Lookup(errorRate, ts), 0f, 1f),
                    GcGen2HeapBytes = MathF.Max(0f, Lookup(gcGen2, ts)),
                    GcPauseRatio = Math.Clamp(Lookup(gcPause, ts), 0f, 1f),
                    ThreadPoolQueueLength = MathF.Max(0f, Lookup(threadPool, ts))
                });
            }

            return snapshots;
        }

        public void Dispose()
        {
            if (_disposed) { return; }
            _disposed = true;
            _http.Dispose();
        }

        // -------------------------------------------------------------------------
        // Range query + parsing
        // -------------------------------------------------------------------------

        private async Task<Dictionary<DateTime, float>> FetchSeriesAsync(
            string promql, CancellationToken ct)
        {
            var start = _config.RangeStart.ToUniversalTime().ToString("O");
            var end = _config.RangeEnd.ToUniversalTime().ToString("O");
            var step = ((int)_config.Step.TotalSeconds).ToString();

            var url = $"{_config.PrometheusBaseUrl}/api/v1/query_range" +
                      $"?query={Uri.EscapeDataString(promql)}" +
                      $"&start={Uri.EscapeDataString(start)}" +
                      $"&end={Uri.EscapeDataString(end)}" +
                      $"&step={step}s";

            using var response = await _http.GetAsync(url, ct).ConfigureAwait(false);
            response.EnsureSuccessStatusCode();

            var body = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            return ParseRangeSeries(body);
        }

        /// <summary>
        /// Parses the Prometheus range query response into a timestamp → value dictionary.
        ///
        /// Response shape:
        /// {
        ///   "data": {
        ///     "result": [ { "values": [[unix_ts, "value"], [unix_ts, "value"], ...] } ]
        ///   }
        /// }
        /// </summary>
        internal static Dictionary<DateTime, float> ParseRangeSeries(string json)
        {
            var result = new Dictionary<DateTime, float>();

            // Find "values":[ — the array of [timestamp, "value"] pairs
            const string valuesKey = "\"values\":[";
            var vi = json.IndexOf(valuesKey, StringComparison.Ordinal);
            if (vi < 0) { return result; }

            var pos = vi + valuesKey.Length;

            // Iterate over [ts,"val"] pairs until the closing ]
            while (pos < json.Length)
            {
                // Skip to opening [
                pos = json.IndexOf('[', pos);
                if (pos < 0 || json[pos - 1] == ']') { break; }

                // Closing ] of outer values array — stop
                var peek = pos - 1;
                while (peek >= 0 && json[peek] == ' ') { peek--; }
                if (peek >= 0 && json[peek] == ']') { break; }

                pos++; // skip [

                // Read unix timestamp (may be float like 1712345678.123)
                var comma = json.IndexOf(',', pos);
                if (comma < 0) { break; }

                var tsStr = json.AsSpan(pos, comma - pos).Trim();
                var closeBracket = json.IndexOf(']', comma);
                if (closeBracket < 0) { break; }

                // Read quoted value string
                var q1 = json.IndexOf('"', comma);
                if (q1 < 0 || q1 > closeBracket)
                {
                    pos = closeBracket + 1;
                    continue;
                }

                var q2 = json.IndexOf('"', q1 + 1);
                if (q2 < 0) { break; }

                var valueStr = json.AsSpan(q1 + 1, q2 - q1 - 1);

                if (double.TryParse(tsStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var unixTs) &&
                    float.TryParse(valueStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var value) &&
                    float.IsFinite(value))
                {
                    var dt = DateTimeOffset.FromUnixTimeMilliseconds((long)(unixTs * 1000))
                        .UtcDateTime;
                    result.TryAdd(dt, value);
                }

                pos = closeBracket + 1;
            }

            return result;
        }

        // -------------------------------------------------------------------------
        // PromQL query builders — same as PrometheusMetricSource
        // -------------------------------------------------------------------------

        private static string BuildCpuUsageQuery(string pod, string ns)
            => $"rate(container_cpu_usage_seconds_total{{pod=\"{pod}\",namespace=\"{ns}\"}}[1m])" +
               $" / on(pod) kube_pod_container_resource_limits{{pod=\"{pod}\",resource=\"cpu\"}}";

        private static string BuildCpuThrottleQuery(string pod, string ns)
            => $"rate(container_cpu_cfs_throttled_periods_total{{pod=\"{pod}\",namespace=\"{ns}\"}}[1m])" +
               $" / rate(container_cpu_cfs_periods_total{{pod=\"{pod}\",namespace=\"{ns}\"}}[1m])";

        private static string BuildMemoryQuery(string pod, string ns)
            => $"container_memory_working_set_bytes{{pod=\"{pod}\",namespace=\"{ns}\"}}";

        private static string BuildOomQuery(string pod, string ns)
            => $"rate(container_oom_events_total{{pod=\"{pod}\",namespace=\"{ns}\"}}[1m])";

        private static string BuildLatencyQuery(string pod, float quantile)
            => $"histogram_quantile({quantile.ToString("F2", CultureInfo.InvariantCulture)}," +
               $" rate(http_server_request_duration_seconds_bucket{{pod=\"{pod}\"}}[1m])) * 1000";

        private static string BuildRpsQuery(string pod)
            => $"rate(http_server_request_duration_seconds_count{{pod=\"{pod}\"}}[1m])";

        private static string BuildErrorRateQuery(string pod)
            => $"rate(http_server_request_duration_seconds_count{{pod=\"{pod}\",http_response_status_code=~\"5..\"}}[1m])" +
               $" / (rate(http_server_request_duration_seconds_count{{pod=\"{pod}\"}}[1m]) or vector(1))";

        private static string BuildGcGen2Query(string pod)
            => $"process_runtime_dotnet_gc_heap_size_bytes{{pod=\"{pod}\",generation=\"2\"}}";

        private static string BuildGcPauseRatioQuery(string pod)
            => $"rate(process_runtime_dotnet_gc_pause_total_seconds_total{{pod=\"{pod}\"}}[1m])";

        private static string BuildThreadPoolQuery(string pod)
            => $"process_runtime_dotnet_thread_pool_queue_length{{pod=\"{pod}\"}}";

        // -------------------------------------------------------------------------
        // Helpers
        // -------------------------------------------------------------------------

        /// <summary>
        /// Looks up the value for a given timestamp, returning 0 when the
        /// timestamp is absent (gap in data / scrape failure).
        /// </summary>
        private static float Lookup(Dictionary<DateTime, float> series, DateTime ts)
            => series.TryGetValue(ts, out var v) ? v : 0f;

        private static HttpClient BuildHttpClient(PrometheusHistoricalSourceConfig config)
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