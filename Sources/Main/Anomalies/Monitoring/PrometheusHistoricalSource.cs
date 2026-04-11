// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Net.Http.Headers;
using System.Text.Json;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    /// Fetches historical metric data from Prometheus for all pods matching a regex
    /// and produces batches of RawMetricSeries ready for MonitoringPipeline.
    ///
    /// Each call to FetchAsync returns one entry per scrape timestamp:
    ///   (ScrapeTimestampMs, List&lt;RawMetricSeries&gt;)
    ///
    /// These are passed directly to OfflineTrainingJob:
    /// <code>
    ///   using var source = new PrometheusHistoricalSource(config);
    ///   var scrapes = await source.FetchAsync(ct);
    ///   var result  = await job.RunAsync(scrapes, trainingConfig, log, ct);
    /// </code>
    /// </summary>
    public sealed class PrometheusHistoricalSource : IDisposable
    {
        private readonly PrometheusHistoricalSourceConfig _config;
        private readonly HttpClient _http;
        private bool _disposed;

        private static readonly JsonSerializerOptions _jsonOptions = new()
        {
            PropertyNameCaseInsensitive = true
        };

        public PrometheusHistoricalSource(
            PrometheusHistoricalSourceConfig config,
            HttpClient httpClient = null)
        {
            ArgumentNullException.ThrowIfNull(config);
            _config = config;
            _http   = httpClient ?? BuildHttpClient(config);
        }

        // ---------------------------------------------------------------------------
        // FetchAsync — main entry point
        // ---------------------------------------------------------------------------

        /// <summary>
        /// Fetches all 12 metrics for all pods matching PodRegex over the configured
        /// time range and assembles them into scrape batches.
        ///
        /// Issues 12 parallel range queries (one per metric × both DCs).
        /// Returns one entry per scrape step covering the full Golden Window.
        /// </summary>
        public async Task<IReadOnlyList<(long ScrapeTimestampMs, List<RawMetricSeries> Series)>> FetchAsync(
            CancellationToken ct = default)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            var stepSeconds = (int)_config.Step.TotalSeconds;
            var startSec    = new DateTimeOffset(_config.RangeStart.ToUniversalTime()).ToUnixTimeSeconds();
            var endSec      = new DateTimeOffset(_config.RangeEnd.ToUniversalTime()).ToUnixTimeSeconds();

            // Fetch all metrics in parallel — 12 queries × 2 DCs = 24 parallel requests
            var tasks = new List<Task<List<RawMetricSeries>>>();

            foreach (DataCenter dc in Enum.GetValues<DataCenter>())
            {
                var dcLabel = dc == DataCenter.West ? _config.DcWestLabel : _config.DcEastLabel;

                for (var m = 0; m < (int)MetricIndex.Count; m++)
                {
                    var metric  = (MetricIndex)m;
                    var metricId = (byte)m;
                    var query   = BuildQuery(metric, _config.PodRegex, dcLabel);
                    var capturedDc = dc;

                    tasks.Add(FetchMetricSeriesAsync(
                        query, metricId, capturedDc, startSec, endSec, stepSeconds, ct));
                }
            }

            await Task.WhenAll(tasks).ConfigureAwait(false);

            // Merge all series into a flat list then group by scrape timestamp
            var allSeries = new List<RawMetricSeries>();
            foreach (var task in tasks)
            {
                allSeries.AddRange(await task);
            }

            return GroupByScrapeTimestamp(allSeries, startSec, endSec, stepSeconds);
        }

        public void Dispose()
        {
            if (_disposed) { return; }
            _disposed = true;
            _http.Dispose();
        }

        // ---------------------------------------------------------------------------
        // Range query → List<RawMetricSeries>
        // ---------------------------------------------------------------------------

        private async Task<List<RawMetricSeries>> FetchMetricSeriesAsync(
            string promql,
            byte metricTypeId,
            DataCenter dc,
            long startSec,
            long endSec,
            int stepSeconds,
            CancellationToken ct)
        {
            var url = $"{_config.PrometheusBaseUrl}/api/v1/query_range"
                    + $"?query={Uri.EscapeDataString(promql)}"
                    + $"&start={startSec}&end={endSec}&step={stepSeconds}s";

            using var response = await _http.GetAsync(url, ct).ConfigureAwait(false);
            response.EnsureSuccessStatusCode();

            var body = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            return ParseRangeResponse(body, metricTypeId, dc);
        }

        private static List<RawMetricSeries> ParseRangeResponse(
            string json,
            byte metricTypeId,
            DataCenter dc)
        {
            var result = new List<RawMetricSeries>();

            using var doc  = JsonDocument.Parse(json);
            var root       = doc.RootElement;

            if (!root.TryGetProperty("data", out var data)) return result;
            if (!data.TryGetProperty("result", out var results)) return result;

            foreach (var series in results.EnumerateArray())
            {
                if (!series.TryGetProperty("metric", out var metric)) continue;

                // Pod name from label
                if (!metric.TryGetProperty("pod", out var podProp)) continue;
                var podName = podProp.GetString() ?? string.Empty;

                if (!series.TryGetProperty("values", out var values)) continue;

                var samples = new List<RawSample>(values.GetArrayLength());

                foreach (var point in values.EnumerateArray())
                {
                    /*
                    var arr = point.EnumerateArray().ToArray();

                    if (arr.Length < 2) continue;

                    var tsMs  = (long)(arr[0].GetDouble() * 1000.0);
                    var valStr = arr[1].GetString();

                    if (!float.TryParse(valStr, 
                            NumberStyles.Float,
                            CultureInfo.InvariantCulture,
                            out var value) || !float.IsFinite(value))
                    {
                        value = float.NaN;
                    }

                    samples.Add(new RawSample { Timestamp = tsMs, Value = value });
                    */
                }

                if (samples.Count == 0) continue;

                var rawSeries = new RawMetricSeries
                {
                    Pod          = new PodKey { DC = dc, PodName = podName },
                    MetricTypeId = metricTypeId
                };
                rawSeries.Samples.AddRange(samples);
                result.Add(rawSeries);
            }

            return result;
        }

        // ---------------------------------------------------------------------------
        // Group flat series into per-scrape batches
        // ---------------------------------------------------------------------------

        private static IReadOnlyList<(long ScrapeTimestampMs, List<RawMetricSeries> Series)>
            GroupByScrapeTimestamp(
                List<RawMetricSeries> allSeries,
                long startSec,
                long endSec,
                int stepSeconds)
        {
            // Build ordered list of scrape timestamps
            var timestamps = new List<long>();
            for (var t = startSec; t <= endSec; t += stepSeconds)
            {
                timestamps.Add(t * 1000L);
            }

            // Each scrape batch contains ALL series — TimeSeriesAligner will
            // extract the relevant window around each scrapeTimestamp
            var batches = new List<(long, List<RawMetricSeries>)>(timestamps.Count);

            foreach (var tsMs in timestamps)
            {
                // For each scrape we pass all series — aligner handles windowing
                batches.Add((tsMs, allSeries));
            }

            return batches;
        }

        // ---------------------------------------------------------------------------
        // PromQL query builders
        // ---------------------------------------------------------------------------

        private static string BuildQuery(MetricIndex metric, string podRegex, string dcLabel)
        {
            return metric switch
            {
                MetricIndex.CpuUsageRatio =>
                    $"rate(container_cpu_usage_seconds_total{{pod=~\"{podRegex}\",dc=\"{dcLabel}\"}}[1m])",

                MetricIndex.CpuThrottleRatio =>
                    $"rate(container_cpu_cfs_throttled_periods_total{{pod=~\"{podRegex}\",dc=\"{dcLabel}\"}}[1m])" +
                    $" / rate(container_cpu_cfs_periods_total{{pod=~\"{podRegex}\",dc=\"{dcLabel}\"}}[1m])",

                MetricIndex.MemoryWorkingSetBytes =>
                    $"container_memory_working_set_bytes{{pod=~\"{podRegex}\",dc=\"{dcLabel}\"}}",

                MetricIndex.OomEventsRate =>
                    $"rate(container_oom_events_total{{pod=~\"{podRegex}\",dc=\"{dcLabel}\"}}[1m])",

                MetricIndex.LatencyP50Ms =>
                    $"histogram_quantile(0.50,rate(http_server_request_duration_seconds_bucket{{pod=~\"{podRegex}\"}}[1m]))*1000",

                MetricIndex.LatencyP95Ms =>
                    $"histogram_quantile(0.95,rate(http_server_request_duration_seconds_bucket{{pod=~\"{podRegex}\"}}[1m]))*1000",

                MetricIndex.LatencyP99Ms =>
                    $"histogram_quantile(0.99,rate(http_server_request_duration_seconds_bucket{{pod=~\"{podRegex}\"}}[1m]))*1000",

                MetricIndex.RequestsPerSecond =>
                    $"rate(http_server_request_duration_seconds_count{{pod=~\"{podRegex}\"}}[1m])",

                MetricIndex.ErrorRate =>
                    $"rate(http_server_request_duration_seconds_count{{pod=~\"{podRegex}\",http_response_status_code=~\"5..\"}}[1m])" +
                    $" / (rate(http_server_request_duration_seconds_count{{pod=~\"{podRegex}\"}}[1m]) or vector(1))",

                MetricIndex.GcGen2HeapBytes =>
                    $"process_runtime_dotnet_gc_heap_size_bytes{{pod=~\"{podRegex}\",generation=\"2\"}}",

                MetricIndex.GcPauseRatio =>
                    $"rate(process_runtime_dotnet_gc_pause_total_seconds_total{{pod=~\"{podRegex}\"}}[1m])",

                MetricIndex.ThreadPoolQueueLength =>
                    $"process_runtime_dotnet_thread_pool_queue_length{{pod=~\"{podRegex}\"}}",

                _ => throw new ArgumentOutOfRangeException(nameof(metric), metric, null)
            };
        }

        private static HttpClient BuildHttpClient(PrometheusHistoricalSourceConfig config)
        {
            var client = new HttpClient { Timeout = config.HttpTimeout };
            client.DefaultRequestHeaders.Accept.Add(
                new MediaTypeWithQualityHeaderValue("application/json"));
            return client;
        }
    }
}