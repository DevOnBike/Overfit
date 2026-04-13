// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Net.Http.Headers;
using System.Text.Json;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    ///     Real-time metric scraper — issues instant PromQL queries every ScrapeInterval
    ///     and returns a flat List&lt;RawMetricSeries&gt; ready for MonitoringPipeline.Process().
    ///     Each ReadAsync call issues 12 × 2DC = 24 parallel instant queries
    ///     and returns one RawMetricSeries per (pod, metric) combination found.
    ///     Usage in inference loop:
    ///     <code>
    ///   using var source = new PrometheusMetricSource(config);
    ///   while (!ct.IsCancellationRequested)
    ///   {
    ///       var series        = await source.ReadAsync(ct);
    ///       var scrapeEndMs   = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
    ///       var scaledResult  = pipeline.Process(series, scrapeEndMs - windowMs, scrapeEndMs);
    ///       var errors        = fleetModel.ReconstructionError(scaledResult.FleetBaseline);
    ///       alertEngine.Evaluate(errors, scaledResult.PodIndex);
    ///   }
    /// </code>
    /// </summary>
    public sealed class PrometheusMetricSource : IDisposable
    {

        private static readonly JsonSerializerOptions _jsonOptions = new()
        {
            PropertyNameCaseInsensitive = true
        };

        private readonly PrometheusMetricSourceConfig _config;
        private readonly HttpClient _http;
        private bool _disposed;

        public PrometheusMetricSource(
            PrometheusMetricSourceConfig config,
            HttpClient httpClient = null)
        {
            ArgumentNullException.ThrowIfNull(config);
            _config = config;
            _http = httpClient ?? BuildHttpClient(config);
        }

        public void Dispose()
        {
            if (_disposed) { return; }
            _disposed = true;
            _http.Dispose();
        }

        // ---------------------------------------------------------------------------
        // ReadAsync — called once per scrape interval
        // ---------------------------------------------------------------------------

        /// <summary>
        ///     Waits ScrapeInterval, then issues 24 parallel instant queries.
        ///     Returns one RawMetricSeries per (pod, metric) found in Prometheus.
        /// </summary>
        public async Task<List<RawMetricSeries>> ReadAsync(CancellationToken ct = default)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            await Task.Delay(_config.ScrapeInterval, ct).ConfigureAwait(false);

            var tasks = new List<Task<List<RawMetricSeries>>>();

            foreach (var dc in Enum.GetValues<DataCenter>())
            {
                var dcLabel = dc == DataCenter.West ? _config.DcWestLabel : _config.DcEastLabel;

                for (var m = 0; m < (int)MetricIndex.Count; m++)
                {
                    var metric = (MetricIndex)m;
                    var metricId = (byte)m;
                    var query = BuildInstantQuery(metric, _config.PodRegex, dcLabel);
                    var capturedDc = dc;

                    tasks.Add(FetchInstantAsync(query, metricId, capturedDc, ct));
                }
            }

            await Task.WhenAll(tasks).ConfigureAwait(false);

            var result = new List<RawMetricSeries>();
            foreach (var task in tasks)
            {
                result.AddRange(await task);
            }

            return result;
        }

        // ---------------------------------------------------------------------------
        // Instant query → List<RawMetricSeries>
        // ---------------------------------------------------------------------------

        private async Task<List<RawMetricSeries>> FetchInstantAsync(
            string promql,
            byte metricTypeId,
            DataCenter dc,
            CancellationToken ct)
        {
            var url = $"{_config.PrometheusBaseUrl}/api/v1/query"
                      + $"?query={Uri.EscapeDataString(promql)}";

            using var response = await _http.GetAsync(url, ct).ConfigureAwait(false);
            response.EnsureSuccessStatusCode();

            var body = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            return ParseInstantResponse(body, metricTypeId, dc);
        }

        private static List<RawMetricSeries> ParseInstantResponse(
            string json,
            byte metricTypeId,
            DataCenter dc)
        {
            var result = new List<RawMetricSeries>();

            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            if (!root.TryGetProperty("data", out var data))
            {
                return result;
            }
            if (!data.TryGetProperty("result", out var results))
            {
                return result;
            }

            var nowMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();

            foreach (var series in results.EnumerateArray())
            {
                /*
                if (!series.TryGetProperty("metric", out var metric)) continue;
                if (!metric.TryGetProperty("pod", out var podProp)) continue;

                var podName = podProp.GetString() ?? string.Empty;

                if (!series.TryGetProperty("value", out var value)) continue;

                var arr = value.EnumerateArray().ToArray();
                if (arr.Length < 2) continue;

                var tsMs   = (long)(arr[0].GetDouble() * 1000.0);
                var valStr = arr[1].GetString();

                if (!float.TryParse(valStr,
                        NumberStyles.Float,
                        CultureInfo.InvariantCulture,
                        out var floatValue) || !float.IsFinite(floatValue))
                {
                    floatValue = float.NaN;
                }

                var rawSeries = new RawMetricSeries
                {
                    Pod          = new PodKey { DC = dc, PodName = podName },
                    MetricTypeId = metricTypeId
                };
                rawSeries.Samples.Add(new RawSample { Timestamp = tsMs, Value = floatValue });
                result.Add(rawSeries);
                */

            }

            return result;
        }

        // ---------------------------------------------------------------------------
        // PromQL instant query builders
        // ---------------------------------------------------------------------------

        private static string BuildInstantQuery(MetricIndex metric, string podRegex, string dcLabel)
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

        private static HttpClient BuildHttpClient(PrometheusMetricSourceConfig config)
        {
            var client = new HttpClient
            {
                Timeout = config.HttpTimeout
            };
            client.DefaultRequestHeaders.Accept.Add(
            new MediaTypeWithQualityHeaderValue("application/json"));
            return client;
        }
    }
}