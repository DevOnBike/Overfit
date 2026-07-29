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
    ///     Fetches historical metric data from Prometheus for all pods matching a regex
    ///     and produces batches of RawMetricSeries ready for MonitoringPipeline.
    ///     Each call to FetchAsync returns one entry per scrape timestamp:
    ///     (ScrapeTimestampMs, List&lt;RawMetricSeries&gt;)
    ///     These are passed directly to OfflineTrainingJob:
    ///     <code>
    ///   using var source = new PrometheusHistoricalSource(config);
    ///   var scrapes = await source.FetchAsync(ct);
    ///   var result  = await job.RunAsync(scrapes, trainingConfig, log, ct);
    /// </code>
    /// </summary>
    public sealed class PrometheusHistoricalSource : IDisposable
    {
        private static readonly JsonSerializerOptions _jsonOptions = new()
        {
            PropertyNameCaseInsensitive = true
        };

        private readonly PrometheusHistoricalSourceConfig _config;
        private readonly HttpClient _http;
        private readonly int[] _seriesFromLastFetch = new int[(int)MetricIndex.Count];
        private bool _disposed;

        public PrometheusHistoricalSource(
            PrometheusHistoricalSourceConfig config,
            HttpClient? httpClient = null)
        {
            ArgumentNullException.ThrowIfNull(config);
            _config = config;
            _http = httpClient ?? BuildHttpClient(config);
        }

        /// <summary>
        /// How many series the last <see cref="FetchAsync"/> obtained for one feature. A count of 0 on a
        /// cluster known to be running pods means the query is wrong, not that the system was quiet — feature
        /// assembly cannot tell those apart, so this is where the difference is visible.
        /// </summary>
        public int SeriesReturned(MetricIndex metric)
        {
            var index = (int)metric;

            if ((uint)index >= (uint)MetricIndex.Count)
            {
                throw new ArgumentOutOfRangeException(nameof(metric), metric, "Unknown metric.");
            }

            return _seriesFromLastFetch[index];
        }

        /// <summary>
        /// Whether this deployment has a query for <paramref name="metric"/> at all. False when the
        /// configuration maps it to an empty template.
        /// </summary>
        public bool IsMapped(MetricIndex metric) => PromqlCatalog.ResolveTemplate(_config, metric).Length > 0;

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
            _http.Dispose();
        }

        // ---------------------------------------------------------------------------
        // FetchAsync — main entry point
        // ---------------------------------------------------------------------------

        /// <summary>
        ///     Fetches all 12 metrics for all pods matching PodRegex over the configured
        ///     time range and assembles them into scrape batches.
        ///     Issues 12 parallel range queries (one per metric × both DCs).
        ///     Returns one entry per scrape step covering the full Golden Window.
        /// </summary>
        public async Task<IReadOnlyList<(long ScrapeTimestampMs, List<RawMetricSeries> Series)>> FetchAsync(
            CancellationToken ct = default)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            var stepSeconds = (int)_config.Step.TotalSeconds;
            var startSec = new DateTimeOffset(_config.RangeStart.ToUniversalTime()).ToUnixTimeSeconds();
            var endSec = new DateTimeOffset(_config.RangeEnd.ToUniversalTime()).ToUnixTimeSeconds();

            // Fetch all metrics in parallel — 12 queries × 2 DCs = 24 parallel requests
            var tasks = new List<Task<List<RawMetricSeries>>>();
            var metricOfTask = new List<MetricIndex>();

            foreach (var dc in Enum.GetValues<DataCenter>())
            {
                var selector = PromqlCatalog.BuildSelector(_config, dc);

                for (var m = 0; m < (int)MetricIndex.Count; m++)
                {
                    var metric = (MetricIndex)m;
                    var query = PromqlCatalog.Build(_config, metric, selector);

                    // No template means this deployment has no source for the feature. Issuing a query built
                    // from a metric name that is not there returns an empty result indistinguishable from a
                    // real one, so it is not issued at all.
                    if (query is null)
                    {
                        continue;
                    }

                    tasks.Add(FetchMetricSeriesAsync(
                        query, (byte)m, dc, startSec, endSec, stepSeconds, ct));
                    metricOfTask.Add(metric);
                }

                // One data centre means one pass: the selector carries no dc matcher, so a second identical
                // round would double every query and every series.
                if (PromqlCatalog.IsSingleDataCenter(_config))
                {
                    break;
                }
            }

            await Task.WhenAll(tasks).ConfigureAwait(false);

            // Merge all series into a flat list then group by scrape timestamp, counting coverage on the
            // way through: a query that matched nothing is otherwise indistinguishable from a quiet system.
            Array.Clear(_seriesFromLastFetch);

            var allSeries = new List<RawMetricSeries>();
            for (var i = 0; i < tasks.Count; i++)
            {
                var series = await tasks[i].ConfigureAwait(false);

                _seriesFromLastFetch[(int)metricOfTask[i]] += series.Count;
                allSeries.AddRange(series);
            }

            return GroupByScrapeTimestamp(allSeries, startSec, endSec, stepSeconds);
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

        internal static List<RawMetricSeries> ParseRangeResponse(
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

            foreach (var series in results.EnumerateArray())
            {
                if (!series.TryGetProperty("metric", out var metric))
                {
                    continue;
                }

                // Pod name from label
                if (!metric.TryGetProperty("pod", out var podProp))
                {
                    continue;
                }
                var podName = podProp.GetString() ?? string.Empty;

                if (!series.TryGetProperty("values", out var values))
                {
                    continue;
                }

                var samples = new List<RawSample>(values.GetArrayLength());

                foreach (var point in values.EnumerateArray())
                {
                    // Prometheus matrix point: 2-tuple [ unixTimeSeconds, "sampleValue" ].
                    // Indexed JsonElement access avoids LINQ (.ToArray() is banned in Sources/Main).
                    if (point.ValueKind != JsonValueKind.Array || point.GetArrayLength() < 2)
                    {
                        continue;
                    }

                    var tsMs = (long)(point[0].GetDouble() * 1000.0);
                    var valStr = point[1].GetString();

                    if (!float.TryParse(valStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var value)
                        || !float.IsFinite(value))
                    {
                        value = float.NaN;
                    }

                    samples.Add(new RawSample { Timestamp = tsMs, Value = value });
                }

                if (samples.Count == 0)
                {
                    continue;
                }

                var rawSeries = new RawMetricSeries
                {
                    Pod = new PodKey
                    {
                        DC = dc,
                        PodName = podName
                    },
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

        private static HttpClient BuildHttpClient(PrometheusHistoricalSourceConfig config)
        {
            var client = new HttpClient
            {
                Timeout = config.HttpTimeout
            };

            client.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));

            return client;
        }
    }
}