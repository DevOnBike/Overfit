// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Net.Http.Headers;
using System.Text.Json;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;

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
    public sealed class PrometheusMetricSource : IDisposable, IRawMetricSource
    {

        private static readonly JsonSerializerOptions _jsonOptions = new()
        {
            PropertyNameCaseInsensitive = true
        };

        private readonly PrometheusMetricSourceConfig _config;
        private readonly HttpClient _http;
        private readonly int[] _seriesFromLastRead = new int[(int)MetricIndex.Count];
        private bool _disposed;

        public PrometheusMetricSource(
            PrometheusMetricSourceConfig config,
            HttpClient? httpClient = null)
        {
            ArgumentNullException.ThrowIfNull(config);
            _config = config;
            _http = httpClient ?? BuildHttpClient(config);
        }

        /// <summary>
        /// How many series the last <see cref="ReadAsync"/> obtained for one feature.
        ///
        /// <para><b>Check this.</b> Feature assembly cannot distinguish a metric that returned nothing from
        /// one that returned zero, so a query naming a metric this deployment does not export produces a
        /// column of zeroes that looks like a calm, well-behaved signal — and a detector will happily learn
        /// it. A count of 0 here, on a cluster known to be running pods, means the query is wrong, not that
        /// the system is quiet.</para>
        ///
        /// <para>Reflects the most recent read only; one instance is not meant to be read concurrently.</para>
        /// </summary>
        public int SeriesReturned(MetricIndex metric)
        {
            var index = (int)metric;

            if ((uint)index >= (uint)MetricIndex.Count)
            {
                throw new ArgumentOutOfRangeException(nameof(metric), metric, "Unknown metric.");
            }

            return _seriesFromLastRead[index];
        }

        /// <summary>
        /// Whether this deployment has a query for <paramref name="metric"/> at all. False when the
        /// configuration maps it to an empty template, i.e. the metric has no source here and the
        /// corresponding feature will never be populated.
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
        // ReadAsync — called once per scrape interval
        // ---------------------------------------------------------------------------

        /// <summary>
        ///     Waits ScrapeInterval, then issues 24 parallel instant queries.
        ///     Returns one RawMetricSeries per (pod, metric) found in Prometheus.
        /// </summary>
        public async Task<List<RawMetricSeries>> ReadAsync(CancellationToken ct)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            await Task.Delay(_config.ScrapeInterval, ct).ConfigureAwait(false);

            var tasks = new List<Task<List<RawMetricSeries>>>();
            var metricOfTask = new List<MetricIndex>();

            foreach (var dc in Enum.GetValues<DataCenter>())
            {
                var selector = BuildSelector(dc);

                for (var m = 0; m < (int)MetricIndex.Count; m++)
                {
                    var metric = (MetricIndex)m;
                    var query = BuildInstantQuery(metric, selector);

                    // No template means this deployment has no source for the feature. Issuing a query built
                    // from a metric name that is not there would return an empty result indistinguishable
                    // from a real one, so it is not issued at all and IsMapped says why.
                    if (query == null)
                    {
                        continue;
                    }

                    tasks.Add(FetchInstantAsync(query, (byte)m, dc, ct));
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

            Array.Clear(_seriesFromLastRead);

            var result = new List<RawMetricSeries>();
            for (var i = 0; i < tasks.Count; i++)
            {
                var series = await tasks[i].ConfigureAwait(false);

                _seriesFromLastRead[(int)metricOfTask[i]] += series.Count;
                result.AddRange(series);
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

        internal static List<RawMetricSeries> ParseInstantResponse(
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
                if (!metric.TryGetProperty("pod", out var podProp))
                {
                    continue;
                }

                var podName = podProp.GetString() ?? string.Empty;

                // Prometheus instant vector: "value" is the 2-tuple
                // [ unixTimeSeconds, "sampleValue" ]. Indexed JsonElement access
                // avoids LINQ (.ToArray() is banned in Sources/Main).
                if (!series.TryGetProperty("value", out var value)
                    || value.ValueKind != JsonValueKind.Array
                    || value.GetArrayLength() < 2)
                {
                    continue;
                }

                var tsMs = (long)(value[0].GetDouble() * 1000.0);
                var valStr = value[1].GetString();

                if (!float.TryParse(valStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var floatValue)
                    || !float.IsFinite(floatValue))
                {
                    floatValue = float.NaN;
                }

                var rawSeries = new RawMetricSeries
                {
                    Pod = new PodKey { DC = dc, PodName = podName },
                    MetricTypeId = metricTypeId
                };
                rawSeries.Samples.Add(new RawSample { Timestamp = tsMs, Value = floatValue });
                result.Add(rawSeries);
            }

            return result;
        }

        // ---------------------------------------------------------------------------
        // PromQL instant query builders
        // ---------------------------------------------------------------------------

        /// <summary>Label matchers for one data centre — see <see cref="PromqlCatalog.BuildSelector"/>.</summary>
        internal string BuildSelector(DataCenter dc) => PromqlCatalog.BuildSelector(_config, dc);

        /// <summary>
        /// The PromQL for one feature, or <c>null</c> when this deployment has no source for it.
        /// </summary>
        internal string? BuildInstantQuery(MetricIndex metric, string selector)
            => PromqlCatalog.Build(_config, metric, selector);

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