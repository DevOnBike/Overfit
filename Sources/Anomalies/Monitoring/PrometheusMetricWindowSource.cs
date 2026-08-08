// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    /// Reads a rolling window of Prometheus into the shape the detectors take.
    ///
    /// <para>The bridge between <see cref="PrometheusHistoricalSource"/>, whose range is fixed at
    /// construction, and a loop that wants a fresh window every few minutes. One HTTP client is held for the
    /// lifetime of this object and lent to a per-cycle source, which is why the borrowed-client case had to
    /// be fixed there first — it used to dispose whatever it was given.</para>
    ///
    /// <para><b>The alignment here is the whole job, and it is the one this API invites you to get wrong.</b>
    /// <c>FetchAsync</c> returns one entry per scrape step and every entry holds <i>the same series list</i>.
    /// Reading a value per entry yields the last sample repeated once per step — a constant, which every
    /// detector accepts as a well-formed series and none can reject. Two separate callers in this repository
    /// made exactly that mistake. So the first entry's series are taken once, and each sample is placed on
    /// the grid by its own timestamp.</para>
    ///
    /// <para>Samples with no matching grid slot are dropped rather than snapped to the nearest one: a value
    /// nudged onto a neighbouring step is a fabricated observation, and the gap it would have left is
    /// information the detectors already know how to handle.</para>
    ///
    /// <para>The live implementation of <see cref="IMetricWindowSource"/>, and currently the only one. It
    /// already answers for any range the caller asks for — <see cref="ReadAsync"/> builds a fresh
    /// <see cref="PrometheusHistoricalSourceConfig"/> per call — so replaying a window a Prometheus still
    /// retains needs no new implementation, only a caller that supplies a past <c>end</c>.</para>
    /// </summary>
    public sealed class PrometheusMetricWindowSource : IMetricWindowSource
    {
        private readonly PrometheusHistoricalSourceConfig _template;
        private readonly HttpClient _http;
        private readonly bool _ownsHttpClient;
        private readonly int[] _seriesFromLastRead = new int[(int)MetricIndex.Count];
        private readonly KeyValuePair<string, string>[] _customQueries;
        private readonly string[] _customNames;
        private string[] _stalePods = [];
        private bool _disposed;

        /// <summary>
        /// How many scrape steps behind the freshest sample in the window a pod may fall before it is treated
        /// as gone.
        ///
        /// <para>Two, so a single missed scrape does not evict a live replica, and a deleted one disappears
        /// within a step or two of stopping. It is measured against the freshest sample ANY pod produced
        /// rather than against the end of the window, because the last slot is routinely empty for everyone —
        /// range expressions are still filling in at the window's edge, and a fixed offset from the end would
        /// declare the entire deployment gone every cycle.</para>
        /// </summary>
        private const int StaleStepTolerance = 2;

        /// <param name="template">
        /// Base configuration. Its <c>RangeStart</c>/<c>RangeEnd</c> are replaced per read; everything else —
        /// base URL, namespace, pod regex, query overrides, step — is reused.
        /// </param>
        /// <param name="httpClient">Optional shared client; one is created and owned when omitted.</param>
        /// <param name="customQueries">
        /// PromQL per metric outside the modelled set, keyed by the name findings will carry — as produced by
        /// <see cref="MetricMap.CustomQueries"/>. These cannot travel through <c>QueryOverrides</c>, which is
        /// keyed by <see cref="MetricIndex"/>, so they are issued alongside it and land in the window's
        /// custom channels.
        /// </param>
        public PrometheusMetricWindowSource(
            PrometheusHistoricalSourceConfig template,
            HttpClient? httpClient = null,
            IReadOnlyDictionary<string, string>? customQueries = null)
        {
            ArgumentNullException.ThrowIfNull(template);

            _template = template;
            _ownsHttpClient = httpClient is null;
            _http = httpClient ?? new HttpClient { Timeout = template.HttpTimeout };

            var usable = new List<KeyValuePair<string, string>>();

            if (customQueries is not null)
            {
                foreach (var (name, query) in customQueries)
                {
                    if (!string.IsNullOrWhiteSpace(name) && !string.IsNullOrWhiteSpace(query))
                    {
                        usable.Add(new KeyValuePair<string, string>(name, query));
                    }
                }
            }

            _customQueries = [.. usable];
            _customNames = new string[_customQueries.Length];

            for (var i = 0; i < _customQueries.Length; i++)
            {
                _customNames[i] = _customQueries[i].Key;
            }
        }

        /// <summary>Custom channels this source fetches, in the order the window carries them.</summary>
        public IReadOnlyList<string> CustomChannels => _customNames;

        /// <summary>
        /// Pods present in the last window's data that had stopped reporting before it ended, and were
        /// therefore left out of it.
        ///
        /// <para><b>Named rather than merely counted, because dropping a pod silently is the failure this
        /// subsystem exists to prevent.</b> A replica that vanishes from the data is usually gone — deleted,
        /// scaled down, rolled over — but it can also be one whose scraping broke while it kept serving, and
        /// those two look identical from here. The first needs no action and the second needs urgent action,
        /// so the guard reports the names and lets an operator tell them apart.</para>
        /// </summary>
        public IReadOnlyList<string> StalePodsExcluded => _stalePods;

        /// <summary>
        /// How many series the last read obtained for one metric. Zero on a cluster known to be running pods
        /// means the query is wrong, not that the system was quiet.
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
        /// Reads the window ending at <paramref name="end"/>.
        /// </summary>
        /// <param name="end">
        /// End of the window. <b>Leave a margin behind "now."</b> Rate expressions are computed over a
        /// trailing range, so samples taken at the current instant are still filling in — on a lab run that
        /// ended a load test, the last two minutes of every RED signal were decaying, and the trend family
        /// reported a cluster-wide decline that was an artefact of when the measurement stopped.
        /// </param>
        /// <param name="window">How much history to evaluate.</param>
        /// <param name="ct">Cancellation.</param>
        /// <returns><c>null</c> when no pod returned anything at all — a cluster this source cannot see is
        /// not an empty cluster, and handing back a window of NaN would read as one.</returns>
        public async Task<MetricWindow?> ReadAsync(
            DateTimeOffset end,
            TimeSpan window,
            CancellationToken ct = default)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(window, TimeSpan.Zero);

            var start = end - window;
            var config = _template with
            {
                RangeStart = start.UtcDateTime,
                RangeEnd = end.UtcDateTime,
            };

            using var source = new PrometheusHistoricalSource(config, _http);
            var batches = await source.FetchAsync(ct).ConfigureAwait(false);

            Array.Clear(_seriesFromLastRead);

            // Cleared with the rest of the per-read state, or a read that returns early leaves the previous
            // read's exclusions on display as though they were this one's.
            _stalePods = [];

            for (var m = 0; m < (int)MetricIndex.Count; m++)
            {
                _seriesFromLastRead[m] = source.SeriesReturned((MetricIndex)m);
            }

            if (batches.Count == 0)
            {
                return null;
            }

            // One entry, not all of them — see the remarks on this class and on FetchAsync.
            var series = batches[0].Series;
            var grid = new Dictionary<long, int>(batches.Count);
            var pods = new SortedSet<string>(StringComparer.Ordinal);

            for (var i = 0; i < batches.Count; i++)
            {
                grid[batches[i].ScrapeTimestampMs] = i;
            }

            // Last grid slot at which each pod produced anything, and the freshest slot anyone reached.
            //
            // A pod deleted part-way through the window keeps its samples in Prometheus for the rest of it, so
            // it is still in this data and, without this, still gets compared against its living peers and
            // named in findings. That is not a hypothetical: a rollout of twelve replicas made the guard
            // report `pods=24` and raise findings on replicas that no longer existed — including one saying a
            // series had FALLEN, which was the pod being deleted. An operator following that name finds
            // nothing there.
            var lastSlot = new Dictionary<string, int>(StringComparer.Ordinal);
            var freshest = -1;

            for (var i = 0; i < series.Count; i++)
            {
                var raw = series[i];

                if (raw.Pod.PodName.Length == 0)
                {
                    continue;
                }

                pods.Add(raw.Pod.PodName);

                for (var s = 0; s < raw.Samples.Count; s++)
                {
                    if (!grid.TryGetValue(raw.Samples[s].Timestamp, out var slot))
                    {
                        continue;
                    }

                    if (!lastSlot.TryGetValue(raw.Pod.PodName, out var seen) || slot > seen)
                    {
                        lastSlot[raw.Pod.PodName] = slot;
                    }

                    if (slot > freshest)
                    {
                        freshest = slot;
                    }
                }
            }

            if (pods.Count == 0)
            {
                return null;
            }

            var stale = new List<string>();

            foreach (var pod in pods)
            {
                // No slot at all means every sample fell outside the grid — already the "cannot be evaluated"
                // case rather than the "no longer exists" one, so it is left in and the detectors refuse it.
                if (lastSlot.TryGetValue(pod, out var slot) && slot < freshest - StaleStepTolerance)
                {
                    stale.Add(pod);
                }
            }

            _stalePods = [.. stale];

            for (var i = 0; i < stale.Count; i++)
            {
                pods.Remove(stale[i]);
            }

            // No emptiness check here, and that is a property of the rule rather than an oversight: staleness
            // is measured against the freshest sample any pod produced, so whichever pod produced it has a
            // lag of zero and can never be excluded. The set cannot be emptied by this filter, which is what
            // makes it safe to apply before anything has been evaluated. Pinned by
            // `ThePodDefiningTheFreshestSampleIsNeverExcluded`.

            var podList = new List<string>(pods);
            var index = new Dictionary<string, int>(podList.Count, StringComparer.Ordinal);

            for (var i = 0; i < podList.Count; i++)
            {
                index[podList[i]] = i;
            }

            var result = new MetricWindow(
                podList,
                batches.Count,
                DateTimeOffset.FromUnixTimeMilliseconds(batches[0].ScrapeTimestampMs),
                _template.Step,
                _customNames);

            for (var i = 0; i < series.Count; i++)
            {
                var raw = series[i];

                if (!index.TryGetValue(raw.Pod.PodName, out var pod))
                {
                    continue;
                }

                var metric = (MetricIndex)raw.MetricTypeId;

                if ((uint)raw.MetricTypeId >= (uint)MetricIndex.Count)
                {
                    continue;
                }

                var destination = result.Series(pod, metric);

                for (var s = 0; s < raw.Samples.Count; s++)
                {
                    var sample = raw.Samples[s];

                    if (grid.TryGetValue(sample.Timestamp, out var slot))
                    {
                        destination[slot] = sample.Value;
                    }
                }
            }

            await FillCustomAsync(result, index, grid, start, end, ct).ConfigureAwait(false);

            return result;
        }

        /// <summary>
        /// Issues one range query per custom metric and places its samples on the same grid.
        ///
        /// <para>A pod the main fetch did not see is skipped rather than added: the window's pod set is fixed
        /// at construction, and a custom metric is not a reason to invent a pod that reported none of the
        /// modelled features.</para>
        /// </summary>
        private async Task FillCustomAsync(
            MetricWindow window,
            Dictionary<string, int> podIndex,
            Dictionary<long, int> grid,
            DateTimeOffset start,
            DateTimeOffset end,
            CancellationToken ct)
        {
            if (_customQueries.Length == 0)
            {
                return;
            }

            var selector = PromqlCatalog.BuildSelector(_template, DataCenter.West);
            var stepSeconds = (int)_template.Step.TotalSeconds;
            var startSec = start.ToUnixTimeSeconds();
            var endSec = end.ToUnixTimeSeconds();

            for (var i = 0; i < _customQueries.Length; i++)
            {
                var (name, template) = _customQueries[i];
                var query = template.Replace(
                    PromqlCatalog.SelectorToken, selector, StringComparison.Ordinal);

                var url = $"{_template.PrometheusBaseUrl.TrimEnd('/')}/api/v1/query_range"
                          + $"?query={Uri.EscapeDataString(query)}"
                          + $"&start={startSec}&end={endSec}&step={stepSeconds}s";

                using var response = await _http.GetAsync(url, ct).ConfigureAwait(false);

                if (!response.IsSuccessStatusCode)
                {
                    // A custom metric that cannot be fetched leaves its channel NaN, which the guard already
                    // counts as blindness. Failing the whole cycle over one optional signal would be worse.
                    continue;
                }

                var body = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
                var series = PrometheusHistoricalSource.ParseRangeResponse(body, 0, DataCenter.West);

                for (var s = 0; s < series.Count; s++)
                {
                    if (!podIndex.TryGetValue(series[s].Pod.PodName, out var pod))
                    {
                        continue;
                    }

                    var destination = window.Series(pod, name);

                    for (var v = 0; v < series[s].Samples.Count; v++)
                    {
                        var sample = series[s].Samples[v];

                        if (grid.TryGetValue(sample.Timestamp, out var slot))
                        {
                            destination[slot] = sample.Value;
                        }
                    }
                }
            }
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;

            if (_ownsHttpClient)
            {
                _http.Dispose();
            }
        }
    }
}
