// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Net;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// The Prometheus-to-window adapter, and above all the trap it exists to avoid.
    ///
    /// <para><b>`FetchAsync` returns one entry per scrape step and every entry holds the same series list.</b>
    /// Reading a value per entry yields the last sample repeated once per step — a constant, which every
    /// detector accepts as a well-formed series and none can reject: a trend over a constant is exactly zero
    /// by construction, and a peer comparison between constants derives significance from a sample count that
    /// does not exist. Two separate callers in this repository made that mistake, and it was only caught by
    /// comparing a recorded fixture against a direct query. This test is what should have caught it.</para>
    /// </summary>
    public sealed class PrometheusMetricWindowSourceTests
    {
        private static readonly DateTimeOffset End = new(2026, 7, 30, 12, 0, 0, TimeSpan.Zero);

        /// <summary>The regression: distinct samples must arrive distinct.</summary>
        [Fact]
        public async Task SamplesArriveAsATimeSeries_NotAsAConstant()
        {
            double[] latency = [900.0, 950.0, 1000.0, 1050.0, 1100.0];

            using var source = Source(("pod-a", latency));
            var window = await source.ReadAsync(End, TimeSpan.FromSeconds(60), TestContext.Current.CancellationToken);

            Assert.NotNull(window);

            var series = window.Series(0, MetricIndex.LatencyP95Ms);
            var distinct = new SortedSet<double>();

            for (var i = 0; i < series.Length; i++)
            {
                if (double.IsFinite(series[i]))
                {
                    distinct.Add(series[i]);
                }
            }

            Assert.Equal(latency.Length, distinct.Count);
            Assert.Contains(900.0, distinct);
            Assert.Contains(1100.0, distinct);
        }

        /// <summary>
        /// A step with no sample stays a gap. Snapping it to a neighbour would fabricate an observation, and
        /// the gap is information every detector here already knows how to drop.
        /// </summary>
        [Fact]
        public async Task AStepWithNoSampleStaysNaN()
        {
            using var source = Source(("pod-a", [900.0, double.NaN, 1000.0]), skipIndex: 1);
            var window = await source.ReadAsync(End, TimeSpan.FromSeconds(60), TestContext.Current.CancellationToken);

            Assert.NotNull(window);

            var series = window.Series(0, MetricIndex.LatencyP95Ms);

            Assert.Equal(900.0, series[0]);
            Assert.True(double.IsNaN(series[1]), "a step the series had no sample for must remain a gap");
            Assert.Equal(1000.0, series[2]);
        }

        /// <summary>
        /// Nothing returned is not an empty cluster — it is a cluster this source cannot see, and a window of
        /// NaN would read as the former.
        /// </summary>
        [Fact]
        public async Task NoSeriesAtAllYieldsNull_NotAWindowOfNaN()
        {
            using var source = Source();
            var window = await source.ReadAsync(End, TimeSpan.FromSeconds(60), TestContext.Current.CancellationToken);

            Assert.Null(window);
        }

        [Fact]
        public async Task PodsAreIndexedInAStableOrder()
        {
            using var source = Source(
                ("pod-c", [1.0, 2.0]), ("pod-a", [3.0, 4.0]), ("pod-b", [5.0, 6.0]));

            var window = await source.ReadAsync(End, TimeSpan.FromSeconds(60), TestContext.Current.CancellationToken);

            Assert.NotNull(window);
            Assert.Equal(["pod-a", "pod-b", "pod-c"], window.Pods);
        }

        /// <summary>
        /// A replica that stopped reporting part-way through the window is left out of it.
        ///
        /// <para><b>The defect this pins was seen in the lab, not imagined.</b> Rolling twelve replicas made
        /// the guard report <c>pods=24</c> for a full window and raise findings naming replicas that had
        /// already been deleted — including one reporting that a series had FALLEN, which was the pod being
        /// terminated. Prometheus keeps a deleted pod's samples for the rest of the window, so from inside the
        /// data a dead replica is indistinguishable from a live one unless recency is checked.</para>
        /// </summary>
        [Fact]
        public async Task APodThatStoppedReportingIsLeftOutOfTheWindow()
        {
            using var source = Source(
                ("pod-live", [1.0, 2.0, 3.0, 4.0, 5.0]),
                ("pod-deleted", [1.0, 2.0]));

            var window = await source.ReadAsync(End, TimeSpan.FromSeconds(60), TestContext.Current.CancellationToken);

            Assert.NotNull(window);
            Assert.Equal(["pod-live"], window.Pods);
            Assert.Equal(["pod-deleted"], source.StalePodsExcluded);
        }

        /// <summary>
        /// One missed scrape is not a death. The tolerance exists because evicting a live replica for a gap
        /// would hide it exactly when it is least healthy, and a hidden pod looks like a healthy cluster.
        /// </summary>
        [Fact]
        public async Task APodThatMissedTheLastScrapeIsKept()
        {
            using var source = Source(
                ("pod-a", [1.0, 2.0, 3.0, 4.0, 5.0]),
                ("pod-b", [1.0, 2.0, 3.0, 4.0]));

            var window = await source.ReadAsync(End, TimeSpan.FromSeconds(60), TestContext.Current.CancellationToken);

            Assert.NotNull(window);
            Assert.Equal(["pod-a", "pod-b"], window.Pods);
            Assert.Empty(source.StalePodsExcluded);
        }

        /// <summary>
        /// The invariant that makes the filter safe to apply before anything is evaluated: recency is measured
        /// against the freshest sample ANY pod produced, so the pod that produced it has zero lag and cannot
        /// be excluded. The filter therefore cannot empty the deployment, however far behind everything is —
        /// which matters because the last grid slot is routinely empty for every pod at once, and a rule
        /// measured from the end of the window instead would declare the whole cluster gone every cycle.
        /// </summary>
        [Fact]
        public async Task ThePodDefiningTheFreshestSampleIsNeverExcluded()
        {
            using var source = Source(
                ("pod-a", [1.0]),
                ("pod-b", [1.0]),
                ("pod-c", [1.0]));

            var window = await source.ReadAsync(End, TimeSpan.FromSeconds(60), TestContext.Current.CancellationToken);

            Assert.NotNull(window);
            Assert.Equal(["pod-a", "pod-b", "pod-c"], window.Pods);
            Assert.Empty(source.StalePodsExcluded);
        }

        /// <summary>The exclusion list belongs to the last read, not to every read since the first.</summary>
        [Fact]
        public async Task TheExclusionListIsClearedBetweenReads()
        {
            using var source = Source(
                ("pod-live", [1.0, 2.0, 3.0, 4.0, 5.0]),
                ("pod-deleted", [1.0, 2.0]));

            await source.ReadAsync(End, TimeSpan.FromSeconds(60), TestContext.Current.CancellationToken);

            Assert.NotEmpty(source.StalePodsExcluded);

            using var clean = Source(("pod-live", [1.0, 2.0, 3.0, 4.0, 5.0]));

            await clean.ReadAsync(End, TimeSpan.FromSeconds(60), TestContext.Current.CancellationToken);

            Assert.Empty(clean.StalePodsExcluded);
        }

        /// <summary>A borrowed client must survive the source that used it.</summary>
        [Fact]
        public async Task ASharedHttpClientIsNotDisposedByTheSource()
        {
            var handler = new StubHandler([("pod-a", [1.0, 2.0])], skipIndex: -1);
            using var client = new HttpClient(handler);

            using (var source = new PrometheusMetricWindowSource(Template(), client))
            {
                await source.ReadAsync(End, TimeSpan.FromSeconds(60), TestContext.Current.CancellationToken);
            }

            // Would throw ObjectDisposedException if the source had disposed what it was lent.
            using var again = new PrometheusMetricWindowSource(Template(), client);

            Assert.NotNull(await again.ReadAsync(End, TimeSpan.FromSeconds(60), TestContext.Current.CancellationToken));
        }

        private static PrometheusMetricWindowSource Source(
            params (string Pod, double[] Values)[] series)
        {
            return Source(-1, series);
        }

        private static PrometheusMetricWindowSource Source(
            (string Pod, double[] Values) single,
            int skipIndex)
        {
            return Source(skipIndex, single);
        }

        private static PrometheusMetricWindowSource Source(
            int skipIndex,
            params (string Pod, double[] Values)[] series)
        {
            return new PrometheusMetricWindowSource(
                Template(), new HttpClient(new StubHandler(series, skipIndex)));
        }

        private static PrometheusHistoricalSourceConfig Template()
        {
            return PrometheusHistoricalSourceConfig.ForOverfitServer(
                "http://127.0.0.1:9090", "pod-.*", "overfit",
                End.AddMinutes(-1).UtcDateTime, End.UtcDateTime,
                step: TimeSpan.FromSeconds(15));
        }

        /// <summary>
        /// Answers every query with a Prometheus matrix response, but only for the latency-p95 query — the
        /// others come back empty, which is also what a real deployment missing those metrics looks like.
        /// </summary>
        private sealed class StubHandler : HttpMessageHandler
        {
            private readonly (string Pod, double[] Values)[] _series;
            private readonly int _skipIndex;

            public StubHandler((string Pod, double[] Values)[] series, int skipIndex)
            {
                _series = series;
                _skipIndex = skipIndex;
            }

            protected override Task<HttpResponseMessage> SendAsync(
                HttpRequestMessage request,
                CancellationToken cancellationToken)
            {
                var query = request.RequestUri?.Query ?? string.Empty;
                var isLatencyP95 = query.Contains("0.95", StringComparison.Ordinal);

                var body = isLatencyP95 ? Matrix() : "{\"status\":\"success\",\"data\":{\"result\":[]}}";

                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent(body),
                });
            }

            /// <summary>Timestamps land on the same 15 s grid the config declares.</summary>
            private string Matrix()
            {
                var baseSeconds = new DateTimeOffset(End.AddMinutes(-1).UtcDateTime, TimeSpan.Zero)
                    .ToUnixTimeSeconds();
                var results = new List<string>(_series.Length);

                foreach (var (pod, values) in _series)
                {
                    var points = new List<string>(values.Length);

                    for (var i = 0; i < values.Length; i++)
                    {
                        if (i == _skipIndex)
                        {
                            continue;
                        }

                        points.Add(string.Create(
                            CultureInfo.InvariantCulture,
                            $"[{baseSeconds + (i * 15)},\"{values[i].ToString("R", CultureInfo.InvariantCulture)}\"]"));
                    }

                    results.Add(
                        $"{{\"metric\":{{\"pod\":\"{pod}\"}},\"values\":[{string.Join(",", points)}]}}");
                }

                return $"{{\"status\":\"success\",\"data\":{{\"resultType\":\"matrix\",\"result\":[{string.Join(",", results)}]}}}}";
            }
        }
    }
}
