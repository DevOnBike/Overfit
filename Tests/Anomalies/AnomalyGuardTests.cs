// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Incidents.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// The composed cycle — and, in the last two tests, the guard running against a <b>recorded window of the
    /// real cluster</b> rather than against a simulator.
    ///
    /// <para>That distinction is the point of having recorded it. Every threshold in this pipeline was tuned
    /// against a generator that was found to be wrong three separate times in one day; a fixture nobody tuned
    /// anything against is the only thing here that can contradict it.</para>
    /// </summary>
    public sealed class AnomalyGuardTests
    {
        private static readonly DateTimeOffset T0 = new(2026, 7, 30, 12, 0, 0, TimeSpan.Zero);

        /// <summary>Nothing wrong with the population means nothing to tell anyone about.</summary>
        [Fact]
        public void AHealthyPopulationOpensNoIncident()
        {
            var result = Guard(new CapturingSink())
                .RunCycle(Synthetic(pods: 4, length: 60, degraded: -1), T0);

            Assert.Equal(0, result.Opened);
        }

        /// <summary>
        /// The distinction the coverage channel exists for: a metric nobody reports produces no findings,
        /// which is indistinguishable from health everywhere below this. The count is asserted exactly rather
        /// than as "more than none", because a vague assertion here would pass on a guard that had stopped
        /// looking at anything at all.
        /// </summary>
        [Fact]
        public void AMetricNoPodReportsIsCountedAsBlindness_NotAsHealth()
        {
            var result = Guard(new CapturingSink())
                .RunCycle(Synthetic(pods: 4, length: 60, degraded: -1), T0);

            // Synthetic fills exactly two channels — latency p95 and request rate. Every other one is
            // genuinely absent and must be reported as such.
            Assert.Equal((int)MetricIndex.Count - 2, result.BlindMetrics);
            Assert.True(result.IsPartiallyBlind);
            Assert.Equal(0, result.PartialMetrics);
        }

        /// <summary>A problem that persists is opened once and then updated — the deployability property.</summary>
        [Fact]
        public void APersistingProblemOpensOnceAcrossCycles()
        {
            var sink = new CapturingSink();
            var guard = Guard(sink);

            var opened = 0;

            for (var cycle = 0; cycle < 6; cycle++)
            {
                var result = guard.RunCycle(
                    Synthetic(pods: 4, length: 60, degraded: 3), T0.AddMinutes(5 * cycle));

                opened += result.Opened;
            }

            Assert.Equal(1, opened);
        }

        [Fact]
        public void RejectsNulls()
        {
            var sink = new CapturingSink();

            Assert.Throws<ArgumentNullException>(
                () => new AnomalyGuard(null!, sink, IncidentTrackingOptions.Balanced));
            Assert.Throws<ArgumentNullException>(
                () => new AnomalyGuard(new AnomalyGuardOptions(), null!, IncidentTrackingOptions.Balanced));
            Assert.Throws<ArgumentNullException>(() => Guard(sink).RunCycle(null!, T0));
        }

        // ---- against the recorded cluster ------------------------------------------------------------

        /// <summary>
        /// Runs the guard over the recorded lab window. Not asserted on incident counts — the honest bound on
        /// four pods and one fault is not known — but the cycle must complete, report, and see the metrics
        /// the recording actually contains.
        /// </summary>
        [Fact]
        public void RunsAgainstTheRecordedLabWindow()
        {
            if (!LabWindowFixture.Exists)
            {
                return;
            }

            var (window, faulted) = LabWindowFixture.Load();
            var sink = new CapturingSink();

            var result = Guard(sink).RunCycle(window, T0);

            Assert.True(window.Pods.Count >= 3, "a peer group needs at least three members");
            Assert.Single(faulted);

            // CpuThrottleRatio exists only on containers carrying a CPU limit, so partial coverage here is
            // the CFS property rather than a fault — but it must be visible as partial, not as complete.
            Assert.True(result.PartialMetrics > 0);
            Assert.Equal(1, sink.Calls);
        }

        /// <summary>
        /// The labelled half of the fixture: whatever else the guard says about that window, the replica that
        /// was deliberately throttled must be among the pods it names.
        /// </summary>
        [Fact]
        public void NamesTheDeliberatelyDegradedReplica()
        {
            if (!LabWindowFixture.Exists)
            {
                return;
            }

            var (window, faulted) = LabWindowFixture.Load();
            var sink = new CapturingSink();

            Guard(sink).RunCycle(window, T0);

            var accused = new SortedSet<string>(StringComparer.Ordinal);

            foreach (var row in sink.Rows)
            {
                if (row.Kind == IncidentLogRecordKind.Finding && row.NamesAPod)
                {
                    accused.Add(row.Pod);
                }
            }

            Assert.Contains(faulted[0], accused);
        }

        private static AnomalyGuard Guard(IIncidentSink sink)
        {
            return new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "overfit",
                    Workload = "overfit-server",
                    Grouping = IncidentGroupingOptions.Balanced with
                    {
                        Topology = TopologyWeights.SingleNode
                    },
                },
                sink,
                IncidentTrackingOptions.Balanced);
        }

        /// <summary>
        /// A small population with an optional degraded member. Only a few channels are filled, which also
        /// exercises the blindness counter.
        /// </summary>
        private static MetricWindow Synthetic(int pods, int length, int degraded)
        {
            var names = new List<string>(pods);

            for (var p = 0; p < pods; p++)
            {
                names.Add($"pod-{p}");
            }

            var window = new MetricWindow(names, length, T0, TimeSpan.FromSeconds(15));
            var rng = new Random(20260730);

            for (var p = 0; p < pods; p++)
            {
                var slow = p == degraded;
                var latency = window.Series(p, MetricIndex.LatencyP95Ms);
                var rps = window.Series(p, MetricIndex.RequestsPerSecond);

                for (var i = 0; i < length; i++)
                {
                    latency[i] = (slow ? 3000.0 : 950.0) * (1.0 + ((rng.NextDouble() - 0.5) * 0.2));
                    rps[i] = 5.0 * (1.0 + ((rng.NextDouble() - 0.5) * 0.1));
                }
            }

            return window;
        }

        private sealed class CapturingSink : IIncidentSink
        {
            public List<IncidentLogRecord> Rows { get; } = [];

            public int Calls
            {
                get; private set;
            }

            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
                Calls++;

                for (var i = 0; i < rows.Length; i++)
                {
                    Rows.Add(rows[i]);
                }
            }
        }
    }
}
