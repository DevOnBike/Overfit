// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Hosting;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;
using Microsoft.Extensions.Logging.Abstractions;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// The guard's own loop, driven without a cluster and without the wall clock.
    ///
    /// <para><b>The first tests in this project to construct <see cref="AnomalyGuardService"/> at all.</b>
    /// Its cycle — window fetch, evaluation, coverage reporting, floor-proposal cadence — was reachable only
    /// from <c>[LabFact]</c> diagnostics against a live Prometheus, so the one property the replay work exists
    /// to deliver had nothing checking it.</para>
    ///
    /// <para><b>Determinism here means two cold runs, not one instance called twice.</b> The guard is
    /// deliberately stateful across cycles: the same window fed twice to one instance opens an incident and
    /// then continues it, so a same-instance comparison would fail for a reason that has nothing to do with
    /// the clock. Two independently constructed services driven through the same windows and the same
    /// <c>now</c> sequence is the comparison that answers "does replaying this produce the same report".</para>
    ///
    /// <para><b>Scoped to store-less replay.</b> <see cref="AnomalyGuard"/> still falls back to the wall clock
    /// for <c>restoredAt</c> when a durable incident store is supplied; this service passes none, and these
    /// tests do not widen the claim past that.</para>
    /// </summary>
    public sealed class AnomalyGuardServiceCycleTests
    {
        /// <summary>Well in the past, so a cycle that read the wall clock instead could not produce it.</summary>
        private static readonly DateTimeOffset T0 = new(2026, 7, 30, 12, 0, 0, TimeSpan.Zero);

        private static readonly TimeSpan Cadence = TimeSpan.FromMinutes(5);

        /// <summary>
        /// Two degraded cycles then three healthy ones: enough to open an incident, continue it, and — at
        /// <see cref="IncidentTrackingOptions.Balanced"/>'s two missed cycles — close it. The lifecycle is the
        /// point, because it is what makes the comparison touch <c>IncidentTracker</c>'s cross-cycle matching
        /// rather than five independent single-cycle evaluations.
        /// </summary>
        private const int DegradedCycles = 2;
        private const int HealthyCycles = 3;

        /// <summary>
        /// <b>The parameter is load-bearing, not decorative.</b> The window a cycle asks for is derived from
        /// the <c>now</c> it was handed, so a historical <c>now</c> must produce a historical <c>end</c>. If
        /// the cycle went back to reading <see cref="DateTimeOffset.UtcNow"/> internally the source would be
        /// asked for today's window and this fails by years — which is exactly the silent corruption the
        /// seam would otherwise ship with: a fake source injected, and real wall-clock timestamps flowing
        /// through incident ages and the floor-proposal gate anyway.
        /// </summary>
        [Fact]
        public async Task TheCycleAsksForTheWindowEndingAtTheMomentItWasGiven()
        {
            var options = Options();

            using var source = new ScriptedMetricWindowSource([Synthetic(degraded: 3)]);
            using var service = Service(options, source, new NullSink());

            var result = await service.RunCycleAsync(T0, CancellationToken.None);

            Assert.NotNull(result);
            Assert.Equal(1, source.Reads);
            Assert.Equal(T0 - options.EndOffset, source.LastEnd);
            Assert.Equal(options.Window, source.LastWindow);
        }

        /// <summary>
        /// <b>The success metric: replaying the same window sequence with the same configuration twice
        /// produces the same report.</b>
        ///
        /// <para>Two independently constructed services, both cold, both store-less, both handed the very same
        /// <see cref="MetricWindow"/> objects and the very same <c>now</c> values. Every cycle's
        /// <see cref="GuardCycleResult"/> must match, and so must every incident row the sinks saw — the rows
        /// are the stricter half, since they carry the incident ids and the narratives that the counts do not.
        /// Measured: with <c>IncidentTracker</c> mutated to draw its incident ids from
        /// <c>Random.Shared</c>, the per-cycle result comparison stayed green and only the row comparison went
        /// red.</para>
        ///
        /// <para><b>What this does NOT catch, measured rather than reasoned.</b> A cycle that ignored its
        /// <c>now</c> and read <see cref="DateTimeOffset.UtcNow"/> instead leaves this test <i>green</i>:
        /// <c>observedAt</c> reaches incident ages and the floor-proposal gate, and neither is observable in a
        /// five-cycle store-less run — the emitted rows take their <c>Start</c>/<c>End</c> from the window, not
        /// from the cycle timestamp. Two runs milliseconds apart therefore agree either way. The clock is
        /// pinned by <see cref="TheCycleAsksForTheWindowEndingAtTheMomentItWasGiven"/>, which is a different
        /// property and needs its own test; determinism and clock-decoupling are not the same claim.</para>
        ///
        /// <para>The lifecycle assertions below are not decoration: without them a scripted fixture that
        /// happened to produce nothing at all would make the equality check pass while proving nothing.</para>
        /// </summary>
        [Fact]
        public async Task TwoColdRunsOfTheSameWindowsAndTimestampsProduceTheSameReport()
        {
            // Shared between both runs on purpose: identical input is the premise, and building the windows
            // twice would only be testing that the builder is deterministic.
            var script = Script();
            var moments = Moments(script.Count);

            var (resultsA, rowsA) = await Replay(script, moments);
            var (resultsB, rowsB) = await Replay(script, moments);

            for (var cycle = 0; cycle < script.Count; cycle++)
            {
                Assert.NotNull(resultsA[cycle]);
                Assert.Equal(resultsA[cycle], resultsB[cycle]);
            }

            Assert.Equal(rowsA, rowsB);

            // Non-vacuity: the runs compared above actually walked an incident through the tracker.
            var opened = 0;
            var ongoing = 0;
            var resolved = 0;

            for (var cycle = 0; cycle < resultsA.Count; cycle++)
            {
                var result = resultsA[cycle]!.Value;

                opened += result.Opened;
                ongoing += result.Ongoing;
                resolved += result.Resolved;
            }

            Assert.Equal(1, opened);
            Assert.True(ongoing > 0, $"no cycle continued the incident, so cross-cycle matching was never exercised (opened={opened}, resolved={resolved})");
            Assert.Equal(1, resolved);
        }

        /// <summary>
        /// One cold run: a fresh service, a fresh scripted source over the shared windows, a fresh sink.
        /// </summary>
        private static async Task<(IReadOnlyList<GuardCycleResult?> Results, IReadOnlyList<IncidentLogRecord> Rows)> Replay(
            IReadOnlyList<MetricWindow?> script,
            IReadOnlyList<DateTimeOffset> moments)
        {
            var sink = new CapturingSink();
            var results = new List<GuardCycleResult?>(script.Count);

            using var source = new ScriptedMetricWindowSource(script);
            using var service = Service(Options(), source, sink);

            for (var cycle = 0; cycle < script.Count; cycle++)
            {
                results.Add(await service.RunCycleAsync(moments[cycle], CancellationToken.None));
            }

            Assert.Equal(script.Count, source.Reads);

            return (results, sink.Rows);
        }

        /// <summary>Degraded for the first cycles, healthy after — one incident, opened once and closed once.</summary>
        private static IReadOnlyList<MetricWindow?> Script()
        {
            var script = new List<MetricWindow?>(DegradedCycles + HealthyCycles);

            for (var cycle = 0; cycle < DegradedCycles; cycle++)
            {
                script.Add(Synthetic(degraded: 3));
            }

            for (var cycle = 0; cycle < HealthyCycles; cycle++)
            {
                script.Add(Synthetic(degraded: -1));
            }

            return script;
        }

        private static IReadOnlyList<DateTimeOffset> Moments(int cycles)
        {
            var moments = new List<DateTimeOffset>(cycles);

            for (var cycle = 0; cycle < cycles; cycle++)
            {
                moments.Add(T0 + (Cadence * cycle));
            }

            return moments;
        }

        private static AnomalyGuardService Service(
            AnomalyGuardServiceOptions options, IMetricWindowSource source, IIncidentSink sink)
        {
            // No topology, no incident store, no learned state: the store-less replay this determinism claim
            // is scoped to. A topology would reach the live cluster, and a store would take AnomalyGuard's
            // restoredAt fallback down to the wall clock.
            return new AnomalyGuardService(
                options,
                source,
                sink,
                NullLogger<AnomalyGuardService>.Instance);
        }

        private static AnomalyGuardServiceOptions Options()
        {
            return new AnomalyGuardServiceOptions
            {
                Guard = new AnomalyGuardOptions
                {
                    Namespace = "overfit",
                    Workload = "overfit-server",
                    Grouping = IncidentGroupingOptions.Balanced with
                    {
                        Topology = TopologyWeights.SingleNode
                    },
                },
                Tracking = IncidentTrackingOptions.Balanced,
            };
        }

        /// <summary>
        /// A four-pod population with an optional degraded member, seeded so two calls with the same argument
        /// produce the same numbers. Only two channels are filled, which also puts the coverage counters to
        /// work — <c>BlindMetrics</c> is part of the result being compared.
        /// </summary>
        private static MetricWindow Synthetic(int degraded)
        {
            const int Pods = 4;
            const int Length = 60;

            var names = new List<string>(Pods);

            for (var p = 0; p < Pods; p++)
            {
                names.Add($"pod-{p}");
            }

            var window = new MetricWindow(names, Length, T0, TimeSpan.FromSeconds(15));
            var rng = new Random(20260730);

            for (var p = 0; p < Pods; p++)
            {
                var slow = p == degraded;
                var latency = window.Series(p, MetricIndex.LatencyP95Ms);
                var rps = window.Series(p, MetricIndex.RequestsPerSecond);

                for (var i = 0; i < Length; i++)
                {
                    latency[i] = (slow ? 3000.0 : 950.0) * (1.0 + ((rng.NextDouble() - 0.5) * 0.2));
                    rps[i] = 5.0 * (1.0 + ((rng.NextDouble() - 0.5) * 0.1));
                }
            }

            return window;
        }

        /// <summary>
        /// Serves scripted windows in call order and ignores the range it is asked for — the sibling of
        /// <c>LiveMonitoringPipelineTests.ScriptedRawMetricSource</c> one contract down.
        ///
        /// <para>Ignoring <c>end</c> is what makes it usable as a fixture, and recording it is what makes the
        /// clock testable: the loop's only observable use of <c>now</c> before evaluation is the window it
        /// asks for.</para>
        /// </summary>
        private sealed class ScriptedMetricWindowSource : IMetricWindowSource
        {
            private static readonly string[] None = [];

            private readonly IReadOnlyList<MetricWindow?> _script;

            public ScriptedMetricWindowSource(IReadOnlyList<MetricWindow?> script) => _script = script;

            public IReadOnlyList<string> StalePodsExcluded => None;

            /// <summary>How many windows have been served. Asserted, so an over-run cannot pass unnoticed.</summary>
            public int Reads
            {
                get; private set;
            }

            public DateTimeOffset LastEnd
            {
                get; private set;
            }

            public TimeSpan LastWindow
            {
                get; private set;
            }

            public Task<MetricWindow?> ReadAsync(
                DateTimeOffset end, TimeSpan window, CancellationToken ct = default)
            {
                LastEnd = end;
                LastWindow = window;

                if (Reads >= _script.Count)
                {
                    throw new InvalidOperationException(
                        $"The script holds {_script.Count} window(s) and a {Reads + 1}th cycle asked for one.");
                }

                return Task.FromResult(_script[Reads++]);
            }

            public void Dispose()
            {
            }
        }

        private sealed class NullSink : IIncidentSink
        {
            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
            }
        }

        /// <summary>
        /// Keeps every row a consumer would have seen, in order. The rows carry incident ids and start/end
        /// timestamps, so comparing them between two runs is the stricter half of the determinism check.
        /// </summary>
        private sealed class CapturingSink : IIncidentSink
        {
            public List<IncidentLogRecord> Rows { get; } = [];

            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
                for (var i = 0; i < rows.Length; i++)
                {
                    Rows.Add(rows[i]);
                }
            }
        }
    }
}
