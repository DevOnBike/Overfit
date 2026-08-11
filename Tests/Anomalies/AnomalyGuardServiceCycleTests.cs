// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Hosting;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tests.TestSupport;
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
    /// <para><b>Was scoped to store-less replay, and the reason given for that was wrong.</b> This paragraph
    /// used to say the service "passes none" — no durable store — which it never did: it passes the store it
    /// is given. The real limit was narrower and worse, and is now fixed: the service accepted an
    /// <c>IClock</c> and did not hand it to <see cref="AnomalyGuard"/>, so restore fell back to the wall clock
    /// whatever a caller injected. <see cref="RestoreJudgesIncidentAgeByTheInjectedClock"/> pins the fix;
    /// without it a store-backed replay begins from a state that depends on what time it was run at.</para>
    ///
    /// <para><b>All three cycle outcomes are driven here, not only the one a healthy fixture produces.</b>
    /// <see cref="GuardCycleOutcome"/> exists to keep "the source saw nothing" apart from "the cycle threw";
    /// a suite that only ever completes cycles would let those two swap places — or let a caught exception
    /// report itself as a quiet cluster — without going red.</para>
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

            var outcome = await service.RunCycleAsync(T0, CancellationToken.None);

            Assert.Equal(GuardCycleKind.Completed, outcome.Kind);
            Assert.Equal(1, source.Reads);
            Assert.Equal(T0 - options.EndOffset, source.LastEnd);
            Assert.Equal(options.Window, source.LastWindow);
        }

        /// <summary>
        /// A cycle whose source returned no window at all is <see cref="GuardCycleKind.Blind"/>, and carries
        /// nothing to read.
        ///
        /// <para>The kind is half the assertion; <see cref="GuardCycleOutcome.TryGetResult"/> answering
        /// <c>false</c> is the other half, because the failure this contract was written against is a caller
        /// reading an all-zero result and reporting a cluster nobody could see as a quiet one.</para>
        ///
        /// <para><b>Measured:</b> with the blind return mutated to <c>GuardCycleOutcome.Completed(default)</c>
        /// this goes red on both assertions.</para>
        /// </summary>
        [Fact]
        public async Task ACycleWhoseSourceSawNothingIsBlindAndCarriesNoResult()
        {
            using var source = new ScriptedMetricWindowSource([null]);
            using var service = Service(Options(), source, new NullSink());

            var outcome = await service.RunCycleAsync(T0, CancellationToken.None);

            Assert.Equal(GuardCycleKind.Blind, outcome.Kind);
            Assert.False(outcome.TryGetResult(out var result));
            Assert.Equal(default(GuardCycleResult), result);
            Assert.Equal(1, source.Reads);
        }

        /// <summary>
        /// A cycle whose source throws is <see cref="GuardCycleKind.Failed"/> — reported, not rethrown — and
        /// the next cycle still evaluates.
        ///
        /// <para>Both halves matter to a replay driver: it needs to count the crashing cycles separately from
        /// the blind ones (a run of either produces the same incident totals as a quiet run), and it needs the
        /// loop to survive one, because a day of history with one unreachable minute in it should not end the
        /// replay.</para>
        ///
        /// <para><b>Measured:</b> with the failed return mutated to <c>GuardCycleOutcome.Blind</c> the kind
        /// assertion goes red.</para>
        /// </summary>
        [Fact]
        public async Task ACycleThatThrowsIsFailedAndTheNextOneStillRuns()
        {
            using var source = new ThrowingFirstReadSource(Synthetic(degraded: 3));
            using var service = Service(Options(), source, new NullSink());

            var failed = await service.RunCycleAsync(T0, CancellationToken.None);

            Assert.Equal(GuardCycleKind.Failed, failed.Kind);
            Assert.False(failed.TryGetResult(out var none));
            Assert.Equal(default(GuardCycleResult), none);

            var next = await service.RunCycleAsync(T0 + Cadence, CancellationToken.None);

            // Completed is only ever returned after the window was evaluated, so this is the assertion that
            // the service kept working rather than merely not throwing.
            Assert.Equal(GuardCycleKind.Completed, next.Kind);
            Assert.True(next.TryGetResult(out _));
            Assert.Equal(2, source.Reads);
        }

        /// <summary>
        /// An outcome that ran is never equal to one that did not, and no two kinds are equal to each other.
        ///
        /// <para><b>This is the guard on the trap the contract was designed around</b>, which until now was
        /// held up by nothing but the code happening to be right today: a blind or failed cycle carries a
        /// <c>default</c> result, so an equality that compared only the result — or a
        /// <see cref="GuardCycleKind"/> whose zero value stopped being <see cref="GuardCycleKind.Blind"/> —
        /// would make "nothing ran" compare equal to "ran and found nothing". Two replays compared
        /// outcome-for-outcome would then agree while one of them was blind throughout.</para>
        ///
        /// <para>The positive assertions are not padding: without them an <c>Equals</c> mutated to return
        /// <c>false</c> unconditionally would satisfy every inequality above it.</para>
        ///
        /// <para><b>Measured:</b> with <c>GuardCycleOutcome.Equals</c> mutated to compare only the result
        /// (dropping <c>Kind</c>) the first three assertions go red; mutated to a constant <c>false</c>, the
        /// positive ones do.</para>
        /// </summary>
        [Fact]
        public void AnOutcomeThatRanIsNeverEqualToOneThatDidNot()
        {
            var quiet = GuardCycleOutcome.Completed(default);
            var busy = GuardCycleOutcome.Completed(new GuardCycleResult(1, 1, 1, 0, 0, 0, 0));

            Assert.NotEqual(GuardCycleOutcome.Blind, quiet);
            Assert.NotEqual(GuardCycleOutcome.Failed, quiet);
            Assert.NotEqual(GuardCycleOutcome.Failed, GuardCycleOutcome.Blind);
            Assert.True(GuardCycleOutcome.Blind != quiet);
            Assert.False(GuardCycleOutcome.Blind == quiet);

            Assert.Equal(GuardCycleOutcome.Blind, GuardCycleOutcome.Blind);
            Assert.Equal(GuardCycleOutcome.Failed, GuardCycleOutcome.Failed);
            Assert.Equal(quiet, GuardCycleOutcome.Completed(default));
            Assert.NotEqual(quiet, busy);

            // An uninitialised outcome — one element of an array a driver has not filled yet — must not read
            // as a completed cycle, which is why Blind is the zero value.
            Assert.Equal(GuardCycleKind.Blind, default(GuardCycleOutcome).Kind);
            Assert.False(default(GuardCycleOutcome).TryGetResult(out _));
        }

        /// <summary>
        /// <b>The success metric: replaying the same window sequence with the same configuration twice
        /// produces the same report.</b>
        ///
        /// <para>Two independently constructed services, both cold, both store-less, both handed the very same
        /// <see cref="MetricWindow"/> objects and the very same <c>now</c> values. Every cycle's
        /// <see cref="GuardCycleOutcome"/> must match — kind as well as counts, so a run that went blind where
        /// the other completed is a difference rather than a pair of matching zeros — and so must every
        /// incident row the sinks saw; the rows
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

            var (outcomesA, rowsA) = await Replay(script, moments);
            var (outcomesB, rowsB) = await Replay(script, moments);

            for (var cycle = 0; cycle < script.Count; cycle++)
            {
                Assert.Equal(GuardCycleKind.Completed, outcomesA[cycle].Kind);
                Assert.Equal(outcomesA[cycle], outcomesB[cycle]);
            }

            Assert.Equal(rowsA, rowsB);

            // Non-vacuity: the runs compared above actually walked an incident through the tracker.
            var opened = 0;
            var ongoing = 0;
            var resolved = 0;

            for (var cycle = 0; cycle < outcomesA.Count; cycle++)
            {
                Assert.True(outcomesA[cycle].TryGetResult(out var result));

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
        private static async Task<(IReadOnlyList<GuardCycleOutcome> Outcomes, IReadOnlyList<IncidentLogRecord> Rows)> Replay(
            IReadOnlyList<MetricWindow?> script,
            IReadOnlyList<DateTimeOffset> moments)
        {
            var sink = new CapturingSink();
            var outcomes = new List<GuardCycleOutcome>(script.Count);

            using var source = new ScriptedMetricWindowSource(script);
            using var service = Service(Options(), source, sink);

            for (var cycle = 0; cycle < script.Count; cycle++)
            {
                outcomes.Add(await service.RunCycleAsync(moments[cycle], CancellationToken.None));
            }

            Assert.Equal(script.Count, source.Reads);

            return (outcomes, sink.Rows);
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
            AnomalyGuardServiceOptions options,
            IMetricWindowSource source,
            IIncidentSink sink,
            IIncidentStore? store = null,
            IClock? clock = null)
        {
            // No topology by default, and no store: a topology would reach the live cluster. The store and
            // clock are supplied only by the restore test, which is the one case where either is observable —
            // restore happens once, in the constructor, and nothing on the cycle path reads them.
            return new AnomalyGuardService(
                options,
                source,
                sink,
                NullLogger<AnomalyGuardService>.Instance,
                store: store,
                clock: clock);
        }

        /// <summary>
        /// The clock this service is given must be the one its guard judges restored incidents against.
        ///
        /// <para><b>What was wrong.</b> The service took an <see cref="IClock"/>, used it for the cycle
        /// timestamp, and constructed <see cref="AnomalyGuard"/> without it — so the guard kept its own
        /// <c>SystemClock</c>. With <c>restoredAt: null</c>, that clock is what
        /// <c>IncidentTracker.Restore</c> compares against <c>MaxRestoredIncidentAge</c>
        /// (<c>now - saved.LastSeen &gt; maxAge</c>), so a caller that injected a clock still had its restore
        /// decided by the wall clock. Nothing could see it: the cycle results were unaffected, only the state
        /// the run started from.</para>
        ///
        /// <para><b>Why the store straddles the bound rather than sitting inside it.</b> A fake clock this far
        /// from the wall clock makes every stored incident look ancient, so an assertion of "some were
        /// dropped" would pass on the broken code too — the wall clock drops them as well, for the wrong
        /// reason. Two records either side of the two-hour bound pin it from both directions: the answer is 1
        /// only if the bound was applied AND applied against <see cref="T0"/>. Unwired, the wall clock drops
        /// both and this reads 0; with no age bound at all it would read 2.</para>
        /// </summary>
        [Fact]
        public void RestoreJudgesIncidentAgeByTheInjectedClock()
        {
            var options = Options();
            var maxAge = options.Guard.MaxRestoredIncidentAge;

            // The premise this test's discrimination rests on, asserted rather than assumed: T0 must be
            // further behind the wall clock than the age bound, or the wall-clock path would keep the recent
            // record too and a broken wiring would pass. If T0 is ever moved forward, this fails here and
            // says why instead of going quietly green.
            Assert.True(
                DateTimeOffset.UtcNow - T0 > maxAge,
                $"T0 ({T0:O}) is within {maxAge} of now, so this test can no longer tell the injected clock "
                + "from the wall clock. Move T0 further into the past.");

            var store = new MemoryStore();

            store.Save(IncidentStateFormat.Write(
                [
                    Saved(id: 1, lastSeen: T0 - (maxAge / 2)),    // inside the bound as of T0
                    Saved(id: 2, lastSeen: T0 - (maxAge * 2)),    // outside it as of T0
                ],
                nextId: 3));

            using var source = new ScriptedMetricWindowSource([]);
            using var service = Service(options, source, new NullSink(), store, new ManualClock(T0));

            Assert.Equal(1, service.RestoredIncidents);
        }

        /// <summary>A saved incident that differs from its siblings only where this test looks: identity and age.</summary>
        private static PersistedIncident Saved(long id, DateTimeOffset lastSeen)
        {
            return new PersistedIncident(
                id, lastSeen, lastSeen, 2, 0, $"overfit/pod-{id}", [$"overfit/pod-{id}"],
                "overfit", "overfit-server", "rs", $"pod-{id}", "node", "LatencyP95Ms",
                SignalClass.Symptom, 0.75, lastSeen, lastSeen, 1, 1, "scripted");
        }

        private sealed class MemoryStore : IIncidentStore
        {
            private string? _state;

            /// <summary>Memory does not fail in a test; there is nothing to report.</summary>
            public string? LastError => null;

            public string? Load() => _state;

            public void Save(string state) => _state = state;
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
                DateTimeOffset end, TimeSpan window, CancellationToken ct)
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

        /// <summary>
        /// Throws on its first read the way an unreachable Prometheus does, then behaves.
        ///
        /// <para>A separate double rather than a flag on <see cref="ScriptedMetricWindowSource"/>: the script
        /// is a list of windows, and a list that can also hold "throw here" is a small state machine in a
        /// fixture, which is the sort of thing that ends up needing its own test.</para>
        /// </summary>
        private sealed class ThrowingFirstReadSource : IMetricWindowSource
        {
            private static readonly string[] None = [];

            private readonly MetricWindow _afterwards;

            public ThrowingFirstReadSource(MetricWindow afterwards) => _afterwards = afterwards;

            public IReadOnlyList<string> StalePodsExcluded => None;

            /// <summary>Reads attempted, the throwing one included.</summary>
            public int Reads
            {
                get; private set;
            }

            public Task<MetricWindow?> ReadAsync(
                DateTimeOffset end, TimeSpan window, CancellationToken ct)
            {
                Reads++;

                if (Reads == 1)
                {
                    throw new HttpRequestException("connection refused (scripted)");
                }

                return Task.FromResult<MetricWindow?>(_afterwards);
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
