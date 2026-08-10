// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// A pod that answers <b>some</b> scrapes and misses others — the gap between "reporting" and "reporting
    /// completely" that `AN-D9` exists to close.
    ///
    /// <para><b>Total silence was already covered and partial reporting was not, and one line is why.</b>
    /// <c>AnomalyGuard.RunSilentPods</c> removes a pod's counter the moment it appears in the window at all,
    /// so a pod answering two scrapes in three never reaches two consecutive silent cycles, keeps full
    /// membership, and has its incomplete series used as though it were complete. That is what a pod at its
    /// connection limit looks like (<c>RS-6</c>), and <c>GuardCycleResult.PartialMetrics</c> is not it: that
    /// counts CHANNELS reported by fewer pods than the window holds, has no subject and produces no
    /// finding.</para>
    ///
    /// <para><b>The fixture band is the whole point of this file.</b> Every test here that proves the signal
    /// works uses a pod sustaining ~65% coverage, not 0%. A 0% fixture would pass against
    /// <c>RunSilentPods</c> alone and prove nothing about this channel — two designs for it were refuted on
    /// 2026-08-10 and both would have passed a healthy-arm-only suite perfectly.</para>
    /// </summary>
    public sealed class ScrapeCoverageTests
    {
        private const string Coverage = "ScrapeCoverage";

        /// <summary>Pods, sized past the masking bound the peer detector documents for a single outlier.</summary>
        private const int Pods = 8;

        /// <summary>What the degraded replica sustains. Inside the partial band, nowhere near silence.</summary>
        private const double Degraded = 0.65;

        private static readonly DateTimeOffset T0 = new(2026, 8, 10, 12, 0, 0, TimeSpan.Zero);

        /// <summary>
        /// The positive arm, and the capability check that has to come with it: a sustained partial-coverage
        /// pod is named, by name, on the coverage channel.
        ///
        /// <para>Asserted on the signal rather than on "any row", because the window also carries a request
        /// rate and the guard has thirteen other channels — a test satisfied by somebody else's finding is
        /// the failure this suite has shipped before.</para>
        /// </summary>
        [Fact]
        public void APodSustainingPartialCoverageIsNamed()
        {
            var sink = new CapturingSink();
            var guard = Guard(sink, Binding() with { RequirePersistence = false });

            guard.RunCycle(SmoothedWindow(), T0);

            Assert.Contains(sink.Rows, r => string.Equals(r.Signal, Coverage, StringComparison.Ordinal));
            Assert.Contains(
                sink.Rows,
                r => string.Equals(r.Signal, Coverage, StringComparison.Ordinal)
                     && r.Pod.EndsWith("pod07", StringComparison.Ordinal));
        }

        /// <summary>
        /// A fleet at full coverage says nothing. The values are exactly <c>1.0</c> because that is what the
        /// lab measured: <c>avg_over_time(up[15m])</c> read 1 on all twelve pods, and a 20-minute range at the
        /// 15-second grid returned 81 of 81 slots with no distinct value but 1.
        /// </summary>
        [Fact]
        public void AFleetAtFullCoverageIsQuiet()
        {
            var sink = new CapturingSink();

            Guard(sink, Binding()).RunCycle(SmoothedWindow(degraded: 1.0), T0);

            Assert.DoesNotContain(sink.Rows, r => string.Equals(r.Signal, Coverage, StringComparison.Ordinal));
        }

        /// <summary>
        /// The negative arm that can actually fail: every pod loses the odd scrape, which is what an ordinary
        /// cluster does, and none of it is reportable.
        ///
        /// <para>One missed scrape inside a 15-minute range at a 15-second interval moves that pod's value by
        /// <c>1/60</c> — about 1.7%, against a relative gate of 8%. This fixture puts two pods a miss below
        /// the rest and asserts the guard stays quiet, which is the shape of the ordinary-noise question
        /// §5 of the plan asks to measure. It is a fixture, <b>not</b> the measurement: the real floor needs a
        /// faulted arm on the cluster, and this only shows the gate is not trivially loose.</para>
        /// </summary>
        [Fact]
        public void OrdinaryScrapeJitterIsNotReported()
        {
            var sink = new CapturingSink();
            var window = SmoothedWindow(degraded: 1.0);

            // Two pods a single miss below full coverage; one pod two misses below.
            Fill(window, 1, 1.0 - (1.0 / 60.0));
            Fill(window, 4, 1.0 - (1.0 / 60.0));
            Fill(window, 6, 1.0 - (2.0 / 60.0));

            Guard(sink, Binding()).RunCycle(window, T0);

            Assert.DoesNotContain(sink.Rows, r => string.Equals(r.Signal, Coverage, StringComparison.Ordinal));
        }

        /// <summary>
        /// <b>Why the binding carries a verbatim <c>avg_over_time</c> query instead of the raw <c>up</c>
        /// gauge, as an executable fact rather than a paragraph.</b>
        ///
        /// <para>Both windows describe the same pod missing the same 35% of its scrapes. The smoothed one is
        /// reported; the raw <c>0</c>/<c>1</c> one is not, and cannot be — the peer detector's size gate reads
        /// each member's MEDIAN (<c>PeerGroupOutlierDetector.MeasureGaps</c>), and the median of a binary
        /// series is <c>1.0</c> for any pod above 50% coverage, identical to a healthy peer. Raw <c>up</c>
        /// only moves the median once coverage falls below half, which is near-total silence and already
        /// <c>RunSilentPods</c>' case.</para>
        ///
        /// <para>Delete the <c>query</c> key from the binding and this is the test that goes red.</para>
        /// </summary>
        [Fact]
        public void TheRawUpSeriesCannotProduceTheFindingTheSmoothedOneDoes()
        {
            var smoothed = new CapturingSink();
            var raw = new CapturingSink();
            var binding = Binding() with { RequirePersistence = false };

            Guard(smoothed, binding).RunCycle(SmoothedWindow(), T0);
            Guard(raw, binding).RunCycle(RawUpWindow(), T0);

            Assert.Contains(smoothed.Rows, r => string.Equals(r.Signal, Coverage, StringComparison.Ordinal));
            Assert.DoesNotContain(raw.Rows, r => string.Equals(r.Signal, Coverage, StringComparison.Ordinal));
        }

        /// <summary>
        /// Persistence: the first cycle is held, the second reports. A pod being replaced stops answering for
        /// a moment on every rollout and a rollout replaces all of them, so a same-cycle gate would add a
        /// false positive per replica per rollout to a rate that already fails <c>AN-A1</c>.
        /// </summary>
        [Fact]
        public void AFindingIsHeldForOneCycleAndReportedOnTheNext()
        {
            var sink = new CapturingSink();
            var guard = Guard(sink, Binding());

            guard.RunCycle(SmoothedWindow(), T0);

            Assert.DoesNotContain(sink.Rows, r => string.Equals(r.Signal, Coverage, StringComparison.Ordinal));

            guard.RunCycle(SmoothedWindow(), T0.AddMinutes(5));

            Assert.Contains(sink.Rows, r => string.Equals(r.Signal, Coverage, StringComparison.Ordinal));
        }

        /// <summary>
        /// A pod that recovers loses its accumulated evidence, or a one-off dip matures into an incident
        /// hours later on the strength of cycles it has already come back from.
        /// </summary>
        [Fact]
        public void RecoveringResetsTheCount()
        {
            var sink = new CapturingSink();
            var guard = Guard(sink, Binding());

            guard.RunCycle(SmoothedWindow(), T0);
            guard.RunCycle(SmoothedWindow(degraded: 1.0), T0.AddMinutes(5));
            guard.RunCycle(SmoothedWindow(), T0.AddMinutes(10));

            Assert.DoesNotContain(sink.Rows, r => string.Equals(r.Signal, Coverage, StringComparison.Ordinal));
        }

        /// <summary>
        /// A cycle that reached no verdict is evidence in neither direction, so it neither advances the
        /// counter nor clears it.
        ///
        /// <para>The middle cycle here holds the <b>same eight pods</b>, each contributing too few usable
        /// samples to clear <c>PeerOutlierOptions.MinimumSamplesPerPeer</c>, so every member is excluded and
        /// the group returns <c>InsufficientData</c> — with the findings buffer populated and every
        /// <c>Deviation</c> at <c>None</c>. Reading that as "every pod matched its peers" would discard
        /// evidence a fault had already produced, which is the same reasoning that makes
        /// <c>RunSilentPods</c> drop its counters rather than trust a roster nobody could confirm.</para>
        ///
        /// <para><b>The fixture had to be rebuilt to mean this, and the first version did not.</b> It used a
        /// two-pod middle cycle, which is unjudgeable for a different reason — too few PEERS — and which does
        /// not contain the degraded pod at all, so its counter was untouched whether the guard existed or
        /// not. Deleting the <c>InsufficientData</c> check left the test green. Every member of the group has
        /// to be present and starved for the check to be the thing under test.</para>
        /// </summary>
        [Fact]
        public void ACycleThatCouldNotBeJudgedNeitherAdvancesNorClearsTheCount()
        {
            var sink = new CapturingSink();
            var guard = Guard(sink, Binding());

            guard.RunCycle(SmoothedWindow(), T0);
            guard.RunCycle(StarvedWindow(), T0.AddMinutes(5));
            guard.RunCycle(SmoothedWindow(), T0.AddMinutes(10));

            Assert.Contains(sink.Rows, r => string.Equals(r.Signal, Coverage, StringComparison.Ordinal));
        }

        /// <summary>
        /// A scale-down drops counters for pods the window no longer holds, and the pod re-earns its cycles
        /// if it comes back.
        ///
        /// <para><b>The accepted cost is stated rather than hidden</b>: this makes the guard momentarily
        /// quieter, which is the unsafe direction. It is bounded — a pod only leaves the window by reporting
        /// nothing at all, which is <c>RunSilentPods</c>' case and produces its own finding — and the
        /// alternative is a counter map that grows for the life of the process.</para>
        ///
        /// <para>Twelve pods with four degraded, then three: the map has to exceed the window's pod count
        /// before pruning is even attempted, so a fixture with one outlier would exercise nothing.</para>
        /// </summary>
        [Fact]
        public void PodsTheWindowNoLongerHoldsLoseTheirCount()
        {
            var sink = new CapturingSink();
            var guard = Guard(sink, Binding());

            guard.RunCycle(SmoothedWindow(pods: 12, degradedPods: 4), T0);
            guard.RunCycle(SmoothedWindow(pods: 3, degraded: 1.0), T0.AddMinutes(5));
            guard.RunCycle(SmoothedWindow(pods: 12, degradedPods: 4), T0.AddMinutes(10));

            // Without the prune the counters from the first cycle survive, the third cycle reaches two, and
            // this reports.
            Assert.DoesNotContain(sink.Rows, r => string.Equals(r.Signal, Coverage, StringComparison.Ordinal));
        }

        /// <summary>
        /// <b>The non-regression the plan requires</b>: the five channels already on this path do not carry
        /// <c>RequirePersistence</c>, and must report on the first cycle exactly as they always have.
        ///
        /// <para>Asserted through the same guard and the same window as the held case above, with only the
        /// flag differing — so a change that made the gate unconditional fails here rather than passing
        /// because the fixture was gentler.</para>
        /// </summary>
        [Fact]
        public void AChannelThatDidNotAskForPersistenceStillReportsOnTheFirstCycle()
        {
            var sink = new CapturingSink();

            Guard(sink, Binding() with { RequirePersistence = false }).RunCycle(SmoothedWindow(), T0);

            Assert.Contains(sink.Rows, r => string.Equals(r.Signal, Coverage, StringComparison.Ordinal));
        }

        /// <summary>
        /// <b>The persistence gate covers the PEER family only, and this pins the hole rather than hiding
        /// it.</b> A pod whose coverage falls across the window is judged by <c>RunCustomTrend</c> as well,
        /// and nothing there waits for a second cycle — so a coverage ramp is reportable immediately.
        ///
        /// <para>Measured, not assumed: this fixture was run before the assertion was written. It matters
        /// because the case that produces a ramp is a pod being replaced, which is what the persistence gate
        /// exists to absorb — so the gate is narrower than the reason for having it. Closing it means gating
        /// the trend path too, which is a change to five other channels' behaviour and belongs to whoever
        /// owns that decision, not to this task.</para>
        /// </summary>
        [Fact]
        public void ACoverageRampIsReportedByTheTrendFamilyWithoutWaitingForASecondCycle()
        {
            var sink = new CapturingSink();

            Guard(sink, Binding()).RunCycle(RampWindow(), T0);

            // "of typical" is the trend family's own wording, so this names WHICH family reported rather
            // than merely that something did — the peer path on this channel is gated and must not be the
            // one satisfying this assertion.
            Assert.Contains(
                sink.Rows,
                r => string.Equals(r.Signal, Coverage, StringComparison.Ordinal)
                     && r.Message.Contains("of typical", StringComparison.Ordinal));
        }

        /// <summary>
        /// A coverage channel nobody reported is <b>blindness</b>, never health — the rule the whole subsystem
        /// turns on, applied to the one channel whose job is to notice missing data.
        ///
        /// <para>Counted as a difference against a guard with no custom channel, because the thirteen built-in
        /// channels satisfy <c>BlindMetrics >= 1</c> on their own in a window that fills none of them: an
        /// absolute assertion here would pass with the custom accounting deleted. That exact test was measured
        /// green under that mutation on 2026-08-09.</para>
        /// </summary>
        [Fact]
        public void ACoverageChannelNobodyReportedCountsAsBlind()
        {
            var window = new MetricWindow(
                Names(), 80, T0, TimeSpan.FromSeconds(15), [Coverage]);

            var withChannel = Guard(new CapturingSink(), Binding()).RunCycle(window, T0);
            var withoutChannel = GuardWithoutCustomChannels().RunCycle(window, T0);

            Assert.Equal(withoutChannel.BlindMetrics + 1, withChannel.BlindMetrics);
        }

        /// <summary>
        /// One pod missing the channel while the rest report it is <c>PartialMetrics</c>, not silence — the
        /// case where the join that produces `up` loses a target rather than the query failing outright.
        /// </summary>
        [Fact]
        public void OnePodMissingTheChannelIsCountedAsPartial()
        {
            var window = SmoothedWindow(degraded: 1.0);

            // NaN throughout for one pod: the window's own convention for "nothing arrived", never zero.
            Fill(window, 3, double.NaN);

            var result = Guard(new CapturingSink(), Binding()).RunCycle(window, T0);

            Assert.True(
                result.PartialMetrics >= 1,
                $"a pod missing the coverage channel must be counted partial; got {result.PartialMetrics}");
        }

        private static CustomMetricBinding Binding()
        {
            return new CustomMetricBinding(
                Name: Coverage,
                Source: "up",
                Kind: MetricSourceKind.Ratio,
                SignalKind: PeerSignalKind.LoadIndependent,
                Class: SignalClass.Symptom,
                Query: "avg_over_time(up{%selector%}[15m])",
                RequirePersistence: true,
                Calibrated: false);
        }

        private static AnomalyGuard Guard(IIncidentSink sink, CustomMetricBinding binding)
        {
            return new AnomalyGuard(Options() with { CustomMetrics = [binding] },
                sink, IncidentTrackingOptions.Balanced);
        }

        private static AnomalyGuard GuardWithoutCustomChannels()
        {
            return new AnomalyGuard(Options(), new CapturingSink(), IncidentTrackingOptions.Balanced);
        }

        private static AnomalyGuardOptions Options()
        {
            return new AnomalyGuardOptions
            {
                Namespace = "lab",
                Workload = "lab-workload",
                NonCalibratedCustomChannels = [Coverage],
                Grouping = IncidentGroupingOptions.Balanced with
                {
                    Topology = TopologyWeights.SingleNode
                },
            };
        }

        private static List<string> Names(int pods = Pods)
        {
            var names = new List<string>(pods);

            for (var p = 0; p < pods; p++)
            {
                names.Add($"lab-workload-8697c6f4c8-pod{p:d2}");
            }

            return names;
        }

        /// <summary>
        /// What the deployed binding actually delivers: one already-averaged fraction per grid slot, the last
        /// <paramref name="degradedPods"/> replicas sustaining <paramref name="degraded"/> and the rest at
        /// full coverage.
        /// </summary>
        private static MetricWindow SmoothedWindow(
            double degraded = Degraded, int pods = Pods, int degradedPods = 1)
        {
            var window = new MetricWindow(Names(pods), 80, T0, TimeSpan.FromSeconds(15), [Coverage]);

            for (var pod = 0; pod < pods; pod++)
            {
                Fill(window, pod, pod >= pods - degradedPods ? degraded : 1.0);

                var rps = window.Series(pod, MetricIndex.RequestsPerSecond);

                for (var i = 0; i < window.Length; i++)
                {
                    rps[i] = 5.0;
                }
            }

            return window;
        }

        /// <summary>
        /// The same fleet and the same 35% of scrapes lost, expressed as the raw <c>up</c> gauge. The pattern
        /// is deterministic and evenly spread so the series carries no slope — this fixture must fail to
        /// produce a PEER finding, and it would be a different test if the trend family caught it instead.
        /// </summary>
        private static MetricWindow RawUpWindow()
        {
            var window = new MetricWindow(Names(), 80, T0, TimeSpan.FromSeconds(15), [Coverage]);

            for (var pod = 0; pod < Pods; pod++)
            {
                var series = window.Series(pod, Coverage);
                var rps = window.Series(pod, MetricIndex.RequestsPerSecond);

                for (var i = 0; i < window.Length; i++)
                {
                    // 7 of every 20 attempts fail on the last pod: 65% coverage, the same as the smoothed
                    // fixture, and above the 50% line where a binary median still reads 1.0.
                    series[i] = pod == Pods - 1 && i % 20 < 7 ? 0.0 : 1.0;
                    rps[i] = 5.0;
                }
            }

            return window;
        }

        /// <summary>
        /// The same fleet, with every pod contributing fewer usable samples than
        /// <c>PeerOutlierOptions.MinimumSamplesPerPeer</c> — the shape a cycle takes when a scrape gap swallows
        /// the recent tail. Every member is excluded, so the verdict is <c>InsufficientData</c> and no
        /// deviation is recorded for anybody.
        ///
        /// <para>Only the first ten slots carry data: the peer families read the trailing
        /// <c>RecentWindow</c>, which at fifteen minutes over a fifteen-second grid is the last sixty of the
        /// eighty here, and that tail is entirely <c>NaN</c>.</para>
        /// </summary>
        private static MetricWindow StarvedWindow()
        {
            var window = new MetricWindow(Names(), 80, T0, TimeSpan.FromSeconds(15), [Coverage]);

            for (var pod = 0; pod < Pods; pod++)
            {
                var series = window.Series(pod, Coverage);
                var rps = window.Series(pod, MetricIndex.RequestsPerSecond);

                for (var i = 0; i < 10; i++)
                {
                    series[i] = 1.0;
                    rps[i] = 5.0;
                }
            }

            return window;
        }

        /// <summary>
        /// One pod's coverage sliding from full to <see cref="Degraded"/> across the window while its peers
        /// hold at 1.0 — the shape a pod takes as it starts failing scrapes, and the shape a pod being
        /// replaced takes as it stops.
        /// </summary>
        private static MetricWindow RampWindow()
        {
            var window = new MetricWindow(Names(), 80, T0, TimeSpan.FromSeconds(15), [Coverage]);

            for (var pod = 0; pod < Pods; pod++)
            {
                var series = window.Series(pod, Coverage);
                var rps = window.Series(pod, MetricIndex.RequestsPerSecond);

                for (var i = 0; i < window.Length; i++)
                {
                    var fraction = (double)i / (window.Length - 1);

                    series[i] = pod == Pods - 1 ? 1.0 - ((1.0 - Degraded) * fraction) : 1.0;
                    rps[i] = 5.0;
                }
            }

            return window;
        }

        private static void Fill(MetricWindow window, int pod, double value)
        {
            var series = window.Series(pod, Coverage);

            for (var i = 0; i < window.Length; i++)
            {
                series[i] = value;
            }
        }

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
