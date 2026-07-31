// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// The lifecycle, which is what makes the guard deployable at all.
    ///
    /// <para>The whole point is the first test: a problem that persists across cycles must be reported once
    /// and then updated. Everything else here guards a way of getting that wrong — closing on a flicker,
    /// merging unrelated problems into one immortal incident, or letting the state grow without bound.</para>
    /// </summary>
    public sealed class IncidentTrackerTests
    {
        private static readonly DateTimeOffset T0 = new(2026, 7, 30, 12, 0, 0, TimeSpan.Zero);

        /// <summary>The reason this class exists.</summary>
        [Fact]
        public void APersistingProblem_OpensOnce_AndIsOngoingAfterwards()
        {
            var tracker = new IncidentTracker(IncidentTrackingOptions.Balanced);

            var first = tracker.Observe(Cycle(("pod-a", "cpu"), ("pod-a", "latency")), T0);

            Assert.Single(first);
            Assert.Equal(IncidentState.Opened, first[0].State);

            var id = first[0].Id;
            var opened = 0;

            for (var cycle = 1; cycle <= 11; cycle++)
            {
                var next = tracker.Observe(
                    Cycle(("pod-a", "cpu"), ("pod-a", "latency")), T0.AddMinutes(5 * cycle));

                Assert.Single(next);
                Assert.Equal(id, next[0].Id);
                opened += next[0].State == IncidentState.Opened ? 1 : 0;
            }

            // An hour at a five-minute cadence: one notification, not twelve.
            Assert.Equal(0, opened);
            Assert.Equal(1, tracker.OpenCount);
        }

        [Fact]
        public void AgeAndCycleCountTrackTheRealDuration_NotTheWindow()
        {
            var tracker = new IncidentTracker(IncidentTrackingOptions.Balanced);

            tracker.Observe(Cycle(("pod-a", "cpu")), T0);
            tracker.Observe(Cycle(("pod-a", "cpu")), T0.AddMinutes(5));
            var third = tracker.Observe(Cycle(("pod-a", "cpu")), T0.AddMinutes(10));

            Assert.Equal(TimeSpan.FromMinutes(10), third[0].Age);
            Assert.Equal(3, third[0].CyclesSeen);
        }

        /// <summary>
        /// A group gains and loses findings every cycle. Overlap, not equality — otherwise every one of those
        /// changes opens a fresh incident, which is the behaviour being removed.
        /// </summary>
        [Fact]
        public void AGroupThatGainsAndLosesFindings_IsStillTheSameIncident()
        {
            var tracker = new IncidentTracker(IncidentTrackingOptions.Balanced);

            var first = tracker.Observe(Cycle(("pod-a", "cpu"), ("pod-a", "latency")), T0);
            var second = tracker.Observe(
                Cycle(("pod-a", "cpu"), ("pod-a", "latency"), ("pod-b", "memory")), T0.AddMinutes(5));
            var third = tracker.Observe(Cycle(("pod-a", "cpu"), ("pod-b", "memory")), T0.AddMinutes(10));

            Assert.Equal(first[0].Id, second[0].Id);
            Assert.Equal(first[0].Id, third[0].Id);
            Assert.All(new[] { second[0], third[0] }, r => Assert.Equal(IncidentState.Ongoing, r.State));
        }

        /// <summary>
        /// <b>The shadow run this was written for.</b> A fault on one pod produced a group of three subjects
        /// in one cycle and one subject in the next, as the collateral findings cleared. Jaccard is then
        /// exactly 1/3 — and the matcher required 0.34, so the same fault on the same pod opened a second
        /// incident, one hundredth short.
        ///
        /// <para>The bar was measuring the wrong thing rather than being set too high, which is why this test
        /// pins the overlap at 0.33 instead of asserting some new number: a group shrinking as it recovers is
        /// one incident getting better, and no threshold on periphery can express that.</para>
        /// </summary>
        [Fact]
        public void AShrinkingGroup_WithTheSameCentre_IsStillTheSameIncident()
        {
            var tracker = new IncidentTracker(IncidentTrackingOptions.Balanced);

            var first = tracker.Observe(
                [Severity(("pod-a", "cpu", 0.95), ("pod-b", "cpu", 0.40), ("pod-c", "cpu", 0.40))], T0);

            var traces = new List<IncidentMatchTrace>();
            var second = tracker.Observe(
                [Severity(("pod-a", "cpu", 0.95))], T0.AddMinutes(5), traces.Add);

            // The scenario really is the borderline one, not merely a case that happens to pass.
            Assert.Equal(1.0 / 3.0, traces.Single().BestOverlap, 6);
            Assert.Contains("pod-a", traces.Single().PrimaryKey, StringComparison.Ordinal);

            Assert.Equal(first[0].Id, second[0].Id);
            Assert.Equal(IncidentState.Ongoing, second[0].State);
            Assert.Equal(IncidentMatchOutcome.Continued, traces.Single().Outcome);
        }

        /// <summary>
        /// The churn the veto caused, at the scale an operator would feel it. The periphery alternates every
        /// cycle — which is what a fault near a threshold actually does — and that used to cross the bar in
        /// both directions, opening an incident every other cycle for one unchanging problem.
        /// </summary>
        [Fact]
        public void AGroupWhosePeripheryAlternates_OpensOnce_NotEveryOtherCycle()
        {
            var tracker = new IncidentTracker(IncidentTrackingOptions.Balanced);
            var opened = 0;

            for (var cycle = 0; cycle < 8; cycle++)
            {
                var group = cycle % 2 == 0
                    ? Severity(("pod-a", "cpu", 0.95), ("pod-b", "cpu", 0.40), ("pod-c", "cpu", 0.40))
                    : Severity(("pod-a", "cpu", 0.95));

                var rows = tracker.Observe([group], T0.AddMinutes(5 * cycle));
                opened += rows.Count(r => r.State == IncidentState.Opened);
            }

            Assert.Equal(1, opened);
            Assert.Equal(1, tracker.OpenCount);
        }

        /// <summary>
        /// <b>The bug the removed veto was introduced alongside, which must stay fixed.</b> When the grouper
        /// merges a whole deployment, the group about the surviving healthy pods shares three subjects out of
        /// four with the group about the degraded one — 0.75, comfortably over any overlap bar. Identity has
        /// to be refused on the centre, not on the periphery, or an incident silently changes what it is
        /// about while keeping its number.
        /// </summary>
        [Fact]
        public void AGroupWithADifferentCentre_DoesNotInheritTheIdentity_EvenAtHighOverlap()
        {
            var tracker = new IncidentTracker(IncidentTrackingOptions.Balanced);

            var first = tracker.Observe(
                [Severity(
                    ("pod-a", "cpu", 0.95), ("pod-b", "cpu", 0.40),
                    ("pod-c", "cpu", 0.40), ("pod-d", "cpu", 0.40))],
                T0);

            var traces = new List<IncidentMatchTrace>();
            var second = tracker.Observe(
                [Severity(("pod-b", "cpu", 0.95), ("pod-c", "cpu", 0.40), ("pod-d", "cpu", 0.40))],
                T0.AddMinutes(5),
                traces.Add);

            Assert.Equal(0.75, traces.Single().BestOverlap, 6);
            Assert.NotEqual(first[0].Id, second[0].Id);
            Assert.Equal(IncidentState.Opened, second[0].State);
            Assert.Equal(IncidentMatchOutcome.PrimaryChanged, traces.Single().Outcome);
        }

        /// <summary>Sharing nothing is a different problem, however similar it looks.</summary>
        [Fact]
        public void AnUnrelatedGroup_OpensItsOwnIncident()
        {
            var tracker = new IncidentTracker(IncidentTrackingOptions.Balanced);

            var first = tracker.Observe(Cycle(("pod-a", "cpu")), T0);
            var second = tracker.Observe(Cycle(("pod-z", "errors")), T0.AddMinutes(5));

            Assert.Equal(IncidentState.Opened, second[0].State);
            Assert.NotEqual(first[0].Id, second[0].Id);

            // The first is now inside its grace period: aged, not yet closed, and not reported.
            Assert.Equal(2, tracker.OpenCount);
        }

        /// <summary>
        /// The grace period, and the reason it is not one cycle. A borderline finding drops out and returns;
        /// closing immediately turns that into resolve/open/resolve/open.
        /// </summary>
        [Fact]
        public void AFlickeringIncident_DoesNotResolveAndReopen()
        {
            var tracker = new IncidentTracker(IncidentTrackingOptions.Balanced);

            var first = tracker.Observe(Cycle(("pod-a", "cpu")), T0);
            var gap = tracker.Observe([], T0.AddMinutes(5));
            var back = tracker.Observe(Cycle(("pod-a", "cpu")), T0.AddMinutes(10));

            Assert.Empty(gap);
            Assert.Single(back);
            Assert.Equal(IncidentState.Ongoing, back[0].State);
            Assert.Equal(first[0].Id, back[0].Id);
            Assert.Equal(0, back[0].CyclesMissing);
        }

        [Fact]
        public void AnIncidentThatStaysAway_ResolvesExactlyOnce()
        {
            var tracker = new IncidentTracker(IncidentTrackingOptions.Balanced);

            var first = tracker.Observe(Cycle(("pod-a", "cpu")), T0);

            Assert.Empty(tracker.Observe([], T0.AddMinutes(5)));

            var closing = tracker.Observe([], T0.AddMinutes(10));

            Assert.Single(closing);
            Assert.Equal(IncidentState.Resolved, closing[0].State);
            Assert.Equal(first[0].Id, closing[0].Id);

            // Closed means gone: no second resolve, and nothing left tracked.
            Assert.Empty(tracker.Observe([], T0.AddMinutes(15)));
            Assert.Equal(0, tracker.OpenCount);
        }

        /// <summary>A resolved row still carries what the incident was — a ticket being closed needs it.</summary>
        [Fact]
        public void AResolvedRowCarriesTheLastKnownIncident()
        {
            var tracker = new IncidentTracker(IncidentTrackingOptions.Strict);

            tracker.Observe(Cycle(("pod-a", "cpu"), ("pod-a", "latency")), T0);
            var closing = tracker.Observe([], T0.AddMinutes(5));

            Assert.Equal(IncidentState.Resolved, closing[0].State);
            Assert.NotNull(closing[0].Incident);
            Assert.Equal(2, closing[0].Incident.Findings.Count);
        }

        /// <summary>Two concurrent problems must not be matched onto one another.</summary>
        [Fact]
        public void TwoConcurrentIncidents_KeepSeparateIdentities()
        {
            var tracker = new IncidentTracker(IncidentTrackingOptions.Balanced);

            var first = tracker.Observe(
                [Group(("pod-a", "cpu")), Group(("pod-z", "errors"))], T0);

            Assert.Equal(2, first.Count);
            Assert.NotEqual(first[0].Id, first[1].Id);

            var second = tracker.Observe(
                [Group(("pod-z", "errors")), Group(("pod-a", "cpu"))], T0.AddMinutes(5));

            // Order changed; identity must follow the content, not the position.
            Assert.All(second, r => Assert.Equal(IncidentState.Ongoing, r.State));
            Assert.Equal(
                new SortedSet<long> { first[0].Id, first[1].Id },
                new SortedSet<long> { second[0].Id, second[1].Id });
        }

        /// <summary>State that only grows is a leak in a process meant to run for months.</summary>
        [Fact]
        public void TrackedStateIsBounded()
        {
            var tracker = new IncidentTracker(
                IncidentTrackingOptions.Balanced with { MaxOpenIncidents = 4, ResolveAfterMissingCycles = 99 });

            for (var i = 0; i < 40; i++)
            {
                tracker.Observe(Cycle(($"pod-{i}", "cpu")), T0.AddMinutes(5 * i));
            }

            Assert.Equal(4, tracker.OpenCount);
        }

        [Fact]
        public void RejectsUnusableOptions()
        {
            Assert.Throws<ArgumentException>(() => new IncidentTracker(default));
            Assert.Throws<ArgumentNullException>(
                () => new IncidentTracker(IncidentTrackingOptions.Balanced).Observe(null!, T0));
        }

        /// <summary>
        /// One group whose members carry <b>different</b> severities, so which pod is the incident's centre is
        /// decided rather than incidental. Findings otherwise share a class and a start time, which is what
        /// leaves severity as the tie-break — see <c>Incident.Findings</c> for the ordering.
        /// </summary>
        private static Incident Severity(params (string Pod, string Signal, double Breach)[] findings)
        {
            var pipeline = new IncidentPipeline();

            foreach (var (pod, signal, breach) in findings)
            {
                pipeline.ObserveRule(
                    new IncidentSubject("overfit", "overfit-server", string.Empty, pod, string.Empty),
                    signal,
                    new SustainedThresholdResult(
                        Status: DetectionStatus.Anomalous,
                        Reason: $"{signal} on {pod}",
                        BreachFraction: breach,
                        BreachedSamples: (int)(breach * 40),
                        UsableSamples: 40,
                        PeakValue: breach * 10.0,
                        MedianValue: breach * 5.0),
                    T0, T0.AddMinutes(12), default, SignalClass.Resource);
            }

            var grouped = pipeline.Group(IncidentGroupingOptions.Balanced with
            {
                Topology = TopologyWeights.SingleNode
            });

            Assert.Single(grouped);

            return grouped[0];
        }

        private static IReadOnlyList<Incident> Cycle(params (string Pod, string Signal)[] findings)
        {
            return [Group(findings)];
        }

        private static Incident Group(params (string Pod, string Signal)[] findings)
        {
            var pipeline = new IncidentPipeline();

            foreach (var (pod, signal) in findings)
            {
                pipeline.ObserveRule(
                    new IncidentSubject("overfit", "overfit-server", string.Empty, pod, string.Empty),
                    signal,
                    new SustainedThresholdResult(
                        Status: DetectionStatus.Anomalous,
                        Reason: $"{signal} on {pod}",
                        BreachFraction: 0.8,
                        BreachedSamples: 30,
                        UsableSamples: 40,
                        PeakValue: 1.0,
                        MedianValue: 0.5),
                    T0, T0.AddMinutes(12), default, SignalClass.Resource);
            }

            var grouped = pipeline.Group(IncidentGroupingOptions.Balanced with
            {
                Topology = TopologyWeights.SingleNode
            });

            Assert.Single(grouped);

            return grouped[0];
        }
    }
}
