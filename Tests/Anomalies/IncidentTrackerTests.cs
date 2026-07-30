// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System;
using System.Collections.Generic;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Contracts;
using DevOnBike.Overfit.Anomalies.Rules.Contracts;
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
