// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// The change-in-gap mechanism behind AN-D1, on the five shapes the plan names.
    ///
    /// <para><b>The oracle is the recorded lab sequence, not a shape invented here.</b> The flat case replays
    /// the fifteen <c>MemoryWorkingSetBytes</c> gaps logged for <c>lab-workload-7765564ff6-pj7r8</c> between
    /// 2026-08-06T08:29Z and 2026-08-07T08:54Z — 9.62 to 10.30 MB, 6.8% of the median end to end, no drift.
    /// That is the pod whose incident stayed open ~97% of a day, so if the mechanism cannot call <i>that</i>
    /// standing it is worth nothing.</para>
    ///
    /// <para><b>Both arms, always.</b> Every "goes quiet" assertion here has a matching "still fires" one, and
    /// the fail-open cases are asserted separately from the measured-stable one — a gate that suppressed on
    /// missing history would pass a test written only around the quiet arm.</para>
    /// </summary>
    public sealed class PeerNoveltyTrackerTests
    {
        private const string Pod = "lab-workload-7765564ff6-pj7r8";

        private static readonly DateTimeOffset T0 = new(2026, 8, 6, 8, 0, 0, TimeSpan.Zero);

        private static readonly DateTimeOffset Created = new(2026, 8, 5, 6, 0, 0, TimeSpan.Zero);

        /// <summary>One megabyte of movement in the gap; a test-local choice, not a shipped default.</summary>
        private const double Floor = 1_000_000.0;

        /// <summary>Recorded gaps, in bytes, from <c>Tests/bin/fp-run-guard-log-FINAL.txt</c>.</summary>
        private static readonly double[] RecordedGaps =
        [
            10200000, 9630000, 9990000, 9820000, 9820000, 9620000, 9720000, 10100000,
            10100000, 9630000, 9670000, 10300000, 10300000, 9990000, 9990000
        ];

        /// <summary>Their offsets in seconds from the first, same source.</summary>
        private static readonly double[] RecordedSeconds =
        [
            0, 1800, 3600, 20999, 21599, 33899, 34799, 54299,
            54899, 74098, 74698, 86723, 86996, 87295, 87895
        ];

        private static PeerNoveltyOptions Options => PeerNoveltyOptions.Daily with
        {
            MinimumCycles = 12,
            RetainedCyclesPerSeries = 24,
        };

        /// <summary>The recorded flat sequence must end up Standing, and stop being forwarded.</summary>
        [Fact]
        public void TheRecordedFlatGapBecomesStanding()
        {
            var tracker = new PeerNoveltyTracker(Options);
            var last = NoveltyDecision.Unknown;

            for (var i = 0; i < RecordedGaps.Length; i++)
            {
                last = tracker.Observe(
                    Pod,
                    Created,
                    MetricIndex.MemoryWorkingSetBytes,
                    RecordedGaps[i],
                    T0.AddSeconds(RecordedSeconds[i]),
                    Floor);
            }

            Assert.Equal(NoveltyKind.Standing, last.Kind);
            Assert.Equal(DetectionStatus.Healthy, last.Status);
            Assert.False(last.Forward);
        }

        /// <summary>
        /// Before the cycle floor is reached the answer is <c>WarmingUp</c> and the finding is forwarded.
        ///
        /// <para>Asserted on the status as well as the classification, because both fail-open states produce
        /// <see cref="NoveltyKind.New"/> and only the status says whether anything was measured. A gate that
        /// could not tell them apart would be indistinguishable from one that had stopped working.</para>
        /// </summary>
        [Fact]
        public void TooLittleHistoryIsWarmingUpAndForwards()
        {
            var tracker = new PeerNoveltyTracker(Options);

            for (var i = 0; i < Options.MinimumCycles - 1; i++)
            {
                var decision = tracker.Observe(
                    Pod, Created, MetricIndex.MemoryWorkingSetBytes, 9_900_000.0,
                    T0.AddMinutes(5 * i), Floor);

                Assert.Equal(NoveltyKind.New, decision.Kind);
                Assert.True(decision.Forward);
                Assert.Equal(DetectionStatus.WarmingUp, decision.Status);
            }
        }

        /// <summary>A gap that is growing stays New and keeps forwarding, at every cycle.</summary>
        [Fact]
        public void AGrowingGapStaysNew()
        {
            var tracker = new PeerNoveltyTracker(Options);
            var decisions = Feed(tracker, 24, cycle => 9_900_000.0 + (400_000.0 * cycle));

            var last = decisions[^1];

            Assert.Equal(NoveltyKind.New, last.Kind);
            Assert.Equal(DetectionStatus.Anomalous, last.Status);
            Assert.Equal(TrendDirection.Rising, last.Direction);

            for (var i = 0; i < decisions.Count; i++)
            {
                Assert.True(decisions[i].Forward);
            }
        }

        /// <summary>
        /// A shrinking gap is stability, not news — the lab's own −2.65 MB over three hours.
        ///
        /// <para>Without the direction test this reads as "something changed" and keeps paging for a replica
        /// that is becoming more like its peers, which is backwards.</para>
        /// </summary>
        [Fact]
        public void AShrinkingGapIsStanding()
        {
            var tracker = new PeerNoveltyTracker(Options);
            var decisions = Feed(tracker, 24, cycle => 19_900_000.0 - (400_000.0 * cycle));
            var last = decisions[^1];

            Assert.Equal(NoveltyKind.Standing, last.Kind);
            Assert.Equal(DetectionStatus.Anomalous, last.Status);
            Assert.Equal(TrendDirection.Falling, last.Direction);
            Assert.False(last.Forward);
        }

        /// <summary>
        /// A pod that restarts keeps its name and gets a new creation time; its history must not survive.
        ///
        /// <para>This is the case roster-based pruning cannot see, and the reason the design carries a
        /// creation-time echo at all: a StatefulSet pod restarting is still on the roster throughout.</para>
        /// </summary>
        [Fact]
        public void ARestartUnderTheSameNameDiscardsTheHistory()
        {
            var tracker = new PeerNoveltyTracker(Options);
            var before = Feed(tracker, 24, _ => 9_900_000.0)[^1];

            Assert.Equal(NoveltyKind.Standing, before.Kind);

            var after = tracker.Observe(
                Pod,
                Created.AddHours(9),
                MetricIndex.MemoryWorkingSetBytes,
                9_900_000.0,
                T0.AddMinutes(5 * 24),
                Floor);

            Assert.Equal(NoveltyKind.New, after.Kind);
            Assert.Equal(DetectionStatus.WarmingUp, after.Status);
            Assert.True(after.Forward);
            Assert.Equal(1, after.Samples);
        }

        /// <summary>
        /// Growth resuming after a settled period puts the pod back on New, with no separate un-suppress step.
        ///
        /// <para><b>It is not the very next cycle, and the exact count is pinned rather than bounded.</b> The
        /// plan says "reverts on the next cycle the rise is observed"; what the mechanism actually does is
        /// re-fit the whole retained window each cycle, so the rise has to outweigh the flat part of it
        /// first. Measured here at <b>13 cycles</b> — 65 minutes at a five-minute cadence, against a
        /// twenty-four-cycle retained window. The number is asserted exactly so that a change in either
        /// direction, faster or slower, fails rather than passes quietly.</para>
        /// </summary>
        [Fact]
        public void ResumedGrowthRevertsToNew()
        {
            var tracker = new PeerNoveltyTracker(Options);

            Assert.Equal(NoveltyKind.Standing, Feed(tracker, 24, _ => 9_900_000.0)[^1].Kind);

            var reverted = -1;

            for (var i = 0; i < Options.RetainedCyclesPerSeries; i++)
            {
                var decision = tracker.Observe(
                    Pod,
                    Created,
                    MetricIndex.MemoryWorkingSetBytes,
                    9_900_000.0 + (400_000.0 * (i + 1)),
                    T0.AddMinutes(5 * (24 + i)),
                    Floor);

                if (decision.Kind == NoveltyKind.New)
                {
                    reverted = i + 1;

                    Assert.True(decision.Forward);

                    break;
                }
            }

            Assert.Equal(13, reverted);
        }

        /// <summary>
        /// A standing deviation is quiet between reassertions and loud on the cadence — never permanently
        /// silent, which is the client's own condition on accepting the trade.
        /// </summary>
        [Fact]
        public void AStandingDeviationReassertsOnTheInterval()
        {
            var options = Options with
            {
                StandingReassertionInterval = TimeSpan.FromMinutes(30),
            };

            var tracker = new PeerNoveltyTracker(options);
            var forwardedWhileStanding = 0;
            var standing = 0;

            for (var i = 0; i < 60; i++)
            {
                var decision = tracker.Observe(
                    Pod, Created, MetricIndex.MemoryWorkingSetBytes, 9_900_000.0,
                    T0.AddMinutes(5 * i), Floor);

                if (decision.Kind != NoveltyKind.Standing)
                {
                    continue;
                }

                standing++;
                forwardedWhileStanding += decision.Forward ? 1 : 0;
            }

            // Cycles 0..10 are WarmingUp (the twelfth observation is the first that can be judged), leaving
            // 49 standing cycles across four hours; a thirty-minute cadence is eight reminders inside them.
            Assert.Equal(49, standing);
            Assert.Equal(8, forwardedWhileStanding);
        }

        /// <summary>
        /// A customer channel named exactly like an enum member keeps its own history — the collision the
        /// <c>~</c> marker exists for, asserted across a serialisation round trip where it would actually bite.
        /// </summary>
        [Fact]
        public void ACustomChannelNamedLikeAMetricDoesNotShareState()
        {
            var tracker = new PeerNoveltyTracker(Options);

            Feed(tracker, 24, _ => 9_900_000.0);

            var custom = tracker.Observe(
                Pod, Created, nameof(MetricIndex.MemoryWorkingSetBytes), 9_900_000.0,
                T0.AddMinutes(5 * 24), Floor);

            Assert.Equal(NoveltyKind.New, custom.Kind);
            Assert.Equal(1, custom.Samples);

            var restored = PeerNoveltyTracker.Read(tracker.Write(), Options);

            var builtIn = restored.Observe(
                Pod, Created, MetricIndex.MemoryWorkingSetBytes, 9_900_000.0,
                T0.AddMinutes(5 * 25), Floor);

            Assert.Equal(NoveltyKind.Standing, builtIn.Kind);
        }

        /// <summary>
        /// State survives a restart, which is the client's decision recorded in ADR 0001 — otherwise every
        /// rollout of the guard reintroduces a bounded copy of the noise this removes.
        /// </summary>
        [Fact]
        public void StateSurvivesARoundTrip()
        {
            var tracker = new PeerNoveltyTracker(Options);

            Feed(tracker, 24, _ => 9_900_000.0);

            var restored = PeerNoveltyTracker.Read(tracker.Write(), Options);

            var decision = restored.Observe(
                Pod, Created, MetricIndex.MemoryWorkingSetBytes, 9_900_000.0,
                T0.AddMinutes(5 * 24), Floor);

            Assert.Equal(NoveltyKind.Standing, decision.Kind);
            Assert.Equal(1, restored.TrackedPods);
        }

        /// <summary>
        /// A row whose pod name the roster still knows, carrying a different creation time, is refused at
        /// load — the ADR's reuse check, on the restore path rather than the observe path.
        /// </summary>
        [Fact]
        public void ARowForAReusedPodNameIsNotAdopted()
        {
            var tracker = new PeerNoveltyTracker(Options);

            Feed(tracker, 24, _ => 9_900_000.0);

            var roster = new Dictionary<string, DateTimeOffset>(StringComparer.Ordinal)
            {
                [Pod] = Created.AddHours(9),
            };

            var restored = PeerNoveltyTracker.Read(tracker.Write(), Options, roster);

            Assert.Equal(0, restored.TrackedPods);
        }

        /// <summary>Pods the cluster no longer lists stop costing memory.</summary>
        [Fact]
        public void PruningDropsPodsTheRosterNoLongerLists()
        {
            var tracker = new PeerNoveltyTracker(Options);

            for (var p = 0; p < 4; p++)
            {
                tracker.Observe(
                    $"pod-{p}", Created, MetricIndex.MemoryWorkingSetBytes, 9_900_000.0, T0, Floor);
            }

            Assert.Equal(4, tracker.TrackedPods);

            tracker.Prune(["pod-0", "pod-1"]);

            Assert.Equal(2, tracker.TrackedPods);
        }

        /// <summary><c>default</c> is refused, the same way every other options type here refuses it.</summary>
        [Fact]
        public void DefaultOptionsAreRefused()
        {
            Assert.Throws<ArgumentException>(() => new PeerNoveltyTracker(default));
        }

        private static List<NoveltyDecision> Feed(
            PeerNoveltyTracker tracker, int cycles, Func<int, double> gap)
        {
            var decisions = new List<NoveltyDecision>(cycles);

            for (var i = 0; i < cycles; i++)
            {
                decisions.Add(tracker.Observe(
                    Pod,
                    Created,
                    MetricIndex.MemoryWorkingSetBytes,
                    gap(i),
                    T0.AddMinutes(5 * i),
                    Floor));
            }

            return decisions;
        }
    }
}
