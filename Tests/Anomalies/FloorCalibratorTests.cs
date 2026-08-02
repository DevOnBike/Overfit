// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// Turning a shadow week into configuration.
    ///
    /// <para>The last test is the one that matters: a floor proposed from a healthy period must actually
    /// silence that period. Everything else here is arithmetic; that one is the claim.</para>
    /// </summary>
    public sealed class FloorCalibratorTests
    {
        private static readonly DateTimeOffset T0 = new(2026, 8, 1, 12, 0, 0, TimeSpan.Zero);

        private readonly ITestOutputHelper _output;

        public FloorCalibratorTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void ProposesFromWhatAHealthyDeploymentDid()
        {
            var calibrator = new FloorCalibrator();

            for (var cycle = 0; cycle < 20; cycle++)
            {
                calibrator.Observe(Healthy(seed: cycle));
            }

            var proposal = calibrator.Propose()[(int)MetricIndex.GcGen2HeapBytes];

            Assert.True(proposal.IsUsable, $"only {proposal.Samples} samples");

            // The generated heap sits around 4 MB, which is the scale the lab actually showed and the reason
            // a percentage gate is defenceless on it.
            Assert.InRange(proposal.TypicalMagnitude, 3.0e6, 5.0e6);

            // A floor has to sit above everything healthy, so above the largest gap seen, not at it.
            Assert.True(proposal.ProposedMinAbsoluteGap > proposal.PeerGapMax,
                $"proposed {proposal.ProposedMinAbsoluteGap} is not above the observed max {proposal.PeerGapMax}");

            Assert.True(proposal.PeerGapMax >= proposal.PeerGapP99);
        }

        [Fact]
        public void WithoutEnoughObservations_TheProposalSaysSo()
        {
            var calibrator = new FloorCalibrator();
            calibrator.Observe(Healthy(seed: 1));

            Assert.False(calibrator.Propose()[(int)MetricIndex.GcGen2HeapBytes].IsUsable);
        }

        [Fact]
        public void AMetricNobodyReports_ProposesNothing()
        {
            var calibrator = new FloorCalibrator();

            for (var cycle = 0; cycle < 10; cycle++)
            {
                calibrator.Observe(Healthy(seed: cycle));
            }

            // ErrorRate is never filled by Healthy(), so there is nothing to propose from and nothing is
            // invented — a floor conjured out of no data would gate a signal nobody has measured.
            var proposal = calibrator.Propose()[(int)MetricIndex.ErrorRate];

            Assert.Equal(0, proposal.Samples);
            Assert.Equal(0.0, proposal.ProposedMinAbsoluteGap);
        }

        /// <summary>
        /// A restart is the finding, so its threshold is not the calibrator's to move.
        ///
        /// <para>This is not a hypothetical guard. Fitted over a healthy synthetic day, the calibrator
        /// proposed a <c>ContainerRestarts</c> floor of <b>1.25</b> — arithmetically correct, since pods there
        /// restart about once a day, and it would have made a single restart permanently unreportable.</para>
        /// </summary>
        [Fact]
        public void ACountedEventIsNeverFittedFromData()
        {
            var calibrator = new FloorCalibrator();

            for (var cycle = 0; cycle < 20; cycle++)
            {
                calibrator.Observe(Restarting(seed: cycle));
            }

            var proposal = calibrator.Propose()[(int)MetricIndex.ContainerRestarts];

            // Observed, so the report can still say what the cluster did...
            Assert.True(proposal.PeerGapMax >= 1.0,
                "the window was built so peers differ by at least one restart; nothing was observed");

            // ...and not turned into a threshold.
            Assert.Equal(0.0, proposal.ProposedMinAbsoluteGap);
            Assert.Equal(0.0, proposal.ProposedMinAbsoluteTrendChange);
        }

        /// <summary>
        /// A channel the enum does not have is calibrated like any other.
        ///
        /// <para>It was not, and the consequence was silent: a custom binding's floor defaults to zero, zero
        /// means the gate is off, and nothing ever proposed a value to replace it. The customer's own metrics
        /// were the one part of the configuration with no fallback at all.</para>
        /// </summary>
        [Fact]
        public void ACustomChannelIsCalibratedLikeAnyOther()
        {
            var calibrator = new FloorCalibrator();

            for (var cycle = 0; cycle < 20; cycle++)
            {
                calibrator.Observe(WithQueueDepth(seed: cycle));
            }

            var proposal = calibrator.Propose("myapp_queue_depth");

            Assert.True(proposal.IsUsable, $"only {proposal.Samples} samples");
            Assert.True(proposal.ProposedMinAbsoluteGap > proposal.PeerGapMax,
                $"proposed {proposal.ProposedMinAbsoluteGap} is not above the observed max {proposal.PeerGapMax}");

            // A name nobody observed proposes nothing rather than inventing a floor from no data.
            Assert.False(calibrator.Propose("never_seen").IsUsable);
        }

        /// <summary>
        /// The learned floors have to survive a restart, or a rolling update of the guard puts it back into
        /// the state that measured at 209 false incidents a day for as long as it takes to relearn them.
        /// </summary>
        [Fact]
        public void ACustomChannelSurvivesARoundTrip()
        {
            var calibrator = new FloorCalibrator();

            for (var cycle = 0; cycle < 20; cycle++)
            {
                calibrator.Observe(WithQueueDepth(seed: cycle));
            }

            var before = calibrator.Propose("myapp_queue_depth");
            var after = FloorCalibrator.Read(calibrator.Write()).Propose("myapp_queue_depth");

            Assert.True(after.IsUsable);
            Assert.Equal(before.PeerGapMax, after.PeerGapMax, 6);
            Assert.Equal(before.ProposedMinAbsoluteGap, after.ProposedMinAbsoluteGap, 6);
        }

        /// <summary>The healthy window plus one channel the enum knows nothing about.</summary>
        private static MetricWindow WithQueueDepth(int seed)
        {
            var names = new List<string>(12);

            for (var p = 0; p < 12; p++)
            {
                names.Add($"pod-{p}");
            }

            var window = new MetricWindow(
                names, 80, T0.AddMinutes(5 * seed), TimeSpan.FromSeconds(15), ["myapp_queue_depth"]);
            var rng = new Random(20260802 + seed);

            for (var pod = 0; pod < names.Count; pod++)
            {
                var depth = window.Series(pod, "myapp_queue_depth");
                var cpu = window.Series(pod, MetricIndex.CpuUsageRatio);
                var level = 40.0 * (1.0 + ((rng.NextDouble() - 0.5) * 0.5));

                for (var i = 0; i < window.Length; i++)
                {
                    depth[i] = level * (1.0 + ((rng.NextDouble() - 0.5) * 0.06));
                    cpu[i] = 0.02;
                }
            }

            return window;
        }

        /// <summary>Twelve replicas, some of which have restarted once and some twice.</summary>
        private static MetricWindow Restarting(int seed)
        {
            var window = Healthy(seed);

            for (var pod = 0; pod < window.Pods.Count; pod++)
            {
                var restarts = window.Series(pod, MetricIndex.ContainerRestarts);

                for (var i = 0; i < window.Length; i++)
                {
                    restarts[i] = pod % 3;
                }
            }

            return window;
        }

        /// <summary>
        /// <b>The claim.</b> Applying the proposed floors to the very period they were derived from must
        /// leave it quiet — otherwise the proposal is arithmetic that does not do the job it exists for.
        /// </summary>
        [Fact]
        public void TheProposedFloorsSilenceThePeriodTheyCameFrom()
        {
            var calibrator = new FloorCalibrator();
            var windows = new List<MetricWindow>();

            for (var cycle = 0; cycle < 20; cycle++)
            {
                var window = Healthy(seed: cycle);
                windows.Add(window);
                calibrator.Observe(window);
            }

            var proposals = calibrator.Propose();
            var gaps = new double[(int)MetricIndex.Count];
            var changes = new double[(int)MetricIndex.Count];

            for (var m = 0; m < proposals.Length; m++)
            {
                gaps[m] = proposals[m].ProposedMinAbsoluteGap;
                changes[m] = proposals[m].ProposedMinAbsoluteTrendChange;
            }

            var heap = proposals[(int)MetricIndex.GcGen2HeapBytes];

            _output.WriteLine(
                $"heap: typical {heap.TypicalMagnitude / 1e6:F1} MB, gap p99 {heap.PeerGapP99 / 1e6:F2} MB, " +
                $"gap max {heap.PeerGapMax / 1e6:F2} MB -> floor {heap.ProposedMinAbsoluteGap / 1e6:F2} MB");

            var before = Count(windows, null, null);
            var after = Count(windows, gaps, changes);

            _output.WriteLine($"incidents opened: {before} without floors, {after} with");

            Assert.True(before > 0, "the healthy period produced no findings at all, so nothing is proven");
            Assert.Equal(0, after);
        }

        private static int Count(
            List<MetricWindow> windows, IReadOnlyList<double>? gaps, IReadOnlyList<double>? changes)
        {
            var sink = new CountingSink();

            var guard = new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "lab",
                    Workload = "svc",
                    MinAbsoluteGap = gaps,
                    MinAbsoluteTrendChange = changes,
                    Grouping = IncidentGroupingOptions.Balanced with { Topology = TopologyWeights.SingleNode },
                },
                sink,
                IncidentTrackingOptions.Balanced);

            var opened = 0;

            for (var i = 0; i < windows.Count; i++)
            {
                opened += guard.RunCycle(windows[i], T0.AddMinutes(5 * i)).Opened;
            }

            return opened;
        }

        /// <summary>
        /// Twelve replicas doing nothing wrong, at the scale the lab measured: a gen2 heap of a few
        /// megabytes and CPU at a few percent of a core.
        ///
        /// <para>The shape matters more than the scale. Each replica gets a <b>persistent</b> offset — one
        /// happens to sit at 5 MB while the group sits at 4 — because that is what allocation history and
        /// collection timing actually produce, and because it is the only shape a rank test can see. White
        /// noise around a shared mean separates nothing and would make this test vacuous, which is what the
        /// first version of it did. A persistent megabyte gives Cliff's delta 1.0 and a 25% relative gap: a
        /// clean, confident, entirely meaningless finding.</para>
        /// </summary>
        private static MetricWindow Healthy(int seed)
        {
            var names = new List<string>(12);

            for (var p = 0; p < 12; p++)
            {
                names.Add($"pod-{p}");
            }

            var window = new MetricWindow(names, 80, T0.AddMinutes(5 * seed), TimeSpan.FromSeconds(15));
            var rng = new Random(20260801 + seed);

            for (var pod = 0; pod < names.Count; pod++)
            {
                var heap = window.Series(pod, MetricIndex.GcGen2HeapBytes);
                var cpu = window.Series(pod, MetricIndex.CpuUsageRatio);
                var rps = window.Series(pod, MetricIndex.RequestsPerSecond);

                // Where this replica happens to sit this cycle, and how fast it is filling up between
                // collections. Both are re-drawn per cycle: no pod is persistently the odd one out, which is
                // precisely why none of these findings is worth waking anyone for.
                var heapLevel = 4.0e6 * (1.0 + ((rng.NextDouble() - 0.5) * 0.5));
                var heapRamp = 1.2e6 * rng.NextDouble();
                var cpuLevel = 0.02 * (1.0 + ((rng.NextDouble() - 0.5) * 0.5));

                for (var i = 0; i < window.Length; i++)
                {
                    var phase = i / (double)(window.Length - 1);

                    heap[i] = (heapLevel + (heapRamp * phase)) * (1.0 + ((rng.NextDouble() - 0.5) * 0.04));
                    cpu[i] = cpuLevel * (1.0 + ((rng.NextDouble() - 0.5) * 0.06));
                    rps[i] = 6.0 * (1.0 + ((rng.NextDouble() - 0.5) * 0.1));
                }
            }

            return window;
        }

        private sealed class CountingSink : IIncidentSink
        {
            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
            }
        }
    }
}
