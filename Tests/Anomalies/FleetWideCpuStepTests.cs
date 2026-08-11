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
    /// The one fault the peer family is structurally unable to see — <c>AN-D3</c>.
    ///
    /// <para><b>What is owed and what is not.</b> The row claimed a CPU rise on every replica at once is
    /// invisible to all four families. Reading <c>AnomalyGuard.RunTrend</c> end to end refutes that as
    /// written: the cross-peer common component is built and tested separately against
    /// <c>WorkloadSubject()</c>. What nobody had shown is that the chain actually produces a finding — the
    /// 0.81-cores-against-a-0.39-core-step numbers quoted in the row are about a floor bug, not about a
    /// detection arm. This class is the fixture half of that debt: it proves the CODE PATH. An injected
    /// fleet-wide fault on the lab is what would prove the CHAIN, and it is a different claim.</para>
    ///
    /// <para><b>The gates the input must clear, quoted rather than assumed</b>
    /// (<c>LevelShiftOptions.Balanced</c>, and <c>AnomalyGuard.cs:1364</c> above them):
    /// at least <c>CrossPeerBaseline.MinimumPeers</c> = 3 pods, p &lt;= 0.01, Cliff's delta &gt;= 0.35,
    /// a relative move &gt;= 25%, and an absolute move &gt;= the floor. The absolute floor for a step
    /// resolves from the TREND table (<c>ConfiguredFloorSource.MinAbsoluteLevelShift</c>), which is
    /// 0.000326 cores in the lab's config. Against a 0.0015-core baseline the 25% gate is 0.000375, so
    /// <b>the relative gate binds and the floor does not</b> — worth knowing before anybody tunes the
    /// wrong number.</para>
    ///
    /// <para><b>The step must pass THROUGH the window.</b> <c>LevelShiftDetector</c> splits the window in
    /// half and compares the halves, so a shift that finished before the window opened is invisible by
    /// construction — "once both halves sit at the new level there is nothing to compare". Every arm here
    /// puts the step mid-window for that reason, and a finished step is pinned separately below so the
    /// limitation is recorded as behaviour rather than discovered as a bug.</para>
    /// </summary>
    public sealed class FleetWideCpuStepTests
    {
        private const int Pods = 12;

        /// <summary>Twenty minutes at fifteen seconds — the deployed window, so the sample count is real.</summary>
        private const int Samples = 80;

        /// <summary>Cores. Measured on the lab fleet, not chosen: the median replica sits here.</summary>
        private const double Baseline = 0.0015;

        private static readonly DateTimeOffset T0 = new(2026, 8, 11, 9, 0, 0, TimeSpan.Zero);

        /// <summary>
        /// <b>Capability, and it comes first on purpose.</b> Nothing this class reports about a quiet arm
        /// means anything until the mechanism has been shown able to say <c>Anomalous</c> at all — a
        /// structurally silent detector and a healthy fleet produce identical output.
        /// </summary>
        [Fact]
        public void AFleetWideStepIsReportedAgainstTheWorkload()
        {
            var rows = Run(Window(step: 1.5 * Baseline, onlyPod: null));

            var cpu = rows.Where(r => string.Equals(r.Signal, nameof(MetricIndex.CpuUsageRatio),
                StringComparison.Ordinal)).ToList();

            Assert.NotEmpty(cpu);

            // Not merely "something was reported against the workload" — the TREND family reports there too,
            // so without this an arm could pass on a neighbour's finding and the mechanism under test could
            // be dead. This sentence belongs to LevelShiftDetector alone.
            Assert.All(cpu, r => Assert.Contains("This is a step, not a drift", r.Message,
                StringComparison.Ordinal));
        }

        /// <summary>
        /// <b>The band edge, and it is where the honest positive arm sits.</b> The arm above is a 2.5x rise —
        /// the detection matrix's reference case — and with a healthy 5% scatter it separates at delta 1.00
        /// and p 7e-15, which proves the mechanism runs but says nothing about where it stops. A 30% step is
        /// just over the 25% relative gate, so this is the smallest fleet-wide movement the deployed
        /// configuration claims to catch.
        /// </summary>
        [Fact]
        public void AStepJustOverTheRelativeGateIsStillCaught()
        {
            var rows = Run(Window(step: 0.30 * Baseline, onlyPod: null));

            Assert.Contains(rows, r =>
                string.Equals(r.Signal, nameof(MetricIndex.CpuUsageRatio), StringComparison.Ordinal)
                && !r.NamesAPod);
        }

        /// <summary>
        /// The half that makes the finding useful rather than merely present: nobody is accused. Twelve
        /// findings naming twelve pods would be the failure this decomposition exists to prevent, and it is
        /// what judging each pod against the seasonal expectation produced before `AN-F1`.
        /// </summary>
        [Fact]
        public void NoIndividualReplicaIsImplicated()
        {
            var rows = Run(Window(step: 1.5 * Baseline, onlyPod: null));

            Assert.NotEmpty(rows);
            Assert.All(rows, r => Assert.False(r.NamesAPod,
                $"'{r.Signal}' was blamed on pod '{r.Pod}'; a movement every replica made belongs to the workload."));
        }

        /// <summary>The negative arm. Read only because the arm above showed the mechanism can fire.</summary>
        [Fact]
        public void AFlatFleetIsQuiet()
        {
            var rows = Run(Window(step: 0.0, onlyPod: null));

            Assert.DoesNotContain(rows, r => string.Equals(r.Signal, nameof(MetricIndex.CpuUsageRatio),
                StringComparison.Ordinal));
        }

        /// <summary>
        /// <b>The refuting arm.</b> If the workload-level step gate fired for a step on ONE replica, the
        /// common component would be contaminated by the outlier and the "everybody moved together"
        /// explanation would be wrong — the finding would mean something else entirely. It has to land
        /// differently, and this is where.
        /// </summary>
        [Fact]
        public void AStepOnASingleReplicaIsNotReportedAsAWorkloadStep()
        {
            var rows = Run(Window(step: 1.5 * Baseline, onlyPod: 0));

            Assert.DoesNotContain(rows, r =>
                string.Equals(r.Signal, nameof(MetricIndex.CpuUsageRatio), StringComparison.Ordinal)
                && !r.NamesAPod);
        }

        /// <summary>
        /// The documented blind spot, pinned so it stays a known limitation rather than becoming a
        /// surprise: a step that completed before the window opened has both halves at the new level and
        /// cannot be recovered from this window by any threshold.
        /// </summary>
        [Fact]
        public void AStepThatFinishedBeforeTheWindowOpenedIsInvisible()
        {
            var window = Window(step: 0.0, onlyPod: null, offset: 1.5 * Baseline);

            var rows = Run(window);

            Assert.DoesNotContain(rows, r => string.Equals(r.Signal, nameof(MetricIndex.CpuUsageRatio),
                StringComparison.Ordinal));
        }

        /// <summary>
        /// Below the 25% relative gate the same shape is refused, which is what stops the daily traffic
        /// curve being reported as a deployment. Sized at 10% — comfortably above the 0.000326 absolute
        /// floor, so this pins the RELATIVE gate specifically and not the floor.
        /// </summary>
        [Fact]
        public void AStepUnderTheRelativeGateIsRefused()
        {
            var rows = Run(Window(step: 0.10 * Baseline, onlyPod: null));

            Assert.DoesNotContain(rows, r => string.Equals(r.Signal, nameof(MetricIndex.CpuUsageRatio),
                StringComparison.Ordinal));
        }

        private static List<IncidentLogRecord> Run(MetricWindow window)
        {
            var sink = new CapturingSink();

            var guard = new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "lab",
                    Workload = "lab-workload",

                    // The path under test. Stated rather than inherited: the default is true, and a silent
                    // flip of it is what this whole family stops working under.
                    DecomposeCommonMode = true,

                    // No seasonal reference, so `Adjust` returns the series untouched and this fixture is
                    // about the step gate alone. With history it subtracts yesterday and adds the median
                    // back, which is a different measurement.
                    MinimumHistoryDays = 0,
                    ApplyCalibratedFloors = false,
                    WarmUpGrace = TimeSpan.Zero,
                    MinAbsoluteTrendChange = LabFloors(),
                    Grouping = IncidentGroupingOptions.Balanced with
                    {
                        Topology = TopologyWeights.SingleNode
                    },
                },
                sink,
                IncidentTrackingOptions.Balanced);

            guard.RunCycle(window, T0.AddMinutes(20));

            return sink.Rows;
        }

        /// <summary>The lab's configured floors, so the gate under test is the deployed one.</summary>
        private static double[] LabFloors()
        {
            var floors = new double[(int)MetricIndex.Count];

            for (var i = 0; i < floors.Length; i++)
            {
                // Everything except CPU pushed out of reach: this class is about one channel, and a
                // neighbouring channel firing would make an arm pass for the wrong reason.
                floors[i] = double.MaxValue;
            }

            floors[(int)MetricIndex.CpuUsageRatio] = 0.000326;

            return floors;
        }

        /// <summary>
        /// A fleet at <see cref="Baseline"/> with a step of <paramref name="step"/> applied half-way
        /// through the window — to every pod, or to one when <paramref name="onlyPod"/> is set.
        /// <paramref name="offset"/> raises the whole window instead, which is a step that already finished.
        /// </summary>
        private static MetricWindow Window(double step, int? onlyPod, double offset = 0.0)
        {
            var names = new List<string>(Pods);

            for (var p = 0; p < Pods; p++)
            {
                names.Add($"lab-workload-7765564ff6-pod{p:d2}");
            }

            var window = new MetricWindow(names, Samples, T0, TimeSpan.FromSeconds(15));

            // Fixed seed: the gates are statistical, so an arm that passes on one draw and fails on the next
            // is not evidence about the detector.
            var rng = new Random(20260811);
            var half = Samples / 2;

            for (var pod = 0; pod < Pods; pod++)
            {
                var series = window.Series(pod, MetricIndex.CpuUsageRatio);
                var stepped = onlyPod is null || onlyPod == pod;

                for (var i = 0; i < Samples; i++)
                {
                    // 5% scatter, which is what a healthy replica does between scrapes.
                    var noise = (rng.NextDouble() - 0.5) * 0.05 * Baseline;

                    series[i] = Baseline + offset + noise + (stepped && i >= half ? step : 0.0);
                }
            }

            return window;
        }

        private sealed class CapturingSink : IIncidentSink
        {
            public List<IncidentLogRecord> Rows { get; } = [];

            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
                foreach (var row in rows)
                {
                    Rows.Add(row);
                }
            }
        }
    }
}
