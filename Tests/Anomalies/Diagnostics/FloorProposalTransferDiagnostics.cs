// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Anomalies.Diagnostics
{
    /// <summary>
    /// Whether floors proposed from one healthy week are worth anything on a different one — and whether a
    /// guard configured from them can still see a fault.
    ///
    /// <para><b>Why this exists at all.</b> <see cref="FalsePositiveRateDiagnostics"/> states the trap in its
    /// own words: floors picked off the healthy spread and then scored against that same spread produce a
    /// result that is guaranteed rather than earned. <see cref="FloorCalibrator"/> does exactly that kind of
    /// fitting, so the only honest way to read it is to <b>fit on one draw and score on another</b>. That is
    /// what this runs.</para>
    ///
    /// <para><b>And the second half is the one that can fail badly.</b> Any floor high enough silences any
    /// detector; a false-positive count on its own is a metric that rewards deafness. So the same three
    /// configurations are also run against a population with a deliberate leak in one replica, and a
    /// configuration that goes quiet on the healthy draw <i>and</i> quiet on the broken one has not been
    /// calibrated, it has been switched off.</para>
    ///
    /// <para>Three arms: no absolute floors at all, the hand-reasoned floors this project argued out from what
    /// the numbers mean, and the calibrator's proposal. Knobs: <c>OVERFIT_FLOOR_PODS</c>,
    /// <c>OVERFIT_FLOOR_HOURS</c>, <c>OVERFIT_FLOOR_FIT_SEED</c>, <c>OVERFIT_FLOOR_TEST_SEED</c>.</para>
    /// </summary>
    public sealed class FloorProposalTransferDiagnostics
    {
        private const int WindowMinutes = 20;
        private const int StepMinutes = 5;
        private const double ScrapeSeconds = 15.0;

        private static readonly DateTimeOffset Origin = new(2026, 7, 29, 0, 0, 0, TimeSpan.Zero);

        private readonly ITestOutputHelper _output;

        public FloorProposalTransferDiagnostics(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact("33s")]
        public void ProposedFloorsTransferToAPopulationTheyWereNotFittedOn()
        {
            var pods = Env("OVERFIT_FLOOR_PODS", 20);
            var hours = Env("OVERFIT_FLOOR_HOURS", 24);
            var fitSeed = Env("OVERFIT_FLOOR_FIT_SEED", 20260729);
            var testSeed = Env("OVERFIT_FLOOR_TEST_SEED", 20260801);

            var fit = new SyntheticCluster(pods, hours, ScrapeSeconds, fitSeed, restartsPerPodPerDay: 1.0);
            var test = new SyntheticCluster(pods, hours, ScrapeSeconds, testSeed, restartsPerPodPerDay: 1.0);

            var calibrator = new FloorCalibrator();

            foreach (var window in Windows(fit, pods))
            {
                calibrator.Observe(window);
            }

            var proposals = calibrator.Propose();
            var proposed = new double[(int)MetricIndex.Count];

            for (var m = 0; m < proposals.Length; m++)
            {
                proposed[m] = proposals[m].IsUsable ? proposals[m].ProposedMinAbsoluteGap : 0.0;
            }

            var report = new StringBuilder();

            report.Append($"population    {pods} pods, {hours} h, {WindowMinutes} min window every {StepMinutes} min\n");
            report.Append($"fitted on     seed {fitSeed}\n");
            report.Append($"scored on     seed {testSeed}  (held out — never seen by the calibrator)\n\n");

            report.Append($"   {"metric",-24}{"typical",12}{"healthy max gap",18}{"proposed floor",16}{"hand-reasoned",16}\n");

            for (var m = 0; m < proposals.Length; m++)
            {
                if (!proposals[m].IsUsable || proposals[m].ProposedMinAbsoluteGap <= 0.0)
                {
                    continue;
                }

                report.Append($"   {(MetricIndex)m,-24}{proposals[m].TypicalMagnitude,12:G4}"
                              + $"{proposals[m].PeerGapMax,18:G4}{proposed[m],16:G4}"
                              + $"{HandReasonedFloor((MetricIndex)m),16:G4}\n");
            }

            var hand = new double[(int)MetricIndex.Count];

            for (var m = 0; m < hand.Length; m++)
            {
                hand[m] = HandReasonedFloor((MetricIndex)m);
            }

            var leaking = InjectLeak(
                new SyntheticCluster(pods, hours, ScrapeSeconds, testSeed, restartsPerPodPerDay: 1.0), pod: 3);

            report.Append("\n   arm                     healthy incidents      per day     leak found in\n");

            Append(report, "no absolute floors", null, test, leaking, pods, hours);
            Append(report, "hand-reasoned", hand, test, leaking, pods, hours);
            Append(report, "calibrator proposal", proposed, test, leaking, pods, hours);

            report.Append("\nThe last column is the veto. An arm that is quiet on both populations has not been\n");
            report.Append("calibrated — it has been switched off, and the false-positive column is meaningless.\n");

            _output.WriteLine(report.ToString());

            Assert.True(proposals[(int)MetricIndex.GcGen2HeapBytes].IsUsable,
                "the fitting population produced too few observations to propose anything");
        }

        private void Append(
            StringBuilder report, string name, double[]? floors,
            SyntheticCluster healthy, SyntheticCluster leaking, int pods, int hours)
        {
            var quiet = Count(healthy, pods, floors, out _);
            var broken = Count(leaking, pods, floors, out var leakCycles);

            report.Append($"   {name,-24}{quiet,10}{quiet * 24.0 / hours,14:F0}"
                          + $"{leakCycles,15} cycle(s)\n");

            // `broken` is the total on the leaking population and is not reported as a column: it mixes the
            // leak's own incidents with the same background noise the healthy column already measures, so it
            // would say less than either number it is made of.
            _ = broken;
        }

        /// <summary>
        /// Incidents opened over the whole history, plus how many cycles named the leaking pod.
        /// </summary>
        private static int Count(SyntheticCluster cluster, int pods, double[]? floors, out int leakCycles)
        {
            var sink = new LeakWatchingSink(SyntheticCluster.PodName(3));

            var guard = new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "overfit",
                    Workload = "overfit-server",
                    MinAbsoluteGap = floors,
                    Grouping = IncidentGroupingOptions.Balanced with { Topology = TopologyWeights.SingleNode },
                },
                sink,
                IncidentTrackingOptions.Balanced);

            var opened = 0;
            var at = Origin;

            foreach (var window in Windows(cluster, pods))
            {
                sink.NamedThisCycle = false;
                opened += guard.RunCycle(window, at).Opened;
                at = at.AddMinutes(StepMinutes);

                if (sink.NamedThisCycle)
                {
                    sink.Cycles++;
                }
            }

            leakCycles = sink.Cycles;

            return opened;
        }

        /// <summary>Walks the generated history in the guard's own window-and-cadence shape.</summary>
        private static IEnumerable<MetricWindow> Windows(SyntheticCluster cluster, int pods)
        {
            var samples = (int)(WindowMinutes * 60 / ScrapeSeconds);
            var step = (int)(StepMinutes * 60 / ScrapeSeconds);
            var names = new List<string>(pods);

            for (var p = 0; p < pods; p++)
            {
                names.Add(SyntheticCluster.PodName(p));
            }

            for (var start = 0; start + samples <= cluster.Samples; start += step)
            {
                var window = new MetricWindow(
                    names, samples,
                    Origin.AddSeconds(start * ScrapeSeconds),
                    TimeSpan.FromSeconds(ScrapeSeconds));

                for (var p = 0; p < pods; p++)
                {
                    for (var m = 0; m < (int)MetricIndex.Count; m++)
                    {
                        cluster.Window(p, (MetricIndex)m, start, samples)
                            .CopyTo(window.Series(p, (MetricIndex)m));
                    }
                }

                yield return window;
            }
        }

        /// <summary>
        /// A real leak in one replica: working set and gen2 heap climb 5 MB per minute over the back half of
        /// the history, which is the shape the lab injects and the size an operator would act on.
        /// </summary>
        private static SyntheticCluster InjectLeak(SyntheticCluster cluster, int pod)
        {
            var from = cluster.Samples / 2;
            var perSample = 5e6 / (60.0 / ScrapeSeconds);

            var memory = cluster.Series(pod, MetricIndex.MemoryWorkingSetBytes);
            var heap = cluster.Series(pod, MetricIndex.GcGen2HeapBytes);

            for (var i = from; i < cluster.Samples; i++)
            {
                var climb = (i - from) * perSample;

                memory[i] += climb;
                heap[i] += climb;
            }

            return cluster;
        }

        /// <summary>
        /// The floors this project argued out from what each number means, restated here so the two
        /// approaches are scored against each other rather than each against nothing.
        /// </summary>
        private static double HandReasonedFloor(MetricIndex metric)
        {
            return metric switch
            {
                MetricIndex.GcPauseRatio => 0.01,
                MetricIndex.LatencyP50Ms => 50.0,
                MetricIndex.LatencyP95Ms => 50.0,
                MetricIndex.LatencyP99Ms => 50.0,
                MetricIndex.RequestsPerSecond => 1.0,
                MetricIndex.MemoryWorkingSetBytes => 100e6,
                MetricIndex.GcGen2HeapBytes => 100e6,
                MetricIndex.CpuUsageRatio => 0.1,
                MetricIndex.CpuThrottleRatio => 0.05,
                MetricIndex.ContainerRestarts => 1.0,
                MetricIndex.OomEventsRate => 0.0001,
                MetricIndex.ErrorRate => 0.01,
                MetricIndex.ThreadPoolQueueLength => 5.0,
                _ => 0.0
            };
        }

        private static int Env(string name, int fallback)
        {
            var raw = Environment.GetEnvironmentVariable(name);

            return int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out var value)
                ? value
                : fallback;
        }

        /// <summary>Notices whether a cycle said anything about the pod that was deliberately broken.</summary>
        private sealed class LeakWatchingSink : IIncidentSink
        {
            private readonly string _pod;

            public LeakWatchingSink(string pod)
            {
                _pod = pod;
            }

            public bool NamedThisCycle
            {
                get; set;
            }

            public int Cycles
            {
                get; set;
            }

            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
                for (var i = 0; i < rows.Length; i++)
                {
                    if (string.Equals(rows[i].Pod, _pod, StringComparison.Ordinal))
                    {
                        NamedThisCycle = true;
                    }
                }
            }
        }
    }
}
