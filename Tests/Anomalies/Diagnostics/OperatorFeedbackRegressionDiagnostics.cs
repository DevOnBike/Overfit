// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Statistics;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.Anomalies.Diagnostics
{
    /// <summary>
    /// Does a week of an operator dismissing alerts leave the guard able to see anything?
    ///
    /// <para><b>The question this answers is the one that decides whether operator feedback is safe to
    /// ship.</b> Every response an operator can give makes the guard quieter, so the loop's failure mode is
    /// not a bug — it is the feature working as asked, repeatedly, until nothing is reported. Measuring "did
    /// false positives fall" would be circular; they fall by construction. The only question worth an
    /// experiment is whether <b>detection survives</b>.</para>
    ///
    /// <para><b>The sequence mirrors a real engagement rather than a convenient one.</b> A shadow week runs
    /// first, with the shipped defaults and no calibrated floors, which is the noisy state a client's first
    /// week is genuinely in. The operator dismisses everything it produces, with a seven-day mute on each —
    /// the most aggressive thing a reasonable person would do. Only then is the guard armed and shown the ten
    /// injected faults. Running it the other way round, calibrating first, would produce almost nothing to
    /// dismiss and the test would pass by having nothing to test.</para>
    /// </summary>
    public sealed class OperatorFeedbackRegressionDiagnostics
    {
        private const int WindowMinutes = 20;
        private const int StepMinutes = 5;
        private const double ScrapeSeconds = 15.0;
        private const int Target = 3;
        private const int Pods = 12;

        private static readonly DateTimeOffset Origin = new(2026, 7, 29, 0, 0, 0, TimeSpan.Zero);

        private readonly ITestOutputHelper _output;

        public OperatorFeedbackRegressionDiagnostics(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact("10s")]
        public void DetectionSurvivesAWeekOfDismissals()
        {
            var report = new StringBuilder();

            // ---- the shadow week: noisy on purpose, and every alert dismissed with a seven-day mute ----
            var state = new MemoryStore();
            var shadowSink = new RecordingSink();
            var shadow = new AnomalyGuard(
                Options(), shadowSink, IncidentTrackingOptions.Balanced,
                store: null, restoredAt: null, historyStore: state);

            var healthy = new SyntheticCluster(Pods, hours: 4, ScrapeSeconds, seed: 20260802,
                restartsPerPodPerDay: 0.0);

            var dismissed = 0;
            var refused = 0;

            RunCycles(shadow, healthy, (guard, at, _) =>
            {
                foreach (var id in shadowSink.Drain())
                {
                    try
                    {
                        guard.Acknowledge(
                            id, OperatorLabelKind.Noise, TimeSpan.FromDays(7), "shadow week", at);
                        dismissed++;
                    }
                    catch (ArgumentException)
                    {
                        // An incident reported by a previous process is not acknowledgeable. Counted rather
                        // than swallowed, because a silent zero here would make the whole test vacuous.
                        refused++;
                    }
                }
            });

            report.Append($"shadow week   {dismissed} incident(s) dismissed, {refused} refused\n");
            report.Append($"              {shadow.Labels.Count} label(s), "
                          + $"{shadow.Suppressions.ActiveCount(Origin.AddHours(4))} suppression(s) active\n\n");

            Assert.True(dismissed > 0,
                "the shadow week produced nothing to dismiss, so this experiment proves nothing about what "
                + "dismissals cost");

            // ---- armed, carrying every one of those dismissals through a restart ----
            var faults = Faults();
            var missed = new List<string>();
            var neverCaught = new List<string>();
            var calibrationLost = new List<string>();

            // The third arm: the same shadow week, learning the same calibration, with nobody
            // acknowledging anything. If a fault is missed here too, the operator loop is exonerated and the
            // cause is the calibration — which is the hazard FloorCalibrator documents about itself.
            var quiet = new MemoryStore();
            var quietGuard = new AnomalyGuard(
                Options(), new SubjectSink(string.Empty, true, string.Empty), IncidentTrackingOptions.Balanced,
                store: null, restoredAt: null, historyStore: quiet);

            RunCycles(quietGuard,
                new SyntheticCluster(Pods, hours: 4, ScrapeSeconds, seed: 20260802,
                    restartsPerPodPerDay: 0.0),
                (_, _, _) => { });

            report.Append($"   {"fault",-28}{"clean",8}{"week only",12}{"week+dismissals",17}\n");

            foreach (var fault in faults)
            {
                // The control runs the identical fault against a guard that has never been told anything.
                // Without it, "not detected" is unreadable: it could be the dismissals or it could be a
                // configuration that never caught this fault at all, and those call for opposite responses.
                var clean = Detect(fault, historyStore: null);
                var weekOnly = Detect(fault, historyStore: quiet);
                var afterFeedback = Detect(fault, historyStore: state);

                report.Append($"   {fault.Name,-28}{(clean >= 0 ? $"yes({clean})" : "NO"),8}"
                              + $"{(weekOnly >= 0 ? $"yes({weekOnly})" : "NO"),12}"
                              + $"{(afterFeedback >= 0 ? $"yes({afterFeedback})" : "NO"),17}\n");

                // Only a fault the guard could catch and then could not is a regression. One it never caught
                // is a gap in the configuration, reported separately rather than blamed on the operator.
                // Only a fault the shadow week's calibration still catches, and the acknowledgements then
                // lose, is a cost of operator feedback. Anything the calibration already lost belongs to the
                // calibration.
                if (weekOnly >= 0 && afterFeedback < 0)
                {
                    missed.Add(fault.Name);
                }

                if (clean >= 0 && weekOnly < 0)
                {
                    calibrationLost.Add(fault.Name);
                }

                if (clean < 0)
                {
                    neverCaught.Add(fault.Name);
                }
            }

            if (calibrationLost.Count > 0)
            {
                report.Append("\nlost to the shadow week's own calibration, before any operator acted: "
                              + string.Join(", ", calibrationLost) + "\n");
            }

            if (neverCaught.Count > 0)
            {
                report.Append("\nnot caught even clean (a configuration gap, not a feedback one): "
                              + string.Join(", ", neverCaught) + "\n");
            }

            _output.WriteLine(report.ToString());

            Assert.True(missed.Count == 0,
                "a week of dismissals cost the guard these faults: " + string.Join(", ", missed));
        }

        /// <summary>
        /// Runs one fault and returns how many cycles after the injection it was first named, or -1.
        ///
        /// <para><b>Only windows that contain the fault count.</b> The first version of this counted any
        /// naming of the subject at any time, and on a configuration with no calibrated floors the target pod
        /// gets named early by ordinary noise — so every fault read as "detected in cycle 0", including in
        /// windows that ended before the fault existed. That made the whole comparison meaningless and
        /// produced a regression that was not there.</para>
        /// </summary>
        private static int Detect(Fault fault, IIncidentStore? historyStore)
        {
            var cluster = new SyntheticCluster(Pods, hours: 4, ScrapeSeconds, seed: 20260802,
                restartsPerPodPerDay: 0.0);
            var injectedAt = cluster.Samples / 2;

            fault.Inject(cluster, injectedAt);

            var sink = new SubjectSink(
                fault.ClusterWide ? string.Empty : SyntheticCluster.PodName(Target),
                fault.ClusterWide,
                fault.Signal.ToString());

            var guard = new AnomalyGuard(
                Options(), sink, IncidentTrackingOptions.Balanced,
                store: null, restoredAt: null, historyStore: historyStore);

            var samples = (int)(WindowMinutes * 60 / ScrapeSeconds);
            var step = (int)(StepMinutes * 60 / ScrapeSeconds);
            var first = -1;

            RunCycles(guard, cluster, (_, _, start) =>
            {
                var containsFault = start + samples > injectedAt;

                if (sink.Named && containsFault && first < 0)
                {
                    // Measured from the first window that could possibly hold the fault, so a straddling
                    // window counts as a fast detection rather than as a negative latency.
                    first = Math.Max(0, (start - (injectedAt - samples + step)) / step);
                }

                sink.Named = false;
            });

            return first;
        }

        private static void RunCycles(
            AnomalyGuard guard,
            SyntheticCluster cluster,
            Action<AnomalyGuard, DateTimeOffset, int> afterCycle)
        {
            var samples = (int)(WindowMinutes * 60 / ScrapeSeconds);
            var step = (int)(StepMinutes * 60 / ScrapeSeconds);
            var names = new List<string>(Pods);

            for (var p = 0; p < Pods; p++)
            {
                names.Add(SyntheticCluster.PodName(p));
            }

            for (var start = 0; start + samples <= cluster.Samples; start += step)
            {
                var window = new MetricWindow(
                    names, samples,
                    Origin.AddSeconds(start * ScrapeSeconds),
                    TimeSpan.FromSeconds(ScrapeSeconds));

                for (var p = 0; p < Pods; p++)
                {
                    for (var m = 0; m < (int)MetricIndex.Count; m++)
                    {
                        cluster.Window(p, (MetricIndex)m, start, samples)
                            .CopyTo(window.Series(p, (MetricIndex)m));
                    }
                }

                var at = Origin.AddSeconds(start * ScrapeSeconds).AddMinutes(WindowMinutes);

                guard.RunCycle(window, at);
                afterCycle(guard, at, start);
            }
        }

        private static AnomalyGuardOptions Options()
        {
            // Identical to DetectionMatrixDiagnostics.Options(all families on). That matrix is where the
            // claim "these faults are detected" comes from; an experiment asking whether feedback costs
            // detection has to start from the configuration the claim was measured in, or a fault it never
            // caught reads as one the operator lost.
            return new AnomalyGuardOptions
            {
                Namespace = "overfit",
                Workload = "overfit-server",
                Peer = PeerOutlierOptions.Balanced,
                Trend = TrendOptions.Balanced,
                Rules = AnomalyGuardOptions.DefaultRules,
                LevelShift = LevelShiftOptions.Balanced,
                Grouping = IncidentGroupingOptions.Balanced with { Topology = TopologyWeights.SingleNode },
            };
        }

        private static List<Fault> Faults()
        {
            return
            [
                new Fault("leak 5 MB/min (one pod)", false, MetricIndex.MemoryWorkingSetBytes,
                    (c, at) => Leak(c, at, Target, 5e6)),
                new Fault("leak 20 MB/min (one pod)", false, MetricIndex.MemoryWorkingSetBytes,
                    (c, at) => Leak(c, at, Target, 20e6)),
                new Fault("leak 20 MB/min (EVERY pod)", true, MetricIndex.MemoryWorkingSetBytes,
                    (c, at) => LeakAll(c, at, 20e6)),
                new Fault("latency 3x (one pod)", false, MetricIndex.LatencyP95Ms,
                    (c, at) => Scale(c, at, Target, MetricIndex.LatencyP95Ms, 3.0)),
                new Fault("cpu 2.5x (one pod)", false, MetricIndex.CpuUsageRatio,
                    (c, at) => Scale(c, at, Target, MetricIndex.CpuUsageRatio, 2.5)),
                new Fault("cpu 2.5x (EVERY pod)", true, MetricIndex.CpuUsageRatio,
                    (c, at) => ScaleAll(c, at, MetricIndex.CpuUsageRatio, 2.5)),
            ];
        }

        private static void Leak(SyntheticCluster cluster, int at, int pod, double perMinute)
        {
            var series = cluster.Series(pod, MetricIndex.MemoryWorkingSetBytes);
            var perSample = perMinute * (ScrapeSeconds / 60.0);

            for (var i = at; i < series.Length; i++)
            {
                series[i] += perSample * (i - at);
            }
        }

        private static void LeakAll(SyntheticCluster cluster, int at, double perMinute)
        {
            for (var p = 0; p < Pods; p++)
            {
                Leak(cluster, at, p, perMinute);
            }
        }

        private static void Scale(SyntheticCluster cluster, int at, int pod, MetricIndex metric, double factor)
        {
            var series = cluster.Series(pod, metric);

            for (var i = at; i < series.Length; i++)
            {
                series[i] *= factor;
            }
        }

        private static void ScaleAll(SyntheticCluster cluster, int at, MetricIndex metric, double factor)
        {
            for (var p = 0; p < Pods; p++)
            {
                Scale(cluster, at, p, metric, factor);
            }
        }

        /// <param name="Signal">
        /// The channel the fault moves. Checked against the reported row, because matching on the subject
        /// alone let an unrelated <c>GcPauseRatio</c> incident on the same pod count as detecting a CPU
        /// fault — which manufactured a regression in the operator loop that did not exist.
        /// </param>
        private sealed record Fault(
            string Name, bool ClusterWide, MetricIndex Signal, Action<SyntheticCluster, int> Inject);

        /// <summary>Collects every incident identifier reported, so the operator can dismiss them.</summary>
        private sealed class RecordingSink : IIncidentSink
        {
            private readonly List<long> _ids = [];

            public IReadOnlyList<long> Drain()
            {
                var copy = _ids.ToArray();
                _ids.Clear();

                return copy;
            }

            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
                for (var i = 0; i < rows.Length; i++)
                {
                    if (rows[i].State == IncidentState.Opened)
                    {
                        _ids.Add(rows[i].IncidentId);
                    }
                }
            }
        }

        /// <summary>Named only when the subject <b>and</b> the signal match.</summary>
        private sealed class SubjectSink : IIncidentSink
        {
            private readonly string _pod;
            private readonly bool _workloadLevel;
            private readonly string _signal;

            public SubjectSink(string pod, bool workloadLevel, string signal)
            {
                _pod = pod;
                _workloadLevel = workloadLevel;
                _signal = signal;
            }

            public bool Named
            {
                get; set;
            }

            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
                for (var i = 0; i < rows.Length; i++)
                {
                    var namesTheSubject = _workloadLevel
                        ? rows[i].Pod.Length == 0
                        : string.Equals(rows[i].Pod, _pod, StringComparison.Ordinal);

                    Named |= namesTheSubject
                             && string.Equals(rows[i].Signal, _signal, StringComparison.Ordinal)
                             && rows[i].State != IncidentState.Resolved;
                }
            }
        }

        private sealed class MemoryStore : IIncidentStore
        {
            private string? _state;

            /// <summary>Memory does not fill up in a test; there is nothing to report.</summary>
            public string? LastError => null;

            public string? Load() => _state;

            public void Save(string state) => _state = state;
        }
    }
}
