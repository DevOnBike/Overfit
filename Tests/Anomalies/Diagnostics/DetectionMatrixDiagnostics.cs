// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Statistics;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Anomalies.Diagnostics
{
    /// <summary>
    /// What the guard actually catches, per fault, and which detector family caught it.
    ///
    /// <para><b>The project has one column of this story and not the other.</b>
    /// <see cref="FalsePositiveRateDiagnostics"/> says how often the guard cries wolf on a healthy cluster;
    /// nothing said what it sees when something is genuinely wrong, beyond a single degraded replica in a
    /// four-pod lab. A detector is only describable by both numbers together — a configuration with no false
    /// positives and no detections is not a good one, it is switched off.</para>
    ///
    /// <para><b>Family attribution is by ablation, not by inspection.</b> A finding does not record which
    /// detector produced it — peer, trend and rule all report under the metric's own name, deliberately, so
    /// that a query groups them. Rather than parse detector wording, each fault is run four times: everything
    /// on, then each family alone. What a family catches by itself is what that family contributes, and the
    /// difference against "all" is what the combination adds.</para>
    ///
    /// <para><b>The row that justifies the architecture is the cluster-wide leak.</b> Peer comparison asks
    /// whether a replica is unlike its peers; when every replica leaks together there is no outlier and the
    /// rank family is structurally silent. That is not a threshold to tune, and it is the reason the absolute
    /// rules family exists at all.</para>
    ///
    /// <para><b>What the first run found, recorded because it is the point of having built this.</b> A
    /// replica running <b>2.5× hotter than its eleven peers</b> — relative gap 120%, Cliff's delta 0.69,
    /// p = 0 — is <b>never named by the peer family</b>. The trace shows why: the group verdict is
    /// <c>Inconclusive</c> in every cycle, before and after injection alike, because
    /// <c>PeerGroupOutlierDetector</c> counts a member as "departing" purely on the <b>fixed 8% relative
    /// gap</b>, with no statistical test, and CPU's natural between-pod scatter on this population is 34%.
    /// A third of the group is therefore always more than 8% from the median, the group is declared to have
    /// no coherent norm, and the finding with a 120% gap is discarded along with the noise. Trend catches the
    /// step in <b>one</b> cycle and never again, because a level that has finished changing has no slope.
    /// The size gate added for <c>MinRelativeGap</c> rescued which deviations are <i>reported</i>; the
    /// coherence gate ahead of it still counts raw gaps.</para>
    ///
    /// <para>Needs no cluster. Knobs: <c>OVERFIT_MATRIX_PODS</c>, <c>OVERFIT_MATRIX_HOURS</c>,
    /// <c>OVERFIT_MATRIX_SEED</c>, <c>OVERFIT_MATRIX_TRACE</c>.</para>
    /// </summary>
    public sealed class DetectionMatrixDiagnostics
    {
        private const int WindowMinutes = 20;
        private const int StepMinutes = 5;
        private const double ScrapeSeconds = 15.0;

        /// <summary>The replica every single-pod fault is injected into.</summary>
        private const int Target = 3;

        private static readonly DateTimeOffset Origin = new(2026, 7, 29, 0, 0, 0, TimeSpan.Zero);

        private readonly ITestOutputHelper _output;

        public DetectionMatrixDiagnostics(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact]
        public void MeasuresWhatEachFamilyCatches()
        {
            var pods = Env("OVERFIT_MATRIX_PODS", 12);
            var hours = Env("OVERFIT_MATRIX_HOURS", 6);
            var seed = Env("OVERFIT_MATRIX_SEED", 20260801);

            var faults = Faults();
            var arms = new (string Name, AnomalyGuardOptions Options)[]
            {
                ("all", Options(peer: true, trend: true, rules: true, shift: true)),
                ("peer", Options(peer: true, trend: false, rules: false, shift: false)),
                ("trend", Options(peer: false, trend: true, rules: false, shift: false)),
                ("rules", Options(peer: false, trend: false, rules: true, shift: false)),
                ("shift", Options(peer: false, trend: false, rules: false, shift: true)),
            };

            var report = new StringBuilder();

            report.Append($"population   {pods} pods, {hours} h, seed {seed}\n");
            report.Append($"cadence      {WindowMinutes} min window every {StepMinutes} min\n");
            report.Append($"injection    at the half-way point, into {SyntheticCluster.PodName(Target)}"
                          + " unless the fault says otherwise\n\n");

            report.Append($"   {"fault",-26}{"detected",10}{"latency",9}{"during",8}{"after",7}"
                          + $"{"  families that caught it alone",-34}{"pre",5}\n");

            foreach (var fault in faults)
            {
                var results = new List<(string Name, Detection Detection)>();

                foreach (var arm in arms)
                {
                    var cluster = new SyntheticCluster(pods, hours, ScrapeSeconds, seed, restartsPerPodPerDay: 0.0);
                    var injectedAt = cluster.Samples / 2;

                    fault.Inject(cluster, injectedAt);

                    results.Add((arm.Name, Run(cluster, pods, arm.Options, injectedAt, fault.ClusterWide)));
                }

                var all = results[0].Detection;
                var solo = new List<string>();

                for (var i = 1; i < results.Count; i++)
                {
                    if (results[i].Detection.Found)
                    {
                        solo.Add(results[i].Name);
                    }
                }

                var latency = all.Found
                    ? $"{all.CyclesToFirst * StepMinutes} min"
                    : "-";

                report.Append($"   {fault.Name,-26}{(all.Found ? "YES" : "no"),10}{latency,9}"
                              + $"{all.CyclesDuring,8}{all.CyclesAfter,7}"
                              + $"  {(solo.Count == 0 ? "(none)" : string.Join(" + ", solo)),-32}"
                              + $"{all.CyclesBefore,5}\n");
            }

            report.Append("\ndetected   the guard named the affected subject once the fault existed\n");
            report.Append("latency    wall-clock from injection to the first cycle that named it\n");
            report.Append("during     cycles whose window STRADDLES the injection. A step is only visible\n");
            report.Append("           here: once both halves sit at the new level there is nothing to compare\n");
            report.Append("after      cycles whose window lies entirely after it. A ramp lives here; a step\n");
            report.Append("           cannot, and a low number is the shape of the fault, not a weakness\n");
            report.Append("pre        cycles that named the same subject BEFORE injection. Non-zero means the\n");
            report.Append("           'detection' is partly the background false-positive rate, not the fault\n");

            Trace(report, faults, pods, hours, seed);

            _output.WriteLine(report.ToString());

            Assert.NotEmpty(faults);
        }

        /// <summary>
        /// Every gate on the injected pod, for one named fault. Set <c>OVERFIT_MATRIX_TRACE</c> to a substring
        /// of a fault name.
        ///
        /// <para>"Not detected" has several causes that call for opposite fixes — too few usable samples, an
        /// effect size below the bar, a relative gap below it, an absolute floor above the difference, or a
        /// group that never reached a verdict at all. The numbers that decided are printed rather than the
        /// verdict, because the verdict cannot distinguish them.</para>
        /// </summary>
        private static void Trace(
            StringBuilder report, List<Fault> faults, int pods, int hours, int seed)
        {
            var wanted = Environment.GetEnvironmentVariable("OVERFIT_MATRIX_TRACE");

            if (string.IsNullOrWhiteSpace(wanted))
            {
                return;
            }

            foreach (var fault in faults)
            {
                if (!fault.Name.Contains(wanted, StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                var cluster = new SyntheticCluster(pods, hours, ScrapeSeconds, seed, restartsPerPodPerDay: 0.0);
                var injectedAt = cluster.Samples / 2;

                fault.Inject(cluster, injectedAt);

                var traces = new List<PeerDecisionTrace>();

                Run(cluster, pods, Options(peer: true, trend: false, rules: false, shift: false),
                    injectedAt, fault.ClusterWide, traces);

                report.Append($"\npeer decisions on {SyntheticCluster.PodName(Target)} — {fault.Name}\n");
                report.Append($"   {"signal",-24}{"status",-16}{"out",5}{"gap",9}{"delta",8}"
                              + $"{"p",11}{"n",5}\n");

                var shown = new HashSet<string>(StringComparer.Ordinal);

                foreach (var t in traces)
                {
                    // One line per signal per distinct status: 33 identical rows say nothing 1 does not.
                    if (!shown.Add($"{t.Signal}|{t.Status}|{t.IsOutlier}"))
                    {
                        continue;
                    }

                    report.Append($"   {t.Signal,-24}{t.Status,-16}{(t.IsOutlier ? "Y" : "n"),5}"
                                  + $"{t.RelativeGap,9:P0}{t.EffectSize,8:F2}{t.PValue,11:G3}"
                                  + $"{t.UsableSamples,5}\n");
                }
            }
        }

        /// <summary>
        /// One arm's verdict on one fault.
        ///
        /// <para><b>A window that straddles the injection is counted separately, and getting that wrong made
        /// this tool lie.</b> It used to fold straddling windows into "before", on the reasoning that such a
        /// window is mostly healthy data and crediting it would flatter the latency. That holds for a ramp and
        /// is exactly backwards for a <b>step</b>: once both halves of a window sit at the new level there is
        /// nothing left to compare, so the straddling windows are the <i>only</i> ones that can see it. The
        /// step detector duly showed up as "1 cycle, 3 before" — its real detections filed under the
        /// false-positive column.</para>
        /// </summary>
        private readonly record struct Detection(
            bool Found, int CyclesToFirst, int CyclesDuring, int CyclesAfter, int CyclesBefore)
        {
            /// <summary>Cycles that named the subject once the fault existed, straddling or not.</summary>
            public int CyclesNaming => CyclesDuring + CyclesAfter;
        }

        private static Detection Run(
            SyntheticCluster cluster, int pods, AnomalyGuardOptions options, int injectedAt, bool clusterWide,
            List<PeerDecisionTrace>? peerTrace = null)
        {
            var sink = new SubjectWatchingSink(
                clusterWide ? string.Empty : SyntheticCluster.PodName(Target), clusterWide);

            var guard = new AnomalyGuard(options, sink, IncidentTrackingOptions.Balanced);

            var samples = (int)(WindowMinutes * 60 / ScrapeSeconds);
            var step = (int)(StepMinutes * 60 / ScrapeSeconds);
            var names = new List<string>(pods);

            for (var p = 0; p < pods; p++)
            {
                names.Add(SyntheticCluster.PodName(p));
            }

            var cycle = 0;
            var first = -1;
            var during = 0;
            var after = 0;
            var before = 0;

            for (var start = 0; start + samples <= cluster.Samples; start += step, cycle++)
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

                sink.Named = false;

                var at = Origin.AddSeconds(start * ScrapeSeconds).AddMinutes(WindowMinutes);

                if (peerTrace is null)
                {
                    guard.RunCycle(window, at);
                }
                else
                {
                    guard.RunCycle(window, at, null, t =>
                    {
                        if (string.Equals(t.Pod, SyntheticCluster.PodName(Target), StringComparison.Ordinal))
                        {
                            peerTrace.Add(t);
                        }
                    });
                }

                if (!sink.Named)
                {
                    continue;
                }

                if (start >= injectedAt)
                {
                    after++;
                }
                else if (start + samples > injectedAt)
                {
                    during++;
                }
                else
                {
                    before++;

                    continue;
                }

                if (first < 0)
                {
                    // From the injection, so a straddling window that names it before the fault fills the
                    // window counts as a fast detection rather than as a negative latency.
                    first = Math.Max(0, cycle - ((injectedAt - samples + step) / step));
                }
            }

            return new Detection(first >= 0, Math.Max(first, 0), during, after, before);
        }

        /// <summary>
        /// Ablation. A family is removed by making its precondition unsatisfiable rather than by a flag,
        /// because there is no flag — and a precondition is the honest way to say "this family had nothing to
        /// contribute", since it is the same code path a real cluster takes when it cannot decide.
        /// </summary>
        private static AnomalyGuardOptions Options(bool peer, bool trend, bool rules, bool shift)
        {
            return new AnomalyGuardOptions
            {
                Namespace = "overfit",
                Workload = "overfit-server",

                // 999 peers are never available, so every group is undecidable.
                Peer = peer
                    ? PeerOutlierOptions.Balanced
                    : PeerOutlierOptions.Balanced with { MinimumPeers = 999 },

                // A window is 80 samples; demanding 100000 makes every verdict InsufficientData.
                Trend = trend
                    ? TrendOptions.Balanced
                    : TrendOptions.Balanced with { MinimumSamples = 100_000 },

                Rules = rules ? AnomalyGuardOptions.DefaultRules : [],

                // Same trick, same reason: 100000 samples are never available, so the step detector returns
                // InsufficientData and contributes nothing. Without this the attribution column was a lie —
                // the step detector ran in every arm, so every arm appeared to catch what only it caught.
                LevelShift = shift
                    ? LevelShiftOptions.Balanced
                    : LevelShiftOptions.Balanced with { MinimumSamples = 100_000 },

                Grouping = IncidentGroupingOptions.Balanced with { Topology = TopologyWeights.SingleNode },
            };
        }

        /// <summary>
        /// The fault panel, matching what <c>Demo/LabWorkload</c> can actually inject, plus the two
        /// cluster-wide variants a single lab replica cannot produce.
        /// </summary>
        private static List<Fault> Faults()
        {
            return
            [
                new Fault("leak 5 MB/min (one pod)", false, (c, at) => Leak(c, at, Target, 5e6)),
                new Fault("leak 20 MB/min (one pod)", false, (c, at) => Leak(c, at, Target, 20e6)),
                new Fault("leak 20 MB/min (EVERY pod)", true, (c, at) => LeakAll(c, at, 20e6)),
                new Fault("latency 3x (one pod)", false, (c, at) => Latency(c, at, Target, 3.0)),
                new Fault("errors 15% (one pod)", false, (c, at) => Errors(c, at, Target, 0.15)),
                new Fault("cpu 2.5x (one pod)", false, (c, at) => Cpu(c, at, Target, 2.5)),
                new Fault("cpu 2.5x (EVERY pod)", true, (c, at) => CpuAll(c, at, 2.5)),
                new Fault("throttling 30% (one pod)", false, (c, at) => Throttle(c, at, Target, 0.30)),
                new Fault("OOM kill (one pod)", false, (c, at) => Oom(c, at, Target)),
                new Fault("crash-restart (one pod)", false, (c, at) => Restart(c, at, Target)),
            ];
        }

        private static void Leak(SyntheticCluster cluster, int at, int pod, double perMinute)
        {
            var perSample = perMinute / (60.0 / ScrapeSeconds);
            var memory = cluster.Series(pod, MetricIndex.MemoryWorkingSetBytes);
            var heap = cluster.Series(pod, MetricIndex.GcGen2HeapBytes);

            for (var i = at; i < cluster.Samples; i++)
            {
                var climb = (i - at) * perSample;

                memory[i] += climb;
                heap[i] += climb;
            }
        }

        private static void LeakAll(SyntheticCluster cluster, int at, double perMinute)
        {
            for (var pod = 0; pod < cluster.Pods; pod++)
            {
                Leak(cluster, at, pod, perMinute);
            }
        }

        private static void Latency(SyntheticCluster cluster, int at, int pod, double factor)
        {
            Scale(cluster, at, pod, MetricIndex.LatencyP50Ms, factor);
            Scale(cluster, at, pod, MetricIndex.LatencyP95Ms, factor);
            Scale(cluster, at, pod, MetricIndex.LatencyP99Ms, factor);
        }

        private static void Errors(SyntheticCluster cluster, int at, int pod, double rate)
        {
            var errors = cluster.Series(pod, MetricIndex.ErrorRate);
            var rps = cluster.Series(pod, MetricIndex.RequestsPerSecond);

            for (var i = at; i < cluster.Samples; i++)
            {
                errors[i] = rps[i] * rate;
            }
        }

        private static void Cpu(SyntheticCluster cluster, int at, int pod, double factor)
            => Scale(cluster, at, pod, MetricIndex.CpuUsageRatio, factor);

        private static void CpuAll(SyntheticCluster cluster, int at, double factor)
        {
            for (var pod = 0; pod < cluster.Pods; pod++)
            {
                Cpu(cluster, at, pod, factor);
            }
        }

        private static void Throttle(SyntheticCluster cluster, int at, int pod, double ratio)
        {
            var series = cluster.Series(pod, MetricIndex.CpuThrottleRatio);

            for (var i = at; i < cluster.Samples; i++)
            {
                series[i] = ratio;
            }
        }

        /// <summary>An OOM kill moves two channels: the event itself, and the restart it causes.</summary>
        private static void Oom(SyntheticCluster cluster, int at, int pod)
        {
            var events = cluster.Series(pod, MetricIndex.OomEventsRate);

            // A rate over the scrape interval: one event, spread across the samples a rate expression would
            // see it in, rather than an impulse on a single sample.
            for (var i = at; i < Math.Min(at + 12, cluster.Samples); i++)
            {
                events[i] = 1.0 / (12 * ScrapeSeconds);
            }

            Restart(cluster, at, pod);
        }

        /// <summary>A restart on its own — a crash loop's first iteration, with no OOM behind it.</summary>
        private static void Restart(SyntheticCluster cluster, int at, int pod)
        {
            var restarts = cluster.Series(pod, MetricIndex.ContainerRestarts);

            for (var i = at; i < cluster.Samples; i++)
            {
                restarts[i] += 1.0;
            }
        }

        private static void Scale(SyntheticCluster cluster, int at, int pod, MetricIndex metric, double factor)
        {
            var series = cluster.Series(pod, metric);

            for (var i = at; i < cluster.Samples; i++)
            {
                series[i] *= factor;
            }
        }

        private static int Env(string name, int fallback)
        {
            var raw = Environment.GetEnvironmentVariable(name);

            return int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out var value)
                ? value
                : fallback;
        }

        private sealed record Fault(string Name, bool ClusterWide, Action<SyntheticCluster, int> Inject);

        /// <summary>
        /// Watches for rows about one subject.
        ///
        /// <para><b>A cluster-wide fault watches for the workload subject, not for "any row at all".</b> The
        /// first version of this counted every row, and the pre-injection column duly came back at 36 of 36
        /// cycles: it was measuring the background false-positive rate and calling it detection. A fault
        /// shared by every replica is reported against the workload — an incident row with no pod — and that
        /// is the only row that can be attributed to it.</para>
        /// </summary>
        private sealed class SubjectWatchingSink : IIncidentSink
        {
            private readonly string _pod;
            private readonly bool _workloadLevel;

            public SubjectWatchingSink(string pod, bool workloadLevel)
            {
                _pod = pod;
                _workloadLevel = workloadLevel;
            }

            public bool Named { get; set; }

            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
                for (var i = 0; i < rows.Length; i++)
                {
                    var matches = _workloadLevel
                        ? !rows[i].NamesAPod
                        : string.Equals(rows[i].Pod, _pod, StringComparison.Ordinal);

                    if (matches)
                    {
                        Named = true;
                    }
                }
            }
        }
    }
}
