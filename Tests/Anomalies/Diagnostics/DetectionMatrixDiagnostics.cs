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
    /// <para><b>Detection requires the right signal, and it did not always.</b> Until 2026-08-03 a cycle
    /// counted as detecting a fault if it named the affected <i>subject</i> — any row about that pod, on any
    /// channel. Against a population with a background false-positive rate that is not a detection test, it
    /// is a coincidence test: an unrelated <c>GcPauseRatio</c> incident on the same pod scored as catching a
    /// CPU fault. The <c>during</c>/<c>after</c>/<c>pre</c> split limited the damage and could not remove it,
    /// because the background rate does not stop at the injection. Both criteria are now reported — the loose
    /// one and the strict one — because the gap between them is the size of the error, and it belongs in the
    /// output rather than in a changelog nobody reads next to the table.</para>
    ///
    /// <para><b>Which signals count is derived from the injection, not declared beside it.</b> A hand-written
    /// list per fault is a second copy of the same fact, and the two copies drift — which is the exact
    /// failure being repaired here. Instead the fault is applied to a throwaway clone and the channels that
    /// moved are diffed out. A fault cannot be credited for a channel it never touched, and adding a fault
    /// cannot forget to update a table.</para>
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
                          + $"{"  families that caught it alone",-34}{"pre",5}{"opened",8}{"  loose",9}\n");

            var channels = new List<string>();
            var wantRows = Environment.GetEnvironmentVariable("OVERFIT_MATRIX_ROWS");
            var dump = new StringBuilder();

            foreach (var fault in faults)
            {
                var signals = AffectedSignals(fault, pods, hours, seed);

                channels.Add($"   {fault.Name,-26}{string.Join(", ", signals)}");

                var dumping = !string.IsNullOrWhiteSpace(wantRows)
                              && fault.Name.Contains(wantRows, StringComparison.OrdinalIgnoreCase);

                var results = new List<(string Name, Detection Detection)>();

                foreach (var arm in arms)
                {
                    var cluster = new SyntheticCluster(pods, hours, ScrapeSeconds, seed, restartsPerPodPerDay: 0.0);
                    var injectedAt = cluster.Samples / 2;

                    fault.Inject(cluster, injectedAt);

                    var rows = dumping ? new Dictionary<string, int>(StringComparer.Ordinal) : null;
                    var cycles = dumping ? new List<GuardCycleResult>() : null;

                    results.Add((
                        arm.Name,
                        Run(cluster, pods, arm.Options, injectedAt, fault.ClusterWide, signals, null, rows,
                            cycles)));

                    if (rows is null || cycles is null)
                    {
                        continue;
                    }

                    var findings = 0;
                    var grouped = 0;
                    var unevaluable = 0;

                    for (var c = 0; c < cycles.Count; c++)
                    {
                        findings += cycles[c].Findings;
                        grouped += cycles[c].Incidents;
                        unevaluable += cycles[c].UnevaluableMetrics;
                    }

                    dump.Append($"\nrows reported — {fault.Name}, arm '{arm.Name}'"
                                + $" ({rows.Count} distinct; over {cycles.Count} cycles"
                                + $" findings {findings}, incidents {grouped},"
                                + $" unevaluable {unevaluable})\n");

                    foreach (var key in rows.Keys.Order(StringComparer.Ordinal))
                    {
                        dump.Append($"   {rows[key],4}x  {key}\n");
                    }
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

                // The loose column is printed only where it disagrees. Printing it everywhere would invite
                // reading the pair as two results; the only thing it has to say is where the old criterion
                // was crediting a coincidence.
                var loose = all.Found == all.LooseFound
                    ? string.Empty
                    : all.LooseFound ? "YES*" : "no*";

                report.Append($"   {fault.Name,-26}{(all.Found ? "YES" : "no"),10}{latency,9}"
                              + $"{all.CyclesDuring,8}{all.CyclesAfter,7}"
                              + $"  {(solo.Count == 0 ? "(none)" : string.Join(" + ", solo)),-32}"
                              + $"{all.CyclesBefore,5}{all.OpenedTotal,8}{loose,9}\n");
            }

            report.Append("\ndetected   a cycle named the affected subject AND a channel the fault moved\n");
            report.Append("latency    wall-clock from injection to the first cycle that named it\n");
            report.Append("during     cycles whose window STRADDLES the injection. A step is only visible\n");
            report.Append("           here: once both halves sit at the new level there is nothing to compare\n");
            report.Append("after      cycles whose window lies entirely after it. A ramp lives here; a step\n");
            report.Append("           cannot, and a low number is the shape of the fault, not a weakness\n");
            report.Append("pre        cycles that named it BEFORE injection. Non-zero means the 'detection'\n");
            report.Append("           is partly the background false-positive rate, not the fault\n");
            report.Append("loose      shown only where the OLD subject-only criterion disagreed. YES* is a\n");
            report.Append("           row that used to read as detected on an unrelated channel\n");

            report.Append("\nchannels the injection moves, diffed from a clean clone rather than declared:\n");

            foreach (var line in channels)
            {
                report.Append(line);
                report.Append('\n');
            }

            report.Append(dump);

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
                    injectedAt, fault.ClusterWide, AffectedSignals(fault, pods, hours, seed), traces);

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
        /// <param name="LooseFound">
        /// What the subject-only criterion would have said. Carried alongside rather than replaced, so the
        /// correction is visible in the tool's own output instead of only in its history.
        /// </param>
        /// <param name="OpenedTotal">
        /// Incidents opened across the whole run, about <b>any</b> subject and on any signal.
        ///
        /// <para>Here because a detection column on its own cannot be read. Every gate in this subsystem
        /// trades detections against noise, so a change that lights up this table may simply have lowered a
        /// bar — and on a population with one faulty replica out of twelve, most of what this counts is
        /// noise. Detection and cost belong in the same output or the cost is measured later, elsewhere, by
        /// somebody who does not connect the two.</para>
        /// </param>
        private readonly record struct Detection(
            bool Found, int CyclesToFirst, int CyclesDuring, int CyclesAfter, int CyclesBefore,
            bool LooseFound, int OpenedTotal)
        {
            /// <summary>Cycles that named the subject once the fault existed, straddling or not.</summary>
            public int CyclesNaming => CyclesDuring + CyclesAfter;
        }

        private static Detection Run(
            SyntheticCluster cluster, int pods, AnomalyGuardOptions options, int injectedAt, bool clusterWide,
            IReadOnlyList<string> signals, List<PeerDecisionTrace>? peerTrace = null,
            Dictionary<string, int>? rows = null, List<GuardCycleResult>? cycles = null)
        {
            var sink = new SubjectWatchingSink(
                clusterWide ? string.Empty : SyntheticCluster.PodName(Target), clusterWide, signals)
            {
                Rows = rows,
            };

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
            var loose = false;
            var openedTotal = 0;

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

                sink.NamedSubject = false;
                sink.NamedSignal = false;

                var at = Origin.AddSeconds(start * ScrapeSeconds).AddMinutes(WindowMinutes);

                GuardCycleResult outcome;

                if (peerTrace is null)
                {
                    outcome = guard.RunCycle(window, at);
                }
                else
                {
                    outcome = guard.RunCycle(window, at, null, t =>
                    {
                        if (string.Equals(t.Pod, SyntheticCluster.PodName(Target), StringComparison.Ordinal))
                        {
                            peerTrace.Add(t);
                        }
                    });
                }

                cycles?.Add(outcome);
                openedTotal += outcome.Opened;

                var faultExisted = start + samples > injectedAt;

                if (sink.NamedSubject && faultExisted)
                {
                    loose = true;
                }

                if (!sink.NamedSignal)
                {
                    continue;
                }

                if (start >= injectedAt)
                {
                    after++;
                }
                else if (faultExisted)
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

            return new Detection(
                first >= 0, Math.Max(first, 0), during, after, before, loose, openedTotal);
        }

        /// <summary>
        /// Which channels a fault actually moves, obtained by applying it to a throwaway clone and diffing.
        ///
        /// <para>The clean and injected clusters are built with the same seed, so every sample is identical
        /// except where the injector wrote. An exact comparison is therefore right and a tolerance would be
        /// wrong: there is no arithmetic between the two runs to accumulate error, and a tolerance would
        /// quietly excuse an injector that moves a channel by a trivial amount.</para>
        ///
        /// <para><b>Compared with <see cref="double.Equals(double)"/> rather than <c>!=</c>, and the
        /// difference is the whole method.</b> The generator punches ~0.5% of samples to
        /// <see cref="double.NaN"/> for missed scrapes and holds <c>CpuThrottleRatio</c> at NaN throughout,
        /// both deliberately. Under IEEE comparison NaN is unequal to itself, so <c>!=</c> reported every
        /// channel as moved by every fault — the derived signal set came back as "all thirteen", the strict
        /// criterion silently degenerated into the loose one it was written to replace, and the first run of
        /// this table looked plausible and was worthless. <c>double.Equals</c> treats NaN as equal to NaN,
        /// which is the identity comparison wanted here.</para>
        /// </summary>
        private static string[] AffectedSignals(Fault fault, int pods, int hours, int seed)
        {
            var clean = new SyntheticCluster(pods, hours, ScrapeSeconds, seed, restartsPerPodPerDay: 0.0);
            var dirty = new SyntheticCluster(pods, hours, ScrapeSeconds, seed, restartsPerPodPerDay: 0.0);

            fault.Inject(dirty, dirty.Samples / 2);

            var moved = new List<string>();

            for (var m = 0; m < (int)MetricIndex.Count; m++)
            {
                var metric = (MetricIndex)m;

                for (var pod = 0; pod < pods && !moved.Contains(metric.ToString()); pod++)
                {
                    var a = clean.Series(pod, metric);
                    var b = dirty.Series(pod, metric);

                    for (var i = 0; i < a.Length; i++)
                    {
                        if (!a[i].Equals(b[i]))
                        {
                            moved.Add(metric.ToString());

                            break;
                        }
                    }
                }
            }

            return [.. moved];
        }

        /// <summary>
        /// Ablation. A family is removed by making its precondition unsatisfiable rather than by a flag,
        /// because there is no flag — and a precondition is the honest way to say "this family had nothing to
        /// contribute", since it is the same code path a real cluster takes when it cannot decide.
        ///
        /// <para><b>The trend ablation is NOT isolated, and this column has already been misread because of
        /// it.</b> <c>FloorCalibrator</c> fits its trend-change samples with the same <c>TrendOptions</c>
        /// instance the guard was configured with, so setting <c>MinimumSamples</c> to 100 000 to silence the
        /// trend family also stops the calibrator learning, and every calibrated absolute floor collapses to
        /// zero. Any family whose gate reads <c>MinAbsoluteTrendChange</c> — the level-shift detector does —
        /// therefore looks stronger without trend than with it, and the difference is the gate, not the
        /// family.</para>
        ///
        /// <para>Measured on the cluster-wide CPU fault: the step detector reports it in the <c>shift</c> arm
        /// with the floor at 0, and is silent in every arm where the calibrator was allowed to learn, where
        /// the floor sits at 0.81–1.74 against a step of 0.39. Reading that as "trend suppresses shift" is
        /// wrong and was the first conclusion drawn.</para>
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
                    : PeerOutlierOptions.Balanced with
                    {
                        MinimumPeers = 999
                    },

                // A window is 80 samples; demanding 100000 makes every verdict InsufficientData.
                Trend = trend
                    ? TrendOptions.Balanced
                    : TrendOptions.Balanced with
                    {
                        MinimumSamples = 100_000
                    },

                Rules = rules ? AnomalyGuardOptions.DefaultRules : [],

                // Same trick, same reason: 100000 samples are never available, so the step detector returns
                // InsufficientData and contributes nothing. Without this the attribution column was a lie —
                // the step detector ran in every arm, so every arm appeared to catch what only it caught.
                LevelShift = shift
                    ? LevelShiftOptions.Balanced
                    : LevelShiftOptions.Balanced with
                    {
                        MinimumSamples = 100_000
                    },

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

                // The control, and the row to read first. Nothing is injected, so `detected` MUST be no —
                // a yes here means the criterion is matching background noise and every row above it is
                // worth less than it looks. Its `opened` is the population's own false-positive count over
                // the same cycles, which is what every other row's `opened` has to be compared against.
                new Fault("NO FAULT (control)", false, (_, _) => { }),
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
        /// Watches for rows about one subject, on one of the channels the fault moved.
        ///
        /// <para><b>A cluster-wide fault watches for the workload subject, not for "any row at all".</b> The
        /// first version of this counted every row, and the pre-injection column duly came back at 36 of 36
        /// cycles: it was measuring the background false-positive rate and calling it detection. A fault
        /// shared by every replica is reported against the workload — an incident row with no pod — and that
        /// is the only row that can be attributed to it.</para>
        ///
        /// <para><b>Both criteria are kept because the difference is the finding.</b> Subject-only was the
        /// second version and it has the same disease in a milder form: the right pod on the wrong channel.
        /// Deleting it would leave the correction invisible, so it is measured alongside and printed where
        /// the two disagree.</para>
        ///
        /// <para><b>Any row in the group counts, not only the incident row.</b> A group's incident row
        /// carries its <i>primary</i> finding's signal, so a genuine detection whose group was led by some
        /// other signal would be missed if only that row were read. Finding rows carry their own signal, and
        /// the fault is detected if any row in the cycle names both the subject and an affected channel.</para>
        /// </summary>
        private sealed class SubjectWatchingSink : IIncidentSink
        {
            private readonly string _pod;
            private readonly bool _workloadLevel;
            private readonly IReadOnlyList<string> _signals;

            public SubjectWatchingSink(string pod, bool workloadLevel, IReadOnlyList<string> signals)
            {
                _pod = pod;
                _workloadLevel = workloadLevel;
                _signals = signals;
            }

            /// <summary>
            /// Every row seen, when set. Exists for the case the matrix cannot explain from its own columns:
            /// a fault a single family catches alone and the full configuration does not, where the question
            /// is not "did a detector fire" but "what happened to the finding afterwards".
            /// </summary>
            public Dictionary<string, int>? Rows
            {
                get; set;
            }

            /// <summary>A row about the right subject, on any channel — the old, loose criterion.</summary>
            public bool NamedSubject
            {
                get; set;
            }

            /// <summary>A row about the right subject on a channel the fault moved.</summary>
            public bool NamedSignal
            {
                get; set;
            }

            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
                for (var i = 0; i < rows.Length; i++)
                {
                    if (Rows is not null)
                    {
                        var subject = rows[i].NamesAPod ? rows[i].Pod : "(workload)";
                        var key = $"{rows[i].Kind,-8} {subject,-38} {rows[i].Signal,-24} {rows[i].State,-8}"
                                  + (rows[i].IsSuppressed ? $" suppressed:{rows[i].SuppressedBy}" : string.Empty);

                        Rows[key] = Rows.TryGetValue(key, out var seen) ? seen + 1 : 1;
                    }

                    var matches = _workloadLevel
                        ? !rows[i].NamesAPod
                        : string.Equals(rows[i].Pod, _pod, StringComparison.Ordinal);

                    if (!matches)
                    {
                        continue;
                    }

                    NamedSubject = true;

                    for (var s = 0; s < _signals.Count; s++)
                    {
                        if (string.Equals(rows[i].Signal, _signals[s], StringComparison.Ordinal))
                        {
                            NamedSignal = true;

                            break;
                        }
                    }
                }
            }
        }
    }
}
