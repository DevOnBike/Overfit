// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Hosting;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Xunit.Abstractions;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.Anomalies.Diagnostics
{
    /// <summary>
    /// Replays a day of the lab's real Prometheus history through the guard's own cycle, as fast as the
    /// queries come back.
    ///
    /// <para><b>What this is for.</b> Tuning a threshold used to cost another day of cluster time, because the
    /// only way to see what a setting produces over a day was to live through one. The cycle now takes the
    /// moment it is evaluated as of, so the same ~288 cycles can be driven back-to-back against history
    /// Prometheus still retains. This is the first end-to-end demonstration of that, and the first measured
    /// figure this repository has for what a live <c>ReadAsync</c> round trip costs.</para>
    ///
    /// <para><b>The ceiling is 30 minutes and it is deliberately generous.</b> Nobody had measured this before
    /// the first run, so a tight bound would have been a number invented rather than observed.
    /// <c>AnomalyGuardScaleBenchmark</c> bounds only the in-memory half — it evaluates a pre-built
    /// <see cref="MetricWindow"/> with no I/O at all — so at the lab's 12 replicas it says the processing
    /// component of 288 cycles sits between roughly 1.2 s and 8.8 s, and says nothing about the ~288 round
    /// trips this makes. Thirty minutes is still about 48x faster than living through the day, and it is
    /// genuinely failable: a driver that regressed into sleeping one cadence per cycle would land near 24 h.
    /// <b>The number to read out of a run is the measured one printed at the end, not the pass.</b></para>
    ///
    /// <para><b>Three outcomes are counted separately and must stay separate.</b> A replay whose cycles went
    /// blind and a replay whose cycles threw can produce the identical incident total, and folding either into
    /// the completed count would describe a broken run as a quiet cluster. Blind cycles are expected wherever
    /// the lab was not running: the retained history has gaps, and a cycle whose whole window falls inside one
    /// is correctly blind rather than healthy.</para>
    ///
    /// <para><b>This replay is NOT yet a like-for-like stand-in for the deployed guard, and the difference is
    /// structural rather than incidental.</b> The container runs through <c>Sources/Cli</c>, which registers a
    /// <c>FileLearnedStateStore</c>; <c>AddOverfitAnomalyGuard</c> registers none, so every replay starts cold
    /// with no seasonal baseline and no accumulated floor calibration. Those exist precisely to absorb the
    /// diurnal ramp, so a cold replay should be expected to report <i>more</i> than the live guard did over
    /// the same kind of day — which is what the counts below show. Until a replay can be handed the learned
    /// state, its incident total answers "what would a cold guard have said", not "what did this guard say".</para>
    ///
    /// <para><b>Grouping uses TODAY's topology against yesterday's data</b>, because
    /// <c>PrometheusTopologySource</c> resolves pod ownership from the current cluster. A replica that has
    /// since been replaced is grouped by a heuristic instead. That is a known limit of historical replay, not
    /// a defect introduced here, and it is one reason the incident total below is a sanity check rather than a
    /// gate.</para>
    ///
    /// <para><b>REQUIRES A PORT-FORWARD TO PROMETHEUS ON :9090</b> and a guard configuration to replay with.
    /// Without the forward this fails with a bare <c>HttpRequestException: connection refused</c>, which names
    /// no cause and reads like a defect in the guard.</para>
    /// <code>
    ///   k8s\monitoring\forward.cmd
    ///   kubectl -n lab get configmap anomaly-guard-config -o jsonpath="{.data.guard\.json}" &gt; Tests\bin\lab-guard.json
    /// </code>
    /// <para>Knobs: <c>OVERFIT_LAB_PROMETHEUS</c>, <c>OVERFIT_REPLAY_CONFIG</c>,
    /// <c>OVERFIT_REPLAY_CYCLES</c>, <c>OVERFIT_REPLAY_CADENCE_SECONDS</c>,
    /// <c>OVERFIT_REPLAY_START_UTC</c> — the last one is <b>required</b>, and the run refuses to start without
    /// it; see <see cref="ResolveReplayStart"/> for the measurement that turned that from advice into a
    /// gate.</para>
    /// </summary>
    public sealed class AnomalyGuardReplayDiagnostics
    {
        /// <summary>
        /// The pass/fail bound on the whole replay, fetch and processing together. See the class remarks for
        /// why it is this loose and why it is still worth having.
        /// </summary>
        private static readonly TimeSpan Ceiling = TimeSpan.FromMinutes(30);

        /// <summary>
        /// The A1 live run of 2026-08-06/07: <b>11 incidents opened over 298 cycles</b> (24.83 h, 10.63/day),
        /// read out of <c>Tests/bin/fp-run-guard-log-FINAL.txt</c> and recorded in
        /// <c>docs/aiops/aiops-backlog.md</c>.
        ///
        /// <para>Printed beside the replayed total as a sanity check only, and <b>per completed cycle rather
        /// than per day</b>, because a replay with blind cycles evaluated fewer windows than a live run of the
        /// same length — comparing daily totals would credit the replay for the hours it could not see. Even
        /// normalised the two are not the same experiment: a different day, and grouping done with today's
        /// topology against pods that may no longer exist. A difference is something to explain, not to fail
        /// on.</para>
        /// </summary>
        private const int A1IncidentsOpened = 11;

        /// <summary>Cycles the A1 run evaluated. See <see cref="A1IncidentsOpened"/>.</summary>
        private const int A1Cycles = 298;

        private readonly ITestOutputHelper _output;

        public AnomalyGuardReplayDiagnostics(ITestOutputHelper output)
        {
            _output = output;
        }

        [LabFact(LabEndpoint.Prometheus)]
        public async Task ReplaysADayOfHistoryThroughTheGuardsCycleWithoutWaitingForIt()
        {
            var prometheus = Environment.GetEnvironmentVariable("OVERFIT_LAB_PROMETHEUS")
                             ?? "http://127.0.0.1:9090";
            var cycles = Setting("OVERFIT_REPLAY_CYCLES", 288);
            var cadence = TimeSpan.FromSeconds(Setting("OVERFIT_REPLAY_CADENCE_SECONDS", 300));

            var file = ReadConfiguration(prometheus);
            var problems = new List<string>();

            var services = new ServiceCollection();
            services.AddLogging();

            // Registered before the guard, so its own logging sink is not added on top: the counts this
            // diagnostic reports come from the cycle results, and the rows are the corroborating half.
            var sink = new CountingSink();
            services.AddSingleton<IIncidentSink>(sink);
            services.AddOverfitAnomalyGuard(
                file,
                new AnomalyGuardServiceOptions
                {
                    Cadence = cadence,
                },
                problems.Add);

            await using var provider = services.BuildServiceProvider();

            // The production wiring builds the source, the topology and the resolved options; this only wraps
            // two of them in timers. Rebuilding that configuration here by hand is what would drift from the
            // deployed guard, which is the thing the replayed counts are supposed to be comparable to.
            using var window = new TimedWindowSource(provider.GetRequiredService<IMetricWindowSource>());
            var topology = new TimedTopology(provider.GetRequiredService<IRefreshablePodTopology>());
            var options = provider.GetRequiredService<AnomalyGuardServiceOptions>();

            using var service = new AnomalyGuardService(
                options,
                window,
                sink,
                provider.GetRequiredService<ILogger<AnomalyGuardService>>(),
                topology);

            var first = ResolveReplayStart(
                Environment.GetEnvironmentVariable("OVERFIT_REPLAY_START_UTC"),
                cadence,
                cycles,
                DateTimeOffset.UtcNow);
            var perCycleMs = new List<double>(cycles);
            var report = new StringBuilder();

            var last = first + (cadence * (cycles - 1));

            report.Append(CultureInfo.InvariantCulture,
                $"replay: {cycles} cycles at {cadence.TotalSeconds:F0}s cadence, {options.Window.TotalMinutes:F0} min window, {options.EndOffset.TotalMinutes:F0} min end offset\n");
            report.Append(CultureInfo.InvariantCulture,
                $"source:  {file.Prometheus}  ns={file.Namespace} pods={file.PodRegex} workload='{file.Workload}'\n");
            report.Append(CultureInfo.InvariantCulture, $"from:    {first:u}\n");
            report.Append(CultureInfo.InvariantCulture, $"to:      {last:u}\n");

            for (var i = 0; i < problems.Count; i++)
            {
                report.Append("config:  ").Append(problems[i]).Append('\n');
            }

            report.Append('\n');
            report.Append($"{"cycle",6}{"at (utc)",22}{"kind",11}{"pods",6}{"find",6}{"inc",5}");
            report.Append($"{"open",6}{"ongo",6}{"resv",6}{"blindm",8}{"ms",8}\n");

            var completed = 0;
            var blind = 0;
            var failed = 0;
            var opened = 0;
            var resolved = 0;
            var findings = 0;

            // No Task.Delay anywhere in this loop, and that absence is the whole point: the cycle's `now` is a
            // parameter, so a day of history is walked at the speed of the queries rather than at the speed of
            // the day.
            var elapsed = Stopwatch.StartNew();

            for (var cycle = 0; cycle < cycles; cycle++)
            {
                var now = first + (cadence * cycle);
                var one = Stopwatch.GetTimestamp();
                var outcome = await service.RunCycleAsync(now, CancellationToken.None);
                var oneMs = Stopwatch.GetElapsedTime(one).TotalMilliseconds;

                perCycleMs.Add(oneMs);

                if (!outcome.TryGetResult(out var result))
                {
                    blind += outcome.Kind == GuardCycleKind.Blind ? 1 : 0;
                    failed += outcome.Kind == GuardCycleKind.Failed ? 1 : 0;

                    report.Append(CultureInfo.InvariantCulture,
                        $"{cycle,6}{now,22:u}{outcome.Kind,11}{string.Empty,37}{oneMs,8:F0}\n");

                    continue;
                }

                completed++;
                opened += result.Opened;
                resolved += result.Resolved;
                findings += result.Findings;

                // Only the cycles that decided something, plus the first, or 288 rows bury the six that
                // matter. The counts below are over every cycle regardless.
                if (cycle == 0 || result.HasNews)
                {
                    report.Append(CultureInfo.InvariantCulture,
                        $"{cycle,6}{now,22:u}{outcome.Kind,11}{window.LastPods,6}{result.Findings,6}");
                    report.Append(CultureInfo.InvariantCulture,
                        $"{result.Incidents,5}{result.Opened,6}{result.Ongoing,6}{result.Resolved,6}{result.BlindMetrics,8}{oneMs,8:F0}\n");
                }
            }

            elapsed.Stop();

            var rest = elapsed.Elapsed.TotalSeconds - window.TotalSeconds - topology.TotalSeconds;

            report.Append("\n=== outcomes ===\n");
            report.Append(CultureInfo.InvariantCulture,
                $"completed {completed}   blind {blind}   failed {failed}   (total {cycles})\n");
            report.Append(CultureInfo.InvariantCulture,
                $"findings {findings}   opened {opened}   resolved {resolved}   still open {service.Guard.OpenIncidents}   sink rows {sink.Rows}\n");
            // Which channels produced them. Without this the total answers "did anything happen" and
            // nothing else, and a replay is nearly always run to find out whether ONE channel reacted.
            report.Append("rows per signal:");

            var bySignal = sink.BySignal;

            if (bySignal.Count == 0)
            {
                report.Append(" (none)");
            }

            for (var i = 0; i < bySignal.Count; i++)
            {
                report.Append(CultureInfo.InvariantCulture, $"  {bySignal[i].Key}={bySignal[i].Value}");
            }

            report.Append('\n');

            report.Append(CultureInfo.InvariantCulture,
                $"opened per 100 completed cycles: {(completed > 0 ? opened * 100.0 / completed : 0.0):F1} (replay, {opened}/{completed}) vs {A1IncidentsOpened * 100.0 / A1Cycles:F1} (A1 live, {A1IncidentsOpened}/{A1Cycles}) — a sanity check, not a gate.\n");
            report.Append(CultureInfo.InvariantCulture,
                $"topology resolved {topology.Hits} pod lookups and missed {topology.Misses}; a miss falls back to the pod-name heuristic and groups differently.\n");
            report.Append(
                "this replay runs COLD: no learned state, so no seasonal baseline and no accumulated floor "
                + "calibration. The deployed guard has both. Expect more incidents here than it reported, and "
                + "do not read the gap as a detection regression.\n");

            report.Append("\n=== wall clock (ONE run, not best-of-N; this box also hosts the lab) ===\n");
            report.Append(CultureInfo.InvariantCulture,
                $"total          {elapsed.Elapsed.TotalSeconds,10:F2} s   ceiling {Ceiling.TotalMinutes:F0} min\n");
            report.Append(CultureInfo.InvariantCulture,
                $"window fetch   {window.TotalSeconds,10:F2} s   over {window.Reads} read(s)\n");
            report.Append(CultureInfo.InvariantCulture,
                $"topology       {topology.TotalSeconds,10:F2} s   over {topology.Refreshes} refresh(es)\n");
            report.Append(CultureInfo.InvariantCulture,
                $"the rest       {rest,10:F2} s   <-- in-memory guard processing plus logging\n");

            report.Append(Distribution("per cycle, ms   ", perCycleMs));

            var starts = window.WindowStarts;
            var walked = 0;

            for (var i = 1; i < starts.Count; i++)
            {
                walked += starts[i] > starts[i - 1] ? 1 : 0;
            }

            report.Append("\n=== did it actually replay? ===\n");
            report.Append(CultureInfo.InvariantCulture,
                $"windows returned {starts.Count}, of which {walked + (starts.Count > 0 ? 1 : 0)} advanced on the one before\n");

            if (starts.Count > 0)
            {
                report.Append(CultureInfo.InvariantCulture,
                    $"first window starts {starts[0]:u}, last {starts[^1]:u}, span {(starts[^1] - starts[0]).TotalHours:F2} h\n");
            }

            _output.WriteLine(report.ToString());

            // Non-vacuity first: 288 blind cycles would sail under the ceiling and prove nothing at all about
            // the replay, so the timing claim is only meaningful once something was actually evaluated.
            Assert.True(
                completed > 0,
                $"no cycle evaluated a window: {blind} blind, {failed} failed. Check the port-forward to "
                + $"{file.Prometheus}, the namespace '{file.Namespace}' and the regex '{file.PodRegex}', and "
                + "that Prometheus still retains the replayed range.");

            // The oracle for the property this whole task is about, and the only assertion here that a cycle
            // ignoring its `now` would redden. Everything else — counts, incidents, wall clock — reads the
            // same whether history was walked or the present was re-evaluated 288 times.
            Assert.Equal(starts.Count - 1, walked);

            Assert.True(
                elapsed.Elapsed < Ceiling,
                $"the {cycles}-cycle replay took {elapsed.Elapsed.TotalMinutes:F1} min against a "
                + $"{Ceiling.TotalMinutes:F0} min ceiling. A replay that approaches the cadence it is "
                + "replaying has stopped being a replay — look for a reintroduced per-cycle delay or a "
                + "serial re-fetch before widening this bound.");
        }

        /// <summary>
        /// Where the replay starts. <c>OVERFIT_REPLAY_START_UTC</c> is <b>required</b>, and anything that is
        /// not a timestamp is refused rather than quietly replaced.
        ///
        /// <para><b>An anchor is the difference between a tuning tool and a coin toss, so it is a gate and no
        /// longer a comment.</b> Anchoring the range at the wall clock means every invocation replays a
        /// slightly different body of history, and the point of replaying at all is to re-evaluate a
        /// <i>fixed</i> one after a threshold change. Measured: three back-to-back 288-cycle runs whose
        /// anchors differed by 7 s and 2.5 min opened 47, 43 and 37 incidents — a threshold change worth ±10%
        /// would have been indistinguishable from the anchor moving. That sensitivity is real and belongs to
        /// the detectors (the sample grid shifts phase with the anchor); pinning the anchor is what stops it
        /// contaminating a comparison.</para>
        ///
        /// <para><b>The silent half was the worse half.</b> A variable that was set but misspelt — a date with
        /// no time, a stray quote — parsed as nothing and fell back to the wall clock, so an operator who
        /// believed both runs were pinned got the unpinned spread above and no indication of it. Both the
        /// missing and the unparseable case now stop the run.</para>
        ///
        /// <para>What this does <b>not</b> enforce: that two separate runs used the <i>same</i> anchor. One
        /// process cannot see the other's environment. It enforces that every run has an explicit anchor,
        /// which is what makes the comparison the operator's stated choice rather than an accident of when
        /// they pressed enter — and the anchor is echoed into the report so a run's own output says which
        /// history it read.</para>
        /// </summary>
        /// <param name="configured">The raw <c>OVERFIT_REPLAY_START_UTC</c> value, or <c>null</c> when unset.</param>
        /// <param name="cadence">Cycle cadence, used only to suggest a whole-cadence anchor in the message.</param>
        /// <param name="cycles">Cycle count, same.</param>
        /// <param name="utcNow">The wall clock, taken by the caller so this stays testable.</param>
        /// <returns>The pinned anchor, exactly as configured.</returns>
        internal static DateTimeOffset ResolveReplayStart(
            string? configured, TimeSpan cadence, int cycles, DateTimeOffset utcNow)
        {
            // A day back, snapped down to a whole cadence — not used, only offered, so the operator's first
            // run costs one paste rather than a decision about what a good anchor looks like.
            var back = utcNow - (cadence * cycles);
            var suggestion = new DateTimeOffset(back.UtcTicks - (back.UtcTicks % cadence.Ticks), TimeSpan.Zero);

            if (string.IsNullOrWhiteSpace(configured))
            {
                throw new InvalidOperationException(
                    "OVERFIT_REPLAY_START_UTC is not set, and this replay will not run without it: two runs "
                    + "anchored seconds apart opened 47, 43 and 37 incidents over the same 288 cycles, so an "
                    + "unpinned run cannot be compared with anything. Pick an anchor and keep it for every "
                    + $"run you intend to compare, for example the last whole cadence a day back: {suggestion:u}");
            }

            if (!DateTimeOffset.TryParse(
                    configured, CultureInfo.InvariantCulture,
                    DateTimeStyles.AdjustToUniversal | DateTimeStyles.AssumeUniversal, out var parsed))
            {
                throw new InvalidOperationException(
                    $"OVERFIT_REPLAY_START_UTC is '{configured}', which is not a timestamp. It is refused "
                    + "rather than replaced, because falling back to the wall clock here produces an unpinned "
                    + $"run that looks pinned. Expected an ISO-8601 instant, for example {suggestion:u}");
            }

            return parsed;
        }

        /// <summary>
        /// The guard's own configuration, read rather than restated. Pointed at the local forward, because the
        /// deployed copy names an in-cluster DNS host that does not resolve from a dev box.
        /// </summary>
        private static AnomalyGuardConfigFile ReadConfiguration(string prometheus)
        {
            var path = Environment.GetEnvironmentVariable("OVERFIT_REPLAY_CONFIG")
                       ?? RepositoryPaths.TestsBin("lab-guard.json");

            Assert.True(File.Exists(path),
                $"no guard configuration at {path}. Pull the deployed one out of the cluster first — "
                + "kubectl -n lab get configmap anomaly-guard-config -o jsonpath=\"{.data.guard\\.json}\" — "
                + "or point OVERFIT_REPLAY_CONFIG at one. Replaying with built-in query templates would "
                + "match nothing in this lab and report a day of blind cycles.");

            var file = JsonSerializer.Deserialize<AnomalyGuardConfigFile>(
                File.ReadAllText(path),
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

            Assert.NotNull(file);

            file.Prometheus = prometheus;

            return file;
        }

        /// <summary>
        /// Median, 95th percentile and maximum. A mean alone hides the shape that matters here — one slow
        /// query in three hundred is a different finding from three hundred slow ones.
        /// </summary>
        private static string Distribution(string label, List<double> values)
        {
            if (values.Count == 0)
            {
                return $"{label} (no samples)\n";
            }

            var sorted = new List<double>(values);
            sorted.Sort();

            var p50 = sorted[sorted.Count / 2];
            var p95 = sorted[Math.Min(sorted.Count - 1, (int)(sorted.Count * 0.95))];

            return string.Create(CultureInfo.InvariantCulture,
                $"{label} min {sorted[0]:F0}   p50 {p50:F0}   p95 {p95:F0}   max {sorted[^1]:F0}\n");
        }

        private static int Setting(string name, int fallback)
        {
            var raw = Environment.GetEnvironmentVariable(name);

            return int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out var value)
                ? value
                : fallback;
        }

        /// <summary>
        /// Times what the fetch half of a cycle costs, and nothing else.
        ///
        /// <para>A decorator rather than a stopwatch inside the loop, because the loop cannot see inside
        /// <c>RunCycleAsync</c>. It deliberately does <b>not</b> dispose what it wraps: the container owns that
        /// instance and disposes it with the provider.</para>
        /// </summary>
        private sealed class TimedWindowSource : IMetricWindowSource
        {
            private readonly IMetricWindowSource _inner;
            private long _ticks;

            public TimedWindowSource(IMetricWindowSource inner) => _inner = inner;

            public IReadOnlyList<string> StalePodsExcluded => _inner.StalePodsExcluded;

            public int Reads
            {
                get; private set;
            }

            /// <summary>Pods in the window the last read produced; zero when it produced none.</summary>
            public int LastPods
            {
                get; private set;
            }

            /// <summary>
            /// Where each returned window actually starts, in read order.
            ///
            /// <para><b>This is the evidence that the replay replayed.</b> Every number in this diagnostic —
            /// the outcome counts, the incident total, the wall clock — looks exactly the same whether the
            /// cycle honoured the <c>now</c> it was handed or quietly read the clock and re-evaluated the
            /// present 288 times. The window's own start comes back from the source's response grid, so a
            /// sequence that does not walk forward is a driver that is not replaying anything.</para>
            /// </summary>
            public List<DateTimeOffset> WindowStarts { get; } = [];

            public double TotalSeconds => (double)_ticks / Stopwatch.Frequency;

            public async Task<MetricWindow?> ReadAsync(
                DateTimeOffset end, TimeSpan window, CancellationToken ct = default)
            {
                var started = Stopwatch.GetTimestamp();

                try
                {
                    var result = await _inner.ReadAsync(end, window, ct);

                    Reads++;
                    LastPods = result?.Pods.Count ?? 0;

                    if (result is not null)
                    {
                        WindowStarts.Add(result.Start);
                    }

                    return result;
                }
                finally
                {
                    _ticks += Stopwatch.GetTimestamp() - started;
                }
            }

            public void Dispose()
            {
                // Owned by the container, which disposes it with the provider. Disposing here would close the
                // shared HttpClient underneath a source the provider still believes is alive.
            }
        }

        /// <summary>
        /// Times the topology refresh separately from the window fetch, so "the rest" really is the in-memory
        /// component and can be read against <c>AnomalyGuardScaleBenchmark</c>'s I/O-free numbers.
        /// </summary>
        private sealed class TimedTopology : IRefreshablePodTopology
        {
            private readonly IRefreshablePodTopology _inner;
            private long _ticks;

            public TimedTopology(IRefreshablePodTopology inner) => _inner = inner;

            public int Refreshes
            {
                get; private set;
            }

            /// <summary>
            /// Pod lookups the current cluster could answer, and those it could not.
            ///
            /// <para><b>The measured size of historical replay's one known distortion.</b> The topology is
            /// refreshed against the cluster as it is <i>now</i>, so a replica that has since been rolled away
            /// is unknown to it and the grouper falls back to a name heuristic for that pod. The plan calls
            /// this out and leaves it out of scope to fix; counting it is what turns "grouping may differ"
            /// into a number an operator can weigh a replayed incident total against.</para>
            /// </summary>
            public int Hits
            {
                get; private set;
            }

            /// <inheritdoc cref="Hits"/>
            public int Misses
            {
                get; private set;
            }

            public double TotalSeconds => (double)_ticks / Stopwatch.Frequency;

            public bool TryResolve(string pod, out PodPlacement placement)
            {
                var resolved = _inner.TryResolve(pod, out placement);

                Hits += resolved ? 1 : 0;
                Misses += resolved ? 0 : 1;

                return resolved;
            }

            public async Task<int> RefreshAsync(CancellationToken ct = default)
            {
                var started = Stopwatch.GetTimestamp();

                try
                {
                    return await _inner.RefreshAsync(ct);
                }
                finally
                {
                    _ticks += Stopwatch.GetTimestamp() - started;
                    Refreshes++;
                }
            }
        }

        /// <summary>
        /// Counts what a consumer would have been told. The cycle results carry the authoritative counts; this
        /// is the corroborating half, and a mismatch between the two is itself worth seeing.
        /// </summary>
        private sealed class CountingSink : IIncidentSink
        {
            /// <summary>
            /// Rows per signal, because the total on its own cannot answer the question a replay is usually
            /// run to answer.
            ///
            /// <para><b>Measured need, 2026-08-09.</b> A replay of a window containing a deliberate stall
            /// reported "findings 31" and nothing else, so it could not say whether the channel under test
            /// had fired — which was the entire reason for running it. A count without the breakdown is the
            /// same defect this subsystem keeps producing in other forms: a number that is present, correct,
            /// and carries none of the information the reader came for.</para>
            /// </summary>
            private readonly Dictionary<string, int> _bySignal = new(StringComparer.Ordinal);

            public int Rows
            {
                get; private set;
            }

            /// <summary>Signals that produced rows, most frequent first.</summary>
            public IReadOnlyList<KeyValuePair<string, int>> BySignal
            {
                get
                {
                    var ordered = _bySignal.ToList();

                    ordered.Sort((a, b) => b.Value != a.Value
                        ? b.Value.CompareTo(a.Value)
                        : string.CompareOrdinal(a.Key, b.Key));

                    return ordered;
                }
            }

            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
                Rows += rows.Length;

                for (var i = 0; i < rows.Length; i++)
                {
                    var signal = rows[i].Signal;

                    // An empty signal is a real value here — an incident-level row rather than a finding —
                    // and naming it keeps the breakdown summing to the total.
                    var key = signal.Length > 0 ? signal : "(incident row)";

                    _bySignal[key] = _bySignal.GetValueOrDefault(key) + 1;
                }
            }
        }
    }
}
