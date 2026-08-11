// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text;
using System.Text.Json;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Hosting;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;
using DevOnBike.Overfit.Tests.TestSupport;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace DevOnBike.Overfit.Tests.Anomalies.Diagnostics
{
    /// <summary>
    /// An experiment, not a feature: why does a cold replay of lab history open ~21 incidents per 100
    /// completed cycles when the live guard opened 3.7 per 100 over the same kind of day?
    ///
    /// <para>Three candidate causes, and this drives them apart by measurement rather than by argument.
    /// <b>(1) Learned state</b> — <c>AddOverfitAnomalyGuard</c> registers no <see cref="ILearnedStateStore"/>,
    /// so every replay starts with no seasonal baseline and no floor calibration while the deployed guard has
    /// days of both. <b>(2) A different sample of time</b> — the recent 24 h contains a 7.7 h cluster outage
    /// and a 12→15→12 scaling experiment that the live run's day did not. <b>(3) Configuration drift</b> —
    /// settled outside this test: the live ConfigMap is byte-identical to <c>k8s/lab/anomaly-guard.yaml</c>
    /// at HEAD, last changed 2026-08-05, before the live run started.</para>
    ///
    /// <para>Cause 2 is separated by replaying <b>the live run's own window</b>, whose cycle timestamps come
    /// out of its log, and cause 1 by handing the same window the guard's real learned state.</para>
    ///
    /// <para><b>The determinism half.</b> The replay's determinism claim is scoped to "store-less replay
    /// only", because <c>AnomalyGuard</c> falls back to <c>DateTimeOffset.UtcNow</c> for <c>restoredAt</c>.
    /// That fallback is guarded by the <i>incident</i> store, not the learned-state store, so the two arms
    /// are measured separately: learned state alone, and learned state plus a non-empty incident store.</para>
    ///
    /// <para><b>REQUIRES a port-forward to Prometheus on :9090 and two files pulled out of the cluster.</b>
    /// Both reads, no writes:</para>
    /// <code>
    ///   k8s\monitoring\forward.cmd
    ///   kubectl -n lab get configmap anomaly-guard-config -o jsonpath="{.data.guard\.json}" &gt; Tests\bin\lab-guard.json
    ///   kubectl -n lab exec deploy/anomaly-guard -- cat /state/learned-state.txt &gt; Tests\bin\lab-learned-state.txt
    /// </code>
    /// </summary>
    public sealed class AnomalyGuardLearnedStateExperiment
    {
        /// <summary>
        /// The live 24-hour run of 2026-08-06/07, read out of <c>Tests/bin/fp-run-guard-log-FINAL.txt</c>:
        /// 298 cycle lines from 08-06T08:04:19Z to 08-07T08:54:14Z, 11 opened, 301 findings, 10 resolved,
        /// 12 pods in every one.
        /// </summary>
        private const int LiveOpened = 11;

        /// <inheritdoc cref="LiveOpened"/>
        private const int LiveCycles = 298;

        /// <inheritdoc cref="LiveOpened"/>
        private const int LiveFindings = 301;

        /// <summary>The first cycle of that run, so a replay can walk the same 298 moments it did.</summary>
        private static readonly DateTimeOffset LiveStart =
            new(2026, 8, 6, 8, 4, 19, TimeSpan.Zero);

        private const string HistoryMarker = "### history";
        private const string CalibrationMarker = "### calibration";
        private const string LabelMarker = "### labels";

        private readonly ITestOutputHelper _output;

        public AnomalyGuardLearnedStateExperiment(ITestOutputHelper output)
        {
            _output = output;
        }

        [LabFact(LabEndpoint.Prometheus)]
        public async Task DoesLearnedStateExplainTheReplayIncidentRateGap()
        {
            var prometheus = Environment.GetEnvironmentVariable("OVERFIT_LAB_PROMETHEUS")
                             ?? "http://127.0.0.1:9090";
            var cadence = TimeSpan.FromSeconds(300);
            var file = ReadConfiguration(prometheus);
            var pristine = ReadLearnedState();

            var services = new ServiceCollection();
            services.AddLogging();
            services.AddOverfitAnomalyGuard(file, new AnomalyGuardServiceOptions { Cadence = cadence });

            await using var provider = services.BuildServiceProvider();

            var source = provider.GetRequiredService<IMetricWindowSource>();
            var topology = provider.GetRequiredService<IRefreshablePodTopology>();
            var options = provider.GetRequiredService<AnomalyGuardServiceOptions>();
            var logger = provider.GetRequiredService<ILogger<AnomalyGuardService>>();

            // Pinned once, here, for every arm that uses it. Three back-to-back replays whose anchors
            // differed by seconds opened 47/43/37 incidents, so an anchor read per arm would make the
            // comparison a measurement of the clock.
            var recentStart = Snap(DateTimeOffset.UtcNow - (cadence * 288), cadence);

            var report = new StringBuilder();

            report.Append(CultureInfo.InvariantCulture,
                $"prometheus {file.Prometheus}  ns={file.Namespace} workload='{file.Workload}'\n");
            report.Append(CultureInfo.InvariantCulture,
                $"window R (recent):  {recentStart:u} .. {recentStart + (cadence * 287):u}  288 cycles\n");
            report.Append(CultureInfo.InvariantCulture,
                $"window L (the live run's own):  {LiveStart:u} .. {LiveStart + (cadence * (LiveCycles - 1)):u}  {LiveCycles} cycles\n");
            report.Append(CultureInfo.InvariantCulture,
                $"learned state: {pristine.Length} chars, {Buckets(pristine)} history buckets, "
                + $"{BucketsWithAtLeast(pristine, 2)} of them holding >= {options.Guard.MinimumHistoryDays} days "
                + $"(MinimumHistoryDays={options.Guard.MinimumHistoryDays})\n\n");

            var arms = new List<Arm>();

            // Order is deliberate: two cold runs of window R bracket everything else, so if the answer moved
            // because Prometheus gained or lost data mid-experiment rather than because of an arm, the
            // bracket says so. A canary, not decoration.
            arms.Add(await RunArm("R cold #1", recentStart, 288, null, null));
            arms.Add(await RunArm("R warm (full state) #1", recentStart, 288, pristine, null));
            arms.Add(await RunArm("R warm (full state) #2", recentStart, 288, pristine, null));
            arms.Add(await RunArm("R warm (calibration only)", recentStart, 288, WithoutHistory(pristine), null));
            arms.Add(await RunArm("R warm (history only)", recentStart, 288, WithoutCalibration(pristine), null));
            arms.Add(await RunArm("L cold", LiveStart, LiveCycles, null, null));
            arms.Add(await RunArm("L warm (full state)", LiveStart, LiveCycles, pristine, null));

            // The store-ful determinism question. A seed run first, because an incident store that restores
            // nothing cannot exercise the wall-clock fallback it is being tested for: `restoredAt ?? UtcNow`
            // only reaches the tracker when there is something saved to age.
            var seed = await RunArm("R warm + incident store (seed)", recentStart, 288, pristine, string.Empty);
            var seededIncidents = File.ReadAllText(seed.IncidentStatePath!);

            arms.Add(seed);
            arms.Add(await RunArm("R warm + incident store #1", recentStart, 288, pristine, seededIncidents));
            arms.Add(await RunArm("R warm + incident store #2", recentStart, 288, pristine, seededIncidents));
            arms.Add(await RunArm("R cold #2 (canary)", recentStart, 288, null, null));

            const int ColdOne = 0;
            const int WarmOne = 1;
            const int WarmTwo = 2;
            const int StoreOne = 8;
            const int StoreTwo = 9;
            const int ColdTwo = 10;

            report.Append(
                $"{"arm",34}{"cyc",5}{"done",6}{"blind",7}{"fail",6}{"find",6}{"open",6}{"resv",6}"
                + $"{"still",6}{"rows",6}{"rest",6}{"per100",8}\n");

            for (var i = 0; i < arms.Count; i++)
            {
                var a = arms[i];

                report.Append(CultureInfo.InvariantCulture,
                    $"{a.Name,34}{a.Cycles,5}{a.Completed,6}{a.Blind,7}{a.Failed,6}{a.Findings,6}"
                    + $"{a.Opened,6}{a.Resolved,6}{a.StillOpen,6}{a.SinkRows,6}{a.Restored,6}"
                    + $"{a.PerHundred,8:F1}\n");
            }

            report.Append(CultureInfo.InvariantCulture,
                $"\nthe live run, for comparison: {LiveOpened} opened and {LiveFindings} findings over "
                + $"{LiveCycles} cycles = {LiveOpened * 100.0 / LiveCycles:F1} per 100.\n");

            report.Append("\n=== did the warm arms actually load anything? ===\n");

            for (var i = 0; i < arms.Count; i++)
            {
                var a = arms[i];

                report.Append(CultureInfo.InvariantCulture,
                    $"{a.Name,34}  calibrator samples before cycle 1: {a.SamplesAtStart,6}   "
                    + $"restored incidents: {a.Restored}   state error: {a.StateError ?? "none"}\n");
            }

            report.Append("\n=== when did they open? (cycle index of each opened incident) ===\n");

            for (var i = 0; i < arms.Count; i++)
            {
                report.Append(CultureInfo.InvariantCulture,
                    $"{arms[i].Name,34}  {string.Join(" ", arms[i].OpenedAt)}\n");
            }

            _output.WriteLine(report.ToString());

            // Non-vacuity: an arm that saw nothing proves nothing about anything.
            for (var i = 0; i < arms.Count; i++)
            {
                Assert.True(
                    arms[i].Completed > 0,
                    $"arm '{arms[i].Name}' completed no cycle ({arms[i].Blind} blind, {arms[i].Failed} "
                    + "failed) — check the port-forward and that Prometheus still retains this range.");

                Assert.Equal(arms[i].WindowsReturned - 1, arms[i].WindowsAdvanced);
            }

            // The warm arms must differ from the cold ones in the one way that is checkable from outside:
            // a guard handed a calibration starts with samples in it, a cold one starts with none. Without
            // this the whole experiment could be two identical configurations wearing different labels.
            Assert.Equal(0, arms[ColdOne].SamplesAtStart);
            Assert.True(
                arms[WarmOne].SamplesAtStart > 0,
                "the warm arm started with an empty calibrator, so the learned state was not loaded and "
                + "every 'warm' number here is a second cold run under another name.");

            _output.WriteLine(
                $"canary       cold R #1 {Signature(arms[ColdOne])}\n"
                + $"canary       cold R #2 {Signature(arms[ColdTwo])}\n"
                + $"determinism  warm R #1 {Signature(arms[WarmOne])}\n"
                + $"determinism  warm R #2 {Signature(arms[WarmTwo])}\n"
                + $"determinism  warm+incidents #1 {Signature(arms[StoreOne])}\n"
                + $"determinism  warm+incidents #2 {Signature(arms[StoreTwo])}");

            // The canary. Same window, same configuration, no store, run at both ends of the experiment: if
            // these two disagree, the body of history moved underneath the experiment and no comparison
            // between the arms in between means anything.
            Assert.Equal(Signature(arms[ColdOne]), Signature(arms[ColdTwo]));

            // The property the replay was built for, on a store-ful configuration. Reported as a failure
            // rather than as a note: if learned state fixes the rate but costs reproducibility, the two
            // things the replay exists for are mutually exclusive and somebody has to decide which wins.
            Assert.Equal(Signature(arms[WarmOne]), Signature(arms[WarmTwo]));
            Assert.Equal(Signature(arms[StoreOne]), Signature(arms[StoreTwo]));

            await ProbeTheWallClockAsync(options, source, logger, topology);

            return;

            async Task<Arm> RunArm(
                string name, DateTimeOffset start, int cycles, string? learned, string? incidents)
            {
                var slot = (arms.Count + 1).ToString("00", CultureInfo.InvariantCulture);
                var learnedPath = Path.Combine(AppContext.BaseDirectory, $"exp-learned-{slot}.txt");
                var incidentPath = Path.Combine(AppContext.BaseDirectory, $"exp-incidents-{slot}.json");

                ILearnedStateStore? learnedStore = null;
                IIncidentStore? incidentStore = null;

                if (learned is not null)
                {
                    // A fresh copy per arm: the guard rewrites this file every cycle, so two arms sharing
                    // one would have the second start from whatever the first left behind.
                    File.WriteAllText(learnedPath, learned);
                    learnedStore = new FileLearnedStateStore(learnedPath);
                }

                if (incidents is not null)
                {
                    File.WriteAllText(incidentPath, incidents);
                    incidentStore = new FileIncidentStore(incidentPath);
                }

                var sink = new CountingSink();
                var recorder = new RecordingSource(source);

                using var service = new AnomalyGuardService(
                    options, recorder, sink, logger, topology, incidentStore, learnedStore);

                var arm = new Arm(name, cycles, incidents is null ? null : incidentPath)
                {
                    Restored = service.RestoredIncidents,
                    SamplesAtStart = MaxSamples(service.Guard.FloorProposals),
                };

                for (var cycle = 0; cycle < cycles; cycle++)
                {
                    var outcome = await service.RunCycleAsync(start + (cadence * cycle), CancellationToken.None);

                    if (!outcome.TryGetResult(out var result))
                    {
                        arm.Blind += outcome.Kind == GuardCycleKind.Blind ? 1 : 0;
                        arm.Failed += outcome.Kind == GuardCycleKind.Failed ? 1 : 0;

                        continue;
                    }

                    arm.Completed++;
                    arm.Findings += result.Findings;
                    arm.Opened += result.Opened;
                    arm.Resolved += result.Resolved;

                    for (var k = 0; k < result.Opened; k++)
                    {
                        arm.OpenedAt.Add(cycle);
                    }
                }

                arm.SinkRows = sink.Rows;
                arm.StillOpen = service.Guard.OpenIncidents;
                arm.StateError = service.Guard.StateError;
                arm.WindowsReturned = recorder.Starts.Count;
                arm.WindowsAdvanced = Advanced(recorder.Starts);

                return arm;
            }
        }

        /// <summary>
        /// Whether constructing a store-ful guard depends on the wall clock — measured directly, because the
        /// replay arms above could not answer it.
        ///
        /// <para><b>Those arms were vacuous for this question and the reason is worth stating.</b> They were
        /// seeded from a replay that ended with nothing open, so <c>Restore</c> had no record to age and
        /// <c>restoredAt ?? DateTimeOffset.UtcNow</c> was reached and then made no difference. Two identical
        /// signatures out of that is not evidence that a store-ful replay is reproducible; it is evidence
        /// that the clock was not consulted about anything.</para>
        ///
        /// <para>So this constructs the guard twice, six seconds apart, over one hand-written state holding
        /// two incidents: one comfortably inside <c>MaxRestoredIncidentAge</c> and one that crosses the
        /// boundary between the two constructions. No cycles are run — the question is entirely about what
        /// the constructor adopts.</para>
        /// </summary>
        private async Task ProbeTheWallClockAsync(
            AnomalyGuardServiceOptions options,
            IMetricWindowSource source,
            ILogger<AnomalyGuardService> logger,
            IRefreshablePodTopology topology)
        {
            var margin = TimeSpan.FromSeconds(5);
            var gap = TimeSpan.FromSeconds(6);
            var now = DateTimeOffset.UtcNow;

            var young = now - TimeSpan.FromHours(1);
            var borderline = now - options.Guard.MaxRestoredIncidentAge + margin;
            var path = Path.Combine(AppContext.BaseDirectory, "exp-clock-probe.json");

            File.WriteAllText(path, "overfit-incident-state\tv1\n3\n"
                                    + Incident(1, young) + Incident(2, borderline));

            var first = Adopted();

            await Task.Delay(gap);

            var second = Adopted();

            _output.WriteLine(
                $"\n=== does a store-ful guard read the wall clock? ===\n"
                + $"MaxRestoredIncidentAge {options.Guard.MaxRestoredIncidentAge}, two saved incidents last "
                + $"seen {young:u} and {borderline:u}\n"
                + $"adopted at construction: {first} then, {gap.TotalSeconds:F0} s later, {second}\n");

            Assert.Equal(2, first);
            Assert.Equal(1, second);

            int Adopted()
            {
                using var probe = new AnomalyGuardService(
                    options, source, new CountingSink(), logger, topology, new FileIncidentStore(path));

                return probe.RestoredIncidents;
            }

            static string Incident(long id, DateTimeOffset lastSeen)
            {
                var ticks = lastSeen.UtcTicks.ToString(CultureInfo.InvariantCulture);

                return string.Join('\t',
                    id.ToString(CultureInfo.InvariantCulture), ticks, ticks, "3", "0",
                    "probe-" + id.ToString(CultureInfo.InvariantCulture),
                    "probe-" + id.ToString(CultureInfo.InvariantCulture),
                    "lab", "lab-workload", "lab-workload-7765564ff6", "lab-workload-7765564ff6-probe",
                    "docker-desktop", "CpuUsageRatio", "0", "0.9", ticks, ticks, "1", "1",
                    "a hand-written record, so the age filter has something to filter") + "\n";
            }
        }

        /// <summary>Everything an arm decided, in one comparable string.</summary>
        private static string Signature(Arm a)
            => string.Create(CultureInfo.InvariantCulture,
                $"done={a.Completed} blind={a.Blind} fail={a.Failed} find={a.Findings} open={a.Opened} "
                + $"resv={a.Resolved} still={a.StillOpen} rows={a.SinkRows} restored={a.Restored}");

        private static int Advanced(List<DateTimeOffset> starts)
        {
            var walked = 0;

            for (var i = 1; i < starts.Count; i++)
            {
                walked += starts[i] > starts[i - 1] ? 1 : 0;
            }

            return walked;
        }

        private static int MaxSamples(FloorProposal[] proposals)
        {
            var max = 0;

            for (var i = 0; i < proposals.Length; i++)
            {
                max = Math.Max(max, proposals[i].Samples);
            }

            return max;
        }

        private static DateTimeOffset Snap(DateTimeOffset at, TimeSpan cadence)
            => new(at.UtcTicks - (at.UtcTicks % cadence.Ticks), TimeSpan.Zero);

        /// <summary>The learned payload with its seasonal baseline removed, calibration untouched.</summary>
        private static string WithoutHistory(string state)
        {
            var calibration = state.IndexOf(CalibrationMarker, StringComparison.Ordinal);

            return HistoryMarker + "\n" + state[calibration..];
        }

        /// <summary>The learned payload with its calibration removed, seasonal baseline untouched.</summary>
        private static string WithoutCalibration(string state)
        {
            var calibration = state.IndexOf(CalibrationMarker, StringComparison.Ordinal);
            var labels = state.IndexOf(LabelMarker, StringComparison.Ordinal);

            return state[..calibration] + CalibrationMarker + "\n" + state[labels..];
        }

        private static int Buckets(string state) => CountBuckets(state, 1);

        private static int BucketsWithAtLeast(string state, int days) => CountBuckets(state, days);

        /// <summary>
        /// How many (workload, metric, hour) buckets the payload holds at least <paramref name="days"/>
        /// distinct days for. Read straight off the text rather than through
        /// <c>MetricHistory</c>, which does not expose it — and the count is the thing that decides whether
        /// a seasonal expectation can be formed at all.
        /// </summary>
        private static int CountBuckets(string state, int days)
        {
            var calibration = state.IndexOf(CalibrationMarker, StringComparison.Ordinal);
            var history = calibration >= 0 ? state[..calibration] : state;
            var counts = new Dictionary<string, HashSet<int>>(StringComparer.Ordinal);

            foreach (var line in history.Split('\n'))
            {
                var fields = line.Split('\t');

                if (fields.Length != 5 || !int.TryParse(
                        fields[3], NumberStyles.Integer, CultureInfo.InvariantCulture, out var day))
                {
                    continue;
                }

                var key = fields[0] + "\t" + fields[1] + "\t" + fields[2];

                if (!counts.TryGetValue(key, out var seen))
                {
                    seen = [];
                    counts[key] = seen;
                }

                seen.Add(day);
            }

            var qualifying = 0;

            foreach (var (_, seen) in counts)
            {
                qualifying += seen.Count >= days ? 1 : 0;
            }

            return qualifying;
        }

        private static string ReadLearnedState()
        {
            var path = Environment.GetEnvironmentVariable("OVERFIT_LEARNED_STATE")
                       ?? RepositoryPaths.TestsBin("lab-learned-state.txt");

            Assert.True(File.Exists(path),
                $"no learned state at {path}. Pull the deployed guard's own out of the cluster — "
                + "kubectl -n lab exec deploy/anomaly-guard -- cat /state/learned-state.txt — or point "
                + "OVERFIT_LEARNED_STATE at one. Without it there is no warm arm and this experiment is "
                + "two cold runs.");

            return File.ReadAllText(path).Replace("\r\n", "\n", StringComparison.Ordinal);
        }

        private static AnomalyGuardConfigFile ReadConfiguration(string prometheus)
        {
            var path = Environment.GetEnvironmentVariable("OVERFIT_REPLAY_CONFIG")
                       ?? RepositoryPaths.TestsBin("lab-guard.json");

            Assert.True(File.Exists(path), $"no guard configuration at {path}.");

            var file = JsonSerializer.Deserialize<AnomalyGuardConfigFile>(
                File.ReadAllText(path),
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

            Assert.NotNull(file);

            file.Prometheus = prometheus;

            return file;
        }

        /// <summary>What one configuration decided over one body of history.</summary>
        private sealed class Arm
        {
            public Arm(string name, int cycles, string? incidentStatePath)
            {
                Name = name;
                Cycles = cycles;
                IncidentStatePath = incidentStatePath;
            }

            public string Name
            {
                get;
            }

            public int Cycles
            {
                get;
            }

            /// <summary>Where this arm's incident store ended up, so a later arm can be seeded from it.</summary>
            public string? IncidentStatePath
            {
                get;
            }

            public int Completed
            {
                get; set;
            }

            public int Blind
            {
                get; set;
            }

            public int Failed
            {
                get; set;
            }

            public int Findings
            {
                get; set;
            }

            public int Opened
            {
                get; set;
            }

            public int Resolved
            {
                get; set;
            }

            public int StillOpen
            {
                get; set;
            }

            public int SinkRows
            {
                get; set;
            }

            public int Restored
            {
                get; set;
            }

            /// <summary>Observations the calibrator already held before the first cycle. Zero when cold.</summary>
            public int SamplesAtStart
            {
                get; set;
            }

            public string? StateError
            {
                get; set;
            }

            public int WindowsReturned
            {
                get; set;
            }

            public int WindowsAdvanced
            {
                get; set;
            }

            public List<int> OpenedAt { get; } = [];

            public double PerHundred => Completed > 0 ? Opened * 100.0 / Completed : 0.0;
        }

        /// <summary>
        /// Records where each returned window starts. The only evidence, per arm, that history was walked
        /// rather than the present re-evaluated N times — every other number reads the same either way.
        /// </summary>
        private sealed class RecordingSource : IMetricWindowSource
        {
            private readonly IMetricWindowSource _inner;

            public RecordingSource(IMetricWindowSource inner) => _inner = inner;

            public IReadOnlyList<string> StalePodsExcluded => _inner.StalePodsExcluded;

            public List<DateTimeOffset> Starts { get; } = [];

            public async Task<MetricWindow?> ReadAsync(
                DateTimeOffset end, TimeSpan window, CancellationToken ct = default)
            {
                var result = await _inner.ReadAsync(end, window, ct);

                if (result is not null)
                {
                    Starts.Add(result.Start);
                }

                return result;
            }

            public void Dispose()
            {
                // The container owns the inner source and disposes it with the provider; closing its shared
                // HttpClient here would break every arm after this one.
            }
        }

        private sealed class CountingSink : IIncidentSink
        {
            public int Rows
            {
                get; private set;
            }

            public void Report(ReadOnlySpan<IncidentLogRecord> rows) => Rows += rows.Length;
        }
    }
}
