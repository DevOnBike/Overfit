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
using DevOnBike.Overfit.Statistics;
using DevOnBike.Overfit.Tests.TestSupport;
using Microsoft.Extensions.DependencyInjection;

namespace DevOnBike.Overfit.Tests.Anomalies.Diagnostics
{
    /// <summary>
    /// Spike 2 of <c>docs/specs/an-d1-peer-novelty-plan.md</c>: what does the <b>change</b> in a pod's peer gap
    /// do on a healthy population, in the metric's own units?
    ///
    /// <para><b>The unit is the whole question, so it is stated before the measurement.</b>
    /// <c>PeerNoveltyOptions</c>'s change floor reaches <see cref="TrendDetector"/> as
    /// <see cref="TrendOptions.MinAbsoluteChangeOverWindow"/>, and the quantity it is compared against is
    /// <c>Math.Abs(slope) * windowSeconds</c> — the Theil-Sen fit extrapolated across the span of the
    /// <i>retained</i> gap ring, not across one cycle and not across the guard's 20-minute metric window. At
    /// the shipped profiles (<c>RetainedCyclesPerSeries</c> 96, five-minute cadence) that span is <b>eight
    /// hours</b>. A number fitted over a shorter stretch answers a different question, and the direction of
    /// the error depends on whether the gap drifts or merely wobbles — so the span this run covered is printed
    /// beside every figure.</para>
    ///
    /// <para><b>Nothing here re-implements the arithmetic.</b> The gaps come out of the guard's own
    /// <see cref="PeerDecisionTrace"/>, produced by the same <c>RunPeer</c> the cluster runs, on windows read
    /// by the same <c>PrometheusMetricWindowSource</c> and configured by the same <c>guard.json</c>. The fit is
    /// <see cref="TrendDetector"/> itself with the options <c>PeerNoveltyTracker.Classify</c> builds. A
    /// diagnostic that recomputed any of it would be measuring the reproduction.</para>
    ///
    /// <para><b>Pods are compared only where they coexist</b>, because that is what the peer detector does:
    /// one cycle's comparison is inside one <see cref="MetricWindow"/>, whose members are the pods present at
    /// those timestamps. Cycles whose pod set differs from the first are counted and reported rather than
    /// folded in — a window spanning a rollout ranks replicas that never ran together.</para>
    ///
    /// <para><b>The window has to be clean, and this cannot check that for you.</b> A floor fitted over an
    /// injected fault raises the bar above the fault and blinds the guard to it at that size — the same rule
    /// <c>LabWindowValidator</c> enforces on recorded fixtures. Screen the range against the fault indicators
    /// <c>k8s/anomaly-guard/guard.lab-workload.json</c> documents healthy values for (CpuPressure,
    /// LockContentions, Exceptions, ActiveRequests) before believing anything printed below.</para>
    ///
    /// <para><b>REQUIRES A PORT-FORWARD TO PROMETHEUS</b> (<c>k8s\monitoring\forward.cmd</c>) and the deployed
    /// configuration at <c>Tests\bin\lab-guard.json</c>. Knobs: <c>OVERFIT_LAB_PROMETHEUS</c>,
    /// <c>OVERFIT_GAPCHANGE_CONFIG</c>, <c>OVERFIT_GAPCHANGE_START_UTC</c> (<b>required</b>, for the reason
    /// <c>AnomalyGuardReplayDiagnostics.ResolveReplayStart</c> records), <c>OVERFIT_GAPCHANGE_CYCLES</c>,
    /// <c>OVERFIT_GAPCHANGE_CADENCE_SECONDS</c>.</para>
    /// </summary>
    public sealed class PeerGapChangeCalibrationDiagnostics
    {
        /// <summary>
        /// Cycle floor for the fit, matching <c>PeerNoveltyOptions.PerShift/Daily/Weekly</c>. Below it
        /// <see cref="TrendDetector"/> answers <c>WarmingUp</c>, which is a refusal rather than a number.
        /// </summary>
        private const int MinimumCycles = 12;

        /// <summary>The margin every other floor in this subsystem carries over its measured healthy peak.</summary>
        private const double Margin = 1.25;

        private readonly ITestOutputHelper _output;

        public PeerGapChangeCalibrationDiagnostics(ITestOutputHelper output)
        {
            _output = output;
        }

        [LabFact(LabEndpoint.Prometheus)]
        public async Task MeasuresTheChangeInPeerGapOverAHealthyPopulation()
        {
            var prometheus = Environment.GetEnvironmentVariable("OVERFIT_LAB_PROMETHEUS")
                             ?? "http://127.0.0.1:9090";
            var cycles = Setting("OVERFIT_GAPCHANGE_CYCLES", 96);
            var cadence = TimeSpan.FromSeconds(Setting("OVERFIT_GAPCHANGE_CADENCE_SECONDS", 300));
            var first = ResolveStart(Environment.GetEnvironmentVariable("OVERFIT_GAPCHANGE_START_UTC"));

            var file = ReadConfiguration(prometheus);
            var problems = new List<string>();
            var sink = new NullSink();

            var services = new ServiceCollection();
            services.AddLogging();
            services.AddSingleton<IIncidentSink>(sink);
            services.AddOverfitAnomalyGuard(
                file, new AnomalyGuardServiceOptions { Cadence = cadence }, problems.Add);

            await using var provider = services.BuildServiceProvider();

            var source = provider.GetRequiredService<IMetricWindowSource>();
            var options = provider.GetRequiredService<AnomalyGuardServiceOptions>();
            var guard = new AnomalyGuard(options.Guard, sink, options.Tracking);

            var series = new Dictionary<string, GapSeries>(StringComparer.Ordinal);
            var report = new StringBuilder();

            report.Append(CultureInfo.InvariantCulture,
                $"prometheus {file.Prometheus}  ns={file.Namespace}  pods=/{file.PodRegex}/\n");
            report.Append(CultureInfo.InvariantCulture,
                $"{cycles} cycles at {cadence.TotalSeconds:F0}s from {first:u} to "
                + $"{first + (cadence * (cycles - 1)):u} ({cycles * cadence.TotalHours:F2} h)\n");
            report.Append(CultureInfo.InvariantCulture,
                $"window {options.Window.TotalMinutes:F0} min, end offset {options.EndOffset.TotalMinutes:F0} min\n");

            for (var i = 0; i < problems.Count; i++)
            {
                report.Append("config:  ").Append(problems[i]).Append('\n');
            }

            var read = 0;
            var blind = 0;
            var roster = string.Empty;
            var rosterChanges = 0;

            for (var cycle = 0; cycle < cycles; cycle++)
            {
                var now = first + (cadence * cycle);
                var window = await source.ReadAsync(now - options.EndOffset, options.Window);

                if (window is null)
                {
                    blind++;

                    continue;
                }

                read++;

                // The coexistence condition, checked rather than assumed: a cycle whose members differ from
                // the first is a different comparison, and folding its gaps into the same series would rank
                // replicas that never ran together.
                var members = string.Join('|', window.Pods.OrderBy(p => p, StringComparer.Ordinal));

                if (roster.Length == 0)
                {
                    roster = members;
                }

                if (!string.Equals(roster, members, StringComparison.Ordinal))
                {
                    rosterChanges++;

                    continue;
                }

                var at = now.ToUnixTimeMilliseconds() / 1000.0;

                guard.RunCycle(window, now, null, trace => Fold(series, trace, at));
            }

            report.Append(CultureInfo.InvariantCulture,
                $"\ncycles: {read} read, {blind} blind, {rosterChanges} skipped for a changed pod set\n");
            report.Append(CultureInfo.InvariantCulture,
                $"pods in the comparison: {(roster.Length == 0 ? 0 : roster.Split('|').Length)}\n\n");

            Assert.True(read > 0, $"no window came back from {file.Prometheus} — is the forward up, and does "
                                  + "Prometheus still retain this range?");

            report.Append(Describe(series));

            _output.WriteLine(report.ToString());

            // Non-vacuity. A run that folded nothing prints an empty table and reads exactly like a population
            // whose gaps never move, which is the failure mode this whole subsystem is built around.
            Assert.True(series.Count > 0,
                "no peer comparison produced a gap. The cycles above returned windows, so this is a "
                + "configuration or cohort problem, not a missing forward.");
        }

        /// <summary>Adds one member's gap for one channel to its own series.</summary>
        private static void Fold(
            Dictionary<string, GapSeries> series, in PeerDecisionTrace trace, double at)
        {
            if (!double.IsFinite(trace.AbsoluteGap))
            {
                return;
            }

            var key = trace.Signal + "\t" + trace.Pod;

            if (!series.TryGetValue(key, out var found))
            {
                found = new GapSeries(trace.Signal, trace.Pod);
                series[key] = found;
            }

            found.Times.Add(at);
            found.Gaps.Add(trace.AbsoluteGap);
            found.OutlierCycles += trace.IsOutlier ? 1 : 0;
        }

        /// <summary>
        /// Per channel: what the fit says every pod's gap moved across the span, and what a floor at
        /// <see cref="Margin"/> times the largest of them would be.
        /// </summary>
        private static string Describe(Dictionary<string, GapSeries> series)
        {
            var text = new StringBuilder();
            var byChannel = new Dictionary<string, List<Fit>>(StringComparer.Ordinal);
            var detector = new TrendDetector();

            // The options PeerNoveltyTracker.Classify builds, with the absolute gate open — the point is to
            // read the fitted change, not to have it suppressed by the value being calibrated.
            var options = new TrendOptions(
                TrendOptions.Balanced.MaxPValue,
                TrendOptions.Balanced.MinTau,
                TrendOptions.Balanced.MinRelativeChangeOverWindow,
                MinimumCycles,
                0.0);

            foreach (var (_, gaps) in series)
            {
                if (gaps.Times.Count < MinimumCycles)
                {
                    continue;
                }

                var times = gaps.Times.ToArray();
                var values = gaps.Gaps.ToArray();
                var result = detector.Detect(values, times, options);
                var span = times[times.Length - 1] - times[0];

                byChannel.TryAdd(gaps.Signal, []);
                byChannel[gaps.Signal].Add(new Fit(
                    gaps.Pod,
                    Math.Abs(result.ProjectedChangeOver(span)),
                    span,
                    result.SampleCount,
                    gaps.OutlierCycles,
                    result.Status,
                    result.Direction));
            }

            foreach (var channel in byChannel.Keys.OrderBy(k => k, StringComparer.Ordinal))
            {
                var fits = byChannel[channel];

                fits.Sort((a, b) => b.FittedChange.CompareTo(a.FittedChange));

                text.Append(CultureInfo.InvariantCulture,
                    $"=== {channel}   {fits.Count} pod(s), span {fits[0].SpanSeconds / 3600.0:F2} h, "
                    + $"{fits[0].Samples} samples\n");
                text.Append($"   {"pod",-36}{"fitted change",16}{"status",16}{"dir",9}{"outlier cyc",12}\n");

                for (var i = 0; i < fits.Count; i++)
                {
                    var fit = fits[i];

                    text.Append(CultureInfo.InvariantCulture,
                        $"   {fit.Pod,-36}{fit.FittedChange,16:G6}{fit.Status,16}{fit.Direction,9}"
                        + $"{fit.OutlierCycles,12}\n");
                }

                var max = fits[0].FittedChange;

                text.Append(CultureInfo.InvariantCulture,
                    $"   max {max:G6}  x{Margin} = {max * Margin:G6}   <-- candidate MinAbsoluteGapChange, "
                    + $"VALID ONLY FOR A {fits[0].SpanSeconds / 3600.0:F2} h RETAINED WINDOW\n\n");
            }

            if (byChannel.Count == 0)
            {
                text.Append($"no channel reached {MinimumCycles} cycles of gap history.\n");
            }

            return text.ToString();
        }

        /// <summary>
        /// The anchor, required for the reason <c>AnomalyGuardReplayDiagnostics.ResolveReplayStart</c> records
        /// in full: an unpinned range makes a calibration a measurement of when somebody pressed enter.
        /// </summary>
        private static DateTimeOffset ResolveStart(string? configured)
        {
            Assert.False(string.IsNullOrWhiteSpace(configured),
                "OVERFIT_GAPCHANGE_START_UTC is not set. A floor fitted over a range chosen by the wall clock "
                + "cannot be reproduced or challenged, and this one has to be defended against a window "
                + "containing no injected fault. Pass an explicit ISO-8601 instant.");

            Assert.True(
                DateTimeOffset.TryParse(
                    configured, CultureInfo.InvariantCulture,
                    DateTimeStyles.AdjustToUniversal | DateTimeStyles.AssumeUniversal, out var parsed),
                $"OVERFIT_GAPCHANGE_START_UTC is '{configured}', which is not a timestamp. It is refused "
                + "rather than replaced, because a fallback here produces an unpinned run that looks pinned.");

            return parsed;
        }

        private static AnomalyGuardConfigFile ReadConfiguration(string prometheus)
        {
            var path = Environment.GetEnvironmentVariable("OVERFIT_GAPCHANGE_CONFIG")
                       ?? RepositoryPaths.TestsBin("lab-guard.json");

            Assert.True(File.Exists(path),
                $"no guard configuration at {path}. Pull the deployed one out of the cluster — "
                + "kubectl -n lab get configmap anomaly-guard-config -o jsonpath=\"{.data.guard\\.json}\" — "
                + "or point OVERFIT_GAPCHANGE_CONFIG at one. The built-in query templates match nothing in "
                + "this lab and would calibrate a floor over a day of blind cycles.");

            var file = JsonSerializer.Deserialize<AnomalyGuardConfigFile>(
                File.ReadAllText(path),
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

            Assert.NotNull(file);

            file.Prometheus = prometheus;

            return file;
        }

        private static int Setting(string name, int fallback)
        {
            var raw = Environment.GetEnvironmentVariable(name);

            return int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out var value)
                ? value
                : fallback;
        }

        /// <summary>One (channel, pod)'s gap over time, in cycle order.</summary>
        private sealed class GapSeries
        {
            public GapSeries(string signal, string pod)
            {
                Signal = signal;
                Pod = pod;
            }

            public string Signal
            {
                get;
            }

            public string Pod
            {
                get;
            }

            public List<double> Times { get; } = [];

            public List<double> Gaps { get; } = [];

            /// <summary>
            /// Cycles in which this pod actually cleared the material gate. Printed because the tracker only
            /// folds outliers, so a pod with zero of these never reaches the mechanism at all and its fit
            /// describes the ranking's noise rather than a deviation's drift.
            /// </summary>
            public int OutlierCycles
            {
                get; set;
            }
        }

        /// <summary>What the fit made of one pod's gap series.</summary>
        private readonly record struct Fit(
            string Pod,
            double FittedChange,
            double SpanSeconds,
            int Samples,
            int OutlierCycles,
            DetectionStatus Status,
            TrendDirection Direction);

        /// <summary>Rows are not the subject here; the traces are.</summary>
        private sealed class NullSink : IIncidentSink
        {
            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
            }
        }
    }
}
