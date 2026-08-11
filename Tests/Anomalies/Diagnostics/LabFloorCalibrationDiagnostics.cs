// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text;
using System.Text.Json;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Statistics;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.Anomalies.Diagnostics
{
    /// <summary>
    /// Runs <see cref="FloorCalibrator"/> against the live lab, from outside it, and prints both floors.
    ///
    /// <para><b>Read-only, and that is the point.</b> The deployed guard already calibrates, but it logs only
    /// the peer-gap floor, and the fix for that is a code change that would mean a third restart of a running
    /// twenty-four-hour measurement. The same numbers can be had by reading the same Prometheus with the same
    /// metric map — no restart, no lost hours, and the two paths sharing <see cref="FloorCalibrator"/> means
    /// this is the deployed calculation rather than a re-implementation of it.</para>
    ///
    /// <para><b>The trend floor is the one that matters here.</b> The lab's dominant false-positive source is
    /// the trend family, whose gate is <c>MinAbsoluteTrendChange</c>; the peer gate governs a smaller share.
    /// Both are printed, in the units the guard reads them in.</para>
    ///
    /// <para><b>REQUIRES A PORT-FORWARD TO PROMETHEUS ON :9098</b> — note the port: this one does not use
    /// 9090. Without the forward it fails with a bare <c>HttpRequestException: connection refused
    /// (127.0.0.1:9098)</c>, which names no cause and reads like a defect in <see cref="FloorCalibrator"/>.
    /// It is not one — measured 2026-08-07, five of these lab diagnostics failed that way in the first
    /// release-gate run purely because nothing was forwarding, and two of them, this one included, were
    /// unreachable even after following the documented <c>forward.cmd</c>, because that script forwarded
    /// only 9090. It now forwards all three.</para>
    /// <code>
    ///   k8s\monitoring\forward.cmd                                            (all three ports at once)
    ///   kubectl port-forward -n monitoring svc/overfit-lab-prometheus 9098:9090   (this one only)
    /// </code>
    /// <para>Verify it before believing a red result:
    /// <c>curl "http://127.0.0.1:9098/api/v1/query?query=up"</c>. Knobs: <c>OVERFIT_LAB_PROMETHEUS</c>,
    /// <c>OVERFIT_LAB_CONFIG</c> (path to the same guard.json the cluster runs),
    /// <c>OVERFIT_LAB_WINDOWS</c>, <c>OVERFIT_LAB_WINDOW_MINUTES</c>, <c>OVERFIT_LAB_STEP_MINUTES</c>.</para>
    /// </summary>
    public sealed class LabFloorCalibrationDiagnostics
    {
        private readonly ITestOutputHelper _output;

        public LabFloorCalibrationDiagnostics(ITestOutputHelper output)
        {
            _output = output;
        }

        [LabFact(LabEndpoint.Prometheus)]
        public async Task ProposesFloorsFromTheLiveLab()
        {
            var prometheus = Environment.GetEnvironmentVariable("OVERFIT_LAB_PROMETHEUS")
                             ?? "http://127.0.0.1:9098";
            var configPath = Environment.GetEnvironmentVariable("OVERFIT_LAB_CONFIG")
                             ?? RepositoryPaths.TestsBin("lab-guard.json");
            var windows = Setting("OVERFIT_LAB_WINDOWS", 24);
            var windowMinutes = Setting("OVERFIT_LAB_WINDOW_MINUTES", 20);
            var stepMinutes = Setting("OVERFIT_LAB_STEP_MINUTES", 5);

            Assert.True(File.Exists(configPath), $"no guard config at {configPath}");

            var file = JsonSerializer.Deserialize<AnomalyGuardConfigFile>(
                await File.ReadAllTextAsync(configPath),
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

            Assert.NotNull(file);

            var map = AnomalyGuardConfigReader.ReadMap(file, out var problems);
            var (gap, trendChange, _) = AnomalyGuardConfigReader.ReadThresholds(file, out var floorProblems);

            var report = new StringBuilder();

            report.Append($"prometheus {prometheus}\n");
            report.Append($"namespace {file.Namespace}, pods /{file.PodRegex}/\n");
            report.Append($"{windows} windows of {windowMinutes} min, {stepMinutes} min apart "
                          + $"({windows * stepMinutes / 60.0:F1} h of history)\n");

            foreach (var problem in problems.Concat(floorProblems))
            {
                report.Append($"   CONFIG PROBLEM: {problem}\n");
            }

            var config = PrometheusHistoricalSourceConfig.ForOverfitServer(
                prometheus,
                podRegex: file.PodRegex,
                namespaceName: file.Namespace,
                rangeStart: DateTime.UtcNow.AddMinutes(-windowMinutes),
                rangeEnd: DateTime.UtcNow,
                step: TimeSpan.FromSeconds(15)) with
            {
                QueryOverrides = map.ToQueryOverrides(),
            };

            using var source = new PrometheusMetricWindowSource(
                config, httpClient: null, customQueries: map.CustomQueries());
            var calibrator = new FloorCalibrator();
            var window = TimeSpan.FromMinutes(windowMinutes);
            var read = 0;

            // Walk backwards from now, so the sampled history is the recent one — the same stretch the
            // deployed guard has been judging.
            for (var i = 0; i < windows; i++)
            {
                var end = DateTimeOffset.UtcNow.AddMinutes(-2 - (i * stepMinutes));
                var slice = await source.ReadAsync(end, window, TestContext.Current.CancellationToken);

                if (slice is null)
                {
                    continue;
                }

                calibrator.Observe(slice);
                read++;
            }

            report.Append($"\nwindows read: {read} of {windows}\n\n");

            Assert.True(read > 0, $"no window came back from {prometheus} — is the port-forward up?");

            var proposals = calibrator.Propose();

            report.Append($"   {"metric",-24}{"typical",13}{"PEER gap floor",18}{"configured",13}"
                          + $"{"TREND floor",16}{"configured",13}{"n",6}\n");

            for (var m = 0; m < proposals.Length; m++)
            {
                var proposal = proposals[m];

                if (proposal.Samples == 0)
                {
                    continue;
                }

                var metric = (MetricIndex)m;
                var configuredGap = AnomalyGuardOptions.FloorFor(gap, metric);
                var configuredTrend = AnomalyGuardOptions.FloorFor(trendChange, metric);

                report.Append($"   {metric,-24}{proposal.TypicalMagnitude,13:G4}"
                              + $"{Show(proposal.ProposedMinAbsoluteGap),18}{Show(configuredGap),13}"
                              + $"{Show(proposal.ProposedMinAbsoluteTrendChange),16}{Show(configuredTrend),13}"
                              + $"{proposal.Samples,6}{(proposal.IsUsable ? "" : "  (thin)")}\n");
            }

            report.Append("\nPEER gap floor   MinAbsoluteGap — how far two replicas may sit apart\n");
            report.Append("TREND floor      MinAbsoluteTrendChange — how far one may move across a window\n");
            report.Append("configured       what the cluster is running with today; 0 means the gate is off\n");
            report.Append("\nA proposal is what a HEALTHY period did, with a margin. It is a noise floor, not\n");
            report.Append("an operational threshold: it says what to ignore, never what is worth waking for.\n");
            report.Append("Valid only if this period really was healthy — a fault inside it raises the bar\n");
            report.Append("above itself and the guard goes permanently blind to that fault at that size.\n");

            _output.WriteLine(report.ToString());
        }

        private static string Show(double value)
        {
            return value <= 0.0 ? "-" : value.ToString("G4", CultureInfo.InvariantCulture);
        }

        private static int Setting(string name, int fallback)
        {
            var raw = Environment.GetEnvironmentVariable(name);

            return int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out var value)
                ? value
                : fallback;
        }
    }
}
