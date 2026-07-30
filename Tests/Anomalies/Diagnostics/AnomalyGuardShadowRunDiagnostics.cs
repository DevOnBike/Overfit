// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Incidents.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Anomalies.Diagnostics
{
    /// <summary>
    /// Runs the guard against the live cluster lab for several cycles — <b>shadow mode</b>: it counts,
    /// records and wakes nobody.
    ///
    /// <para><b>This is the first time the incident tracker sees real multi-cycle data.</b> Everything
    /// measured until now was either one window of the lab or a synthetic population; the lifecycle — opened
    /// once, ongoing, resolved — has only ever been exercised against fixtures. A cluster that is genuinely
    /// running, with a genuinely degraded replica, is the only thing that can say whether an incident stays
    /// one incident across cycles or fragments into a new one every time a finding moves.</para>
    ///
    /// <para><b>What to read out of a run.</b> `opened` is what would have paged somebody; anything above one
    /// per real problem is the tracker failing at its job. `blind` is the coverage channel — a non-zero count
    /// with zero incidents means the guard saw nothing because it was looking at nothing, which is the
    /// failure mode that looks exactly like health.</para>
    ///
    /// <para>Needs the lab from <c>k8s/</c>, a port-forward to Prometheus, and <b>traffic</b> — the RED
    /// signals sit at zero until something drives the server, and a degraded replica is invisible on an idle
    /// one. Knobs: <c>OVERFIT_LAB_PROMETHEUS</c>, <c>OVERFIT_SHADOW_CYCLES</c>,
    /// <c>OVERFIT_SHADOW_CADENCE_SECONDS</c>, <c>OVERFIT_SHADOW_WINDOW_MINUTES</c>.</para>
    /// </summary>
    public sealed class AnomalyGuardShadowRunDiagnostics
    {
        private readonly ITestOutputHelper _output;

        public AnomalyGuardShadowRunDiagnostics(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact]
        public async Task RunsInShadowAgainstTheLab()
        {
            var prometheus = Environment.GetEnvironmentVariable("OVERFIT_LAB_PROMETHEUS")
                             ?? "http://127.0.0.1:9090";
            var cycles = Setting("OVERFIT_SHADOW_CYCLES", 8);
            var cadence = TimeSpan.FromSeconds(Setting("OVERFIT_SHADOW_CADENCE_SECONDS", 60));
            var window = TimeSpan.FromMinutes(Setting("OVERFIT_SHADOW_WINDOW_MINUTES", 12));
            var endOffset = TimeSpan.FromMinutes(Setting("OVERFIT_SHADOW_END_OFFSET_MINUTES", 2));

            var config = PrometheusHistoricalSourceConfig.ForOverfitServer(
                prometheus,
                podRegex: "overfit-server-.*",
                namespaceName: "overfit",
                rangeStart: DateTime.UtcNow.AddMinutes(-window.TotalMinutes),
                rangeEnd: DateTime.UtcNow,
                step: TimeSpan.FromSeconds(15));

            using var source = new PrometheusMetricWindowSource(config);
            var sink = new RecordingSink();

            var guard = new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "overfit",
                    Workload = "overfit-server",
                    Grouping = IncidentGroupingOptions.Balanced with
                    {
                        // One node means SameNode is a constant, which would relate everything to everything.
                        Topology = TopologyWeights.SingleNode
                    },
                },
                sink,
                IncidentTrackingOptions.Balanced);

            var report = new StringBuilder();
            report.Append($"shadow run: {cycles} cycles, {cadence.TotalSeconds:F0}s cadence, "
                          + $"{window.TotalMinutes:F0} min window, {endOffset.TotalMinutes:F0} min end offset\n\n");
            report.Append($"{"cycle",6}{"pods",6}{"blind",7}{"partial",9}{"findings",10}"
                          + $"{"incidents",11}{"opened",8}{"ongoing",9}{"resolved",10}\n");

            var totalOpened = 0;
            var evaluated = 0;

            for (var cycle = 1; cycle <= cycles; cycle++)
            {
                var now = DateTimeOffset.UtcNow;
                var read = await source.ReadAsync(now - endOffset, window);

                if (read is null)
                {
                    report.Append($"{cycle,6}   no series returned — check the port-forward and the regex\n");
                }
                else
                {
                    var result = guard.RunCycle(read, now);

                    evaluated++;
                    totalOpened += result.Opened;

                    report.Append($"{cycle,6}{read.Pods.Count,6}{result.BlindMetrics,7}{result.PartialMetrics,9}"
                                  + $"{result.Findings,10}{result.Incidents,11}{result.Opened,8}"
                                  + $"{result.Ongoing,9}{result.Resolved,10}\n");
                }

                if (cycle < cycles)
                {
                    await Task.Delay(cadence, CancellationToken.None);
                }
            }

            report.Append($"\nevaluated {evaluated} of {cycles} cycles, {totalOpened} incident(s) opened, "
                          + $"{guard.OpenIncidents} still open\n");

            report.Append("\nincidents opened during the run — one line each is the point of the tracker\n");

            foreach (var line in sink.IncidentRows)
            {
                report.Append($"   {line}\n");
            }

            _output.WriteLine(report.ToString());

            Assert.True(evaluated > 0,
                $"no cycle produced a window from {prometheus} — is the port-forward up and does "
                + "'overfit-server-.*' match anything in namespace 'overfit'?");
        }

        private static int Setting(string name, int fallback)
        {
            var raw = Environment.GetEnvironmentVariable(name);

            return int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out var value)
                ? value
                : fallback;
        }

        /// <summary>
        /// Keeps every incident row so a run can be read after the fact. <b>One row per cycle the
        /// incident was observed in</b>, not one per notification — an eight-cycle run of one problem
        /// leaves eight rows here and opens exactly one incident.
        /// </summary>
        private sealed class RecordingSink : IIncidentSink
        {
            public List<string> IncidentRows { get; } = [];

            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
                for (var i = 0; i < rows.Length; i++)
                {
                    if (rows[i].Kind != IncidentLogRecordKind.Incident)
                    {
                        continue;
                    }

                    var where = rows[i].NamesAPod ? rows[i].Pod : $"{rows[i].Workload} (deployment)";

                    IncidentRows.Add(
                        $"[sev {rows[i].Severity:F2}] {where} — {rows[i].Signal}: {rows[i].Message}");
                }
            }
        }
    }
}
