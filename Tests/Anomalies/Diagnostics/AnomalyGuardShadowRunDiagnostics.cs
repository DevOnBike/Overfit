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
using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;
using DevOnBike.Overfit.Statistics;
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
    /// <para><b>REQUIRES A PORT-FORWARD TO PROMETHEUS ON :9090.</b> Without it this fails with a bare
    /// <c>HttpRequestException: connection refused (127.0.0.1:9090)</c>, which names no cause and reads
    /// like a defect in the guard. It is not one — measured 2026-08-07, five of these lab diagnostics
    /// failed that way in the first release-gate run purely because nothing was forwarding.</para>
    /// <code>
    ///   k8s\monitoring\forward.cmd                                            (all three ports at once)
    ///   kubectl port-forward -n monitoring svc/overfit-lab-prometheus 9090:9090   (this one only)
    /// </code>
    /// <para>Verify it before believing a red result:
    /// <c>curl "http://127.0.0.1:9090/api/v1/query?query=up"</c>.</para>
    ///
    /// <para>Also needs the lab from <c>k8s/</c> and <b>traffic</b> — the RED signals sit at zero until
    /// something drives the server, and a degraded replica is invisible on an idle one. Knobs:
    /// <c>OVERFIT_LAB_PROMETHEUS</c>, <c>OVERFIT_SHADOW_CYCLES</c>,
    /// <c>OVERFIT_SHADOW_CADENCE_SECONDS</c>, <c>OVERFIT_SHADOW_WINDOW_MINUTES</c>.</para>
    /// </summary>
    public sealed class AnomalyGuardShadowRunDiagnostics
    {
        private readonly ITestOutputHelper _output;

        public AnomalyGuardShadowRunDiagnostics(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact]  // runtime unmeasured — the test failed after 4s (2026-08-07)
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

            // The path production takes: real ownership from kube-state-metrics rather than the pod-name
            // heuristic. It matters here — the lab's degraded replica is its own Deployment, and telling the
            // grouper otherwise merges it with the healthy three.
            using var topology = new PrometheusTopologySource(prometheus, config);
            var sink = new RecordingSink();

            var guard = new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "overfit",
                    Workload = "overfit-server",
                    PodTopology = topology,
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
                          + $"{"incidents",11}{"opened",8}{"ongoing",9}{"resolved",10}{"topo",7}\n");

            var totalOpened = 0;
            var evaluated = 0;

            for (var cycle = 1; cycle <= cycles; cycle++)
            {
                var now = DateTimeOffset.UtcNow;
                var resolvedPods = await topology.RefreshAsync();
                var read = await source.ReadAsync(now - endOffset, window);

                if (read is null)
                {
                    report.Append($"{cycle,6}   no series returned — check the port-forward and the regex\n");
                }
                else
                {
                    var traces = new List<IncidentMatchTrace>();
                    var peer = new List<PeerDecisionTrace>();
                    var result = guard.RunCycle(read, now, traces.Add, peer.Add);

                    evaluated++;
                    totalOpened += result.Opened;

                    report.Append($"{cycle,6}{read.Pods.Count,6}{result.BlindMetrics,7}{result.PartialMetrics,9}"
                                  + $"{result.Findings,10}{result.Incidents,11}{result.Opened,8}"
                                  + $"{result.Ongoing,9}{result.Resolved,10}{resolvedPods,7}\n");

                    // Every gate on the pod the lab deliberately degraded. "No finding" has five causes and
                    // they call for opposite fixes, so the numbers that decided are printed rather than the
                    // verdict.
                    for (var i = 0; i < peer.Count; i++)
                    {
                        if (!peer[i].Pod.Contains("degraded", StringComparison.Ordinal))
                        {
                            continue;
                        }

                        if (peer[i].Status == DetectionStatus.Healthy && !peer[i].IsOutlier)
                        {
                            continue;
                        }

                        report.Append($"      PEER {peer[i].Signal,-22} {peer[i].Status,-16} "
                                      + $"high={peer[i].High} low={peer[i].Low} "
                                      + $"outlier={(peer[i].IsOutlier ? "Y" : "n")} "
                                      + $"gap={peer[i].RelativeGap,7:P0} "
                                      + $"delta={peer[i].EffectSize,6:F2} "
                                      + $"p={peer[i].PValue:G3} "
                                      + $"n={peer[i].UsableSamples}")
                            .Append('\n');
                    }

                    // Only the decisions that were not "it continued" — a run where everything continues has
                    // nothing to explain, and printing it would bury the cycles that do.
                    for (var i = 0; i < traces.Count; i++)
                    {
                        if (traces[i].Outcome == IncidentMatchOutcome.Continued)
                        {
                            continue;
                        }

                        report.Append($"          {traces[i].Outcome,-16} {Short(traces[i].PrimaryKey),-8} "
                                      + $"subjects={traces[i].SubjectCount,-3} "
                                      + $"bestOverlap={traces[i].BestOverlap:F2} "
                                      + $"vs #{traces[i].BestOverlapId} "
                                      + $"({Short(traces[i].BestOverlapPrimaryKey)})\n");
                    }
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

        /// <summary>Last segment of a subject key, so a table stays readable.</summary>
        private static string Short(string key)
        {
            if (key.Length == 0)
            {
                return "(none)";
            }

            var slash = key.LastIndexOf('/');
            var tail = slash >= 0 ? key[(slash + 1)..] : key;

            return tail.Length <= 8 ? tail : tail[^8..];
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
