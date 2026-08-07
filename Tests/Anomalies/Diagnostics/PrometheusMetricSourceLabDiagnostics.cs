// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Anomalies.Diagnostics
{
    /// <summary>
    /// Points <see cref="PrometheusMetricSource"/> at the cluster lab and reports what each of the twelve
    /// features actually resolves to.
    ///
    /// <para><b>REQUIRES A PORT-FORWARD TO PROMETHEUS ON :9099</b>, plus the lab from <c>k8s/</c>. Without
    /// the forward this fails with a bare <c>HttpRequestException: connection refused (127.0.0.1:9099)</c>
    /// — a message that names no cause and reads like a defect in the source under test. It is not one:
    /// measured 2026-08-07, five lab diagnostics failed exactly that way in the first release-gate run,
    /// solely because nothing was forwarding.</para>
    /// <code>
    ///   k8s\monitoring\forward.cmd                                            (all three ports at once)
    ///   kubectl port-forward -n monitoring svc/overfit-lab-prometheus 9099:9090   (this one only)
    /// </code>
    /// <para>Override the endpoint with <c>OVERFIT_LAB_PROMETHEUS</c>. Verify the forward before believing
    /// a red result: <c>curl "http://127.0.0.1:9099/api/v1/query?query=up"</c>.</para>
    ///
    /// <para><b>What this is for.</b> Feature assembly cannot tell a query that matched nothing from a metric
    /// that reads zero, so a wrong metric name produces a flat, plausible, entirely fictional column. The only
    /// defence is looking — which is what this does, per metric and per pod, before anything is trained on it.</para>
    /// </summary>
    public sealed class PrometheusMetricSourceLabDiagnostics
    {
        private readonly ITestOutputHelper _output;

        public PrometheusMetricSourceLabDiagnostics(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact]  // runtime unmeasured — the test failed after 2s (2026-08-07)
        public async Task ReportsCoverageAgainstTheLab()
        {
            var baseUrl = Environment.GetEnvironmentVariable("OVERFIT_LAB_PROMETHEUS")
                          ?? "http://127.0.0.1:9099";

            var config = PrometheusMetricSourceConfig.ForOverfitServer(
                baseUrl,
                podRegex: "overfit-server-.*",
                namespaceName: "overfit") with
            {
                // The diagnostic should answer immediately rather than sit through a production scrape gap.
                ScrapeInterval = TimeSpan.FromMilliseconds(50)
            };

            using var source = new PrometheusMetricSource(config);

            var series = await source.ReadAsync();

            var report = new StringBuilder();
            report.Append("metric".PadRight(24)).Append("mapped  series\n");

            var missing = 0;
            for (var m = 0; m < (int)MetricIndex.Count; m++)
            {
                var metric = (MetricIndex)m;
                var mapped = source.IsMapped(metric);
                var count = source.SeriesReturned(metric);

                if (mapped && count == 0)
                {
                    missing++;
                }

                report.Append(metric.ToString().PadRight(24))
                      .Append(mapped ? "yes     " : "NO      ")
                      .Append(count)
                      .Append(count == 0 && mapped ? "   <-- query matched nothing" : string.Empty)
                      .Append('\n');
            }

            report.Append("\ntotal series: ").Append(series.Count).Append('\n');
            _output.WriteLine(report.ToString());

            // The cAdvisor-backed features need no application instrumentation at all, so if even those are
            // empty the problem is the connection or the selector, not the exporter — and the rest of the
            // report is not worth reading.
            Assert.True(
                source.SeriesReturned(MetricIndex.MemoryWorkingSetBytes) > 0,
                "no container_memory_working_set_bytes series: check the port-forward, namespace and pod regex "
                + $"before believing anything else in this report.\n{report}");

            _output.WriteLine(missing == 0
                ? "All mapped metrics returned data."
                : $"{missing} mapped metric(s) returned nothing — see the marked rows.");
        }
    }
}
