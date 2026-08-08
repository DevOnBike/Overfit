// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// A configured binding overrides the built-in template, so fixing a template is not enough to fix a
    /// deployment. Measured 2026-08-08: <c>OomEventsRate</c> was corrected in <see cref="PromqlCatalog"/> and
    /// the lab kept reading the dead series, because its config named one. These pin the passthrough that
    /// closes that gap and the guard rail that stops it becoming a hole.
    /// </summary>
    public sealed class VerbatimQueryBindingTests
    {
        private const string Join =
            "sum by (pod) (increase(kube_pod_container_status_restarts_total{%selector%}[2m])"
            + " * on (namespace, pod, container) group_left()"
            + " kube_pod_container_status_last_terminated_reason{%selector%,reason=\"OOMKilled\"})";

        [Fact]
        public void AVerbatimQueryIsUsedInsteadOfTheNameAndKind()
        {
            var map = new MetricMap([
                new MetricBinding(MetricIndex.OomEventsRate, "ignored_name",
                    MetricSourceKind.EventCount, 0.0, Join)
            ]);

            var query = map.ToQueryOverrides()[MetricIndex.OomEventsRate];

            Assert.Equal(Join, query);
            Assert.DoesNotContain("ignored_name", query, StringComparison.Ordinal);
        }

        [Fact]
        public void WithoutAVerbatimQueryTheNameAndKindStillDecide()
        {
            // The control: if this were not here, a bug that always took the verbatim branch would pass the
            // test above and break every other channel.
            var map = new MetricMap([
                new MetricBinding(MetricIndex.OomEventsRate, "some_counter_total",
                    MetricSourceKind.EventCount)
            ]);

            var query = map.ToQueryOverrides()[MetricIndex.OomEventsRate];

            Assert.Contains("increase(some_counter_total", query, StringComparison.Ordinal);
        }

        [Fact]
        public void AQueryWithoutTheSelectorTokenIsRejected()
        {
            // Without the token the query ignores the namespace and pod matchers and reports on the whole
            // cluster — quietly, and looking entirely healthy.
            var file = new AnomalyGuardConfigFile();

            file.Metrics["OomEventsRate"] = new AnomalyGuardConfigFile.MetricEntry
            {
                Source = "kube_pod_container_status_restarts_total",
                Kind = nameof(MetricSourceKind.EventCount),
                Query = "sum by (pod) (increase(kube_pod_container_status_restarts_total[2m]))"
            };

            AnomalyGuardConfigReader.ReadMap(file, out var problems);

            Assert.Contains(problems, p => p.Contains("%selector%", StringComparison.Ordinal));
        }

        [Fact]
        public void TheShippedLabConfigsBindTheOomChannelToKubeStateMetrics()
        {
            // The measurement said container_oom_events_total is zero on every series this runtime produces.
            // This is what stops it coming back in a file nobody re-reads.
            foreach (var name in new[] { "guard.lab-workload.json", "guard.lab.json" })
            {
                var path = RepositoryPaths.FromRoot("k8s", "anomaly-guard", name);

                Assert.True(File.Exists(path), $"missing {path}");

                var file = JsonSerializer.Deserialize<AnomalyGuardConfigFile>(
                    File.ReadAllText(path),
                    new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

                Assert.NotNull(file);

                var entry = file.Metrics["OomEventsRate"];

                Assert.Contains("kube_pod_container_status_last_terminated_reason", entry.Query,
                    StringComparison.Ordinal);
                Assert.Contains("%selector%", entry.Query, StringComparison.Ordinal);

                // The whole file has to survive the reader, not just this entry — a query that parses as
                // JSON and is then rejected would leave the channel unbound and blind.
                AnomalyGuardConfigReader.ReadMap(file, out var problems);

                Assert.Empty(problems);
            }
        }
    }
}
