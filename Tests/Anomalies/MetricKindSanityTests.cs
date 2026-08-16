// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using System.Text.RegularExpressions;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// A `kind` that does not match the shape of the series it names produces a query that is
    /// syntactically perfect and semantically wrong — and nothing else in this system notices.
    ///
    /// <para><b>Both directions have now shipped, a day apart.</b> `OomEventsRate` was pinned to
    /// `container_oom_events_total`, a series this runtime leaves permanently at zero, so the channel could
    /// never fire. `CpuThrottleRatio` was pinned to `container_cpu_cfs_throttled_periods_total` with
    /// `kind: Ratio` — which <see cref="MetricSourceKind.Ratio"/> defines as "already a fraction of one,
    /// passed through untouched" — so the guard queried a raw cumulative counter, the trend family saw a
    /// perfect climb in every window, and the channel became the largest single source of findings: **12 of
    /// 22 rows in a replay window where the true ratio was 0.0000 on all twelve pods**.</para>
    ///
    /// <para><b>What is checkable here, and what is not.</b> Whether a series is really a ratio cannot be
    /// known from a config file. What can: <c>_total</c> is the Prometheus convention for a cumulative
    /// counter, and a counter declared `Gauge` or `Ratio` is summed or passed through — which compares how
    /// long processes have been up rather than what they are doing. That is the exact mistake that shipped,
    /// and it costs one string comparison to refuse.</para>
    ///
    /// <para>An entry carrying an explicit <c>query</c> is exempt: writing the PromQL by hand is a
    /// deliberate act, and the reviewer of that line can see what it computes.</para>
    /// </summary>
    public sealed class MetricKindSanityTests
    {
        /// <summary>Kinds that use the series as-is. A cumulative counter under either is wrong.</summary>
        private static readonly string[] UntouchedKinds = ["Gauge", "Ratio"];

        [Fact]
        public void NoShippedConfigDeclaresACumulativeCounterAsAGaugeOrRatio()
        {
            var problems = new List<string>();

            foreach (var (label, document) in ShippedConfigs())
            {
                foreach (var section in new[] { "metrics", "customMetrics" })
                {
                    foreach (var problem in Suspect(document, section))
                    {
                        problems.Add($"{label}: {problem}");
                    }
                }
            }

            Assert.True(
                problems.Count == 0,
                "a `_total` series is a cumulative counter; declared Gauge or Ratio it is summed or passed "
                + "through, which compares process uptimes rather than values. Give it Counter or "
                + "EventCount, or an explicit `query` if the correct PromQL is a join or a division:\n  "
                + string.Join("\n  ", problems));
        }

        /// <summary>
        /// The exact entry that shipped and was wrong, as a fixture — so this test is known to catch the
        /// thing it was written for rather than merely passing today.
        /// </summary>
        [Fact]
        public void TheEntryThatActuallyShippedIsRejected()
        {
            var broken = JsonDocument.Parse(
                """
                {
                  "metrics": {
                    "CpuThrottleRatio": {
                      "source": "container_cpu_cfs_throttled_periods_total",
                      "kind": "Ratio"
                    }
                  }
                }
                """).RootElement;

            var problems = Suspect(broken, "metrics").ToList();

            Assert.True(problems.Count == 1, $"expected exactly one problem, got {problems.Count}");
            Assert.Contains("CpuThrottleRatio", problems[0], StringComparison.Ordinal);
        }

        [Fact]
        public void AnExplicitQueryIsExemptAndACounterIsFine()
        {
            // The control, and it is what stops this rule from being "reject every _total". Both of these
            // are correct and must pass: a cumulative counter declared Counter, and the same wrong kind
            // rescued by hand-written PromQL — which is how the real defect was fixed.
            var fine = JsonDocument.Parse(
                """
                {
                  "metrics": {
                    "CpuUsageRatio": {
                      "source": "container_cpu_usage_seconds_total",
                      "kind": "Counter"
                    },
                    "CpuThrottleRatio": {
                      "source": "container_cpu_cfs_throttled_periods_total",
                      "kind": "Ratio",
                      "query": "sum by (pod) (rate(a{%selector%}[2m])) / sum by (pod) (rate(b{%selector%}[2m]))"
                    }
                  }
                }
                """).RootElement;

            Assert.Empty(Suspect(fine, "metrics"));
        }

        private static IEnumerable<string> Suspect(JsonElement document, string section)
        {
            if (!document.TryGetProperty(section, out var entries)
                || entries.ValueKind != JsonValueKind.Object)
            {
                yield break;
            }

            foreach (var entry in entries.EnumerateObject())
            {
                if (entry.Value.ValueKind != JsonValueKind.Object)
                {
                    continue;
                }

                var source = Text(entry.Value, "source");
                var kind = Text(entry.Value, "kind");

                if (Text(entry.Value, "query").Length > 0)
                {
                    continue;
                }

                if (source.EndsWith("_total", StringComparison.Ordinal)
                    && UntouchedKinds.Contains(kind, StringComparer.Ordinal))
                {
                    yield return $"{entry.Name} declares `{source}` as {kind}";
                }
            }
        }

        private static string Text(JsonElement element, string name)
        {
            return element.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String
                ? value.GetString() ?? string.Empty
                : string.Empty;
        }

        /// <summary>
        /// Every configuration this repository ships, including the one embedded in the ConfigMap — which is
        /// the one that actually runs, and the one a file-only check would miss.
        /// </summary>
        private static IEnumerable<(string Label, JsonElement Document)> ShippedConfigs()
        {
            foreach (var path in Directory.GetFiles(
                         RepositoryPaths.FromRoot("k8s", "anomaly-guard"), "*.json"))
            {
                yield return (Path.GetFileName(path),
                    JsonDocument.Parse(File.ReadAllText(path)).RootElement.Clone());
            }

            var yaml = File.ReadAllText(RepositoryPaths.FromRoot("k8s", "lab", "anomaly-guard.yaml"))
                .Replace("\r\n", "\n", StringComparison.Ordinal);

            var block = Regex.Match(yaml, @"^  guard\.json: \|\n(.*?)^---",
                RegexOptions.Singleline | RegexOptions.Multiline);

            Assert.True(block.Success, "no `guard.json: |` block in k8s/lab/anomaly-guard.yaml");

            var body = string.Join("\n", block.Groups[1].Value.Split('\n')
                .Where(l => l.Trim().Length > 0)
                .Select(l => l.StartsWith("    ", StringComparison.Ordinal) ? l[4..] : l));

            yield return ("anomaly-guard.yaml (deployed)",
                JsonDocument.Parse(body).RootElement.Clone());
        }
    }
}
