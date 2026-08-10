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
    /// A binding can be in the wrong UNIT while being a perfectly valid query, and nothing else in this
    /// system notices.
    ///
    /// <para><b>This is the gate the ErrorRate defect walked past.</b> `ErrorRate` was bound to
    /// `labapp_errors_total` with `kind: Counter` — errors per second, unbounded — while `MetricSnapshot`
    /// documents feature [8] as `rate(5xx) / rate(total)`, a fraction on [0,1]. It shipped in the ConfigMap
    /// on 2026-08-01, the standalone file's own comment said the channel was left out, and
    /// `DeployedConfigParityTests` — which enforces that the two files agree — resolved the disagreement by
    /// copying the binding into the file. A parity gate makes two artefacts the same and has no opinion
    /// about which one was right. `MetricKindSanityTests` did not catch it either: that test asks whether a
    /// `_total` series is declared Gauge or Ratio, and `Counter` on a `_total` is exactly correct — as a
    /// rate. The missing question was what the channel PROMISED.</para>
    ///
    /// <para><b>What this refuses to check.</b> A gauge passes its series through and a hand-written query
    /// computes whatever it computes; neither unit is visible in configuration. Those are reported as
    /// unverifiable and counted, never silently treated as agreeing — a check that cannot fail on the cases
    /// it skips is worth less than an admitted gap.</para>
    /// </summary>
    public sealed class MetricUnitContractTests
    {
        /// <summary>
        /// Every channel states its unit, and the switch throws for anything unlisted. Without this a new
        /// `MetricIndex` member picks up whatever `Declared` happens to do — which is the failure mode
        /// `MetricIndex.Count` vs `MetricSnapshot.FeatureCount` already exists to warn about.
        /// </summary>
        [Fact]
        public void EveryChannelDeclaresItsUnit()
        {
            for (var index = 0; index < (int)MetricIndex.Count; index++)
            {
                var metric = (MetricIndex)index;
                var unit = MetricUnits.Declared(metric);

                Assert.True(Enum.IsDefined(unit), $"{metric} declared an undefined unit");
            }

            Assert.Throws<ArgumentOutOfRangeException>(() => MetricUnits.Declared(MetricIndex.Count));
        }

        /// <summary>
        /// The defect itself, as a fixture, so this test is known to catch what it was written for rather
        /// than merely passing on today's configuration.
        /// </summary>
        [Fact]
        public void TheErrorRateBindingThatShippedIsRejected()
        {
            var produced = MetricUnits.Produced("labapp_errors_total", MetricSourceKind.Counter, false);

            Assert.Equal(MetricUnit.PerSecond, produced);
            Assert.NotEqual(MetricUnits.Declared(MetricIndex.ErrorRate), produced);
        }

        /// <summary>
        /// The control that stops this becoming "reject every Counter". A rate over a seconds counter IS
        /// dimensionless, which is why `GcPauseRatio` is correctly a fraction and `ErrorRate` is not.
        /// </summary>
        [Fact]
        public void ARateOverASecondsCounterIsAFractionOfTime()
        {
            Assert.Equal(
                MetricUnit.Fraction,
                MetricUnits.Produced("dotnet_gc_pause_seconds_total", MetricSourceKind.Counter, false));

            Assert.Equal(
                MetricUnit.Cores,
                MetricUnits.Produced("container_cpu_usage_seconds_total", MetricSourceKind.Counter, false));
        }

        [Fact]
        public void AGaugeAndAHandWrittenQueryAreUnverifiableRatherThanAssumedCorrect()
        {
            Assert.Null(MetricUnits.Produced("anything_bytes", MetricSourceKind.Gauge, false));
            Assert.Null(MetricUnits.Produced("anything_total", MetricSourceKind.Counter, true));
        }

        /// <summary>
        /// The gate over what actually ships. Every built-in channel bound in every shipped configuration —
        /// including the ConfigMap, which is the one that runs.
        /// </summary>
        [Fact]
        public void NoShippedConfigBindsAChannelInTheWrongUnit()
        {
            var mismatches = new List<string>();
            var unverifiable = 0;
            var checkedCount = 0;

            foreach (var (label, document) in ShippedConfigs())
            {
                if (!document.TryGetProperty("metrics", out var entries)
                    || entries.ValueKind != JsonValueKind.Object)
                {
                    continue;
                }

                foreach (var entry in entries.EnumerateObject())
                {
                    if (entry.Value.ValueKind != JsonValueKind.Object
                        || !Enum.TryParse<MetricIndex>(entry.Name, out var metric)
                        || metric == MetricIndex.Count)
                    {
                        continue;
                    }

                    var source = Text(entry.Value, "source");
                    var hasQuery = Text(entry.Value, "query").Length > 0;

                    if (!Enum.TryParse<MetricSourceKind>(Text(entry.Value, "kind"), out var kind))
                    {
                        continue;
                    }

                    var produced = MetricUnits.Produced(source, kind, hasQuery);

                    if (produced is null)
                    {
                        unverifiable++;

                        continue;
                    }

                    checkedCount++;
                    var declared = MetricUnits.Declared(metric);

                    if (produced != declared)
                    {
                        mismatches.Add(
                            $"{label}: {metric} promises {declared} but `{source}` as {kind} produces "
                            + $"{produced}");
                    }
                }
            }

            Assert.True(
                checkedCount > 0,
                "no binding was decidable — this test would pass on an empty config file, which is not a "
                + "result. Check ShippedConfigs still finds the shipped configurations.");

            Assert.True(
                mismatches.Count == 0,
                $"{checkedCount} binding(s) checked, {unverifiable} unverifiable (gauge or hand-written "
                + "query). A threshold written against the promised unit means nothing against the produced "
                + "one:" + Environment.NewLine + "  "
                + string.Join(Environment.NewLine + "  ", mismatches));
        }

        private static string Text(JsonElement element, string name)
        {
            return element.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String
                ? value.GetString() ?? string.Empty
                : string.Empty;
        }

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

            yield return ("anomaly-guard.yaml (deployed)", JsonDocument.Parse(body).RootElement.Clone());
        }
    }
}
