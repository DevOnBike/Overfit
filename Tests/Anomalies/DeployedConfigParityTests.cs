// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using System.Text.RegularExpressions;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// The guard's deployed ConfigMap and its standalone configuration file must declare the same channels
    /// and the same floors.
    ///
    /// <para><b>Both directions of this have already gone wrong, on consecutive days.</b> On 2026-08-08 an
    /// <c>apply</c> of <c>k8s/lab/anomaly-guard.yaml</c> — which carried no <c>customMetrics</c> block at all
    /// — <b>deleted the live <c>GcCommittedBytes</c> binding</b> and reported <c>configured</c>. Nothing
    /// noticed: a channel that is no longer evaluated produces no findings, and neither does a healthy
    /// cluster. On 2026-08-09 the opposite was found: the file held <b>one</b> threshold while the cluster
    /// held <b>nine</b>, and the one they shared disagreed — the file said <c>256MiB</c>, a round number, and
    /// the cluster said <c>9.52MB</c>, a measured one. A third file, <c>guard.lab-workload.calibrated.json</c>,
    /// held the real values and was referenced by nothing.</para>
    ///
    /// <para><b>The rule existed and nothing enforced it.</b> <c>k8s/README.md</c> already said the two must
    /// agree. That is the part worth noticing: the failure was not a missing rule, it was a rule with no
    /// mechanism, and this repository has now produced that shape three times.</para>
    ///
    /// <para>Keys beginning with <c>_</c> are ignored. Comments belong in the file a human edits and would
    /// bloat a ConfigMap that has no comment syntax, so their absence is not drift.</para>
    /// </summary>
    public sealed class DeployedConfigParityTests
    {
        [Fact]
        public void TheConfigMapAndTheStandaloneConfigDeclareTheSameChannelsAndFloors()
        {
            var deployed = EmbeddedGuardJson();
            var standalone = Parse(File.ReadAllText(
                RepositoryPaths.FromRoot("k8s", "anomaly-guard", "guard.lab-workload.json")));

            foreach (var section in new[] { "metrics", "customMetrics", "thresholds" })
            {
                var left = Section(deployed, section);
                var right = Section(standalone, section);

                Assert.True(
                    left.Keys.SequenceEqual(right.Keys),
                    $"{section}: the ConfigMap declares [{string.Join(", ", left.Keys)}] and the file "
                    + $"declares [{string.Join(", ", right.Keys)}]. Applying the ConfigMap would remove or "
                    + "add channels silently — that is how GcCommittedBytes was lost on 2026-08-08.");

                foreach (var name in left.Keys)
                {
                    Assert.True(
                        left[name] == right[name],
                        $"{section}.{name} differs.\n  ConfigMap: {left[name]}\n  file:      {right[name]}");
                }
            }
        }

        [Fact]
        public void TheDeploymentTargetsAgree()
        {
            var deployed = EmbeddedGuardJson();
            var standalone = Parse(File.ReadAllText(
                RepositoryPaths.FromRoot("k8s", "anomaly-guard", "guard.lab-workload.json")));

            // Not cosmetic: a file that names a different namespace or pod pattern describes a different
            // population, so every threshold in it was calibrated against something else.
            foreach (var key in new[] { "namespace", "workload", "podRegex" })
            {
                Assert.Equal(Scalar(deployed, key), Scalar(standalone, key));
            }

            // `prometheus` is deliberately NOT compared: one runs in-cluster and reaches the service by DNS,
            // the other is read from a workstation through a port-forward. Asserting equality there would
            // make the honest configuration fail.
            Assert.NotEqual(Scalar(deployed, "prometheus"), Scalar(standalone, "prometheus"));
        }

        [Fact]
        public void OnlyOneStandaloneConfigDescribesTheLabWorkloadGuard()
        {
            // A second file holding the real values while the first holds stale ones is how the drift
            // happened: guard.lab-workload.calibrated.json carried nine measured thresholds and was
            // referenced by nothing, while guard.lab-workload.json carried one round number.
            var directory = RepositoryPaths.FromRoot("k8s", "anomaly-guard");
            var candidates = Directory.GetFiles(directory, "guard.lab-workload*.json")
                .Select(Path.GetFileName)
                .Order(StringComparer.Ordinal)
                .ToList();

            Assert.True(
                candidates.Count == 1,
                $"expected exactly one, found [{string.Join(", ", candidates)}]. If a variant is genuinely "
                + "needed, give it a name this test recognises and say in the file which one is deployed.");
        }

        [Fact]
        public void TheConfigMapIsAcceptedByTheReaderItWillBeGivenTo()
        {
            // Parsing as JSON is not the same as being usable: a query missing its selector token parses
            // fine and reports on the whole cluster.
            var file = JsonSerializer.Deserialize<AnomalyGuardConfigFile>(
                EmbeddedGuardJsonText(),
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

            Assert.NotNull(file);

            AnomalyGuardConfigReader.ReadMap(file, out var problems);

            Assert.Empty(problems);
        }

        /// <summary>
        /// The deployed coverage binding produces the <b>smoothed</b> query, not the raw <c>up</c> gauge.
        ///
        /// <para><b>Parsing cleanly is not enough here, and the failure would be silent.</b> Drop the
        /// <c>query</c> key and the entry is still valid — <c>kind: Ratio</c> renders <c>up{selector}</c>,
        /// which returns a series for every pod, fills the channel, satisfies every coverage check, and
        /// cannot produce a finding: the peer detector's size gate reads each member's median, and the
        /// median of a <c>0</c>/<c>1</c> series is <c>1.0</c> for any pod above 50% coverage. A channel that
        /// reports numbers and can never fire is the exact shape <c>InertChannel</c> was written for.</para>
        /// </summary>
        [Fact]
        public void TheDeployedCoverageChannelQueriesTheAveragedUpSeries()
        {
            var file = JsonSerializer.Deserialize<AnomalyGuardConfigFile>(
                EmbeddedGuardJsonText(),
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

            Assert.NotNull(file);

            var queries = AnomalyGuardConfigReader.ReadMap(file, out _).CustomQueries();

            Assert.True(
                queries.TryGetValue("ScrapeCoverage", out var query),
                $"the ConfigMap declares [{string.Join(", ", queries.Keys)}] and not ScrapeCoverage");

            Assert.StartsWith("avg_over_time(", query, StringComparison.Ordinal);
            Assert.Contains(PromqlCatalog.SelectorToken, query, StringComparison.Ordinal);
        }

        /// <summary>
        /// Pulls <c>guard.json</c> out of the ConfigMap's literal block. Hand-rolled rather than a YAML
        /// dependency: one block scalar at a known key is not worth a package, and the alternative is that
        /// nothing checks this at all.
        /// </summary>
        private static string EmbeddedGuardJsonText()
        {
            var yaml = File.ReadAllText(RepositoryPaths.FromRoot("k8s", "lab", "anomaly-guard.yaml"))
                .Replace("\r\n", "\n", StringComparison.Ordinal);

            var block = Regex.Match(yaml, @"^  guard\.json: \|\n(.*?)^---",
                RegexOptions.Singleline | RegexOptions.Multiline);

            Assert.True(block.Success, "no `guard.json: |` block in k8s/lab/anomaly-guard.yaml");

            var lines = block.Groups[1].Value.Split('\n')
                .Where(l => l.Trim().Length > 0)
                .Select(l => l.StartsWith("    ", StringComparison.Ordinal) ? l[4..] : l);

            return string.Join("\n", lines);
        }

        private static JsonElement EmbeddedGuardJson()
        {
            return Parse(EmbeddedGuardJsonText());
        }

        private static JsonElement Parse(string text)
        {
            return JsonDocument.Parse(text).RootElement.Clone();
        }

        /// <summary>
        /// A value rendered so that equal values compare equal.
        ///
        /// <para><c>JsonElement.ToString()</c> hands back the raw text, so <c>0.50</c> and <c>0.5</c> read as
        /// different — which this test reported as configuration drift on its first run. A parity check that
        /// fires on formatting teaches people to ignore it, and an ignored gate is the state the drift
        /// happened in.</para>
        /// </summary>
        private static string Canonical(JsonElement value)
        {
            return value.ValueKind == JsonValueKind.Number
                ? value.GetDouble().ToString("R", System.Globalization.CultureInfo.InvariantCulture)
                : value.ToString();
        }

        private static string Scalar(JsonElement root, string name)
        {
            return root.TryGetProperty(name, out var value) ? value.ToString() : string.Empty;
        }

        /// <summary>
        /// One section, normalised: entries sorted by name, `_`-prefixed keys dropped, each rendered as
        /// canonical JSON so the comparison is about content rather than key order or whitespace.
        /// </summary>
        private static SortedDictionary<string, string> Section(JsonElement root, string name)
        {
            var result = new SortedDictionary<string, string>(StringComparer.Ordinal);

            if (!root.TryGetProperty(name, out var section) || section.ValueKind != JsonValueKind.Object)
            {
                return result;
            }

            foreach (var entry in section.EnumerateObject())
            {
                var fields = new SortedDictionary<string, string>(StringComparer.Ordinal);

                if (entry.Value.ValueKind == JsonValueKind.Object)
                {
                    foreach (var field in entry.Value.EnumerateObject())
                    {
                        if (!field.Name.StartsWith('_'))
                        {
                            fields[field.Name] = Canonical(field.Value);
                        }
                    }
                }

                result[entry.Name] = string.Join(", ", fields.Select(f => $"{f.Key}={f.Value}"));
            }

            return result;
        }
    }
}
