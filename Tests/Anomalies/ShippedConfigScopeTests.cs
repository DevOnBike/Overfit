// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// A shipped configuration that omits `workload` is broken in two ways that both look like silence.
    ///
    /// <para><b>`AnomalyGuardConfigFile.Workload` states the cost and it is not theoretical.</b> Without it a
    /// maintenance window naming a workload can never match, so an operator who declares one for a rollout is
    /// paged through it anyway; and the incident tracker keys a pod-less subject on the workload, so every
    /// deployment-level finding in the namespace collapses to `"namespace/"` — one identity shared by
    /// unrelated problems, reported as a single continuing incident. That was visible in the lab's own logs
    /// as `Anomaly incident in lab/:` with nothing after the slash.</para>
    ///
    /// <para><b>Found 2026-08-10 (`AN-F3`) in `guard.lab.json`</b>, which had carried the gap since it was
    /// written. Nothing caught it because every other check reads `metrics`, `customMetrics` and
    /// `thresholds` — the sections with entries in them — and a missing scalar at the top of the file is not
    /// an entry that can be wrong, it is an entry that is not there.</para>
    /// </summary>
    public sealed class ShippedConfigScopeTests
    {
        /// <summary>
        /// Every configuration this repository ships names its scope completely. The assertion is one line;
        /// the value is that it covers files added later, which is the case that actually recurs.
        /// </summary>
        [Fact]
        public void EveryShippedConfigNamesItsNamespacePodRegexAndWorkload()
        {
            var problems = new List<string>();
            var checkedFiles = 0;

            foreach (var path in Directory.GetFiles(
                         RepositoryPaths.FromRoot("k8s", "anomaly-guard"), "*.json"))
            {
                checkedFiles++;

                var file = JsonSerializer.Deserialize<AnomalyGuardConfigFile>(
                    File.ReadAllText(path),
                    new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

                var name = Path.GetFileName(path);

                if (file is null)
                {
                    problems.Add($"{name}: does not parse as a guard configuration");

                    continue;
                }

                // Scopes are the multi-scope form; a file using them states its scope there instead, and
                // demanding the top-level fields as well would refuse a shape the reader accepts.
                if (file.Scopes.Count > 0)
                {
                    continue;
                }

                foreach (var (field, value) in new[]
                         {
                             ("namespace", file.Namespace),
                             ("podRegex", file.PodRegex),
                             ("workload", file.Workload),
                         })
                {
                    if (string.IsNullOrWhiteSpace(value))
                    {
                        problems.Add($"{name}: `{field}` is blank");
                    }
                }
            }

            Assert.True(
                checkedFiles > 0,
                "no shipped configuration was read — this test would pass on an empty directory, which is "
                + "not a result.");

            Assert.True(
                problems.Count == 0,
                "a blank scope field is not a neutral default. `workload` in particular silently breaks "
                + "maintenance windows and collapses every deployment-level finding onto one identity:\n  "
                + string.Join("\n  ", problems));
        }
    }
}
