// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// A property a deployment cannot set is a feature that is not shipped, however well it is tested.
    ///
    /// <para><b>Written after `MinAbsoluteGapChange` was found unreachable from the config file on
    /// 2026-08-10.</b> `CustomMetricBinding` carried the property, `AnomalyGuardConfigFile.CustomEntry` had no
    /// field for it, and `AnomalyGuardConfigReader` never set it — so it was permanently zero for every
    /// deployment. Nothing caught it, because the peer-novelty gate that requires it had only ever been
    /// exercised with options assigned in code. It was harmless while the gate was off and would have thrown
    /// at startup the day it was switched on, since `AnomalyGuard.RestoreNovelty` refuses a binding whose
    /// value is zero.</para>
    ///
    /// <para><b>What this checks and what it deliberately does not.</b> It compares the binding's numeric
    /// knobs against what a config file can express, by round-tripping a file that sets all of them. It does
    /// not check that the values are sensible — that is a different question — only that they arrive at all.
    /// A test that asserted defaults would pass on a reader that ignored the file entirely, which is the
    /// exact failure being closed here.</para>
    /// </summary>
    public sealed class ConfigSurfaceCompletenessTests
    {
        /// <summary>
        /// Every numeric knob a custom binding carries survives the file. The values are distinct on purpose:
        /// if two shared a number, a reader that assigned the wrong one would still pass.
        /// </summary>
        [Fact]
        public void EveryNumericKnobOnACustomBindingCanBeSetFromTheConfigFile()
        {
            var map = ReadMap(
                """
                {
                  "prometheus": "http://127.0.0.1:9090",
                  "namespace": "lab",
                  "podRegex": "app-.*",
                  "customMetrics": {
                    "Probe": {
                      "source": "probe_bytes",
                      "kind": "Gauge",
                      "class": "Resource",
                      "minGap": "11MB",
                      "minTrendChange": "22MB",
                      "minGapChange": "33MB"
                    }
                  }
                }
                """);

            var binding = Assert.Single(map.Custom);

            // Decimal MB, not MiB — `MetricQuantity` maps "MB" to 1_000_000. Checked in the source
            // rather than assumed, because a wrong constant here would have made this test fail against
            // a correct reader and sent the next reader hunting in the wrong file.
            const double mb = 1_000_000.0;

            Assert.Equal(11.0 * mb, binding.MinAbsoluteGap);
            Assert.Equal(22.0 * mb, binding.MinAbsoluteTrendChange);

            // The one that was unreachable. Asserted against its own distinct value so a reader that copied
            // minGap or minTrendChange into it would fail here rather than look correct.
            Assert.Equal(33.0 * mb, binding.MinAbsoluteGapChange);
        }

        /// <summary>
        /// The regression itself: a file that omits the key leaves the value at zero, which is what the
        /// novelty gate refuses. Pins the behaviour so "unset" stays distinguishable from "set to something".
        /// </summary>
        [Fact]
        public void OmittingTheGapChangeLeavesItZeroRatherThanInventingOne()
        {
            var map = ReadMap(
                """
                {
                  "prometheus": "http://127.0.0.1:9090",
                  "namespace": "lab",
                  "podRegex": "app-.*",
                  "customMetrics": {
                    "Probe": { "source": "probe_bytes", "kind": "Gauge", "minGap": "1MB" }
                  }
                }
                """);

            Assert.Equal(0.0, Assert.Single(map.Custom).MinAbsoluteGapChange);
        }

        /// <summary>
        /// The same class of gap one field along, and the halves reversed: <c>CustomEntry</c> has inherited a
        /// <c>Query</c> property from <c>MetricEntry</c> all along — so the JSON key always parsed — and
        /// <c>AnomalyGuardConfigReader</c>'s custom loop never looked at it. A file could set the key and
        /// watch it be dropped on the floor.
        ///
        /// <para>Not academic: <c>ScrapeCoverage</c> cannot be expressed without it. No
        /// <see cref="MetricSourceKind"/> renders <c>avg_over_time(up{%selector%}[15m])</c>, and the raw gauge
        /// that <c>Ratio</c> does render is one the peer detector's size gate cannot see.</para>
        /// </summary>
        [Fact]
        public void AVerbatimQueryOnACustomBindingSurvivesTheConfigFile()
        {
            var map = ReadMap(
                """
                {
                  "prometheus": "http://127.0.0.1:9090",
                  "namespace": "lab",
                  "podRegex": "app-.*",
                  "customMetrics": {
                    "ScrapeCoverage": {
                      "source": "up",
                      "kind": "Ratio",
                      "query": "avg_over_time(up{%selector%}[15m])",
                      "requirePersistence": true,
                      "calibrated": false
                    }
                  }
                }
                """);

            var binding = Assert.Single(map.Custom);

            Assert.Equal("avg_over_time(up{%selector%}[15m])", binding.Query);
            Assert.True(binding.RequirePersistence);
            Assert.False(binding.Calibrated);

            // And it has to reach the query the source actually issues, not merely the binding: a field that
            // arrives and is then ignored one layer down is the same defect wearing a different coat.
            Assert.Equal(
                "avg_over_time(up{%selector%}[15m])",
                map.CustomQueries()["ScrapeCoverage"]);

            // The name-based exemption the calibrator consumes, derived from the binding rather than
            // configured twice.
            Assert.Equal(["ScrapeCoverage"], map.NonCalibratedChannels);
        }

        /// <summary>
        /// The two flags default to what every channel did before they existed, so an existing file cannot
        /// change behaviour by standing still.
        /// </summary>
        [Fact]
        public void OmittingTheNewFlagsLeavesACustomBindingBehavingAsItAlwaysDid()
        {
            var map = ReadMap(
                """
                {
                  "prometheus": "http://127.0.0.1:9090",
                  "namespace": "lab",
                  "podRegex": "app-.*",
                  "customMetrics": {
                    "Probe": { "source": "probe_bytes", "kind": "Gauge" }
                  }
                }
                """);

            var binding = Assert.Single(map.Custom);

            Assert.Equal(string.Empty, binding.Query);
            Assert.False(binding.RequirePersistence);
            Assert.True(binding.Calibrated);
            Assert.Empty(map.NonCalibratedChannels);

            // The kind still builds the query, which is the half that must not have moved.
            Assert.Equal("sum by (pod) (probe_bytes{%selector%})", map.CustomQueries()["Probe"]);
        }

        /// <summary>
        /// A custom query with no selector token is refused, exactly as a built-in one already was. Without
        /// the token the query ignores the namespace and pod matchers and silently reports on the whole
        /// cluster — the check existed on one loop and not the other purely because the other never read the
        /// field.
        /// </summary>
        [Fact]
        public void ACustomQueryWithoutTheSelectorTokenIsRejected()
        {
            var file = Parse(
                """
                {
                  "prometheus": "http://127.0.0.1:9090",
                  "namespace": "lab",
                  "podRegex": "app-.*",
                  "customMetrics": {
                    "ScrapeCoverage": {
                      "source": "up",
                      "kind": "Ratio",
                      "query": "avg_over_time(up[15m])"
                    }
                  }
                }
                """);

            var map = AnomalyGuardConfigReader.ReadMap(file, out var problems);

            Assert.Empty(map.Custom);
            Assert.Contains(problems, p => p.Contains("%selector%", StringComparison.Ordinal));
        }

        private static AnomalyGuardConfigFile Parse(string json)
        {
            var file = JsonSerializer.Deserialize<AnomalyGuardConfigFile>(
                json,
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

            Assert.NotNull(file);

            return file;
        }

        private static MetricMap ReadMap(string json)
        {
            var map = AnomalyGuardConfigReader.ReadMap(Parse(json), out var problems);

            Assert.True(problems.Count == 0, string.Join("; ", problems));

            return map;
        }
    }
}
