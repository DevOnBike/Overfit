// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Reflection;
using System.Text.Json;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Statistics;

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
    ///
    /// <para><b>The BUILT-IN half was still missing until 2026-08-12</b>, and the custom fix hid it: the
    /// file's `thresholds` entries carried three floors and the guard's options carried four, so a
    /// deployment could set `minGapChange` on a custom channel and had no way at all to set it on
    /// `MemoryWorkingSetBytes`. Enabling the gate from a ConfigMap would have failed at startup on the
    /// per-metric table before any custom channel was reached.</para>
    ///
    /// <para><b>And the floor was only half of it — the GATE itself could not be switched on either, found
    /// 2026-08-12 while closing the half above.</b> `AnomalyGuardOptions.PeerNovelty` had no config key at
    /// all, so a mechanism with a plan, an ADR, a tracker, persistence and 25 tests was reachable only by a
    /// host assembling options in code. The two are one feature: a profile without a floor refuses to start,
    /// and a floor without a profile does nothing.</para>
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

        /// <summary>
        /// Every floor a built-in <c>thresholds</c> entry carries survives the file, at the index the enum
        /// names. Distinct values on purpose: a reader that copied one field into another would still pass
        /// if they shared a number.
        /// </summary>
        [Fact]
        public void EveryFloorOnABuiltInThresholdCanBeSetFromTheConfigFile()
        {
            var (gap, trend, step, gapChange) = ReadThresholds(
                """
                {
                  "prometheus": "http://127.0.0.1:9090",
                  "namespace": "lab",
                  "podRegex": "app-.*",
                  "thresholds": {
                    "MemoryWorkingSetBytes": {
                      "minGap": "11MB",
                      "minTrendChange": "22MB",
                      "minStepChange": "33MB",
                      "minGapChange": "44MB"
                    }
                  }
                }
                """);

            const double mb = 1_000_000.0;
            const int index = (int)MetricIndex.MemoryWorkingSetBytes;

            Assert.Equal(11.0 * mb, gap[index]);
            Assert.Equal(22.0 * mb, trend[index]);
            Assert.Equal(33.0 * mb, step[index]);

            // The one that was unreachable until 2026-08-12.
            Assert.NotNull(gapChange);
            Assert.Equal(44.0 * mb, gapChange[index]);

            // And nowhere else: a reader that filled the table instead of one slot would leave the gate on
            // for twelve features whose floor nobody wrote down.
            Assert.Equal(0.0, gapChange[(int)MetricIndex.LatencyP95Ms]);
        }

        /// <summary>
        /// The arm that matters most: a file that declares no <c>minGapChange</c> hands back <b>no table</b>,
        /// and a guard with the novelty gate switched on still refuses to start.
        ///
        /// <para><b>A zero-filled table would clear that check.</b> <c>AnomalyGuard.RestoreNovelty</c> tests
        /// only that the table exists and is long enough, so returning one full of zeros would start the
        /// guard with a suppression gate running on the relative test alone — a threshold nobody chose,
        /// applied to a decision whose failure mode is silence. This is the mutation that has to stay red.
        /// </para>
        /// </summary>
        [Fact]
        public void OmittingTheGapChangeLeavesNoTableAndTheGateStillRefusesToStart()
        {
            var (_, _, _, gapChange) = ReadThresholds(
                """
                {
                  "prometheus": "http://127.0.0.1:9090",
                  "namespace": "lab",
                  "podRegex": "app-.*",
                  "thresholds": {
                    "MemoryWorkingSetBytes": { "minGap": "11MB", "minTrendChange": "22MB" }
                  }
                }
                """);

            Assert.Null(gapChange);

            var error = Assert.Throws<ArgumentException>(() => new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "lab",
                    Workload = "lab-workload",
                    PeerNovelty = PeerNoveltyOptions.Daily,
                    MinAbsoluteGapChange = gapChange,
                },
                new CapturingSink(),
                IncidentTrackingOptions.Balanced));

            Assert.Contains(nameof(AnomalyGuardOptions.MinAbsoluteGapChange), error.Message,
                StringComparison.Ordinal);
        }

        /// <summary>
        /// An explicit <c>"0"</c> is "not configured", not "gate off" — the same trap `AN-D4b` recorded for
        /// the step floor one field along, and the reason the reader tests the parsed value rather than the
        /// presence of the key.
        /// </summary>
        [Fact]
        public void AZeroGapChangeIsNotAConfiguredFloor()
        {
            var (_, _, _, gapChange) = ReadThresholds(
                """
                {
                  "prometheus": "http://127.0.0.1:9090",
                  "namespace": "lab",
                  "podRegex": "app-.*",
                  "thresholds": {
                    "MemoryWorkingSetBytes": { "minGap": "11MB", "minGapChange": "0" }
                  }
                }
                """);

            Assert.Null(gapChange);
        }

        /// <summary>
        /// A value that is not a number is reported and the table stays absent. The direction matters: a typo
        /// must not be the difference between a guard that refuses to start and one that starts with a
        /// suppression gate switched on and its floor at zero.
        /// </summary>
        [Fact]
        public void AnUnreadableGapChangeIsReportedAndLeavesNoTable()
        {
            var (_, _, _, gapChange) = AnomalyGuardConfigReader.ReadThresholds(
                Parse(
                    """
                    {
                      "prometheus": "http://127.0.0.1:9090",
                      "namespace": "lab",
                      "podRegex": "app-.*",
                      "thresholds": {
                        "MemoryWorkingSetBytes": { "minGapChange": "quite a lot" }
                      }
                    }
                    """),
                out var problems);

            Assert.Null(gapChange);
            Assert.Contains(problems, p => p.Contains("minGapChange", StringComparison.Ordinal)
                                           && p.Contains("gate is OFF", StringComparison.Ordinal));
        }

        /// <summary>
        /// A <c>thresholds</c> key that is not a feature is dropped with the rest of its entry, so a
        /// misspelled metric name cannot smuggle a floor in under a name the guard never reads.
        /// </summary>
        [Fact]
        public void AGapChangeUnderAnUnknownFeatureIsDroppedAndReported()
        {
            var (_, _, _, gapChange) = AnomalyGuardConfigReader.ReadThresholds(
                Parse(
                    """
                    {
                      "prometheus": "http://127.0.0.1:9090",
                      "namespace": "lab",
                      "podRegex": "app-.*",
                      "thresholds": {
                        "MemoryWorkingSetBytesTypo": { "minGapChange": "44MB" }
                      }
                    }
                    """),
                out var problems);

            Assert.Null(gapChange);
            Assert.Contains(problems, p => p.Contains("not a known feature", StringComparison.Ordinal));
        }

        /// <summary>
        /// The file names a cadence profile and gets that profile — by name, because
        /// <see cref="PeerNoveltyOptions"/> ships presets and refuses <c>default</c>.
        /// </summary>
        [Theory]
        [InlineData("PerShift", 8.0)]
        [InlineData("Daily", 24.0)]
        [InlineData("Weekly", 168.0)]
        [InlineData("daily", 24.0)]
        public void ANamedProfileIsReadFromTheConfigFile(string name, double hours)
        {
            var profile = AnomalyGuardConfigReader.ReadPeerNovelty(
                Parse($$"""
                       {
                         "prometheus": "http://127.0.0.1:9090",
                         "namespace": "lab",
                         "podRegex": "app-.*",
                         "peerNovelty": "{{name}}"
                       }
                       """),
                out var problems);

            Assert.Empty(problems);
            Assert.NotNull(profile);

            // The interval is what distinguishes the three, so asserting it is what makes a reader that
            // returned the wrong preset fail rather than look right.
            Assert.Equal(TimeSpan.FromHours(hours), profile.Value.StandingReassertionInterval);
            Assert.True(profile.Value.IsValid);
        }

        /// <summary>
        /// No key means no gate. This is the arm that protects every deployment written before the key
        /// existed, and it is the reason nothing here defaults.
        /// </summary>
        [Fact]
        public void AFileThatNamesNoProfileLeavesTheGateOff()
        {
            var profile = AnomalyGuardConfigReader.ReadPeerNovelty(
                Parse(
                    """
                    {
                      "prometheus": "http://127.0.0.1:9090",
                      "namespace": "lab",
                      "podRegex": "app-.*"
                    }
                    """),
                out var problems);

            Assert.Empty(problems);
            Assert.Null(profile);
        }

        /// <summary>
        /// A misspelled profile is reported and the gate stays OFF — never rounded to the nearest name.
        ///
        /// <para>The direction is the point. Off leaves the guard as noisy as it was, which an operator
        /// notices; guessing a profile from a typo would switch a suppression gate on, and a suppression gate
        /// nobody chose is indistinguishable from a quiet cluster.</para>
        /// </summary>
        [Fact]
        public void AnUnknownProfileNameIsReportedAndRefused()
        {
            var profile = AnomalyGuardConfigReader.ReadPeerNovelty(
                Parse(
                    """
                    {
                      "prometheus": "http://127.0.0.1:9090",
                      "namespace": "lab",
                      "podRegex": "app-.*",
                      "peerNovelty": "Dayly"
                    }
                    """),
                out var problems);

            Assert.Null(profile);

            var problem = Assert.Single(problems);

            Assert.Contains("Dayly", problem, StringComparison.Ordinal);
            Assert.Contains("PerShift, Daily, Weekly", problem, StringComparison.Ordinal);
            Assert.Contains("OFF", problem, StringComparison.Ordinal);
        }

        /// <summary>
        /// Every preset <see cref="PeerNoveltyOptions"/> publishes is reachable by name from the file.
        ///
        /// <para><b>This is the drift gate, and it exists because the mapping is a hand-written switch.</b>
        /// The presets are static properties rather than enum members, so the reader cannot enumerate them
        /// and a fourth one added later would be silently un-nameable — the same shape as the defect this
        /// whole class was written for, one level up. Walking the type is what turns "remember to update the
        /// reader" from a comment into a failing test.</para>
        ///
        /// <para>The count is asserted before the loop: a reflection query that returned nothing would make
        /// this pass vacuously, which is the failure mode of every test that iterates a collection.</para>
        /// </summary>
        [Fact]
        public void EveryPublishedProfileIsReachableByNameFromTheConfigFile()
        {
            var presets = typeof(PeerNoveltyOptions)
                .GetProperties(BindingFlags.Public | BindingFlags.Static)
                .Where(p => p.PropertyType == typeof(PeerNoveltyOptions))
                .ToList();

            Assert.True(
                presets.Count >= 3,
                $"expected at least the three documented presets, found [{presets.Count}] — a reflection "
                + "query that finds nothing would make the loop below assert nothing at all");

            foreach (var preset in presets)
            {
                var expected = (PeerNoveltyOptions)preset.GetValue(null)!;
                var file = new AnomalyGuardConfigFile { PeerNovelty = preset.Name };
                var read = AnomalyGuardConfigReader.ReadPeerNovelty(file, out var problems);

                Assert.True(
                    problems.Count == 0,
                    $"PeerNoveltyOptions.{preset.Name} is published but the config reader does not know the "
                    + $"name: {string.Join("; ", problems)}");

                Assert.Equal(expected, read);
            }
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

        private static (double[] Gap, double[] Trend, double[] Step, double[]? GapChange) ReadThresholds(
            string json)
        {
            var tables = AnomalyGuardConfigReader.ReadThresholds(Parse(json), out var problems);

            Assert.True(problems.Count == 0, string.Join("; ", problems));

            return tables;
        }

        private sealed class CapturingSink : IIncidentSink
        {
            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
            }
        }
    }
}
