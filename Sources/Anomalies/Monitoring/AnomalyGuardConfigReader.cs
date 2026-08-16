// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    /// Turns a filled-in <see cref="AnomalyGuardConfigFile"/> into the objects the guard runs on, and says
    /// what it could not understand.
    ///
    /// <para><b>Problems are collected, not thrown.</b> A configuration with one bad enum name should report
    /// that line and everything else wrong with it in the same pass, rather than failing on the first and
    /// making the operator re-run to find the second. And an unreadable entry is <b>dropped</b> rather than
    /// defaulted: a threshold that silently became zero is a gate that silently stopped gating, which is
    /// exactly the class of failure this whole pipeline keeps paying for.</para>
    /// </summary>
    public static class AnomalyGuardConfigReader
    {
        /// <summary>
        /// Reads the file. <paramref name="problems"/> receives one line per entry that could not be used;
        /// an empty list means everything in the file was understood, not that the file is complete.
        /// </summary>
        public static MetricMap ReadMap(AnomalyGuardConfigFile file, out IReadOnlyList<string> problems)
        {
            ArgumentNullException.ThrowIfNull(file);

            var found = new List<string>();
            var bindings = new List<MetricBinding>();
            var custom = new List<CustomMetricBinding>();

            foreach (var (key, entry) in file.Metrics)
            {
                if (!Enum.TryParse<MetricIndex>(key, ignoreCase: true, out var metric)
                    || metric == MetricIndex.Count)
                {
                    found.Add($"Metrics['{key}'] — not a known feature; see MetricIndex for the list.");

                    continue;
                }

                if (!TryKind(entry.Kind, out var kind))
                {
                    found.Add($"Metrics['{key}'].kind = '{entry.Kind}' — not one of {KindNames()}.");

                    continue;
                }

                var query = entry.Query.Trim();

                if (query.Length > 0 && !query.Contains(PromqlCatalog.SelectorToken, StringComparison.Ordinal))
                {
                    found.Add($"Metrics['{key}'].query does not contain {PromqlCatalog.SelectorToken} — "
                              + "without it the query ignores the namespace and pod matchers and silently "
                              + "reports on the whole cluster.");

                    continue;
                }

                if (string.IsNullOrWhiteSpace(entry.Source) && query.Length == 0)
                {
                    found.Add($"Metrics['{key}'].source is blank — omit the entry instead, which states "
                              + "plainly that this cluster does not have it.");

                    continue;
                }

                bindings.Add(new MetricBinding(
                    metric, entry.Source.Trim(), kind, entry.Quantile, query));
            }

            foreach (var (key, entry) in file.CustomMetrics)
            {
                if (Enum.TryParse<MetricIndex>(key, ignoreCase: true, out _))
                {
                    found.Add($"CustomMetrics['{key}'] — that name is already a modelled feature; put it "
                              + "under Metrics so it reaches the learned family too.");

                    continue;
                }

                if (!TryKind(entry.Kind, out var kind))
                {
                    found.Add($"CustomMetrics['{key}'].kind = '{entry.Kind}' — not one of {KindNames()}.");

                    continue;
                }

                if (!Enum.TryParse<SignalClass>(entry.Class, ignoreCase: true, out var signalClass))
                {
                    found.Add($"CustomMetrics['{key}'].class = '{entry.Class}' — not one of "
                              + "Infrastructure, Resource, Symptom.");

                    continue;
                }

                var customQuery = entry.Query.Trim();

                // The same check the Metrics loop above applies, and it was missing here — a custom entry
                // could carry a query with no selector token and would have reported on every pod in the
                // cluster. Unreachable until now only because the reader never read the field at all.
                if (customQuery.Length > 0
                    && !customQuery.Contains(PromqlCatalog.SelectorToken, StringComparison.Ordinal))
                {
                    found.Add($"CustomMetrics['{key}'].query does not contain "
                              + PromqlCatalog.SelectorToken + " — without it the query ignores the namespace "
                              + "and pod matchers and silently reports on the whole cluster.");

                    continue;
                }

                if (string.IsNullOrWhiteSpace(entry.Source) && customQuery.Length == 0)
                {
                    found.Add($"CustomMetrics['{key}'].source is blank.");

                    continue;
                }

                custom.Add(new CustomMetricBinding(
                    Name: key,
                    Source: entry.Source.Trim(),
                    Kind: kind,
                    SignalKind: entry.LoadSensitive
                        ? PeerSignalKind.LoadSensitive
                        : PeerSignalKind.LoadIndependent,
                    Class: signalClass,
                    MinAbsoluteGap: Quantity($"CustomMetrics['{key}'].minGap", entry.MinGap, found),
                    MinAbsoluteTrendChange:
                        Quantity($"CustomMetrics['{key}'].minTrendChange", entry.MinTrendChange, found),

                    // Zero when absent, which AnomalyGuard.LevelShiftFloor reads as "fall back to the trend
                    // floor" — what the step gate did before this field existed.
                    MinAbsoluteStepChange:
                        Quantity($"CustomMetrics['{key}'].minStepChange", entry.MinStepChange, found),

                    // Required by the peer-novelty gate and by nothing else, so it stays optional here and is
                    // enforced where it is used: AnomalyGuard.RestoreNovelty refuses a zero. Reading it was
                    // missing entirely until 2026-08-10 — the binding carried the property, the file could
                    // not set it, and the gate had only ever been exercised with options assigned in code.
                    MinAbsoluteGapChange:
                        Quantity($"CustomMetrics['{key}'].minGapChange", entry.MinGapChange, found),
                    Quantile: entry.Quantile,
                    Rule: BuildRule(key, entry, found),

                    // Inherited from MetricEntry, so the JSON key has always parsed — and until 2026-08-10
                    // this loop never looked at it, so it parsed into the DTO and was dropped. The same shape
                    // as minGapChange one field along, with the halves reversed: there the field was missing,
                    // here the reader was.
                    Query: customQuery,
                    RequirePersistence: entry.RequirePersistence,
                    Calibrated: entry.Calibrated));
            }

            problems = found;

            return new MetricMap(bindings, custom);
        }

        /// <summary>
        /// The per-feature absolute floors, as the positional tables the guard takes. Built here so the
        /// indexed-by-enum shape stays an implementation detail rather than something a human has to write.
        ///
        /// <para><b><c>GapChange</c> is nullable and the other three are not.</b> Absent from the first three
        /// means "that gate is off", which is a table of zeros; absent from the fourth means "not
        /// configured", which is <c>null</c> — and a guard with the peer-novelty gate switched on refuses to
        /// start on a null rather than suppressing findings against a floor nobody chose
        /// (<c>AnomalyGuard.RestoreNovelty</c>). Returning a zero-filled table there would clear that check
        /// and turn a loud failure into a silent one.</para>
        /// </summary>
        public static (double[] Gap, double[] TrendChange, double[] StepChange, double[]? GapChange)
            ReadThresholds(
                AnomalyGuardConfigFile file,
                out IReadOnlyList<string> problems)
        {
            ArgumentNullException.ThrowIfNull(file);

            var found = new List<string>();
            var gap = new double[(int)MetricIndex.Count];
            var trend = new double[(int)MetricIndex.Count];
            var step = new double[(int)MetricIndex.Count];
            var gapChange = new double[(int)MetricIndex.Count];
            var anyGapChange = false;

            foreach (var (key, entry) in file.Thresholds)
            {
                if (!Enum.TryParse<MetricIndex>(key, ignoreCase: true, out var metric)
                    || metric == MetricIndex.Count)
                {
                    found.Add($"Thresholds['{key}'] — not a known feature.");

                    continue;
                }

                gap[(int)metric] = Quantity($"Thresholds['{key}'].minGap", entry.MinGap, found);
                trend[(int)metric] =
                    Quantity($"Thresholds['{key}'].minTrendChange", entry.MinTrendChange, found);

                // Left at zero when absent, which ConfiguredFloorSource reads as "fall back to the trend
                // floor" — the behaviour every config had before this field existed.
                step[(int)metric] =
                    Quantity($"Thresholds['{key}'].minStepChange", entry.MinStepChange, found);

                // Positive, not merely present: Quantity returns zero for an absent key, for an explicit
                // "0" and for an unreadable value alike, and all three have to mean "not configured" here.
                // Counting any of them as a declaration would hand the novelty gate a full-length table of
                // zeros, which passes AnomalyGuard.RestoreNovelty's length check and then runs the gate on
                // the relative test alone.
                var change = Quantity($"Thresholds['{key}'].minGapChange", entry.MinGapChange, found);
                gapChange[(int)metric] = change;
                anyGapChange |= change > 0.0;
            }

            problems = found;

            return (gap, trend, step, anyGapChange ? gapChange : null);
        }

        /// <summary>
        /// The peer-novelty cadence profile the file names, or <c>null</c> when it names none.
        ///
        /// <para><b>Null is the whole compatibility story.</b> The gate is what makes the peer family
        /// <i>quieter</i>, so a file that says nothing must leave it off and every existing deployment
        /// untouched. There is no default profile to fall into and none is invented here — see
        /// <see cref="PeerNoveltyOptions"/>, which refuses <c>default</c> for the same reason.</para>
        ///
        /// <para><b>An unrecognised name is reported and dropped, not rounded.</b> The gate stays off, which
        /// leaves the guard noisy rather than silent — the safe direction for a typo — and the operator is
        /// told which names exist. Choosing the nearest profile would switch a suppression gate on from a
        /// misspelling.</para>
        /// </summary>
        public static PeerNoveltyOptions? ReadPeerNovelty(
            AnomalyGuardConfigFile file,
            out IReadOnlyList<string> problems)
        {
            ArgumentNullException.ThrowIfNull(file);

            var found = new List<string>();
            var name = file.PeerNovelty.Trim();

            if (name.Length == 0)
            {
                problems = found;

                return null;
            }

            if (TryNovelty(name, out var profile))
            {
                problems = found;

                return profile;
            }

            found.Add($"peerNovelty = '{name}' — not one of {NoveltyNames()}. The peer-novelty gate is OFF.");

            problems = found;

            return null;
        }

        /// <summary>
        /// The declared maintenance windows, with anything unparseable reported and dropped.
        ///
        /// <para><b>Dropped, never widened.</b> A window whose timestamps cannot be read must not become one
        /// that covers everything: a suppression that quietly applies for ever is total, invisible deafness,
        /// and it would arrive through a typo. The operator is told and the guard keeps watching.</para>
        /// </summary>
        public static IReadOnlyList<MaintenanceWindow> ReadMaintenance(
            AnomalyGuardConfigFile file, out IReadOnlyList<string> problems)
        {
            ArgumentNullException.ThrowIfNull(file);

            var found = new List<string>();
            var windows = new List<MaintenanceWindow>(file.Maintenance.Count);

            for (var i = 0; i < file.Maintenance.Count; i++)
            {
                var entry = file.Maintenance[i];

                if (!DateTimeOffset.TryParse(
                        entry.From, CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal, out var from))
                {
                    found.Add($"Maintenance[{i}].from = '{entry.From}' — not an ISO-8601 timestamp. "
                              + "The window is IGNORED.");

                    continue;
                }

                if (!DateTimeOffset.TryParse(
                        entry.To, CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal, out var to))
                {
                    found.Add($"Maintenance[{i}].to = '{entry.To}' — not an ISO-8601 timestamp. "
                              + "The window is IGNORED.");

                    continue;
                }

                if (to <= from)
                {
                    found.Add($"Maintenance[{i}] ends at or before it starts ({entry.From} .. {entry.To}). "
                              + "The window is IGNORED.");

                    continue;
                }

                windows.Add(new MaintenanceWindow(from, to, entry.Workload ?? string.Empty,
                    entry.Reason ?? string.Empty));
            }

            problems = found;

            return windows;
        }

        /// <summary>
        /// A pre-flight summary: what is mapped, what is not, and what could not be read. Meant to be printed
        /// before anything is deployed, because "this guard will be blind to these six things" is a decision
        /// an operator should make knowingly rather than discover from a cycle that saw nothing.
        /// </summary>
        public static string Describe(AnomalyGuardConfigFile file)
        {
            var map = ReadMap(file, out var mapProblems);
            ReadThresholds(file, out var thresholdProblems);
            ReadPeerNovelty(file, out var noveltyProblems);

            var report = new StringBuilder();
            report.Append(map.Describe());

            var problems = new List<string>(mapProblems);
            problems.AddRange(thresholdProblems);

            // A misspelled profile leaves the gate off, which is a decision an operator should see BEFORE
            // deploying rather than infer from the guard staying as noisy as it was.
            problems.AddRange(noveltyProblems);

            if (problems.Count == 0)
            {
                report.Append("\nevery entry in the file was understood\n");

                return report.ToString();
            }

            report.Append($"\n{problems.Count} entr(y/ies) could not be used and were DROPPED:\n");

            for (var i = 0; i < problems.Count; i++)
            {
                report.Append("  ").Append(problems[i]).Append('\n');
            }

            return report.ToString();
        }

        private static SustainedThresholdOptions? BuildRule(
            string key,
            AnomalyGuardConfigFile.CustomEntry entry,
            List<string> problems)
        {
            if (string.IsNullOrWhiteSpace(entry.RuleThreshold))
            {
                return null;
            }

            if (!MetricQuantity.TryParse(entry.RuleThreshold, out var threshold))
            {
                problems.Add($"CustomMetrics['{key}'].ruleThreshold = '{entry.RuleThreshold}' — not a "
                             + "number, optionally with a unit such as 100MB or 50ms.");

                return null;
            }

            return new SustainedThresholdOptions(
                threshold,
                entry.RuleMinBreachFraction,
                MinimumSamples: 20);
        }

        private static double Quantity(string where, string? text, List<string> problems)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                return 0.0;
            }

            if (MetricQuantity.TryParse(text, out var value))
            {
                return value;
            }

            // Dropped, not defaulted. A threshold that quietly became zero is a gate that quietly stopped
            // gating, and it would be found the hard way.
            problems.Add($"{where} = '{text}' — not a number, optionally with a unit such as 100MB, 50ms "
                         + "or 5%. The gate is OFF for this entry.");

            return 0.0;
        }

        private static bool TryKind(string text, out MetricSourceKind kind)
        {
            return Enum.TryParse(text, ignoreCase: true, out kind) && Enum.IsDefined(kind);
        }

        /// <summary>
        /// Maps a profile name onto the preset that carries it.
        ///
        /// <para><b>A switch rather than an enum parse, because these are static properties and not enum
        /// members.</b> The cost is that a fourth preset added to <see cref="PeerNoveltyOptions"/> would not
        /// become nameable here on its own, so a test walks the type's presets and fails if one of them
        /// cannot be reached through this method — the drift is caught by a gate rather than by a comment.
        /// </para>
        /// </summary>
        private static bool TryNovelty(string text, out PeerNoveltyOptions profile)
        {
            if (string.Equals(text, nameof(PeerNoveltyOptions.PerShift), StringComparison.OrdinalIgnoreCase))
            {
                profile = PeerNoveltyOptions.PerShift;

                return true;
            }

            if (string.Equals(text, nameof(PeerNoveltyOptions.Daily), StringComparison.OrdinalIgnoreCase))
            {
                profile = PeerNoveltyOptions.Daily;

                return true;
            }

            if (string.Equals(text, nameof(PeerNoveltyOptions.Weekly), StringComparison.OrdinalIgnoreCase))
            {
                profile = PeerNoveltyOptions.Weekly;

                return true;
            }

            profile = default;

            return false;
        }

        private static string NoveltyNames()
        {
            return nameof(PeerNoveltyOptions.PerShift)
                   + ", " + nameof(PeerNoveltyOptions.Daily)
                   + ", " + nameof(PeerNoveltyOptions.Weekly);
        }

        private static string KindNames()
        {
            return string.Join(", ", Enum.GetNames<MetricSourceKind>());
        }
    }
}
