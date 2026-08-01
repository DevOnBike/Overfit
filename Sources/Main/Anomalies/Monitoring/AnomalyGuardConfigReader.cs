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

                if (string.IsNullOrWhiteSpace(entry.Source))
                {
                    found.Add($"Metrics['{key}'].source is blank — omit the entry instead, which states "
                              + "plainly that this cluster does not have it.");

                    continue;
                }

                bindings.Add(new MetricBinding(metric, entry.Source.Trim(), kind, entry.Quantile));
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

                if (string.IsNullOrWhiteSpace(entry.Source))
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
                    Quantile: entry.Quantile,
                    Rule: BuildRule(key, entry, found)));
            }

            problems = found;

            return new MetricMap(bindings, custom);
        }

        /// <summary>
        /// The per-feature absolute floors, as the positional tables the guard takes. Built here so the
        /// indexed-by-enum shape stays an implementation detail rather than something a human has to write.
        /// </summary>
        public static (double[] Gap, double[] TrendChange) ReadThresholds(
            AnomalyGuardConfigFile file,
            out IReadOnlyList<string> problems)
        {
            ArgumentNullException.ThrowIfNull(file);

            var found = new List<string>();
            var gap = new double[(int)MetricIndex.Count];
            var trend = new double[(int)MetricIndex.Count];

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
            }

            problems = found;

            return (gap, trend);
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

            var report = new StringBuilder();
            report.Append(map.Describe());

            var problems = new List<string>(mapProblems);
            problems.AddRange(thresholdProblems);

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

        private static string KindNames()
        {
            return string.Join(", ", Enum.GetNames<MetricSourceKind>());
        }
    }
}
