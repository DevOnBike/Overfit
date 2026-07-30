// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.Anomalies.Rules.Contracts;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Anomalies.Incidents.Contracts
{
    /// <summary>
    /// Everything <c>AnomalyGuard</c> needs that is not data: thresholds, topology, and the per-metric floors
    /// only the operator can supply.
    /// </summary>
    public sealed record AnomalyGuardOptions
    {
        /// <summary>Kubernetes namespace every subject is reported under.</summary>
        public string Namespace { get; init; } = string.Empty;

        /// <summary>
        /// Deployment or StatefulSet name. Used for the workload coordinate and as the subject of common-mode
        /// findings, which name no pod on purpose.
        /// </summary>
        public string Workload { get; init; } = string.Empty;

        /// <summary>Peer-comparison thresholds.</summary>
        public PeerOutlierOptions Peer { get; init; } = PeerOutlierOptions.Balanced;

        /// <summary>Trend thresholds.</summary>
        public TrendOptions Trend { get; init; } = TrendOptions.Balanced;

        /// <summary>How findings become incidents.</summary>
        public IncidentGroupingOptions Grouping { get; init; } = IncidentGroupingOptions.Balanced;

        /// <summary>
        /// Whether the trend family runs on each pod's <b>residual</b> against the group's common component,
        /// with the common component itself tested once at workload level.
        ///
        /// <para>On by default because the alternative is measurably worse and not in a subtle way: on the
        /// cluster lab, testing pods in isolation made ten of eleven healthy-replica findings the same falling
        /// latency trend on all three pods at once, during warm-up. Turn it off only for a group whose members
        /// are not expected to move together at all — at which point they are not really peers.</para>
        /// </summary>
        public bool DecomposeCommonMode { get; init; } = true;

        /// <summary>
        /// Smallest peer gap worth reporting, per metric, in that metric's own units. Indexed by
        /// <see cref="MetricIndex"/>; entries beyond its length, and a null table, mean the gate is off.
        ///
        /// <para><b>Nobody but the caller can fill this in.</b> One number cannot serve bytes, seconds, ratios
        /// and counts. Left empty, the largest single source of false peer findings on a healthy population
        /// was a GC pause difference of three tenths of a millisecond per second, which cleared a percentage
        /// gate because the metric's whole magnitude is 0.004.</para>
        /// </summary>
        public IReadOnlyList<double>? MinAbsoluteGap
        {
            get; init;
        }

        /// <summary>
        /// Smallest fitted trend change worth reporting, per metric, in that metric's own units. Same
        /// indexing and the same reasoning as <see cref="MinAbsoluteGap"/> — and the case it exists for is a
        /// series sitting at zero, where a relative gate has nothing to be relative to.
        /// </summary>
        public IReadOnlyList<double>? MinAbsoluteTrendChange
        {
            get; init;
        }

        /// <summary>
        /// Where each pod sits. <b>Supply this.</b> Without it the guard falls back to deriving the workload
        /// from the pod name, which is a heuristic that gets StatefulSets, Jobs and bare pods wrong — and
        /// getting it wrong is not cosmetic: the grouper scores <c>SameWorkload</c> at 0.7 against a 0.35
        /// threshold, so a wrong answer merges unrelated pods into one incident.
        /// </summary>
        public IPodTopology? PodTopology
        {
            get; init;
        }

        /// <summary>
        /// Metrics outside the modelled set that this deployment wants watched, keyed by the name they are
        /// reported under. Evaluated by the rules, peer and trend families; <b>not</b> by the learned one —
        /// <c>MetricSnapshot.FeatureCount</c> is a trained model's input contract and cannot move per
        /// deployment.
        /// </summary>
        public IReadOnlyList<CustomMetricBinding> CustomMetrics { get; init; } = [];

        /// <summary>
        /// How old a saved incident may be and still be adopted after a restart.
        ///
        /// <para><b>A bound is required.</b> A guard restarted after a week would otherwise resurrect
        /// week-old incidents and close them immediately, producing a burst of resolutions for problems
        /// nobody remembers — the alert storm the tracker exists to prevent, with the opposite sign.</para>
        /// </summary>
        public TimeSpan MaxRestoredIncidentAge { get; init; } = TimeSpan.FromHours(2);

        /// <summary>Thresholds the absolute rules run with; empty disables that family.</summary>
        public IReadOnlyList<RuleProfile> Rules { get; init; } = DefaultRules;

        /// <summary>
        /// The rules worth running out of the box, each for a reason the relative methods cannot cover.
        ///
        /// <para>CPU throttling because the CFS counters exist only on containers carrying a limit, so the
        /// peer group can hold exactly one member — on precisely the pod being throttled. OOM kills and
        /// restarts because a single event produces a non-zero rate over a small share of the window, which
        /// Cliff's delta scores below any usable effect size: <b>the peer detector is structurally blind to
        /// one OOMKill</b>, measured, and only a rule catches it.</para>
        /// </summary>
        public static IReadOnlyList<RuleProfile> DefaultRules { get; } =
        [
            new(MetricIndex.CpuThrottleRatio, SustainedThresholdOptions.ForCpuThrottling),
            new(MetricIndex.OomEventsRate, SustainedThresholdOptions.ForRareEvent),
            new(MetricIndex.ContainerRestarts, SustainedThresholdOptions.ForRareEvent),
        ];

        /// <summary>Looks up a per-metric floor, treating a short or absent table as "gate off".</summary>
        public static double FloorFor(IReadOnlyList<double>? table, MetricIndex metric)
        {
            var index = (int)metric;

            if (table is null || index < 0 || index >= table.Count)
            {
                return 0.0;
            }

            var value = table[index];

            return double.IsFinite(value) && value > 0.0 ? value : 0.0;
        }
    }
}
