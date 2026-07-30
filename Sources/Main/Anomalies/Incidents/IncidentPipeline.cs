// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;
using DevOnBike.Overfit.Anomalies.Incidents.Contracts;
using DevOnBike.Overfit.Anomalies.Rules;
using DevOnBike.Overfit.Anomalies.Rules.Contracts;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Anomalies.Incidents
{
    /// <summary>
    /// The bridge between the detectors and <see cref="IncidentGrouper"/>: takes the verdicts
    /// <see cref="TrendDetector"/> and <see cref="PeerGroupOutlierDetector"/> hand back, turns the ones that
    /// are actually detections into <see cref="SignalFinding"/>s, and groups them at the end of a cycle.
    ///
    /// <para><b>Severity is the effect size, never the p-value.</b> The two detectors answer different
    /// questions with different statistics, and their outputs have to be ordered against each other. A
    /// p-value cannot do that: it is a statement about evidence, and it shrinks with sample count, so a
    /// trivial difference measured over a long window outranks a large one measured over a short one. Effect
    /// sizes are already on a bounded, sample-count-independent scale — Kendall's tau for a trend, Cliff's
    /// delta for a peer comparison, both |·| ≤ 1 — so those are what severity carries. Significance is not
    /// discarded; it was the gate that let the finding exist at all.</para>
    ///
    /// <para><b>Only decided anomalies become findings.</b> <c>WarmingUp</c>, <c>InsufficientData</c> and
    /// <c>Inconclusive</c> are deliberately distinct from <c>Healthy</c> in this codebase, and collapsing any
    /// of them into a finding would report "we could not tell" as "something is wrong" — the single fastest
    /// way to make an alerting product untrustworthy.</para>
    ///
    /// <para><b>Peer findings carry no series, and that is not an omission.</b> Correlation exists in the
    /// grouper to link subjects that topology cannot reach. Peers are in one group by construction and
    /// normally share a workload, which already scores 0.7 — running a lag scan to rediscover a relationship
    /// the detector established by definition would be paying the expensive term for nothing.</para>
    ///
    /// <para>Not thread-safe. One instance per evaluation cycle, or <see cref="Clear"/> between cycles.</para>
    /// </summary>
    public sealed class IncidentPipeline
    {
        private readonly List<SignalFinding> _findings = [];
        private readonly IncidentGrouper _grouper = new();

        /// <summary>Findings accumulated so far this cycle.</summary>
        public int Count => _findings.Count;

        /// <summary>Discards everything, so the instance can serve the next cycle.</summary>
        public void Clear()
        {
            _findings.Clear();
        }

        /// <summary>
        /// Records a trend verdict, if it is one. Returns <c>false</c> for anything that is not a decided
        /// anomaly, which is the common case and not an error.
        /// </summary>
        /// <param name="subject">Who the series belongs to.</param>
        /// <param name="signal">Metric name — stable across subjects, so the grouper can count distinct signals.</param>
        /// <param name="result">What <see cref="TrendDetector.Detect"/> returned.</param>
        /// <param name="windowStart">Start of the evaluated window in wall-clock time.</param>
        /// <param name="windowEnd">End of it.</param>
        /// <param name="series">Optional observations, enabling correlation-based linking in the grouper.</param>
        /// <param name="signalClass">Overrides <see cref="SignalCatalog"/> classification when the caller
        /// knows better than a name can express.</param>
        public bool Observe(
            IncidentSubject subject,
            string signal,
            in TrendResult result,
            DateTimeOffset windowStart,
            DateTimeOffset windowEnd,
            ReadOnlyMemory<double> series = default,
            SignalClass? signalClass = null)
        {
            ArgumentNullException.ThrowIfNull(signal);

            if (result.Status != DetectionStatus.Anomalous)
            {
                return false;
            }

            RequireCapacity();

            _findings.Add(new SignalFinding(
                subject,
                signal,
                signalClass ?? SignalCatalog.Classify(signal),
                windowStart,
                windowEnd,
                Severity(result.KendallTau),
                result.Reason)
            {
                Series = series
            });

            return true;
        }

        /// <summary>
        /// Records a hard-rule verdict, if it is one. Returns <c>false</c> for anything that is not a decided
        /// anomaly.
        ///
        /// <para>This is the path for signals a comparison cannot reach. CPU throttling is the case that forced
        /// it: the CFS counters exist only on containers carrying a limit, so a peer group can contain exactly
        /// one member and the relative methods are undefined — on precisely the pod that is being throttled.</para>
        ///
        /// <para>Severity is the share of the window in breach rather than the height above the line, because
        /// crossing the line is what already decided the magnitude mattered. See
        /// <see cref="SustainedThresholdResult.Severity"/>.</para>
        /// </summary>
        /// <param name="subject">Who the signal belongs to.</param>
        /// <param name="signal">Metric name.</param>
        /// <param name="result">What <see cref="SustainedThresholdRule.Evaluate"/> returned.</param>
        /// <param name="windowStart">Start of the evaluated window in wall-clock time.</param>
        /// <param name="windowEnd">End of it.</param>
        /// <param name="series">Optional observations, enabling correlation-based linking in the grouper.</param>
        /// <param name="signalClass">Overrides <see cref="SignalCatalog"/> classification.</param>
        public bool ObserveRule(
            IncidentSubject subject,
            string signal,
            in SustainedThresholdResult result,
            DateTimeOffset windowStart,
            DateTimeOffset windowEnd,
            ReadOnlyMemory<double> series = default,
            SignalClass? signalClass = null)
        {
            ArgumentNullException.ThrowIfNull(signal);

            if (result.Status != DetectionStatus.Anomalous)
            {
                return false;
            }

            RequireCapacity();

            _findings.Add(new SignalFinding(
                subject,
                signal,
                signalClass ?? SignalCatalog.Classify(signal),
                windowStart,
                windowEnd,
                result.Severity,
                result.Reason)
            {
                Series = series
            });

            return true;
        }

        /// <summary>
        /// Records every deviating member of a peer group. Returns how many findings were added.
        /// </summary>
        /// <param name="signal">Metric name the group was compared on.</param>
        /// <param name="result">What <see cref="PeerGroupOutlierDetector.Detect"/> returned for the group.</param>
        /// <param name="findings">The per-peer buffer the detector filled, <b>index-aligned with the peer
        /// list that was passed to it</b> — the detector writes one entry per peer, in order.</param>
        /// <param name="subjects">Topology for those same peers, in the same order.</param>
        /// <param name="windowStart">Start of the evaluated window.</param>
        /// <param name="windowEnd">End of it.</param>
        /// <param name="signalClass">Overrides <see cref="SignalCatalog"/> classification.</param>
        public int ObservePeerGroup(
            string signal,
            in PeerOutlierResult result,
            ReadOnlySpan<PeerOutlierFinding> findings,
            ReadOnlySpan<IncidentSubject> subjects,
            DateTimeOffset windowStart,
            DateTimeOffset windowEnd,
            SignalClass? signalClass = null)
        {
            ArgumentNullException.ThrowIfNull(signal);

            if (findings.Length != subjects.Length)
            {
                throw new ArgumentException(
                    $"{findings.Length} findings against {subjects.Length} subjects — the two must be "
                    + "index-aligned with the peer list the detector was given.",
                    nameof(subjects));
            }

            if (result.Status != DetectionStatus.Anomalous)
            {
                return 0;
            }

            var resolvedClass = signalClass ?? SignalCatalog.Classify(signal);
            var added = 0;

            for (var i = 0; i < findings.Length; i++)
            {
                if (!findings[i].IsOutlier)
                {
                    continue;
                }

                RequireCapacity();

                _findings.Add(new SignalFinding(
                    subjects[i],
                    signal,
                    resolvedClass,
                    windowStart,
                    windowEnd,
                    Severity(findings[i].Comparison.EffectSize),
                    Describe(findings[i], result)));

                added++;
            }

            return added;
        }

        /// <summary>
        /// Groups everything recorded this cycle. Does not clear — call <see cref="Clear"/> when the result
        /// has been consumed, so a caller can group under more than one set of thresholds.
        /// </summary>
        public IReadOnlyList<Incident> Group(IncidentGroupingOptions options)
        {
            // The grouper only reads, so handing it the list's backing store beats copying the batch.
            return _grouper.Group(CollectionsMarshal.AsSpan(_findings), options);
        }

        /// <summary>
        /// Effect sizes arrive on −1…+1 and severity is a magnitude, so the sign is dropped here rather than
        /// at each call site. Non-finite input is treated as no evidence rather than propagated into an
        /// ordering, where a NaN would make the comparison non-transitive.
        /// </summary>
        private static double Severity(double effectSize)
        {
            if (!double.IsFinite(effectSize))
            {
                return 0.0;
            }

            return Math.Clamp(Math.Abs(effectSize), 0.0, 1.0);
        }

        private void RequireCapacity()
        {
            if (_findings.Count < IncidentGrouper.MaxFindingsPerCall)
            {
                return;
            }

            throw new InvalidOperationException(
                $"Pipeline already holds {IncidentGrouper.MaxFindingsPerCall} findings, the per-call grouping "
                + "bound. A cycle this large means a detector has stopped filtering; raise its thresholds or "
                + "group in batches rather than scoring half a million pairs.");
        }

        /// <summary>
        /// A peer finding has no reason of its own — the detector reports the group's verdict and each
        /// member's standing separately — so one is composed here from the numbers that decided it.
        /// </summary>
        private static string Describe(in PeerOutlierFinding finding, in PeerOutlierResult result)
        {
            var direction = finding.Deviation == PeerDeviation.High ? "above" : "below";
            var peers = result.PeerCount - 1;
            var comparison = finding.Comparison;

            // Both directions are reported, and the low side is not a curiosity: measured on the cluster lab,
            // a CPU-throttled replica showed the *lowest* cost per request of the four, because it served
            // fewer requests while its fixed overhead stayed put. A detector that only looked upward would
            // have called the broken pod the cheapest one.
            // The relative gap leads, because it is the only one of the three numbers that answers "by how
            // much". Cliff's delta saturates on any well-separated pair and says nothing about size.
            // Both, because a percentage alone hides "14% of a 0.004 ratio" and an absolute figure alone
            // hides how unusual it is for this group.
            var size = double.IsFinite(finding.RelativeGap)
                ? $"by {finding.RelativeGap:P0} ({finding.AbsoluteGap:G3})"
                : $"by {finding.AbsoluteGap:G3} (the peers' median is zero, so there is no proportion to take)";

            return $"'{finding.Name}' sits {direction} the other {peers} peers {size}: "
                   + $"Cliff's delta {Math.Abs(comparison.EffectSize):F2}, p {comparison.PValueCandidateWorse:G3} "
                   + $"against a Bonferroni-corrected alpha of {result.CorrectedAlpha:G3} "
                   + $"({finding.UsableSamples} usable samples).";
        }
    }
}
