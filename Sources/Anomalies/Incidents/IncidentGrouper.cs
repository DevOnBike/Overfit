// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Statistics;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Anomalies.Incidents
{
    /// <summary>
    /// Collapses a batch of findings into incidents — the layer that decides whether an operator sees one
    /// problem or eleven alerts.
    ///
    /// <para><b>Why this and not connected components.</b> The obvious implementation links any two related
    /// findings and takes the connected components, and it fails the same way every time: relatedness is not
    /// transitive, so a chain of individually-plausible links walks an incident across the cluster until
    /// everything is one incident called "something is wrong". The fix here is Kruskal-style agglomeration —
    /// consider the strongest links first, and refuse any merge that would push the group's total span past
    /// <see cref="IncidentGroupingOptions.MaxIncidentSpan"/>. Strong links get to form the group; weak ones
    /// only ever extend it within a bound that was stated up front.</para>
    ///
    /// <para><b>Three sources of evidence, multiplied rather than added.</b> Time proximity and cluster
    /// topology both have to hold — two findings a millisecond apart on unrelated workloads are not one
    /// incident, and neither are two findings on one pod six hours apart. Rank correlation is the third, and
    /// it earns its keep on the case topology cannot reach: pods that share no owner and no node but whose
    /// series move together, which is how a shared dependency shows itself.</para>
    ///
    /// <para><b>What this does not do.</b> It does not establish causation — see <see cref="SignalClass"/>.
    /// It does not carry state between calls, so it cannot track an incident across evaluation cycles; that
    /// belongs to whatever owns incident identity and lifecycle, and mixing the two here would make both
    /// untestable.</para>
    ///
    /// <para>The pairwise scoring runs on pooled scratch. The returned graph of incidents is ordinary managed
    /// objects: it is the product of the call and outlives it, so there is nothing to pool.</para>
    /// </summary>
    public sealed class IncidentGrouper
    {
        /// <summary>
        /// Ceiling on findings per call. Scoring is O(n²) in pairs, and 1024 findings is already ~524 000 of
        /// them; a batch larger than this is a symptom of a detector that has stopped filtering, and silently
        /// spending minutes on it would hide that. Named rather than implicit, per the reliability rules.
        ///
        /// <para><b>Measured, so the bound is a number rather than a feeling</b> (<c>IncidentGroupingBenchmark</c>,
        /// worst-case topology where every pair reaches the correlation step, 120-sample series):</para>
        /// <list type="table">
        /// <item><term>16 findings</term><description>2.7 µs without correlation, 549 µs with</description></item>
        /// <item><term>64 findings</term><description>20 µs / 8.2 ms</description></item>
        /// <item><term>256 findings</term><description>204 µs / 172 ms</description></item>
        /// </list>
        /// <para>Correlation costs three orders of magnitude and grows as O(n²) on top of that, so at this
        /// ceiling the worst case is measured in seconds. Real batches are far cheaper — replicas share a
        /// workload, which scores 0.7 and short-circuits the expensive term before it runs — but a caller
        /// feeding hundreds of topologically unrelated findings should either raise
        /// <see cref="IncidentGroupingOptions.MinRelatedness"/> or use
        /// <see cref="IncidentGroupingOptions.Strict"/>, which turns correlation off entirely.</para>
        /// </summary>
        public const int MaxFindingsPerCall = 1024;

        /// <summary>Identifier for reports and exported metric labels.</summary>
        public string Name => "incident-grouper";

        /// <summary>
        /// Groups findings into incidents, strongest links first.
        /// </summary>
        /// <param name="findings">The batch. Order does not affect the result.</param>
        /// <param name="options">Thresholds; use <see cref="IncidentGroupingOptions.Balanced"/> rather than
        /// <c>default</c>.</param>
        /// <returns>Incidents ordered by peak severity, most severe first. A finding that links to nothing
        /// becomes an incident of one — dropping it would be losing a detection to make the output tidier.</returns>
        public IReadOnlyList<Incident> Group(
            ReadOnlySpan<SignalFinding> findings,
            IncidentGroupingOptions options)
        {
            if (!options.IsValid)
            {
                throw new ArgumentException(
                    "Thresholds are not usable — use IncidentGroupingOptions.Balanced/Strict rather than default.",
                    nameof(options));
            }

            if (findings.Length > MaxFindingsPerCall)
            {
                throw new ArgumentException(
                    $"{findings.Length} findings exceeds the {MaxFindingsPerCall} per-call bound; filter or " +
                    "batch upstream rather than scoring half a million pairs.",
                    nameof(findings));
            }

            if (findings.IsEmpty)
            {
                return [];
            }

            var n = findings.Length;

            using var parentBuffer = new PooledBuffer<int>(n, clearMemory: false);
            using var startBuffer = new PooledBuffer<long>(n, clearMemory: false);
            using var endBuffer = new PooledBuffer<long>(n, clearMemory: false);

            var parent = parentBuffer.Span[..n];
            var componentStart = startBuffer.Span[..n];
            var componentEnd = endBuffer.Span[..n];

            for (var i = 0; i < n; i++)
            {
                parent[i] = i;
                componentStart[i] = findings[i].Start.UtcTicks;
                componentEnd[i] = findings[i].End.UtcTicks;
            }

            Agglomerate(findings, options, parent, componentStart, componentEnd);

            return Build(findings, parent);
        }

        /// <summary>
        /// Scores every pair, then merges in descending order of strength subject to the span bound.
        /// </summary>
        private static void Agglomerate(
            ReadOnlySpan<SignalFinding> findings,
            IncidentGroupingOptions options,
            Span<int> parent,
            Span<long> componentStart,
            Span<long> componentEnd)
        {
            var n = findings.Length;
            var capacity = n * (n - 1) / 2;

            if (capacity == 0)
            {
                return;
            }

            using var weightBuffer = new PooledBuffer<double>(capacity, clearMemory: false);
            using var pairBuffer = new PooledBuffer<long>(capacity, clearMemory: false);

            var weights = weightBuffer.Span;
            var pairs = pairBuffer.Span;
            var edges = 0;

            for (var i = 0; i < n - 1; i++)
            {
                for (var j = i + 1; j < n; j++)
                {
                    var weight = Relatedness(findings[i], findings[j], options);

                    if (weight < options.MinRelatedness)
                    {
                        continue;
                    }

                    weights[edges] = weight;
                    pairs[edges] = ((long)i << 32) | (uint)j;
                    edges++;
                }
            }

            if (edges == 0)
            {
                return;
            }

            // Ascending, then walked backwards: strongest link first, which is what keeps a weak link from
            // deciding the shape of a group that a strong one would have formed differently.
            weights[..edges].Sort(pairs[..edges]);

            var maxSpanTicks = options.MaxIncidentSpan.Ticks;

            for (var e = edges - 1; e >= 0; e--)
            {
                var packed = pairs[e];
                var left = Find(parent, (int)(packed >> 32));
                var right = Find(parent, (int)(packed & 0xFFFFFFFF));

                if (left == right)
                {
                    continue;
                }

                var mergedStart = Math.Min(componentStart[left], componentStart[right]);
                var mergedEnd = Math.Max(componentEnd[left], componentEnd[right]);

                if (mergedEnd - mergedStart > maxSpanTicks)
                {
                    continue;
                }

                parent[right] = left;
                componentStart[left] = mergedStart;
                componentEnd[left] = mergedEnd;
            }
        }

        /// <summary>
        /// Union-find root with path halving. Iterative on purpose: OVERFIT022 forbids recursion, and the
        /// depth here is data-driven, which is exactly the case the rule exists for.
        /// </summary>
        private static int Find(Span<int> parent, int index)
        {
            while (parent[index] != index)
            {
                parent[index] = parent[parent[index]];
                index = parent[index];
            }

            return index;
        }

        /// <summary>
        /// Evidence that two findings describe one event, on 0…1. Time and topology are multiplied because
        /// each is necessary: a perfect topological match hours apart is not one incident, and simultaneity
        /// across unrelated workloads is coincidence.
        /// </summary>
        private static double Relatedness(
            in SignalFinding first,
            in SignalFinding second,
            IncidentGroupingOptions options)
        {
            var temporal = TemporalScore(first, second, options.MaxSeparation);

            if (temporal <= 0.0)
            {
                return 0.0;
            }

            var topology = TopologyScore(first.Subject, second.Subject, options.Topology);

            // Correlation is only consulted when topology cannot already justify the merge — it is the
            // expensive term, and computing it to confirm what is already known would be waste.
            if (topology < options.Topology.Correlated
                && options.MinCorrelation < 1.0
                && IsCorrelated(first, second, options))
            {
                topology = options.Topology.Correlated;
            }

            return temporal * topology;
        }

        /// <summary>
        /// 1.0 for overlapping windows, decaying linearly to 0 across <paramref name="maxSeparation"/>.
        /// </summary>
        private static double TemporalScore(
            in SignalFinding first,
            in SignalFinding second,
            TimeSpan maxSeparation)
        {
            var laterStart = first.Start > second.Start ? first.Start : second.Start;
            var earlierEnd = first.End < second.End ? first.End : second.End;
            var gap = laterStart - earlierEnd;

            if (gap <= TimeSpan.Zero)
            {
                return 1.0;
            }

            if (gap >= maxSeparation)
            {
                return 0.0;
            }

            return 1.0 - (gap.TotalSeconds / maxSeparation.TotalSeconds);
        }

        /// <summary>
        /// The closest relationship the two subjects have in the cluster. Empty coordinates never match —
        /// two findings with no known node are not "on the same node".
        /// </summary>
        private static double TopologyScore(
            in IncidentSubject first,
            in IncidentSubject second,
            TopologyWeights weights)
        {
            if (Matches(first.Pod, second.Pod))
            {
                return weights.SamePod;
            }

            // Between pod and workload on purpose: same ReplicaSet is same software, same workload during a
            // rollout is not.
            if (Matches(first.ReplicaSet, second.ReplicaSet))
            {
                return weights.SameReplicaSet;
            }

            if (Matches(first.Workload, second.Workload))
            {
                return weights.SameWorkload;
            }

            if (Matches(first.Node, second.Node))
            {
                return weights.SameNode;
            }

            if (Matches(first.Namespace, second.Namespace))
            {
                return weights.SameNamespace;
            }

            return 0.0;
        }

        private static bool Matches(string first, string second)
            => first.Length > 0 && string.Equals(first, second, StringComparison.Ordinal);

        /// <summary>
        /// Whether the two findings' series move together strongly enough to stand in for topology. Requires
        /// both series on a common grid: without timestamps there is no honest way to align series of
        /// different lengths, and guessing would invent a lag rather than measure one.
        /// </summary>
        private static bool IsCorrelated(
            in SignalFinding first,
            in SignalFinding second,
            IncidentGroupingOptions options)
        {
            var a = first.Series.Span;
            var b = second.Series.Span;

            if (a.IsEmpty || b.IsEmpty || a.Length != b.Length)
            {
                return false;
            }

            var correlation = options.MaxLagSamples > 0
                ? SpearmanCorrelation.CorrelateWithLag(a, b, options.MaxLagSamples)
                : SpearmanCorrelation.Correlate(a, b);

            return correlation.IsSignificant(options.MinCorrelation, options.MaxCorrelationPValue);
        }

        /// <summary>Materialises components into incidents, ordered most severe first.</summary>
        private static IReadOnlyList<Incident> Build(ReadOnlySpan<SignalFinding> findings, Span<int> parent)
        {
            var n = findings.Length;
            var groups = new Dictionary<int, List<SignalFinding>>();

            for (var i = 0; i < n; i++)
            {
                var root = Find(parent, i);

                if (!groups.TryGetValue(root, out var members))
                {
                    members = [];
                    groups[root] = members;
                }

                members.Add(findings[i]);
            }

            var incidents = new List<Incident>(groups.Count);

            foreach (var members in groups.Values)
            {
                incidents.Add(Assemble(members));
            }

            SortBySeverityDescending(incidents);

            return incidents;
        }

        private static Incident Assemble(List<SignalFinding> members)
        {
            SortCauseFirst(members);

            var start = members[0].Start;
            var end = members[0].End;

            var subjects = new HashSet<string>(StringComparer.Ordinal);
            var signals = new HashSet<string>(StringComparer.Ordinal);

            for (var i = 0; i < members.Count; i++)
            {
                var member = members[i];

                if (member.Start < start)
                {
                    start = member.Start;
                }

                if (member.End > end)
                {
                    end = member.End;
                }

                subjects.Add(member.Subject.Label);
                signals.Add(member.Signal);
            }

            var primary = members[0];
            var summary = Describe(primary, members.Count, subjects.Count, signals.Count);

            return new Incident(members, primary, start, end, subjects.Count, signals.Count, summary);
        }

        /// <summary>
        /// Orders findings cause-first: infrastructure before resource before symptom, then earliest, then
        /// most severe. Insertion sort because a component holds a handful of findings and a clear comparison
        /// is worth more here than an asymptotic one.
        /// </summary>
        private static void SortCauseFirst(List<SignalFinding> members)
        {
            for (var i = 1; i < members.Count; i++)
            {
                var current = members[i];
                var j = i - 1;

                while (j >= 0 && ComesAfter(members[j], current))
                {
                    members[j + 1] = members[j];
                    j--;
                }

                members[j + 1] = current;
            }
        }

        private static bool ComesAfter(in SignalFinding candidate, in SignalFinding reference)
        {
            if (candidate.Class != reference.Class)
            {
                return candidate.Class > reference.Class;
            }

            if (candidate.Start != reference.Start)
            {
                return candidate.Start > reference.Start;
            }

            return candidate.Severity < reference.Severity;
        }

        /// <summary>
        /// Most severe first, tie-broken by start and then by the primary signal's name. The tie-breaks are
        /// not cosmetic: without them the output order falls back on component-discovery order, which shifts
        /// when the input is permuted, and a report that reorders itself between two runs over the same data
        /// is one nobody can diff.
        /// </summary>
        private static void SortBySeverityDescending(List<Incident> incidents)
        {
            for (var i = 1; i < incidents.Count; i++)
            {
                var current = incidents[i];
                var j = i - 1;

                while (j >= 0 && RanksBelow(incidents[j], current))
                {
                    incidents[j + 1] = incidents[j];
                    j--;
                }

                incidents[j + 1] = current;
            }
        }

        private static bool RanksBelow(Incident candidate, Incident reference)
        {
            if (candidate.PeakSeverity != reference.PeakSeverity)
            {
                return candidate.PeakSeverity < reference.PeakSeverity;
            }

            if (candidate.Start != reference.Start)
            {
                return candidate.Start > reference.Start;
            }

            return string.CompareOrdinal(candidate.Primary.Signal, reference.Primary.Signal) > 0;
        }

        /// <summary>
        /// One line naming the thing to look at and the size of what it explains. Built once per incident,
        /// which is once per evaluation cycle — this is reporting code, not a hot path.
        /// </summary>
        private static string Describe(in SignalFinding primary, int findingCount, int subjects, int signals)
        {
            var text = new StringBuilder(160);

            text.Append(primary.Signal)
                .Append(" on ")
                .Append(primary.Subject.Label);

            if (findingCount == 1)
            {
                return text.Append(" — ").Append(primary.Reason).ToString();
            }

            text.Append(" — ")
                .Append(primary.Reason)
                .Append(" (+")
                .Append(findingCount - 1)
                .Append(findingCount == 2 ? " related finding across " : " related findings across ")
                .Append(subjects)
                .Append(subjects == 1 ? " subject, " : " subjects, ")
                .Append(signals)
                .Append(signals == 1 ? " signal)" : " signals)");

            return text.ToString();
        }
    }
}
