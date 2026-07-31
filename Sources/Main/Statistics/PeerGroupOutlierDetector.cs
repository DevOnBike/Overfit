// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Statistics
{
    /// <summary>
    /// Finds the member of a peer group that behaves unlike the rest — "eleven pods sit at 400 MB, one sits at
    /// 1.7 GB" — for memory leaks confined to one replica, a damaged pod, uneven load balancing, a bad node or
    /// a noisy neighbour.
    ///
    /// <para><b>Why this needs no history.</b> The comparison is between members of the group <i>at the same
    /// instant</i>, not against the group's past, so it works from the first minute after installation. That is
    /// the property that makes it the right thing to ship first: every other detector in the family
    /// (seasonality, trend) has to wait days or weeks before it can say anything.</para>
    ///
    /// <para><b>Method: leave-one-out over a two-sample comparator.</b> Each peer's window is tested against
    /// the pooled windows of all the others, through <see cref="ITwoSampleComparer"/>. A rank comparison rather
    /// than a distance from the median, because it asks the question that actually matters — "is this peer's
    /// <i>distribution</i> shifted relative to its siblings" — without assuming a shape the data does not
    /// have, and because it yields a p-value and an effect size instead of a hand-tuned threshold.</para>
    ///
    /// <para><b>Multiple comparisons.</b> Two one-sided tests per peer means twelve pods give twenty-four
    /// chances to be unlucky, so the significance level is Bonferroni-corrected
    /// (<c>MaxPValue / (2 × peers)</c>). The leave-one-out tests are not independent — each peer appears in
    /// every other peer's baseline — but Bonferroni is valid under arbitrary dependence, merely conservative.</para>
    ///
    /// <para><b>Masking, and why both directions are tested.</b> With more than one deviating member the pooled
    /// baseline is itself contaminated, and the effect available to a deviating peer is bounded by the fraction
    /// of clean siblings — roughly <c>(n − k) / (n − 1)</c> for <c>k</c> deviants among <c>n</c> peers. A
    /// one-sided detector therefore <b>fails silently exactly where it matters most</b>: with eight of ten pods
    /// regressed, each one's effect falls under the threshold and the group reads as healthy.</para>
    ///
    /// <para>Testing the low side as well removes that failure. In the same scenario the two untouched pods
    /// stand out sharply <i>below</i> their peers, so the group is reported as deviating rather than healthy —
    /// and when members deviate in <b>both</b> directions at once, the group has no single norm and the answer
    /// is <see cref="DetectionStatus.Inconclusive"/> rather than a confident list.</para>
    ///
    /// <para>What remains genuinely undecidable here is attribution: a relative method has no external
    /// reference, so "eight members regressed" and "two members are unusually idle" produce the same evidence.
    /// The finding states the direction and stops there; resolving it needs the workload's own history, which
    /// is the trend/baseline detector's job (<see cref="TrendDetector"/>). The two are complementary precisely
    /// at this point.</para>
    ///
    /// <para><b>A member that cannot be compared is dropped, not fatal.</b> Requiring every peer to clear
    /// <see cref="PeerOutlierOptions.MinimumSamplesPerPeer"/> before anyone is evaluated makes the newest,
    /// emptiest pod in the group decide whether the group is looked at — and the failure is silent. See
    /// <c>Exclude</c> for the run that demonstrated it. The dropped members are counted in
    /// <see cref="PeerOutlierResult.ExcludedCount"/> so coverage cannot pass for health.</para>
    ///
    /// <para>Allocation-free apart from the caller's findings buffer: scratch comes from the shared pool.</para>
    /// </summary>
    public sealed class PeerGroupOutlierDetector
    {
        private readonly ITwoSampleComparer _comparer;

        /// <summary>Uses <see cref="MannWhitneyComparer.Instance"/>.</summary>
        public PeerGroupOutlierDetector()
            : this(MannWhitneyComparer.Instance)
        {
        }

        /// <param name="comparer">The two-sample test to run per peer.</param>
        public PeerGroupOutlierDetector(ITwoSampleComparer comparer)
        {
            ArgumentNullException.ThrowIfNull(comparer);
            _comparer = comparer;
        }

        /// <summary>Identifier for reports and exported metric labels.</summary>
        public string Name => "peer-group-outlier";

        /// <summary>
        /// Evaluates one peer group. Higher values must mean <i>worse</i> (memory, CPU, latency, error counts);
        /// for a signal where higher is better, negate the observations before calling — the same contract the
        /// underlying comparer states.
        /// </summary>
        /// <param name="peers">Group members. At least <see cref="PeerOutlierOptions.MinimumPeers"/> of them.</param>
        /// <param name="kind">Whether uneven load could explain a deviation; decides if
        /// <see cref="PeerSeries.Work"/> is mandatory.</param>
        /// <param name="options">Thresholds; use <see cref="PeerOutlierOptions.Balanced"/> rather than
        /// <c>default</c>.</param>
        /// <param name="findings">Caller-owned buffer receiving one finding per peer, index-aligned with
        /// <paramref name="peers"/>. Must be at least as long as the group. Populated for every status except
        /// the ones that reject the input outright.</param>
        public PeerOutlierResult Detect(
            IReadOnlyList<PeerSeries> peers,
            PeerSignalKind kind,
            PeerOutlierOptions options,
            PeerOutlierFinding[] findings)
        {
            ArgumentNullException.ThrowIfNull(peers);
            ArgumentNullException.ThrowIfNull(findings);

            if (!options.IsValid)
            {
                throw new ArgumentException(
                    "Thresholds are not usable — use PeerOutlierOptions.Balanced/Strict/FastFeedback rather than default.",
                    nameof(options));
            }

            if (findings.Length < peers.Count)
            {
                throw new ArgumentException("Findings buffer is shorter than the peer group.", nameof(findings));
            }

            if (peers.Count < options.MinimumPeers)
            {
                return Undecidable(
                    DetectionStatus.InsufficientData,
                    $"Peer group has {peers.Count} members; at least {options.MinimumPeers} are needed for one to be an outlier from the rest.",
                    peers.Count);
            }

            var rawTotal = 0;
            for (var i = 0; i < peers.Count; i++)
            {
                rawTotal += peers[i].Values.Length;
            }

            if (rawTotal == 0)
            {
                return Undecidable(DetectionStatus.InsufficientData, "No observations in the peer group.", peers.Count);
            }

            // One rental covers everything: the normalised observations of every peer laid end to end, plus a
            // workspace the same size in which each leave-one-out baseline is assembled.
            using var scratch = new PooledBuffer<double>(2 * rawTotal, clearMemory: false);
            using var boundsBuffer = new PooledBuffer<int>((2 * peers.Count) + 1, clearMemory: false);

            var pooled = scratch.Span[..rawTotal];
            var workspace = scratch.Span[rawTotal..];
            var bounds = boundsBuffer.Span[..(peers.Count + 1)];
            var map = boundsBuffer.Span.Slice(peers.Count + 1, peers.Count);

            var written = Normalise(peers, kind, pooled, bounds);
            var comparable = Exclude(peers, options.MinimumSamplesPerPeer, bounds, map, findings);
            var starved = peers.Count - comparable;

            if (comparable < options.MinimumPeers)
            {
                var detail = kind == PeerSignalKind.LoadSensitive
                    ? " On a load-sensitive signal a sample is only usable when the peer also reported the work behind it, so a missing or idle request-rate series starves the member here."
                    : string.Empty;

                return Undecidable(
                    DetectionStatus.InsufficientData,
                    $"{starved} of {peers.Count} member(s) contributed fewer than {options.MinimumSamplesPerPeer} usable samples and were dropped; {comparable} comparable member(s) remain and {options.MinimumPeers} are required.{detail}",
                    comparable,
                    starved);
            }

            if (starved > 0)
            {
                written = Compact(pooled, bounds, map, comparable);
            }

            // The size gate, measured before any test runs — see PeerOutlierOptions.MinRelativeGap for why a
            // rank effect size cannot stand in for it.
            using var summaryBuffer = new PooledBuffer<double>(3 * comparable, clearMemory: false);
            using var rawBuffer = new PooledBuffer<PeerDeviation>(comparable, clearMemory: true);

            var medians = summaryBuffer.Span[..comparable];
            var gaps = summaryBuffer.Span.Slice(comparable, comparable);
            var absolute = summaryBuffer.Span.Slice(2 * comparable, comparable);
            var rawDeviations = rawBuffer.Span[..comparable];

            for (var i = 0; i < comparable; i++)
            {
                var span = pooled[bounds[i]..bounds[i + 1]];
                span.CopyTo(workspace);
                medians[i] = MedianSelector.MedianInPlace(workspace[..span.Length]);
            }

            var departures = MeasureGaps(medians, gaps, absolute, workspace, options.MinRelativeGap);

            // Two one-sided tests per member, so the family is twice the group size. Dropped members ran no
            // test, so they must not inflate the correction — that would quietly raise the bar for everyone.
            var correctedAlpha = options.MaxPValue / (2.0 * comparable);
            var high = 0;
            var low = 0;
            var rawHigh = 0;
            var rawLow = 0;

            for (var i = 0; i < comparable; i++)
            {
                var start = bounds[i];
                var end = bounds[i + 1];

                // The rest of the group: everything before this peer, then everything after it.
                pooled[..start].CopyTo(workspace);
                pooled[end..written].CopyTo(workspace[start..]);

                var rest = workspace[..(start + (written - end))];
                var peer = pooled[start..end];

                // The size gate never changes which direction the rank test found, only whether that direction
                // is worth reporting — so the raw verdict is kept alongside as the evidence for "this group has
                // no norm".
                // Three gates, and a deviation must clear all of them: consistent (the rank test), large in
                // proportion (the relative gap), and large in the signal's own units (the absolute floor).
                // The third exists because the first two are dimensionless — see MinAbsoluteGap.
                var material = (options.MinRelativeGap <= 0.0 || gaps[i] >= options.MinRelativeGap)
                               && (options.MinAbsoluteGap <= 0.0 || absolute[i] >= options.MinAbsoluteGap);

                // Above its peers: peer as the candidate. Below: swap the arms, so the same one-sided test
                // answers the opposite question and the effect size stays a positive magnitude.
                var above = _comparer.Compare(rest, peer);
                var isHigh = above.IsRegression(correctedAlpha, options.MinEffectSize, options.MinimumSamplesPerPeer);

                if (isHigh)
                {
                    rawDeviations[i] = PeerDeviation.High;
                    rawHigh++;

                    findings[map[i]] = new PeerOutlierFinding(
                        peers[map[i]].Name, above, material ? PeerDeviation.High : PeerDeviation.None,
                        end - start, gaps[i], absolute[i]);

                    if (material)
                    {
                        high++;
                    }

                    continue;
                }

                var below = _comparer.Compare(peer, rest);
                var isLow = below.IsRegression(correctedAlpha, options.MinEffectSize, options.MinimumSamplesPerPeer);

                if (isLow)
                {
                    rawDeviations[i] = PeerDeviation.Low;
                    rawLow++;

                    findings[map[i]] = new PeerOutlierFinding(
                        peers[map[i]].Name, below, material ? PeerDeviation.Low : PeerDeviation.None,
                        end - start, gaps[i], absolute[i]);

                    if (material)
                    {
                        low++;
                    }

                    continue;
                }

                findings[map[i]] = new PeerOutlierFinding(
                    peers[map[i]].Name, above, PeerDeviation.None, end - start, gaps[i], absolute[i]);
            }

            // A third or more of the group standing away from the group's own centre is not one departure from
            // a norm — it is the absence of one, and the size gate must not be allowed to tidy that into a
            // confident list. Reported with the RAW directions, because members pulling both ways is the
            // evidence. Strict inequality so a single outlier among three peers still counts as an outlier.
            if (departures * 3 > comparable)
            {
                for (var i = 0; i < comparable; i++)
                {
                    findings[map[i]] = findings[map[i]] with
                    {
                        Deviation = rawDeviations[i]
                    };
                }

                return new PeerOutlierResult(
                    DetectionStatus.Inconclusive,
                    $"{departures} of {comparable} members sit more than {options.MinRelativeGap:P0} from the group's own median ({rawHigh} above and {rawLow} below their peers): the group has no coherent norm, which points to a workload-level change rather than an outlier. Compare against the workload's own history to attribute it.{Dropped(starved, peers.Count)}",
                    correctedAlpha,
                    comparable,
                    rawHigh,
                    rawLow,
                    starved);
            }

            if (high == 0 && low == 0)
            {
                return new PeerOutlierResult(
                    DetectionStatus.Healthy,
                    $"Every member is consistent with the rest of the group.{Dropped(starved, peers.Count)}",
                    correctedAlpha,
                    comparable,
                    0,
                    0,
                    starved);
            }

            // Members pulling in opposite directions, or a majority departing at once: "the rest" is no longer
            // a norm. Naming a list here would be a confident answer to a question the data cannot settle.
            if ((high > 0 && low > 0) || ((high + low) * 2 > comparable))
            {
                return new PeerOutlierResult(
                    DetectionStatus.Inconclusive,
                    $"{high} member(s) above and {low} below out of {comparable}: the group has no coherent norm, which points to a workload-level change rather than an outlier. Compare against the workload's own history to attribute it.{Dropped(starved, peers.Count)}",
                    correctedAlpha,
                    comparable,
                    high,
                    low,
                    starved);
            }

            return new PeerOutlierResult(
                DetectionStatus.Anomalous,
                $"{high + low} of {comparable} members deviate from their peers beyond noise ({high} above, {low} below).{Dropped(starved, peers.Count)}",
                correctedAlpha,
                comparable,
                high,
                low,
                starved);
        }

        /// <summary>
        /// Names the dropped members in a verdict, or says nothing when none were. Always appended, including
        /// to <see cref="DetectionStatus.Healthy"/> — "healthy" computed over six of ten replicas is a
        /// different statement from "healthy" over all ten, and the reader has to be able to tell.
        /// </summary>
        private static string Dropped(int starved, int total)
        {
            if (starved == 0)
            {
                return string.Empty;
            }

            return $" {starved} of {total} member(s) were dropped for contributing too few usable samples.";
        }

        /// <summary>
        /// Partitions the group into members that can be compared and members that cannot, writing the index
        /// of each comparable member into <paramref name="map"/> and returning how many there are. A dropped
        /// member still gets its finding — with its usable count, which is the number that explains it.
        ///
        /// <para><b>Dropping rather than aborting is the whole point, and it was measured.</b> The obvious
        /// reading of "every peer needs N samples" is that the group cannot be compared until they all have
        /// them, and that is what this did: the first member under the floor ended the evaluation for
        /// everyone. On the cluster lab, scaling one deployment from four replicas to eight made
        /// <b>nine of eleven metrics</b> return <see cref="DetectionStatus.InsufficientData"/> and the
        /// deliberately degraded replica became invisible — not because anything about it changed, but because
        /// the same traffic spread over twice as many pods left some of them with sparse latency quantiles.
        /// A newly created, restarting or lightly loaded pod would do the same thing at a client, and the
        /// symptom is silence, which reads exactly like health.</para>
        ///
        /// <para>The comparison the excluded member would have joined is still sound without it: the pooled
        /// baseline is other members' observations, so removing one shrinks the baseline rather than biasing
        /// it. What is lost is coverage of that member, which is reported rather than hidden — see
        /// <see cref="PeerOutlierResult.ExcludedCount"/>.</para>
        /// </summary>
        private static int Exclude(
            IReadOnlyList<PeerSeries> peers,
            int minimumSamples,
            ReadOnlySpan<int> bounds,
            Span<int> map,
            PeerOutlierFinding[] findings)
        {
            var comparable = 0;

            for (var i = 0; i < peers.Count; i++)
            {
                var usable = bounds[i + 1] - bounds[i];

                if (usable >= minimumSamples)
                {
                    map[comparable] = i;
                    comparable++;

                    continue;
                }

                // NaN rather than zero on both gaps: this member was never measured against anything, and a
                // zero would read as "sits exactly on its peers' median".
                findings[i] = new PeerOutlierFinding(
                    peers[i].Name, default, PeerDeviation.None, usable, double.NaN, double.NaN);
            }

            return comparable;
        }

        /// <summary>
        /// Squeezes the excluded members out of the normalised observations so the comparable ones are again
        /// contiguous, rewrites <paramref name="bounds"/> over them and returns the new total.
        ///
        /// <para>In place, and it can be: the map is ascending, so every span moves towards the front and
        /// <see cref="Span{T}.CopyTo"/> is memmove-safe for the overlap. The alternative was a second rental
        /// the size of the whole window.</para>
        /// </summary>
        private static int Compact(Span<double> pooled, Span<int> bounds, ReadOnlySpan<int> map, int comparable)
        {
            var packed = 0;

            for (var i = 0; i < comparable; i++)
            {
                var source = map[i];

                // Both reads happen before the write, which is what makes writing into the same buffer safe:
                // map is ascending, so bounds[i + 1] can only ever alias a slot already consumed.
                var start = bounds[source];
                var length = bounds[source + 1] - start;

                pooled.Slice(start, length).CopyTo(pooled[packed..]);
                packed += length;
                bounds[i + 1] = packed;
            }

            return packed;
        }

        /// <summary>
        /// Fills <paramref name="gaps"/> with each member's median distance from its peers', as a fraction of
        /// theirs, and returns how many members sit that far from the <b>group's</b> own median.
        ///
        /// <para><b>Medians of medians, not pooled samples.</b> A pooled baseline mixes distributions, so one
        /// deviating member drags the reference every other member is measured against; the median of the
        /// other members' medians barely moves. That difference is the whole reason this gate can be trusted
        /// at four replicas, which is an ordinary deployment size.</para>
        ///
        /// <para>A group centred on zero has no meaningful relative scale, so the gate stands down rather than
        /// dividing by something arbitrarily small — the same guard <see cref="TrendDetector"/> applies to its
        /// own relative-change threshold.</para>
        /// </summary>
        /// <param name="medians">One median per peer, index-aligned with the group.</param>
        /// <param name="gaps">Receives each peer's relative distance from its peers' centre.</param>
        /// <param name="absolute">Receives the same distance in the signal's own units.</param>
        /// <param name="scratch">At least <c>medians.Length</c> doubles; permuted.</param>
        /// <param name="minimumGap">The gate; zero disables it, and every gap is then reported as passing.</param>
        private static int MeasureGaps(
            ReadOnlySpan<double> medians,
            Span<double> gaps,
            Span<double> absolute,
            Span<double> scratch,
            double minimumGap)
        {
            var n = medians.Length;

            // The absolute distance is always measured, even when the relative gate is off: it is reported on
            // every finding, and the absolute gate may be in use on its own.
            for (var i = 0; i < n; i++)
            {
                var written = 0;
                for (var j = 0; j < n; j++)
                {
                    if (j == i)
                    {
                        continue;
                    }

                    scratch[written] = medians[j];
                    written++;
                }

                absolute[i] = Math.Abs(medians[i] - MedianSelector.MedianInPlace(scratch[..written]));
            }

            if (minimumGap <= 0.0)
            {
                gaps.Fill(double.PositiveInfinity);

                return 0;
            }

            for (var i = 0; i < n; i++)
            {
                var written = 0;
                for (var j = 0; j < n; j++)
                {
                    if (j == i)
                    {
                        continue;
                    }

                    scratch[written] = medians[j];
                    written++;
                }

                var centre = MedianSelector.MedianInPlace(scratch[..written]);
                gaps[i] = RelativeGap(medians[i], centre);
            }

            medians.CopyTo(scratch);
            var groupCentre = MedianSelector.MedianInPlace(scratch[..n]);

            // A group centred on zero has no relative scale, so "how many members sit relatively far from it"
            // has no answer — and the answer must be none, not all.
            //
            // Getting this backwards was measured, not imagined: the unscaled case yields an infinite gap,
            // which is the right reading for the per-finding gate ("cannot judge, so do not block") and
            // exactly the wrong one here. On the cluster lab it made four identically-zero counters —
            // container_oom_events_total, the 5xx ratio, GC pause and thread-pool queue — look like fully
            // split groups, so the detector reported "no coherent norm" about metrics on which nobody
            // disagreed, instead of Healthy.
            if (Math.Abs(groupCentre) <= 1e-12)
            {
                return 0;
            }

            var departures = 0;
            for (var i = 0; i < n; i++)
            {
                if (RelativeGap(medians[i], groupCentre) >= minimumGap)
                {
                    departures++;
                }
            }

            return departures;
        }

        /// <summary>
        /// <c>|value − centre| / |centre|</c>, or <see cref="double.PositiveInfinity"/> when the centre carries
        /// no usable scale — an unscaled group is one the gate cannot judge, so it does not block a finding.
        /// </summary>
        private static double RelativeGap(double value, double centre)
        {
            var scale = Math.Abs(centre);

            if (scale <= 1e-12)
            {
                return double.PositiveInfinity;
            }

            return Math.Abs(value - centre) / scale;
        }

        /// <summary>
        /// Copies every peer's usable observations into <paramref name="destination"/> end to end, dividing by
        /// work for load-sensitive signals, and records each peer's span in <paramref name="bounds"/> as a
        /// prefix sum. Returns the total written.
        /// </summary>
        private static int Normalise(
            IReadOnlyList<PeerSeries> peers,
            PeerSignalKind kind,
            Span<double> destination,
            Span<int> bounds)
        {
            var written = 0;
            bounds[0] = 0;

            for (var i = 0; i < peers.Count; i++)
            {
                var peer = peers[i];
                var values = peer.Values.Span;
                var work = peer.Work.Span;
                var loadSensitive = kind == PeerSignalKind.LoadSensitive;

                for (var s = 0; s < values.Length; s++)
                {
                    var value = values[s];
                    if (!double.IsFinite(value))
                    {
                        continue;
                    }

                    if (loadSensitive)
                    {
                        // A sample with no work behind it carries no cost-per-unit information; keeping it as a
                        // division by ~0 would manufacture a spectacular outlier out of an idle scrape.
                        if (s >= work.Length)
                        {
                            continue;
                        }

                        var units = work[s];
                        if (!double.IsFinite(units) || units <= 0.0)
                        {
                            continue;
                        }

                        value /= units;
                    }

                    destination[written] = value;
                    written++;
                }

                bounds[i + 1] = written;
            }

            return written;
        }

        private static PeerOutlierResult Undecidable(
            DetectionStatus status,
            string reason,
            int peerCount,
            int excluded = 0)
            => new(status, reason, 0.0, peerCount, 0, 0, excluded);
    }
}
