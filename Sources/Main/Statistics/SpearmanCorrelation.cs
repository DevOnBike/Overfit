// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Statistics
{
    /// <summary>
    /// Rank correlation between two series, with an optional lag scan — the primitive that turns a pile of
    /// separate alerts into "memory started climbing, and latency followed ninety seconds later".
    ///
    /// <para><b>Rank, not Pearson.</b> Same argument as everywhere else in this namespace: monitoring series
    /// carry scrape spikes, restarts and saturation, and a single outlier can move a Pearson coefficient from
    /// 0.1 to 0.9. Ranking bounds any one sample's influence to one rank position. It also makes the
    /// coefficient invariant to monotone rescaling, so bytes against seconds needs no normalisation.</para>
    ///
    /// <para><b>The lag scan is where honesty costs something.</b> Cause precedes effect, so the useful
    /// question is not "do these move together" but "does one lead the other, and by how much". Scanning
    /// 2·L+1 offsets and keeping the strongest is a multiple-comparison procedure: with 21 offsets and pure
    /// noise, the best of them clears p &lt; 0.05 about two-thirds of the time. The reported p-value is
    /// therefore Bonferroni-corrected by the number of offsets actually evaluated. Uncorrected, this routine
    /// would manufacture a causal story for every pair of unrelated metrics in the cluster — which is exactly
    /// the failure mode that makes correlation engines untrustworthy.</para>
    ///
    /// <para><b>Correlation here is evidence of shared cause, never proof of one.</b> Two pods on a saturated
    /// node correlate perfectly and neither causes the other. The lag sign narrows the candidates; it does not
    /// settle them.</para>
    ///
    /// <para><b>Cost, because this sits under an O(N²) caller.</b> Measured in
    /// <c>SpearmanCorrelationBenchmark</c>, zero allocations throughout:</para>
    /// <list type="table">
    /// <item><term>60 samples</term><description>579 ns unlagged, 2.3 µs across 21 offsets</description></item>
    /// <item><term>120 samples</term><description>1.14 µs / 4.9 µs</description></item>
    /// <item><term>300 samples</term><description>3.34 µs / 12.4 µs</description></item>
    /// </list>
    /// <para>The scan costs roughly 4x the single coefficient rather than the 21x the offset count suggests,
    /// because ranking is hoisted out of the loop and only the linear pass repeats. Doing it the other way —
    /// re-ranking inside every offset, preserved as
    /// <see cref="CorrelateWithLagPerWindowRanks"/> for the benchmark that settles it — costs 21x at 60
    /// samples and 53x at 300, which turned a 256-finding grouping cycle into 1.66 seconds.</para>
    ///
    /// <para>Scratch is pooled; nothing lands on the GC heap.</para>
    /// </summary>
    public static class SpearmanCorrelation
    {
        /// <summary>
        /// Below this many overlapping observations the Fisher transform has no business being applied and the
        /// coefficient is dominated by whichever handful of points happened to arrive.
        /// </summary>
        public const int MinimumSamples = 8;

        /// <summary>Ceiling on the lag scan, so a caller cannot ask for a shift that consumes the window.</summary>
        public const int MaxLagSamples = 128;

        /// <summary>
        /// Correlates two index-aligned series with no lag. Samples where either series is non-finite are
        /// dropped from both — pairwise-complete, so one gap does not discard the whole window.
        /// </summary>
        public static CorrelationResult Correlate(ReadOnlySpan<double> first, ReadOnlySpan<double> second)
        {
            if (first.Length != second.Length)
            {
                throw new ArgumentException("Series must be index-aligned.", nameof(second));
            }

            var n = first.Length;
            if (n == 0)
            {
                return CorrelationResult.Undecidable(0);
            }

            using var scratch = new PooledBuffer<double>(3 * n, clearMemory: false);
            using var order = new PooledBuffer<int>(n, clearMemory: false);

            return CorrelateCore(first, second, 0, 1, scratch.Span, order.Span);
        }

        /// <summary>
        /// Scans lags in −<paramref name="maxLagSamples"/>…+<paramref name="maxLagSamples"/> and returns the
        /// strongest, with the p-value corrected for how many offsets were tried.
        /// </summary>
        /// <param name="first">Candidate leader.</param>
        /// <param name="second">Candidate follower.</param>
        /// <param name="maxLagSamples">Largest shift to consider, in samples. A positive
        /// <see cref="CorrelationResult.LagSamples"/> in the result means <paramref name="first"/> leads.</param>
        public static CorrelationResult CorrelateWithLag(
            ReadOnlySpan<double> first,
            ReadOnlySpan<double> second,
            int maxLagSamples)
        {
            if (first.Length != second.Length)
            {
                throw new ArgumentException("Series must be index-aligned.", nameof(second));
            }

            if (maxLagSamples < 0 || maxLagSamples > MaxLagSamples)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(maxLagSamples), maxLagSamples, $"Lag must be in 0…{MaxLagSamples} samples.");
            }

            var n = first.Length;
            if (n == 0)
            {
                return CorrelationResult.Undecidable(0);
            }

            // A shift only makes sense while enough of the two windows still overlap to test.
            var usableLag = Math.Min(maxLagSamples, n - MinimumSamples);
            if (usableLag < 0)
            {
                return CorrelationResult.Undecidable(n);
            }

            using var scratch = new PooledBuffer<double>(3 * n, clearMemory: false);
            using var order = new PooledBuffer<int>(n, clearMemory: false);

            var ranksFirst = scratch.Span.Slice(0, n);
            var ranksSecond = scratch.Span.Slice(n, n);
            var keys = scratch.Span.Slice(2 * n, n);

            // Rank each series ONCE over the whole window, then slide. Two reasons, and the cheaper one is
            // the less important:
            //
            //   Comparability. Re-ranking inside each offset's overlap gives every offset its own rank scale,
            //   so picking "the strongest" compares coefficients that were never on the same footing — and a
            //   short overlap at an extreme lag produces inflated coefficients for free. One ranking for all
            //   offsets is what makes the scan's argmax mean anything.
            //
            //   Cost. The alternative re-sorts both series 2L+1 times. Measured at 120 samples and L=10 that
            //   is 24 us per pair, which the incident grouper multiplies by O(N^2) pairs: 1.66 SECONDS for a
            //   256-finding batch. This version sorts twice and then walks linearly per offset.
            if (RankFinite(first, ranksFirst, keys, order.Span) == 0
                || RankFinite(second, ranksSecond, keys, order.Span) == 0)
            {
                return CorrelationResult.Undecidable(n);
            }

            var evaluated = 0;
            var best = CorrelationResult.Undecidable(n);

            for (var lag = -usableLag; lag <= usableLag; lag++)
            {
                var firstStart = lag < 0 ? -lag : 0;
                var secondStart = lag > 0 ? lag : 0;
                var overlap = n - Math.Abs(lag);

                var rho = PearsonSkippingGaps(
                    ranksFirst.Slice(firstStart, overlap),
                    ranksSecond.Slice(secondStart, overlap),
                    out var kept);

                if (kept < MinimumSamples || double.IsNaN(rho))
                {
                    continue;
                }

                evaluated++;

                if (Math.Abs(rho) > best.Strength)
                {
                    best = new CorrelationResult(rho, FisherPValue(rho, kept, 1), kept, lag);
                }
            }

            if (evaluated == 0)
            {
                return CorrelationResult.Undecidable(n);
            }

            // Corrected by the offsets that actually produced a verdict, not by the range requested — a short
            // window should not be penalised for offsets it never got to try.
            return best with
            {
                PValue = Math.Min(1.0, best.PValue * evaluated)
            };
        }

        /// <summary>
        /// The per-offset re-ranking this class used to do, kept for the benchmark that justifies not doing
        /// it. Exact Spearman inside every overlap, at 2L+1 pairs of sorts.
        /// </summary>
        internal static CorrelationResult CorrelateWithLagPerWindowRanks(
            ReadOnlySpan<double> first,
            ReadOnlySpan<double> second,
            int maxLagSamples)
        {
            var n = first.Length;
            var usableLag = Math.Min(maxLagSamples, n - MinimumSamples);

            if (usableLag < 0)
            {
                return CorrelationResult.Undecidable(n);
            }

            using var scratch = new PooledBuffer<double>(3 * n, clearMemory: false);
            using var order = new PooledBuffer<int>(n, clearMemory: false);

            var evaluated = 0;
            var best = CorrelationResult.Undecidable(n);

            for (var lag = -usableLag; lag <= usableLag; lag++)
            {
                var candidate = CorrelateCore(first, second, lag, 1, scratch.Span, order.Span);

                if (!candidate.IsUsable)
                {
                    continue;
                }

                evaluated++;

                if (candidate.Strength > best.Strength)
                {
                    best = candidate;
                }
            }

            if (evaluated == 0)
            {
                return CorrelationResult.Undecidable(n);
            }

            return best with
            {
                PValue = Math.Min(1.0, best.PValue * evaluated)
            };
        }

        /// <summary>
        /// Mid-ranks over the finite entries of <paramref name="values"/>, with non-finite positions marked
        /// <see cref="double.NaN"/> so the sliding Pearson can skip them without disturbing the alignment.
        /// Returns how many entries were ranked.
        /// </summary>
        private static int RankFinite(
            ReadOnlySpan<double> values,
            Span<double> ranks,
            Span<double> keys,
            Span<int> order)
        {
            var kept = 0;

            for (var i = 0; i < values.Length; i++)
            {
                ranks[i] = double.NaN;

                if (!double.IsFinite(values[i]))
                {
                    continue;
                }

                keys[kept] = values[i];
                order[kept] = i;
                kept++;
            }

            if (kept == 0)
            {
                return 0;
            }

            var sortedKeys = keys.Slice(0, kept);
            sortedKeys.Sort(order.Slice(0, kept));

            var index = 0;
            while (index < kept)
            {
                var last = index;
                while (last + 1 < kept && sortedKeys[last + 1] == sortedKeys[index])
                {
                    last++;
                }

                var midRank = (index + last + 2) / 2.0;
                for (var k = index; k <= last; k++)
                {
                    ranks[order[k]] = midRank;
                }

                index = last + 1;
            }

            return kept;
        }

        /// <summary>
        /// Pearson over two aligned rank vectors, ignoring index pairs where either side is a gap. One pass,
        /// two accumulators — this is the routine the whole lag scan reduces to once ranking is hoisted out.
        /// </summary>
        private static double PearsonSkippingGaps(
            ReadOnlySpan<double> a,
            ReadOnlySpan<double> b,
            out int kept)
        {
            kept = 0;

            var meanA = 0.0;
            var meanB = 0.0;

            for (var i = 0; i < a.Length; i++)
            {
                if (double.IsNaN(a[i]) || double.IsNaN(b[i]))
                {
                    continue;
                }

                meanA += a[i];
                meanB += b[i];
                kept++;
            }

            if (kept == 0)
            {
                return double.NaN;
            }

            meanA /= kept;
            meanB /= kept;

            var covariance = 0.0;
            var varianceA = 0.0;
            var varianceB = 0.0;

            for (var i = 0; i < a.Length; i++)
            {
                if (double.IsNaN(a[i]) || double.IsNaN(b[i]))
                {
                    continue;
                }

                var da = a[i] - meanA;
                var db = b[i] - meanB;

                covariance += da * db;
                varianceA += da * da;
                varianceB += db * db;
            }

            if (varianceA <= 0.0 || varianceB <= 0.0)
            {
                return double.NaN;
            }

            return Math.Clamp(covariance / Math.Sqrt(varianceA * varianceB), -1.0, 1.0);
        }

        /// <summary>
        /// The whole computation for one fixed offset. <paramref name="lag"/> &gt; 0 pairs
        /// <c>first[i]</c> with <c>second[i + lag]</c>, i.e. the first series leads.
        /// </summary>
        /// <param name="first">Candidate leader.</param>
        /// <param name="second">Candidate follower.</param>
        /// <param name="lag">Offset in samples, positive when the first series leads.</param>
        /// <param name="comparisons">Multiplier applied to the p-value; 1 when no scan is in progress.</param>
        /// <param name="scratch">At least 3·n doubles: two rank vectors and one sort key vector.</param>
        /// <param name="order">At least n ints, the sort permutation.</param>
        private static CorrelationResult CorrelateCore(
            ReadOnlySpan<double> first,
            ReadOnlySpan<double> second,
            int lag,
            int comparisons,
            Span<double> scratch,
            Span<int> order)
        {
            var n = first.Length;

            var firstStart = lag < 0 ? -lag : 0;
            var secondStart = lag > 0 ? lag : 0;
            var overlap = n - Math.Abs(lag);

            if (overlap < MinimumSamples)
            {
                return CorrelationResult.Undecidable(Math.Max(overlap, 0));
            }

            var ranksFirst = scratch.Slice(0, n);
            var ranksSecond = scratch.Slice(n, n);
            var keys = scratch.Slice(2 * n, n);

            // Pairwise-complete: an index survives only if both series are finite there. Compacting into the
            // rank buffers first keeps the two vectors aligned after the drop.
            var kept = 0;
            for (var i = 0; i < overlap; i++)
            {
                var a = first[firstStart + i];
                var b = second[secondStart + i];

                if (!double.IsFinite(a) || !double.IsFinite(b))
                {
                    continue;
                }

                ranksFirst[kept] = a;
                ranksSecond[kept] = b;
                kept++;
            }

            if (kept < MinimumSamples)
            {
                return CorrelationResult.Undecidable(kept);
            }

            RankInPlace(ranksFirst.Slice(0, kept), keys.Slice(0, kept), order.Slice(0, kept));
            RankInPlace(ranksSecond.Slice(0, kept), keys.Slice(0, kept), order.Slice(0, kept));

            var rho = PearsonOnRanks(ranksFirst.Slice(0, kept), ranksSecond.Slice(0, kept));

            if (double.IsNaN(rho))
            {
                // One side is entirely tied — a flat series has no ranking to correlate against, which is a
                // real answer ("no information here"), not an error.
                return CorrelationResult.Undecidable(kept);
            }

            return new CorrelationResult(rho, FisherPValue(rho, kept, comparisons), kept, lag);
        }

        /// <summary>
        /// Replaces <paramref name="values"/> with their mid-ranks. Ties share the average of the positions
        /// they span, which is what makes the resulting Pearson coefficient equal tie-corrected Spearman
        /// rather than an approximation of it.
        /// </summary>
        private static void RankInPlace(Span<double> values, Span<double> keys, Span<int> order)
        {
            var n = values.Length;

            for (var i = 0; i < n; i++)
            {
                keys[i] = values[i];
                order[i] = i;
            }

            keys.Sort(order);

            var index = 0;
            while (index < n)
            {
                var last = index;
                while (last + 1 < n && keys[last + 1] == keys[index])
                {
                    last++;
                }

                var midRank = (index + last + 2) / 2.0;
                for (var k = index; k <= last; k++)
                {
                    values[order[k]] = midRank;
                }

                index = last + 1;
            }
        }

        private static double PearsonOnRanks(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
        {
            var n = a.Length;

            var meanA = 0.0;
            var meanB = 0.0;
            for (var i = 0; i < n; i++)
            {
                meanA += a[i];
                meanB += b[i];
            }

            meanA /= n;
            meanB /= n;

            var covariance = 0.0;
            var varianceA = 0.0;
            var varianceB = 0.0;

            for (var i = 0; i < n; i++)
            {
                var da = a[i] - meanA;
                var db = b[i] - meanB;

                covariance += da * db;
                varianceA += da * da;
                varianceB += db * db;
            }

            if (varianceA <= 0.0 || varianceB <= 0.0)
            {
                return double.NaN;
            }

            return Math.Clamp(covariance / Math.Sqrt(varianceA * varianceB), -1.0, 1.0);
        }

        /// <summary>
        /// Two-sided p-value via the Fisher z-transform, multiplied by <paramref name="comparisons"/>.
        /// atanh(ρ)·√(n−3) is approximately standard normal, which reuses the normal CDF the rest of the
        /// namespace already depends on rather than pulling in a t-distribution for a marginal gain.
        /// </summary>
        private static double FisherPValue(double rho, int sampleCount, int comparisons)
        {
            if (sampleCount <= 3)
            {
                return 1.0;
            }

            // |ρ| = 1 sends atanh to infinity; back it off by one ulp-ish so the z stays finite and extreme,
            // which is the honest reading of a perfect rank match on a finite window.
            var bounded = Math.Clamp(rho, -0.999999999999, 0.999999999999);
            var z = Math.Atanh(bounded) * Math.Sqrt(sampleCount - 3.0);
            var oneSided = 1.0 - NormalDistribution.Cdf(Math.Abs(z));

            return Math.Min(1.0, 2.0 * oneSided * comparisons);
        }
    }
}
