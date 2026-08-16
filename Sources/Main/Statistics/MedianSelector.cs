// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics;

namespace DevOnBike.Overfit.Statistics
{
    /// <summary>
    /// Median of an unordered span, by selection rather than by ordering the whole thing.
    ///
    /// <para><b>Why this exists.</b> The robust estimators in this namespace are built on medians of large
    /// derived sets — <see cref="TrendDetector"/> takes the median of every pairwise slope, which at its
    /// 600-sample cap is 179 700 values. Sorting all of them to read the middle one is an O(n log n) answer
    /// to an O(n) question, and it was not a rounding error: measured at 600 samples, <c>slopes.Sort()</c>
    /// was <b>99%</b> of the detector's entire runtime, 8.02 ms of 8.10 ms. Selection cuts that to 1.21 ms.</para>
    ///
    /// <para><b>The worst case is bounded rather than hoped away.</b> Hoare partitioning is O(n) expected but
    /// O(n²) on a sequence crafted against the pivot rule, and pairwise slopes are derived data — nobody
    /// audits their distribution. After a partition budget of 2·log₂(n)+4 the window is sorted outright, so
    /// the routine is O(n) in practice and O(n log n) always. That is the introselect construction, and the
    /// bound is named here rather than left as an assumption about the data.</para>
    ///
    /// <para>Permutes its input and allocates nothing.</para>
    /// </summary>
    internal static class MedianSelector
    {
        /// <summary>
        /// Median of <paramref name="values"/>, reordering them in the process. Returns 0 for an empty span,
        /// matching the sorted-median helper it replaced.
        /// </summary>
        internal static double MedianInPlace(Span<double> values)
        {
            if (values.Length == 0)
            {
                return 0.0;
            }

            var middle = values.Length / 2;

            if (values.Length % 2 == 1)
            {
                return Select(values, middle);
            }

            // Selection leaves everything before `middle` no greater than the element at it, so the lower of
            // the two central values is the maximum of that prefix — one linear scan instead of a second
            // selection pass.
            var upper = Select(values, middle);
            var lower = double.NegativeInfinity;

            for (var i = 0; i < middle; i++)
            {
                if (values[i] > lower)
                {
                    lower = values[i];
                }
            }

            return 0.5 * (lower + upper);
        }

        /// <summary>
        /// The <paramref name="k"/>-th smallest element, leaving every earlier position no greater than it.
        /// </summary>
        private static double Select(Span<double> values, int k)
        {
            var lo = 0;
            var hi = values.Length - 1;

            // #pragma BOUND: partitions <= 2*log2(n)+4; on exhaustion the remaining window is sorted, so the
            // loop terminates in O(n log n) regardless of how adversarial the input is.
            var budget = (2 * BitOperations.Log2((uint)values.Length)) + 4;

            while (lo < hi)
            {
                if (budget <= 0)
                {
                    values.Slice(lo, (hi + 1) - lo).Sort();

                    return values[k];
                }

                budget--;

                var pivot = MedianOfThree(values, lo, hi);
                var i = lo;
                var j = hi;

                while (i <= j)
                {
                    while (values[i] < pivot)
                    {
                        i++;
                    }

                    while (values[j] > pivot)
                    {
                        j--;
                    }

                    if (i > j)
                    {
                        break;
                    }

                    (values[i], values[j]) = (values[j], values[i]);
                    i++;
                    j--;
                }

                if (k <= j)
                {
                    hi = j;
                    continue;
                }

                if (k >= i)
                {
                    lo = i;
                    continue;
                }

                // Strictly between the two partition boundaries: every element there equals the pivot, so
                // this position already holds its final value.
                return values[k];
            }

            return values[lo];
        }

        /// <summary>
        /// Median of the first, middle and last elements. Cheap insurance against the already-sorted and
        /// reverse-sorted inputs that make a first-element pivot degenerate — and both of those arrive here
        /// routinely, because a clean monotone series produces pairwise slopes that are far from random.
        /// </summary>
        private static double MedianOfThree(ReadOnlySpan<double> values, int lo, int hi)
        {
            var mid = lo + ((hi - lo) / 2);

            var a = values[lo];
            var b = values[mid];
            var c = values[hi];

            if (a > b)
            {
                (a, b) = (b, a);
            }

            if (b > c)
            {
                (b, c) = (c, b);
            }

            if (a > b)
            {
                (a, b) = (b, a);
            }

            return b;
        }
    }
}
