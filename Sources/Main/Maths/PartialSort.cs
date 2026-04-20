// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.Maths
{
    /// <summary>
    ///     Indirect (index-based) sort utilities for float keys.
    ///     All routines operate in-place on an <c>int[]</c> of indices, ordered by the
    ///     corresponding values in a <see cref="ReadOnlySpan{T}"/>. No managed allocations.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         All comparisons are NaN-safe and rely on <see cref="float.CompareTo(float)"/>,
    ///         which treats <see cref="float.NaN"/> as the smallest possible value
    ///         (including below <see cref="float.NegativeInfinity"/>). In practice that means a NaN
    ///         fitness can never win a "top-K-largest" slot and can never corrupt the ordering.
    ///     </para>
    ///     <para>
    ///         Ties on <c>values</c> are broken deterministically by ascending index: among
    ///         entries with equal fitness, the one with the lower original index sorts first in
    ///         ascending order and last in descending order. This gives a total order over
    ///         <c>(value, index)</c> regardless of the initial permutation of <paramref name="indices"/>
    ///         passed in, and regardless of whether the underlying quicksort partitioned a given
    ///         range or fell through to insertion sort. Consequence: two runs with identical inputs
    ///         produce bit-identical rankings, which matters for reproducible training.
    ///     </para>
    ///     <para>
    ///         Designed for the evolutionary pipeline where population sizes are in the hundreds
    ///         to thousands and elite counts are typically 5–20 % of the population. All public
    ///         methods are AOT-safe.
    ///     </para>
    /// </remarks>
    public static class PartialSort
    {
        /// <summary>
        ///     Partial sort by descending value. After the call,
        ///     <paramref name="indices"/>[0..<paramref name="k"/>] contains the indices of the
        ///     k largest values in <paramref name="values"/>, sorted from largest to smallest.
        ///     Remaining positions are left in an unspecified order.
        /// </summary>
        /// <remarks>
        ///     O(n log k) time, O(1) extra memory. Internally maintains a min-heap of size k
        ///     during the selection phase, then a small insertion sort to order the elite set.
        /// </remarks>
        public static void TopKDescending(int[] indices, ReadOnlySpan<float> values, int k)
        {
            ArgumentNullException.ThrowIfNull(indices);

            var n = indices.Length;

            if (values.Length != n)
            {
                throw new ArgumentException("indices and values must have the same length.");
            }

            if ((uint)k > (uint)n)
            {
                throw new ArgumentOutOfRangeException(nameof(k));
            }

            if (k == 0 || n == 0)
            {
                return;
            }

            if (k == n)
            {
                SortIndices(indices, values, ascending: false);
                return;
            }

            // Phase 1 — selection. Min-heap over indices[0..k]; the root holds the index
            // of the smallest value in the current elite set, so any better candidate evicts it.
            for (var i = (k >> 1) - 1; i >= 0; i--)
            {
                SiftDownMin(indices, values, i, k);
            }

            for (var i = k; i < n; i++)
            {
                // Root is the weakest elite under CompareDesc (highest "descending position",
                // i.e. worst of the current top-K). Candidate joins if it outranks root:
                // CompareDesc(candidate, root) < 0 means candidate sorts earlier in output,
                // i.e. candidate is better.
                if (CompareDesc(indices[i], indices[0], values) < 0)
                {
                    (indices[0], indices[i]) = (indices[i], indices[0]);
                    SiftDownMin(indices, values, 0, k);
                }
            }

            // Phase 2 — order. The heap guarantees membership is correct, but only the root
            // is sorted. Finish with an insertion sort over indices[0..k] for descending order.
            // Rationale: elite counts are small (tens to low hundreds in practice); insertion
            // sort is cache- and branch-predictor-friendly at this size, and faster than
            // a second O(k log k) heap-sort pass for realistic k.
            IndirectInsertionSort(indices, values, 0, k - 1, ascending: false);
        }

        /// <summary>
        ///     Full indirect sort of <paramref name="indices"/> by <paramref name="values"/>,
        ///     ascending or descending. NaN always sorts last in ascending and first in
        ///     descending order (as <see cref="float.CompareTo(float)"/> ranks it). Ties on
        ///     value are broken by ascending index.
        /// </summary>
        /// <remarks>
        ///     O(n log n) average, O(n^2) worst case (standard quicksort).
        ///     Zero managed allocations. For inputs ≤ 16 elements, falls back to insertion sort.
        /// </remarks>
        public static void SortIndices(int[] indices, ReadOnlySpan<float> values, bool ascending)
        {
            ArgumentNullException.ThrowIfNull(indices);

            if (values.Length != indices.Length)
            {
                throw new ArgumentException("indices and values must have the same length.");
            }

            if (indices.Length <= 1)
            {
                return;
            }

            IndirectQuickSort(indices, values, 0, indices.Length - 1, ascending);
        }

        // -------------------------------------------------------------------------------------
        // Internals
        // -------------------------------------------------------------------------------------

        /// <summary>
        ///     Ascending total order: primary value ASC, secondary index ASC.
        ///     Returns &lt;0 when <paramref name="left"/> sorts BEFORE <paramref name="right"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CompareAsc(int left, int right, ReadOnlySpan<float> values)
        {
            var cmp = values[left].CompareTo(values[right]);
            return cmp != 0 ? cmp : left.CompareTo(right);
        }

        /// <summary>
        ///     Descending total order: primary value DESC, secondary index ASC (stable tie-break,
        ///     independent of value direction). Returns &lt;0 when <paramref name="left"/>
        ///     sorts BEFORE <paramref name="right"/> in the final output.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CompareDesc(int left, int right, ReadOnlySpan<float> values)
        {
            var cmp = values[right].CompareTo(values[left]);
            return cmp != 0 ? cmp : left.CompareTo(right);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int Compare(int left, int right, ReadOnlySpan<float> values, bool ascending)
        {
            return ascending ? CompareAsc(left, right, values) : CompareDesc(left, right, values);
        }

        private static void IndirectQuickSort(int[] indices, ReadOnlySpan<float> values, int lo, int hi, bool ascending)
        {
            while (lo < hi)
            {
                if (hi - lo < 16)
                {
                    IndirectInsertionSort(indices, values, lo, hi, ascending);
                    return;
                }

                var pivotIndex = indices[lo + ((hi - lo) >> 1)];
                var i = lo;
                var j = hi;

                while (i <= j)
                {
                    while (Compare(indices[i], pivotIndex, values, ascending) < 0)
                    {
                        i++;
                    }

                    while (Compare(indices[j], pivotIndex, values, ascending) > 0)
                    {
                        j--;
                    }

                    if (i <= j)
                    {
                        (indices[i], indices[j]) = (indices[j], indices[i]);
                        i++;
                        j--;
                    }
                }

                // Recurse into the smaller partition; loop on the larger one to bound recursion depth.
                if (j - lo < hi - i)
                {
                    IndirectQuickSort(indices, values, lo, j, ascending);
                    lo = i;
                }
                else
                {
                    IndirectQuickSort(indices, values, i, hi, ascending);
                    hi = j;
                }
            }
        }

        private static void IndirectInsertionSort(int[] indices, ReadOnlySpan<float> values, int lo, int hi, bool ascending)
        {
            for (var i = lo + 1; i <= hi; i++)
            {
                var key = indices[i];
                var j = i - 1;

                while (j >= lo && Compare(indices[j], key, values, ascending) > 0)
                {
                    indices[j + 1] = indices[j];
                    j--;
                }

                indices[j + 1] = key;
            }
        }

        /// <summary>
        ///     Restores the min-heap property rooted at <paramref name="start"/> under the
        ///     DESCENDING total order: the root holds the "smallest-so-far elite" — the
        ///     candidate with the lowest value, breaking ties toward the HIGHER index so that
        ///     the final top-K contains the lower-indexed members of any equal-valued group.
        ///     In CompareDesc terms, "smallest in the heap" = "largest under CompareDesc".
        /// </summary>
        private static void SiftDownMin(int[] indices, ReadOnlySpan<float> values, int start, int heapSize)
        {
            var root = start;

            while (true)
            {
                var left = (root << 1) + 1;

                if (left >= heapSize)
                {
                    return;
                }

                var smallest = root;

                // "smaller in heap" = "later in descending sorted output" = CompareDesc > 0
                if (CompareDesc(indices[left], indices[smallest], values) > 0)
                {
                    smallest = left;
                }

                var right = left + 1;

                if (right < heapSize && CompareDesc(indices[right], indices[smallest], values) > 0)
                {
                    smallest = right;
                }

                if (smallest == root)
                {
                    return;
                }

                (indices[root], indices[smallest]) = (indices[smallest], indices[root]);
                root = smallest;
            }
        }
    }
}
