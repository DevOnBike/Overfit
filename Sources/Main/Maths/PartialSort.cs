using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.Maths
{
    /// <summary>
    /// Indirect partial sort by float keys. Rearranges <paramref name="indices"/>
    /// so that the first <paramref name="k"/> positions hold the indices of the
    /// k largest (or smallest) values, in sorted order. Remaining indices are
    /// left in an unspecified order. NaN values rank last regardless of direction.
    /// </summary>
    public static class PartialSort
    {
        /// <summary>
        /// In-place partial sort, descending. After the call, <paramref name="indices"/>[0..k]
        /// contain the indices of the k largest values in <paramref name="values"/>, sorted
        /// from largest to smallest. Uses a k-element min-heap → O(n log k) time, O(1) extra memory.
        /// NaN values are treated as -infinity (they cannot win a "largest" slot).
        /// </summary>
        public static void TopKDescending(int[] indices, ReadOnlySpan<float> values, int k)
        {
            if (indices is null)
            {
                throw new ArgumentNullException(nameof(indices));
            }

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

            // Build a min-heap over the first k entries. Root = smallest among the current top-k.
            for (var i = k / 2 - 1; i >= 0; i--)
            {
                SiftDownMin(indices, values, i, k);
            }

            // Scan the rest. Anything beating the heap root (= current k-th best) replaces it.
            for (var i = k; i < n; i++)
            {
                if (Better(values, indices[i], indices[0]))
                {
                    (indices[0], indices[i]) = (indices[i], indices[0]);
                    SiftDownMin(indices, values, 0, k);
                }
            }

            // Heap-sort the top-k in place to get descending order at indices[0..k].
            for (var end = k - 1; end > 0; end--)
            {
                (indices[0], indices[end]) = (indices[end], indices[0]);
                SiftDownMin(indices, values, 0, end);
            }
        }

        /// <summary>
        /// Full indirect sort by float key. Ascending if <paramref name="ascending"/> is true,
        /// otherwise descending. NaN values always land at the end. O(n log n), zero alloc.
        /// </summary>
        public static void SortIndices(int[] indices, ReadOnlySpan<float> values, bool ascending)
        {
            if (indices is null)
            {
                throw new ArgumentNullException(nameof(indices));
            }

            if (values.Length != indices.Length)
            {
                throw new ArgumentException("indices and values must have the same length.");
            }

            // .NET's Array.Sort over a key/value pair is not directly applicable to indirect sort
            // without allocating a keys array, so use an introsort-style quicksort on indices.
            IndirectQuickSort(indices, values, 0, indices.Length - 1, ascending);
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

                var pivotIdx = indices[lo + ((hi - lo) >> 1)];
                var pivot = values[pivotIdx];
                var i = lo;
                var j = hi;

                while (i <= j)
                {
                    while (ascending ? LessThan(values[indices[i]], pivot) : GreaterThan(values[indices[i]], pivot))
                    {
                        i++;
                    }

                    while (ascending ? GreaterThan(values[indices[j]], pivot) : LessThan(values[indices[j]], pivot))
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

                // Recurse into the smaller partition; iterate the larger (bounds recursion depth).
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
                var keyVal = values[key];
                var j = i - 1;

                while (j >= lo && (ascending ? GreaterThan(values[indices[j]], keyVal) : LessThan(values[indices[j]], keyVal)))
                {
                    indices[j + 1] = indices[j];
                    j--;
                }

                indices[j + 1] = key;
            }
        }

        private static void SiftDownMin(int[] indices, ReadOnlySpan<float> values, int start, int heapSize)
        {
            var root = start;

            while (true)
            {
                var left = 2 * root + 1;
                if (left >= heapSize)
                {
                    return;
                }

                var smallest = root;
                if (Better(values, indices[smallest], indices[left]))
                {
                    smallest = left;
                }

                var right = left + 1;
                if (right < heapSize && Better(values, indices[smallest], indices[right]))
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

        /// <summary>
        /// Returns true when the element at <paramref name="candidate"/> would push the element at
        /// <paramref name="incumbent"/> out of a top-K-descending set. NaN-aware: a NaN candidate
        /// never wins, and a NaN incumbent is always displaced.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool Better(ReadOnlySpan<float> values, int incumbent, int candidate)
        {
            var a = values[incumbent];
            var b = values[candidate];

            // NaN handling: NaN is always the worst.
            if (float.IsNaN(b))
            {
                return false;
            }

            if (float.IsNaN(a))
            {
                return true;
            }

            return b > a;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool LessThan(float a, float b)
        {
            // NaN sorts last in both directions.
            if (float.IsNaN(a))
            {
                return false;
            }

            if (float.IsNaN(b))
            {
                return true;
            }

            return a < b;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool GreaterThan(float a, float b)
        {
            if (float.IsNaN(a))
            {
                return false;
            }

            if (float.IsNaN(b))
            {
                return true;
            }

            return a > b;
        }
    }
}