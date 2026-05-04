// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Maths;

namespace DevOnBike.Overfit.Tests
{
    public sealed class PartialSortTests
    {
        // ------------------------------------------------------------------
        // TopKDescending — basic correctness
        // ------------------------------------------------------------------

        [Fact]
        public void TopKDescending_ReturnsLargestKInOrder()
        {
            float[] values = [3f, 1f, 4f, 1f, 5f, 9f, 2f, 6f];
            var indices = InitIndices(values.Length);

            PartialSort.TopKDescending(indices, values, k: 3);

            // Expected top-3: 9 (idx 5), 6 (idx 7), 5 (idx 4).
            Assert.Equal(5, indices[0]);
            Assert.Equal(7, indices[1]);
            Assert.Equal(4, indices[2]);
        }

        [Fact]
        public void TopKDescending_KEqualsN_FullySortsAllIndices()
        {
            float[] values = [3f, 1f, 4f, 1f, 5f];
            var indices = InitIndices(values.Length);

            PartialSort.TopKDescending(indices, values, k: values.Length);

            // Full descending sort. Tie: 1 at idx 1 vs idx 3 — ASC tie-break keeps 1 before 3.
            Assert.Equal(4, indices[0]); // 5
            Assert.Equal(2, indices[1]); // 4
            Assert.Equal(0, indices[2]); // 3
            Assert.Equal(1, indices[3]); // 1 at lower idx
            Assert.Equal(3, indices[4]); // 1 at higher idx
        }

        [Fact]
        public void TopKDescending_KZero_IsNoOp()
        {
            float[] values = [1f, 2f, 3f];
            var indices = InitIndices(values.Length);

            PartialSort.TopKDescending(indices, values, k: 0);

            // Indices untouched.
            Assert.Equal(0, indices[0]);
            Assert.Equal(1, indices[1]);
            Assert.Equal(2, indices[2]);
        }

        // ------------------------------------------------------------------
        // Tie-break: ascending-index stable tie-break is the documented contract.
        // ------------------------------------------------------------------

        [Fact]
        public void TopKDescending_AllEqualValues_SelectsLowestIndices()
        {
            // Every value equal → top-K should be {idx 0, idx 1, ..., idx k-1} in order.
            float[] values = [5f, 5f, 5f, 5f, 5f];
            var indices = InitIndices(values.Length);

            PartialSort.TopKDescending(indices, values, k: 2);

            Assert.Equal(0, indices[0]);
            Assert.Equal(1, indices[1]);
        }

        [Fact]
        public void TopKDescending_AllEqualValues_DeterministicFullSort()
        {
            // Full sort of equal values: should produce indices in ascending order.
            float[] values = [7f, 7f, 7f, 7f];
            int[] indices = [3, 1, 0, 2]; // scrambled start

            PartialSort.TopKDescending(indices, values, k: 4);

            Assert.Equal(new[] { 0, 1, 2, 3 }, indices);
        }

        [Fact]
        public void TopKDescending_TiesWithDistinctValues_HonorTieBreak()
        {
            // Values: [9, 9, 3, 9, 2]. Top-3 should be {0, 1, 3} — all the 9s, in index order.
            float[] values = [9f, 9f, 3f, 9f, 2f];
            var indices = InitIndices(values.Length);

            PartialSort.TopKDescending(indices, values, k: 3);

            Assert.Equal(0, indices[0]);
            Assert.Equal(1, indices[1]);
            Assert.Equal(3, indices[2]);
        }

        [Fact]
        public void TopKDescending_OutputIsInvariantUnderInputPermutation()
        {
            // For the tie-break contract to be meaningful, the TOP-K region must not depend
            // on the initial permutation of `indices` — it can only depend on `values`.
            // Positions [k..n) are explicitly "left in an unspecified order" by the contract,
            // so we only compare the first k entries.
            const int k = 3;
            float[] values = [5f, 3f, 5f, 3f, 5f, 3f];

            int[] ordered = [0, 1, 2, 3, 4, 5];
            int[] scrambled = [5, 3, 0, 2, 4, 1];

            PartialSort.TopKDescending(ordered, values, k);
            PartialSort.TopKDescending(scrambled, values, k);

            var orderedTop = ordered.AsSpan(0, k).ToArray();
            var scrambledTop = scrambled.AsSpan(0, k).ToArray();

            Assert.Equal(orderedTop, scrambledTop);
            // Expected top-3: idx 0, 2, 4 (all the 5s, ascending-index tie-break).
            Assert.Equal(new[] { 0, 2, 4 }, orderedTop);
        }

        [Fact]
        public void SortIndices_Ascending_TiesBreakByIndex()
        {
            float[] values = [2f, 1f, 2f, 1f];
            var indices = InitIndices(values.Length);

            PartialSort.SortIndices(indices, values, ascending: true);

            // Expected: 1 (idx 1), 1 (idx 3), 2 (idx 0), 2 (idx 2).
            Assert.Equal(new[] { 1, 3, 0, 2 }, indices);
        }

        [Fact]
        public void SortIndices_Descending_TiesBreakByIndex()
        {
            float[] values = [2f, 1f, 2f, 1f];
            var indices = InitIndices(values.Length);

            PartialSort.SortIndices(indices, values, ascending: false);

            // Expected: 2 (idx 0), 2 (idx 2), 1 (idx 1), 1 (idx 3).
            // Primary DESC, secondary ASC.
            Assert.Equal(new[] { 0, 2, 1, 3 }, indices);
        }

        [Fact]
        public void SortIndices_LargeInputCrossesQuicksortBoundary_StillStable()
        {
            // Size > 16 forces the quicksort path, which is not stable by default. The
            // secondary-index tie-break must make it effectively stable anyway.
            const int n = 200;
            var values = new float[n];
            var indices = InitIndices(n);

            // Alternating pattern of just three distinct values → lots of ties.
            for (var i = 0; i < n; i++)
            {
                values[i] = (i % 3) switch
                {
                    0 => 1f,
                    1 => 2f,
                    _ => 3f,
                };
            }

            PartialSort.SortIndices(indices, values, ascending: true);

            // Within each value group, indices must be in ascending order.
            var prev = -1;
            var prevValue = float.NegativeInfinity;

            for (var i = 0; i < n; i++)
            {
                var currentValue = values[indices[i]];

                if (currentValue == prevValue)
                {
                    Assert.True(indices[i] > prev,
                        $"Tie-break failure at position {i}: prev idx {prev}, current idx {indices[i]}, value {currentValue}.");
                }

                prev = indices[i];
                prevValue = currentValue;
            }
        }

        // ------------------------------------------------------------------
        // NaN safety — must not poison rankings in either direction.
        // ------------------------------------------------------------------

        [Fact]
        public void TopKDescending_WithNaN_NaNNeverEntersTopK()
        {
            float[] values = [float.NaN, 5f, 3f, float.NaN, 1f];
            var indices = InitIndices(values.Length);

            PartialSort.TopKDescending(indices, values, k: 2);

            // Expected top-2: idx 1 (5), idx 2 (3). NaNs excluded.
            Assert.Equal(1, indices[0]);
            Assert.Equal(2, indices[1]);
        }

        [Fact]
        public void TopKDescending_AllNaN_KEqualsN_NaNsSortLast()
        {
            // All NaN is a degenerate input; the sort must still terminate without throwing.
            // Output order of NaNs is unspecified but must be a valid permutation of [0..n).
            float[] values = [float.NaN, float.NaN, float.NaN];
            var indices = InitIndices(values.Length);

            PartialSort.TopKDescending(indices, values, k: values.Length);

            // Must be a permutation.
            Array.Sort(indices);
            Assert.Equal(new[] { 0, 1, 2 }, indices);
        }

        [Fact]
        public void SortIndices_Descending_WithNaN_NaNGoesLast()
        {
            // float.CompareTo ranks NaN below -Infinity. In descending sort that means NaN
            // goes last.
            float[] values = [5f, float.NaN, 3f, 1f, float.NaN];
            var indices = InitIndices(values.Length);

            PartialSort.SortIndices(indices, values, ascending: false);

            // First 3 positions: finite values in descending order.
            Assert.Equal(0, indices[0]); // 5
            Assert.Equal(2, indices[1]); // 3
            Assert.Equal(3, indices[2]); // 1
            // Last 2: the NaNs, in ascending index order (tie-break).
            Assert.Equal(1, indices[3]);
            Assert.Equal(4, indices[4]);
        }

        [Fact]
        public void SortIndices_Ascending_WithNaN_NaNGoesFirst()
        {
            // Ascending + NaN-is-smallest ⇒ NaN slots lead the output.
            float[] values = [5f, float.NaN, 3f, 1f, float.NaN];
            var indices = InitIndices(values.Length);

            PartialSort.SortIndices(indices, values, ascending: true);

            Assert.Equal(1, indices[0]); // NaN at lower idx
            Assert.Equal(4, indices[1]); // NaN at higher idx
            Assert.Equal(3, indices[2]); // 1
            Assert.Equal(2, indices[3]); // 3
            Assert.Equal(0, indices[4]); // 5
        }

        // ------------------------------------------------------------------
        // Allocation stability.
        // ------------------------------------------------------------------

        [Fact]
        public void TopKDescending_IsAllocationStable()
        {
            var values = new float[128];
            var rng = new Random(42);
            for (var i = 0; i < values.Length; i++)
            {
                values[i] = rng.NextSingle();
            }

            var indices = InitIndices(values.Length);
            PartialSort.TopKDescending(indices, values, k: 16); // warmup

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < 1_000; i++)
            {
                indices = InitIndices(values.Length);
                PartialSort.TopKDescending(indices, values, k: 16);
            }

            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            var perCall = allocated / 1_000.0;

            // Each iteration allocates one fresh int[128], ~512 B. The sort itself must add
            // nothing on top. Tolerance well above the int[] cost, well below anything alarming.
            Assert.True(perCall <= 640, $"TopKDescending + InitIndices allocated {perCall:F1} B/call.");
        }

        // ------------------------------------------------------------------
        // Helpers
        // ------------------------------------------------------------------

        private static int[] InitIndices(int length)
        {
            var indices = new int[length];
            for (var i = 0; i < length; i++)
            {
                indices[i] = i;
            }
            return indices;
        }
    }
}