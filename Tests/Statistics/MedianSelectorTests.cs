// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Statistics
{
    /// <summary>
    /// Parity between selection and the sort-then-index median it replaced. The reference is deliberately the
    /// dumbest possible implementation — sort the whole thing, read the middle — because the point is to
    /// check the clever one against something with no room to be clever in.
    /// </summary>
    public sealed class MedianSelectorTests
    {
        [Fact]
        public void EmptySpan_IsZero()
        {
            Assert.Equal(0.0, MedianSelector.MedianInPlace([]));
        }

        [Theory]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(3)]
        [InlineData(4)]
        [InlineData(5)]
        [InlineData(17)]
        public void SmallSpans_MatchTheSortedMedian(int length)
        {
            var rng = new Random(1000 + length);

            for (var trial = 0; trial < 200; trial++)
            {
                var values = new double[length];
                for (var i = 0; i < length; i++)
                {
                    values[i] = (rng.NextDouble() * 200.0) - 100.0;
                }

                AssertMatchesSortedMedian(values);
            }
        }

        [Fact]
        public void RandomShapes_MatchTheSortedMedian()
        {
            // Odd and even lengths both matter: the even case averages two order statistics and is where an
            // off-by-one hides, because it still produces a plausible number.
            var rng = new Random(20260728);

            for (var trial = 0; trial < 400; trial++)
            {
                var length = rng.Next(1, 500);
                var values = new double[length];

                for (var i = 0; i < length; i++)
                {
                    values[i] = (rng.NextDouble() * 1e6) - 5e5;
                }

                AssertMatchesSortedMedian(values);
            }
        }

        [Fact]
        public void HeavilyTiedValues_MatchTheSortedMedian()
        {
            // The case selection is supposed to be bad at, and the one this codebase actually produces: a flat
            // metric yields pairwise slopes that are all exactly zero.
            var rng = new Random(4242);

            for (var trial = 0; trial < 300; trial++)
            {
                var length = rng.Next(1, 400);
                var distinct = rng.Next(1, 4);
                var values = new double[length];

                for (var i = 0; i < length; i++)
                {
                    values[i] = rng.Next(distinct);
                }

                AssertMatchesSortedMedian(values);
            }
        }

        [Fact]
        public void AllEqual_MatchesTheSortedMedian()
        {
            for (var length = 1; length <= 64; length++)
            {
                var values = new double[length];
                Array.Fill(values, 7.5);

                AssertMatchesSortedMedian(values);
            }
        }

        [Theory]
        [InlineData(false)]
        [InlineData(true)]
        public void AlreadyOrderedInput_MatchesTheSortedMedian(bool descending)
        {
            // Sorted and reverse-sorted are the classic degenerate pivots, and both arrive here in practice:
            // a clean monotone series produces pairwise slopes that are anything but randomly arranged.
            for (var length = 1; length <= 300; length += 7)
            {
                var values = new double[length];

                for (var i = 0; i < length; i++)
                {
                    values[i] = descending ? length - i : i;
                }

                AssertMatchesSortedMedian(values);
            }
        }

        [Fact]
        public void OrganPipeInput_MatchesTheSortedMedian()
        {
            // Up then down — a shape that defeats median-of-three more often than random data does.
            for (var length = 2; length <= 400; length += 13)
            {
                var values = new double[length];

                for (var i = 0; i < length; i++)
                {
                    values[i] = i < length / 2 ? i : length - i;
                }

                AssertMatchesSortedMedian(values);
            }
        }

        [Fact]
        public void ExtremeMagnitudes_MatchTheSortedMedian()
        {
            var values = new[]
            {
                double.MaxValue, double.MinValue, 0.0, -0.0, 1e-320, -1e-320,
                1e308, -1e308, 42.0, -42.0
            };

            AssertMatchesSortedMedian(values);
        }

        [Fact]
        public void SelectionLeavesThePrefixNoGreaterThanTheMedian()
        {
            // The even-length path depends on this invariant: after selecting the upper central element, the
            // lower one is the maximum of everything before it. If selection did not guarantee that, the
            // result would be quietly wrong on exactly half of all inputs.
            var rng = new Random(777);

            for (var trial = 0; trial < 200; trial++)
            {
                var length = 2 * rng.Next(1, 120);
                var values = new double[length];

                for (var i = 0; i < length; i++)
                {
                    values[i] = rng.NextDouble() * 1000.0;
                }

                var expected = SortedMedian(values);
                var actual = MedianSelector.MedianInPlace(values);

                Assert.Equal(expected, actual);

                var middle = length / 2;
                for (var i = 0; i < middle; i++)
                {
                    Assert.True(
                        values[i] <= values[middle],
                        $"prefix[{i}]={values[i]} exceeded the selected element {values[middle]}");
                }
            }
        }

        /// <summary>
        /// Bit-exact, not approximate. Both paths take the median of the same multiset of doubles and, for an
        /// even count, average the same two of them — so any difference at all is a defect, and a tolerance
        /// would only hide it.
        /// </summary>
        private static void AssertMatchesSortedMedian(double[] values)
        {
            var expected = SortedMedian(values);
            var actual = MedianSelector.MedianInPlace(values.AsSpan());

            Assert.Equal(expected, actual);
        }

        private static double SortedMedian(double[] values)
        {
            var copy = (double[])values.Clone();
            Array.Sort(copy);

            if (copy.Length == 0)
            {
                return 0.0;
            }

            var middle = copy.Length / 2;

            if (copy.Length % 2 == 1)
            {
                return copy[middle];
            }

            return 0.5 * (copy[middle - 1] + copy[middle]);
        }
    }
}
