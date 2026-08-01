// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Statistics
{
    /// <summary>
    /// Order statistics over a stream that is never stored.
    ///
    /// <para>The property that matters is asymmetric on purpose: the <b>maximum must be exact</b>, because a
    /// floor is set from the largest thing a healthy period did, while the percentiles are context for a human
    /// and tolerate sampling.</para>
    /// </summary>
    public sealed class BoundedSamplesTests
    {
        [Fact]
        public void MemoryDoesNotGrowWithTheStream()
        {
            var samples = new BoundedSamples(capacity: 16);

            for (var i = 0; i < 100_000; i++)
            {
                samples.Add(i);
            }

            Assert.Equal(100_000, samples.Count);

            // Nothing to assert about internals; the guarantee is that Add is O(1) amortised against a fixed
            // array, and the count above is the whole stream while the reservoir stayed at 16.
            Assert.Equal(99_999.0, samples.Max);
        }

        /// <summary>
        /// The maximum survives every halving. If decimation could drop it, the one number a floor is derived
        /// from would be the one number the structure loses.
        /// </summary>
        [Fact]
        public void TheMaximumIsExactEvenWhenItArrivesEarly()
        {
            var samples = new BoundedSamples(capacity: 8);

            samples.Add(1000.0);

            for (var i = 0; i < 5000; i++)
            {
                samples.Add(1.0);
            }

            Assert.Equal(1000.0, samples.Max);
        }

        [Fact]
        public void QuantilesTrackTheDistribution()
        {
            var samples = new BoundedSamples(capacity: 512);

            for (var i = 0; i <= 10_000; i++)
            {
                samples.Add(i / 100.0);
            }

            // A uniform ramp to 100: the median lands near 50 and the 99th near 99, within the error a
            // decimated sample can be expected to carry.
            Assert.InRange(samples.Quantile(0.5), 45.0, 55.0);
            Assert.InRange(samples.Quantile(0.99), 95.0, 100.0);
        }

        [Fact]
        public void AnEmptySampleAnswersZeroRatherThanThrowing()
        {
            var samples = new BoundedSamples();

            Assert.Equal(0, samples.Count);
            Assert.Equal(0.0, samples.Quantile(0.5));
        }

        [Fact]
        public void NonFiniteValuesAreNotMagnitudes()
        {
            var samples = new BoundedSamples();

            samples.Add(double.NaN);
            samples.Add(double.PositiveInfinity);

            Assert.Equal(0, samples.Count);
        }

        [Fact]
        public void ItSurvivesARoundTrip()
        {
            var samples = new BoundedSamples(capacity: 64);

            for (var i = 0; i < 5000; i++)
            {
                samples.Add(i * 0.5);
            }

            var restored = BoundedSamples.Read(samples.Write(), capacity: 64);

            Assert.Equal(samples.Count, restored.Count);
            Assert.Equal(samples.Max, restored.Max);
            Assert.Equal(samples.Quantile(0.5), restored.Quantile(0.5));
        }

        [Theory]
        [InlineData(null)]
        [InlineData("")]
        [InlineData("nonsense")]
        [InlineData("1 2")]
        public void AnUnreadablePayloadIsAnEmptySample(string? text)
        {
            Assert.Equal(0, BoundedSamples.Read(text).Count);
        }

        /// <summary>
        /// Deterministic: the same stream must produce the same answer twice, or a calibration cannot be
        /// argued with.
        /// </summary>
        [Fact]
        public void TheSameStreamGivesTheSameAnswerTwice()
        {
            var first = new BoundedSamples(capacity: 32);
            var second = new BoundedSamples(capacity: 32);

            for (var i = 0; i < 4000; i++)
            {
                var value = ((i * 37) % 1000) / 3.0;

                first.Add(value);
                second.Add(value);
            }

            Assert.Equal(first.Max, second.Max);
            Assert.Equal(first.Quantile(0.5), second.Quantile(0.5));
            Assert.Equal(first.Quantile(0.99), second.Quantile(0.99));
        }
    }
}
