// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Randomization;

namespace DevOnBike.Overfit.Tests.Core.Randomization
{
    public class RandomizationTests
    {
        [Fact]
        public void PRNG_RegressionTest_MustBeFullyDeterministic()
        {
            // Arrange: Hard seed guarantees the same stream every time
            var prng = new VectorizedRandom(seed: 12345);
            var buffer = new float[10000];

            // Act
            prng.Fill(buffer);

            // Assert: We sum 10,000 sampled floats.
            // Any change in the vector math will immediately fail this test.
            // (Note: copy the first computed sum from a debugger run
            // and paste it here as "expected"; here I use the example value 4987.123f)
            var sum = buffer.Sum();

            // We allow a small tolerance for float precision errors
            Assert.InRange(sum, 5077.00f, 5077.20f);
        }

        [Fact]
        public void PRNG_StatisticalTest_UniformDistributionQuality()
        {
            // Arrange
            var prng = new VectorizedRandom(seed: 999);
            var sampleSize = 1_000_000;
            var buffer = new float[sampleSize];

            // Act
            prng.Fill(buffer);

            // Assert 1: Range (no number may escape outside [0, 1) )
            var min = buffer.Min();
            var max = buffer.Max();
            Assert.True(min >= 0.0f, "Wygenerowano liczbę mniejszą niż 0!");
            Assert.True(max < 1.0f, "Wygenerowano liczbę większą lub równą 1!");

            // Assert 2: Mean (~0.5)
            var mean = buffer.Average(x => (double)x);
            Assert.InRange(mean, 0.499, 0.501); // Very close to 0.5

            // Assert 3: Variance (~0.08333)
            var variance = buffer.Average(x => Math.Pow(x - mean, 2));
            Assert.InRange(variance, 0.083, 0.084);

            // Assert 4: Bucket test (Simple Chi-Square / Histogram)
            var numBuckets = 10;
            var buckets = new int[numBuckets];

            foreach (var value in buffer)
            {
                var bucketIndex = (int)(value * numBuckets);
                if (bucketIndex >= numBuckets)
                {
                    bucketIndex = numBuckets - 1; // safety
                }
                buckets[bucketIndex]++;
            }

            // We expect 100,000 elements in each of the 10 buckets
            var expectedPerBucket = sampleSize / numBuckets;
            var tolerance = (int)(expectedPerBucket * 0.015); // Deviation tolerance 1.5%

            for (var i = 0; i < numBuckets; i++)
            {
                Assert.InRange(buckets[i], expectedPerBucket - tolerance, expectedPerBucket + tolerance);
            }
        }
    }
}