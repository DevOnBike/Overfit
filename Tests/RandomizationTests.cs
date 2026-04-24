using DevOnBike.Overfit.Randomization;

namespace DevOnBike.Overfit.Tests
{
    public class RandomizationTests
    {
        [Fact]
        public void PRNG_RegressionTest_MustBeFullyDeterministic()
        {
            // Arrange: Twardy seed gwarantuje zawsze ten sam strumień
            var prng = new VectorizedRandom(seed: 12345);
            var buffer = new float[10000];

            // Act
            prng.Fill(buffer);

            // Assert: Liczymy sumę z 10 000 wylosowanych floatów. 
            // Każda zmiana w matematyce wektorów natychmiast wyrzuci ten test.
            // (Uwaga: pierwszą wyliczoną sumę musisz sobie skopiować z odpalenia debuggera 
            // i wkleić tu jako "expected", tu używam przykładowej 4987.123f)
            var sum = buffer.Sum();

            // Dajemy małą tolerancję na błędy precyzji float
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

            // Assert 1: Zakres (żadna liczba nie może uciec poza [0, 1) )
            var min = buffer.Min();
            var max = buffer.Max();
            Assert.True(min >= 0.0f, "Wygenerowano liczbę mniejszą niż 0!");
            Assert.True(max < 1.0f, "Wygenerowano liczbę większą lub równą 1!");

            // Assert 2: Średnia (~0.5)
            var mean = buffer.Average(x => (double)x);
            Assert.InRange(mean, 0.499, 0.501); // Bardzo blisko 0.5

            // Assert 3: Wariancja (~0.08333)
            var variance = buffer.Average(x => Math.Pow(x - mean, 2));
            Assert.InRange(variance, 0.083, 0.084);

            // Assert 4: Test wiaderkowy (Prosty Chi-Kwadrat / Histogram)
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

            // Spodziewamy się 100,000 elementów w każdym z 10 wiader
            var expectedPerBucket = sampleSize / numBuckets;
            var tolerance = (int)(expectedPerBucket * 0.015); // Tolerancja odchylenia 1.5%

            for (var i = 0; i < numBuckets; i++)
            {
                Assert.InRange(buckets[i], expectedPerBucket - tolerance, expectedPerBucket + tolerance);
            }
        }
    }
}