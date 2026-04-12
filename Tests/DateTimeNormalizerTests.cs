using System;
using Xunit;
using DevOnBike.Overfit.Data.Normalizers;

namespace DevOnBike.Overfit.Tests
{
    public class DateTimeNormalizerPrecisionTests
    {
        // Badamy dokładność do 6 miejsc po przecinku (limit typu float to ~7)
        private const int HighPrecision = 6;

        [Fact]
        public void EncodeFromUnixMilliseconds_ShouldDetectSingleMillisecondChanges()
        {
            // ARRANGE
            long baseMillis = 1704067200000; // 1 Stycznia 2024, 00:00:00.000 UTC
            long plusOneMillis = 1704067200001; // 1 Stycznia 2024, 00:00:00.001 UTC

            // ACT
            var (sinBase, cosBase) = DateTimeNormalizer.EncodeFromUnixMilliseconds(baseMillis);
            var (sinPlusOne, cosPlusOne) = DateTimeNormalizer.EncodeFromUnixMilliseconds(plusOneMillis);

            // ASSERT
            // 1 milisekunda na tarczy 24h to bardzo mały kąt, ale MUSI być różny od zera!
            Assert.NotEqual(sinBase, sinPlusOne);

            // Oczekiwana różnica w radianach dla 1ms:
            // 1 ms to 1 / (24 * 60 * 60 * 1000) części dnia = 1.1574074e-8
            // Różnica na Sinusie wokół zera powinna wynosić mniej więcej: 2 * PI * 1.1574e-8 ≈ 7.27e-8
            float difference = Math.Abs(sinPlusOne - sinBase);

            Assert.True(difference > 0f, "Brak precyzji! 1 milisekunda została zignorowana (pływające zaokrąglenie).");
            Assert.True(difference < 1e-6f, $"Różnica {difference} jest zbyt duża jak na 1 milisekundę!");
        }

        [Theory]
        // 1709208000 = 29 Lutego 2024, 12:00:00 UTC (Czwartek - Dzień Przestępny)
        [InlineData(1709208000, 0f, -1f, DayOfWeek.Thursday)]
        public void EncodeAllTimeFeatures_ShouldPerfectlyHandleLeapYearMidday(
            long unixSeconds, float expectedHourSin, float expectedHourCos, DayOfWeek expectedDay)
        {
            // ACT
            var result = DateTimeNormalizer.EncodeAllTimeFeaturesFromUnixSeconds(unixSeconds);

            // Obliczamy oczekiwane wartości dla czwartku (Thursday = 4)
            float expectedDaySin = (float)Math.Sin((4.0 / 7.0) * 2.0 * Math.PI);
            float expectedDayCos = (float)Math.Cos((4.0 / 7.0) * 2.0 * Math.PI);

            // ASSERT - Czas (Idealne 12:00 w południe -> Sin=0, Cos=-1)
            // Używamy HighPrecision (6), aby upewnić się, że Math.PI nie robi nam śmieci na 7. miejscu po przecinku
            Assert.Equal(expectedHourSin, result.HourSin, HighPrecision);
            Assert.Equal(expectedHourCos, result.HourCos, HighPrecision);

            // ASSERT - Dzień
            Assert.Equal(expectedDaySin, result.DaySin, HighPrecision);
            Assert.Equal(expectedDayCos, result.DayCos, HighPrecision);
        }

        [Fact]
        public void Encode_ZeroCrossing_ShouldNotProduceNaNOrInfinity()
        {
            // ARRANGE
            // 06:00:00 -> dokładnie 90 stopni, Cosinus powinen być równy 0
            var time = new TimeSpan(6, 0, 0);

            // ACT
            var (sin, cos) = DateTimeNormalizer.Encode(time);

            // ASSERT
            Assert.False(float.IsNaN(sin));
            Assert.False(float.IsNaN(cos));
            Assert.False(float.IsInfinity(sin));
            Assert.False(float.IsInfinity(cos));

            // Cosinus dla 90 stopni (PI/2) w C# z powodu zaokrągleń PI nie jest idealnym 0, 
            // ale musi być BARDZO blisko zera (rzędu 10^-7 dla float)
            Assert.Equal(0f, cos, HighPrecision);
            Assert.Equal(1f, sin, HighPrecision);
        }


    }
}