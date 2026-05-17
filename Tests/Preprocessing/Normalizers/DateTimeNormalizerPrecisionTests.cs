// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Data.Normalizers;

namespace DevOnBike.Overfit.Tests.Preprocessing.Normalizers
{
    public class DateTimeNormalizerPrecisionTests
    {
        // Testing precision to 6 decimal places (the limit for float is ~7)
        private const int HighPrecision = 6;

        [Fact]
        public void EncodeFromUnixMilliseconds_ShouldDetectSingleMillisecondChanges()
        {
            // ARRANGE
            var baseMillis = 1704067200000; // 1 Stycznia 2024, 00:00:00.000 UTC
            var plusOneMillis = 1704067200001; // 1 Stycznia 2024, 00:00:00.001 UTC

            // ACT
            var (sinBase, cosBase) = DateTimeNormalizer.EncodeFromUnixMilliseconds(baseMillis);
            var (sinPlusOne, cosPlusOne) = DateTimeNormalizer.EncodeFromUnixMilliseconds(plusOneMillis);

            // ASSERT
            // 1 millisecond on a 24 h clock is a very small angle, but it MUST be non-zero!
            Assert.NotEqual(sinBase, sinPlusOne);

            // Expected difference in radians for 1 ms:
            // 1 ms is 1 / (24 * 60 * 60 * 1000) of a day = 1.1574074e-8
            // The sine difference around zero should be approximately: 2 * PI * 1.1574e-8 ≈ 7.27e-8
            var difference = Math.Abs(sinPlusOne - sinBase);

            Assert.True(difference > 0f, "Brak precyzji! 1 milisekunda została zignorowana (pływające zaokrąglenie).");
            Assert.True(difference < 1e-6f, $"Różnica {difference} jest zbyt duża jak na 1 milisekundę!");
        }

        [Theory]
        // 1709208000 = 29 February 2024, 12:00:00 UTC (Thursday - Leap Day)
        [InlineData(1709208000, 0f, -1f, DayOfWeek.Thursday)]
        public void EncodeAllTimeFeatures_ShouldPerfectlyHandleLeapYearMidday(
            long unixSeconds, float expectedHourSin, float expectedHourCos, DayOfWeek expectedDay)
        {
            // ACT
            var result = DateTimeNormalizer.EncodeAllTimeFeaturesFromUnixSeconds(unixSeconds);

            // Calculate expected values for Thursday (Thursday = 4)
            var expectedDaySin = (float)Math.Sin((4.0 / 7.0) * 2.0 * Math.PI);
            var expectedDayCos = (float)Math.Cos((4.0 / 7.0) * 2.0 * Math.PI);

            // ASSERT - Time (Perfect 12:00 noon -> Sin=0, Cos=-1)
            // We use HighPrecision (6) to ensure Math.PI does not introduce noise at the 7th decimal place
            Assert.Equal(expectedHourSin, result.HourSin, HighPrecision);
            Assert.Equal(expectedHourCos, result.HourCos, HighPrecision);

            // ASSERT - Day
            Assert.Equal(expectedDaySin, result.DaySin, HighPrecision);
            Assert.Equal(expectedDayCos, result.DayCos, HighPrecision);
        }

        [Fact]
        public void Encode_ZeroCrossing_ShouldNotProduceNaNOrInfinity()
        {
            // ARRANGE
            // 06:00:00 -> exactly 90 degrees, cosine should equal 0
            var time = new TimeSpan(6, 0, 0);

            // ACT
            var (sin, cos) = DateTimeNormalizer.Encode(time);

            // ASSERT
            Assert.False(float.IsNaN(sin));
            Assert.False(float.IsNaN(cos));
            Assert.False(float.IsInfinity(sin));
            Assert.False(float.IsInfinity(cos));

            // The cosine of 90 degrees (PI/2) in C# is not exactly 0 due to PI rounding,
            // but it must be VERY close to zero (on the order of 10^-7 for float)
            Assert.Equal(0f, cos, HighPrecision);
            Assert.Equal(1f, sin, HighPrecision);
        }


    }
}