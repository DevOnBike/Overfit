// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Data.Normalizers
{
    public static class DateTimeNormalizer
    {
        private const double TwoPi = 2.0 * Math.PI;

        public static (float Sin, float Cos) Encode(TimeSpan timeOfDay)
        {
            var decimalHour = timeOfDay.TotalHours;
            var radians = decimalHour / 24.0 * TwoPi;

            return (CleanZero(Math.Sin(radians)), CleanZero(Math.Cos(radians)));
        }

        public static (float Sin, float Cos) Encode(DayOfWeek dayOfWeek)
        {
            var dayInt = (int)dayOfWeek;
            var radians = dayInt / 7.0 * TwoPi;

            return (CleanZero(Math.Sin(radians)), CleanZero(Math.Cos(radians)));
        }

        public static (float Sin, float Cos) EncodeFromUnixSeconds(long unixTimeSeconds)
        {
            var utcTime = DateTimeOffset.FromUnixTimeSeconds(unixTimeSeconds).UtcDateTime;
            return Encode(utcTime.TimeOfDay);
        }

        public static (float Sin, float Cos) EncodeFromUnixMilliseconds(long unixTimeMilliseconds)
        {
            var utcTime = DateTimeOffset.FromUnixTimeMilliseconds(unixTimeMilliseconds).UtcDateTime;
            return Encode(utcTime.TimeOfDay);
        }

        public static (float Sin, float Cos) EncodeDayOfWeekFromUnixSeconds(long unixTimeSeconds)
        {
            var utcTime = DateTimeOffset.FromUnixTimeSeconds(unixTimeSeconds).UtcDateTime;
            return Encode(utcTime.DayOfWeek);
        }

        public static (float HourSin, float HourCos, float DaySin, float DayCos) EncodeAllTimeFeaturesFromUnixSeconds(long unixTimeSeconds)
        {
            var utcTime = DateTimeOffset.FromUnixTimeSeconds(unixTimeSeconds).UtcDateTime;

            var (sin, cos) = Encode(utcTime.TimeOfDay);
            var (daySin, dayCos) = Encode(utcTime.DayOfWeek);

            return (sin, cos, daySin, dayCos);
        }

        private static float CleanZero(double value)
        {
            return Math.Abs(value) < 1e-10 ? 0f : (float)value;
        }
    }
}