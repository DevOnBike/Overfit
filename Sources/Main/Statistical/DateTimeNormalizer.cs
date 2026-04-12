using System;

namespace DevOnBike.Overfit.Statistical
{
    /// <summary>
    /// Klasa narzędziowa do cyklicznego kodowania czasu (Cyclical Encoding).
    /// Zamienia liniowy czas na współrzędne na okręgu (Sin/Cos).
    /// </summary>
    public static class DateTimeNormalizer
    {
        private const double TwoPi = 2.0 * Math.PI;

        /// <summary>
        /// Transformuje czas z doby (00:00 - 23:59) na dwie zmienne cykliczne.
        /// </summary>
        public static (float Sin, float Cos) Encode(TimeSpan timeOfDay)
        {
            var decimalHour = timeOfDay.TotalHours;
            var radians = (decimalHour / 24.0) * TwoPi;

            return ((float)Math.Sin(radians), (float)Math.Cos(radians));
        }

        /// <summary>
        /// Transformuje dzień tygodnia na dwie zmienne cykliczne.
        /// </summary>
        public static (float Sin, float Cos) Encode(DayOfWeek dayOfWeek)
        {
            var dayInt = (int)dayOfWeek;
            var radians = (dayInt / 7.0) * TwoPi;

            return ((float)Math.Sin(radians), (float)Math.Cos(radians));
        }

        // ====================================================================
        // NOWE METODY: OBSŁUGA UNIX TIMESTAMP
        // ====================================================================

        /// <summary>
        /// Transformuje standardowy Unix Timestamp (w sekundach) na zmienne cykliczne pory dnia.
        /// Operuje w przestrzeni UTC dla bezpieczeństwa w systemach rozproszonych.
        /// </summary>
        public static (float Sin, float Cos) EncodeFromUnixSeconds(long unixTime)
        {
            var utcTime = ToUtc(unixTime);

            return Encode(utcTime.TimeOfDay);
        }

        /// <summary>
        /// Transformuje Unix Timestamp (w milisekundach) na zmienne cykliczne pory dnia.
        /// Przydatne, gdy dane spływają prosto z Prometheusa, Grafany lub JavaScriptu.
        /// </summary>
        public static (float Sin, float Cos) EncodeFromUnixMilliseconds(long unixTime)
        {
            var utcTime = ToUtc(unixTime);

            return Encode(utcTime.TimeOfDay);
        }

        /// <summary>
        /// Transformuje standardowy Unix Timestamp (w sekundach) na zmienne cykliczne dnia tygodnia (UTC).
        /// </summary>
        public static (float Sin, float Cos) EncodeDayOfWeekFromUnixSeconds(long unixTimeSeconds)
        {
            var utcTime = ToUtc(unixTimeSeconds);

            return Encode(utcTime.DayOfWeek);
        }

        /// <summary>
        /// Metoda typu "All-in-One". Pobiera Unix Timestamp (sekundy) i zwraca 
        /// komplet 4 cech cyklicznych naraz. Idealne prosto do pipeline'u danych.
        /// </summary>
        public static (float HourSin, float HourCos, float DaySin, float DayCos) EncodeAllTimeFeaturesFromUnixSeconds(long unixTimeSeconds)
        {
            var utcTime = ToUtc(unixTimeSeconds);

            var (sin, cos) = Encode(utcTime.TimeOfDay);
            var (daySin, dayCos) = Encode(utcTime.DayOfWeek);

            return (sin, cos, daySin, dayCos);
        }

        private static DateTime ToUtc(long unixMilliseconds)
        {
            return DateTimeOffset.FromUnixTimeMilliseconds(unixMilliseconds).UtcDateTime;
        }

    }
}