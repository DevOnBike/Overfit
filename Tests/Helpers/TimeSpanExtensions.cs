// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tests.Helpers
// ReSharper restore CheckNamespace once
{
    public static class TimeSpanExtensions
    {
        public static long TotalMillisecondsLong(this TimeSpan timeSpan)
        {
            return (long)Math.Round(timeSpan.TotalMilliseconds, MidpointRounding.AwayFromZero);
        }
    }
}