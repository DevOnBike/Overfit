// ReSharper disable CheckNamespace once
namespace System
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