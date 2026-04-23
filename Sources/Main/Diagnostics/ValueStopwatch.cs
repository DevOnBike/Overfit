using System.Diagnostics;

namespace DevOnBike.Overfit.Diagnostics
{
    /// <summary>
    /// This is already familiar <see cref="Stopwatch"/> but readonly struct.
    /// Doesn't allocate memory on the managed heap.
    /// </summary>
    internal readonly struct ValueStopwatch
    {
        private static readonly double TimestampToTicks = TimeSpan.TicksPerSecond / (double)Stopwatch.Frequency;
        private readonly long _startTimestamp;
        public bool IsActive => _startTimestamp != 0;

        public TimeSpan Elapsed
        {
            get
            {
                if (!IsActive)
                {
                    throw new InvalidOperationException("An uninitialized, or 'default', ValueStopwatch cannot be used to get elapsed time.");
                }

                var end = Stopwatch.GetTimestamp();
                var timestampDelta = end - _startTimestamp;
                var ticks = (long)(TimestampToTicks * timestampDelta);

                return new TimeSpan(ticks);
            }
        }

        private ValueStopwatch(long startTimestamp)
        {
            _startTimestamp = startTimestamp;
        }

        public static ValueStopwatch StartNew()
        {
            return new(Stopwatch.GetTimestamp());
        }
    }
}