using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.Diagnostics
{
    /// <summary>
    /// Allocation-free stopwatch stored as a readonly struct.
    /// </summary>
    public readonly struct ValueStopwatch
    {
        private static readonly double TimestampToTicks = TimeSpan.TicksPerSecond / (double)Stopwatch.Frequency;
        private readonly long _startTimestamp;

        public bool IsActive
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _startTimestamp != 0;
        }

        private ValueStopwatch(long startTimestamp)
        {
            _startTimestamp = startTimestamp;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ValueStopwatch StartNew()
        {
            return new ValueStopwatch(Stopwatch.GetTimestamp());
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TimeSpan GetElapsedTime()
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
}