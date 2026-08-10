// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.Tests.TestSupport
{
    /// <summary>
    /// A clock the test moves by hand, so a duration can be tested at the value that actually ships.
    ///
    /// <para><b>What this replaces.</b> The alert cooldown's expiry test used to reconfigure the cooldown
    /// from its shipped five minutes down to 30 ms and then <c>await Task.Delay(60)</c>. That test could
    /// only ever cover a duration nobody runs, and it fails on a loaded box for the same reason
    /// <c>TG-T12</c> does — a wall-clock margin measured in tens of milliseconds is not a margin.</para>
    ///
    /// <para>Advancing is not thread-safe by design. A test that needs concurrency should control the
    /// interleaving explicitly rather than race the clock as well.</para>
    /// </summary>
    internal sealed class ManualClock : IClock
    {
        public ManualClock(DateTimeOffset start)
        {
            UtcNow = start;
        }

        /// <summary>A fixed, obviously-synthetic instant, so a leaked real timestamp is visible in a failure.</summary>
        public ManualClock()
            : this(new DateTimeOffset(2026, 1, 1, 0, 0, 0, TimeSpan.Zero))
        {
        }

        public DateTimeOffset UtcNow
        {
            get;
            private set;
        }

        /// <summary>Moves the clock forward. Backwards is refused: a test that needs it is testing something else.</summary>
        public void Advance(TimeSpan by)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(by, TimeSpan.Zero);
            UtcNow = UtcNow.Add(by);
        }
    }
}
