// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Runtime
{
    /// <summary>
    /// The current instant, as a dependency rather than a static call.
    ///
    /// <para><b>Why anything that decides on elapsed time must take this.</b> A component that reads
    /// <c>DateTime.UtcNow</c> directly cannot be tested at the durations it actually ships with. The alert
    /// cooldown is the worked example: its default is five minutes, and the only test that exercised
    /// expiry had to reconfigure it to 30 ms and sleep 60 ms — so the shipped duration was never covered,
    /// and the test was a wall-clock race of the same family as the one that fails this suite under load.</para>
    ///
    /// <para><b>And a replay is not a clock.</b> The anomaly guard replays recorded history through
    /// <c>RunCycleAsync(DateTimeOffset)</c>, feeding a day of data through in seconds. Anything downstream
    /// that reads the wall clock is then measuring against a different timeline than the data it is
    /// judging — a cooldown that suppresses everything, or a timestamp claiming last week's incident
    /// happened just now.</para>
    ///
    /// <para><b>Ours, deliberately, rather than <c>System.TimeProvider</c>.</b> The BCL has already churned
    /// this abstraction once — ASP.NET Core's <c>ISystemClock</c> was obsoleted in favour of
    /// <c>TimeProvider</c> — and a public library that hands its own interface to callers does not have to
    /// pass that churn on. Adapting to whatever the platform offers is one small class; changing a public
    /// contract is not. <see cref="SystemClock"/> is that adapter today.</para>
    ///
    /// <para><b>Not for measuring durations.</b> Use <c>ValueStopwatch</c>, which is allocation-free and
    /// monotonic. This interface answers "what time is it", which is a different question and the only one
    /// a fake needs to control.</para>
    /// </summary>
    public interface IClock
    {
        /// <summary>The current instant in UTC.</summary>
        DateTimeOffset UtcNow
        {
            get;
        }
    }
}
