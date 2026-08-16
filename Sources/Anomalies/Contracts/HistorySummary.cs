// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// What a workload has normally done on one signal, at one hour of the day.
    /// </summary>
    /// <param name="Days">Distinct days behind the figures. One day is a coincidence, not a baseline.</param>
    /// <param name="Median">The level, across those days.</param>
    /// <param name="InterquartileSpread">
    /// How much it varies between days at this hour. Present because a level without a spread cannot say
    /// whether today is unusual — a workload that swings 40% every Tuesday and one that never moves have the
    /// same median and completely different meanings.
    /// </param>
    /// <param name="Newest">The most recent observation, for a report that wants to say "today, versus usual".</param>
    public readonly record struct HistorySummary(
        int Days,
        double Median,
        double InterquartileSpread,
        double Newest)
    {
        /// <summary>Whether there is enough here to be an expectation rather than an anecdote.</summary>
        public bool IsUsable(int minimumDays) => Days >= minimumDays;
    }
}
