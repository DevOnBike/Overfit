// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;

namespace DevOnBike.Overfit.Anomalies.Hosting
{
    /// <summary>Scheduling for <see cref="AnomalyGuardService"/>, plus the guard's own options.</summary>
    public sealed class AnomalyGuardServiceOptions
    {
        /// <summary>
        /// How often to evaluate. Five minutes is the cadence every measurement in this project used.
        ///
        /// <para>Faster is not better: consecutive windows overlap, so a problem is re-observed rather than
        /// re-found, and the incident tracker absorbs that — but a cadence far below the window length just
        /// spends CPU re-deciding what it already decided.</para>
        /// </summary>
        public TimeSpan Cadence { get; init; } = TimeSpan.FromMinutes(5);

        /// <summary>
        /// How much history each cycle evaluates.
        ///
        /// <para><b>Longer is not safer, and that was measured.</b> Sweeping the window on a healthy synthetic
        /// population: 20 minutes gave 234 false incidents a day, 60 minutes gave 93, and <b>240 minutes gave
        /// 2583</b> — a four-hour window sits on the slope of the daily traffic curve, so the trend family
        /// finds a real, meaningless drift in everything at once.</para>
        /// </summary>
        public TimeSpan Window { get; init; } = TimeSpan.FromMinutes(20);

        /// <summary>
        /// How far behind "now" each window ends.
        ///
        /// <para>Rate expressions look backwards over a trailing range, so the newest samples are still being
        /// filled in. On the lab, a window ending at the instant a load run stopped reported a cluster-wide
        /// downward trend in every RED signal — an artefact of when the measurement ended, not of the
        /// cluster. Two minutes covers the default rate range with margin.</para>
        /// </summary>
        public TimeSpan EndOffset { get; init; } = TimeSpan.FromMinutes(2);

        /// <summary>
        /// Pod label the topology reader uses as the peer-group key. Empty leaves every pod in one group,
        /// which is the behaviour before cohorts existed.
        /// </summary>
        public string PeerGroupLabel { get; init; } = string.Empty;

        /// <summary>
        /// How often the guard reports what floors its own observations imply. <see cref="TimeSpan.Zero"/>
        /// turns it off.
        ///
        /// <para><b>This is the answer to "what do I put in those absolute floors", and it is on by default
        /// because the honest alternative was "only you can know".</b> True, and useless — the numbers are not
        /// knowable in advance, but they are measurable, and a guard that has been running in shadow mode has
        /// been measuring them all along. It reports only the metrics whose configured floor is <i>below</i>
        /// what a healthy period produced, so it goes quiet once the configuration catches up rather than
        /// repeating itself forever.</para>
        ///
        /// <para>An hour, because the proposal is only worth reading once enough windows are behind it and a
        /// shadow deployment runs for days.</para>
        /// </summary>
        public TimeSpan FloorProposalInterval { get; init; } = TimeSpan.FromHours(1);

        /// <summary>Detector thresholds, topology and the per-metric floors.</summary>
        public AnomalyGuardOptions Guard { get; init; } = new();

        /// <summary>Incident lifecycle: how groups are matched across cycles and when they close.</summary>
        public IncidentTrackingOptions Tracking { get; init; } = IncidentTrackingOptions.Balanced;
    }
}
