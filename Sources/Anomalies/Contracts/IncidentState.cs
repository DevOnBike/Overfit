// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// Where an incident sits in its life, which is the difference between a detector and something an
    /// operator can live with.
    /// </summary>
    public enum IncidentState
    {
        /// <summary>
        /// First cycle this has been seen. The only state that should ever produce a notification — every
        /// later cycle of the same problem is the same problem.
        /// </summary>
        Opened = 0,

        /// <summary>
        /// Matched to something already open. Worth updating a dashboard with, worth writing down, and
        /// <b>not</b> worth telling anyone about again: a one-hour incident evaluated every five minutes is
        /// twelve of these, and sending twelve is how an alerting product gets muted.
        /// </summary>
        Ongoing = 1,

        /// <summary>
        /// Was open, and has now been absent for long enough to call it over. Emitted once, in the cycle it
        /// closes, so a consumer can pair it with the <see cref="Opened"/> that started it.
        /// </summary>
        Resolved = 2,
    }
}
