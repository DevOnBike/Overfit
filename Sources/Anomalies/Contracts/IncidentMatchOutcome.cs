// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// What the matcher decided about one incoming group — and, when it opened a new incident, which of the
    /// two very different reasons applied.
    /// </summary>
    public enum IncidentMatchOutcome
    {
        /// <summary>It continued an open incident.</summary>
        Continued = 0,

        /// <summary>Nothing was open to continue. The ordinary first cycle.</summary>
        NoOpenIncidents = 1,

        /// <summary>
        /// An open incident overlaps it substantially, but its <b>primary subject differs</b> — the same
        /// trouble with a different centre. This is the outcome that says the matching key is too strict for
        /// the data, rather than that the data changed.
        /// </summary>
        PrimaryChanged = 2,

        /// <summary>
        /// <b>No longer produced.</b> The matcher used to require a minimum subject overlap on top of a
        /// matching primary, and this said that second gate had failed. It cost a shadow run an incident at
        /// 0.33 against a bar of 0.34 — same pod, same fault — and was removed; see <c>IncidentTracker</c>.
        ///
        /// <para>Kept so a stored or logged trace from before the change still reads back as what it meant.
        /// A matcher that emits this again has grown a second gate, which is the regression.</para>
        /// </summary>
        OverlapTooLow = 3,

        /// <summary>Nothing open resembles it at all — a new problem.</summary>
        NoResemblance = 4,
    }
}
