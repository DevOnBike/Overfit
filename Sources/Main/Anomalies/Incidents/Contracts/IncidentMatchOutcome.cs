// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Incidents.Contracts
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
        /// The best candidate with a matching primary shares too few subjects. The group genuinely moved.
        /// </summary>
        OverlapTooLow = 3,

        /// <summary>Nothing open resembles it at all — a new problem.</summary>
        NoResemblance = 4,
    }
}
