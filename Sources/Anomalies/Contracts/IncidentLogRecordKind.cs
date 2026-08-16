// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>What one <see cref="IncidentLogRecord"/> describes.</summary>
    public enum IncidentLogRecordKind
    {
        /// <summary>
        /// The group: what an operator should be paged about, if anything is. Counting these answers "how
        /// many things went wrong"; counting findings answers "how noisy is the detector", and conflating the
        /// two has already produced one wrong conclusion in this project.
        /// </summary>
        Incident = 0,

        /// <summary>
        /// One signal inside a group — the evidence. Useful for attribution and for measuring the
        /// false-positive budget per metric, not for alerting.
        /// </summary>
        Finding = 1,
    }
}
