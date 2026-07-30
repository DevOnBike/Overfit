// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Incidents.Contracts
{
    /// <summary>Why one group of this cycle did or did not continue an open incident.</summary>
    /// <param name="PrimaryKey">Whose problem the incoming group is about.</param>
    /// <param name="SubjectCount">How many subjects it covers.</param>
    /// <param name="Outcome">What the matcher decided.</param>
    /// <param name="ContinuedId">The incident continued, or 0.</param>
    /// <param name="BestOverlap">
    /// Highest subject overlap found against any open incident, <b>ignoring the primary-key requirement</b>.
    /// That is the number the ordinary path never computes and the one that separates the two ways matching
    /// fails: a high overlap with a different primary means the incident is the same and its centre moved,
    /// while a low overlap means it really is a different group.
    /// </param>
    /// <param name="BestOverlapId">Which open incident that was.</param>
    /// <param name="BestOverlapPrimaryKey">And whose problem <i>it</i> is about.</param>
    public readonly record struct IncidentMatchTrace(
        string PrimaryKey,
        int SubjectCount,
        IncidentMatchOutcome Outcome,
        long ContinuedId,
        double BestOverlap,
        long BestOverlapId,
        string BestOverlapPrimaryKey);
}
