// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// A pod the cluster says exists and which reported no metrics at all.
    ///
    /// <para><b>Its own type rather than a threshold rule, because nothing was measured.</b> A
    /// <see cref="SustainedThresholdResult"/> carries a breach fraction, a peak and a median, and every one of
    /// those would be a fabrication here — the whole finding is that there are no values. Forcing this into a
    /// rule's shape would put three invented numbers in front of an operator.</para>
    /// </summary>
    /// <param name="Status">Anomalous once the pod has been silent long enough; Healthy otherwise.</param>
    /// <param name="Reason">Human-readable justification, always populated.</param>
    /// <param name="SilentCycles">Consecutive evaluation cycles in which the pod reported nothing.</param>
    /// <param name="Severity">
    /// Normalised 0…1 and comparable with the other detectors'. It climbs with how long the silence has
    /// lasted, because a pod that has said nothing for an hour is a different problem from one that has said
    /// nothing for ten minutes — the first is a rollout that failed, the second may still be starting.
    /// </param>
    public readonly record struct SilentPodResult(
        DetectionStatus Status,
        string Reason,
        int SilentCycles,
        double Severity);
}
