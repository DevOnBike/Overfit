// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// Asked, for each finding, whether an operator has said they do not want to hear this one.
    ///
    /// <para><b>One choke point, deliberately.</b> Findings arrive from five families down six code paths,
    /// and checking suppression at each would leave the guarantee spread across six places that have to stay
    /// in step. <c>IncidentPipeline</c> is where every finding already passes, so the question is asked once,
    /// counted once, and cannot be forgotten by a family added later.</para>
    ///
    /// <para><b>A suppressed finding is dropped, not hidden.</b> It never becomes an incident, so nothing
    /// pages — that is what the operator asked for. What it does not do is vanish: the count is on the
    /// telemetry endpoint, because a mute nobody can see is indistinguishable from a detector that stopped
    /// working, which is the failure this subsystem exists to make loud.</para>
    /// </summary>
    public interface ISignalSuppressor
    {
        /// <summary>Whether <paramref name="signal"/> on <paramref name="subject"/> is muted at <paramref name="at"/>.</summary>
        /// <param name="subject">Namespace, workload and pod the finding is about. A pod-scoped mute covers
        /// that pod; one with no pod covers the workload and its replicas.</param>
        /// <param name="signal">The channel name as findings carry it, matched exactly.</param>
        /// <param name="at">When the finding was made, so an expired mute stops applying without anyone
        /// having to remember to remove it.</param>
        /// <param name="magnitude">
        /// The finding's size in the signal's own units, so a mute opened on a small one does not hide a
        /// large one. See <see cref="SignalSuppression.Magnitude"/> — measured, that omission silenced a whole
        /// channel.
        /// </param>
        bool IsSuppressed(
            in IncidentSubject subject, string signal, DateTimeOffset at, double magnitude = double.NaN);
    }
}
