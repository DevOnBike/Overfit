// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// A period the operator has declared abnormal on purpose.
    ///
    /// <para><b>Without this, the first rollout after installation spends the trust.</b> A deployment <i>is</i>
    /// a level shift, and the step detector will say so — correctly, at Cliff's delta 1.00, about something
    /// the operator did deliberately five minutes ago. A monitoring tool whose first act is to page somebody
    /// about their own change is one they mute, and a muted tool detects nothing at all.</para>
    ///
    /// <para><b>It suppresses reporting and, just as importantly, suppresses LEARNING.</b> The calibrator's
    /// one real hazard is that a fault inside the observed period raises the floor above that fault and blinds
    /// the guard to it permanently. A declared window is a period already known to be abnormal, so folding it
    /// into "what this cluster does when it is well" would be taking the one input that is certainly wrong
    /// and treating it as ground truth. Nothing observed inside a window reaches the baseline or the floors.</para>
    ///
    /// <para><b>Findings are marked, never dropped.</b> An operator looking at a failed deploy wants to know
    /// what the guard saw during it; deleting the evidence to keep the log tidy would remove exactly the
    /// record they came for. The suppression travels on the report so the host can route it away from paging
    /// while keeping it in the history.</para>
    /// </summary>
    /// <param name="From">Start, inclusive.</param>
    /// <param name="To">End, exclusive.</param>
    /// <param name="Workload">
    /// Which workload it covers. Empty means the whole scope — the honest default for a cluster-wide change
    /// such as a node pool upgrade, and the wrong one for a single deployment, which is why it is stated
    /// rather than assumed.
    /// </param>
    /// <param name="Reason">
    /// Why, in the operator's words. Carried into the report: "suppressed" without a reason is
    /// indistinguishable from a bug six weeks later.
    /// </param>
    public readonly record struct MaintenanceWindow(
        DateTimeOffset From,
        DateTimeOffset To,
        string Workload = "",
        string Reason = "")
    {
        /// <summary>Whether <paramref name="at"/> falls inside this window for <paramref name="workload"/>.</summary>
        public bool Covers(DateTimeOffset at, string workload)
        {
            if (at < From || at >= To)
            {
                return false;
            }

            return Workload.Length == 0
                   || string.Equals(Workload, workload, StringComparison.Ordinal);
        }
    }
}
