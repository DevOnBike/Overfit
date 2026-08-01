// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// Whether a moment was declared abnormal on purpose.
    ///
    /// <para><b>A seam because the authority on this is usually not us.</b> A list in a configuration file
    /// works and is the default, but the thing that actually knows a deployment is happening is the deployment
    /// pipeline — and the thing that knows a maintenance is approved is a change-management system. Both are
    /// live, and a customer who has one should be able to point the guard at it rather than keep a second copy
    /// of the truth in a ConfigMap they will forget to update.</para>
    ///
    /// <para><b>Consulted every cycle</b>, so an implementation that calls out to a network must cache. A
    /// calendar that blocks the loop has replaced the problem the guard was deployed to detect with one of its
    /// own — the same rule the whole subsystem follows about its own failures.</para>
    ///
    /// <para><b>The safe answer to "I do not know" is <c>false</c>.</b> A calendar that fails open suppresses
    /// everything, and a suppression that quietly applies for ever is total, invisible deafness.</para>
    /// </summary>
    public interface IMaintenanceCalendar
    {
        /// <summary>
        /// Whether <paramref name="at"/> falls in a declared window for <paramref name="workload"/>, and why.
        /// </summary>
        /// <param name="reason">
        /// The operator's words, or a stand-in. Never empty when this returns <c>true</c>: "suppressed" with
        /// no explanation is indistinguishable from a bug six weeks later.
        /// </param>
        bool IsDeclaredAbnormal(DateTimeOffset at, string workload, out string reason);
    }
}
