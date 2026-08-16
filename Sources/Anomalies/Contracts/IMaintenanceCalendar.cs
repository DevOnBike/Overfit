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
        /// <param name="at">The instant being judged — the END of the evaluated window, not the moment of
        /// the call, so a cycle that runs late still asks about the period it actually looked at.</param>
        /// <param name="workload">The deployment the finding is about. Empty matches only windows that are
        /// themselves unscoped; a window naming a workload must not suppress a different one.</param>
        /// <param name="reason">
        /// The operator's words, or a stand-in. Never empty when this returns <c>true</c>: "suppressed" with
        /// no explanation is indistinguishable from a bug six weeks later.
        /// </param>
        bool IsDeclaredAbnormal(DateTimeOffset at, string workload, out string reason);

        /// <summary>
        /// Whether any window this calendar knows about is scoped to a named workload, rather than covering
        /// everything in the namespace.
        ///
        /// <para><b>Asked once at startup, to catch a contradiction that is otherwise silent.</b> A
        /// workload-scoped window compared against an empty workload can never match: the operator declares a
        /// window for their rollout, the guard pages them during it anyway, and nothing anywhere says why. The
        /// combination is detectable before the first cycle, and detecting it there costs one check against a
        /// silence nobody would ever attribute to configuration.</para>
        ///
        /// <para>Defaulted to <c>false</c> so a calendar that cannot enumerate its windows - one backed by a
        /// deployment pipeline, for instance - keeps compiling and simply declines to make the claim.</para>
        /// </summary>
        bool HasWorkloadScopedWindow => false;
    }
}
