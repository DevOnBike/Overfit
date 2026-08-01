// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Monitoring.Abstractions
{
    /// <summary>
    /// Which pods the cluster says exist, as opposed to which pods reported any metrics.
    ///
    /// <para><b>The difference between those two lists is a blind spot with no other way in.</b> Every
    /// detector in this subsystem judges a time series, and a pod that never started has none: it is not an
    /// outlier, it has no trend, it breaches no threshold. It is simply absent from the window, and absence
    /// is what a healthy cluster of eleven pods also looks like. A replica stuck in <c>Pending</c>,
    /// <c>ImagePullBackOff</c> or a crash loop that dies before its first scrape is invisible to the entire
    /// guard, and the operator sees nothing at all — the worst available failure mode, because silence is
    /// exactly what success sounds like.</para>
    ///
    /// <para>Separate from <see cref="IPodTopology"/> rather than added to it, so an existing implementation
    /// keeps compiling and simply does not offer the roster. The guard checks for this interface and skips
    /// the comparison when it is absent, which is the same shape as every other optional capability here.</para>
    /// </summary>
    public interface IPodRoster
    {
        /// <summary>
        /// Pods the cluster knows about, as of the last refresh.
        ///
        /// <para>Empty means "nothing known", never "no pods exist" — the distinction matters because the
        /// second reading would report every pod in the window as unexpected.</para>
        /// </summary>
        IReadOnlyList<string> KnownPods { get; }
    }
}
