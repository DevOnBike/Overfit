// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;

namespace DevOnBike.Overfit.Anomalies.Monitoring.Abstractions
{
    /// <summary>
    /// Supplies one evaluated window at a time to the loop that drives the guard.
    ///
    /// <para><b>The range argument is the whole reason this interface exists.</b> The two older contracts in
    /// this folder — <see cref="IMetricSource"/> and <see cref="IRawMetricSource"/> — are point-in-time reads
    /// by construction: neither method takes a start, an end or an "as of", so no implementation of either can
    /// be asked for a window that has already passed. Tuning the guard's thresholds means re-evaluating a
    /// fixed body of history after each change, and against a point-in-time contract the only way to do that
    /// is to run a cluster for another day.</para>
    ///
    /// <para><b>Three members, and deliberately not four.</b> This is exactly what
    /// <c>AnomalyGuardService</c> consumes today — <see cref="ReadAsync"/> once per cycle and
    /// <see cref="StalePodsExcluded"/> straight after it. <c>SeriesReturned</c> and <c>CustomChannels</c> are
    /// left off even though <see cref="PrometheusMetricWindowSource"/> offers both, because nothing reaches
    /// them through this seam: the two diagnostics that do already hold a concrete reference. Widening an
    /// interface when a second implementer needs a member is a same-day change; narrowing one after callers
    /// have found it is not.</para>
    ///
    /// <para><see cref="IDisposable"/> is part of the contract rather than an implementation detail: the
    /// instance is a host-lifetime singleton holding a connection to whatever it reads, and the container
    /// disposes it at shutdown.</para>
    /// </summary>
    public interface IMetricWindowSource : IDisposable
    {
        /// <summary>
        /// Pods that were present in the last window's data but had stopped reporting before it ended, and
        /// were therefore left out of it.
        ///
        /// <para>Names, not a count, and the reason is in the ambiguity: a replica that vanishes is usually
        /// gone — deleted, scaled down, rolled over — but it can also be one whose scraping broke while it
        /// kept serving traffic, and those two are indistinguishable from here. The first needs no action and
        /// the second needs action today, so an implementation reports who and lets an operator tell them
        /// apart.</para>
        ///
        /// <para>Belongs to the most recent <see cref="ReadAsync"/> only. An implementation that leaves the
        /// previous read's exclusions on display after a read that returned early is reporting last cycle's
        /// cluster as though it were this one's.</para>
        /// </summary>
        IReadOnlyList<string> StalePodsExcluded
        {
            get;
        }

        /// <summary>
        /// Reads the window of length <paramref name="window"/> ending at <paramref name="end"/>.
        /// </summary>
        /// <param name="end">
        /// End of the window. A live implementation should be given a margin behind "now": rate expressions
        /// are computed over a trailing range, so samples at the current instant are still filling in — on a
        /// lab run that ended a load test, the last two minutes of every RED signal were decaying and the
        /// trend family reported a cluster-wide decline that was an artefact of when the measurement stopped.
        /// </param>
        /// <param name="window">How much history to evaluate.</param>
        /// <param name="ct">Cancellation.</param>
        /// <returns>
        /// <c>null</c> when no pod returned anything at all. <b>Not an empty window</b>: a cluster the source
        /// cannot see and a cluster with nothing running produce the same silence here, and a window of NaN
        /// would be evaluated as health by every detector below.
        /// </returns>
        Task<MetricWindow?> ReadAsync(
            DateTimeOffset end,
            TimeSpan window,
            CancellationToken ct = default);
    }
}
