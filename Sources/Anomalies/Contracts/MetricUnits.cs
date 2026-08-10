// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// What each built-in channel promises its consumers, and what a given binding actually produces.
    ///
    /// <para><b>Two numbers that are both correct can still not be the same quantity.</b> Every threshold,
    /// floor and feature in this subsystem is written against the promise in <see cref="Declared"/>; a binding
    /// renders to PromQL whose unit follows from the series and the kind. When those disagree the guard keeps
    /// reporting, the queries stay valid, and only the meaning is wrong — see <c>ErrorRate</c>, bound to a
    /// plain error counter for nine days while its contract said "fraction on [0,1]".</para>
    ///
    /// <para><b>What is decidable here, and what is not.</b> A <see cref="MetricSourceKind.Gauge"/> passes the
    /// series through, so its unit is the series' unit and no config file can reveal it — those are accepted
    /// rather than checked. What IS decidable is the rest, and it is where the mistakes have been:
    /// <c>rate()</c> over a <c>_seconds_total</c> series is seconds per second, which is dimensionless —
    /// a fraction of time, or cores when the seconds are CPU seconds. <c>rate()</c> over a counter of
    /// <i>things</i> is things per second, which is never a fraction however the channel is named.</para>
    /// </summary>
    public static class MetricUnits
    {
        /// <summary>
        /// The unit a channel promises. Adding a <see cref="MetricIndex"/> member without adding it here
        /// throws, which is deliberate — the decision is cheap now and expensive after a threshold is
        /// calibrated against the wrong quantity.
        /// </summary>
        public static MetricUnit Declared(MetricIndex metric)
        {
            return metric switch
            {
                // CORES, not a share of the limit, and the name is the part that is wrong.
                //
                // Decided 2026-08-10 (AN-D13) after the alternative was measured and refused. The obvious
                // repair — divide by the pod's CPU limit — is not available: cAdvisor exports no
                // `container_spec_cpu_quota` here at all, and while kube-state-metrics does carry the limit,
                // **24 of 42 pods in the lab cluster have no CPU limit set**, including the guard itself.
                // Dividing would yield NO SERIES for those, so the channel would go dark on exactly the pods
                // that have no limit — and absence reads as health, which is the failure this subsystem
                // exists to prevent.
                //
                // Fixing the contract instead costs nothing in data and removes the lie. It is safe because
                // the only consumers today are the RELATIVE families, which rank and are indifferent to
                // scale: no absolute rule targets this channel, in `AnomalyGuardOptions.DefaultRules` or in
                // any shipped config. **The day somebody wants a threshold meaning "80% of the limit", this
                // decision has to be revisited** — that is the cost, and it is deferred, not avoided.
                //
                // The enum member is deliberately NOT renamed: `MetricIndex` values are persisted as the
                // `MetricTypeId` byte on every stored series, and the member names are the keys in
                // `guard.json`. Renaming would invalidate historical data and every deployed configuration to
                // correct a comment.
                MetricIndex.CpuUsageRatio => MetricUnit.Cores,
                MetricIndex.CpuThrottleRatio => MetricUnit.Fraction,
                MetricIndex.MemoryWorkingSetBytes => MetricUnit.Bytes,
                MetricIndex.OomEventsRate => MetricUnit.PerSecond,
                MetricIndex.LatencyP50Ms => MetricUnit.Milliseconds,
                MetricIndex.LatencyP95Ms => MetricUnit.Milliseconds,
                MetricIndex.LatencyP99Ms => MetricUnit.Milliseconds,
                MetricIndex.RequestsPerSecond => MetricUnit.PerSecond,
                MetricIndex.ErrorRate => MetricUnit.Fraction,
                MetricIndex.GcGen2HeapBytes => MetricUnit.Bytes,
                MetricIndex.GcPauseRatio => MetricUnit.Fraction,
                MetricIndex.ThreadPoolQueueLength => MetricUnit.Count,
                MetricIndex.ContainerRestarts => MetricUnit.EventsInWindow,
                _ => throw new ArgumentOutOfRangeException(
                    nameof(metric),
                    metric,
                    "No declared unit. A new channel must state what its numbers are before anything can "
                    + "threshold them."),
            };
        }

        /// <summary>
        /// What a binding produces, from its source series and kind — or <c>null</c> when it cannot be told
        /// from configuration alone.
        ///
        /// <para>Null is returned rather than a guess for a gauge and for a hand-written query. Both are
        /// legitimate and neither is inspectable here; a caller that treats null as "matches" is claiming
        /// something this method did not say.</para>
        /// </summary>
        /// <param name="source">Series name as the binding gives it.</param>
        /// <param name="kind">How the map will wrap it.</param>
        /// <param name="hasExplicitQuery">Whether the binding carries verbatim PromQL, which wins over kind.</param>
        public static MetricUnit? Produced(string source, MetricSourceKind kind, bool hasExplicitQuery)
        {
            ArgumentNullException.ThrowIfNull(source);

            if (hasExplicitQuery)
            {
                return null;
            }

            switch (kind)
            {
                case MetricSourceKind.HistogramSeconds:
                    return MetricUnit.Milliseconds;

                case MetricSourceKind.Ratio:
                    return MetricUnit.Fraction;

                case MetricSourceKind.EventCount:
                    return MetricUnit.EventsInWindow;

                case MetricSourceKind.Counter:
                    // The whole point of the check. Seconds per second is dimensionless; anything else per
                    // second is a rate of things and cannot be a fraction whatever the channel is called.
                    if (!source.EndsWith("_seconds_total", StringComparison.Ordinal))
                    {
                        return MetricUnit.PerSecond;
                    }

                    // CPU seconds accumulate faster than wall clock on more than one core, so they are cores,
                    // not a share of anything — until something divides by a limit, which a bare kind cannot.
                    return source.Contains("cpu", StringComparison.OrdinalIgnoreCase)
                        ? MetricUnit.Cores
                        : MetricUnit.Fraction;

                default:
                    // Gauge, and anything added later: the series carries its own unit and configuration
                    // cannot show it.
                    return null;
            }
        }
    }
}
