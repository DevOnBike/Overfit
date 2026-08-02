// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// Where the "how large is large enough to report" numbers come from, per signal.
    ///
    /// <para><b>A seam because there are already two answers and a third is coming.</b> Today a floor is
    /// whatever the operator configured, falling back to what a healthy period turned out to look like. The
    /// third is the operational one — "we do not get up for less than fifty milliseconds, whatever your noise
    /// floor says" — which is a policy nobody but the customer can state, and which the calibrator correctly
    /// refuses to invent: measured on the lab, it proposed a latency floor of <b>0.01 ms</b>, which is exactly
    /// what healthy does and is not a threshold anyone would act on.</para>
    ///
    /// <para><b>Two methods rather than one plus an enum</b>, because the two gates are not interchangeable
    /// and treating them as one parameterised thing is how the step detector came to be gated on the peer
    /// floor. Measured on the lab, those two differ by <b>six times</b> on the same signal — a GC sawtooth
    /// moves a heap far more across a window than two replicas differ at any instant — and the detector fired
    /// about twice an hour on a healthy cluster because of it.</para>
    ///
    /// <para>Zero means "this gate is off". It is a real answer, not a missing one, and callers must not read
    /// it as "unknown".</para>
    /// </summary>
    public interface IAbsoluteFloorSource
    {
        /// <summary>Smallest difference between replicas, in the signal's own units, worth reporting.</summary>
        double MinAbsoluteGap(MetricIndex metric);

        /// <summary>Smallest movement across a window, in the signal's own units, worth reporting.</summary>
        double MinAbsoluteTrendChange(MetricIndex metric);

        /// <summary>
        /// The same question for a channel the enum does not have. Zero means the gate is off.
        ///
        /// <para><b>The named overloads exist because the customer's own metrics were the one part of the
        /// configuration with no fallback at all.</b> A built-in signal left unconfigured falls back to what a
        /// healthy period measured; a custom binding left unconfigured simply had its gate off, which is the
        /// arrangement measured at 209 false incidents a day - on precisely the signals the customer chose to
        /// add, and therefore cares most about.</para>
        ///
        /// <para>Defaulted to zero rather than abstract so an existing implementation keeps compiling and
        /// keeps its current behaviour, which was already "no floor for custom channels".</para>
        /// </summary>
        double MinAbsoluteGap(string signal) => 0.0;

        /// <inheritdoc cref="MinAbsoluteGap(string)"/>
        double MinAbsoluteTrendChange(string signal) => 0.0;
    }
}
