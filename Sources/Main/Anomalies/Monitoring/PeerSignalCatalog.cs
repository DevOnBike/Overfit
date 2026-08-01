// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    /// The single place a <see cref="MetricIndex"/> declares whether uneven load can explain its magnitude,
    /// and therefore whether <see cref="PeerGroupOutlierDetector"/> should divide it by a work metric before
    /// comparing peers.
    ///
    /// <para><b>This is a per-signal physical claim, not a convenience.</b> Dividing by work is right only
    /// when the quantity actually has a per-request component. Divide a quantity that does not, and the
    /// division manufactures a difference the size of the traffic imbalance — permanently, on a perfectly
    /// healthy cluster, because a replica serving 9% less traffic reports 9% more of a fixed cost per
    /// request. That is an invalid comparison, not a threshold that needs raising.</para>
    ///
    /// <para><b>Measured, and it corrected an earlier belief in this codebase.</b>
    /// <see cref="PeerSignalKind"/> listed memory alongside CPU as an example of a signal that "tracks
    /// throughput". On a healthy synthetic population, <see cref="MetricIndex.MemoryWorkingSetBytes"/> then
    /// produced the largest single block of false peer findings — 317, 269 and 35 across three seeds — at a
    /// median gap of 10–12%, which is the traffic spread the generator draws by construction and nothing
    /// else. CPU's per-request component is real and was measured across three traffic skews; a working set's
    /// is close to zero, because it is assemblies, JIT-compiled code, caches and the live set.</para>
    ///
    /// <para><b>The general answer is an affine fit, and this is its degenerate case.</b> Cost is
    /// <c>fixed + marginal × work</c>; plain division conflates the two terms and, as the measurement on
    /// <see cref="PeerSignalKind"/> records, leaves a residue that grows with imbalance and ranks the busiest
    /// replica as the cheapest. For a signal whose marginal term is ~0 the correct affine treatment is simply
    /// not to divide, which is what <see cref="PeerSignalKind.LoadIndependent"/> does. Until the affine fit
    /// exists this catalog is the honest approximation: each signal declares which term dominates it.</para>
    /// </summary>
    public static class PeerSignalCatalog
    {
        /// <summary>
        /// How <paramref name="metric"/> should be compared across peers.
        ///
        /// <para>Unknown values fall to <see cref="PeerSignalKind.LoadIndependent"/>, which is the safe
        /// default in the sense that matters: it compares the number the exporter actually reported instead of
        /// silently dividing it by something whose relationship to it nobody has established.</para>
        /// </summary>
        public static PeerSignalKind Classify(MetricIndex metric)
        {
            return metric switch
            {
                // A per-request cost component that is real and measured — see PeerSignalKind for the three
                // traffic skews it was measured across.
                MetricIndex.CpuUsageRatio => PeerSignalKind.LoadSensitive,

                // Time spent collecting is driven by allocation, and allocation is driven by requests served.
                MetricIndex.GcPauseRatio => PeerSignalKind.LoadSensitive,

                // A queue is work waiting; comparing depths under different arrival rates compares the rates.
                MetricIndex.ThreadPoolQueueLength => PeerSignalKind.LoadSensitive,

                // Everything else is either already a fraction, a rate over a common denominator, an event
                // count, or — the case this catalog exists for — a fixed cost that division would corrupt:
                // working set and gen-2 heap are assemblies, JIT'd code, caches and the live set.
                _ => PeerSignalKind.LoadIndependent,
            };
        }

        /// <summary>
        /// Whether <see cref="Classify"/> asks for a work metric, so a caller can decide whether to fetch one
        /// without repeating the switch.
        /// </summary>
        public static bool RequiresWork(MetricIndex metric)
        {
            return Classify(metric) == PeerSignalKind.LoadSensitive;
        }

        /// <summary>
        /// Whether a single occurrence of <paramref name="metric"/> is itself the finding, which makes its
        /// threshold a matter of meaning rather than of measurement.
        ///
        /// <para><b>This exists because fitting one of these to observed data sets it above a real event.</b>
        /// Measured, and it is not hypothetical: <see cref="FloorCalibrator"/> run over a healthy day proposed
        /// a <see cref="MetricIndex.ContainerRestarts"/> floor of <b>1.25</b>, because pods in that population
        /// restart about once a day and so a difference of one restart is, statistically, entirely normal. It
        /// is also exactly what the operator wants to hear about, and a floor of 1.25 makes one restart
        /// permanently unreportable. The rest of that proposal was good; this part of it was worse than
        /// nothing.</para>
        ///
        /// <para>So these signals opt out of calibration and keep whatever the operator declared. The general
        /// rule the case teaches: a quantity whose <i>scale</i> is arbitrary can be calibrated from data, and
        /// a quantity whose <i>unit</i> is already the thing you care about cannot.</para>
        /// </summary>
        public static bool IsCountedEvent(MetricIndex metric)
        {
            return metric switch
            {
                MetricIndex.ContainerRestarts => true,
                MetricIndex.OomEventsRate => true,
                _ => false,
            };
        }
    }
}
