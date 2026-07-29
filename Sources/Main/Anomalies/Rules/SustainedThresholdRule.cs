// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Rules.Contracts;
using DevOnBike.Overfit.Statistics;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Anomalies.Rules
{
    /// <summary>
    /// A signal held at or above an absolute threshold for a material share of the window. No comparison, no
    /// p-value, no baseline — the first layer of the guard, and the one that answers when the statistics
    /// cannot.
    ///
    /// <para><b>Why an absolute rule is not a lesser tool.</b> A test answers "is this difference real?", which
    /// is the wrong question when the platform is already acting on the workload. CPU throttling is the case
    /// that forced this: <c>container_cpu_cfs_throttled_periods_total</c> exists <b>only on containers that
    /// carry a CPU limit</b>, so on the cluster lab exactly one pod of four had the series at all. A peer
    /// comparison over a group of one is not a hard case, it is undefined — and that pod was the deliberately
    /// degraded one, whose p95 sat 2.75x above its siblings'. The cleanest evidence of the fault was structurally
    /// out of reach of the method most likely to be pointed at it.</para>
    ///
    /// <para><b>Persistence is the guard against the obvious failure of absolute thresholds.</b> Throttling is
    /// bursty — measured median 1.4%, p90 11.8% on the same pod in the same window — so a rule firing on one
    /// sample over the line would fire constantly. Requiring the breach to hold across a share of the window is
    /// what separates a traffic burst from a misconfigured limit. See
    /// <see cref="SustainedThresholdOptions.ForCpuThrottling"/> for both numbers and the measurement they came
    /// from.</para>
    ///
    /// <para><b>Missing is not zero.</b> Non-finite samples are dropped rather than counted as compliant: a
    /// scrape that returned nothing says nothing about whether the container was throttled, and treating it as
    /// "under the threshold" would let a broken query read as a clean bill of health. A window with no usable
    /// samples is <see cref="DetectionStatus.InsufficientData"/>, never <see cref="DetectionStatus.Healthy"/>.</para>
    ///
    /// <para>Scratch is pooled; the rule allocates nothing on the GC heap beyond its reason string.</para>
    /// </summary>
    public sealed class SustainedThresholdRule
    {
        /// <summary>Identifier for reports and exported metric labels.</summary>
        public string Name => "sustained-threshold";

        /// <summary>
        /// Evaluates one window against one threshold.
        /// </summary>
        /// <param name="values">Observations, in any order — only their values matter, not their sequence.</param>
        /// <param name="options">Threshold and persistence; use a named profile rather than <c>default</c>.</param>
        public SustainedThresholdResult Evaluate(
            ReadOnlySpan<double> values,
            SustainedThresholdOptions options)
        {
            if (!options.IsValid)
            {
                throw new ArgumentException(
                    "Thresholds are not usable — use SustainedThresholdOptions.ForCpuThrottling/ForRareEvent "
                    + "rather than default.",
                    nameof(options));
            }

            if (values.IsEmpty)
            {
                return Undecidable(DetectionStatus.InsufficientData, "No observations.", 0);
            }

            using var buffer = new PooledBuffer<double>(values.Length, clearMemory: false);
            var usable = buffer.Span;

            var kept = 0;
            var breached = 0;
            var peak = double.NegativeInfinity;

            for (var i = 0; i < values.Length; i++)
            {
                var value = values[i];

                if (!double.IsFinite(value))
                {
                    continue;
                }

                usable[kept] = value;
                kept++;

                if (value > peak)
                {
                    peak = value;
                }

                if (value >= options.Threshold)
                {
                    breached++;
                }
            }

            if (kept == 0)
            {
                return Undecidable(
                    DetectionStatus.InsufficientData,
                    "No usable observations: every sample was non-finite, which is silence rather than compliance.",
                    0);
            }

            if (kept < options.MinimumSamples)
            {
                // Data is arriving, the window is simply not long enough yet — that resolves itself, and
                // reporting it as a fault would fire on every freshly-started pod.
                return Undecidable(
                    DetectionStatus.WarmingUp,
                    $"Window holds {kept} usable observations; {options.MinimumSamples} are required.",
                    kept);
            }

            var median = MedianSelector.MedianInPlace(usable[..kept]);
            var fraction = (double)breached / kept;

            if (fraction < options.MinBreachFraction)
            {
                return new SustainedThresholdResult(
                    DetectionStatus.Healthy,
                    breached == 0
                        ? $"No sample reached {options.Threshold:G3} across {kept} observations."
                        : $"{fraction:P0} of the window reached {options.Threshold:G3}, under the {options.MinBreachFraction:P0} required to call it sustained (peak {peak:G3}).",
                    fraction,
                    breached,
                    kept,
                    peak,
                    median);
            }

            return new SustainedThresholdResult(
                DetectionStatus.Anomalous,
                $"Held at or above {options.Threshold:G3} for {fraction:P0} of the window ({breached} of {kept} observations, median {median:G3}, peak {peak:G3}).",
                fraction,
                breached,
                kept,
                peak,
                median);
        }

        private static SustainedThresholdResult Undecidable(DetectionStatus status, string reason, int usable)
            => new(status, reason, 0.0, 0, usable, double.NaN, double.NaN);
    }
}
