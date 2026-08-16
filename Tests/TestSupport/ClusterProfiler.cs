// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;

namespace DevOnBike.Overfit.Tests.TestSupport
{
    /// <summary>
    /// Reduces a cluster — recorded or generated — to the few numbers the generator is calibrated on, and
    /// scores one against the other.
    ///
    /// <para>See <c>docs/autoresearch-program.md</c> for what is scored and, more importantly, what is not.
    /// The short version: within-pod spread only. Between-pod spread is a range over three draws on the lab
    /// side, and an objective built on it would have the search chasing a coin flip.</para>
    /// </summary>
    public static class ClusterProfiler
    {
        /// <summary>Channels present on both sides and worth comparing.</summary>
        public static readonly MetricIndex[] Scored =
        [
            MetricIndex.CpuUsageRatio,
            MetricIndex.MemoryWorkingSetBytes,
            MetricIndex.LatencyP50Ms,
            MetricIndex.LatencyP95Ms,
            MetricIndex.LatencyP99Ms,
            MetricIndex.RequestsPerSecond,
            MetricIndex.GcGen2HeapBytes,
        ];

        private const int MinimumSamples = 8;

        /// <summary>Within-pod spread of one channel, at the median pod. Null when nothing usable is there.</summary>
        public static Spread? FromWindow(MetricWindow window, IReadOnlyList<int> pods, MetricIndex metric)
        {
            var series = new List<double[]>(pods.Count);

            for (var i = 0; i < pods.Count; i++)
            {
                series.Add(window.Series(pods[i], metric).ToArray());
            }

            return Characterise(series);
        }

        /// <summary>The same, over the tail of a generated cluster, matched to the reference window length.</summary>
        public static Spread? FromGenerator(SyntheticCluster cluster, MetricIndex metric, int samples)
        {
            var start = Math.Max(0, cluster.Samples - samples - 1);
            var length = Math.Min(samples, cluster.Samples - start);
            var series = new List<double[]>(cluster.Pods);

            for (var p = 0; p < cluster.Pods; p++)
            {
                series.Add(cluster.Series(p, metric).AsSpan(start, length).ToArray());
            }

            return Characterise(series);
        }

        /// <summary>
        /// The generator's spread at the median of several seeds, so one random draw cannot pass for a shape.
        /// </summary>
        public static Spread? OverSeeds(int pods, int seeds, MetricIndex metric, int samples, SyntheticClusterShape shape)
        {
            var ranges = new List<double>(seeds);
            var iqrs = new List<double>(seeds);
            var retraces = new List<double>(seeds);

            for (var s = 0; s < seeds; s++)
            {
                var cluster = new SyntheticCluster(
                    pods, hours: 6, scrapeSeconds: 15.0, seed: 20260729 + s, shape: shape);

                if (FromGenerator(cluster, metric, samples) is not { } spread)
                {
                    continue;
                }

                ranges.Add(spread.Range);
                iqrs.Add(spread.Iqr);
                retraces.Add(spread.Retrace);
            }

            if (ranges.Count == 0)
            {
                return null;
            }

            return new Spread(Median(ranges), Median(iqrs), Median(retraces));
        }

        /// <summary>
        /// Mean absolute log-ratio between generated and recorded spreads, over every scored channel and both
        /// statistics. Lower is better; zero is exact agreement.
        ///
        /// <para><b>Log-ratio so that twice-too-large and half-as-large cost the same.</b> A plain difference
        /// would let a search buy its score on whichever channel happens to carry the biggest numbers and
        /// ignore the rest, which is not "closer to the lab" in any sense anyone cares about.</para>
        ///
        /// <para>Channels absent from the recording are skipped rather than scored as zero — the lab exports
        /// no GC pause ratio, and pretending agreement there would reward the generator for a column nobody
        /// can check.</para>
        /// </summary>
        public static double Score(
            MetricWindow lab,
            IReadOnlyList<int> healthy,
            SyntheticClusterShape shape,
            int pods = 3,
            int seeds = 16)
        {
            var total = 0.0;
            var terms = 0;

            foreach (var metric in Scored)
            {
                if (FromWindow(lab, healthy, metric) is not { } reference)
                {
                    continue;
                }

                if (OverSeeds(pods, seeds, metric, lab.Length, shape) is not { } generated)
                {
                    continue;
                }

                total += LogRatio(generated.Iqr, reference.Iqr);
                total += LogRatio(generated.Range, reference.Range);

                // Absolute difference, not a log ratio, and deliberately so: this term is already a bounded
                // fraction and its informative end is ZERO. A log ratio would blow up exactly where the
                // statistic is most meaningful — a monotone series against a cyclic one — which is the
                // comparison it was added to make.
                total += Math.Abs(generated.Retrace - reference.Retrace);
                terms += 3;
            }

            return terms == 0 ? double.PositiveInfinity : total / terms;
        }

        /// <summary>
        /// Distance between two spreads on a log scale. A spread of zero carries no scale, so it is floored
        /// rather than turned into an infinity that would dominate every average it appears in — the lab's
        /// gen2 heap has an interquartile spread of exactly zero, and that is a fact about the metric, not a
        /// reason to discard the run.
        /// </summary>
        private static double LogRatio(double generated, double reference)
        {
            const double Floor = 1e-4;

            var a = Math.Max(generated, Floor);
            var b = Math.Max(reference, Floor);

            return Math.Abs(Math.Log(a / b));
        }

        private static Spread? Characterise(List<double[]> series)
        {
            var ranges = new List<double>(series.Count);
            var iqrs = new List<double>(series.Count);
            var retraces = new List<double>(series.Count);

            foreach (var values in series)
            {
                var finite = new List<double>(values.Length);

                // Time order, kept until the drawdown has been taken — sorting first would erase the only
                // property this whole statistic exists to measure.
                foreach (var value in values)
                {
                    if (double.IsFinite(value))
                    {
                        finite.Add(value);
                    }
                }

                if (finite.Count < MinimumSamples)
                {
                    continue;
                }

                var drawdown = MaxDrawdown(finite);

                finite.Sort();
                var median = Quantile(finite, 0.5);

                if (Math.Abs(median) <= 1e-12)
                {
                    continue;
                }

                var range = finite[finite.Count - 1] - finite[0];

                ranges.Add(range / median);
                iqrs.Add((Quantile(finite, 0.75) - Quantile(finite, 0.25)) / median);
                retraces.Add(range <= 1e-12 ? 0.0 : drawdown / range);
            }

            if (ranges.Count < 2)
            {
                return null;
            }

            return new Spread(Median(ranges), Median(iqrs), Median(retraces));
        }

        /// <summary>
        /// The largest fall from any preceding high, in the series' own units and in time order.
        ///
        /// <para><b>The order-sensitive statistic, added because its absence let the generator be wrong while
        /// scoring perfectly.</b> Range and interquartile spread are both computed on sorted values, so they
        /// cannot tell a monotone climb from a sawtooth of the same amplitude — any permutation of a window
        /// scores identically. Memory was fitted to within a tenth of a percent on both and still produced a
        /// completely different signal in time: the generator sawed and reset, the lab climbed and held.</para>
        ///
        /// <para>Divided by the range it becomes a bounded ratio that separates exactly those two shapes: near
        /// <b>0</b> for a series that never gives back what it gained (the lab's working set measured 0.0%),
        /// near <b>1</b> for anything stationary or cyclic, because a reset or a noise excursion retraces the
        /// whole span. That is also the property the trend detector keys on, which is why a generator that
        /// gets it wrong cannot be used to measure trend false positives at all.</para>
        /// </summary>
        private static double MaxDrawdown(List<double> ordered)
        {
            var peak = ordered[0];
            var worst = 0.0;

            for (var i = 1; i < ordered.Count; i++)
            {
                peak = Math.Max(peak, ordered[i]);
                worst = Math.Max(worst, peak - ordered[i]);
            }

            return worst;
        }

        private static double Median(List<double> values)
        {
            values.Sort();

            return Quantile(values, 0.5);
        }

        private static double Quantile(List<double> sorted, double q)
            => sorted[Math.Clamp((int)(q * (sorted.Count - 1)), 0, sorted.Count - 1)];

        /// <summary>
        /// The median pod's shape: two measures of how far it moves, and one of whether it comes back.
        /// </summary>
        /// <param name="Range">Full span, relative to the pod's own median.</param>
        /// <param name="Iqr">Interquartile spread, same normalisation.</param>
        /// <param name="Retrace">
        /// Largest drawdown as a fraction of the range. 0 for a series that only climbs, ~1 for anything
        /// stationary or cyclic. The only statistic here that can see time order — see <c>MaxDrawdown</c>.
        /// </param>
        public readonly record struct Spread(double Range, double Iqr, double Retrace);
    }
}
