// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    /// Watches a deployment that is believed to be healthy and works out, per signal, how large a difference
    /// has to be before it is worth reporting.
    ///
    /// <para><b>This is what a shadow week is for.</b> The first question anyone asks about the absolute
    /// gates is "what do I put here", and until now the honest answer was that only they could know — true,
    /// and useless. The floors are not knowable in advance, but they are <i>measurable</i>: run against the
    /// cluster, watch what normal looks like, and set the bar above it.</para>
    ///
    /// <para><b>Both quantities are computed exactly as the gates compute them</b>, or the proposal would be
    /// in the wrong units. The peer gap is a pod's median against its peers' median, which is what
    /// <c>MinAbsoluteGap</c> is compared to; the trend change is the fitted slope multiplied by the window,
    /// which is what <c>MinAbsoluteTrendChange</c> is compared to.</para>
    ///
    /// <para><b>The one thing that can ruin it: the period has to have been healthy.</b> A real fault inside
    /// the observation window raises the maximum, the floor is set above the fault, and the guard is
    /// permanently blind to that fault at that size. This is the same failure as calibrating a generator
    /// against a contaminated recording — fast, repeatable, and wrong — and it is why the proposal is a
    /// suggestion for a human to accept, never something applied on its own.</para>
    ///
    /// <para>Accumulates across cycles; call <see cref="Observe"/> once per window and
    /// <see cref="Propose"/> whenever a report is wanted. Not thread-safe.</para>
    /// </summary>
    public sealed class FloorCalibrator
    {
        /// <summary>Headroom over the largest healthy observation, for what a week did not happen to show.</summary>
        private const double Margin = 1.25;

        private readonly BoundedSamples[] _peerGaps;
        private readonly BoundedSamples[] _trendChanges;
        private readonly BoundedSamples[] _magnitudes;
        private readonly TrendDetector _trend = new();
        private readonly TrendOptions _trendOptions;

        /// <summary>
        /// Accumulates into <see cref="BoundedSamples"/> rather than plain lists, and that is a fix rather
        /// than a style choice: the first version appended one value per pod per metric per cycle for as long
        /// as the process ran — about a million doubles across a shadow week on twelve replicas, and eight
        /// million on a hundred, with no ceiling. The maximum, which is what a floor is actually set from,
        /// stays exact; only the percentiles become sampled.
        /// </summary>
        public FloorCalibrator(TrendOptions? trendOptions = null)
        {
            _trendOptions = trendOptions ?? TrendOptions.Balanced;

            var count = (int)MetricIndex.Count;
            _peerGaps = new BoundedSamples[count];
            _trendChanges = new BoundedSamples[count];
            _magnitudes = new BoundedSamples[count];

            for (var i = 0; i < count; i++)
            {
                _peerGaps[i] = new BoundedSamples();
                _trendChanges[i] = new BoundedSamples();
                _magnitudes[i] = new BoundedSamples();
            }
        }

        /// <summary>Folds one window into the accumulated picture.</summary>
        public void Observe(MetricWindow window)
        {
            ArgumentNullException.ThrowIfNull(window);

            var pods = window.Pods.Count;

            if (pods == 0 || window.Length == 0)
            {
                return;
            }

            var times = new double[window.Length];
            window.WriteTimestampSeconds(times);

            var windowSeconds = times[^1] - times[0];
            var medians = new double[pods];

            for (var m = 0; m < (int)MetricIndex.Count; m++)
            {
                var metric = (MetricIndex)m;
                var usable = 0;

                for (var pod = 0; pod < pods; pod++)
                {
                    var series = window.Series(pod, metric);
                    var median = Median(series);

                    medians[pod] = median;

                    if (!double.IsFinite(median))
                    {
                        continue;
                    }

                    usable++;
                    _magnitudes[m].Add(Math.Abs(median));

                    // The same fitted change the trend gate is compared against, so the proposal lands in the
                    // units the gate reads. A verdict is not needed — only the slope.
                    var verdict = _trend.Detect(series, times, _trendOptions);

                    if (double.IsFinite(verdict.SlopePerSecond) && windowSeconds > 0.0)
                    {
                        _trendChanges[m].Add(Math.Abs(verdict.SlopePerSecond) * windowSeconds);
                    }
                }

                // A gap needs someone to be apart FROM. Below three there is no "rest of the group", which is
                // the same bound the peer detector itself refuses under.
                if (usable < 3)
                {
                    continue;
                }

                for (var pod = 0; pod < pods; pod++)
                {
                    if (!double.IsFinite(medians[pod]))
                    {
                        continue;
                    }

                    var others = MedianOfOthers(medians, pod);

                    if (double.IsFinite(others))
                    {
                        _peerGaps[m].Add(Math.Abs(medians[pod] - others));
                    }
                }
            }
        }

        /// <summary>The proposal so far, one entry per metric, indexed by <see cref="MetricIndex"/>.</summary>
        public FloorProposal[] Propose()
        {
            var proposals = new FloorProposal[(int)MetricIndex.Count];

            for (var m = 0; m < proposals.Length; m++)
            {
                var gaps = _peerGaps[m];
                var changes = _trendChanges[m];
                var magnitudes = _magnitudes[m];

                var gapMax = gaps.Count > 0 ? gaps.Max : 0.0;
                var changeMax = changes.Count > 0 ? changes.Max : 0.0;

                // Counted events are observed and reported like everything else, and proposed for by nobody.
                // The observations are still worth reading — how often peers differ by a restart is a real
                // fact about the cluster — but turning that fact into a floor would set the bar above a single
                // restart, which is the event the signal exists to report. See PeerSignalCatalog.
                var fittable = !PeerSignalCatalog.IsCountedEvent((MetricIndex)m);

                proposals[m] = new FloorProposal(
                    magnitudes.Count,
                    magnitudes.Quantile(0.5),
                    gaps.Quantile(0.99),
                    gapMax,
                    changes.Quantile(0.99),
                    changeMax,
                    fittable ? gapMax * Margin : 0.0,
                    fittable ? changeMax * Margin : 0.0);
            }

            return proposals;
        }

        private static double MedianOfOthers(double[] medians, int skip)
        {
            var others = new List<double>(medians.Length - 1);

            for (var i = 0; i < medians.Length; i++)
            {
                if (i != skip && double.IsFinite(medians[i]))
                {
                    others.Add(medians[i]);
                }
            }

            if (others.Count == 0)
            {
                return double.NaN;
            }

            others.Sort();

            return Quantile(others, 0.5);
        }

        private static double Median(ReadOnlySpan<double> values)
        {
            var finite = new List<double>(values.Length);

            for (var i = 0; i < values.Length; i++)
            {
                if (double.IsFinite(values[i]))
                {
                    finite.Add(values[i]);
                }
            }

            if (finite.Count == 0)
            {
                return double.NaN;
            }

            finite.Sort();

            return Quantile(finite, 0.5);
        }

        /// <summary>Nearest-rank quantile of an already sorted list; zero when there is nothing to rank.</summary>
        private static double Quantile(List<double> sorted, double q)
        {
            if (sorted.Count == 0)
            {
                return 0.0;
            }

            var index = (int)(q * (sorted.Count - 1));

            return sorted[Math.Clamp(index, 0, sorted.Count - 1)];
        }
    }
}
