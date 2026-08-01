// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Statistics
{
    /// <summary>
    /// Finds a step: a series that sat at one level and now sits at another.
    ///
    /// <para><b>This exists because a step is not a trend, and that was measured rather than assumed.</b> The
    /// detection matrix left one fault undetected by every family — CPU rising 2.5× on all twelve replicas at
    /// once. Peer comparison is structurally silent when everybody moves together, no absolute rule was
    /// configured for the signal, and the trend family, which sees the step in every window that straddles it,
    /// reported <c>Healthy</c>.</para>
    ///
    /// <para><b>And the reason it reported Healthy is worth stating, because it rules out every fix that is
    /// merely a lower threshold.</b> On a window split evenly by a step, Theil-Sen recovered the slope
    /// perfectly well — 0.94 across the window against a level of 0.55. What failed was significance.
    /// Mann-Kendall's tau counts only rank ordering, so a step scores tau ≈ 0.51 <b>no matter how large it
    /// is</b>, and the step's own autocorrelation inflates the variance the AR(1) correction applies. Measured
    /// on the same window shape: a <b>2.5× step gave p = 0.0695 and a 10× step gave p = 0.0794</b> — both just
    /// the wrong side of 0.05, and the bigger step scored <i>worse</i>. No threshold reachable from there
    /// separates a real shift from noise, because the statistic being thresholded barely responds to the
    /// magnitude at all.</para>
    ///
    /// <para><b>The remedy is a different question, not a looser gate.</b> Split the window and ask whether
    /// the two halves came from the same distribution. On the same six shapes: every step was separated at
    /// p ≤ 1.1e-3, the mid-window ones at Cliff's delta 1.00 and p = 7e-15, and a flat control was correctly
    /// left alone at delta 0.19, p = 0.92.</para>
    ///
    /// <para><b>Where it belongs.</b> Against the workload's own aggregate — the cross-peer common component —
    /// rather than per replica, because "everyone moved together" is exactly the case the other families
    /// cannot see, and running it per pod would duplicate the peer comparison while adding its false
    /// positives. It is also the answer the peer detector's own <c>Inconclusive</c> verdict has always
    /// recommended in words: <i>compare against the workload's own history</i>.</para>
    ///
    /// <para><b>What it cannot do.</b> See a shift that finished before the window opened. Once both halves
    /// sit at the new level there is nothing to compare, exactly as a finished ramp has no slope. It catches a
    /// shift while it passes through the window — with a 20-minute window and a 5-minute cadence that is three
    /// or four consecutive cycles — and the incident tracker is what keeps the resulting incident open
    /// afterwards.</para>
    /// </summary>
    public sealed class LevelShiftDetector
    {
        private readonly ITwoSampleComparer _comparer;

        /// <summary>Uses <see cref="MannWhitneyComparer.Instance"/>.</summary>
        public LevelShiftDetector()
            : this(MannWhitneyComparer.Instance)
        {
        }

        /// <param name="comparer">The two-sample test to run across the split.</param>
        public LevelShiftDetector(ITwoSampleComparer comparer)
        {
            ArgumentNullException.ThrowIfNull(comparer);
            _comparer = comparer;
        }

        /// <summary>Identifier for reports and exported metric labels.</summary>
        public string Name => "level-shift";

        /// <summary>
        /// Compares the first half of <paramref name="series"/> against the second.
        ///
        /// <para>Higher values must mean <i>worse</i>, the same contract the rest of this namespace states;
        /// the direction is reported either way, so a caller that cares about a drop reads
        /// <see cref="TrendDirection"/> rather than negating its input.</para>
        /// </summary>
        /// <param name="series">Observations in time order. Non-finite entries are skipped.</param>
        /// <param name="options">Thresholds. Use <see cref="LevelShiftOptions.Balanced"/>, not
        /// <c>default</c>.</param>
        public LevelShiftResult Detect(ReadOnlySpan<double> series, LevelShiftOptions options)
        {
            if (!options.IsValid)
            {
                throw new ArgumentException(
                    "Level-shift thresholds are not usable; start from LevelShiftOptions.Balanced.",
                    nameof(options));
            }

            using var buffer = new PooledBuffer<double>(series.Length, clearMemory: false);
            var usable = buffer.Span[..series.Length];
            var written = 0;

            for (var i = 0; i < series.Length; i++)
            {
                if (double.IsFinite(series[i]))
                {
                    usable[written++] = series[i];
                }
            }

            if (written < options.MinimumSamples || written < 4)
            {
                return new LevelShiftResult(
                    DetectionStatus.InsufficientData,
                    TrendDirection.None,
                    $"{written} usable observation(s); {Math.Max(options.MinimumSamples, 4)} are required "
                    + "before the halves can be compared.",
                    0.0,
                    1.0,
                    double.NaN,
                    double.NaN,
                    written);
            }

            // The split point is the middle, which maximises the smaller half and therefore the power of the
            // test. Searching for the best split would find one in any series — that is what a change-point
            // search does, and it needs a multiple-comparisons correction this does not have.
            var split = written / 2;
            var before = usable[..split];
            var after = usable[split..written];

            var beforeMedian = MedianOf(before);
            var afterMedian = MedianOf(after);
            var absolute = Math.Abs(afterMedian - beforeMedian);
            var relative = Math.Abs(beforeMedian) <= 1e-12
                ? double.PositiveInfinity
                : absolute / Math.Abs(beforeMedian);

            // Both directions, because a level that fell is as much a change as one that rose — a deployment
            // that halved throughput is not healthy just because the number went down.
            var rose = _comparer.Compare(before, after);
            var fell = _comparer.Compare(after, before);

            var roseIsBetterSupported = rose.PValueCandidateWorse <= fell.PValueCandidateWorse;
            var chosen = roseIsBetterSupported ? rose : fell;
            var direction = roseIsBetterSupported ? TrendDirection.Rising : TrendDirection.Falling;

            var effect = Math.Abs(chosen.EffectSize);
            var significant = chosen.PValueCandidateWorse <= options.MaxPValue;
            var large = effect >= options.MinEffectSize;

            // Two size gates, and a change must clear both where they are enabled. The relative one cannot be
            // evaluated at all when the level started at zero, which is why the absolute one exists — see
            // LevelShiftOptions.MinAbsoluteChange.
            var proportional = options.MinRelativeChange <= 0.0
                               || relative >= options.MinRelativeChange;
            var material = options.MinAbsoluteChange <= 0.0
                           || absolute >= options.MinAbsoluteChange;

            if (significant && large && proportional && material)
            {
                var word = direction == TrendDirection.Rising ? "rose" : "fell";
                var proportionText = double.IsFinite(relative)
                    ? $"{relative:P0}"
                    : "an unmeasurable proportion (it started from zero)";

                return new LevelShiftResult(
                    DetectionStatus.Anomalous,
                    direction,
                    $"The level {word} from {beforeMedian:G4} to {afterMedian:G4} part-way through the window "
                    + $"— {proportionText} of where it started, with the halves separating at delta "
                    + $"{effect:F2} (p {chosen.PValueCandidateWorse:G3}). This is a step, not a drift: it "
                    + "affects the workload as a whole rather than any one replica.",
                    effect,
                    chosen.PValueCandidateWorse,
                    beforeMedian,
                    afterMedian,
                    written);
            }

            return new LevelShiftResult(
                DetectionStatus.Healthy,
                TrendDirection.None,
                Explain(significant, large, proportional, material, effect, relative, absolute, options),
                effect,
                chosen.PValueCandidateWorse,
                beforeMedian,
                afterMedian,
                written);
        }

        /// <summary>
        /// Says which gate refused, because "healthy" has four causes here and they call for different
        /// responses — one of them is "your floor is above the change you are looking for".
        /// </summary>
        private static string Explain(
            bool significant, bool large, bool proportional, bool material,
            double effect, double relative, double absolute, LevelShiftOptions options)
        {
            if (!significant)
            {
                return "The two halves of the window are consistent with one level.";
            }

            if (!large)
            {
                return $"The halves differ consistently but only weakly (delta {effect:F2}, below "
                       + $"{options.MinEffectSize:F2}) — the level moved less than it wandered.";
            }

            if (!proportional)
            {
                var text = double.IsFinite(relative) ? $"{relative:P0}" : "immeasurable";

                return $"The level moved by {text}, below the {options.MinRelativeChange:P0} worth reporting.";
            }

            _ = material;

            return $"The level moved by {absolute:G4}, below the {options.MinAbsoluteChange:G4} worth "
                   + "reporting in this signal's own units.";
        }

        private static double MedianOf(ReadOnlySpan<double> values)
        {
            using var scratch = new PooledBuffer<double>(values.Length, clearMemory: false);
            var span = scratch.Span[..values.Length];

            values.CopyTo(span);

            return MedianSelector.MedianInPlace(span);
        }
    }
}
