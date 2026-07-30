// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Statistics
{
    /// <summary>
    /// Detects a series drifting in one direction — the memory leak at 20 MB every ten minutes, latency that
    /// settled 15% higher after a deploy, an error rate that walked from 0.1% to 0.8%, a queue that keeps
    /// growing. None of these trips a <c>usage &gt; 80%</c> alert until it is late, which is the whole point.
    ///
    /// <para><b>Three rank-based pieces, matching the rest of the family's gate.</b></para>
    /// <list type="bullet">
    /// <item><b>Magnitude — Theil-Sen slope.</b> The median of every pairwise slope. It tolerates roughly 29%
    /// of the data being garbage before it breaks, which matters because monitoring series are full of scrape
    /// gaps, restarts and spikes that would drag least-squares wherever they liked.</item>
    /// <item><b>Significance — Mann-Kendall.</b> Counts concordant minus discordant pairs, so it assumes no
    /// distribution at all. It is the same pairwise-comparison machinery as <see cref="MannWhitneyU"/>, applied
    /// against time instead of against a second sample.</item>
    /// <item><b>Effect size — Kendall's tau.</b> How <i>monotone</i> the movement is, on −1…+1 — the direct
    /// analogue of Cliff's delta in <see cref="TwoSampleComparison"/>.</item>
    /// </list>
    ///
    /// <para><b>Autocorrelation is corrected, and this is not optional.</b> Mann-Kendall assumes independent
    /// observations. Consecutive scrapes of memory or latency are anything but — each one is nearly the
    /// previous one — and feeding a correlated series to a test that assumes independence <b>manufactures
    /// significance</b>: a smoothly wandering metric produces p-values that look overwhelming while carrying
    /// almost no independent evidence. Left uncorrected, a trend detector on real Kubernetes data fires
    /// constantly, which is precisely the alert fatigue it was bought to reduce.
    ///
    /// <para>The correction estimates lag-1 autocorrelation ρ on the detrended residuals and inflates the
    /// variance of the statistic by (1+ρ)/(1−ρ), the standard effective-sample-size factor for an AR(1)
    /// process. It is an approximation with a named assumption rather than a silent fudge; ρ is clamped to
    /// 0…0.95 (negative values are ignored rather than used to <i>deflate</i> the variance, which would be
    /// anti-conservative) and reported on the result so a reader can see how much was discounted.</para></para>
    ///
    /// <para><b>Two thresholds, because either alone misleads.</b> Tau says how consistent the movement is; a
    /// series creeping up 1% but never once dipping scores near 1.0 and is not worth waking anyone. The
    /// relative-change threshold says how big it is, as a fraction of the series' own median, so one number
    /// serves bytes, seconds and counts. A trend is reported only when it clears both.</para>
    ///
    /// <para>Scratch is pooled; the detector allocates nothing on the GC heap.</para>
    /// </summary>
    public sealed class TrendDetector
    {
        /// <summary>
        /// Above this many observations the series is thinned by a fixed stride before fitting. Theil-Sen is
        /// O(n²) in pairs, and 600 points already means ~180 000 of them; thinning also reduces the
        /// autocorrelation that the variance correction has to pay for. Chosen so a five-hour window at a
        /// 30-second scrape passes through untouched.
        /// </summary>
        private const int MaxSamplesForFit = 600;

        /// <summary>A projection further out than this is not information anybody can act on.</summary>
        private static readonly TimeSpan LongestUsefulProjection = TimeSpan.FromDays(3650);

        /// <summary>Identifier for reports and exported metric labels.</summary>
        public string Name => "robust-trend";

        /// <summary>
        /// Evaluates one series. Samples whose value or timestamp is non-finite are dropped, as are samples
        /// that do not advance the clock — duplicated or out-of-order scrapes carry no slope information.
        /// </summary>
        /// <param name="values">Observations.</param>
        /// <param name="timestampsSeconds">Their timestamps in seconds, index-aligned and non-decreasing.</param>
        /// <param name="options">Thresholds; use <see cref="TrendOptions.Balanced"/> rather than <c>default</c>.</param>
        /// <param name="limit">Optional ceiling (a memory limit, an SLO) for the time-to-limit projection. Pass
        /// <see cref="double.NaN"/> — the default — to skip it.</param>
        /// <param name="expectation">
        /// Optional per-sample expectation, index-aligned with <paramref name="values"/>. When supplied, the
        /// test runs on the <b>residual</b> (observed minus expected) instead of the raw series.
        ///
        /// <para>Two sources build one, and they answer different questions:</para>
        /// <list type="bullet">
        /// <item><see cref="SeasonalBaseline.TryBuild"/> — the same phase of previous periods. Turns "is this
        /// rising?" into "is this rising <i>more than it does every day at this hour</i>?".</item>
        /// <item><see cref="CrossPeerBaseline.TryBuild"/> — the same instant across sibling replicas. Turns it
        /// into "is this rising <i>more than its siblings are</i>?", which is what separates one sick pod from
        /// a deployment that is warming up, being rolled, or simply serving a rising load.</item>
        /// </list>
        ///
        /// <para><b>This is not a refinement, it is the difference between usable and not.</b> Measured on a
        /// healthy synthetic population with a four-hour window, the raw test produced <b>2583 false incidents
        /// a day</b>, almost all of them the daily traffic curve — every one arithmetically correct and none of
        /// them a fault. Leave it empty and that behaviour is what you get; it is also all that is available
        /// before a few periods of history exist.</para>
        ///
        /// <para>The materiality gate keeps using the <i>original</i> series for its scale, because a residual
        /// is centred on zero and a relative threshold against zero is meaningless.</para>
        /// </param>
        public TrendResult Detect(
            ReadOnlySpan<double> values,
            ReadOnlySpan<double> timestampsSeconds,
            TrendOptions options,
            double limit = double.NaN,
            ReadOnlySpan<double> expectation = default)
        {
            if (values.Length != timestampsSeconds.Length)
            {
                throw new ArgumentException("Values and timestamps must be index-aligned.", nameof(timestampsSeconds));
            }

            if (!expectation.IsEmpty && expectation.Length != values.Length)
            {
                throw new ArgumentException(
                    "The expectation must be index-aligned with the observations.",
                    nameof(expectation));
            }

            if (!options.IsValid)
            {
                throw new ArgumentException(
                    "Thresholds are not usable — use TrendOptions.Balanced/Strict/FastFeedback rather than default.",
                    nameof(options));
            }

            if (values.IsEmpty)
            {
                return Undecidable(DetectionStatus.InsufficientData, "No observations.", 0);
            }

            var raw = values.Length;
            var hasExpectation = !expectation.IsEmpty;

            using var series = new PooledBuffer<double>((hasExpectation ? 3 : 2) * raw, clearMemory: false);

            var times = series.Span[..raw];
            var observations = series.Span.Slice(raw, raw);

            // The residual is what gets tested; the original series still supplies the scale the materiality
            // gate is expressed against, and the level the time-to-limit projection needs.
            var scale = double.NaN;
            var expectedAtEnd = 0.0;
            var tested = values;

            if (hasExpectation)
            {
                var residual = series.Span.Slice(2 * raw, raw);
                scale = Subtract(values, expectation, residual, out expectedAtEnd);
                tested = residual;
            }

            var count = Compact(tested, timestampsSeconds, times, observations);

            if (count == 0)
            {
                return Undecidable(
                    DetectionStatus.InsufficientData,
                    "No usable observations: every sample was non-finite or failed to advance the clock.",
                    0);
            }

            count = Thin(times, observations, count);

            if (count < options.MinimumSamples)
            {
                // Data is arriving, the window is simply not long enough yet — that resolves itself, and
                // reporting it as a configuration problem would send someone chasing nothing.
                return Undecidable(
                    DetectionStatus.WarmingUp,
                    $"Window holds {count} usable observations; {options.MinimumSamples} are required.",
                    count);
            }

            times = times[..count];
            observations = observations[..count];

            var pairs = count * (count - 1) / 2;
            using var slopeBuffer = new PooledBuffer<double>(pairs, clearMemory: false);
            using var residualBuffer = new PooledBuffer<double>(count, clearMemory: false);

            var slopes = slopeBuffer.Span;
            var residuals = residualBuffer.Span;

            var s = AccumulatePairs(times, observations, slopes);

            // Selection, not a sort: only the middle value is wanted, and at the 600-sample cap this span
            // holds 179 700 pairwise slopes. Ordering all of them measured as 99% of this method's runtime.
            var slope = MedianSelector.MedianInPlace(slopes);

            // Detrended residuals, in time order, so the autocorrelation estimate sees the noise rather than
            // the trend itself — a strong trend would otherwise read as near-perfect correlation.
            for (var i = 0; i < count; i++)
            {
                residuals[i] = observations[i] - (slope * times[i]);
            }

            var autocorrelation = Lag1Autocorrelation(residuals);

            // Must follow the autocorrelation estimate: that reads the residuals in time order, and selection
            // permutes them.
            var intercept = MedianSelector.MedianInPlace(residuals);

            // The observations are genuinely sorted rather than selected — the tie structure the variance
            // correction needs is only visible in a full ordering, so there is nothing to save here.
            observations.Sort();
            var tiedPairs = CountTiedPairs(observations, out var varianceTieTerm);
            var median = Median(observations);

            var totalPairs = (double)count * (count - 1) / 2.0;
            var variance = (((double)count * (count - 1) * ((2.0 * count) + 5.0)) - varianceTieTerm) / 18.0;
            variance *= (1.0 + autocorrelation) / (1.0 - autocorrelation);

            var z = 0.0;
            if (variance > 0.0 && s != 0)
            {
                // Continuity correction toward zero, as for any discrete statistic tested against a continuous
                // normal.
                var corrected = s > 0 ? s - 1 : s + 1;
                z = corrected / Math.Sqrt(variance);
            }

            var pValue = 1.0 - NormalDistribution.Cdf(Math.Abs(z));
            var tauDenominator = Math.Sqrt((totalPairs - tiedPairs) * totalPairs);
            var tau = tauDenominator > 0.0 ? s / tauDenominator : 0.0;

            var windowSeconds = times[count - 1] - times[0];
            var fittedChange = Math.Abs(slope) * windowSeconds;
            var fittedAtEnd = intercept + (slope * times[count - 1]) + expectedAtEnd;

            var direction = TrendDirection.None;
            if (s > 0)
            {
                direction = TrendDirection.Rising;
            }

            if (s < 0)
            {
                direction = TrendDirection.Falling;
            }

            var significant = pValue <= options.MaxPValue;
            var monotone = Math.Abs(tau) >= options.MinTau;

            // The size gate, and the case it used to get exactly backwards.
            //
            // A series sitting at zero has no meaningful median to be relative to. The previous rule skipped
            // the gate entirely when that happened — treating "the size cannot be judged" as "the size is
            // large" — and on the cluster lab that raised a severity-0.59 incident over GcPauseRatio, a
            // channel that is identically zero on healthy replicas apart from readings around 10^-5.
            //
            // Falling back to silence would be just as wrong in the other direction: a signal climbing away
            // from zero (an error rate leaving zero, a queue starting to build) is precisely what should be
            // caught, and its median is zero for most of the window that matters.
            //
            // So the scale falls back to the largest magnitude the window actually reached. That keeps
            // "climbs away from zero" material and makes "wobbles inside the noise floor" not, because a
            // wobble's fitted change is small against its own peak too. Only a window that never leaves zero
            // has no scale under either rule, and a change within such a window is numerically nothing.
            // The absolute gate is checked independently, in the signal's own units — see TrendOptions.
            var scaleIsUsable = Math.Abs(median) > 1e-12;
            var scaleForSize = scaleIsUsable ? Math.Abs(median) : PeakMagnitude(observations);
            var hasScale = scaleForSize > 1e-12;

            var material = hasScale
                           && fittedChange >= options.MinRelativeChangeOverWindow * scaleForSize
                           && fittedChange >= options.MinAbsoluteChangeOverWindow;

            if (!significant || !monotone || !material || direction == TrendDirection.None)
            {
                return new TrendResult(
                    DetectionStatus.Healthy,
                    TrendDirection.None,
                    Explain(significant, monotone, material, direction),
                    slope,
                    tau,
                    pValue,
                    count,
                    autocorrelation,
                    fittedAtEnd,
                    null);
            }

            // Reported against whichever scale the gate actually used, so the number in the message is the
            // number that decided the verdict. When the median was zero that is the window's peak, and the
            // wording says which — "180% of typical" and "180% of the window's peak" are different claims.
            var relative = fittedChange / scaleForSize;
            var timeToLimit = ProjectTimeToLimit(slope, fittedAtEnd, limit);

            return new TrendResult(
                DetectionStatus.Anomalous,
                direction,
                Describe(direction, relative, windowSeconds, tau, autocorrelation, scaleIsUsable, timeToLimit),
                slope,
                tau,
                pValue,
                count,
                autocorrelation,
                fittedAtEnd,
                timeToLimit);
        }

        /// <summary>
        /// Writes <c>observed − expected</c> into <paramref name="residuals"/>, and returns the median of the
        /// finite <b>observed</b> values — the scale the materiality gate needs, which the residual cannot
        /// supply because it is centred on zero.
        ///
        /// <para>A sample with no expectation becomes <see cref="double.NaN"/> and is dropped downstream, which
        /// is the honest reading: a phase whose history is missing has not been shown to be normal or abnormal.
        /// Substituting the observed value would silently fall back to the raw test for that sample.</para>
        /// </summary>
        /// <param name="values">The observed series.</param>
        /// <param name="expectation">Per-sample expectation, index-aligned with <paramref name="values"/>.</param>
        /// <param name="residuals">Receives <c>observed − expected</c>.</param>
        /// <param name="expectedAtEnd">Receives the last finite expectation, so the projected level can be
        /// reported in the series' own units rather than as a deviation.</param>
        private static double Subtract(
            ReadOnlySpan<double> values,
            ReadOnlySpan<double> expectation,
            Span<double> residuals,
            out double expectedAtEnd)
        {
            expectedAtEnd = 0.0;

            using var finite = new PooledBuffer<double>(values.Length, clearMemory: false);
            var kept = 0;

            for (var i = 0; i < values.Length; i++)
            {
                var observed = values[i];
                var expected = expectation[i];

                residuals[i] = double.IsFinite(observed) && double.IsFinite(expected)
                    ? observed - expected
                    : double.NaN;

                if (double.IsFinite(expected))
                {
                    expectedAtEnd = expected;
                }

                if (!double.IsFinite(observed))
                {
                    continue;
                }

                finite.Span[kept] = observed;
                kept++;
            }

            return kept == 0 ? double.NaN : MedianSelector.MedianInPlace(finite.Span[..kept]);
        }

        /// <summary>
        /// Copies finite samples whose timestamp strictly advances into the destination spans and returns how
        /// many survived.
        /// </summary>
        private static int Compact(
            ReadOnlySpan<double> values,
            ReadOnlySpan<double> timestamps,
            Span<double> times,
            Span<double> observations)
        {
            var written = 0;
            var previous = double.NegativeInfinity;

            for (var i = 0; i < values.Length; i++)
            {
                var t = timestamps[i];
                var v = values[i];

                if (!double.IsFinite(t) || !double.IsFinite(v))
                {
                    continue;
                }

                if (t <= previous)
                {
                    continue;
                }

                times[written] = t;
                observations[written] = v;
                written++;
                previous = t;
            }

            return written;
        }

        /// <summary>Strided down-sample, keeping the first and preserving order.</summary>
        private static int Thin(Span<double> times, Span<double> observations, int count)
        {
            if (count <= MaxSamplesForFit)
            {
                return count;
            }

            var stride = ((count - 1) / MaxSamplesForFit) + 1;
            var written = 0;

            for (var i = 0; i < count; i += stride)
            {
                times[written] = times[i];
                observations[written] = observations[i];
                written++;
            }

            return written;
        }

        /// <summary>
        /// Fills <paramref name="slopes"/> with every pairwise slope and returns the Mann-Kendall S statistic —
        /// concordant minus discordant pairs — from the same sweep.
        /// </summary>
        private static long AccumulatePairs(
            ReadOnlySpan<double> times,
            ReadOnlySpan<double> observations,
            Span<double> slopes)
        {
            var written = 0;
            var s = 0L;

            for (var i = 0; i < times.Length - 1; i++)
            {
                for (var j = i + 1; j < times.Length; j++)
                {
                    var difference = observations[j] - observations[i];
                    slopes[written] = difference / (times[j] - times[i]);
                    written++;

                    if (difference > 0.0)
                    {
                        s++;
                        continue;
                    }

                    if (difference < 0.0)
                    {
                        s--;
                    }
                }
            }

            return s;
        }

        /// <summary>
        /// Largest absolute value in an already-sorted span — the fallback scale when the median is zero.
        ///
        /// <para>Only the two ends can hold it, since the span is ordered, so this is a comparison rather
        /// than a sweep.</para>
        /// </summary>
        private static double PeakMagnitude(ReadOnlySpan<double> sorted)
        {
            if (sorted.Length == 0)
            {
                return 0.0;
            }

            return Math.Max(Math.Abs(sorted[0]), Math.Abs(sorted[^1]));
        }

        /// <summary>Median of an already-sorted span.</summary>
        private static double Median(Span<double> sorted)
        {
            if (sorted.Length == 0)
            {
                return 0.0;
            }

            var middle = sorted.Length / 2;

            if (sorted.Length % 2 == 1)
            {
                return sorted[middle];
            }

            return 0.5 * (sorted[middle - 1] + sorted[middle]);
        }

        /// <summary>
        /// Counts tied pairs in a sorted span (for Kendall's tau-b) and, via
        /// <paramref name="varianceTieTerm"/>, the Σ t(t−1)(2t+5) correction the Mann-Kendall variance needs.
        /// </summary>
        private static double CountTiedPairs(ReadOnlySpan<double> sorted, out double varianceTieTerm)
        {
            var tiedPairs = 0.0;
            varianceTieTerm = 0.0;

            var index = 0;
            while (index < sorted.Length)
            {
                var last = index;
                while (last + 1 < sorted.Length && sorted[last + 1] == sorted[index])
                {
                    last++;
                }

                double tied = last - index + 1;
                tiedPairs += tied * (tied - 1.0) / 2.0;
                varianceTieTerm += tied * (tied - 1.0) * ((2.0 * tied) + 5.0);

                index = last + 1;
            }

            return tiedPairs;
        }

        /// <summary>
        /// Lag-1 autocorrelation of the detrended series, clamped to 0…1. Negative correlation is reported as
        /// zero: using it would <i>shrink</i> the variance and make the test more eager, which is the wrong way
        /// to be wrong.
        /// </summary>
        private static double Lag1Autocorrelation(ReadOnlySpan<double> residuals)
        {
            if (residuals.Length < 3)
            {
                return 0.0;
            }

            var mean = 0.0;
            for (var i = 0; i < residuals.Length; i++)
            {
                mean += residuals[i];
            }

            mean /= residuals.Length;

            var covariance = 0.0;
            var varianceSum = 0.0;

            for (var i = 0; i < residuals.Length; i++)
            {
                var centred = residuals[i] - mean;
                varianceSum += centred * centred;

                if (i + 1 < residuals.Length)
                {
                    covariance += centred * (residuals[i + 1] - mean);
                }
            }

            if (varianceSum <= 0.0)
            {
                return 0.0;
            }

            var rho = covariance / varianceSum;

            // 0.95 keeps the inflation factor finite; past that the window carries so little independent
            // evidence that no correction would rescue it anyway.
            return Math.Clamp(rho, 0.0, 0.95);
        }

        private static TimeSpan? ProjectTimeToLimit(double slope, double fittedAtEnd, double limit)
        {
            if (!double.IsFinite(limit) || slope == 0.0)
            {
                return null;
            }

            var distance = limit - fittedAtEnd;

            // Already past it, or moving away from it: there is no crossing to predict.
            if (distance == 0.0 || Math.Sign(distance) != Math.Sign(slope))
            {
                return null;
            }

            var seconds = distance / slope;
            if (!double.IsFinite(seconds) || seconds <= 0.0 || seconds > LongestUsefulProjection.TotalSeconds)
            {
                return null;
            }

            return TimeSpan.FromSeconds(seconds);
        }

        private static string Explain(bool significant, bool monotone, bool material, TrendDirection direction)
        {
            if (direction == TrendDirection.None)
            {
                return "No monotone movement: rises and falls cancel out across the window.";
            }

            if (!significant)
            {
                return "Movement is not distinguishable from noise once autocorrelation is accounted for.";
            }

            if (!monotone)
            {
                return "Movement is not consistent enough to be a trend rather than drift.";
            }

            if (!material)
            {
                return "Movement is consistent but too small to matter over this window.";
            }

            return "No trend.";
        }

        private static string Describe(
            TrendDirection direction,
            double relative,
            double windowSeconds,
            double tau,
            double autocorrelation,
            bool scaleIsUsable,
            TimeSpan? timeToLimit)
        {
            var moving = direction == TrendDirection.Rising ? "rose" : "fell";

            // Two different claims, so two different words. Against the median it is "of typical"; when the
            // median was zero the gate used the window's peak instead, and saying "of typical" there would
            // describe a proportion of a number the series never held.
            var size = scaleIsUsable
                ? $"{relative * 100.0:F1}% of typical"
                : $"{relative * 100.0:F1}% of the window's peak (the median is zero, so there is no typical)";

            var window = TimeSpan.FromSeconds(windowSeconds);
            var message =
                $"Series {moving} by {size} across {window:g} (tau {tau:F2}, lag-1 autocorrelation {autocorrelation:F2}).";

            if (timeToLimit is null)
            {
                return message;
            }

            return $"{message} At this rate the limit is reached in {timeToLimit.Value:g}.";
        }

        private static TrendResult Undecidable(DetectionStatus status, string reason, int count)
            => new(status, TrendDirection.None, reason, 0.0, 0.0, 1.0, count, 0.0, double.NaN, null);
    }
}
