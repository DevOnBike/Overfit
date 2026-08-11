// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Statistics
{
    /// <summary>
    /// What a series is <i>expected</i> to read at each point of a window, taken from the same phase of previous
    /// periods.
    ///
    /// <para><b>Why this has to exist outside the window.</b> A trend detector given twenty minutes of a
    /// twenty-four-hour cycle sees a straight line, because that is all the information there is: the window
    /// covers 1.4% of the period. Measured on a healthy synthetic population, the consequence is not marginal —
    /// with a four-hour window the guard produced <b>2583 false incidents a day</b>, almost all of them
    /// <c>RequestsPerSecond</c>, because a four-hour window sits squarely on the rising or falling limb of the
    /// daily curve and that limb is a huge, perfectly monotone, statistically overwhelming trend. Every one of
    /// those findings was arithmetically correct. The detector was answering "is this series rising?" when the
    /// question is "is it rising <i>more than it does every day at this hour</i>?".</para>
    ///
    /// <para><b>Median across periods, not mean.</b> One bad day — a deploy, an incident, a load test — would
    /// drag a mean expectation and then suppress detection for the following week. A median over three or four
    /// periods absorbs it. This is the same reason every estimator in this namespace is rank-based.</para>
    ///
    /// <para><b>It reports failure rather than guessing.</b> With fewer than
    /// <c>minimumPeriods</c> complete periods behind the window there is no expectation to build, and inventing
    /// one from a single day would make the first day of operation the definition of normal. The caller then
    /// falls back to a plain trend test, which is honest: it will be noisy on a seasonal signal, and that is
    /// a known cost of having no history yet.</para>
    /// </summary>
    public static class SeasonalBaseline
    {
        /// <summary>
        /// Fewest complete periods that can produce an expectation. Two gives a median of two — the midpoint —
        /// which is weak but not arbitrary; three or more is where it starts absorbing a bad day.
        /// </summary>
        public const int MinimumPeriodsSupported = 2;

        /// <summary>
        /// Fills <paramref name="expectation"/> with the median reading at each phase of the window across the
        /// preceding periods, and returns whether there was enough history to do it.
        /// </summary>
        /// <param name="history">The full series, oldest first, evenly spaced. The evaluated window is part of it.</param>
        /// <param name="windowStart">Index in <paramref name="history"/> where the evaluated window begins.</param>
        /// <param name="windowLength">Samples in the evaluated window.</param>
        /// <param name="samplesPerPeriod">Samples in one seasonal period — 5760 for a day at a 15 s scrape.</param>
        /// <param name="expectation">Receives <paramref name="windowLength"/> values. Positions with no usable
        /// history are <see cref="double.NaN"/>, which the trend detector drops rather than treating as zero.</param>
        /// <param name="minimumPeriods">Complete periods required before an expectation is produced.</param>
        public static bool TryBuild(
            ReadOnlySpan<double> history,
            int windowStart,
            int windowLength,
            int samplesPerPeriod,
            Span<double> expectation,
            int minimumPeriods = 3)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(windowStart);
            ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(windowLength, 0);
            ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(samplesPerPeriod, 0);
            ArgumentOutOfRangeException.ThrowIfLessThan(minimumPeriods, MinimumPeriodsSupported);

            if (expectation.Length < windowLength)
            {
                throw new ArgumentException(
                    $"Destination too short: need {windowLength}, got {expectation.Length}.", nameof(expectation));
            }

            if (windowStart + windowLength > history.Length)
            {
                throw new ArgumentException("The window runs past the end of the history.", nameof(windowLength));
            }

            // How many whole periods fit before the window starts. This is the honest bound: a partial period
            // is not a period, and counting it would let the expectation lean on a few hours of yesterday.
            var available = windowStart / samplesPerPeriod;

            if (available < minimumPeriods)
            {
                return false;
            }

            using var scratch = new PooledBuffer<double>(available, clearMemory: false);
            var samples = scratch.Span;

            for (var i = 0; i < windowLength; i++)
            {
                var kept = 0;

                for (var period = 1; period <= available; period++)
                {
                    var index = windowStart + i - (period * samplesPerPeriod);

                    if (index < 0)
                    {
                        break;
                    }

                    var value = history[index];

                    if (!double.IsFinite(value))
                    {
                        continue;
                    }

                    samples[kept] = value;
                    kept++;
                }

                // A phase whose history is entirely missing gets no expectation. NaN says "unknown" and the
                // detector drops the sample; zero would say "we expected nothing here", which is a different
                // and false claim.
                expectation[i] = kept == 0
                    ? double.NaN
                    : MedianSelector.MedianInPlace(samples.Slice(0, kept));
            }

            return true;
        }

        /// <summary>Samples in one period for a given period length and scrape interval.</summary>
        public static int SamplesPerPeriod(TimeSpan period, TimeSpan scrapeInterval)
        {
            if (period <= TimeSpan.Zero || scrapeInterval <= TimeSpan.Zero)
            {
                throw new ArgumentOutOfRangeException(nameof(period), "Period and scrape interval must be positive.");
            }

            return (int)Math.Round(period.TotalSeconds / scrapeInterval.TotalSeconds);
        }
    }
}
