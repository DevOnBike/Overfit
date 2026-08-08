// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    /// The exact smallest and largest <b>raw sample</b> a channel has produced, across every pod and every
    /// window since the process started.
    ///
    /// <para><b>It exists because every other accumulator in the calibrator is median-based, and a median
    /// cannot see a rare event.</b> That is correct for setting a floor — a floor is about what normal looks
    /// like — but it is exactly wrong for asking whether a channel has ever moved. Measured while building
    /// the inert-channel check: a single non-zero sample injected into a 1200-observation history left
    /// <c>Magnitudes</c> constant at zero, because the median of a window containing one spike is still zero.
    /// A real OOM kill produces a non-zero rate across roughly a tenth of a window, so the channel the check
    /// was built for would have been reported dead <i>after successfully detecting a kill</i>.</para>
    ///
    /// <para><b>Deliberately not serialised.</b> <see cref="FloorCalibrator.Write"/> exists so a restart does
    /// not relearn its floors from nothing, and floors are what a stale state would damage. Inertness is a
    /// question about a binding, it is answered within hours, and carrying it across restarts would mean a
    /// format change for a judgement that is cheap to re-earn. A restarted guard simply waits out its
    /// observation threshold again.</para>
    /// </summary>
    public readonly struct ObservedRange
    {
        private ObservedRange(double min, double max)
        {
            Min = min;
            Max = max;
        }

        /// <summary>Nothing observed yet — both bounds are NaN, and <see cref="HasSamples"/> is false.</summary>
        public static ObservedRange Empty => new(double.NaN, double.NaN);

        /// <summary>The smallest finite sample seen, or NaN.</summary>
        public double Min
        {
            get;
        }

        /// <summary>The largest finite sample seen, or NaN.</summary>
        public double Max
        {
            get;
        }

        /// <summary>Whether any finite sample has been folded in.</summary>
        public bool HasSamples => double.IsFinite(Min) && double.IsFinite(Max);

        /// <summary>
        /// Whether every sample ever seen was the same value. False when nothing has been observed: an
        /// unobserved channel is unknown, not constant, and the two must not collapse.
        /// </summary>
        public bool IsConstant => HasSamples && Min.Equals(Max);

        /// <summary>Folds one series in, ignoring non-finite samples.</summary>
        public ObservedRange Fold(ReadOnlySpan<double> values)
        {
            var min = Min;
            var max = Max;

            for (var i = 0; i < values.Length; i++)
            {
                var value = values[i];

                if (!double.IsFinite(value))
                {
                    continue;
                }

                if (!double.IsFinite(min) || value < min)
                {
                    min = value;
                }

                if (!double.IsFinite(max) || value > max)
                {
                    max = value;
                }
            }

            return new ObservedRange(min, max);
        }
    }
}
