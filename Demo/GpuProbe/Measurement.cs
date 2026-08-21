// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// The timings of one arm in one cell. Min, median and max are all reported and a mean is
    /// deliberately not offered: a single reading is not a fact, and a mean hides the spread that says
    /// whether the sitting was quiet.
    /// </summary>
    internal sealed class Measurement
    {
        public Measurement(string arm, IReadOnlyList<double> samplesMs)
        {
            Arm = arm;
            Samples = samplesMs;

            var sorted = samplesMs.ToArray();
            Array.Sort(sorted);

            MinMs = sorted[0];
            MaxMs = sorted[^1];
            MedianMs = sorted.Length % 2 == 1
                ? sorted[sorted.Length / 2]
                : 0.5 * (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]);
        }

        public string Arm { get; }

        public IReadOnlyList<double> Samples { get; }

        public double MinMs { get; }

        public double MedianMs { get; }

        public double MaxMs { get; }

        /// <summary>Spread as a fraction of the median — a quiet sitting keeps this small.</summary>
        public double SpreadFraction => MedianMs > 0 ? (MaxMs - MinMs) / MedianMs : 0;

        public double GFlops(long flops) => MedianMs > 0 ? flops / (MedianMs * 1e6) : 0;
    }
}
