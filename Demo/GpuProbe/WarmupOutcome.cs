// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// What the warm-up phase established about one arm. This is reported, not kept internal: "the
    /// timings had stopped moving after 14 rounds" is the evidence that the number below it is a
    /// measurement of the arm rather than of a compiler, a cold cache or a clocked-down device.
    /// </summary>
    internal sealed class WarmupOutcome
    {
        public WarmupOutcome(
            string arm,
            int rounds,
            bool settled,
            int settledAtRound,
            double previousWindowMedianMs,
            double lastWindowMedianMs,
            double noiseBandMs)
        {
            Arm = arm;
            Rounds = rounds;
            Settled = settled;
            SettledAtRound = settledAtRound;
            PreviousWindowMedianMs = previousWindowMedianMs;
            LastWindowMedianMs = lastWindowMedianMs;
            NoiseBandMs = noiseBandMs;
        }

        public string Arm { get; }

        /// <summary>Warm-up rounds this arm actually ran.</summary>
        public int Rounds { get; }

        /// <summary>True when the stopping rule held at the round the loop stopped on.</summary>
        public bool Settled { get; }

        /// <summary>First round at which the rule held, or 0 if it never did.</summary>
        public int SettledAtRound { get; }

        public double PreviousWindowMedianMs { get; }

        public double LastWindowMedianMs { get; }

        /// <summary>
        /// How far two window medians of this arm are expected to sit apart from scatter alone. A move
        /// inside this band is not evidence of drift, because the arm cannot resolve drift that small.
        /// </summary>
        public double NoiseBandMs { get; }

        /// <summary>Relative move between the two window medians at the stopping round.</summary>
        public double RelativeMove => PreviousWindowMedianMs > 0
            ? Math.Abs(LastWindowMedianMs - PreviousWindowMedianMs) / PreviousWindowMedianMs
            : 0;

        public string Describe() => Settled
            ? string.Create(
                CultureInfo.InvariantCulture,
                $"settled after {SettledAtRound} rounds ({Rounds} ran; last two window medians {PreviousWindowMedianMs:F3} and {LastWindowMedianMs:F3} ms, {RelativeMove * 100:F1} % apart, noise band {NoiseBandMs:F3} ms)")
            : string.Create(
                CultureInfo.InvariantCulture,
                $"DID NOT SETTLE in {Rounds} rounds (last two window medians {PreviousWindowMedianMs:F3} and {LastWindowMedianMs:F3} ms, {RelativeMove * 100:F1} % apart, noise band {NoiseBandMs:F3} ms)");
    }
}
