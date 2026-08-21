// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// Everything one interleaved run of a cell produced: the timings, and the evidence that the timings
    /// were taken after the arms stopped moving. The two travel together on purpose — a timing whose
    /// warm-up outcome has been dropped is a number nobody can weigh.
    /// </summary>
    internal sealed class ArmRunResult
    {
        public ArmRunResult(
            IReadOnlyDictionary<string, Measurement> timings,
            IReadOnlyDictionary<string, WarmupOutcome> warmups,
            int warmupRounds,
            string warmupStopReason)
        {
            Timings = timings;
            Warmups = warmups;
            WarmupRounds = warmupRounds;
            WarmupStopReason = warmupStopReason;
        }

        public IReadOnlyDictionary<string, Measurement> Timings { get; }

        public IReadOnlyDictionary<string, WarmupOutcome> Warmups { get; }

        /// <summary>Rounds the warm-up phase ran before it stopped.</summary>
        public int WarmupRounds { get; }

        /// <summary>Which of the three terminating conditions ended the warm-up phase.</summary>
        public string WarmupStopReason { get; }

        /// <summary>True when every arm satisfied the stopping rule at the round the loop stopped on.</summary>
        public bool AllSettled => Warmups.Values.All(w => w.Settled);
    }
}
