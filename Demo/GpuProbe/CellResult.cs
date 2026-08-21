// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>One cell at one token count: its timings, its parity verdicts and its upload cost.</summary>
    internal sealed class CellResult
    {
        public CellResult(Cell cell, int n)
        {
            Cell = cell;
            N = n;
        }

        public Cell Cell { get; }

        public int N { get; }

        /// <summary>Forward FLOPs of one call: two per multiply-accumulate.</summary>
        public long ForwardFlops => 2L * N * Cell.K * Cell.M;

        public double WeightUploadMs { get; set; }

        public Dictionary<string, Measurement> Timings { get; } = [];

        public Dictionary<string, ParityResult> Parity { get; } = [];

        /// <summary>
        /// What the warm-up phase established about each arm, kept beside the timings rather than
        /// discarded: a number whose warm-up evidence has been dropped is a number nobody can weigh.
        /// </summary>
        public Dictionary<string, WarmupOutcome> Warmups { get; } = [];

        /// <summary>Warm-up rounds this cell ran, and which condition ended them.</summary>
        public int WarmupRounds { get; set; }

        public string WarmupStopReason { get; set; } = string.Empty;

        public List<string> Notes { get; } = [];

        /// <summary>
        /// True when this arm may show a number. An arm with no parity entry is a CPU reference arm and
        /// is always printable; an arm with a failed check is not, whatever it measured.
        /// </summary>
        public bool MayPrint(string arm) => !Parity.TryGetValue(arm, out var parity) || parity.Passed;
    }
}
