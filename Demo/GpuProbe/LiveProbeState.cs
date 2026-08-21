// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// What the live view knows about the run. Every throughput here comes from a
    /// <see cref="Measurement"/> the report itself prints - the view starts NO timer of its own.
    /// <para>
    /// Two timers disagree, and then nobody can say which figure the report means. So the view is a
    /// second rendering of the probe's numbers, never a second measurement of them.
    /// </para>
    /// </summary>
    internal sealed class LiveProbeState
    {
        public string Cell { get; set; } = "-";

        public int N { get; set; }

        /// <summary>"warming up", "timing", "checking parity" - what the probe is doing right now.</summary>
        public string Phase { get; set; } = "starting";

        public int WarmupRound { get; set; }

        public int CellsDone { get; set; }

        public int CellsTotal { get; set; }

        /// <summary>Arm name to its last COMPLETED median in ms, straight from the report's own numbers.</summary>
        public Dictionary<string, double> LastMedianMs { get; } = [];

        /// <summary>Arm name to its last COMPLETED throughput in GFLOP/s, from the same numbers.</summary>
        public Dictionary<string, double> LastGflops { get; } = [];

        /// <summary>The cell those completed figures belong to, so the view cannot mislabel them.</summary>
        public string CompletedLabel { get; set; } = "nothing has completed yet";

        public void RecordCompleted(CellResult cell)
        {
            LastMedianMs.Clear();
            LastGflops.Clear();

            foreach (var (arm, measurement) in cell.Timings)
            {
                if (!cell.MayPrint(arm))
                {
                    continue;
                }

                LastMedianMs[arm] = measurement.MedianMs;
                LastGflops[arm] = measurement.GFlops(cell.ForwardFlops);
            }

            CompletedLabel = $"{cell.Cell.Name} n={cell.N}";
            CellsDone++;
        }
    }
}
