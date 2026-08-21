// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// Decides whether the report may print a headline ratio, and when it may not, why not and what the
    /// operator should do about it.
    /// <para>
    /// This exists because of a measured failure on 2026-08-21. The probe printed
    /// <c>HEADLINE forward ... 1036.58x</c> while its own <c>CANARY MOVED - SITTING SUSPECT</c> line sat
    /// higher up the same page. A number printed next to a warning gets quoted without the warning; that
    /// is not a hypothesis, it happened three times in one session. The probe already refuses to print a
    /// timing for a cell whose parity check failed, and that principle is right — this class applies the
    /// identical principle to the headline. A suspect sitting SUPPRESSES the ratio rather than
    /// annotating it.
    /// </para>
    /// <para>
    /// The per-arm timings are still printed. They are facts about the arms and a reader can weigh them.
    /// The ratio is the thing that travels out of the report as an answer, so the ratio is the thing
    /// that has to be withheld.
    /// </para>
    /// </summary>
    internal sealed class HeadlineVerdict
    {
        private HeadlineVerdict(IReadOnlyList<string> blockers)
        {
            Blockers = blockers;
        }

        /// <summary>Each entry is one reason the ratio is absent, followed by what to do about it.</summary>
        public IReadOnlyList<string> Blockers { get; }

        public bool MayPrint => Blockers.Count == 0;

        /// <summary>
        /// The reasons that apply to the whole sitting rather than to one cell: the machine moved under
        /// the measurement, or the arms did not run on the hardware the probe exists to price.
        /// </summary>
        public static HeadlineVerdict ForRun(Report report)
        {
            var blockers = new List<string>();

            if (report.CanaryMeasured && Math.Abs(report.CanaryMove) > Canary.MoveThreshold)
            {
                blockers.Add(string.Create(
                    CultureInfo.InvariantCulture,
                    $"the canary moved {report.CanaryMove * 100:+0.0;-0.0} % between the start and the end of this run " +
                    $"({report.CanaryStartMs:F2} ms then {report.CanaryEndMs:F2} ms), which is outside the " +
                    $"{Canary.MoveThreshold * 100:F0} % threshold. The machine changed underneath the measurement, so " +
                    $"every timing in this report was taken against a moving baseline. " +
                    $"WHAT TO DO: close the browser and anything else heavy, let the machine idle for a minute, run again."));
            }

            if (report.CanaryMeasured && !report.CanarySettled)
            {
                blockers.Add(
                    $"the canary itself did not settle ({report.CanaryStart?.Warmup.Describe()} at the start, " +
                    $"{report.CanaryEnd?.Warmup.Describe()} at the end), so it cannot say whether the machine held " +
                    $"still. An unsettled canary is not evidence that nothing moved. " +
                    $"WHAT TO DO: run on a quieter machine, or raise --warmup-max.");
            }

            if (!report.IsCuda)
            {
                blockers.Add(
                    $"the device arms ran on {report.AcceleratorType}, not CUDA. This probe exists to price a port to " +
                    "an NVIDIA GPU, and a ratio measured on anything else does not answer that question - an OpenCL " +
                    "backend is a much weaker path than PTX, and ILGPU's CPU emulator is not a GPU at all. " +
                    "WHAT TO DO: run this on a machine with an NVIDIA card and a current NVIDIA driver.");
            }

            return new HeadlineVerdict(blockers);
        }

        /// <summary>
        /// The run-level reasons plus everything specific to this cell and these two arms: a failed
        /// parity check, a missing arm, or an arm whose timings had not stopped moving when the clock
        /// started.
        /// </summary>
        public static HeadlineVerdict ForRatio(
            HeadlineVerdict run,
            CellResult cell,
            string baseline,
            string candidate)
        {
            var blockers = new List<string>(run.Blockers);

            foreach (var arm in new[] { baseline, candidate })
            {
                if (!cell.Timings.ContainsKey(arm))
                {
                    blockers.Add($"arm '{arm}' was not measured in this cell, so there is nothing to form a ratio from.");
                    continue;
                }

                if (!cell.MayPrint(arm))
                {
                    blockers.Add(
                        $"arm '{arm}' FAILED its parity check ({cell.Parity[arm].Detail}). A fast wrong kernel is the " +
                        "failure this probe is most exposed to. WHAT TO DO: the kernel is wrong; nothing about the " +
                        "hardware can be read from this cell.");
                    continue;
                }

                if (cell.Warmups.TryGetValue(arm, out var warmup) && !warmup.Settled)
                {
                    blockers.Add(string.Create(
                        CultureInfo.InvariantCulture,
                        $"arm '{arm}' {warmup.Describe()}, so its timings were still drifting when the clock started. " +
                        $"WHAT TO DO: raise --warmup-max or --warmup-budget-ms, or run on a quieter machine."));
                }
            }

            return new HeadlineVerdict(blockers);
        }
    }
}
