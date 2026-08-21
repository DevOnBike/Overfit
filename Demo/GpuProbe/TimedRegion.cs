// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// Whether a clock is running right now, so that anything expensive can refuse to run inside one.
    /// <para>
    /// This exists because the first rule of the live view - no telemetry call and no repaint inside a
    /// timed region, ever - was otherwise enforced by remembering it at each call site. An NVML read is a
    /// driver round trip and a Spectre frame is tens of kilobytes of allocation; either one inside a
    /// clock moves the number and looks entirely normal in the report. The flag makes the rule checkable:
    /// <see cref="LiveView.Refresh"/> consults it, drops a repaint that arrives inside a clock, and
    /// records that it happened so the report can say the timings are suspect.
    /// </para>
    /// <para>
    /// <b>Both writes sit OUTSIDE the timestamps they bracket</b>, so the flag costs the measured region
    /// nothing. The probe is single-threaded by design - see <see cref="LiveView"/> on why the repaint is
    /// pulled rather than pushed - so a plain static is the whole mechanism and no synchronisation is
    /// implied by it. Timed regions do not nest here either, which is why a bool is enough; a nested one
    /// would need a counter, and <see cref="Enter"/> would be the wrong shape for it.
    /// </para>
    /// </summary>
    internal static class TimedRegion
    {
        /// <summary>True between <see cref="Enter"/> and <see cref="Leave"/>.</summary>
        public static bool IsInside { get; private set; }

        public static void Enter() => IsInside = true;

        public static void Leave() => IsInside = false;
    }
}
