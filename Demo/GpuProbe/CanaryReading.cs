// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// One reading of the canary: the settled median, and the evidence that it is settled.
    /// <para>
    /// The evidence travels with the number because of a measured failure on 2026-08-21. The canary was
    /// timed three times with no warm-up at all, so its FIRST reading carried the JIT of its own inner
    /// loop and a cold cache while its SECOND reading did not. Measured: 99.14 ms at the start against
    /// 5.30 ms at the end of the same quiet run, a move of -94.7 % that says nothing whatever about the
    /// machine. A canary that fires on every run is not a canary, and it would have suppressed every
    /// headline ratio the probe could ever print.
    /// </para>
    /// </summary>
    internal sealed class CanaryReading
    {
        public CanaryReading(double medianMs, WarmupOutcome warmup)
        {
            MedianMs = medianMs;
            Warmup = warmup;
        }

        public double MedianMs { get; }

        public WarmupOutcome Warmup { get; }

        public bool Settled => Warmup.Settled;
    }
}
