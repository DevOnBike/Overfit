// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.Tests.TestSupport
{
    /// <summary>
    /// Turns <see cref="OverfitParallel.MeasureOccupancy"/> on for the duration of the scope and puts the
    /// previous value back. Mirrors <see cref="NonRepackedKernelScope"/> and <c>TiledPrefillQ4KScope</c>.
    ///
    /// <para><b>It restores the previous value, not <c>false</c>.</b> An unconditional reset is not a scope:
    /// two nested ones leave the flag wrong after the inner one exits. That defect is why the sibling types
    /// exist in this shape.</para>
    ///
    /// <para><b>This flag must NOT become <see cref="System.ThreadStaticAttribute"/>, and the reason is the
    /// opposite of the one that made a per-thread override right for <c>UseTiledPrefillQ4K</c>.</b> That
    /// flag's single production read was proved to happen on the calling thread before any fan-out —
    /// stamped 2026-08-21 at 234 reads on 1 thread — so binding it per thread costs nothing. This one is
    /// read inside <c>OverfitParallel.ExecuteChunk</c>, which runs on the WORKER threads. A per-thread
    /// override set by a test would be invisible to every worker, the chunk timings would never be
    /// recorded, and the measurement would report nothing while looking like it had run. Where the read
    /// happens decides the mechanism; the two flags look alike and are not.</para>
    ///
    /// <para><b>So the flag stays process-wide, and the isolation has to come from the runner instead.</b>
    /// A class using this scope belongs in
    /// <see cref="ExclusiveProcessMeasurementCollection"/>, which xunit runs without parallelising against
    /// other collections. Without that, a concurrent test in another class observes the flag mid-assertion —
    /// the cross-class collision that <c>XC-56</c> and <c>XC-57</c> were both about.</para>
    ///
    /// <para><b>Construct it, never <c>default</c> it.</b> <c>default(OccupancyMeasurementScope)</c> skips
    /// the constructor, so its <c>Dispose</c> writes <c>false</c> rather than restoring anything.</para>
    /// </summary>
    internal readonly struct OccupancyMeasurementScope : IDisposable
    {
        private readonly bool _previous;

        /// <summary>Saves the current value and turns measurement on.</summary>
        public OccupancyMeasurementScope()
        {
            _previous = OverfitParallel.MeasureOccupancy;
            OverfitParallel.MeasureOccupancy = true;
        }

        public void Dispose() => OverfitParallel.MeasureOccupancy = _previous;
    }
}
