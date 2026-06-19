// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime;

namespace DevOnBike.Overfit.Runtime
{
    /// <summary>
    /// Scopes <see cref="GCSettings.LatencyMode"/> to <see cref="GCLatencyMode.SustainedLowLatency"/> for the
    /// duration of a generation, restoring the previous mode on dispose — it suppresses blocking gen-2 collections
    /// (trading some memory headroom for fewer GC pauses), which trims tail latency on the allocating paths
    /// (prefill, the server's per-request marshalling) even though steady-state decode is already zero-allocation.
    /// A <c>ref struct</c> (like <see cref="GcHandleScope"/> / <see cref="PooledArray"/>), so it can't escape its
    /// scope or be boxed — the latency mode is restored exactly when the <c>using</c> ends.
    /// <para>
    /// <b>Process-wide by design.</b> <see cref="GCSettings.LatencyMode"/> is a process-global knob, so the library
    /// never flips it implicitly — that would hijack the GC of a host that merely embeds Overfit. Use this scope
    /// only from a component that OWNS the process (the <c>overfit serve</c> server, a CLI, a benchmark). It is a
    /// no-op when the current mode is already as aggressive or more so.
    /// </para>
    /// </summary>
    public readonly ref struct GcLatencyScope
    {
        private readonly GCLatencyMode _previous;
        private readonly bool _changed;

        private GcLatencyScope(GCLatencyMode previous, bool changed)
        {
            _previous = previous;
            _changed = changed;
        }

        /// <summary>
        /// Enters a <see cref="GCLatencyMode.SustainedLowLatency"/> scope. Skips the change (and the restore) when
        /// the current mode is already SustainedLowLatency or the no-GC region, so nested scopes compose safely.
        /// </summary>
        public static GcLatencyScope SustainedLowLatency()
        {
            var previous = GCSettings.LatencyMode;
            if (previous is GCLatencyMode.SustainedLowLatency or GCLatencyMode.NoGCRegion)
            {
                return new GcLatencyScope(previous, changed: false);
            }

            GCSettings.LatencyMode = GCLatencyMode.SustainedLowLatency;
            return new GcLatencyScope(previous, changed: true);
        }

        public void Dispose()
        {
            if (_changed)
            {
                GCSettings.LatencyMode = _previous;
            }
        }
    }
}
