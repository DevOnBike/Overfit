// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Linq;
using DevOnBike.Overfit.Diagnostics.Contracts;

namespace DevOnBike.Overfit.Diagnostics
{
    /// <summary>
    /// Lekki kolektor w pamięci, zoptymalizowany pod kątem braku alokacji (Zero-Alloc) w gorących ścieżkach.
    /// Wykorzystuje krotki wartości (ValueTuple) jako klucze słowników, aby uniknąć interpolacji stringów.
    /// </summary>
    public sealed class EpochTraceCollector : IOverfitDiagnosticsSink
    {
        // Zamiast stringów używamy krotek, które są typami wartościowymi (nie alokują na stercie)
        private readonly Dictionary<(string Category, string Name, string Phase), Aggregate> _kernels = new();
        private readonly Dictionary<(string ModuleType, string Phase), Aggregate> _modules = new();
        private readonly Dictionary<string, long> _counters = new(StringComparer.Ordinal);
        private readonly Lock _lock = new();

        public long GraphCount { get; private set; }
        public long TapeOps { get; private set; }
        public double GraphBackwardMs { get; private set; }
        public long GraphAllocatedBytes { get; private set; }
        public int GraphGc0 { get; private set; }
        public int GraphGc1 { get; private set; }
        public int GraphGc2 { get; private set; }
        public long AllocationBytes { get; private set; }

        public void Reset()
        {
            lock (_lock)
            {
                _kernels.Clear();
                _modules.Clear();
                _counters.Clear();

                GraphCount = 0;
                TapeOps = 0;
                GraphBackwardMs = 0;
                GraphAllocatedBytes = 0;
                GraphGc0 = 0;
                GraphGc1 = 0;
                GraphGc2 = 0;
                AllocationBytes = 0;
            }
        }

        public void OnKernelCompleted(in KernelDiagnosticEvent evt)
        {
            // Tworzenie krotki na stosie - zero alokacji
            var key = (evt.Category, evt.Name, evt.Phase);

            lock (_lock)
            {
                if (_kernels.TryGetValue(key, out var agg))
                {
                    _kernels[key] = agg with
                    {
                        Count = agg.Count + 1,
                        DurationMs = agg.DurationMs + evt.DurationMs
                    };
                }
                else
                {
                    _kernels[key] = new Aggregate(1, evt.DurationMs, 0);
                }
            }
        }

        public void OnModuleCompleted(in ModuleDiagnosticEvent evt)
        {
            // Eliminacja interpolacji stringa $"{evt.ModuleType}/{evt.Phase}"
            var key = (evt.ModuleType, evt.Phase);

            lock (_lock)
            {
                if (_modules.TryGetValue(key, out var agg))
                {
                    _modules[key] = agg with
                    {
                        Count = agg.Count + 1,
                        DurationMs = agg.DurationMs + evt.DurationMs,
                        AllocatedBytes = agg.AllocatedBytes + evt.AllocatedBytes
                    };
                }
                else
                {
                    _modules[key] = new Aggregate(1, evt.DurationMs, evt.AllocatedBytes);
                }
            }
        }

        public void OnGraphCompleted(in GraphDiagnosticEvent evt)
        {
            lock (_lock)
            {
                GraphCount++;
                TapeOps += evt.TapeOpCount;
                GraphBackwardMs += evt.BackwardMs;
                GraphAllocatedBytes += evt.AllocatedBytes;
                GraphGc0 += evt.Gen0Collections;
                GraphGc1 += evt.Gen1Collections;
                GraphGc2 += evt.Gen2Collections;
            }
        }

        public void OnAllocation(in AllocationDiagnosticEvent evt)
        {
            lock (_lock)
            {
                AllocationBytes += evt.Bytes;
            }
        }

        public void OnCounter(string name, long value)
        {
            lock (_lock)
            {
                if (_counters.TryGetValue(name, out var existing))
                {
                    _counters[name] = existing + value;
                }
                else
                {
                    _counters[name] = value;
                }
            }
        }

        /// <summary>
        /// Snapshot jest wykonywany rzadko (np. raz na epokę), więc tutaj alokacja stringów 
        /// na potrzeby czytelnego raportu jest akceptowalna.
        /// </summary>
        public EpochTraceSnapshot Snapshot()
        {
            lock (_lock)
            {
                return new EpochTraceSnapshot(
                    _kernels.ToDictionary(
                        kvp => $"{kvp.Key.Category}/{kvp.Key.Name}/{kvp.Key.Phase}",
                        kvp => kvp.Value),
                    _modules.ToDictionary(
                        kvp => $"{kvp.Key.ModuleType}/{kvp.Key.Phase}",
                        kvp => kvp.Value),
                    new Dictionary<string, long>(_counters, StringComparer.Ordinal),
                    GraphCount,
                    TapeOps,
                    GraphBackwardMs,
                    GraphAllocatedBytes,
                    GraphGc0,
                    GraphGc1,
                    GraphGc2,
                    AllocationBytes);
            }
        }

        public readonly record struct Aggregate(long Count, double DurationMs, long AllocatedBytes);

        public readonly record struct EpochTraceSnapshot(
            IReadOnlyDictionary<string, Aggregate> Kernels,
            IReadOnlyDictionary<string, Aggregate> Modules,
            IReadOnlyDictionary<string, long> Counters,
            long GraphCount,
            long TapeOps,
            double GraphBackwardMs,
            long GraphAllocatedBytes,
            int GraphGc0,
            int GraphGc1,
            int GraphGc2,
            long AllocationBytes);
    }
}