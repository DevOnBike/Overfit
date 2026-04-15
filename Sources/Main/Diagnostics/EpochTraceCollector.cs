// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Diagnostics.Contracts;

namespace DevOnBike.Overfit.Diagnostics
{
    /// <summary>
    /// Lightweight in-memory collector useful for tests and benchmarks.
    /// It aggregates duration and allocation by kernel/module/category for one run.
    /// </summary>
    public sealed class EpochTraceCollector : IOverfitDiagnosticsSink
    {
        private readonly Dictionary<string, Aggregate> _kernels = new(StringComparer.Ordinal);
        private readonly Dictionary<string, Aggregate> _modules = new(StringComparer.Ordinal);
        private readonly Dictionary<string, long> _counters = new(StringComparer.Ordinal);

        public IReadOnlyDictionary<string, Aggregate> Kernels => _kernels;
        public IReadOnlyDictionary<string, Aggregate> Modules => _modules;
        public IReadOnlyDictionary<string, long> Counters => _counters;

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

        public void OnKernelCompleted(in KernelDiagnosticEvent evt)
        {
            var key = $"{evt.Category}/{evt.Name}/{evt.Phase}";
            
            Add(_kernels, key, evt.DurationMs, 0);
        }

        public void OnModuleCompleted(in ModuleDiagnosticEvent evt)
        {
            var key = $"{evt.ModuleType}/{evt.Phase}";
            
            Add(_modules, key, evt.DurationMs, evt.AllocatedBytes);
        }

        public void OnGraphCompleted(in GraphDiagnosticEvent evt)
        {
            GraphCount++;
            TapeOps += evt.TapeOpCount;
            GraphBackwardMs += evt.BackwardMs;
            GraphAllocatedBytes += evt.AllocatedBytes;
            GraphGc0 += evt.Gen0Collections;
            GraphGc1 += evt.Gen1Collections;
            GraphGc2 += evt.Gen2Collections;
        }

        public void OnAllocation(in AllocationDiagnosticEvent evt)
        {
            AllocationBytes += evt.Bytes;
        }

        public void OnCounter(string name, long value)
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

        public EpochTraceSnapshot Snapshot()
        {
            return new EpochTraceSnapshot(
            new Dictionary<string, Aggregate>(_kernels, StringComparer.Ordinal),
            new Dictionary<string, Aggregate>(_modules, StringComparer.Ordinal),
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

        private static void Add(Dictionary<string, Aggregate> map, string key, double durationMs, long allocatedBytes)
        {
            if (map.TryGetValue(key, out var agg))
            {
                map[key] = agg with
                {
                    Count = agg.Count + 1,
                    DurationMs = agg.DurationMs + durationMs,
                    AllocatedBytes = agg.AllocatedBytes + allocatedBytes
                };
            }
            else
            {
                map[key] = new Aggregate(1, durationMs, allocatedBytes);
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
