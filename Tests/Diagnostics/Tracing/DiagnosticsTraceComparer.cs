// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;

namespace DevOnBike.Overfit.Tests
{
    internal static class DiagnosticsTraceComparer
    {
        public static DiagnosticsTraceDiff Compare(DiagnosticsTraceModel baseline, DiagnosticsTraceModel current)
        {
            var diff = new DiagnosticsTraceDiff
            {
                BaselineEpoch = baseline.Epoch,
                CurrentEpoch = current.Epoch,
                GraphBackwardMsDelta = current.GraphBackwardMs - baseline.GraphBackwardMs,
                GraphAllocatedBytesDelta = current.GraphAllocatedBytes - baseline.GraphAllocatedBytes,
                TapeOpsDelta = current.TapeOps - baseline.TapeOps
            };

            CompareEntries(baseline.Modules!, current.Modules!, diff.ModuleDiffs);
            CompareEntries(baseline.Kernels!, current.Kernels!, diff.KernelDiffs);

            return diff;
        }

        private static void CompareEntries(
            Dictionary<string, DiagnosticsTraceEntry> baseline,
            Dictionary<string, DiagnosticsTraceEntry> current,
            List<DiagnosticsTraceEntryDiff> target)
        {
            foreach (var kv in current)
            {
                if (baseline.TryGetValue(kv.Key, out var b))
                {
                    target.Add(new DiagnosticsTraceEntryDiff
                    {
                        Name = kv.Key,
                        CountDelta = kv.Value.Count - b.Count,
                        DurationMsDelta = kv.Value.DurationMs - b.DurationMs,
                        AllocatedBytesDelta = kv.Value.AllocatedBytes - b.AllocatedBytes,
                        CurrentDurationMs = kv.Value.DurationMs,
                        CurrentAllocatedBytes = kv.Value.AllocatedBytes
                    });
                }
                else
                {
                    target.Add(new DiagnosticsTraceEntryDiff
                    {
                        Name = kv.Key,
                        CountDelta = kv.Value.Count,
                        DurationMsDelta = kv.Value.DurationMs,
                        AllocatedBytesDelta = kv.Value.AllocatedBytes,
                        CurrentDurationMs = kv.Value.DurationMs,
                        CurrentAllocatedBytes = kv.Value.AllocatedBytes
                    });
                }
            }
        }

        public static string Format(DiagnosticsTraceDiff diff, int top = 10)
        {
            var sb = new StringBuilder();

            sb.AppendLine("=== DIAGNOSTICS DIFF ===");
            sb.AppendLine($"baseline.epoch:         {diff.BaselineEpoch}");
            sb.AppendLine($"current.epoch:          {diff.CurrentEpoch}");
            sb.AppendLine($"graph.backward.delta:   {diff.GraphBackwardMsDelta:F1} ms");
            sb.AppendLine($"graph.alloc.delta:      {diff.GraphAllocatedBytesDelta / 1024.0 / 1024.0:F2} MB");
            sb.AppendLine($"tape_ops.delta:         {diff.TapeOpsDelta}");

            AppendTop("top.module.deltas", diff.ModuleDiffs, sb, top);
            AppendTop("top.kernel.deltas", diff.KernelDiffs, sb, top);

            return sb.ToString();
        }

        private static void AppendTop(string title, List<DiagnosticsTraceEntryDiff> items, StringBuilder sb, int top)
        {
            sb.AppendLine(title + ":");

            var sorted = new List<DiagnosticsTraceEntryDiff>(items);
            sorted.Sort(static (a, b) => Math.Abs(b.DurationMsDelta).CompareTo(Math.Abs(a.DurationMsDelta)));

            var count = Math.Min(top, sorted.Count);
            for (var i = 0; i < count; i++)
            {
                var x = sorted[i];
                sb.AppendLine(
                $"  {x.Name,-32} | d.ms {x.DurationMsDelta,10:F1} | d.alloc {(x.AllocatedBytesDelta / 1024.0 / 1024.0),10:F2} MB | cur.ms {x.CurrentDurationMs,10:F1}");
            }
        }
    }

    internal sealed class DiagnosticsTraceDiff
    {
        public int BaselineEpoch { get; set; }
        public int CurrentEpoch { get; set; }
        public double GraphBackwardMsDelta { get; set; }
        public long GraphAllocatedBytesDelta { get; set; }
        public long TapeOpsDelta { get; set; }
        public List<DiagnosticsTraceEntryDiff> ModuleDiffs { get; } = new();
        public List<DiagnosticsTraceEntryDiff> KernelDiffs { get; } = new();
    }

    internal sealed class DiagnosticsTraceEntryDiff
    {
        public string Name { get; set; } = string.Empty;
        public long CountDelta { get; set; }
        public double DurationMsDelta { get; set; }
        public long AllocatedBytesDelta { get; set; }
        public double CurrentDurationMs { get; set; }
        public long CurrentAllocatedBytes { get; set; }
    }
}
