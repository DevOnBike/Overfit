// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.Diagnostics;

namespace DevOnBike.Overfit.Tests
{
    internal static class BenchmarkTraceFormatter
    {
        public static string Format(EpochTraceCollector.EpochTraceSnapshot snapshot, int top = 12)
        {
            var sb = new StringBuilder();

            sb.AppendLine("=== RUNTIME DIAGNOSTICS ===");
            sb.AppendLine($"graph.count:            {snapshot.GraphCount}");
            sb.AppendLine($"graph.tape_ops:         {snapshot.TapeOps}");
            sb.AppendLine($"graph.backward.ms:      {snapshot.GraphBackwardMs:F1}");
            sb.AppendLine($"graph.alloc.mb:         {snapshot.GraphAllocatedBytes / 1024.0 / 1024.0:F2}");
            sb.AppendLine($"graph.gc0/gc1/gc2:      {snapshot.GraphGc0}/{snapshot.GraphGc1}/{snapshot.GraphGc2}");
            sb.AppendLine($"allocation.total.mb:    {snapshot.AllocationBytes / 1024.0 / 1024.0:F2}");

            if (snapshot.Modules.Count > 0)
            {
                sb.AppendLine("top.modules:");
                foreach (var kv in snapshot.Modules
                             .OrderByDescending(x => x.Value.DurationMs)
                             .Take(top))
                {
                    sb.AppendLine(
                    $"  {kv.Key,-30} | count {kv.Value.Count,6} | ms {kv.Value.DurationMs,10:F1} | alloc {(kv.Value.AllocatedBytes / 1024.0 / 1024.0),10:F2} MB");
                }
            }

            if (snapshot.Kernels.Count > 0)
            {
                sb.AppendLine("top.kernels:");
                foreach (var kv in snapshot.Kernels
                             .OrderByDescending(x => x.Value.DurationMs)
                             .Take(top))
                {
                    sb.AppendLine(
                    $"  {kv.Key,-30} | count {kv.Value.Count,6} | ms {kv.Value.DurationMs,10:F1}");
                }
            }

            if (snapshot.Counters.Count > 0)
            {
                sb.AppendLine("top.counters:");
                foreach (var kv in snapshot.Counters
                             .OrderByDescending(x => x.Value)
                             .Take(top))
                {
                    sb.AppendLine($"  {kv.Key,-30} | {kv.Value}");
                }
            }

            return sb.ToString();
        }
    }
}
