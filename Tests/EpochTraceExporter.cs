using System.Text;
using System.Text.Json;
using DevOnBike.Overfit.Diagnostics;

namespace DevOnBike.Overfit.Tests
{
    internal static class EpochTraceExporter
    {
        public static void WriteJson(string path, int epochIndex, EpochTraceCollector.EpochTraceSnapshot snapshot)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);

            var payload = new
            {
                epoch = epochIndex,
                graphCount = snapshot.GraphCount,
                tapeOps = snapshot.TapeOps,
                graphBackwardMs = snapshot.GraphBackwardMs,
                graphAllocatedBytes = snapshot.GraphAllocatedBytes,
                graphGc0 = snapshot.GraphGc0,
                graphGc1 = snapshot.GraphGc1,
                graphGc2 = snapshot.GraphGc2,
                allocationBytes = snapshot.AllocationBytes,
                modules = snapshot.Modules.ToDictionary(
                static kv => kv.Key,
                static kv => new
                {
                    count = kv.Value.Count,
                    durationMs = kv.Value.DurationMs,
                    allocatedBytes = kv.Value.AllocatedBytes
                }),
                kernels = snapshot.Kernels.ToDictionary(
                static kv => kv.Key,
                static kv => new
                {
                    count = kv.Value.Count,
                    durationMs = kv.Value.DurationMs,
                    allocatedBytes = kv.Value.AllocatedBytes
                }),
                counters = snapshot.Counters
            };

            var json = JsonSerializer.Serialize(payload, new JsonSerializerOptions
            {
                WriteIndented = true
            });

            File.WriteAllText(path, json, Encoding.UTF8);
        }

        public static void WriteCsv(string path, EpochTraceCollector.EpochTraceSnapshot snapshot)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);

            var sb = new StringBuilder();
            sb.AppendLine("kind,name,count,duration_ms,allocated_bytes,value");

            foreach (var kv in snapshot.Modules.OrderBy(static x => x.Key, StringComparer.Ordinal))
            {
                sb.AppendLine($"module,{Escape(kv.Key)},{kv.Value.Count},{kv.Value.DurationMs:F6},{kv.Value.AllocatedBytes},");
            }

            foreach (var kv in snapshot.Kernels.OrderBy(static x => x.Key, StringComparer.Ordinal))
            {
                sb.AppendLine($"kernel,{Escape(kv.Key)},{kv.Value.Count},{kv.Value.DurationMs:F6},{kv.Value.AllocatedBytes},");
            }

            foreach (var kv in snapshot.Counters.OrderBy(static x => x.Key, StringComparer.Ordinal))
            {
                sb.AppendLine($"counter,{Escape(kv.Key)},0,0,0,{kv.Value}");
            }

            sb.AppendLine($"graph,graph.count,0,{snapshot.GraphBackwardMs:F6},{snapshot.GraphAllocatedBytes},{snapshot.GraphCount}");
            sb.AppendLine($"graph,graph.tape_ops,0,0,0,{snapshot.TapeOps}");
            sb.AppendLine($"graph,graph.gc0,0,0,0,{snapshot.GraphGc0}");
            sb.AppendLine($"graph,graph.gc1,0,0,0,{snapshot.GraphGc1}");
            sb.AppendLine($"graph,graph.gc2,0,0,0,{snapshot.GraphGc2}");
            sb.AppendLine($"allocation,allocation.total,0,0,{snapshot.AllocationBytes},0");

            File.WriteAllText(path, sb.ToString(), Encoding.UTF8);

            static string Escape(string s) => "\"" + s.Replace("\"", "\"\"") + "\"";
        }
    }
}