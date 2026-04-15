using System.Text.Json;
using System.Text.Json.Serialization;

namespace DevOnBike.Overfit.Tests
{
    internal sealed class DiagnosticsTraceModel
    {
        [JsonPropertyName("epoch")]
        public int Epoch { get; set; }

        [JsonPropertyName("graphCount")]
        public long GraphCount { get; set; }

        [JsonPropertyName("tapeOps")]
        public long TapeOps { get; set; }

        [JsonPropertyName("graphBackwardMs")]
        public double GraphBackwardMs { get; set; }

        [JsonPropertyName("graphAllocatedBytes")]
        public long GraphAllocatedBytes { get; set; }

        [JsonPropertyName("graphGc0")]
        public int GraphGc0 { get; set; }

        [JsonPropertyName("graphGc1")]
        public int GraphGc1 { get; set; }

        [JsonPropertyName("graphGc2")]
        public int GraphGc2 { get; set; }

        [JsonPropertyName("allocationBytes")]
        public long AllocationBytes { get; set; }

        [JsonPropertyName("modules")]
        public Dictionary<string, DiagnosticsTraceEntry>? Modules { get; set; }

        [JsonPropertyName("kernels")]
        public Dictionary<string, DiagnosticsTraceEntry>? Kernels { get; set; }

        [JsonPropertyName("counters")]
        public Dictionary<string, long>? Counters { get; set; }

        public static DiagnosticsTraceModel Load(string path)
        {
            var json = File.ReadAllText(path);
            var model = JsonSerializer.Deserialize<DiagnosticsTraceModel>(json);

            if (model == null)
            {
                throw new InvalidOperationException($"Failed to deserialize diagnostics trace: {path}");
            }

            model.Modules ??= new Dictionary<string, DiagnosticsTraceEntry>(StringComparer.Ordinal);
            model.Kernels ??= new Dictionary<string, DiagnosticsTraceEntry>(StringComparer.Ordinal);
            model.Counters ??= new Dictionary<string, long>(StringComparer.Ordinal);

            return model;
        }
    }

    internal sealed class DiagnosticsTraceEntry
    {
        [JsonPropertyName("count")]
        public long Count { get; set; }

        [JsonPropertyName("durationMs")]
        public double DurationMs { get; set; }

        [JsonPropertyName("allocatedBytes")]
        public long AllocatedBytes { get; set; }
    }
}
