// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics.Metrics;
using System.Security.Cryptography;
using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.Demo.LocalAgent.Observability
{
    /// <summary>
    /// In-process metrics for the local agent, instrumented with the built-in
    /// <see cref="System.Diagnostics.Metrics"/> (<see cref="Meter"/>) API — the idiomatic ASP.NET Core
    /// approach (learn.microsoft.com/aspnet/core/log-mon/metrics). The metrics are exported to a
    /// Prometheus scrape endpoint at <c>/metrics</c> by OpenTelemetry (wired in <c>Program.cs</c> via
    /// <c>AddPrometheusExporter()</c> + <c>MapPrometheusScrapingEndpoint()</c>) — no hand-rolled text.
    ///
    /// Records per-generation stats sourced from <see cref="GenerationStats"/> (the same numbers the
    /// engine reports): prompt/generated tokens, allocated bytes per generation (Overfit's headline
    /// "≈ 0 B/token"), decode throughput, plus tool-call counts and RAG retrieval latency. Static build
    /// info (model fingerprint, mmap flag, load time) is published as observable gauges. All
    /// <see cref="Meter"/> instruments are thread-safe.
    /// </summary>
    public sealed class MetricsCollector : IDisposable
    {
        /// <summary>Meter name — registered with OpenTelemetry via <c>AddMeter(MetricsCollector.MeterName)</c>.</summary>
        public const string MeterName = "Overfit.LocalAgent";

        private readonly Meter _meter;
        private readonly Counter<long> _requests;
        private readonly Counter<long> _generations;
        private readonly Counter<long> _promptTokens;
        private readonly Counter<long> _generatedTokens;
        private readonly Counter<long> _allocatedBytes;
        private readonly Counter<long> _toolCallsByName;
        private readonly Histogram<double> _decodeRate;
        private readonly Histogram<double> _ragSearch;

        // ── Static build info (set once at startup, published as observable gauges) ──
        public string ModelFile { get; init; } = "unknown";
        public string ModelFingerprint { get; init; } = "unknown";
        public bool MmapEnabled
        {
            get; init;
        }
        public double ModelLoadSeconds
        {
            get; init;
        }

        public MetricsCollector()
        {
            _meter = new Meter(MeterName);

            _requests = _meter.CreateCounter<long>(
                "overfit.requests", unit: "{request}", description: "Requests handled, tagged by endpoint.");
            _generations = _meter.CreateCounter<long>(
                "overfit.generations", unit: "{generation}", description: "Model generations.");
            _promptTokens = _meter.CreateCounter<long>(
                "overfit.prompt.tokens", unit: "{token}", description: "Prompt tokens processed.");
            _generatedTokens = _meter.CreateCounter<long>(
                "overfit.generated.tokens", unit: "{token}", description: "Tokens generated.");
            _allocatedBytes = _meter.CreateCounter<long>(
                "overfit.allocated", unit: "By", description: "Bytes allocated during generation (Overfit targets ~0 B/token).");
            _toolCallsByName = _meter.CreateCounter<long>(
                "overfit.tool.calls", unit: "{call}", description: "Tool calls dispatched, tagged by tool name.");
            _decodeRate = _meter.CreateHistogram<double>(
                "overfit.decode.rate", unit: "{token}/s", description: "Decode throughput per generation.");
            _ragSearch = _meter.CreateHistogram<double>(
                "overfit.rag.search", unit: "s", description: "RAG retrieval (embed + cosine scan) latency.");

            // Static build info as always-1 gauge carrying identifying labels (Prometheus build_info idiom).
            _meter.CreateObservableGauge(
                "overfit.build.info",
                () => new Measurement<int>(1,
                    new KeyValuePair<string, object?>("model", ModelFile),
                    new KeyValuePair<string, object?>("fingerprint", ModelFingerprint),
                    new KeyValuePair<string, object?>("mmap", MmapEnabled ? "true" : "false")),
                description: "Static info about the loaded model (value is always 1).");
            _meter.CreateObservableGauge(
                "overfit.model.load", () => ModelLoadSeconds, unit: "s",
                description: "Time to load the model at startup.");
        }

        public void RecordGeneration(string endpoint, in GenerationStats stats)
        {
            var endpointTag = new KeyValuePair<string, object?>("endpoint", endpoint);
            _requests.Add(1, endpointTag);
            _generations.Add(1);
            _promptTokens.Add(stats.PromptTokens);
            _generatedTokens.Add(stats.GeneratedTokens);
            _allocatedBytes.Add(stats.AllocatedBytes);
            if (stats.TokensPerSecond > 0)
            {
                _decodeRate.Record(stats.TokensPerSecond, endpointTag);
            }
        }

        public void RecordToolCall(string tool) =>
            _toolCallsByName.Add(1, new KeyValuePair<string, object?>("tool", tool));

        public void RecordRagSearch(double seconds) => _ragSearch.Record(seconds);

        public void Dispose() => _meter.Dispose();

        /// <summary>
        /// A fast, stable fingerprint of a model file: SHA-256 over its length plus the first and last
        /// 1 MiB. Near-instant even for multi-GB files (no full read), and identifies a specific model
        /// build in practice. This is a partial fingerprint, not a full content hash — labelled as such.
        /// </summary>
        public static string FingerprintModel(string path)
        {
            try
            {
                const int window = 1024 * 1024;
                var length = new FileInfo(path).Length;

                using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
                Span<byte> lengthBytes = stackalloc byte[8];
                BitConverter.TryWriteBytes(lengthBytes, length);
                hash.AppendData(lengthBytes);

                using var fs = File.OpenRead(path);
                var buffer = new byte[window];

                var head = fs.Read(buffer, 0, window);
                if (head > 0)
                {
                    hash.AppendData(buffer, 0, head);
                }

                if (length > 2L * window)
                {
                    fs.Seek(-window, SeekOrigin.End);
                    var tail = fs.Read(buffer, 0, window);
                    if (tail > 0)
                    {
                        hash.AppendData(buffer, 0, tail);
                    }
                }

                return Convert.ToHexString(hash.GetHashAndReset())[..16].ToLowerInvariant();
            }
            catch
            {
                return "unavailable";
            }
        }
    }
}
