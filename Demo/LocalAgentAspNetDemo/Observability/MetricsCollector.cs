// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Concurrent;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.Demo.LocalAgent.Observability
{
    /// <summary>
    /// In-process metrics for the local agent, exposed at <c>/metrics</c> in Prometheus text exposition
    /// format — scrapeable by the Prometheus container in <c>compose.yaml</c>, no exporter package needed.
    ///
    /// Records per-generation stats sourced from <see cref="GenerationStats"/> (the same numbers the
    /// engine reports): prompt/generated tokens, allocated bytes per generation (Overfit's headline
    /// "≈ 0 B/token"), decode throughput, plus tool-call counts and RAG search latency. Static build
    /// info (model fingerprint, mmap flag, load time) is captured once at startup.
    ///
    /// Counters use <see cref="Interlocked"/>; concurrent recording is safe.
    /// </summary>
    public sealed class MetricsCollector
    {
        private readonly ConcurrentDictionary<string, long> _requestsByEndpoint = new(StringComparer.Ordinal);
        private readonly ConcurrentDictionary<string, long> _toolCalls = new(StringComparer.Ordinal);

        private long _promptTokensTotal;
        private long _generatedTokensTotal;
        private long _allocatedBytesTotal;
        private long _generationsTotal;
        private double _lastTokensPerSecond;

        private long _ragSearches;
        private double _ragSearchSecondsTotal;
        private readonly object _ragLock = new();

        // ── Static build info (set once at startup) ──
        public string ModelFile { get; init; } = "unknown";
        public string ModelFingerprint { get; init; } = "unknown";
        public bool MmapEnabled { get; init; }
        public double ModelLoadSeconds { get; init; }

        public void RecordGeneration(string endpoint, in GenerationStats stats)
        {
            _requestsByEndpoint.AddOrUpdate(endpoint, 1, static (_, v) => v + 1);
            Interlocked.Add(ref _promptTokensTotal, stats.PromptTokens);
            Interlocked.Add(ref _generatedTokensTotal, stats.GeneratedTokens);
            Interlocked.Add(ref _allocatedBytesTotal, stats.AllocatedBytes);
            Interlocked.Increment(ref _generationsTotal);
            Interlocked.Exchange(ref _lastTokensPerSecond, stats.TokensPerSecond);
        }

        public void RecordToolCall(string tool)
        {
            _toolCalls.AddOrUpdate(tool, 1, static (_, v) => v + 1);
        }

        public void RecordRagSearch(double seconds)
        {
            lock (_ragLock)
            {
                _ragSearches++;
                _ragSearchSecondsTotal += seconds;
            }
        }

        /// <summary>Renders the current metrics as a Prometheus text exposition payload.</summary>
        public string ToPrometheus()
        {
            var sb = new StringBuilder(1024);

            sb.AppendLine("# HELP overfit_build_info Static info about the loaded model (value is always 1).");
            sb.AppendLine("# TYPE overfit_build_info gauge");
            sb.Append("overfit_build_info{model=\"").Append(Escape(ModelFile))
              .Append("\",fingerprint=\"").Append(Escape(ModelFingerprint))
              .Append("\",mmap=\"").Append(MmapEnabled ? "true" : "false")
              .Append("\"} 1").AppendLine();

            Gauge(sb, "overfit_model_load_seconds", "Time to load the model at startup.", ModelLoadSeconds);

            // Requests by endpoint.
            sb.AppendLine("# HELP overfit_requests_total Requests handled, by endpoint.");
            sb.AppendLine("# TYPE overfit_requests_total counter");
            foreach (var kv in _requestsByEndpoint)
            {
                sb.Append("overfit_requests_total{endpoint=\"").Append(Escape(kv.Key)).Append("\"} ")
                  .Append(kv.Value).AppendLine();
            }

            Counter(sb, "overfit_generations_total", "Total model generations.", Volatile.Read(ref _generationsTotal));
            Counter(sb, "overfit_prompt_tokens_total", "Total prompt tokens processed.", Volatile.Read(ref _promptTokensTotal));
            Counter(sb, "overfit_generated_tokens_total", "Total tokens generated.", Volatile.Read(ref _generatedTokensTotal));
            Counter(sb, "overfit_allocated_bytes_total", "Total bytes allocated during generation (Overfit targets ~0 B/token).", Volatile.Read(ref _allocatedBytesTotal));
            Gauge(sb, "overfit_decode_tokens_per_second", "Decode throughput of the most recent generation.", Volatile.Read(ref _lastTokensPerSecond));

            // Tool calls by tool.
            sb.AppendLine("# HELP overfit_tool_calls_total Tool calls dispatched, by tool name.");
            sb.AppendLine("# TYPE overfit_tool_calls_total counter");
            foreach (var kv in _toolCalls)
            {
                sb.Append("overfit_tool_calls_total{tool=\"").Append(Escape(kv.Key)).Append("\"} ")
                  .Append(kv.Value).AppendLine();
            }

            long ragSearches;
            double ragSeconds;
            lock (_ragLock) { ragSearches = _ragSearches; ragSeconds = _ragSearchSecondsTotal; }
            Counter(sb, "overfit_rag_searches_total", "Total RAG retrieval searches.", ragSearches);
            Counter(sb, "overfit_rag_search_seconds_total", "Cumulative RAG retrieval (embed + cosine scan) time.", ragSeconds);

            return sb.ToString();
        }

        private static void Counter(StringBuilder sb, string name, string help, long value)
        {
            sb.Append("# HELP ").Append(name).Append(' ').AppendLine(help);
            sb.Append("# TYPE ").Append(name).AppendLine(" counter");
            sb.Append(name).Append(' ').Append(value).AppendLine();
        }

        private static void Counter(StringBuilder sb, string name, string help, double value)
        {
            sb.Append("# HELP ").Append(name).Append(' ').AppendLine(help);
            sb.Append("# TYPE ").Append(name).AppendLine(" counter");
            sb.Append(name).Append(' ').Append(value.ToString("0.######", CultureInfo.InvariantCulture)).AppendLine();
        }

        private static void Gauge(StringBuilder sb, string name, string help, double value)
        {
            sb.Append("# HELP ").Append(name).Append(' ').AppendLine(help);
            sb.Append("# TYPE ").Append(name).AppendLine(" gauge");
            sb.Append(name).Append(' ').Append(value.ToString("0.######", CultureInfo.InvariantCulture)).AppendLine();
        }

        private static string Escape(string value) =>
            value.Replace("\\", "\\\\").Replace("\"", "\\\"");

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
                if (head > 0) { hash.AppendData(buffer, 0, head); }

                if (length > 2L * window)
                {
                    fs.Seek(-window, SeekOrigin.End);
                    var tail = fs.Read(buffer, 0, window);
                    if (tail > 0) { hash.AppendData(buffer, 0, tail); }
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
