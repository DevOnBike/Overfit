// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Concurrent;
using System.Diagnostics.Metrics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.Demo.LocalAgent.Observability
{
    /// <summary>
    /// In-process metrics for the local agent, instrumented with the built-in
    /// <see cref="System.Diagnostics.Metrics"/> (<see cref="Meter"/>) API — the idiomatic ASP.NET Core
    /// approach (learn.microsoft.com/aspnet/core/log-mon/metrics).
    ///
    /// <para><b>The Prometheus text at <c>/metrics</c> is written here (see <see cref="WriteExposition"/>)
    /// rather than by an exporter package, and that changed on 2026-08-11.</b> It used to come from
    /// <c>OpenTelemetry.Exporter.Prometheus.AspNetCore</c>, which has been in prerelease since 2022-08-18 —
    /// 33 versions, not one stable, while every other package in its suite shipped stable 1.17.0 — and was
    /// the only prerelease pin in the repository. The product does not use it either: both
    /// <c>Sources/Server.AspNet/Endpoints/MetricsEndpoints.cs</c> and
    /// <c>Sources/Anomalies/Monitoring/GuardTelemetry.cs</c> write the exposition by hand, so the demo was
    /// showing the opposite of what the product does.</para>
    ///
    /// <para><b>The <see cref="Meter"/> instruments are KEPT, deliberately.</b> They are the idiomatic
    /// surface and anyone who wants OpenTelemetry can attach it to <see cref="MeterName"/> without touching
    /// this file. What was removed is the exporter, not the instrumentation. The shadow counters below exist
    /// because a <see cref="Meter"/> is write-only from the producer's side — reading it back needs a
    /// listener, which is more machinery than a demo should carry to print eight numbers.</para>
    ///
    /// <para>Series names follow what the OpenTelemetry exporter emitted (dots to underscores, <c>_total</c>
    /// on counters, unit suffixes), so an existing scrape config and any dashboard built against it keep
    /// working. <c>Observability/prometheus.yml</c> is unchanged.</para>
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

        // ── Shadow state for the exposition ──
        // Written alongside every Meter instrument above. The two must be updated together; a Record method
        // that touches one and forgets the other reports a number that is quietly stale rather than absent,
        // which is the harder kind to notice.
        private readonly ConcurrentDictionary<string, long> _requestsByEndpoint = new(StringComparer.Ordinal);
        private readonly ConcurrentDictionary<string, long> _toolCallsByTool = new(StringComparer.Ordinal);

        // The same boundaries the OpenTelemetry view used to configure, kept verbatim: the defaults start at
        // 0 and jump to 5 s, which is useless for the 10-50 ms searches this actually sees.
        private readonly BucketHistogram _ragSearchHistogram =
            new([0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1]);

        private readonly BucketHistogram _decodeRateHistogram =
            new([1, 2, 5, 10, 20, 50, 100, 200, 500]);

        private long _generationsTotal;
        private long _promptTokensTotal;
        private long _generatedTokensTotal;
        private long _allocatedBytesTotal;

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

            _requestsByEndpoint.AddOrUpdate(endpoint, 1, static (_, current) => current + 1);
            Interlocked.Increment(ref _generationsTotal);
            Interlocked.Add(ref _promptTokensTotal, stats.PromptTokens);
            Interlocked.Add(ref _generatedTokensTotal, stats.GeneratedTokens);
            Interlocked.Add(ref _allocatedBytesTotal, stats.AllocatedBytes);

            if (stats.TokensPerSecond > 0)
            {
                _decodeRate.Record(stats.TokensPerSecond, endpointTag);
                _decodeRateHistogram.Record(stats.TokensPerSecond);
            }
        }

        public void RecordToolCall(string tool)
        {
            _toolCallsByName.Add(1, new KeyValuePair<string, object?>("tool", tool));
            _toolCallsByTool.AddOrUpdate(tool, 1, static (_, current) => current + 1);
        }

        public void RecordRagSearch(double seconds)
        {
            _ragSearch.Record(seconds);
            _ragSearchHistogram.Record(seconds);
        }

        /// <summary>
        /// Writes the Prometheus text exposition served at <c>GET /metrics</c>.
        ///
        /// <para>One <c># HELP</c> and one <c># TYPE</c> per metric name, immediately before its samples —
        /// a second pair for the same name makes Prometheus reject the whole document, which is why the
        /// grouping is per metric and never per line.</para>
        /// </summary>
        public string WriteExposition()
        {
            var text = new StringBuilder(2048);

            WriteLabelled(text, "overfit_requests_total", "counter",
                "Requests handled, by endpoint.", "endpoint", _requestsByEndpoint);

            WriteScalar(text, "overfit_generations_total", "counter",
                "Model generations.", Interlocked.Read(ref _generationsTotal));
            WriteScalar(text, "overfit_prompt_tokens_total", "counter",
                "Prompt tokens processed.", Interlocked.Read(ref _promptTokensTotal));
            WriteScalar(text, "overfit_generated_tokens_total", "counter",
                "Tokens generated.", Interlocked.Read(ref _generatedTokensTotal));
            WriteScalar(text, "overfit_allocated_bytes_total", "counter",
                "Bytes allocated during generation (Overfit's headline is near zero per token).",
                Interlocked.Read(ref _allocatedBytesTotal));

            WriteLabelled(text, "overfit_tool_calls_total", "counter",
                "Tool calls, by tool name.", "tool", _toolCallsByTool);

            _decodeRateHistogram.Write(text, "overfit_decode_rate", "Decode throughput, tokens per second.");
            _ragSearchHistogram.Write(text, "overfit_rag_search_seconds",
                "RAG retrieval (embed + cosine scan) latency.");

            text.Append("# HELP overfit_build_info Static info about the loaded model (value is always 1).\n");
            text.Append("# TYPE overfit_build_info gauge\n");
            text.Append("overfit_build_info{model=\"").Append(Escape(ModelFile))
                .Append("\",fingerprint=\"").Append(Escape(ModelFingerprint))
                .Append("\",mmap=\"").Append(MmapEnabled ? "true" : "false").Append("\"} 1\n");

            text.Append("# HELP overfit_model_load_seconds Time to load the model at startup.\n");
            text.Append("# TYPE overfit_model_load_seconds gauge\n");
            text.Append("overfit_model_load_seconds ")
                .Append(ModelLoadSeconds.ToString("G17", CultureInfo.InvariantCulture)).Append('\n');

            return text.ToString();
        }

        private static void WriteScalar(StringBuilder text, string name, string type, string help, long value)
        {
            text.Append("# HELP ").Append(name).Append(' ').Append(help).Append('\n');
            text.Append("# TYPE ").Append(name).Append(' ').Append(type).Append('\n');
            text.Append(name).Append(' ').Append(value).Append('\n');
        }

        /// <summary>
        /// A metric with one label. Emitted even when empty — with the HELP and TYPE lines and no samples —
        /// because a series that vanishes when its count is zero is indistinguishable from one that was
        /// never registered, and an alert written against it silently never fires.
        /// </summary>
        private static void WriteLabelled(
            StringBuilder text, string name, string type, string help, string label,
            ConcurrentDictionary<string, long> values)
        {
            text.Append("# HELP ").Append(name).Append(' ').Append(help).Append('\n');
            text.Append("# TYPE ").Append(name).Append(' ').Append(type).Append('\n');

            foreach (var pair in values)
            {
                text.Append(name).Append('{').Append(label).Append("=\"").Append(Escape(pair.Key))
                    .Append("\"} ").Append(pair.Value).Append('\n');
            }
        }

        /// <summary>
        /// Escapes a label VALUE per the exposition format: backslash, double quote and newline. A model
        /// path on Windows is full of backslashes, so this is load-bearing rather than defensive — an
        /// unescaped one makes the whole document unparseable.
        /// </summary>
        private static string Escape(string value)
        {
            return value
                .Replace("\\", "\\\\", StringComparison.Ordinal)
                .Replace("\"", "\\\"", StringComparison.Ordinal)
                .Replace("\n", "\\n", StringComparison.Ordinal);
        }

        public void Dispose() => _meter.Dispose();

        /// <summary>
        /// A fast, stable fingerprint of a model file: SHA-256 over its length plus the first and last
        /// 1 MiB. Near-instant even for multi-GB files (no full read), and identifies a specific model
        /// build in practice. This is a partial fingerprint, not a full content hash — labelled as such.
        /// </summary>
        // OVERFIT040 — synchronous by design: a one-shot STARTUP path. The single caller is Program.Main,
        // before the host is built, to label the build-info metric once. Nothing is serving at that point,
        // so there is no request thread to free and no pool to starve; the two 1 MiB reads happen while the
        // process is doing nothing else.
#pragma warning disable OVERFIT040
        public static string FingerprintModel(string path)
#pragma warning restore OVERFIT040
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

                return Convert.ToHexString(hash.GetHashAndReset()).Substring(0, 16).ToLowerInvariant();
            }
            catch
            {
                return "unavailable";
            }
        }
    }
}
