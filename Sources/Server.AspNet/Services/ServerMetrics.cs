// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics.Metrics;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.Server.OpenAi;

namespace DevOnBike.Overfit.Server.AspNet.Services
{
    /// <summary>
    /// Collects the server's request / token / session-pool metrics through the idiomatic .NET metrics API —
    /// a <see cref="Meter"/> with observable instruments — so the same numbers are visible to
    /// <c>dotnet-counters</c>, an OpenTelemetry pipeline, or any other <c>System.Diagnostics.Metrics</c>
    /// consumer, not only the built-in Prometheus endpoint.
    ///
    /// <para><b>Why the Meter, but not the OpenTelemetry Prometheus exporter.</b> The instrumentation side
    /// (<see cref="Meter"/>, observable counters/gauges) is part of the framework and Native-AOT-clean. The
    /// usual Prometheus <i>exporter</i> is not — it drags the same reflection-heavy dependencies that trip the
    /// AOT guard (the trap the OpenAPI generator hit). So the values are published to the Meter for tooling,
    /// and exposed to Prometheus by the hand-rolled <c>/metrics</c> endpoint, which reads the counters below
    /// directly.</para>
    ///
    /// <para>It doubles as an <see cref="IChatExchangeObserver"/>: the shared chat exchange calls
    /// <see cref="OnCompleted"/> at the end of every completion, which is exactly where the token counts are
    /// known. Counters are interlocked — several pooled sessions decode concurrently.</para>
    /// </summary>
    public sealed class ServerMetrics : IChatExchangeObserver, IDisposable
    {
        private readonly Meter _meter = new("DevOnBike.Overfit.Server", "1.0.0");

        private long _chatRequests;
        private long _embeddingRequests;
        private long _speechRequests;
        private long _promptTokens;
        private long _generatedTokens;

        private readonly Histogram<double> _ttftMeter;
        private readonly Histogram<double> _responseMeter;

        // Prometheus-side accumulation of the same samples (the Meter histograms don't expose their buckets).
        internal LatencyHistogram Ttft { get; } = new();
        internal LatencyHistogram ResponseTime { get; } = new();

        public ServerMetrics()
        {
            _ttftMeter = _meter.CreateHistogram<double>("overfit.chat.ttft",
                unit: "s", description: "Server-side time to first token (streaming).");
            _responseMeter = _meter.CreateHistogram<double>("overfit.chat.response_time",
                unit: "s", description: "Chat completion wall-clock time.");

            // Observable counters read the interlocked totals on demand — no double-bookkeeping, and the
            // hand-rolled /metrics endpoint reads the same fields.
            _meter.CreateObservableCounter("overfit.chat.requests", () => ChatRequests,
                unit: "{request}", description: "Completed chat-completion requests.");
            _meter.CreateObservableCounter("overfit.embedding.requests", () => EmbeddingRequests,
                unit: "{request}", description: "Completed embedding requests.");
            _meter.CreateObservableCounter("overfit.speech.requests", () => SpeechRequests,
                unit: "{request}", description: "Completed text-to-speech requests.");
            _meter.CreateObservableCounter("overfit.chat.prompt_tokens", () => PromptTokens,
                unit: "{token}", description: "Prompt tokens processed across all chat requests.");
            _meter.CreateObservableCounter("overfit.chat.generated_tokens", () => GeneratedTokens,
                unit: "{token}", description: "Tokens generated across all chat requests.");
        }

        public long ChatRequests => Interlocked.Read(ref _chatRequests);
        public long EmbeddingRequests => Interlocked.Read(ref _embeddingRequests);
        public long SpeechRequests => Interlocked.Read(ref _speechRequests);
        public long PromptTokens => Interlocked.Read(ref _promptTokens);
        public long GeneratedTokens => Interlocked.Read(ref _generatedTokens);

        /// <summary>
        /// Registers the live session-pool gauges on the Meter. Called once by the service, which owns the
        /// pool; the same snapshot backs the <c>/metrics</c> pool gauges.
        /// </summary>
        public void BindPool(Func<PoolStatus> poolStatus)
        {
            ArgumentNullException.ThrowIfNull(poolStatus);
            _meter.CreateObservableGauge("overfit.pool.size", () => poolStatus().Size,
                description: "Total sessions in the pool (max concurrent decodes).");
            _meter.CreateObservableGauge("overfit.pool.active_sessions", () => poolStatus().Active,
                description: "Sessions currently decoding a request.");
            _meter.CreateObservableGauge("overfit.pool.available_sessions", () => poolStatus().Available,
                description: "Sessions free to serve a request right now.");
        }

        public void RecordEmbeddingRequest() => Interlocked.Increment(ref _embeddingRequests);

        public void RecordSpeechRequest() => Interlocked.Increment(ref _speechRequests);

        /// <summary>Records the wall-clock time of one chat completion (called by the service).</summary>
        public void RecordResponseTime(double seconds)
        {
            _responseMeter.Record(seconds);
            ResponseTime.Record(seconds);
        }

        /// <summary>Server-side time to first streamed token (called by the chat exchange for streaming requests).</summary>
        public void OnFirstToken(double elapsedMs)
        {
            var seconds = elapsedMs / 1000.0;
            _ttftMeter.Record(seconds);
            Ttft.Record(seconds);
        }

        /// <summary>Records one completed chat request and its token counts (called by the chat exchange).</summary>
        public void OnCompleted(bool streamed, GenerationStats stats, int cachedPromptTokens)
        {
            Interlocked.Increment(ref _chatRequests);
            Interlocked.Add(ref _promptTokens, stats.PromptTokens);
            Interlocked.Add(ref _generatedTokens, stats.GeneratedTokens);
        }

        public void Dispose() => _meter.Dispose();
    }
}
