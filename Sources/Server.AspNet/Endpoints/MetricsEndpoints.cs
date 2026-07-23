// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Globalization;
using System.Text;
using DevOnBike.Overfit.Server.AspNet.Services;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;

namespace DevOnBike.Overfit.Server.AspNet.Endpoints
{
    /// <summary>
    /// <c>GET /metrics</c> — process metrics in the Prometheus text exposition format, hand-rolled so the
    /// whole path stays reflection-free and Native-AOT-clean (the OpenTelemetry Prometheus exporter drags
    /// reflection-heavy dependencies that would break the AOT guard, the same trap as the OpenAPI generator).
    ///
    /// <para>The headline gauge is <c>process_resident_memory_bytes</c> (the working set): with the model
    /// memory-mapped it grows as weight pages are touched, so it is the honest "how much RAM the server holds
    /// with the GGUF loaded" number. CPU is exposed the Prometheus way — <c>process_cpu_seconds_total</c> as a
    /// counter, from which a dashboard derives utilisation via <c>rate()</c>.</para>
    /// </summary>
    internal static class MetricsEndpoints
    {
        private static readonly double StartUnixSeconds =
            new DateTimeOffset(Process.GetCurrentProcess().StartTime.ToUniversalTime()).ToUnixTimeMilliseconds() / 1000.0;

        public static WebApplication MapMetrics(this WebApplication app)
        {
            app.MapGet("/metrics", (ServerMetrics metrics, IOpenAiInferenceService service) =>
                Results.Text(Render(metrics, service.PoolStatus), "text/plain; version=0.0.4; charset=utf-8"));

            return app;
        }

        private static string Render(ServerMetrics metrics, PoolStatus pool)
        {
            using var process = Process.GetCurrentProcess();
            var sb = new StringBuilder(2048);

            // ── Server metrics: requests, tokens (rate() -> tokens/s), and live session-pool load. ──
            Counter(sb, "overfit_chat_requests_total", "Completed chat-completion requests.", metrics.ChatRequests);
            Counter(sb, "overfit_embedding_requests_total", "Completed embedding requests.", metrics.EmbeddingRequests);
            Counter(sb, "overfit_speech_requests_total", "Completed text-to-speech requests.", metrics.SpeechRequests);
            Counter(sb, "overfit_prompt_tokens_total", "Prompt tokens processed across all chat requests.", metrics.PromptTokens);
            Counter(sb, "overfit_generated_tokens_total",
                "Tokens generated across all chat requests (rate() gives tokens/second).", metrics.GeneratedTokens);

            Gauge(sb, "overfit_pool_size", "Total sessions in the pool (max concurrent decodes).", pool.Size);
            Gauge(sb, "overfit_pool_active_sessions", "Sessions currently decoding a request.", pool.Active);
            Gauge(sb, "overfit_pool_available_sessions", "Sessions free to serve a request right now.", pool.Available);
            Gauge(sb, "overfit_pool_peak_active_sessions", "High-water mark of concurrent active sessions.", pool.PeakActive);
            Counter(sb, "overfit_pool_rejected_total", "Requests shed with HTTP 503 because the pool was full.", pool.RejectedTotal);

            metrics.Ttft.Write(sb, "overfit_chat_ttft", "Server-side time to first streamed token, in seconds.");
            metrics.ResponseTime.Write(sb, "overfit_chat_response_time", "Chat completion wall-clock time, in seconds.");

            Gauge(sb, "process_resident_memory_bytes",
                "Resident set size (working set) in bytes — includes paged-in mmap'd model weights.",
                process.WorkingSet64);
            Gauge(sb, "process_private_memory_bytes",
                "Private (committed) memory in bytes.", process.PrivateMemorySize64);
            Gauge(sb, "process_virtual_memory_bytes",
                "Virtual address space in bytes (includes the mmap'd model, mostly not resident).",
                process.VirtualMemorySize64);

            Counter(sb, "process_cpu_seconds_total",
                "Total user + system CPU time consumed by the process, in seconds.",
                process.TotalProcessorTime.TotalSeconds);
            Gauge(sb, "process_start_time_seconds",
                "Process start time since the unix epoch, in seconds.", StartUnixSeconds);
            Gauge(sb, "process_num_threads", "Number of OS threads.", process.Threads.Count);

            Gauge(sb, "dotnet_total_memory_bytes",
                "Managed GC heap memory currently allocated, in bytes.", GC.GetTotalMemory(forceFullCollection: false));
            var gc = GC.GetGCMemoryInfo();
            Gauge(sb, "dotnet_gc_heap_size_bytes", "GC heap size after the last collection, in bytes.", gc.HeapSizeBytes);
            Gauge(sb, "dotnet_gc_committed_bytes", "Committed GC memory, in bytes.", gc.TotalCommittedBytes);

            sb.Append("# HELP dotnet_gc_collections_total Number of GC collections, by generation.\n");
            sb.Append("# TYPE dotnet_gc_collections_total counter\n");
            for (var generation = 0; generation <= GC.MaxGeneration; generation++)
            {
                sb.Append("dotnet_gc_collections_total{generation=\"")
                  .Append(generation)
                  .Append("\"} ")
                  .Append(GC.CollectionCount(generation).ToString(CultureInfo.InvariantCulture))
                  .Append('\n');
            }

            return sb.ToString();
        }

        private static void Gauge(StringBuilder sb, string name, string help, long value)
            => Metric(sb, name, help, "gauge", value.ToString(CultureInfo.InvariantCulture));

        private static void Gauge(StringBuilder sb, string name, string help, double value)
            => Metric(sb, name, help, "gauge", Format(value));

        private static void Counter(StringBuilder sb, string name, string help, double value)
            => Metric(sb, name, help, "counter", Format(value));

        private static void Counter(StringBuilder sb, string name, string help, long value)
            => Metric(sb, name, help, "counter", value.ToString(CultureInfo.InvariantCulture));

        private static void Metric(StringBuilder sb, string name, string help, string type, string value)
        {
            sb.Append("# HELP ").Append(name).Append(' ').Append(help).Append('\n');
            sb.Append("# TYPE ").Append(name).Append(' ').Append(type).Append('\n');
            sb.Append(name).Append(' ').Append(value).Append('\n');
        }

        // Prometheus wants a plain decimal (no thousands separators, no scientific notation).
        private static string Format(double value) => value.ToString("0.######", CultureInfo.InvariantCulture);
    }
}
