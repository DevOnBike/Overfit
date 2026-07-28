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

            // Families are collected first and emitted in name order at the end. Sorting has to happen at
            // this granularity, never per line: a metric's `# HELP` and `# TYPE` belong to the samples that
            // follow them, and a flat sort of the rendered text would separate them and produce output that
            // is no longer valid exposition format.
            //
            // Prometheus itself does not require any order. Two readers do: a human diffing /metrics between
            // two replicas — which is the whole premise of the peer comparison this server is instrumented
            // for — and anyone eyeballing the endpoint for a name they expect to be there.
            var families = new List<(string Name, string Text)>(24);

            // ── Server metrics: requests, tokens (rate() -> tokens/s), and live session-pool load. ──
            Counter(families, "overfit_chat_requests_total", "Completed chat-completion requests.", metrics.ChatRequests);
            Counter(families, "overfit_embedding_requests_total", "Completed embedding requests.", metrics.EmbeddingRequests);
            Counter(families, "overfit_speech_requests_total", "Completed text-to-speech requests.", metrics.SpeechRequests);
            Counter(families, "overfit_prompt_tokens_total", "Prompt tokens processed across all chat requests.", metrics.PromptTokens);
            Counter(families, "overfit_generated_tokens_total",
                "Tokens generated across all chat requests (rate() gives tokens/second).", metrics.GeneratedTokens);

            Gauge(families, "overfit_pool_size", "Total sessions in the pool (max concurrent decodes).", pool.Size);
            Gauge(families, "overfit_pool_active_sessions", "Sessions currently decoding a request.", pool.Active);
            Gauge(families, "overfit_pool_available_sessions", "Sessions free to serve a request right now.", pool.Available);
            Gauge(families, "overfit_pool_peak_active_sessions", "High-water mark of concurrent active sessions.", pool.PeakActive);
            Counter(families, "overfit_pool_rejected_total", "Requests shed with HTTP 503 because the pool was full.", pool.RejectedTotal);

            Histogram(families, metrics.Ttft, "overfit_chat_ttft",
                "Server-side time to first streamed token, in seconds.");
            Histogram(families, metrics.ResponseTime, "overfit_chat_response_time",
                "Chat completion wall-clock time, in seconds.");

            Gauge(families, "process_resident_memory_bytes",
                "Resident set size (working set) in bytes — includes paged-in mmap'd model weights.",
                process.WorkingSet64);
            Gauge(families, "process_private_memory_bytes",
                "Private (committed) memory in bytes.", process.PrivateMemorySize64);
            Gauge(families, "process_virtual_memory_bytes",
                "Virtual address space in bytes (includes the mmap'd model, mostly not resident).",
                process.VirtualMemorySize64);

            Counter(families, "process_cpu_seconds_total",
                "Total user + system CPU time consumed by the process, in seconds.",
                process.TotalProcessorTime.TotalSeconds);
            Gauge(families, "process_start_time_seconds",
                "Process start time since the unix epoch, in seconds.", StartUnixSeconds);
            Gauge(families, "process_num_threads", "Number of OS threads.", process.Threads.Count);

            Gauge(families, "dotnet_total_memory_bytes",
                "Managed GC heap memory currently allocated, in bytes.", GC.GetTotalMemory(forceFullCollection: false));
            var gc = GC.GetGCMemoryInfo();
            Gauge(families, "dotnet_gc_heap_size_bytes", "GC heap size after the last collection, in bytes.", gc.HeapSizeBytes);
            Gauge(families, "dotnet_gc_committed_bytes", "Committed GC memory, in bytes.", gc.TotalCommittedBytes);

            // Fraction of wall-clock time spent in GC pauses is what a consumer wants, but a ratio computed
            // here would be a ratio over the process lifetime and would flatten out. Exposed as the cumulative
            // counter Prometheus expects, so `rate()` gives the pause seconds per second over any window.
            Counter(families, "dotnet_gc_pause_seconds_total",
                "Cumulative time the runtime spent in GC pauses, in seconds (rate() gives the pause ratio).",
                GC.GetTotalPauseDuration().TotalSeconds);

            Gauge(families, "dotnet_threadpool_queue_length",
                "Work items queued to the thread pool and not yet started — an early thread-starvation signal.",
                ThreadPool.PendingWorkItemCount);

            families.Add(("dotnet_gc_collections_total", RenderGcCollections()));
            families.Add(("overfit_http_responses_total", RenderResponsesByStatus(metrics)));

            // Ordinal, not culture-aware: metric names are identifiers, and a culture-sensitive comparison
            // would reorder the endpoint depending on the server's locale — a difference between two
            // replicas that means nothing and would look like a real one.
            families.Sort(static (left, right) => string.CompareOrdinal(left.Name, right.Name));

            var sb = new StringBuilder(2048);
            for (var i = 0; i < families.Count; i++)
            {
                sb.Append(families[i].Text);
            }

            return sb.ToString();
        }

        /// <summary>
        /// Completed responses grouped by status class. All five classes are emitted even when zero, so that
        /// <c>rate(...{status="5xx"})</c> resolves from the first scrape instead of returning "no data" —
        /// which a dashboard renders identically to "no errors" and a detector cannot tell apart either.
        /// </summary>
        private static string RenderResponsesByStatus(ServerMetrics metrics)
        {
            var sb = new StringBuilder(256);

            sb.Append("# HELP overfit_http_responses_total Completed HTTP responses, by status class.\n");
            sb.Append("# TYPE overfit_http_responses_total counter\n");

            for (var statusClass = 1; statusClass <= 5; statusClass++)
            {
                sb.Append("overfit_http_responses_total{status=\"")
                  .Append(statusClass)
                  .Append("xx\"} ")
                  .Append(metrics.ResponsesInClass(statusClass).ToString(CultureInfo.InvariantCulture))
                  .Append('\n');
            }

            return sb.ToString();
        }

        private static string RenderGcCollections()
        {
            var sb = new StringBuilder(256);

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

        private static void Histogram(
            List<(string Name, string Text)> families,
            LatencyHistogram histogram,
            string name,
            string help)
        {
            var sb = new StringBuilder(512);
            histogram.Write(sb, name, help);
            families.Add((name, sb.ToString()));
        }

        private static void Gauge(List<(string Name, string Text)> families, string name, string help, long value)
            => Metric(families, name, help, "gauge", value.ToString(CultureInfo.InvariantCulture));

        private static void Gauge(List<(string Name, string Text)> families, string name, string help, double value)
            => Metric(families, name, help, "gauge", Format(value));

        private static void Counter(List<(string Name, string Text)> families, string name, string help, double value)
            => Metric(families, name, help, "counter", Format(value));

        private static void Counter(List<(string Name, string Text)> families, string name, string help, long value)
            => Metric(families, name, help, "counter", value.ToString(CultureInfo.InvariantCulture));

        private static void Metric(
            List<(string Name, string Text)> families,
            string name,
            string help,
            string type,
            string value)
        {
            var sb = new StringBuilder(160);

            sb.Append("# HELP ").Append(name).Append(' ').Append(help).Append('\n');
            sb.Append("# TYPE ").Append(name).Append(' ').Append(type).Append('\n');
            sb.Append(name).Append(' ').Append(value).Append('\n');

            families.Add((name, sb.ToString()));
        }

        // Prometheus wants a plain decimal (no thousands separators, no scientific notation).
        private static string Format(double value) => value.ToString("0.######", CultureInfo.InvariantCulture);
    }
}
