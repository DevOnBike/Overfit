// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;
using System.Diagnostics;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text.Json;
using DevOnBike.Overfit.Serving;

namespace DevOnBike.Overfit.Cli
{
    /// <summary>
    /// Concurrent serving load test for any OpenAI-compatible streaming chat endpoint (Overfit's own
    /// <c>overfit serve</c>, or another provider for an apples-to-apples comparison). N virtual users
    /// hammer <c>POST {url}/chat/completions</c> with <c>stream:true</c>; per request it measures
    /// time-to-first-token, end-to-end latency and the streamed chunk count, then folds everything into
    /// a <see cref="ServingLoadReport"/> (latency percentiles, throughput, goodput, holistic score).
    /// </summary>
    internal static class ServingBenchmark
    {
        public static int Run(
            string url,
            string model,
            int users,
            int requests,
            int maxTokens,
            string prompt,
            int warmup,
            double costUnits)
            => RunAsync(url, model, users, requests, maxTokens, prompt, warmup, costUnits).GetAwaiter().GetResult();

        private static async Task<int> RunAsync(
            string url,
            string model,
            int users,
            int requests,
            int maxTokens,
            string prompt,
            int warmup,
            double costUnits)
        {
            users = Math.Max(1, users);
            requests = Math.Max(users, requests);
            var endpoint = url.TrimEnd('/') + "/chat/completions";

            using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(5) };

            Console.WriteLine($"Serving benchmark → {endpoint}");
            Console.WriteLine($"  model={model}  users={users}  requests={requests}  max_tokens={maxTokens}  warmup={warmup}");
            Console.WriteLine();

            // Warm-up requests are NOT measured — they let the server JIT, load the model and fill caches
            // so the scored window reflects steady state, not cold start (the GPU-leaderboard discipline).
            if (warmup > 0)
            {
                Console.Write($"warming up ({warmup} requests)... ");
                var warm = new Task[Math.Min(warmup, users)];
                var warmRemaining = warmup;
                for (var i = 0; i < warm.Length; i++)
                {
                    warm[i] = Task.Run(async () =>
                    {
                        while (Interlocked.Decrement(ref warmRemaining) >= 0)
                        {
                            await OneRequestAsync(http, endpoint, model, prompt, maxTokens);
                        }
                    });
                }

                await Task.WhenAll(warm);
                Console.WriteLine("done.");
            }

            var samples = new List<ServingRequestSample>(requests);
            var sync = new object();
            var remaining = requests;
            var completed = 0;

            Console.Write($"running {requests} requests at {users} concurrent users...");
            var clock = Stopwatch.StartNew();

            var workers = new Task[users];
            for (var w = 0; w < users; w++)
            {
                workers[w] = Task.Run(async () =>
                {
                    while (Interlocked.Decrement(ref remaining) >= 0)
                    {
                        var sample = await OneRequestAsync(http, endpoint, model, prompt, maxTokens);
                        lock (sync)
                        {
                            samples.Add(sample);
                        }

                        Interlocked.Increment(ref completed);
                    }
                });
            }

            await Task.WhenAll(workers);
            clock.Stop();
            Console.WriteLine($" {completed} done in {clock.Elapsed.TotalSeconds:F1}s.");
            Console.WriteLine();

            var report = ServingLoadReport.From(samples, users, clock.Elapsed.TotalSeconds, costUnits);
            PrintReport(report);
            return report.SuccessfulRequests > 0 ? 0 : 1;
        }

        private static async Task<ServingRequestSample> OneRequestAsync(
            HttpClient http, string endpoint, string model, string prompt, int maxTokens)
        {
            try
            {
                using var content = BuildRequestBody(model, prompt, maxTokens);
                using var request = new HttpRequestMessage(HttpMethod.Post, endpoint) { Content = content };

                var sw = Stopwatch.StartNew();
                using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);

                if (!response.IsSuccessStatusCode)
                {
                    return ServingRequestSample.Failure();
                }

                await using var stream = await response.Content.ReadAsStreamAsync();
                using var reader = new StreamReader(stream);

                var firstTokenMs = -1.0;
                var tokens = 0;

                string? line;
                while ((line = await reader.ReadLineAsync()) != null)
                {
                    if (line.Length == 0 || !line.StartsWith("data:", StringComparison.Ordinal))
                    {
                        continue;
                    }

                    var data = line.AsSpan(5).Trim();
                    if (data.SequenceEqual("[DONE]"))
                    {
                        break;
                    }

                    if (HasContentDelta(data))
                    {
                        if (firstTokenMs < 0.0)
                        {
                            firstTokenMs = sw.Elapsed.TotalMilliseconds;
                        }

                        tokens++;
                    }
                }

                sw.Stop();

                // No content streamed at all → treat as a failed request, not a 0-token success.
                return firstTokenMs < 0.0
                    ? ServingRequestSample.Failure()
                    : ServingRequestSample.Success(firstTokenMs, sw.Elapsed.TotalMilliseconds, tokens);
            }
            catch
            {
                return ServingRequestSample.Failure();
            }
        }

        /// <summary>True when the SSE data line carries a non-empty <c>choices[0].delta.content</c> chunk.</summary>
        private static bool HasContentDelta(ReadOnlySpan<char> data)
        {
            try
            {
                using var doc = JsonDocument.Parse(data.ToString());
                var root = doc.RootElement;

                if (root.ValueKind != JsonValueKind.Object
                    || !root.TryGetProperty("choices", out var choices)
                    || choices.ValueKind != JsonValueKind.Array
                    || choices.GetArrayLength() == 0)
                {
                    return false;
                }

                var first = choices[0];

                if (first.TryGetProperty("delta", out var delta)
                    && delta.TryGetProperty("content", out var contentEl)
                    && contentEl.ValueKind == JsonValueKind.String)
                {
                    return !string.IsNullOrEmpty(contentEl.GetString());
                }

                return false;
            }
            catch (JsonException)
            {
                return false;
            }
        }

        private static ByteArrayContent BuildRequestBody(string model, string prompt, int maxTokens)
        {
            var buffer = new ArrayBufferWriter<byte>();

            using (var writer = new Utf8JsonWriter(buffer))
            {
                writer.WriteStartObject();
                writer.WriteString("model", model);

                writer.WriteStartArray("messages");
                writer.WriteStartObject();
                writer.WriteString("role", "user");
                writer.WriteString("content", prompt);
                writer.WriteEndObject();
                writer.WriteEndArray();

                writer.WriteBoolean("stream", true);
                writer.WriteNumber("max_tokens", maxTokens);
                writer.WriteNumber("temperature", 0);
                writer.WriteEndObject();
            }

            var content = new ByteArrayContent(buffer.WrittenSpan.ToArray());
            content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
            return content;
        }

        private static void PrintReport(ServingLoadReport r)
        {
            Console.WriteLine("── Serving benchmark report ───────────────────────────────");
            Console.WriteLine($"  requests           {r.SuccessfulRequests}/{r.TotalRequests} ok   (error rate {r.ErrorRate * 100:F1}%)");
            Console.WriteLine($"  concurrency        {r.Concurrency} users   ·   window {r.WallClockSeconds:F1}s   ·   cost {r.CostUnits:F0} unit(s)");
            Console.WriteLine();
            Console.WriteLine($"  TTFT  (ms)         p50 {r.TimeToFirstTokenP50Ms,7:F1}   p95 {r.TimeToFirstTokenP95Ms,7:F1}   p99 {r.TimeToFirstTokenP99Ms,7:F1}");
            Console.WriteLine($"  ITL   (ms/token)   p50 {r.InterTokenLatencyP50Ms,7:F2}   p95 {r.InterTokenLatencyP95Ms,7:F2}   p99 {r.InterTokenLatencyP99Ms,7:F2}");
            Console.WriteLine($"  E2E   (ms)         p50 {r.EndToEndP50Ms,7:F1}   p95 {r.EndToEndP95Ms,7:F1}   p99 {r.EndToEndP99Ms,7:F1}");
            Console.WriteLine();
            Console.WriteLine($"  throughput         {r.ThroughputTokensPerSecond,8:F1} tok/s   (goodput {r.Goodput,8:F1} tok/s)");
            Console.WriteLine($"  SCORE              {r.Score,12:N0}");
            Console.WriteLine("────────────────────────────────────────────────────────────");
            Console.WriteLine("  score = (goodput × users) / (TTFT_p95_s × ITL_p95_s × cost) — higher is better.");
        }
    }
}
