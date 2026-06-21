// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Text;
using DevOnBike.Overfit.Redaction;
using DevOnBike.Overfit.Server;

namespace DevOnBike.Overfit.Tests.Redaction
{
    /// <summary>
    /// Streaming (SSE) path of the gateway, fully in-process: a mock upstream emits a chat-completion stream with a
    /// redaction placeholder deliberately split across SSE chunks; the gateway must forward the REDACTED request, then
    /// re-hydrate the placeholder on the streamed response so the client reassembles the original value. No external
    /// exe — the gateway runs on a background thread.
    /// </summary>
    public sealed class RedactionGatewayStreamingTests
    {
        [Fact]
        public void Streaming_RedactsRequest_RestoresSplitPlaceholderAcrossChunks()
        {
            string? upstreamReceived = null;
            var upstreamPort = FreePort();
            using var upstream = new HttpListener();
            upstream.Prefixes.Add($"http://127.0.0.1:{upstreamPort}/");
            upstream.Start();

            var upstreamThread = new Thread(() =>
            {
                while (upstream.IsListening)
                {
                    HttpListenerContext c;
                    try { c = upstream.GetContext(); }
                    catch { break; }

                    using (var reader = new StreamReader(c.Request.InputStream))
                    {
                        upstreamReceived = reader.ReadToEnd();
                    }

                    // SSE stream that splits "[REDACTED_EMAIL_0]" across two data chunks.
                    c.Response.ContentType = "text/event-stream";
                    var sb = new StringBuilder();
                    sb.Append(Chunk("Reach me at "));
                    sb.Append(Chunk("[REDACTED_EMA"));
                    sb.Append(Chunk("IL_0] anytime."));
                    sb.Append("data: [DONE]\n\n");
                    var bytes = Encoding.UTF8.GetBytes(sb.ToString());
                    c.Response.OutputStream.Write(bytes, 0, bytes.Length);
                    c.Response.OutputStream.Close();
                }
            })
            { IsBackground = true };
            upstreamThread.Start();

            var gatewayPort = FreePort();
            var gatewayThread = new Thread(() =>
            {
                RedactionGateway.Serve(
                    "127.0.0.1",
                    gatewayPort,
                    $"http://127.0.0.1:{upstreamPort}/v1",
                    "sk-not-leaked",
                    Redactor.CreateDefault(),
                    new NullAuditSink(),
                    RedactionPolicy.Default());
            })
            { IsBackground = true };
            gatewayThread.Start();

            try
            {
                WaitForHealth($"http://127.0.0.1:{gatewayPort}/health");

                using var client = new HttpClient();
                const string request =
                    "{\"model\":\"gpt-4\",\"stream\":true,"
                    + "\"messages\":[{\"role\":\"user\",\"content\":\"reach me at bob@example.com\"}]}";

                using var msg = new HttpRequestMessage(
                    HttpMethod.Post, $"http://127.0.0.1:{gatewayPort}/v1/chat/completions")
                {
                    Content = new StringContent(request, Encoding.UTF8, "application/json")
                };
                using var resp = client.Send(msg, HttpCompletionOption.ResponseHeadersRead);
                Assert.Equal(HttpStatusCode.OK, resp.StatusCode);

                // Concatenate the delta.content across the streamed chunks the client receives.
                var assembled = new StringBuilder();
                using (var stream = resp.Content.ReadAsStream())
                using (var sr = new StreamReader(stream))
                {
                    string? line;
                    while ((line = sr.ReadLine()) is not null)
                    {
                        if (!line.StartsWith("data:", StringComparison.Ordinal))
                        {
                            continue;
                        }
                        var payload = line.Substring("data:".Length).Trim();
                        if (payload is "[DONE]" or "")
                        {
                            continue;
                        }
                        assembled.Append(ExtractContent(payload));
                    }
                }

                var clientText = assembled.ToString();

                // The upstream saw only the placeholder; the streamed client output got the original restored whole.
                Assert.NotNull(upstreamReceived);
                Assert.DoesNotContain("bob@example.com", upstreamReceived);
                Assert.Contains("REDACTED_EMAIL", upstreamReceived);

                Assert.Equal("Reach me at bob@example.com anytime.", clientText);
                Assert.DoesNotContain("REDACTED", clientText);
            }
            finally
            {
                upstream.Stop();
            }
        }

        private static string Chunk(string content)
        {
            var escaped = content.Replace("\"", "\\\"");
            return "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\","
                + "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" + escaped + "\"}}]}\n\n";
        }

        // Minimal extractor for the test's own assertions — pulls delta.content out of a chunk JSON.
        private static string ExtractContent(string chunkJson)
        {
            const string marker = "\"content\":\"";
            var i = chunkJson.IndexOf(marker, StringComparison.Ordinal);
            if (i < 0)
            {
                return string.Empty;
            }
            i += marker.Length;
            var end = chunkJson.IndexOf('"', i);
            return end < 0 ? string.Empty : chunkJson.Substring(i, end - i);
        }

        private sealed class NullAuditSink : IRedactionAuditSink
        {
            public void Record(RedactionAuditRecord record)
            {
            }
        }

        private static int FreePort()
        {
            var l = new TcpListener(IPAddress.Loopback, 0);
            l.Start();
            var port = ((IPEndPoint)l.LocalEndpoint).Port;
            l.Stop();
            return port;
        }

        private static void WaitForHealth(string url)
        {
            using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(1) };
            var deadline = Environment.TickCount64 + 15_000;
            while (Environment.TickCount64 < deadline)
            {
                try
                {
                    var r = client.GetAsync(url).GetAwaiter().GetResult();
                    if (r.IsSuccessStatusCode)
                    {
                        return;
                    }
                }
                catch
                {
                    // not up yet
                }
                Thread.Sleep(100);
            }
            throw new TimeoutException($"gateway did not become healthy at {url}");
        }
    }
}
