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
    /// Streaming response-side scanning end-to-end: with <c>scanResponses</c> on, a model that leaks an e-mail split
    /// across two SSE chunks gets it masked before it reaches the client — even though no single chunk contains the
    /// whole address. In-process, no external exe.
    /// </summary>
    public sealed class RedactionGatewayStreamingScanTests
    {
        [Fact]
        public void Streaming_MasksModelSecret_SplitAcrossChunks()
        {
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

                    try
                    {
                        using (var reader = new StreamReader(c.Request.InputStream))
                        {
                            _ = reader.ReadToEnd();
                        }

                        // The leaked e-mail "leaked@corp.io" is split across two delta chunks.
                        c.Response.ContentType = "text/event-stream";
                        var sb = new StringBuilder();
                        sb.Append(Chunk("Reach me at leaked@cor"));
                        sb.Append(Chunk("p.io today."));
                        sb.Append("data: [DONE]\n\n");
                        var bytes = Encoding.UTF8.GetBytes(sb.ToString());
                        c.Response.OutputStream.Write(bytes, 0, bytes.Length);
                        c.Response.OutputStream.Close();
                    }
                    catch
                    {
                        // listener stopped during teardown — ignore on this background thread.
                    }
                }
            })
            { IsBackground = true };
            upstreamThread.Start();

            var gatewayPort = FreePort();
            var gatewayThread = new Thread(() =>
            {
                try
                {
                    RedactionGateway.Serve(
                        "127.0.0.1",
                        gatewayPort,
                        $"http://127.0.0.1:{upstreamPort}/v1",
                        "sk-not-leaked",
                        Redactor.CreateDefault(),
                        new NullAuditSink(),
                        RedactionPolicy.Default(),
                        clientKeys: null,
                        scanResponses: true);
                }
                catch
                {
                    // bind race / listener torn down at test end — never crash the test host from this thread.
                }
            })
            { IsBackground = true };
            gatewayThread.Start();

            var baseUrl = $"http://127.0.0.1:{gatewayPort}";
            try
            {
                WaitForHealth($"{baseUrl}/health");
                using var client = new HttpClient();

                // No PII in the request — exercises the scan-only path (no placeholder restore involved).
                const string request =
                    "{\"model\":\"gpt-4\",\"stream\":true,"
                    + "\"messages\":[{\"role\":\"user\",\"content\":\"say hi\"}]}";
                using var msg = new HttpRequestMessage(HttpMethod.Post, $"{baseUrl}/v1/chat/completions")
                {
                    Content = new StringContent(request, Encoding.UTF8, "application/json")
                };
                using var resp = client.Send(msg, HttpCompletionOption.ResponseHeadersRead);
                Assert.Equal(HttpStatusCode.OK, resp.StatusCode);

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
                Assert.DoesNotContain("leaked@corp.io", clientText);
                Assert.Contains("REDACTED-RESPONSE-EMAIL", clientText);
                Assert.Contains("Reach me at", clientText);
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
                    if (client.GetAsync(url).GetAwaiter().GetResult().IsSuccessStatusCode)
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
