// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Specialized;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Text;
using DevOnBike.Overfit.Redaction;
using DevOnBike.Overfit.Server;

namespace DevOnBike.Overfit.Tests.Redaction
{
    /// <summary>
    /// Header transparency of the gateway: a caller's custom request headers (OpenAI-Beta, X-*) reach the upstream
    /// so client features keep working, the caller's Authorization (its gateway key) is NEVER forwarded — the real
    /// upstream key is injected in its place — and the upstream's response headers (rate limits, request id) flow
    /// back to the caller. In-process, no external exe.
    /// </summary>
    public sealed class RedactionGatewayHeaderForwardingTests
    {
        [Fact]
        public void ForwardsClientHeaders_StripsClientAuth_ReturnsUpstreamResponseHeaders()
        {
            NameValueCollection? upstreamHeaders = null;
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
                        // Snapshot the headers the upstream actually received.
                        upstreamHeaders = new NameValueCollection(c.Request.Headers);
                        using (var reader = new StreamReader(c.Request.InputStream))
                        {
                            _ = reader.ReadToEnd();
                        }

                        const string body =
                            "{\"id\":\"x\",\"object\":\"chat.completion\",\"created\":0,\"model\":\"m\",\"choices\":"
                            + "[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"ok\"},\"finish_reason\":\"stop\"}],"
                            + "\"usage\":{}}";
                        var bytes = Encoding.UTF8.GetBytes(body);
                        c.Response.Headers["x-ratelimit-remaining-requests"] = "42";
                        c.Response.ContentType = "application/json";
                        c.Response.ContentLength64 = bytes.Length;
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
                        "sk-upstream-real",
                        Redactor.CreateDefault(),
                        new NullAuditSink(),
                        RedactionPolicy.Default());
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

                using var msg = new HttpRequestMessage(HttpMethod.Post, $"{baseUrl}/v1/chat/completions")
                {
                    Content = new StringContent(
                        "{\"model\":\"gpt-4\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
                        Encoding.UTF8, "application/json")
                };
                msg.Headers.TryAddWithoutValidation("OpenAI-Beta", "assistants=v2");
                msg.Headers.TryAddWithoutValidation("X-Custom-Trace", "abc123");
                msg.Headers.TryAddWithoutValidation("Authorization", "Bearer sk-client-key-should-not-leak");

                using var resp = client.Send(msg);
                Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
                Assert.NotNull(upstreamHeaders);

                // Custom client headers reached the upstream.
                Assert.Equal("assistants=v2", upstreamHeaders!["OpenAI-Beta"]);
                Assert.Equal("abc123", upstreamHeaders["X-Custom-Trace"]);

                // SECURITY: the caller's Authorization (its gateway key) was replaced by the injected upstream key.
                Assert.Equal("Bearer sk-upstream-real", upstreamHeaders["Authorization"]);
                Assert.DoesNotContain("sk-client-key-should-not-leak", upstreamHeaders["Authorization"] ?? string.Empty);

                // Upstream response headers (rate limits) flow back to the caller.
                Assert.True(resp.Headers.TryGetValues("x-ratelimit-remaining-requests", out var values));
                Assert.Equal("42", string.Concat(values!));
            }
            finally
            {
                upstream.Stop();
            }
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
