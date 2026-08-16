// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Net;
using System.Net.Sockets;
using System.Text;
using DevOnBike.Overfit.Redaction;
using DevOnBike.Overfit.Server;

namespace DevOnBike.Overfit.Tests.Redaction
{
    /// <summary>
    /// Endpoint coverage of the gateway beyond chat: the generic redacting proxy must redact a JSON body on
    /// <c>/v1/embeddings</c> (the upstream sees only the placeholder), restore placeholders on the response, map the
    /// path to the upstream correctly, and pass a <c>GET /v1/models</c> straight through. In-process — gateway on a
    /// background thread, mock upstream echoes the request body so both directions can be asserted.
    /// </summary>
    public sealed class RedactionGatewayEndpointsTests
    {
        [Fact]
        public async Task GenericProxy_RedactsEmbeddingsBody_RestoresResponse_PassesThroughGet()
        {
            string? upstreamPath = null;
            string? upstreamBody = null;
            var upstreamPort = FreePort();
            using var upstream = new HttpListener();
            upstream.Prefixes.Add($"http://127.0.0.1:{upstreamPort}/");
            upstream.Start();

            var upstreamThread = new Thread(() =>
            {
                while (upstream.IsListening)
                {
                    HttpListenerContext c;
                    try
                    {
                        c = upstream.GetContext();
                    }
                    catch { break; }

                    // Guard the whole response: the test's finally{} stops the listener, which can dispose this
                    // response mid-write on this background thread — that must never become an unhandled crash.
                    try
                    {
                        upstreamPath = c.Request.Url?.AbsolutePath;

                        string body;
                        using (var reader = new StreamReader(c.Request.InputStream))
                        {
                            body = reader.ReadToEnd();
                        }

                        string responseBody;
                        if (c.Request.HttpMethod == "GET")
                        {
                            responseBody = "{\"object\":\"list\",\"data\":[{\"id\":\"gpt-4\",\"object\":\"model\"}]}";
                        }
                        else
                        {
                            upstreamBody = body;
                            // Echo the received (already-redacted) body back so the gateway's restore step is exercised.
                            responseBody = body;
                        }

                        var bytes = Encoding.UTF8.GetBytes(responseBody);
                        c.Response.ContentType = "application/json";
                        c.Response.ContentLength64 = bytes.Length;
                        c.Response.OutputStream.Write(bytes, 0, bytes.Length);
                        c.Response.OutputStream.Close();
                    }
                    catch
                    {
                        // listener stopped / client gone during teardown — ignore on this background thread.
                    }
                }
            })
            {
                IsBackground = true
            };
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
                        CancellationToken.None);
                }
                catch
                {
                    // bind race / listener torn down at test end — never crash the test host from this thread.
                }
            })
            {
                IsBackground = true
            };
            gatewayThread.Start();

            var baseUrl = $"http://127.0.0.1:{gatewayPort}";
            try
            {
                await WaitForHealth($"{baseUrl}/health");
                using var client = new HttpClient();

                // ── /v1/embeddings: body carries an e-mail; the upstream must see only the placeholder. ──
                const string embedRequest =
                    "{\"model\":\"text-embedding-3-small\",\"input\":\"email me at bob@example.com\"}";
                using var embedMsg = new HttpRequestMessage(HttpMethod.Post, $"{baseUrl}/v1/embeddings")
                {
                    Content = new StringContent(embedRequest, Encoding.UTF8, "application/json")
                };
                using var embedResp = client.Send(embedMsg);
                var clientGot = await embedResp.Content.ReadAsStringAsync();

                Assert.Equal(HttpStatusCode.OK, embedResp.StatusCode);

                // Path mapped /v1/embeddings → upstream .../v1/embeddings (the /v1 is not doubled).
                Assert.Equal("/v1/embeddings", upstreamPath);

                // Egress: the upstream never saw the e-mail, only the placeholder.
                Assert.NotNull(upstreamBody);
                Assert.DoesNotContain("bob@example.com", upstreamBody);
                Assert.Contains("REDACTED_EMAIL", upstreamBody);

                // Restore: the client got the original value re-hydrated from the echoed placeholder.
                Assert.Contains("bob@example.com", clientGot);
                Assert.DoesNotContain("REDACTED", clientGot);

                // ── GET /v1/models passes straight through. ──
                using var modelsResp = await client.GetAsync($"{baseUrl}/v1/models");
                Assert.Equal(HttpStatusCode.OK, modelsResp.StatusCode);
                Assert.Contains("gpt-4", await modelsResp.Content.ReadAsStringAsync());
                Assert.Equal("/v1/models", upstreamPath);
            }
            finally
            {
                upstream.Stop();
            }
        }

        private sealed class NullAuditSink : IRedactionAuditSink
        {
            public void Record(in RedactionAuditEntry entry)
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

        /// <summary>
        /// Polls until the gateway answers, or gives up after fifteen seconds.
        ///
        /// <para>Asynchronous since 2026-08-11 (<c>XC-18</c>): it used to block on the probe with
        /// <c>GetAwaiter().GetResult()</c>, which xunit v3's <c>xUnit1031</c> flags as a deadlock risk. The
        /// <c>Task.Delay</c> between attempts is deliberate and is NOT the sleep-as-synchronisation smell —
        /// the loop is polling an external process's startup, which has no event to wait on.</para>
        /// </summary>
        private static async Task WaitForHealth(string url)
        {
            using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(1) };
            var deadline = Environment.TickCount64 + 15_000;
            while (Environment.TickCount64 < deadline)
            {
                try
                {
                    if ((await client.GetAsync(url)).IsSuccessStatusCode)
                    {
                        return;
                    }
                }
                catch
                {
                    // not up yet
                }
                await Task.Delay(100);
            }
            throw new TimeoutException($"gateway did not become healthy at {url}");
        }
    }
}
