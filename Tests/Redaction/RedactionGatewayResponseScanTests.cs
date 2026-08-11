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
    /// Response-side scanning end-to-end: with <c>scanResponses</c> on, a model reply that BOTH echoes the caller's
    /// redacted value (restored to the original) AND leaks a different model-generated e-mail must come back with the
    /// caller's own value visible and the model-generated leak masked. Proves the pre-restore scan order distinguishes
    /// "the user's data coming home" from "the model leaking something new". In-process, no external exe.
    /// </summary>
    public sealed class RedactionGatewayResponseScanTests
    {
        [Fact]
        public async Task ScanResponses_MasksModelLeak_ButRestoresCallerValue()
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
                    try
                    {
                        c = upstream.GetContext();
                    }
                    catch { break; }

                    // Guard the whole response: the test's finally{} stops the listener, which can dispose this
                    // response mid-write on this background thread — that must never become an unhandled crash.
                    try
                    {
                        using (var reader = new StreamReader(c.Request.InputStream))
                        {
                            _ = reader.ReadToEnd();
                        }

                        // Assistant content echoes the caller's placeholder ([REDACTED_EMAIL_0], deterministic for the
                        // first e-mail) AND leaks a different, model-generated address.
                        const string content =
                            "Confirmed [REDACTED_EMAIL_0]; our internal contact is leaked@internal.corp.";
                        var body =
                            "{\"id\":\"x\",\"object\":\"chat.completion\",\"created\":0,\"model\":\"m\",\"choices\":"
                            + "[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"" + content + "\"},"
                            + "\"finish_reason\":\"stop\"}],\"usage\":{}}";
                        var bytes = Encoding.UTF8.GetBytes(body);
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
                        clientKeys: null,
                        scanResponses: true);
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

                const string request =
                    "{\"model\":\"gpt-4\",\"messages\":[{\"role\":\"user\",\"content\":\"reach me at bob@example.com\"}]}";
                using var msg = new HttpRequestMessage(HttpMethod.Post, $"{baseUrl}/v1/chat/completions")
                {
                    Content = new StringContent(request, Encoding.UTF8, "application/json")
                };
                using var resp = client.Send(msg);
                var clientGot = await resp.Content.ReadAsStringAsync();

                Assert.Equal(HttpStatusCode.OK, resp.StatusCode);

                // The caller's own value came home (restored from its placeholder).
                Assert.Contains("bob@example.com", clientGot);

                // The model-generated leak was masked — never reaches the client.
                Assert.DoesNotContain("leaked@internal.corp", clientGot);
                Assert.Contains("REDACTED-RESPONSE-EMAIL", clientGot);
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
