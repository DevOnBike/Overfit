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
    /// Client authentication enforcement on the live gateway: with a gateway key configured, a chat request without
    /// (or with a wrong) bearer token is refused with 401 BEFORE anything is forwarded upstream, while <c>/health</c>
    /// stays open for liveness probes. In-process — the gateway runs on a background thread; no upstream is needed
    /// because rejection happens at the door.
    /// </summary>
    public sealed class RedactionGatewayAuthTests
    {
        [Fact]
        public void AuthEnabled_RejectsMissingAndWrongKey_HealthStaysOpen()
        {
            var gatewayPort = FreePort();
            var gatewayThread = new Thread(() =>
            {
                RedactionGateway.Serve(
                    "127.0.0.1",
                    gatewayPort,
                    "http://127.0.0.1:1/v1", // unreachable upstream — auth must reject before we ever touch it
                    "sk-upstream-not-leaked",
                    Redactor.CreateDefault(),
                    new NullAuditSink(),
                    RedactionPolicy.Default(),
                    new[] { "sk-gateway-client" });
            })
            { IsBackground = true };
            gatewayThread.Start();

            var baseUrl = $"http://127.0.0.1:{gatewayPort}";
            WaitForHealth($"{baseUrl}/health");

            using var client = new HttpClient();

            // /health needs no key.
            Assert.Equal(HttpStatusCode.OK, client.GetAsync($"{baseUrl}/health").GetAwaiter().GetResult().StatusCode);

            // Chat without a key → 401.
            Assert.Equal(HttpStatusCode.Unauthorized, Post(client, baseUrl, authHeader: null));

            // Chat with a wrong key → 401.
            Assert.Equal(HttpStatusCode.Unauthorized, Post(client, baseUrl, authHeader: "Bearer sk-wrong"));
        }

        private static HttpStatusCode Post(HttpClient client, string baseUrl, string? authHeader)
        {
            const string body =
                "{\"model\":\"gpt-4\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
            using var msg = new HttpRequestMessage(HttpMethod.Post, $"{baseUrl}/v1/chat/completions")
            {
                Content = new StringContent(body, Encoding.UTF8, "application/json")
            };
            if (authHeader is not null)
            {
                msg.Headers.TryAddWithoutValidation("Authorization", authHeader);
            }
            using var resp = client.Send(msg);
            return resp.StatusCode;
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
