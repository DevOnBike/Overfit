// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Net;
using System.Net.Sockets;
using DevOnBike.Overfit.Redaction;
using DevOnBike.Overfit.Server;

namespace DevOnBike.Overfit.Tests.Redaction
{
    /// <summary>
    /// Cancelling the token handed to <see cref="RedactionGateway.Serve"/> unwinds the gateway: the call returns
    /// and the port stops answering. Every other gateway test passes <c>CancellationToken.None</c> and lets the
    /// background thread die at process end, so before this one nothing anywhere exercised the shutdown path —
    /// <c>Task.Delay(Timeout.Infinite, token)</c> and the <c>app.StopAsync()</c> that follows it.
    ///
    /// <para><b>What this does NOT cover, stated because the gap is the interesting part (<c>XC-23</c>).</b>
    /// The defect was in <c>Sources/Cli/Commands.cs</c>, which wired no <c>Console.CancelKeyPress</c> handler
    /// and therefore had no token to pass. This test pins the callee's contract, not that wiring: a
    /// <c>Console.CancelKeyPress</c> handler is raised by the OS console control handler and cannot be raised
    /// in-process, and <c>Tests.csproj</c> carries no <c>ProjectReference</c> to <c>Sources/Cli</c>, so the
    /// command is not constructible from the suite either. The CLI half is verified by reading it. What this
    /// test establishes is that the token the command now passes is one the gateway acts on — without which
    /// the fix would be cosmetic.</para>
    /// </summary>
    public sealed class RedactionGatewayShutdownTests
    {
        [Fact]
        public async Task Serve_ReturnsAndStopsAnswering_WhenItsTokenIsCancelled()
        {
            var port = FreePort();
            using var cts = new CancellationTokenSource();
            using var returned = new ManualResetEventSlim(false);
            Exception? failure = null;

            var thread = new Thread(() =>
            {
                try
                {
                    RedactionGateway.Serve(
                        "127.0.0.1",
                        port,
                        "http://127.0.0.1:1/v1", // unreachable upstream — nothing is proxied here.
                        "sk-upstream-not-leaked",
                        Redactor.CreateDefault(),
                        new NullAuditSink(),
                        RedactionPolicy.Default(),
                        cts.Token);
                }
                catch (Exception ex)
                {
                    failure = ex;
                }
                finally
                {
                    returned.Set();
                }
            })
            {
                IsBackground = true
            };
            thread.Start();

            // Capability check before the verdict: it really was serving, and it had NOT returned on its own.
            // Without both, a Serve that threw at startup would satisfy the assertions below for free.
            var baseUrl = $"http://127.0.0.1:{port}";
            await WaitForHealth($"{baseUrl}/health");
            Assert.False(returned.IsSet, "Serve returned before the token was cancelled");

            cts.Cancel();

            Assert.True(
                returned.Wait(TimeSpan.FromSeconds(15)),
                "Serve did not return after its token was cancelled");
            Assert.Null(failure);

            // Returning is not the same as having shut down — assert the listener is gone, which is what
            // `app.StopAsync()` buys and what a bare `return` out of the delay would not.
            Assert.True(
                await StoppedAnswering($"{baseUrl}/health"),
                "the gateway kept answering after Serve returned");
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
        /// Polls until the gateway answers, or gives up after fifteen seconds. Asynchronous for the same reason
        /// as its twin in <c>RedactionGatewayAuthTests</c> (xunit v3's <c>xUnit1031</c>).
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

        /// <summary>
        /// True once the port refuses or fails the probe. Polled rather than checked once: the socket is closed
        /// by the host's shutdown, which finishes shortly after <c>Serve</c> returns rather than before it.
        /// </summary>
        private static async Task<bool> StoppedAnswering(string url)
        {
            using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(1) };
            var deadline = Environment.TickCount64 + 10_000;
            while (Environment.TickCount64 < deadline)
            {
                try
                {
                    await client.GetAsync(url);
                }
                catch
                {
                    return true;
                }
                await Task.Delay(100);
            }
            return false;
        }
    }
}
