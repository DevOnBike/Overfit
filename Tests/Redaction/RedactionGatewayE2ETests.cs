// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Text;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Redaction
{
    /// <summary>
    /// Full proxy path of <c>overfit gateway</c>: a real client → the gateway process → a mock upstream. Asserts the
    /// upstream received the REDACTED payload (the secret never left the box), the client got the RESTORED answer,
    /// and the audit log recorded the category — proving the egress firewall end-to-end. Integration — needs the
    /// built exe, so [LongFact]; skips cleanly when the exe is absent.
    /// </summary>
    public sealed class RedactionGatewayE2ETests
    {
        private readonly ITestOutputHelper _output;

        public RedactionGatewayE2ETests(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact]
        public void Gateway_RedactsOutbound_ForwardsRestores_Audits()
        {
            var exe = LocateOverfitExe();
            if (exe is null)
            {
                _output.WriteLine("overfit exe not built — skipping");
                return;
            }

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

                    // Guard the whole response: the test's finally{} stops the listener, which can dispose this
                    // response mid-write on this background thread — that must never become an unhandled crash.
                    try
                    {
                        using (var reader = new StreamReader(c.Request.InputStream))
                        {
                            upstreamReceived = reader.ReadToEnd();
                        }

                        const string body =
                            "{\"id\":\"x\",\"object\":\"chat.completion\",\"created\":0,\"model\":\"m\",\"choices\":"
                            + "[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"ok\"},\"finish_reason\":\"stop\"}],"
                            + "\"usage\":{}}";
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
            { IsBackground = true };
            upstreamThread.Start();

            var gatewayPort = FreePort();
            var auditPath = Path.Combine(Path.GetTempPath(), $"gw_audit_{Guid.NewGuid():N}.jsonl");

            var psi = new ProcessStartInfo(exe)
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false
            };
            psi.ArgumentList.Add("gateway");
            psi.ArgumentList.Add("--upstream");
            psi.ArgumentList.Add($"http://127.0.0.1:{upstreamPort}/v1");
            psi.ArgumentList.Add("--port");
            psi.ArgumentList.Add(gatewayPort.ToString());
            psi.ArgumentList.Add("--upstream-key-env");
            psi.ArgumentList.Add("OVERFIT_GW_TEST_KEY");
            psi.ArgumentList.Add("--audit");
            psi.ArgumentList.Add(auditPath);
            psi.Environment["OVERFIT_GW_TEST_KEY"] = "sk-secret-not-leaked";

            using var gateway = Process.Start(psi)!;
            try
            {
                WaitForHealth($"http://127.0.0.1:{gatewayPort}/health");

                using var client = new HttpClient();
                const string request =
                    "{\"model\":\"gpt-4\",\"messages\":[{\"role\":\"user\",\"content\":\"reach me at bob@example.com\"}]}";
                var resp = client.PostAsync(
                    $"http://127.0.0.1:{gatewayPort}/v1/chat/completions",
                    new StringContent(request, Encoding.UTF8, "application/json")).GetAwaiter().GetResult();

                Assert.Equal(HttpStatusCode.OK, resp.StatusCode);

                // The upstream NEVER saw the e-mail — only a placeholder.
                Assert.NotNull(upstreamReceived);
                Assert.DoesNotContain("bob@example.com", upstreamReceived);
                Assert.Contains("REDACTED_EMAIL", upstreamReceived);

                // The audit log recorded the category (counts only — no value). Read with a shared handle —
                // the gateway process still holds the file open for append.
                Assert.True(File.Exists(auditPath));
                string audit;
                using (var fs = new FileStream(auditPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
                using (var sr = new StreamReader(fs))
                {
                    audit = sr.ReadToEnd();
                }
                Assert.Contains("EMAIL", audit);
                Assert.DoesNotContain("bob@example.com", audit);
            }
            finally
            {
                try { gateway.Kill(entireProcessTree: true); } catch { }
                upstream.Stop();
                try { File.Delete(auditPath); } catch { }
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
                Thread.Sleep(150);
            }
            throw new TimeoutException($"gateway did not become healthy at {url}");
        }

        private static string? LocateOverfitExe()
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "Overfit.sln")))
            {
                dir = dir.Parent;
            }

            if (dir is null)
            {
                return null;
            }

            var cliBin = Path.Combine(dir.FullName, "Sources", "Cli", "bin");
            if (!Directory.Exists(cliBin))
            {
                return null;
            }

            var name = OperatingSystem.IsWindows() ? "overfit.exe" : "overfit";
            var matches = Directory.GetFiles(cliBin, name, SearchOption.AllDirectories);
            return matches.Length > 0 ? matches[0] : null;
        }
    }
}
