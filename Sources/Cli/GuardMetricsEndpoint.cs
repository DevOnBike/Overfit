// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Net;
using System.Text;
using DevOnBike.Overfit.Anomalies.Monitoring;
using Microsoft.Extensions.Logging;

namespace DevOnBike.Overfit.Cli
{
    /// <summary>
    /// Serves the guard's own metrics so Prometheus can scrape the guard.
    ///
    /// <para><b>Without this the telemetry is a property nobody reads.</b> The counters existed, and the one
    /// consumer that mattered — a scrape, and therefore an alert on
    /// <c>overfit_guard_last_cycle_timestamp_seconds</c> going stale — had no way to reach them. A guard that
    /// has stopped is worse than one that never started, because somebody is relying on it, and until it is
    /// scrapeable that condition is undetectable from outside.</para>
    ///
    /// <para><b>A bare <see cref="HttpListener"/> rather than a web framework.</b> This process exists to
    /// watch a cluster; giving it a dependency injection container, routing and middleware to serve two
    /// hundred bytes of text would be a larger attack surface and a slower start for no gain. It answers
    /// <c>/metrics</c> and <c>/healthz</c> and refuses everything else.</para>
    ///
    /// <para><b>A failure to serve must never stop the guard.</b> Monitoring one's own monitoring is useful;
    /// dying because the port was taken is not, and would replace the problem the guard was deployed to detect
    /// with one of its own.</para>
    /// </summary>
    internal sealed class GuardMetricsEndpoint : IDisposable
    {
        private readonly HttpListener _listener = new();
        private readonly GuardTelemetry _telemetry;
        private readonly ILogger _logger;
        private readonly CancellationTokenSource _stopping = new();

        private GuardMetricsEndpoint(GuardTelemetry telemetry, ILogger logger, string prefix)
        {
            _telemetry = telemetry;
            _logger = logger;

            _listener.Prefixes.Add(prefix);
        }

        /// <summary>
        /// Starts the endpoint, or returns <c>null</c> and says why.
        /// </summary>
        /// <param name="port">Port to listen on. Zero or below disables it entirely.</param>
        public static GuardMetricsEndpoint? TryStart(GuardTelemetry telemetry, ILogger logger, int port)
        {
            ArgumentNullException.ThrowIfNull(telemetry);
            ArgumentNullException.ThrowIfNull(logger);

            if (port <= 0)
            {
                return null;
            }

            // All interfaces first, because in a pod that is the only binding a scrape can reach. It needs a
            // URL reservation on Windows and none on Linux, so a developer box falls back to loopback rather
            // than losing the endpoint entirely — and is told, because "metrics work on my machine and not in
            // the cluster" is the wrong lesson to learn later.
            if (TryBind(telemetry, logger, $"http://+:{port}/") is { } wide)
            {
                return wide;
            }

            if (TryBind(telemetry, logger, $"http://127.0.0.1:{port}/") is { } local)
            {
                logger.LogWarning(
                    "Guard metrics are bound to LOOPBACK only on port {Port} — binding all interfaces needs a "
                    + "URL reservation (Windows) or root (low ports). Inside a container this would be "
                    + "unreachable from Prometheus.", port);

                return local;
            }

            // Reported loudly, because an operator who asked for metrics and silently did not get them will
            // later write an alert against a series that never arrives — and a missing series reads as a
            // healthy one in most alerting rules.
            logger.LogError(
                "Could not serve guard metrics on port {Port}. The guard continues WITHOUT them, so 'this "
                + "guard has stopped' will not be detectable from outside.", port);

            return null;
        }

        private static GuardMetricsEndpoint? TryBind(GuardTelemetry telemetry, ILogger logger, string prefix)
        {
            var endpoint = new GuardMetricsEndpoint(telemetry, logger, prefix);

            try
            {
                endpoint._listener.Start();
            }
            catch (Exception ex) when (ex is HttpListenerException or ObjectDisposedException)
            {
                endpoint.Dispose();

                return null;
            }

            _ = endpoint.ServeAsync();

            return endpoint;
        }

        public void Dispose()
        {
            _stopping.Cancel();

            if (_listener.IsListening)
            {
                _listener.Stop();
            }

            _listener.Close();
            _stopping.Dispose();
        }

        private async Task ServeAsync()
        {
            // #pragma BOUND: exits when Dispose cancels the token or stops the listener, which makes
            // GetContextAsync throw — there is no other path out and no way to spin.
            while (!_stopping.IsCancellationRequested)
            {
                HttpListenerContext context;

                try
                {
                    context = await _listener.GetContextAsync().ConfigureAwait(false);
                }
                catch (Exception ex) when (ex is HttpListenerException or ObjectDisposedException)
                {
                    return;
                }

                try
                {
                    Respond(context);
                }
                catch (Exception ex) when (ex is HttpListenerException or ObjectDisposedException or IOException)
                {
                    // A scrape that hung up mid-write is the client's business, not a reason to stop serving.
                    _logger.LogDebug(ex, "Guard metrics request failed.");
                }
            }
        }

        private void Respond(HttpListenerContext context)
        {
            var path = context.Request.Url?.AbsolutePath ?? "/";

            if (string.Equals(path, "/healthz", StringComparison.Ordinal))
            {
                Write(context, 200, "text/plain; charset=utf-8", "ok\n");

                return;
            }

            if (!string.Equals(path, "/metrics", StringComparison.Ordinal))
            {
                Write(context, 404, "text/plain; charset=utf-8", "try /metrics or /healthz\n");

                return;
            }

            Write(context, 200, "text/plain; version=0.0.4; charset=utf-8", _telemetry.ToPrometheusText());
        }

        private static void Write(HttpListenerContext context, int status, string contentType, string body)
        {
            var bytes = Encoding.UTF8.GetBytes(body);

            context.Response.StatusCode = status;
            context.Response.ContentType = contentType;
            context.Response.ContentLength64 = bytes.Length;
            context.Response.OutputStream.Write(bytes, 0, bytes.Length);
            context.Response.Close();
        }
    }
}
