// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Net;
using System.Text;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Monitoring;
using Microsoft.Extensions.Logging;

namespace DevOnBike.Overfit.Cli
{
    /// <summary>
    /// Serves the guard's own metrics so Prometheus can scrape the guard.
    ///
    /// <para><b>Without this the telemetry is a property nobody reads.</b> The counters existed, and the one
    /// consumer that mattered — a scrape, and therefore an alert on
    /// <c>overfit_guard_last_cycle_timestamp_seconds</c> — had no way to reach them. A guard that has stopped
    /// is worse than one that never started, because somebody is relying on it, and until it is scrapeable
    /// that condition is undetectable from outside.</para>
    ///
    /// <para><b>Scrapeable is necessary and not sufficient.</b> The rule written against this series must use
    /// <c>absent()</c> as well as a staleness comparison: when the pod goes, the series goes with it, and a
    /// <c>time()</c> comparison alone then evaluates over an empty vector and reports healthy. Measured
    /// 2026-08-05 — six minutes of <c>inactive</c> with the guard scaled to zero. The working rule is in
    /// <c>k8s/lab/guard-alerts.yaml</c>.</para>
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
        private readonly AnomalyGuard? _guard;
        private readonly ILogger _logger;
        private readonly CancellationTokenSource _stopping = new();

        private GuardMetricsEndpoint(
            GuardTelemetry telemetry, AnomalyGuard? guard, ILogger logger, string prefix)
        {
            _telemetry = telemetry;
            _guard = guard;
            _logger = logger;

            _listener.Prefixes.Add(prefix);
        }

        /// <summary>
        /// Starts the endpoint, or returns <c>null</c> and says why.
        /// </summary>
        /// <param name="port">Port to listen on. Zero or below disables it entirely.</param>
        /// <param name="guard">
        /// The guard, so <c>/ack</c> and <c>/suppressions</c> can be served. Null serves metrics only, which
        /// is what a host that does not want a write endpoint gets.
        /// </param>
        public static GuardMetricsEndpoint? TryStart(
            GuardTelemetry telemetry, ILogger logger, int port, AnomalyGuard? guard = null)
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
            if (TryBind(telemetry, guard, logger, $"http://+:{port}/") is { } wide)
            {
                return wide;
            }

            if (TryBind(telemetry, guard, logger, $"http://127.0.0.1:{port}/") is { } local)
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

        private static GuardMetricsEndpoint? TryBind(
            GuardTelemetry telemetry, AnomalyGuard? guard, ILogger logger, string prefix)
        {
            var endpoint = new GuardMetricsEndpoint(telemetry, guard, logger, prefix);

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

            if (string.Equals(path, "/suppressions", StringComparison.Ordinal))
            {
                Suppressions(context);

                return;
            }

            if (string.Equals(path, "/ack", StringComparison.Ordinal))
            {
                Acknowledge(context);

                return;
            }

            if (!string.Equals(path, "/metrics", StringComparison.Ordinal))
            {
                Write(context, 404, "text/plain; charset=utf-8",
                    "try /metrics, /suppressions, /ack or /healthz\n");

                return;
            }

            Write(context, 200, "text/plain; version=0.0.4; charset=utf-8", _telemetry.ToPrometheusText());
        }

        /// <summary>
        /// Lists what is muted right now — every subject, signal, expiry and the reason somebody gave.
        ///
        /// <para>Readable by a human on purpose. A mute nobody can enumerate is indistinguishable from a
        /// detector that stopped working, and the whole argument for letting an operator silence anything is
        /// that the silence stays visible and expires.</para>
        /// </summary>
        private void Suppressions(HttpListenerContext context)
        {
            if (_guard is null)
            {
                Write(context, 501, "text/plain; charset=utf-8",
                    "this host serves metrics only; no guard was supplied to the endpoint\n");

                return;
            }

            var now = DateTimeOffset.UtcNow;
            var active = _guard.ActiveSuppressions(now);
            var text = new StringBuilder();

            text.Append(active.Length).Append(" active suppression(s) at ")
                .Append(now.ToString("u", CultureInfo.InvariantCulture)).Append('\n');

            for (var i = 0; i < active.Length; i++)
            {
                var s = active[i];
                var who = s.Pod.Length > 0 ? s.Pod : s.Workload;

                text.Append("  incident ").Append(s.IncidentId).Append("  ").Append(s.Signal)
                    .Append(" on ").Append(who)
                    .Append("  until ").Append(s.Until.ToString("u", CultureInfo.InvariantCulture))
                    .Append("  \"").Append(s.Reason).Append("\"\n");
            }

            Write(context, 200, "text/plain; charset=utf-8", text.ToString());
        }

        /// <summary>
        /// Records an operator's judgement about an incident.
        ///
        /// <para><b>POST, and it is not pedantry.</b> This changes what the guard will report; a GET that
        /// mutates would be replayed by every crawler, proxy and browser prefetch that ever saw the URL, and
        /// the change it makes is <i>silence</i>.</para>
        ///
        /// <para>Query parameters: <c>id</c> (required), <c>kind</c> = <c>noise</c>|<c>real</c> (required),
        /// <c>for</c> = a duration like <c>7d</c>, <c>90m</c> (noise only), and <c>reason</c>.</para>
        /// </summary>
        private void Acknowledge(HttpListenerContext context)
        {
            if (_guard is null)
            {
                Write(context, 501, "text/plain; charset=utf-8",
                    "this host serves metrics only; no guard was supplied to the endpoint\n");

                return;
            }

            if (!string.Equals(context.Request.HttpMethod, "POST", StringComparison.OrdinalIgnoreCase))
            {
                Write(context, 405, "text/plain; charset=utf-8",
                    "POST /ack?id=<n>&kind=noise|real&for=7d&reason=... — this changes what the guard "
                    + "reports, so it is not a GET\n");

                return;
            }

            var query = context.Request.QueryString;

            if (!long.TryParse(query["id"], NumberStyles.Integer, CultureInfo.InvariantCulture, out var id))
            {
                Write(context, 400, "text/plain; charset=utf-8", "id is required and must be a number\n");

                return;
            }

            var kindText = query["kind"] ?? string.Empty;
            var isReal = string.Equals(kindText, "real", StringComparison.OrdinalIgnoreCase);

            if (!isReal && !string.Equals(kindText, "noise", StringComparison.OrdinalIgnoreCase))
            {
                Write(context, 400, "text/plain; charset=utf-8",
                    "kind must be 'noise' or 'real'. There is no default: one of them silences a signal and "
                    + "the other pins it, and guessing between those is not a thing to do quietly\n");

                return;
            }

            TimeSpan? mute = null;

            if (query["for"] is { Length: > 0 } forText)
            {
                if (!TryParseDuration(forText, out var parsed))
                {
                    Write(context, 400, "text/plain; charset=utf-8",
                        $"could not read '{forText}' as a duration; use 30m, 12h or 7d\n");

                    return;
                }

                mute = parsed;
            }

            try
            {
                var echo = _guard.Acknowledge(
                    id,
                    isReal ? OperatorLabelKind.Real : OperatorLabelKind.Noise,
                    mute,
                    query["reason"] ?? string.Empty,
                    DateTimeOffset.UtcNow);

                _logger.LogWarning(
                    "Operator acknowledged incident {Incident} as {Kind}: {Outcome}", id, kindText, echo);

                Write(context, 200, "text/plain; charset=utf-8", echo + "\n");
            }
            catch (ArgumentException ex)
            {
                Write(context, 404, "text/plain; charset=utf-8", ex.Message + "\n");
            }
        }

        /// <summary>Reads <c>30m</c>, <c>12h</c>, <c>7d</c>. Deliberately tiny — a wrong duration is refused.</summary>
        private static bool TryParseDuration(string text, out TimeSpan value)
        {
            value = default;

            if (text.Length < 2)
            {
                return false;
            }

            var unit = text[^1];
            var number = text[..^1];

            if (!double.TryParse(number, NumberStyles.Float, CultureInfo.InvariantCulture, out var amount)
                || amount <= 0.0)
            {
                return false;
            }

            value = unit switch
            {
                'm' or 'M' => TimeSpan.FromMinutes(amount),
                'h' or 'H' => TimeSpan.FromHours(amount),
                'd' or 'D' => TimeSpan.FromDays(amount),
                _ => TimeSpan.Zero,
            };

            return value > TimeSpan.Zero;
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
