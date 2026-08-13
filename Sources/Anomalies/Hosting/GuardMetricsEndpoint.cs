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
using DevOnBike.Overfit.Runtime;
using Microsoft.Extensions.Logging;

namespace DevOnBike.Overfit.Anomalies.Hosting
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
    public sealed class GuardMetricsEndpoint : IDisposable
    {
        private readonly IClock _clock;
        private readonly HttpListener _listener = new();
        private readonly GuardTelemetry _telemetry;
        private readonly AnomalyGuard? _guard;
        private readonly ILogger _logger;
        private readonly CancellationTokenSource _stopping = new();

        /// <summary>
        /// The serving loop, held rather than discarded.
        ///
        /// <para><b>It was <c>_ = ServeAsync()</c>, and that is a silent hole rather than a style point.</b>
        /// <c>CS4014</c> is an error repo-wide, but an explicit discard satisfies it — so the one guard that
        /// exists for an unawaited task does not fire here, and a terminal failure of this loop went to a
        /// task object nobody held. An unobserved exception has not crashed the process since .NET 4.5: the
        /// endpoint would stop serving, the process would keep running, and nothing anywhere would say so.
        /// That is the guard's own observability channel failing in exactly the shape the rest of this
        /// subsystem is built to make loud.</para>
        ///
        /// <para><b>Observed, not awaited</b>, because there is nothing to await it from:
        /// <see cref="TryBind"/> is synchronous and this loop is meant to run until <see cref="Dispose"/>,
        /// so awaiting it at the call site would hang the guard at startup instead of starting it.</para>
        /// </summary>
        private Task? _serving;

        private GuardMetricsEndpoint(
            GuardTelemetry telemetry, AnomalyGuard? guard, ILogger logger, string prefix, IClock? clock)
        {
            _clock = clock ?? SystemClock.Instance;
            _telemetry = telemetry;
            _guard = guard;
            _logger = logger;

            _listener.Prefixes.Add(prefix);
        }

        /// <summary>
        /// Starts the endpoint, or returns <c>null</c> and says why.
        /// </summary>
        /// <param name="telemetry">
        /// The guard's counters. Read — never mutated — once per <c>/metrics</c> scrape and rendered as
        /// Prometheus text. Borrowed: the caller that owns the guard owns this too, and the endpoint neither
        /// disposes nor takes a copy of it.
        /// </param>
        /// <param name="logger">
        /// Where this endpoint says what an operator cannot otherwise see: that the port could not be bound,
        /// that it fell back to loopback, that a request failed for a reason which is not a transport fault,
        /// and that the serving loop has ended. All of those are invisible in the metrics themselves,
        /// because a channel that is not serving publishes nothing.
        /// </param>
        /// <param name="port">Port to listen on. Zero or below disables it entirely.</param>
        /// <param name="guard">
        /// The guard, so <c>/ack</c> and <c>/suppressions</c> can be served. Null serves metrics only, which
        /// is what a host that does not want a write endpoint gets.
        /// </param>
        /// <param name="clock">
        /// The instant stamped on <c>/suppressions</c> (which suppressions are active <i>now</i>) and passed
        /// to <c>AnomalyGuard.Acknowledge</c> as the moment a mute begins. Null takes
        /// <see cref="SystemClock.Instance"/>; a test or a replay supplies its own so those timestamps are on
        /// the same timeline as the data being judged.
        /// </param>
        public static GuardMetricsEndpoint? TryStart(
            GuardTelemetry telemetry, ILogger logger, int port, AnomalyGuard? guard = null,
            IClock? clock = null)
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
            if (TryBind(telemetry, guard, logger, $"http://+:{port}/", clock) is { } wide)
            {
                return wide;
            }

            if (TryBind(telemetry, guard, logger, $"http://127.0.0.1:{port}/", clock) is { } local)
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
            GuardTelemetry telemetry, AnomalyGuard? guard, ILogger logger, string prefix, IClock? clock)
        {
            var endpoint = new GuardMetricsEndpoint(telemetry, guard, logger, prefix, clock);

            try
            {
                endpoint._listener.Start();
            }
            catch (Exception ex) when (ex is HttpListenerException or ObjectDisposedException)
            {
                endpoint.Dispose();

                return null;
            }

            endpoint._serving = endpoint.ServeAsync();

            return endpoint;
        }

        // OVERFIT040 on Dispose: `CancellationTokenSource.Cancel` has a `CancelAsync` sibling, and this method
        // cannot use it. BOUND BY THE CONTRACT: this is `IDisposable.Dispose`, which returns void — there is
        // nothing here for a task to be awaited from, and `IAsyncDisposable` is a different lifetime decision
        // from a lint sweep's. What `CancelAsync` buys is not running registered callbacks on the caller's
        // thread; `_stopping` has NO registrations at all — its token is only ever read through
        // `_stopping.IsCancellationRequested` in ServeAsync — so Cancel here sets a flag and returns, and the
        // asynchronous form would have nothing to move off this thread.
#pragma warning disable OVERFIT040
        public void Dispose()
#pragma warning restore OVERFIT040
        {
            _stopping.Cancel();

            if (_listener.IsListening)
            {
                _listener.Stop();
            }

            _listener.Close();

            // Reading the loop's terminal state back instead of dropping it. ServeAsync now catches
            // everything, so a faulted task here means its own handler failed — rare, and still better as
            // one logged line than as nothing. Not awaited: this method is what ends that loop, so waiting
            // on it here is waiting on itself.
            if (_serving is { IsFaulted: true })
            {
                _logger.LogError(
                    _serving.Exception, "Guard metrics serving loop ended in an unhandled fault.");
            }

            _stopping.Dispose();
        }

        /// <summary>
        /// Accepts and answers scrapes until the endpoint is disposed.
        ///
        /// <para><b>Every exit is a logged one.</b> The two handlers below used to leave a gap between them:
        /// anything from <see cref="RespondAsync"/> that was not a transport fault escaped the loop, faulted
        /// the returned task and — because that task was discarded — vanished.</para>
        ///
        /// <para><b>Correction, 2026-08-13: the case this paragraph used to call reachable is NOT, and the
        /// fix stands without it.</b> It named a concrete race — <c>_guard.ActiveSuppressions</c> and
        /// <c>_guard.Acknowledge</c> reading guard state from this thread while a cycle mutates it on
        /// another, giving a <c>Collection was modified</c> <see cref="InvalidOperationException"/> that
        /// matched neither filter. All three members take the same <c>lock (_gate)</c>
        /// (<c>AnomalyGuard.RunCycle</c>, <c>Acknowledge</c>, <c>ActiveSuppressions</c>), so a scrape
        /// arriving mid-cycle blocks rather than enumerating a moving list. Dates settle it rather than
        /// reasoning: the lock arrived in <c>e9a5642</c> (2026-08-05) and this paragraph in <c>fc52686</c>
        /// (2026-08-11), six days after the hazard it describes was closed.</para>
        ///
        /// <para><b>An unexpected request failure no longer ends the channel, whatever produced it.</b> That
        /// is the property, and it needs no story about how the exception arises: an unhandled one out of a
        /// request used to kill metrics permanently for the life of the process, leaving a running process
        /// with nothing serving — which from outside is indistinguishable from a healthy guard with nothing
        /// to report. It is logged at <c>Error</c> — not <c>Debug</c>, which is where a client hanging up
        /// mid-write belongs — and the loop continues. Pinned by
        /// <c>GuardMetricsEndpointTests.AnUnexpectedFailureOnOneRequestDoesNotKillTheMetricsChannel</c>,
        /// which raises an <see cref="InvalidOperationException"/> from inside a handler and requires
        /// <c>/metrics</c> to answer afterwards.</para>
        /// </summary>
        private async Task ServeAsync()
        {
            try
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
                        await RespondAsync(context).ConfigureAwait(false);
                    }
                    catch (Exception ex) when (ex is HttpListenerException or ObjectDisposedException or IOException)
                    {
                        // A scrape that hung up mid-write is the client's business, not a reason to stop serving.
                        _logger.LogDebug(ex, "Guard metrics request failed.");
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(
                            ex,
                            "Guard metrics request failed unexpectedly and the endpoint kept serving. This is "
                            + "not a transport fault, so it is a defect in the endpoint or a concurrent read "
                            + "of guard state.");
                    }
                }
            }
            catch (Exception ex)
            {
                // The backstop. Nothing above should reach here, and if something does, the endpoint is
                // finished: the process keeps running, the listener stops being served, and a scrape then
                // hangs until it times out. Said loudly because from outside it is indistinguishable from a
                // healthy guard with nothing to report — see the class remarks on `absent()`.
                _logger.LogError(
                    ex,
                    "Guard metrics endpoint has STOPPED serving and will not recover; the guard itself "
                    + "continues. 'This guard has stopped' is no longer detectable from its own metrics.");
            }
        }

        private async Task RespondAsync(HttpListenerContext context)
        {
            var path = context.Request.Url?.AbsolutePath ?? "/";

            if (string.Equals(path, "/healthz", StringComparison.Ordinal))
            {
                await WriteAsync(context, 200, "text/plain; charset=utf-8", "ok\n").ConfigureAwait(false);

                return;
            }

            if (string.Equals(path, "/suppressions", StringComparison.Ordinal))
            {
                await SuppressionsAsync(context).ConfigureAwait(false);

                return;
            }

            if (string.Equals(path, "/ack", StringComparison.Ordinal))
            {
                await AcknowledgeAsync(context).ConfigureAwait(false);

                return;
            }

            if (!string.Equals(path, "/metrics", StringComparison.Ordinal))
            {
                await WriteAsync(context, 404, "text/plain; charset=utf-8",
                    "try /metrics, /suppressions, /ack or /healthz\n").ConfigureAwait(false);

                return;
            }

            await WriteAsync(context, 200, "text/plain; version=0.0.4; charset=utf-8",
                _telemetry.ToPrometheusText()).ConfigureAwait(false);
        }

        /// <summary>
        /// Lists what is muted right now — every subject, signal, expiry and the reason somebody gave.
        ///
        /// <para>Readable by a human on purpose. A mute nobody can enumerate is indistinguishable from a
        /// detector that stopped working, and the whole argument for letting an operator silence anything is
        /// that the silence stays visible and expires.</para>
        /// </summary>
        private async Task SuppressionsAsync(HttpListenerContext context)
        {
            if (_guard == null)
            {
                await WriteAsync(context, 501, "text/plain; charset=utf-8",
                    "this host serves metrics only; no guard was supplied to the endpoint\n").ConfigureAwait(false);

                return;
            }

            var now = _clock.UtcNow;
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

            await WriteAsync(context, 200, "text/plain; charset=utf-8", text.ToString()).ConfigureAwait(false);
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
        /// <summary>
        /// The shared secret <c>POST /ack</c> requires, read once at startup.
        ///
        /// <para>Empty means none was configured, and <see cref="IsAuthorised"/> then refuses every call —
        /// fail closed, because the endpoint's effect is to silence a finding.</para>
        /// </summary>
        private static string AckToken
        {
            get;
        } =
            Environment.GetEnvironmentVariable(OverfitEnvironment.GuardAckToken)?.Trim() ?? string.Empty;

        /// <summary>
        /// Whether the caller presented the configured bearer token.
        ///
        /// <para><b>Fail closed on an unset secret.</b> An unauthenticated write endpoint that silences
        /// incidents is worse than no endpoint: the guard keeps running, keeps looking healthy, and reports
        /// nothing. The `NetworkPolicy` written to protect this port was measured **inert** on a
        /// Docker-Desktop-class cluster — no policy-capable CNI, probe pod still reached it — so a control
        /// that assumes the customer's CNI enforces policy is not a control.</para>
        ///
        /// <para><b>Compared in constant time.</b> A short-circuiting comparison over a secret leaks its
        /// prefix to anyone who can time the response, and this endpoint is reachable by whoever can reach
        /// the scrape port.</para>
        ///
        /// <para><c>/metrics</c>, <c>/healthz</c> and <c>/suppressions</c> are deliberately NOT behind this.
        /// The first two are what Prometheus scrapes, and the third is the transparency guarantee — a mute
        /// nobody can enumerate is indistinguishable from a detector that stopped working. That leaves the
        /// mute list readable by anyone who reaches the port, which is a smaller exposure than a writable
        /// one and is stated here rather than left to be discovered.</para>
        /// </summary>
        private static bool IsAuthorised(HttpListenerContext context)
        {
            return GuardAckAuthorization.IsAuthorised(
                context.Request.Headers["Authorization"], AckToken);
        }

        private async Task AcknowledgeAsync(HttpListenerContext context)
        {
            if (_guard == null)
            {
                await WriteAsync(context, 501, "text/plain; charset=utf-8",
                    "this host serves metrics only; no guard was supplied to the endpoint\n").ConfigureAwait(false);

                return;
            }

            if (!string.Equals(context.Request.HttpMethod, "POST", StringComparison.OrdinalIgnoreCase))
            {
                await WriteAsync(context, 405, "text/plain; charset=utf-8",
                    "POST /ack?id=<n>&kind=noise|real&for=7d&reason=... — this changes what the guard "
                    + "reports, so it is not a GET\n").ConfigureAwait(false);

                return;
            }

            if (!IsAuthorised(context))
            {
                // 503 rather than 401 when no secret is configured, because the two are different problems
                // and only one is the caller's: an operator holding a valid token needs to know the guard was
                // never given one, and telling them "unauthorised" sends them hunting for their own mistake.
                var configured = AckToken.Length > 0;

                _logger.LogWarning(
                    "Refused POST /ack from {Remote}: {Reason}. This request would have suppressed a finding.",
                    context.Request.RemoteEndPoint?.Address,
                    configured ? "bad or missing bearer token" : "no ack token configured on this guard");

                await WriteAsync(
                    context,
                    configured ? 401 : 503,
                    "text/plain; charset=utf-8",
                    configured
                        ? "POST /ack requires `Authorization: Bearer <token>` matching this guard's "
                          + OverfitEnvironment.GuardAckToken + "\n"
                        : "POST /ack is disabled because " + OverfitEnvironment.GuardAckToken
                          + " is not set on this guard. It suppresses findings for a caller-chosen "
                          + "duration, so it refuses rather than serving unauthenticated writes\n").ConfigureAwait(false);

                return;
            }

            var query = context.Request.QueryString;

            if (!long.TryParse(query["id"], NumberStyles.Integer, CultureInfo.InvariantCulture, out var id))
            {
                await WriteAsync(context, 400, "text/plain; charset=utf-8",
                    "id is required and must be a number\n").ConfigureAwait(false);

                return;
            }

            var kindText = query["kind"] ?? string.Empty;
            var isReal = string.Equals(kindText, "real", StringComparison.OrdinalIgnoreCase);

            if (!isReal && !string.Equals(kindText, "noise", StringComparison.OrdinalIgnoreCase))
            {
                await WriteAsync(context, 400, "text/plain; charset=utf-8",
                    "kind must be 'noise' or 'real'. There is no default: one of them silences a signal and "
                    + "the other pins it, and guessing between those is not a thing to do quietly\n").ConfigureAwait(false);

                return;
            }

            TimeSpan? mute = null;

            if (query["for"] is { Length: > 0 } forText)
            {
                if (!TryParseDuration(forText, out var parsed))
                {
                    await WriteAsync(context, 400, "text/plain; charset=utf-8",
                        $"could not read '{forText}' as a duration; use 30m, 12h or 7d\n").ConfigureAwait(false);

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
                    _clock.UtcNow);

                _logger.LogWarning(
                    "Operator acknowledged incident {Incident} as {Kind}: {Outcome}", id, kindText, echo);

                await WriteAsync(context, 200, "text/plain; charset=utf-8", echo + "\n").ConfigureAwait(false);
            }
            catch (ArgumentException ex)
            {
                await WriteAsync(context, 404, "text/plain; charset=utf-8", ex.Message + "\n").ConfigureAwait(false);
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

            var unit = text[text.Length - 1];
            var number = text.Substring(0, text.Length - 1);

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

        /// <summary>
        /// Answers one request.
        ///
        /// <para><b>Asynchronous because this runs on a pool thread, on the serving path.</b> It used to be a
        /// synchronous <c>OutputStream.Write</c> called from <see cref="ServeAsync"/>'s loop (OVERFIT040): a
        /// scrape that connected and then stopped reading held that thread until the socket timed out, and the
        /// endpoint serves one request at a time, so the thread held is the whole channel. The body is small —
        /// a few hundred bytes of Prometheus text — which bounds the cost but does not remove it, because the
        /// duration is set by the client's read rate rather than by the payload.</para>
        /// </summary>
        private static async Task WriteAsync(
            HttpListenerContext context, int status, string contentType, string body)
        {
            var bytes = Encoding.UTF8.GetBytes(body);

            context.Response.StatusCode = status;
            context.Response.ContentType = contentType;
            context.Response.ContentLength64 = bytes.Length;
            await context.Response.OutputStream.WriteAsync(bytes).ConfigureAwait(false);
            context.Response.Close();
        }
    }
}
