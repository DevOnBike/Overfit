// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using DevOnBike.Overfit.Demo.LocalAgent.Observability;

namespace DevOnBike.Overfit.Demo.LocalAgent.Infrastructure
{
    /// <summary>
    /// Writes one audit record per handled request: timestamp, request id, method/path, status, latency, the caller's
    /// pseudonymous key fingerprint (SHA-256, never the key), the loaded model's fingerprint, and — when an endpoint
    /// stashed them in <see cref="HttpContext.Items"/> — the RAG sources retrieved (<c>audit.sources</c>) and the tool
    /// invoked (<c>audit.tool</c>). Runs OUTERMOST (before auth) so 401s are audited too. Probe/docs paths are skipped
    /// to keep the trail signal-rich. Metadata only — no prompt or response content (see <see cref="AuditLog"/>).
    /// </summary>
    public sealed class AuditMiddleware
    {
        private static readonly string[] SkipPrefixes =
            ["/health", "/healthz", "/readyz", "/metrics", "/swagger", "/openapi"];

        private readonly RequestDelegate _next;

        public AuditMiddleware(RequestDelegate next) => _next = next;

        public async Task InvokeAsync(HttpContext context, AuditLog audit, MetricsCollector metrics)
        {
            var path = context.Request.Path.Value ?? "/";
            if (IsSkipped(path))
            {
                await _next(context);
                return;
            }

            var stopwatch = Stopwatch.StartNew();
            try
            {
                await _next(context);
            }
            finally
            {
                stopwatch.Stop();
                audit.Record(new
                {
                    ts = DateTimeOffset.UtcNow,
                    id = context.TraceIdentifier,
                    method = context.Request.Method,
                    path,
                    status = context.Response.StatusCode,
                    ms = Math.Round(stopwatch.Elapsed.TotalMilliseconds, 1),
                    client = ClientFingerprint(context.Request),
                    model = metrics.ModelFingerprint,
                    sources = context.Items.TryGetValue("audit.sources", out var s) ? s : null,
                    tool = context.Items.TryGetValue("audit.tool", out var t) ? t : null,
                });
            }
        }

        private static bool IsSkipped(string path)
        {
            if (path == "/")
            {
                return true;
            }

            foreach (var prefix in SkipPrefixes)
            {
                if (path.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }

            return false;
        }

        // Pseudonymous caller id: the first 8 hex of SHA-256(key) — stable per key, never the key itself.
        private static string ClientFingerprint(HttpRequest request)
        {
            var key = request.Headers["X-API-Key"].ToString();
            if (string.IsNullOrEmpty(key))
            {
                var auth = request.Headers.Authorization.ToString();
                if (auth.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
                {
                    key = auth["Bearer ".Length..];
                }
            }

            if (string.IsNullOrWhiteSpace(key))
            {
                return "anonymous";
            }

            return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(key.Trim())))[..8].ToLowerInvariant();
        }
    }
}
