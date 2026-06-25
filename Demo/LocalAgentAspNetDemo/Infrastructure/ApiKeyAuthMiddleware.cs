// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Security.Cryptography;
using System.Text;

namespace DevOnBike.Overfit.Demo.LocalAgent.Infrastructure
{
    /// <summary>
    /// API-key authentication for the agent endpoints — the first production gate. When an <c>ApiKey</c> is
    /// configured (appsettings or the <c>ApiKey</c> env var), every request must present it as
    /// <c>X-API-Key: &lt;key&gt;</c> or <c>Authorization: Bearer &lt;key&gt;</c>; otherwise the call is refused with
    /// 401 <i>before</i> it reaches the model. The key is never stored in plaintext — only its SHA-256 hash is kept
    /// and compared in constant time, so neither the key nor its length leaks through timing. Probe and docs paths
    /// (<c>/health</c>, <c>/healthz</c>, <c>/readyz</c>, <c>/metrics</c>, <c>/swagger</c>, <c>/openapi</c>, <c>/</c>)
    /// stay open so liveness checks and the Swagger UI work without a key. With no key configured, auth is OFF (a
    /// startup warning is logged) — so the demo runs out of the box, but a deployment must set a key.
    /// </summary>
    public sealed class ApiKeyAuthMiddleware
    {
        private static readonly string[] OpenPrefixes =
            ["/health", "/healthz", "/readyz", "/metrics", "/swagger", "/openapi"];

        private readonly RequestDelegate _next;
        private readonly byte[]? _keyHash;

        public ApiKeyAuthMiddleware(RequestDelegate next, IConfiguration configuration, ILogger<ApiKeyAuthMiddleware> logger)
        {
            _next = next;
            var key = configuration["ApiKey"];
            if (string.IsNullOrWhiteSpace(key))
            {
                _keyHash = null;
                logger.LogWarning(
                    "API-key auth is OFF — any caller can reach the agent. Set 'ApiKey' (config or env) before exposing it.");
            }
            else
            {
                _keyHash = SHA256.HashData(Encoding.UTF8.GetBytes(key.Trim()));
                logger.LogInformation("API-key auth is ON — callers must present 'X-API-Key' or 'Authorization: Bearer'.");
            }
        }

        public async Task InvokeAsync(HttpContext context)
        {
            if (_keyHash is null || IsOpenPath(context.Request.Path.Value))
            {
                await _next(context);
                return;
            }

            if (!IsAuthorized(context.Request))
            {
                context.Response.StatusCode = StatusCodes.Status401Unauthorized;
                await context.Response.WriteAsJsonAsync(new
                {
                    error = "unauthorized",
                    detail = "Provide a valid API key as 'X-API-Key: <key>' or 'Authorization: Bearer <key>'.",
                });
                return;
            }

            await _next(context);
        }

        private static bool IsOpenPath(string? path)
        {
            if (string.IsNullOrEmpty(path) || path == "/")
            {
                return true;
            }

            foreach (var prefix in OpenPrefixes)
            {
                if (path.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }

            return false;
        }

        private bool IsAuthorized(HttpRequest request)
        {
            var presented = ExtractKey(request);
            if (string.IsNullOrEmpty(presented))
            {
                return false;
            }

            var presentedHash = SHA256.HashData(Encoding.UTF8.GetBytes(presented));
            return CryptographicOperations.FixedTimeEquals(presentedHash, _keyHash);
        }

        private static string? ExtractKey(HttpRequest request)
        {
            if (request.Headers.TryGetValue("X-API-Key", out var apiKey) && !string.IsNullOrWhiteSpace(apiKey))
            {
                return apiKey.ToString().Trim();
            }

            var auth = request.Headers.Authorization.ToString();
            if (auth.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
            {
                return auth["Bearer ".Length..].Trim();
            }

            return null;
        }
    }
}
