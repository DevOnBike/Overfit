// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Net;
using System.Text;
using System.Text.Json;
using DevOnBike.Overfit.Redaction;
using DevOnBike.Overfit.Server.OpenAi;

namespace DevOnBike.Overfit.Server
{
    /// <summary>
    /// LLM egress firewall — an OpenAI-compatible reverse proxy that redacts outbound PII/secrets from each request
    /// before it leaves the box, forwards it to an upstream LLM (injecting the gateway-held upstream key so clients
    /// never see it), restores the placeholders on the way back, and audits every redaction. "Change one base URL."
    ///
    /// <para>Phase 2a walking skeleton: <c>/v1/chat/completions</c>, NON-streaming, restore-on-response,
    /// JSON-lines audit. Streaming SSE, response-side scanning, and the BLOCK policy are follow-ons.</para>
    /// </summary>
    public static class RedactionGateway
    {
        /// <summary>
        /// Binds an HTTP listener on <paramref name="host"/>:<paramref name="port"/> and proxies chat completions
        /// to <paramref name="upstreamBaseUrl"/> (e.g. <c>https://api.openai.com/v1</c>), redacting via
        /// <paramref name="redactor"/> and auditing via <paramref name="audit"/>. <paramref name="upstreamApiKey"/>
        /// is the gateway-held secret injected as the upstream <c>Authorization</c> — clients authenticate to the
        /// gateway, not the upstream. Blocks until the process is stopped.
        /// </summary>
        public static void Serve(
            string host,
            int port,
            string upstreamBaseUrl,
            string? upstreamApiKey,
            Redactor redactor,
            IRedactionAuditSink audit)
        {
            ArgumentException.ThrowIfNullOrEmpty(upstreamBaseUrl);
            ArgumentNullException.ThrowIfNull(redactor);
            ArgumentNullException.ThrowIfNull(audit);

            var upstream = upstreamBaseUrl.TrimEnd('/');

            using var http = new HttpClient
            {
                Timeout = TimeSpan.FromSeconds(120)
            };

            using var listener = new HttpListener();
            listener.Prefixes.Add($"http://{host}:{port}/");
            listener.Start();

            Console.WriteLine($"Redaction gateway listening on http://{host}:{port}");
            Console.WriteLine($"  → forwarding to {upstream}   (outbound PII/secrets redaction, audit on)");
            Console.WriteLine($"  point your OpenAI client's base_url here; the real upstream key never leaves the gateway.");

            while (true)
            {
                HttpListenerContext ctx;
                try
                {
                    ctx = listener.GetContext();
                }
                catch (HttpListenerException)
                {
                    break;
                }

                HandleRequest(ctx, upstream, upstreamApiKey, redactor, audit, http);
            }
        }

        /// <summary>
        /// Redacts every message's content in <paramref name="req"/> in place, returning the removed spans (for
        /// restoring the response) and per-category counts (for the audit). Pure — the testable core of the proxy.
        /// </summary>
        public static (List<RedactionMatch> Matches, Dictionary<string, int> Counts) RedactRequest(
            ChatCompletionRequest req,
            Redactor redactor)
        {
            ArgumentNullException.ThrowIfNull(req);
            ArgumentNullException.ThrowIfNull(redactor);

            var matches = new List<RedactionMatch>();
            var counts = new Dictionary<string, int>(StringComparer.Ordinal);

            foreach (var message in req.Messages)
            {
                if (string.IsNullOrEmpty(message.Content))
                {
                    continue;
                }

                var result = redactor.Redact(message.Content);
                if (!result.HasRedactions)
                {
                    continue;
                }

                message.Content = result.Text;
                foreach (var match in result.Matches)
                {
                    matches.Add(match);
                    counts[match.Category] = counts.GetValueOrDefault(match.Category) + 1;
                }
            }

            return (matches, counts);
        }

        /// <summary>Restores the original values for any placeholder a response echoed back (default policy).</summary>
        public static void RestoreResponse(ChatCompletionResponse response, IReadOnlyList<RedactionMatch> matches)
        {
            ArgumentNullException.ThrowIfNull(response);
            ArgumentNullException.ThrowIfNull(matches);

            if (matches.Count == 0)
            {
                return;
            }

            foreach (var choice in response.Choices)
            {
                if (choice.Message?.Content is { } content)
                {
                    choice.Message.Content = Redactor.Restore(content, matches);
                }
            }
        }

        private static void HandleRequest(
            HttpListenerContext ctx,
            string upstream,
            string? upstreamApiKey,
            Redactor redactor,
            IRedactionAuditSink audit,
            HttpClient http)
        {
            try
            {
                var path = ctx.Request.Url?.AbsolutePath ?? string.Empty;
                var method = ctx.Request.HttpMethod;

                if (method == "GET" && path == "/health")
                {
                    WriteText(ctx.Response, HttpStatusCode.OK, "ok");
                    return;
                }

                if (method == "POST" && path == "/v1/chat/completions")
                {
                    HandleChatCompletions(ctx, upstream, upstreamApiKey, redactor, audit, http);
                    return;
                }

                WriteText(ctx.Response, HttpStatusCode.NotFound, "not found");
            }
            catch (Exception ex)
            {
                try
                {
                    WriteText(ctx.Response, HttpStatusCode.BadGateway, $"gateway error: {ex.Message}");
                }
                catch
                {
                    // client gone — nothing to do.
                }
            }
        }

        private static void HandleChatCompletions(
            HttpListenerContext ctx,
            string upstream,
            string? upstreamApiKey,
            Redactor redactor,
            IRedactionAuditSink audit,
            HttpClient http)
        {
            var req = JsonSerializer.Deserialize(ctx.Request.InputStream, OpenAiJsonContext.Default.ChatCompletionRequest);
            if (req is null)
            {
                WriteText(ctx.Response, HttpStatusCode.BadRequest, "invalid request body");
                return;
            }

            // ── Redact every message's content; keep the matches for restore + audit. ──
            var (matches, counts) = RedactRequest(req, redactor);

            // Phase 2a: non-streaming (redacting a token stream across chunk boundaries is a follow-on).
            req.Stream = false;

            // ── Forward to the upstream, injecting the gateway-held key. ──
            var requestJson = JsonSerializer.Serialize(req, OpenAiJsonContext.Default.ChatCompletionRequest);

            using var upstreamRequest = new HttpRequestMessage(HttpMethod.Post, $"{upstream}/chat/completions")
            {
                Content = new StringContent(requestJson, Encoding.UTF8, "application/json")
            };
            if (!string.IsNullOrEmpty(upstreamApiKey))
            {
                upstreamRequest.Headers.TryAddWithoutValidation("Authorization", $"Bearer {upstreamApiKey}");
            }

            using var upstreamResponse = http.Send(upstreamRequest);
            var responseJson = upstreamResponse.Content.ReadAsStringAsync().GetAwaiter().GetResult();

            // ── Restore placeholders on the way back (default policy). ──
            if (matches.Count > 0 && upstreamResponse.IsSuccessStatusCode)
            {
                var response = JsonSerializer.Deserialize(responseJson, OpenAiJsonContext.Default.ChatCompletionResponse);
                if (response is not null)
                {
                    RestoreResponse(response, matches);
                    responseJson = JsonSerializer.Serialize(response, OpenAiJsonContext.Default.ChatCompletionResponse);
                }
            }

            // ── Audit (counts only — never the values). ──
            if (matches.Count > 0)
            {
                audit.Record(new RedactionAuditRecord(
                    Guid.NewGuid().ToString("N"),
                    DateTimeOffset.UtcNow,
                    matches.Count,
                    counts));
            }

            WriteRaw(ctx.Response, (int)upstreamResponse.StatusCode, responseJson);
        }

        private static void WriteRaw(HttpListenerResponse response, int status, string json)
        {
            var bytes = Encoding.UTF8.GetBytes(json);
            response.StatusCode = status;
            response.ContentType = "application/json";
            response.ContentLength64 = bytes.Length;
            response.OutputStream.Write(bytes, 0, bytes.Length);
            response.OutputStream.Close();
        }

        private static void WriteText(HttpListenerResponse response, HttpStatusCode status, string text)
        {
            var bytes = Encoding.UTF8.GetBytes(text);
            response.StatusCode = (int)status;
            response.ContentType = "text/plain";
            response.ContentLength64 = bytes.Length;
            response.OutputStream.Write(bytes, 0, bytes.Length);
            response.OutputStream.Close();
        }
    }
}
