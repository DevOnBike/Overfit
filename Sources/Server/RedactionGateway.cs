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
    /// <para><c>/v1/chat/completions</c>, streaming (SSE) and non-streaming, restore-on-response, BLOCK policy,
    /// JSON-lines audit. Response-side scanning and client authentication are follow-ons.</para>
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
            IRedactionAuditSink audit,
            RedactionPolicy policy,
            IReadOnlyCollection<string>? clientKeys = null,
            bool scanResponses = false)
        {
            ArgumentException.ThrowIfNullOrEmpty(upstreamBaseUrl);
            ArgumentNullException.ThrowIfNull(redactor);
            ArgumentNullException.ThrowIfNull(audit);
            ArgumentNullException.ThrowIfNull(policy);

            var upstream = upstreamBaseUrl.TrimEnd('/');
            var auth = new GatewayClientAuth(clientKeys);

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
            Console.WriteLine(auth.Enabled
                ? "  client authentication: ON (callers must present a configured gateway key)."
                : "  client authentication: OFF — any caller can reach this gateway. Set gateway keys before exposing it.");
            if (scanResponses)
            {
                Console.WriteLine("  response scanning: ON (model-generated secrets/PII masked on non-streaming responses).");
            }

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

                // Dispatch each request to the thread pool so a slow (or streaming) call never blocks the next caller.
                var captured = ctx;
                ThreadPool.QueueUserWorkItem(
                    _ => HandleRequest(captured, upstream, upstreamApiKey, redactor, audit, policy, auth, scanResponses, http));
            }
        }

        /// <summary>
        /// Redacts every message's content in <paramref name="req"/> in place, returning the removed spans (for
        /// restoring the response) and per-category counts (for the audit). Pure — the testable core of the proxy.
        /// </summary>
        public static (List<RedactionMatch> Matches, Dictionary<string, int> Counts, bool Blocked, List<string> BlockedCategories) RedactRequest(
            ChatCompletionRequest req,
            Redactor redactor,
            RedactionPolicy policy)
        {
            ArgumentNullException.ThrowIfNull(req);
            ArgumentNullException.ThrowIfNull(redactor);
            ArgumentNullException.ThrowIfNull(policy);

            var matches = new List<RedactionMatch>();
            var counts = new Dictionary<string, int>(StringComparer.Ordinal);
            var blocked = false;
            var blockedCategories = new List<string>();

            foreach (var message in req.Messages)
            {
                if (string.IsNullOrEmpty(message.Content))
                {
                    continue;
                }

                var decision = redactor.Redact(message.Content, policy);

                if (decision.Blocked)
                {
                    blocked = true;
                    foreach (var category in decision.BlockedCategories)
                    {
                        if (!blockedCategories.Contains(category))
                        {
                            blockedCategories.Add(category);
                        }
                    }
                }

                if (decision.RedactedMatches.Count == 0)
                {
                    continue;
                }

                message.Content = decision.Text;
                foreach (var match in decision.RedactedMatches)
                {
                    matches.Add(match);
                    counts[match.Category] = counts.GetValueOrDefault(match.Category) + 1;
                }
            }

            return (matches, counts, blocked, blockedCategories);
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
            RedactionPolicy policy,
            GatewayClientAuth auth,
            bool scanResponses,
            HttpClient http)
        {
            try
            {
                var path = ctx.Request.Url?.AbsolutePath ?? string.Empty;
                var method = ctx.Request.HttpMethod;

                // /health is unauthenticated so liveness probes work without a key.
                if (method == "GET" && path == "/health")
                {
                    WriteText(ctx.Response, HttpStatusCode.OK, "ok");
                    return;
                }

                // Everything that proxies upstream requires a valid gateway client key (when auth is enabled).
                if (!auth.IsAuthorized(ctx.Request.Headers["Authorization"]))
                {
                    WriteText(ctx.Response, HttpStatusCode.Unauthorized,
                        "Unauthorized: present a valid gateway key as 'Authorization: Bearer <key>'. "
                        + "The gateway holds the real upstream key — clients authenticate to the gateway, not upstream.");
                    return;
                }

                // Chat completions get the structured path (per-message redaction + streaming SSE restore).
                if (method == "POST" && path == "/v1/chat/completions")
                {
                    HandleChatCompletions(ctx, upstream, upstreamApiKey, redactor, audit, policy, scanResponses, http);
                    return;
                }

                // Everything else (embeddings, legacy completions, models, future endpoints) goes through the
                // generic redacting proxy: JSON bodies are scanned + redacted (and restored on the response),
                // GETs and non-JSON bodies pass straight through. Keeps the firewall transparent without 404s.
                HandleGenericProxy(ctx, upstream, upstreamApiKey, redactor, audit, policy, scanResponses, http);
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
            RedactionPolicy policy,
            bool scanResponses,
            HttpClient http)
        {
            var req = JsonSerializer.Deserialize(ctx.Request.InputStream, OpenAiJsonContext.Default.ChatCompletionRequest);
            if (req is null)
            {
                WriteText(ctx.Response, HttpStatusCode.BadRequest, "invalid request body");
                return;
            }

            // ── Detect + apply the policy across every message. ──
            var (matches, counts, blocked, blockedCategories) = RedactRequest(req, redactor, policy);

            // ── BLOCK policy: a forbidden category must never leave the box — refuse, don't forward. ──
            if (blocked)
            {
                RespondBlocked(ctx, audit, blockedCategories);
                return;
            }

            var streaming = req.Stream == true;

            // ── Forward to the upstream, injecting the gateway-held key. ──
            var requestJson = JsonSerializer.Serialize(req, OpenAiJsonContext.Default.ChatCompletionRequest);

            using var upstreamRequest = new HttpRequestMessage(HttpMethod.Post, $"{upstream}/chat/completions")
            {
                Content = new StringContent(requestJson, Encoding.UTF8, "application/json")
            };
            ForwardRequestHeaders(ctx.Request, upstreamRequest);
            if (!string.IsNullOrEmpty(upstreamApiKey))
            {
                upstreamRequest.Headers.TryAddWithoutValidation("Authorization", $"Bearer {upstreamApiKey}");
            }

            // ── Audit (counts only — never the values). Recorded once per request, before the body streams back. ──
            if (matches.Count > 0)
            {
                audit.Record(new RedactionAuditRecord(
                    Guid.NewGuid().ToString("N"),
                    DateTimeOffset.UtcNow,
                    matches.Count,
                    counts));
            }

            if (streaming)
            {
                StreamResponse(ctx, upstreamRequest, matches, redactor, policy, audit, scanResponses, http);
                return;
            }

            using var upstreamResponse = http.Send(upstreamRequest);
            var responseJson = upstreamResponse.Content.ReadAsStringAsync().GetAwaiter().GetResult();

            // ── Response-side scan: mask any secret/PII the MODEL produced (run before restore, while the caller's
            //    own values are still placeholders, so only genuinely model-generated content is caught). ──
            if (scanResponses && upstreamResponse.IsSuccessStatusCode)
            {
                responseJson = ScanResponseBody(responseJson, redactor, policy, audit);
            }

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

            ForwardResponseHeaders(upstreamResponse, ctx.Response);
            WriteRaw(ctx.Response, (int)upstreamResponse.StatusCode, responseJson);
        }

        // Scans a raw response body for model-generated sensitive content, masks Redact/Block-category spans, and
        // audits what it masked (counts only). Returns the (possibly) masked body. Placeholders the request inserted
        // are inert here (they are not sensitive patterns) and are restored separately afterwards.
        private static string ScanResponseBody(string body, Redactor redactor, RedactionPolicy policy, IRedactionAuditSink audit)
        {
            var scan = redactor.ScanResponse(body, policy);
            if (scan.Matches.Count == 0)
            {
                return body;
            }

            AuditRedactions(audit, scan.Matches);
            return scan.Text;
        }

        /// <summary>
        /// Streams the upstream's Server-Sent-Events response back to the client. Each <c>data:</c> chunk's
        /// <c>delta.content</c> is (optionally) scanned for model-generated secrets via a per-choice
        /// <see cref="StreamingResponseScanner"/>, then fed through a per-choice <see cref="StreamingRestorer"/> so a
        /// placeholder split across SSE chunks is still restored. Both detect across chunk boundaries; the token
        /// stream is never fully buffered — chunks are rewritten and forwarded as they arrive.
        /// </summary>
        private static void StreamResponse(
            HttpListenerContext ctx,
            HttpRequestMessage upstreamRequest,
            IReadOnlyList<RedactionMatch> matches,
            Redactor redactor,
            RedactionPolicy policy,
            IRedactionAuditSink audit,
            bool scanResponses,
            HttpClient http)
        {
            using var upstreamResponse = http.Send(upstreamRequest, HttpCompletionOption.ResponseHeadersRead);

            var clientResponse = ctx.Response;
            clientResponse.StatusCode = (int)upstreamResponse.StatusCode;
            ForwardResponseHeaders(upstreamResponse, clientResponse);
            // Set the SSE-framing headers AFTER forwarding so the gateway's values win over any upstream duplicates.
            clientResponse.ContentType = "text/event-stream";
            clientResponse.Headers["Cache-Control"] = "no-cache";
            clientResponse.SendChunked = true;

            using var upstreamStream = upstreamResponse.Content.ReadAsStream();
            using var reader = new StreamReader(upstreamStream, Encoding.UTF8);
            var output = clientResponse.OutputStream;

            // Per choice index (usually one, but n>1 is legal): a response scanner (mask model-generated secrets) and
            // a restorer (re-hydrate the caller's own placeholders). The scanner runs first while the caller's values
            // are still placeholders, so only genuinely model-generated content is masked.
            var restorers = new Dictionary<int, StreamingRestorer>();
            var scanners = scanResponses ? new Dictionary<int, StreamingResponseScanner>() : null;

            string? line;
            while ((line = reader.ReadLine()) is not null)
            {
                if (!line.StartsWith("data:", StringComparison.Ordinal))
                {
                    // Comments / blank separators — forward verbatim to preserve SSE framing.
                    WriteLine(output, line);
                    continue;
                }

                var payload = line.Substring("data:".Length).TrimStart();

                if (payload == "[DONE]")
                {
                    // Release any held-back tails before closing the stream, then audit what the scanner masked.
                    FlushStreams(output, scanners, restorers);
                    if (scanners is not null)
                    {
                        AuditStreamScanned(audit, scanners);
                    }
                    WriteLine(output, "data: [DONE]");
                    WriteLine(output, string.Empty);
                    break;
                }

                if (matches.Count == 0 && !scanResponses)
                {
                    WriteLine(output, line);
                    WriteLine(output, string.Empty);
                    continue;
                }

                var rewritten = RewriteChunk(payload, matches, redactor, policy, scanners, restorers);
                WriteLine(output, "data: " + rewritten);
                WriteLine(output, string.Empty);
                output.Flush();
            }

            output.Close();
        }

        // Scans (model secrets) then restores (caller placeholders) a single SSE chunk's delta content per choice,
        // returning the chunk re-serialized. Falls back to the raw payload if it is not a parseable chunk.
        private static string RewriteChunk(
            string payload,
            IReadOnlyList<RedactionMatch> matches,
            Redactor redactor,
            RedactionPolicy policy,
            Dictionary<int, StreamingResponseScanner>? scanners,
            Dictionary<int, StreamingRestorer> restorers)
        {
            ChatCompletionChunk? chunk;
            try
            {
                chunk = JsonSerializer.Deserialize(payload, OpenAiJsonContext.Default.ChatCompletionChunk);
            }
            catch (JsonException)
            {
                return payload;
            }

            if (chunk is null)
            {
                return payload;
            }

            foreach (var choice in chunk.Choices)
            {
                if (choice.Delta?.Content is not { Length: > 0 } content)
                {
                    continue;
                }

                if (scanners is not null)
                {
                    if (!scanners.TryGetValue(choice.Index, out var scanner))
                    {
                        scanner = new StreamingResponseScanner(redactor, policy);
                        scanners[choice.Index] = scanner;
                    }
                    content = scanner.Push(content);
                }

                if (!restorers.TryGetValue(choice.Index, out var restorer))
                {
                    restorer = new StreamingRestorer(matches);
                    restorers[choice.Index] = restorer;
                }

                choice.Delta.Content = restorer.Push(content);
            }

            return JsonSerializer.Serialize(chunk, OpenAiJsonContext.Default.ChatCompletionChunk);
        }

        // Emits any text the scanners/restorers held back, as a final synthetic chunk per choice, before [DONE].
        // Per choice: flush the scanner (mask remaining model secrets) → feed through the restorer → flush it.
        private static void FlushStreams(
            System.IO.Stream output,
            Dictionary<int, StreamingResponseScanner>? scanners,
            Dictionary<int, StreamingRestorer> restorers)
        {
            foreach (var pair in restorers)
            {
                var restorer = pair.Value;
                var tail = string.Empty;

                if (scanners is not null && scanners.TryGetValue(pair.Key, out var scanner))
                {
                    tail = restorer.Push(scanner.Flush());
                }

                tail += restorer.Flush();

                if (tail.Length == 0)
                {
                    continue;
                }

                var chunk = new ChatCompletionChunk
                {
                    Choices = [new ChatChoice { Index = pair.Key, Delta = new OpenAiMessage { Content = tail } }]
                };
                WriteLine(output, "data: " + JsonSerializer.Serialize(chunk, OpenAiJsonContext.Default.ChatCompletionChunk));
                WriteLine(output, string.Empty);
            }
        }

        // Audits (counts only) everything the streaming scanners masked across the whole response.
        private static void AuditStreamScanned(IRedactionAuditSink audit, Dictionary<int, StreamingResponseScanner> scanners)
        {
            var masked = new List<RedactionMatch>();
            foreach (var scanner in scanners.Values)
            {
                masked.AddRange(scanner.MaskedMatches);
            }

            AuditRedactions(audit, masked);
        }

        private static void WriteLine(System.IO.Stream output, string text)
        {
            var bytes = Encoding.UTF8.GetBytes(text + "\n");
            output.Write(bytes, 0, bytes.Length);
        }

        /// <summary>
        /// Transparent redacting proxy for every endpoint other than chat completions (embeddings, legacy
        /// completions, <c>/v1/models</c>, anything future). A JSON request body is scanned with the policy: a
        /// blocked category is refused (403, nothing forwarded), otherwise sensitive spans are redacted before the
        /// body leaves the box and restored in the response. GETs and non-JSON bodies (e.g. multipart audio) pass
        /// through untouched so they are not corrupted. The gateway's own client-auth header is never forwarded;
        /// only the upstream key is injected.
        /// </summary>
        private static void HandleGenericProxy(
            HttpListenerContext ctx,
            string upstream,
            string? upstreamApiKey,
            Redactor redactor,
            IRedactionAuditSink audit,
            RedactionPolicy policy,
            bool scanResponses,
            HttpClient http)
        {
            var request = ctx.Request;
            var method = request.HttpMethod;
            var targetUrl = BuildUpstreamUrl(upstream, request.Url?.AbsolutePath ?? "/") + (request.Url?.Query ?? string.Empty);

            using var upstreamRequest = new HttpRequestMessage(new HttpMethod(method), targetUrl);

            IReadOnlyList<RedactionMatch> matches = [];
            var carriesBody = method is "POST" or "PUT" or "PATCH";

            if (carriesBody)
            {
                string body;
                using (var reader = new StreamReader(request.InputStream, Encoding.UTF8))
                {
                    body = reader.ReadToEnd();
                }

                var contentType = request.ContentType ?? "application/octet-stream";
                var isJson = contentType.Contains("json", StringComparison.OrdinalIgnoreCase);

                if (isJson && body.Length > 0)
                {
                    // Redact the raw JSON body as text — placeholders ([REDACTED_…]) are JSON-string-safe, and
                    // Restore re-hydrates them on any response shape by exact token replacement.
                    var decision = redactor.Redact(body, policy);
                    if (decision.Blocked)
                    {
                        RespondBlocked(ctx, audit, decision.BlockedCategories);
                        return;
                    }

                    body = decision.Text;
                    matches = decision.RedactedMatches;
                    AuditRedactions(audit, matches);
                }

                upstreamRequest.Content = new StringContent(body, Encoding.UTF8, isJson ? "application/json" : contentType);
            }

            ForwardRequestHeaders(request, upstreamRequest);
            if (!string.IsNullOrEmpty(upstreamApiKey))
            {
                upstreamRequest.Headers.TryAddWithoutValidation("Authorization", $"Bearer {upstreamApiKey}");
            }

            using var upstreamResponse = http.Send(upstreamRequest);
            var responseBody = upstreamResponse.Content.ReadAsStringAsync().GetAwaiter().GetResult();

            // Response-side scan (model-generated leaks), then restore the caller's own placeholders — same order as
            // the chat path: scan first while originals are still placeholders, restore second.
            if (scanResponses && upstreamResponse.IsSuccessStatusCode)
            {
                responseBody = ScanResponseBody(responseBody, redactor, policy, audit);
            }

            if (matches.Count > 0 && upstreamResponse.IsSuccessStatusCode)
            {
                responseBody = Redactor.Restore(responseBody, matches);
            }

            ForwardResponseHeaders(upstreamResponse, ctx.Response);
            var responseContentType = upstreamResponse.Content.Headers.ContentType?.ToString() ?? "application/json";
            WriteRaw(ctx.Response, (int)upstreamResponse.StatusCode, responseBody, responseContentType);
        }

        // Maps an incoming gateway path to the upstream URL. The upstream base already ends in the version segment
        // (e.g. .../v1), so a "/v1/…" request path has its "/v1" stripped before being appended.
        private static string BuildUpstreamUrl(string upstream, string path)
        {
            var sub = path.StartsWith("/v1/", StringComparison.Ordinal) ? path.Substring("/v1".Length) : path;
            return upstream + sub;
        }

        // Headers never forwarded upstream: the caller's gateway key (the real upstream key is injected instead),
        // hop-by-hop framing headers, and content/encoding headers owned by the rewritten body. Accept-Encoding is
        // dropped so the upstream replies in identity encoding — the body is read and re-serialized as text.
        private static readonly HashSet<string> NonForwardableRequestHeaders = new(StringComparer.OrdinalIgnoreCase)
        {
            "Host", "Authorization", "Content-Length", "Content-Type", "Connection", "Keep-Alive",
            "Proxy-Connection", "Proxy-Authorization", "Transfer-Encoding", "Upgrade", "TE", "Trailer",
            "Expect", "Accept-Encoding",
        };

        // Upstream response headers the gateway sets itself, or that describe a body we re-encode as identity text.
        private static readonly HashSet<string> NonForwardableResponseHeaders = new(StringComparer.OrdinalIgnoreCase)
        {
            "Content-Length", "Content-Type", "Content-Encoding", "Transfer-Encoding", "Connection",
            "Keep-Alive", "Trailer", "Upgrade",
        };

        // Passes the caller's request headers (OpenAI-Beta, OpenAI-Organization/Project, X-*, User-Agent, Accept, …)
        // through to the upstream so client features keep working — minus the security/framing denylist. The client's
        // Authorization (its gateway key) is dropped here; the real upstream key is injected separately by the caller.
        private static void ForwardRequestHeaders(HttpListenerRequest src, HttpRequestMessage dst)
        {
            var headers = src.Headers;
            for (var i = 0; i < headers.Count; i++)
            {
                var name = headers.GetKey(i);
                if (name is null || NonForwardableRequestHeaders.Contains(name))
                {
                    continue;
                }

                var values = headers.GetValues(i);
                if (values is not null)
                {
                    dst.Headers.TryAddWithoutValidation(name, values);
                }
            }
        }

        // Passes upstream response headers (x-request-id, x-ratelimit-*, openai-*, …) back to the caller so clients
        // can see rate limits and request ids — minus headers the gateway manages itself.
        private static void ForwardResponseHeaders(HttpResponseMessage upstream, HttpListenerResponse client)
        {
            CopyResponseHeaders(upstream.Headers, client);
            if (upstream.Content is not null)
            {
                CopyResponseHeaders(upstream.Content.Headers, client);
            }
        }

        private static void CopyResponseHeaders(System.Net.Http.Headers.HttpHeaders headers, HttpListenerResponse client)
        {
            foreach (var header in headers)
            {
                if (NonForwardableResponseHeaders.Contains(header.Key))
                {
                    continue;
                }

                try
                {
                    client.Headers[header.Key] = string.Join(", ", header.Value);
                }
                catch (ArgumentException)
                {
                    // Restricted header the HttpListener manages itself — skip it.
                }
            }
        }

        // Records a redaction audit entry (per-category counts only — never the values).
        private static void AuditRedactions(IRedactionAuditSink audit, IReadOnlyList<RedactionMatch> matches)
        {
            if (matches.Count == 0)
            {
                return;
            }

            var counts = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var match in matches)
            {
                counts[match.Category] = counts.GetValueOrDefault(match.Category) + 1;
            }

            audit.Record(new RedactionAuditRecord(
                Guid.NewGuid().ToString("N"), DateTimeOffset.UtcNow, matches.Count, counts));
        }

        // Refuses a request whose payload carried a Block-policy category: 403, audit the blocked category, no forward.
        private static void RespondBlocked(HttpListenerContext ctx, IRedactionAuditSink audit, IReadOnlyList<string> blockedCategories)
        {
            var blockCounts = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var category in blockedCategories)
            {
                blockCounts["BLOCKED:" + category] = 1;
            }
            audit.Record(new RedactionAuditRecord(
                Guid.NewGuid().ToString("N"), DateTimeOffset.UtcNow, blockedCategories.Count, blockCounts));

            WriteText(ctx.Response, HttpStatusCode.Forbidden,
                $"Request refused by the redaction gateway: it contains forbidden category(ies) "
                + $"[{string.Join(", ", blockedCategories)}] that must not leave the box. Nothing was forwarded.");
        }

        private static void WriteRaw(HttpListenerResponse response, int status, string json)
        {
            WriteRaw(response, status, json, "application/json");
        }

        private static void WriteRaw(HttpListenerResponse response, int status, string body, string contentType)
        {
            var bytes = Encoding.UTF8.GetBytes(body);
            response.StatusCode = status;
            response.ContentType = contentType;
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
