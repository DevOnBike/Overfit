// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Net;
using System.Text;
using System.Text.Json;
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.Server.OpenAi;

namespace DevOnBike.Overfit.Server
{
    /// <summary>
    /// A dependency-free, OpenAI-compatible HTTP server over <see cref="HttpListener"/> — no ASP.NET Core, so it
    /// drops cleanly into the Native-AOT <c>overfit</c> CLI. Exposes <c>/v1/chat/completions</c> (streaming SSE +
    /// non-streaming), <c>/v1/models</c> and <c>/health</c>; point any OpenAI client at the base URL and only
    /// change the model name. Requests are served STRICTLY ONE AT A TIME (single-tenant model session, one KV
    /// cache) — exactly like a local llama.cpp server; for multi-tenant use a session-per-request pool.
    /// </summary>
    public static class OverfitOpenAiServer
    {
        /// <summary>
        /// Binds an <see cref="HttpListener"/> on <paramref name="host"/>:<paramref name="port"/> and serves
        /// requests until <paramref name="cancellationToken"/> is cancelled. Blocks the calling thread. Each
        /// request replays its full <c>messages[]</c> and restores the baseline system turn afterwards, so the
        /// shared session never accumulates state across calls.
        /// </summary>
        /// <param name="client">A loaded model client; owned by the caller (not disposed here).</param>
        /// <param name="modelName">The id reported by <c>/v1/models</c> and echoed in responses.</param>
        /// <param name="host">Bind host. <c>127.0.0.1</c>/<c>localhost</c> need no elevation; <c>0.0.0.0</c>/<c>*</c> bind all interfaces (may need a URL ACL / admin on Windows).</param>
        /// <param name="port">TCP port.</param>
        /// <param name="systemMessage">Baseline system prompt restored after every request.</param>
        /// <param name="onListening">Optional callback invoked once the listener is up, with the base URL.</param>
        /// <param name="cancellationToken">Cancel to stop the server gracefully.</param>
        public static void Serve(
            OverfitClient client,
            string modelName,
            string host,
            int port,
            string systemMessage,
            Action<string>? onListening = null,
            CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(client);

            var bindHost = host is "0.0.0.0" or "*" or "+" ? "+" : host;
            var prefix = $"http://{bindHost}:{port}/";

            using var listener = new HttpListener();
            listener.Prefixes.Add(prefix);
            listener.Start();

            var displayHost = bindHost == "+" ? "0.0.0.0" : host;
            onListening?.Invoke($"http://{displayHost}:{port}");

            using var stop = cancellationToken.Register(() =>
            {
                try
                {
                    listener.Stop();
                }
                catch
                {
                    // listener already torn down — nothing to do.
                }
            });

            var created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

            while (!cancellationToken.IsCancellationRequested)
            {
                HttpListenerContext ctx;
                try
                {
                    ctx = listener.GetContext();
                }
                catch (HttpListenerException)
                {
                    break;   // Stop() was called.
                }
                catch (InvalidOperationException)
                {
                    break;   // listener disposed.
                }

                try
                {
                    Handle(ctx, client, modelName, systemMessage, created);
                }
                catch (Exception ex)
                {
                    TryWriteError(ctx.Response, HttpStatusCode.InternalServerError, ex.Message);
                }
                finally
                {
                    try
                    {
                        ctx.Response.Close();
                    }
                    catch
                    {
                        // client may have already disconnected (e.g. aborted a stream).
                    }
                }
            }
        }

        private static void Handle(HttpListenerContext ctx, OverfitClient client, string modelName, string systemMessage, long created)
        {
            var req = ctx.Request;
            var path = req.Url?.AbsolutePath ?? "/";
            var method = req.HttpMethod;

            if (method == "GET" && path is "/health" or "/")
            {
                WriteRaw(ctx.Response, HttpStatusCode.OK, "application/json", "{\"status\":\"ok\"}");
                return;
            }

            if (method == "GET" && path == "/v1/models")
            {
                var models = new ModelsResponse { Data = [new ModelInfo { Id = modelName, Created = created }] };
                WriteJson(ctx.Response, HttpStatusCode.OK, models, OpenAiJsonContext.Default.ModelsResponse);
                return;
            }

            if (method == "POST" && path == "/v1/chat/completions")
            {
                HandleChatCompletions(ctx, client, modelName, systemMessage);
                return;
            }

            if (method == "POST" && path == "/v1/embeddings")
            {
                // The CLI loads a chat/generation GGUF, not a sentence-embedding model; embeddings need a
                // separate embedder (SentenceEmbedder). Surface a clear, actionable 501 rather than a 500.
                TryWriteError(ctx.Response, HttpStatusCode.NotImplemented,
                    "embeddings are not served by 'overfit serve' (load a sentence-embedding model via the API/SDK).");
                return;
            }

            TryWriteError(ctx.Response, HttpStatusCode.NotFound, $"no route for {method} {path}");
        }

        private static void HandleChatCompletions(HttpListenerContext ctx, OverfitClient client, string modelName, string systemMessage)
        {
            ChatCompletionRequest? req;
            try
            {
                req = JsonSerializer.Deserialize(ctx.Request.InputStream, OpenAiJsonContext.Default.ChatCompletionRequest);
            }
            catch (JsonException ex)
            {
                TryWriteError(ctx.Response, HttpStatusCode.BadRequest, $"invalid JSON body: {ex.Message}");
                return;
            }

            if (req is null || req.Messages is not { Count: > 0 })
            {
                TryWriteError(ctx.Response, HttpStatusCode.BadRequest, "'messages' is required and must be non-empty.");
                return;
            }

            var last = req.Messages[^1];
            if (!string.Equals(last.Role, "user", StringComparison.OrdinalIgnoreCase))
            {
                TryWriteError(ctx.Response, HttpStatusCode.BadRequest, "the last message must have role 'user'.");
                return;
            }

            var (sampling, maxTokens) = OpenAiChatMapping.BuildSampling(req);
            var options = new GenerationOptions(maxTokens, maxContextLength: 8192, sampling, stopOnEndOfTextToken: true);
            var id = "chatcmpl-" + Guid.NewGuid().ToString("N");
            var ts = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

            ITokenConstraint? constraint;
            try
            {
                constraint = OpenAiChatMapping.BuildResponseFormatConstraint(req.ResponseFormat, client.Tokenizer);
            }
            catch (JsonException ex)
            {
                TryWriteError(ctx.Response, HttpStatusCode.BadRequest, $"invalid response_format: {ex.Message}");
                return;
            }

            try
            {
                OpenAiChatMapping.ReplayHistory(client.Chat, req.Messages);
                var userContent = last.Content ?? string.Empty;

                if (!req.Stream)
                {
                    var reply = client.Chat.Send(userContent, in options, onText: null, constraint: constraint);
                    var s = client.Chat.LastStats;

                    var response = new ChatCompletionResponse
                    {
                        Id = id,
                        Created = ts,
                        Model = modelName,
                        Choices =
                        [
                            new ChatChoice
                            {
                                Index = 0,
                                Message = new OpenAiMessage { Role = "assistant", Content = reply },
                                FinishReason = s.GeneratedTokens >= maxTokens ? "length" : "stop",
                            },
                        ],
                        Usage = new OpenAiUsage
                        {
                            PromptTokens = s.PromptTokens,
                            CompletionTokens = s.GeneratedTokens,
                            TotalTokens = s.PromptTokens + s.GeneratedTokens,
                        },
                    };
                    WriteJson(ctx.Response, HttpStatusCode.OK, response, OpenAiJsonContext.Default.ChatCompletionResponse);
                    return;
                }

                // Streaming (SSE). One in-flight request at a time, so the request thread owns the stream.
                var resp = ctx.Response;
                resp.StatusCode = (int)HttpStatusCode.OK;
                resp.ContentType = "text/event-stream";
                resp.Headers["Cache-Control"] = "no-cache";
                resp.SendChunked = true;

                WriteChunk(resp, id, ts, modelName, new OpenAiMessage { Role = "assistant" }, finishReason: null);
                client.Chat.Send(userContent, in options,
                    onText: delta => WriteChunk(resp, id, ts, modelName, new OpenAiMessage { Content = delta }, finishReason: null),
                    constraint: constraint);

                var streamStats = client.Chat.LastStats;
                var streamFinish = streamStats.GeneratedTokens >= maxTokens ? "length" : "stop";
                WriteChunk(resp, id, ts, modelName, new OpenAiMessage(), finishReason: streamFinish);
                WriteSseRaw(resp, "[DONE]");
            }
            finally
            {
                // Restore the baseline system turn so the shared single-tenant session stays clean.
                client.Reset();
                if (!string.IsNullOrEmpty(systemMessage))
                {
                    client.AddSystem(systemMessage);
                }
            }
        }

        private static void WriteChunk(HttpListenerResponse resp, string id, long created, string model, OpenAiMessage delta, string? finishReason)
        {
            var chunk = new ChatCompletionChunk
            {
                Id = id,
                Created = created,
                Model = model,
                Choices = [new ChatChoice { Index = 0, Delta = delta, FinishReason = finishReason }],
            };
            WriteSseRaw(resp, JsonSerializer.Serialize(chunk, OpenAiJsonContext.Default.ChatCompletionChunk));
        }

        private static void WriteSseRaw(HttpListenerResponse resp, string data)
        {
            var bytes = Encoding.UTF8.GetBytes($"data: {data}\n\n");
            resp.OutputStream.Write(bytes, 0, bytes.Length);
            resp.OutputStream.Flush();
        }

        private static void WriteJson<T>(HttpListenerResponse resp, HttpStatusCode status, T body, System.Text.Json.Serialization.Metadata.JsonTypeInfo<T> typeInfo)
        {
            var json = JsonSerializer.Serialize(body, typeInfo);
            WriteRaw(resp, status, "application/json", json);
        }

        private static void WriteRaw(HttpListenerResponse resp, HttpStatusCode status, string contentType, string body)
        {
            var bytes = Encoding.UTF8.GetBytes(body);
            resp.StatusCode = (int)status;
            resp.ContentType = contentType;
            resp.ContentLength64 = bytes.Length;
            resp.OutputStream.Write(bytes, 0, bytes.Length);
        }

        private static void TryWriteError(HttpListenerResponse resp, HttpStatusCode status, string message)
        {
            try
            {
                var body = new OpenAiErrorResponse { Error = new OpenAiError { Message = message } };
                var json = JsonSerializer.Serialize(body, OpenAiJsonContext.Default.OpenAiErrorResponse);
                WriteRaw(resp, status, "application/json", json);
            }
            catch
            {
                // headers already sent (e.g. mid-stream) — can't change the status now.
            }
        }
    }
}
