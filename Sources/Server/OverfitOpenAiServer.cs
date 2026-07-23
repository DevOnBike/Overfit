// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Net;
using System.Text;
using System.Text.Json;
using DevOnBike.Overfit.Audio;
using DevOnBike.Overfit.Audio.Tts;
using DevOnBike.Overfit.Audio.Tts.Orpheus;
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Embeddings;
using DevOnBike.Overfit.Server.OpenAi;
using DevOnBike.Overfit.Serving;
using DevOnBike.Overfit.Diagnostics;
using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.Server
{
    /// <summary>
    /// A dependency-free, OpenAI-compatible HTTP server over <see cref="HttpListener"/> — no ASP.NET Core, so it
    /// drops cleanly into the Native-AOT <c>overfit</c> CLI. Exposes <c>/v1/chat/completions</c> (streaming SSE +
    /// non-streaming), <c>/v1/models</c>, <c>/v1/embeddings</c>, <c>/v1/audio/speech</c> and <c>/health</c>, plus
    /// the self-describing <c>/openapi.yaml</c> (the API contract) and <c>/docs</c> (Swagger UI). Point any OpenAI
    /// client at the base URL and only change the model name. Concurrency is bounded by the client pool: the
    /// single-client <see cref="Serve(OverfitClient, string, string, int, string, SentenceEmbedder, OrpheusVoiceEngine, Action{string}, CancellationToken)"/>
    /// overload serialises requests through one session (like a local llama.cpp server); the
    /// <see cref="Serve(OverfitResourcePool{OverfitClient}, string, string, int, string, SentenceEmbedder, OrpheusVoiceEngine, Action{string}, CancellationToken)"/>
    /// overload decodes up to <c>pool.Size</c> chat requests at once and sheds excess load with HTTP 503.
    /// </summary>
    public static class OverfitOpenAiServer
    {
        // How long a chat request waits for a free session before the server sheds it with HTTP 503.
        private const int RentTimeoutSeconds = 30;

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
        /// <param name="embedder">Optional in-process sentence embedder. When supplied, <c>/v1/embeddings</c>
        /// serves it (pure .NET, no data egress); when null that route returns 501. Owned by the caller.</param>
        /// <param name="onListening">Optional callback invoked once the listener is up, with the base URL.</param>
        /// <param name="cancellationToken">Cancel to stop the server gracefully.</param>
        public static void Serve(
            OverfitClient client,
            string modelName,
            string host,
            int port,
            string systemMessage,
            SentenceEmbedder? embedder = null,
            OrpheusVoiceEngine? tts = null,
            Action<string>? onListening = null,
            CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(client);

            // Single caller-owned client → a pool-of-1 that does NOT own it (the caller still disposes it).
            // Behaviour is identical to before: one session, requests serialised through the single client.
            using var pool = new OverfitResourcePool<OverfitClient>([client], ownsItems: false);
            Serve(pool, modelName, host, port, systemMessage, embedder, tts, onListening, cancellationToken);
        }

        /// <summary>
        /// Multi-session overload: serves requests across a <see cref="OverfitResourcePool{T}"/> of clients so up
        /// to <c>pool.Size</c> chat completions decode concurrently (each client owns its KV cache; the weights are
        /// shared via mmap). Requests beyond the pool wait up to a timeout and are otherwise shed with HTTP 503.
        /// Embeddings and TTS use single shared engines and are serialised. <c>/health</c> reports pool load.
        /// </summary>
        public static void Serve(
            OverfitResourcePool<OverfitClient> pool,
            string modelName,
            string host,
            int port,
            string systemMessage,
            SentenceEmbedder? embedder = null,
            OrpheusVoiceEngine? tts = null,
            Action<string>? onListening = null,
            CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(pool);

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

            var state = new ServerState
            {
                Pool = pool,
                ModelName = modelName,
                SystemMessage = systemMessage,
                Embedder = embedder,
                Tts = tts,
                Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                EmbedGate = new SemaphoreSlim(1, 1),
                TtsGate = new SemaphoreSlim(1, 1),
                RentTimeout = TimeSpan.FromSeconds(RentTimeoutSeconds),
                StopToken = cancellationToken,
            };

            var inFlight = 0;
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

                // One task per request: up to pool.Size chat decodes run concurrently; the rest wait/shed.
                Interlocked.Increment(ref inFlight);
                _ = Task.Run(() =>
                {
                    try
                    {
                        HandleRequest(ctx, state);
                    }
                    finally
                    {
                        Interlocked.Decrement(ref inFlight);
                    }
                });
            }

            // Drain in-flight requests (bounded) so pooled clients aren't disposed mid-decode by the caller.
            for (var i = 0; i < 200 && Volatile.Read(ref inFlight) > 0; i++)
            {
                Thread.Sleep(50);
            }

            state.EmbedGate.Dispose();
            state.TtsGate.Dispose();
        }

        private static void HandleRequest(HttpListenerContext ctx, ServerState s)
        {
            try
            {
                var req = ctx.Request;
                var path = req.Url?.AbsolutePath ?? "/";
                var method = req.HttpMethod;

                if (method == "GET" && path is "/health" or "/")
                {
                    var m = s.Pool.Metrics;
                    WriteRaw(ctx.Response, HttpStatusCode.OK, "application/json",
                        $"{{\"status\":\"ok\",\"sessions\":{{\"size\":{m.Size},\"active\":{m.Active},"
                        + $"\"available\":{m.Available},\"rented\":{m.TotalRented},\"rejected\":{m.TotalRejected},"
                        + $"\"peak\":{m.PeakActive}}}}}");
                    return;
                }

                if (method == "GET" && path == "/openapi.yaml")
                {
                    // The machine-readable contract — import into Swagger UI / Postman / an OpenAPI codegen.
                    WriteRaw(ctx.Response, HttpStatusCode.OK, "application/yaml; charset=utf-8", OpenApiYaml());
                    return;
                }

                if (method == "GET" && path is "/docs" or "/docs/")
                {
                    // Swagger UI for this server's /openapi.yaml. The viewer assets load from a CDN, so /docs
                    // needs internet to render (the API itself stays fully local — no prompt/data leaves).
                    WriteRaw(ctx.Response, HttpStatusCode.OK, "text/html; charset=utf-8", SwaggerUiHtml);
                    return;
                }

                if (method == "GET" && path == "/v1/models")
                {
                    var models = new ModelsResponse { Data = [new ModelInfo { Id = s.ModelName, Created = s.Created }] };
                    WriteJson(ctx.Response, HttpStatusCode.OK, models, OpenAiJsonContext.Default.ModelsResponse);
                    return;
                }

                if (method == "POST" && path == "/v1/chat/completions")
                {
                    // Rent a session for the duration of the decode. Full pool → wait up to RentTimeout, then shed
                    // load with 503 rather than queue unboundedly. A cancelled wait (server stopping) is also a 503.
                    OverfitResourcePool<OverfitClient>.Lease lease;
                    try
                    {
                        if (!s.Pool.TryRent(s.RentTimeout, s.StopToken, out lease))
                        {
                            TryWriteError(ctx.Response, HttpStatusCode.ServiceUnavailable,
                                $"server busy — all {s.Pool.Size} sessions in use; retry shortly.");
                            return;
                        }
                    }
                    catch (OperationCanceledException)
                    {
                        TryWriteError(ctx.Response, HttpStatusCode.ServiceUnavailable, "server is shutting down.");
                        return;
                    }

                    using (lease)
                    {
                        HandleChatCompletions(ctx, lease.Value, s.ModelName, s.SystemMessage);
                    }
                    return;
                }

                if (method == "POST" && path == "/v1/embeddings")
                {
                    if (s.Embedder is null)
                    {
                        // No embedder loaded — a chat GGUF alone can't serve sentence embeddings. Clear, actionable 501.
                        TryWriteError(ctx.Response, HttpStatusCode.NotImplemented,
                            "embeddings are not served — start with an embedding model (e.g. 'overfit serve <model> --embed-model <dir>').");
                        return;
                    }

                    // SentenceEmbedder holds a single scratch arena — serialise concurrent embedding calls.
                    s.EmbedGate.Wait(s.StopToken);
                    try
                    {
                        HandleEmbeddings(ctx, s.Embedder, s.ModelName);
                    }
                    finally
                    {
                        s.EmbedGate.Release();
                    }
                    return;
                }

                if (method == "POST" && path == "/v1/audio/speech")
                {
                    if (s.Tts is null)
                    {
                        TryWriteError(ctx.Response, HttpStatusCode.NotImplemented,
                            "text-to-speech is not served — start with a TTS model (e.g. 'overfit serve <model> "
                            + "--tts-model <orpheus.gguf> --tts-snac <dir>').");
                        return;
                    }

                    // Single TTS engine — serialise.
                    s.TtsGate.Wait(s.StopToken);
                    try
                    {
                        HandleAudioSpeech(ctx, s.Tts);
                    }
                    finally
                    {
                        s.TtsGate.Release();
                    }
                    return;
                }

                TryWriteError(ctx.Response, HttpStatusCode.NotFound, $"no route for {method} {path}");
            }
            catch (OperationCanceledException)
            {
                TryWriteError(ctx.Response, HttpStatusCode.ServiceUnavailable, "server is shutting down.");
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

        /// <summary>Per-server shared state handed to each request task.</summary>
        private sealed class ServerState
        {
            public required OverfitResourcePool<OverfitClient> Pool;
            public required string ModelName;
            public required string SystemMessage;
            public SentenceEmbedder? Embedder;
            public OrpheusVoiceEngine? Tts;
            public long Created;
            public required SemaphoreSlim EmbedGate;
            public required SemaphoreSlim TtsGate;
            public TimeSpan RentTimeout;
            public CancellationToken StopToken;
        }

        private static void HandleAudioSpeech(HttpListenerContext ctx, OrpheusVoiceEngine tts)
        {
            SpeechRequest? req;
            try
            {
                req = JsonSerializer.Deserialize(ctx.Request.InputStream, OpenAiJsonContext.Default.SpeechRequest);
            }
            catch (JsonException ex)
            {
                TryWriteError(ctx.Response, HttpStatusCode.BadRequest, $"invalid JSON body: {ex.Message}");
                return;
            }

            SpeechExchange.Handle(req, tts, new HttpListenerResponseSink(ctx.Response));
        }

        /// <summary>Opt-in per-request phase trace (<c>OVERFIT_SERVER_TRACE=1</c>) for TTFT attribution.</summary>
        private static readonly bool ServerTrace =
            Environment.GetEnvironmentVariable(OverfitEnvironment.ServerTrace) == "1";

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

            // Everything past the body parse — validation, sampling, replay, streaming shape, finish-reason,
            // system-turn restore — is the shared protocol, run once in ChatCompletionExchange. This host
            // supplies only the wire adapter and (opt-in) the phase trace.
            var sink = new HttpListenerResponseSink(ctx.Response);
            var observer = ServerTrace ? ConsoleTraceObserver.Instance : null;
            ChatCompletionExchange.Handle(req, client, modelName, systemMessage, sink, observer);
        }

        private static void HandleEmbeddings(HttpListenerContext ctx, SentenceEmbedder embedder, string modelName)
        {
            EmbeddingsRequest? req;
            try
            {
                req = JsonSerializer.Deserialize(ctx.Request.InputStream, OpenAiJsonContext.Default.EmbeddingsRequest);
            }
            catch (JsonException ex)
            {
                TryWriteError(ctx.Response, HttpStatusCode.BadRequest, $"invalid JSON body: {ex.Message}");
                return;
            }

            EmbeddingsExchange.Handle(req, embedder, modelName, new HttpListenerResponseSink(ctx.Response));
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

        private static string? _openApiYaml;

        /// <summary>The embedded <c>openapi.yaml</c> contract, read once and cached. The server handles one
        /// request at a time (single-threaded accept loop), so a lock-free lazy init is safe here.</summary>
        private static string OpenApiYaml()
        {
            // Returning from each branch rather than falling through to a shared `return`: the field is
            // nullable, and the two assignments above a common exit are not enough for the compiler to prove
            // it was set (CS8603). With `else` banned, an early return per branch is the honest shape.
            if (_openApiYaml is not null)
            {
                return _openApiYaml;
            }

            using var stream = typeof(OverfitOpenAiServer).Assembly.GetManifestResourceStream("openapi.yaml");
            if (stream is null)
            {
                _openApiYaml = "openapi: 3.0.3\ninfo:\n  title: Overfit\n  version: '1.0.0'\npaths: {}\n";
                return _openApiYaml;
            }

            using var reader = new StreamReader(stream, Encoding.UTF8);
            _openApiYaml = reader.ReadToEnd();
            return _openApiYaml;
        }

        // Swagger UI viewer for /openapi.yaml. Lean by design: the UI bundle loads from a CDN instead of
        // bloating the single self-contained binary with ~1.5 MB of assets. Same-origin spec fetch + "try it out".
        private const string SwaggerUiHtml = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset="UTF-8" />
              <meta name="viewport" content="width=device-width, initial-scale=1" />
              <title>Overfit API — Swagger UI</title>
              <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
            </head>
            <body>
              <div id="swagger-ui"></div>
              <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js" crossorigin></script>
              <script>
                window.ui = SwaggerUIBundle({ url: '/openapi.yaml', dom_id: '#swagger-ui' });
              </script>
            </body>
            </html>
            """;
    }
}
