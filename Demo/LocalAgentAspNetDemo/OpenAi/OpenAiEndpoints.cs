// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using DevOnBike.Overfit.Demo.LocalAgent.Observability;
using DevOnBike.Overfit.Demo.LocalAgent.Rag;
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.Server.OpenAi;
using Microsoft.AspNetCore.Http.Features;

namespace DevOnBike.Overfit.Demo.LocalAgent.OpenAi
{
    /// <summary>
    /// OpenAI-compatible HTTP surface (<c>/v1/chat/completions</c>, <c>/v1/embeddings</c>, <c>/v1/models</c>)
    /// so existing OpenAI tooling points at Overfit by only changing the base URL — the in-process .NET moat
    /// behind the standard API. Chat completions are STATELESS per request (the full <c>messages[]</c> is
    /// replayed each call), serialized through a single-flight gate because the demo shares one model session
    /// (single-tenant; for multi-tenant use a session-per-request pool). No continuous batching — same model
    /// as dotLLM's server (sequential), which is fine for an in-process embed.
    /// </summary>
    public static class OpenAiEndpoints
    {
        // The shared model session is single-tenant — serialize generation so concurrent callers don't
        // interleave on one KV cache. One in-flight request at a time (like a local llama.cpp server).
        private static readonly SemaphoreSlim Gate = new(1, 1);

        // Omit null role/content/finish_reason (streaming deltas carry only what changed).
        private static readonly JsonSerializerOptions Json = new()
        {
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        };

        public static void MapOpenAiApi(this WebApplication app, string modelName, string systemMessage)
        {
            var created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

            app.MapGet("/v1/models", () => Results.Json(new ModelsResponse { Data = [new ModelInfo { Id = modelName, Created = created }] }, Json));

            app.MapPost("/v1/chat/completions", async (ChatCompletionRequest req, HttpContext ctx, OverfitClient client, MetricsCollector metrics) =>
            {
                if (req.Messages is not { Count: > 0 })
                {
                    return Results.BadRequest(new
                    {
                        error = "'messages' is required and must be non-empty."
                    });
                }

                var last = req.Messages[^1];
                if (!string.Equals(last.Role, "user", StringComparison.OrdinalIgnoreCase))
                {
                    return Results.BadRequest(new
                    {
                        error = "the last message must have role 'user'."
                    });
                }

                var (sampling, maxTokens) = OpenAiChatMapping.BuildSampling(req);
                var options = new GenerationOptions(maxTokens, maxContextLength: 8192, sampling, stopOnEndOfTextToken: true);
                var id = "chatcmpl-" + Guid.NewGuid().ToString("N");
                var ts = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

                // response_format → an optional decode-time constraint (well-formed JSON, or schema-conforming).
                ITokenConstraint? constraint;
                try
                {
                    constraint = OpenAiChatMapping.BuildResponseFormatConstraint(req.ResponseFormat, client.Tokenizer);
                }
                catch (JsonException ex)
                {
                    return Results.BadRequest(new
                    {
                        error = $"invalid response_format: {ex.Message}"
                    });
                }

                await Gate.WaitAsync(ctx.RequestAborted);
                try
                {
                    OpenAiChatMapping.ReplayHistory(client.Chat, req.Messages);
                    var userContent = last.Content ?? string.Empty;

                    if (!req.Stream)
                    {
                        var reply = client.Chat.Send(userContent, in options, onText: null, constraint: constraint);
                        var s = client.Chat.LastStats;
                        metrics.RecordGeneration("openai_chat", s);

                        return Results.Json(new ChatCompletionResponse
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
                                    // "length" when the cap truncated generation, else "stop" (per OpenAI spec).
                                    FinishReason = s.GeneratedTokens >= maxTokens ? "length" : "stop",
                                },
                            ],
                            Usage = new OpenAiUsage
                            {
                                PromptTokens = s.PromptTokens,
                                CompletionTokens = s.GeneratedTokens,
                                TotalTokens = s.PromptTokens + s.GeneratedTokens,
                            },
                        }, Json);
                    }

                    // Streaming (SSE). Blocking writes from the synchronous generate callback are fine here:
                    // the gate guarantees a single in-flight request, so the request thread owns the stream.
                    ctx.Response.Headers.ContentType = "text/event-stream";
                    ctx.Response.Headers.CacheControl = "no-cache";
                    var bodyControl = ctx.Features.Get<IHttpBodyControlFeature>();
                    if (bodyControl is not null)
                    {
                        bodyControl.AllowSynchronousIO = true;
                    }

                    WriteChunk(ctx, id, ts, modelName, new OpenAiMessage { Role = "assistant" }, finishReason: null);
                    client.Chat.Send(userContent, in options,
                        onText: delta => WriteChunk(ctx, id, ts, modelName, new OpenAiMessage { Content = delta }, finishReason: null),
                        constraint: constraint);
                    var streamStats = client.Chat.LastStats;
                    var streamFinish = streamStats.GeneratedTokens >= maxTokens ? "length" : "stop";
                    WriteChunk(ctx, id, ts, modelName, new OpenAiMessage(), finishReason: streamFinish);
                    WriteSseRaw(ctx, "[DONE]");

                    metrics.RecordGeneration("openai_chat_stream", streamStats);
                    return Results.Empty;
                }
                finally
                {
                    // Restore the baseline system turn so the shared single-tenant session keeps /chat working.
                    client.Reset();
                    client.AddSystem(systemMessage);
                    Gate.Release();
                }
            });

            app.MapPost("/v1/embeddings", (EmbeddingsRequest req, RagService rag, MetricsCollector metrics) =>
            {
                var inputs = OpenAiChatMapping.ParseInputs(req.Input);
                if (inputs.Count == 0)
                {
                    return Results.BadRequest(new
                    {
                        error = "'input' is required (a string or array of strings)."
                    });
                }

                try
                {
                    var vectors = rag.EmbedAll(inputs);
                    var data = new List<EmbeddingData>(vectors.Count);
                    for (var i = 0; i < vectors.Count; i++)
                    {
                        data.Add(new EmbeddingData { Index = i, Embedding = vectors[i] });
                    }

                    var approxTokens = 0;
                    foreach (var t in inputs)
                    {
                        approxTokens += Math.Max(1, t.Length / 4);
                    }   // rough proxy

                    return Results.Json(new EmbeddingsResponse
                    {
                        Model = modelName,
                        Data = data,
                        Usage = new OpenAiUsage { PromptTokens = approxTokens, TotalTokens = approxTokens },
                    }, Json);
                }
                catch (InvalidOperationException ex)
                {
                    // No embedding model configured / found — actionable client error, not a 500.
                    return Results.Problem(detail: ex.Message, statusCode: StatusCodes.Status400BadRequest);
                }
            });
        }

        private static void WriteChunk(HttpContext ctx, string id, long created, string model, OpenAiMessage delta, string? finishReason)
        {
            var chunk = new ChatCompletionChunk
            {
                Id = id,
                Created = created,
                Model = model,
                Choices = [new ChatChoice { Index = 0, Delta = delta, FinishReason = finishReason }],
            };
            WriteSseRaw(ctx, JsonSerializer.Serialize(chunk, Json));
        }

        private static void WriteSseRaw(HttpContext ctx, string data)
        {
            var bytes = Encoding.UTF8.GetBytes($"data: {data}\n\n");
            ctx.Response.Body.Write(bytes, 0, bytes.Length);
            ctx.Response.Body.Flush();
        }
    }
}
