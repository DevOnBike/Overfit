// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using DevOnBike.Overfit.Diagnostics;
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.Server.OpenAi
{
    /// <summary>
    /// The one implementation of <c>POST /v1/chat/completions</c>, shared by every host. It owns the whole
    /// OpenAI protocol — validating the request, mapping sampling / response-format, replaying history,
    /// running the streaming and non-streaming generation, computing <c>finish_reason</c>, and restoring the
    /// baseline system turn — and writes exclusively through an <see cref="IOpenAiResponseSink"/>, so a host
    /// contributes only a byte-output adapter plus an optional <see cref="IChatExchangeObserver"/>.
    ///
    /// <para>Before this existed the orchestration lived twice (the <c>HttpListener</c> CLI server and the
    /// ASP.NET host), and the two drifted the moment either gained a feature the other lacked — the TTFT
    /// phase trace was added to one and the prompt-cache reuse count reported by only one. Keeping it in one
    /// place is the point.</para>
    ///
    /// <para><b>Session lifetime is the caller's.</b> This method neither rents from a pool nor holds a
    /// concurrency gate — a session decodes one request at a time, and serializing access to it is the host's
    /// job (the CLI rents a pooled client; the ASP.NET host holds a single-flight semaphore). The handler only
    /// resets the session afterwards so the next caller starts clean.</para>
    /// </summary>
    public static class ChatCompletionExchange
    {
        /// <summary>
        /// Runs one chat-completion exchange to completion, writing the whole response through
        /// <paramref name="sink"/>. <paramref name="req"/> may be <c>null</c> (a body that failed to parse) —
        /// it is validated here so every host rejects malformed input identically.
        /// </summary>
        public static void Handle(
            ChatCompletionRequest? req,
            OverfitClient client,
            string modelName,
            string systemMessage,
            IOpenAiResponseSink sink,
            IChatExchangeObserver? observer = null,
            IClock? clock = null)
        {
            ArgumentNullException.ThrowIfNull(client);
            ArgumentNullException.ThrowIfNull(sink);

            if (req is null || req.Messages is not { Count: > 0 })
            {
                WriteError(sink, 400, "'messages' is required and must be non-empty.");
                return;
            }

            var last = req.Messages[^1];
            if (!string.Equals(last.Role, "user", StringComparison.OrdinalIgnoreCase))
            {
                WriteError(sink, 400, "the last message must have role 'user'.");
                return;
            }

            var (sampling, maxTokens) = OpenAiChatMapping.BuildSampling(req);
            var options = new GenerationOptions(maxTokens, maxContextLength: 8192, sampling, stopOnEndOfTextToken: true);
            var id = "chatcmpl-" + Guid.NewGuid().ToString("N");
            var ts = (clock ?? SystemClock.Instance).UtcNow.ToUnixTimeSeconds();

            ITokenConstraint? constraint;
            try
            {
                constraint = OpenAiChatMapping.BuildResponseFormatConstraint(req.ResponseFormat, client.Tokenizer);
            }
            catch (JsonException ex)
            {
                WriteError(sink, 400, $"invalid response_format: {ex.Message}");
                return;
            }

            try
            {
                var replayStarted = observer is null ? default : ValueStopwatch.StartNew();
                OpenAiChatMapping.ReplayHistory(client.Chat, req.Messages);
                observer?.OnHistoryReplayed(req.Messages.Count, replayStarted.GetElapsedTime().TotalMilliseconds);

                var userContent = last.Content ?? string.Empty;

                if (!req.Stream)
                {
                    HandleNonStreaming(client, modelName, id, ts, maxTokens, userContent, options, constraint, sink, observer);
                    return;
                }

                HandleStreaming(client, modelName, id, ts, maxTokens, userContent, options, constraint, sink, observer);
            }
            finally
            {
                // Restore the baseline system turn so the shared single-tenant session stays clean for the
                // next caller. The host owns concurrency; by here it still holds the session exclusively.
                client.Reset();
                if (!string.IsNullOrEmpty(systemMessage))
                {
                    client.AddSystem(systemMessage);
                }
            }
        }

        private static void HandleNonStreaming(
            OverfitClient client, string modelName, string id, long ts, int maxTokens,
            string userContent, GenerationOptions options, ITokenConstraint? constraint,
            IOpenAiResponseSink sink, IChatExchangeObserver? observer)
        {
            var reply = client.Chat.Send(userContent, in options, onText: null, constraint: constraint);
            var stats = client.Chat.LastStats;

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
                        FinishReason = stats.GeneratedTokens >= maxTokens ? "length" : "stop",
                    },
                ],
                Usage = new OpenAiUsage
                {
                    PromptTokens = stats.PromptTokens,
                    CompletionTokens = stats.GeneratedTokens,
                    TotalTokens = stats.PromptTokens + stats.GeneratedTokens,
                },
            };

            var json = JsonSerializer.Serialize(response, OpenAiJsonContext.Default.ChatCompletionResponse);
            sink.WriteBody(200, "application/json", json);
            observer?.OnCompleted(streamed: false, stats, client.Chat.CachedPromptTokens);
        }

        private static void HandleStreaming(
            OverfitClient client, string modelName, string id, long ts, int maxTokens,
            string userContent, GenerationOptions options, ITokenConstraint? constraint,
            IOpenAiResponseSink sink, IChatExchangeObserver? observer)
        {
            sink.BeginEventStream();
            WriteChunk(sink, id, ts, modelName, new OpenAiMessage { Role = "assistant" }, finishReason: null);

            var sendStarted = observer is null ? default : ValueStopwatch.StartNew();
            var firstDelta = true;

            client.Chat.Send(userContent, in options,
                onText: delta =>
                {
                    if (observer is not null && firstDelta)
                    {
                        firstDelta = false;
                        observer.OnFirstToken(sendStarted.GetElapsedTime().TotalMilliseconds);
                    }

                    WriteChunk(sink, id, ts, modelName, new OpenAiMessage { Content = delta }, finishReason: null);
                },
                constraint: constraint);

            var stats = client.Chat.LastStats;
            var finish = stats.GeneratedTokens >= maxTokens ? "length" : "stop";
            WriteChunk(sink, id, ts, modelName, new OpenAiMessage(), finishReason: finish);
            sink.WriteEvent("[DONE]");
            observer?.OnCompleted(streamed: true, stats, client.Chat.CachedPromptTokens);
        }

        private static void WriteChunk(
            IOpenAiResponseSink sink, string id, long created, string model, OpenAiMessage delta, string? finishReason)
        {
            var chunk = new ChatCompletionChunk
            {
                Id = id,
                Created = created,
                Model = model,
                Choices = [new ChatChoice { Index = 0, Delta = delta, FinishReason = finishReason }],
            };
            sink.WriteEvent(JsonSerializer.Serialize(chunk, OpenAiJsonContext.Default.ChatCompletionChunk));
        }

        private static void WriteError(IOpenAiResponseSink sink, int status, string message)
        {
            var body = new OpenAiErrorResponse { Error = new OpenAiError { Message = message } };
            var json = JsonSerializer.Serialize(body, OpenAiJsonContext.Default.OpenAiErrorResponse);
            sink.WriteBody(status, "application/json", json);
        }
    }
}
