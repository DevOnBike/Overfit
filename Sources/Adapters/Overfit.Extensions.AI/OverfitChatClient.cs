// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using System.Threading.Channels;
using DevOnBike.Overfit.LanguageModels.Contracts;
using Microsoft.Extensions.AI;
using ChatSession = DevOnBike.Overfit.LanguageModels.Chat.ChatSession;

namespace DevOnBike.Overfit.Extensions.AI
{
    /// <summary>
    /// Exposes an in-process Overfit <see cref="ChatSession"/> as a standard <see cref="IChatClient"/>, so the
    /// Overfit runtime drops into Semantic Kernel and any <c>Microsoft.Extensions.AI</c> pipeline (caching,
    /// telemetry, function-invocation middleware, DI) by changing one line. Stateless per call — the full
    /// <c>messages</c> list is replayed into the underlying <see cref="ChatSession"/> each request (M.E.AI
    /// convention), serialized through a single-flight gate because the wrapped session is single-tenant.
    ///
    /// <code>
    /// using var overfit = OverfitClient.LoadGguf("qwen.q4km.gguf");
    /// IChatClient chat = overfit.AsChatClient();
    /// Console.WriteLine(await chat.GetResponseAsync("What is the capital of France?"));
    /// </code>
    /// </summary>
    public sealed class OverfitChatClient : IChatClient
    {
        private readonly ChatSession _session;
        private readonly ChatClientMetadata _metadata;
        private readonly SemaphoreSlim _gate = new(1, 1);
        private bool _disposed;

        /// <param name="session">The chat session (borrowed — NOT disposed by this adapter).</param>
        /// <param name="modelId">Optional model id surfaced through <see cref="ChatClientMetadata"/>.</param>
        public OverfitChatClient(ChatSession session, string? modelId = null)
        {
            _session = session ?? throw new ArgumentNullException(nameof(session));
            _metadata = new ChatClientMetadata("overfit", providerUri: null, defaultModelId: modelId ?? "overfit");
        }

        /// <inheritdoc />
        public async Task<ChatResponse> GetResponseAsync(
            IEnumerable<ChatMessage> messages,
            ChatOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(messages);
            ObjectDisposedException.ThrowIf(_disposed, this);

            var (sampling, maxTokens) = BuildOptions(options);
            var genOptions = new GenerationOptions(maxTokens, maxContextLength: 8192, sampling, stopOnEndOfTextToken: true);
            var materialized = Materialize(messages);

            await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
            
            try
            {
                var reply = await Task.Run(() =>
                {
                    var userContent = Replay(materialized);
                    return _session.Send(userContent, in genOptions, onText: null, constraint: null);
                }, cancellationToken).ConfigureAwait(false);

                var stats = _session.LastStats;
                
                return new ChatResponse(new ChatMessage(ChatRole.Assistant, reply))
                {
                    ModelId = _metadata.DefaultModelId,
                    FinishReason = stats.GeneratedTokens >= maxTokens ? ChatFinishReason.Length : ChatFinishReason.Stop,
                    Usage = new UsageDetails
                    {
                        InputTokenCount = stats.PromptTokens,
                        OutputTokenCount = stats.GeneratedTokens,
                        TotalTokenCount = stats.PromptTokens + stats.GeneratedTokens,
                    },
                };
            }
            finally
            {
                _gate.Release();
            }
        }

        /// <inheritdoc />
        public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
            IEnumerable<ChatMessage> messages,
            ChatOptions? options = null,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(messages);
            ObjectDisposedException.ThrowIf(_disposed, this);

            var (sampling, maxTokens) = BuildOptions(options);
            var genOptions = new GenerationOptions(maxTokens, maxContextLength: 8192, sampling, stopOnEndOfTextToken: true);
            var materialized = Materialize(messages);

            await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                // Bridge the synchronous per-token onText callback to the async stream through a channel:
                // generation runs on a worker, deltas flow out as ChatResponseUpdate as they arrive.
                var channel = Channel.CreateUnbounded<string>(new UnboundedChannelOptions { SingleReader = true, SingleWriter = true });
                var generation = Task.Run(() =>
                {
                    try
                    {
                        var userContent = Replay(materialized);
                        _session.Send(userContent, in genOptions,
                            onText: delta => channel.Writer.TryWrite(delta), constraint: null);
                    }
                    finally
                    {
                        channel.Writer.Complete();
                    }
                }, cancellationToken);

                var modelId = _metadata.DefaultModelId;
                yield return new ChatResponseUpdate(ChatRole.Assistant, (string?)null) { ModelId = modelId };

                await foreach (var delta in channel.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
                {
                    yield return new ChatResponseUpdate(ChatRole.Assistant, delta) { ModelId = modelId };
                }

                await generation.ConfigureAwait(false);

                var stats = _session.LastStats;
                yield return new ChatResponseUpdate(ChatRole.Assistant, (string?)null)
                {
                    ModelId = modelId,
                    FinishReason = stats.GeneratedTokens >= maxTokens ? ChatFinishReason.Length : ChatFinishReason.Stop,
                };
            }
            finally
            {
                _gate.Release();
            }
        }

        /// <inheritdoc />
        public object? GetService(Type serviceType, object? serviceKey = null)
        {
            ArgumentNullException.ThrowIfNull(serviceType);
            if (serviceKey is null && serviceType == typeof(ChatClientMetadata))
            {
                return _metadata;
            }
            return serviceType.IsInstanceOfType(this) ? this : null;
        }

        /// <summary>Disposes adapter-owned state only — the wrapped <see cref="OverfitClient"/> is borrowed.</summary>
        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
            _gate.Dispose();
        }

        // Maps M.E.AI ChatOptions to Overfit sampling. temperature 0 → greedy; >0 → top-p sampling.
        private static (SamplingOptions Sampling, int MaxTokens) BuildOptions(ChatOptions? options)
        {
            var maxTokens = options?.MaxOutputTokens ?? 512;
            if (maxTokens <= 0) { maxTokens = 512; }

            var temperature = options?.Temperature ?? 1.0f;
            var sampling = temperature <= 0.0001f
                ? SamplingOptions.Greedy
                : new SamplingOptions(SamplingStrategy.TopP, temperature, topK: 0, topP: options?.TopP ?? 1.0f, seed: 0);

            return (sampling, maxTokens);
        }

        private static IReadOnlyList<ChatMessage> Materialize(IEnumerable<ChatMessage> messages)
        {
            return messages as IReadOnlyList<ChatMessage> ?? messages.ToList();
        }

        // Replays all but the final message as conversation history, returns the final message's text to
        // generate against. Mirrors the OpenAI-endpoint replay (stateless per request).
        private string Replay(IReadOnlyList<ChatMessage> messages)
        {
            if (messages.Count == 0)
            {
                throw new ArgumentException("messages must contain at least one message.", nameof(messages));
            }

            _session.ResetConversation();
            for (var i = 0; i < messages.Count - 1; i++)
            {
                var text = messages[i].Text ?? string.Empty;
                var role = messages[i].Role;
                if (role == ChatRole.System) { _session.AddSystem(text); }
                else if (role == ChatRole.Assistant) { _session.AddAssistant(text); }
                else { _session.AddUser(text); }   // user / tool / unknown → user turn
            }
            
            return messages[^1].Text ?? string.Empty;
        }
    }
}
