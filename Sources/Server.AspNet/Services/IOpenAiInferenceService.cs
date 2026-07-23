// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Server.OpenAi;

namespace DevOnBike.Overfit.Server.AspNet.Services
{
    /// <summary>
    /// The inference operations behind the OpenAI-compatible endpoints, resolved from DI so the Minimal-API
    /// endpoints stay thin (parse the body, hand it here) and the logic — session pooling, concurrency gating,
    /// the shared protocol exchanges, availability of embeddings/TTS — lives in one testable place.
    ///
    /// <para>Transport-neutral by design: every method writes through an <see cref="IOpenAiResponseSink"/> and
    /// never touches an ASP.NET type, so the same service is unit-testable against a fake sink and could back
    /// a non-Kestrel host unchanged. Methods are synchronous — they drive the synchronous, zero-allocation
    /// decode path — and take the request's <see cref="CancellationToken"/> for the rent/gate waits.</para>
    /// </summary>
    public interface IOpenAiInferenceService
    {
        /// <summary>The served model's id, for <c>GET /v1/models</c>.</summary>
        ModelsResponse ListModels();

        /// <summary>
        /// Runs one chat completion (streaming or not) — rents a session, replays history, generates, restores
        /// the baseline system turn — writing the whole response through <paramref name="sink"/>. Sheds with
        /// HTTP 503 through the sink when the pool is exhausted.
        /// </summary>
        void CompleteChat(ChatCompletionRequest? request, IOpenAiResponseSink sink, CancellationToken cancellationToken);

        /// <summary>
        /// Embeds the request's input in-process. Writes HTTP 501 through <paramref name="sink"/> when no
        /// embedding model was loaded.
        /// </summary>
        void Embed(EmbeddingsRequest? request, IOpenAiResponseSink sink, CancellationToken cancellationToken);

        /// <summary>
        /// Synthesizes speech (WAV / PCM). Writes HTTP 501 through <paramref name="sink"/> when no TTS model
        /// was loaded.
        /// </summary>
        void Synthesize(SpeechRequest? request, IOpenAiResponseSink sink, CancellationToken cancellationToken);
    }
}
