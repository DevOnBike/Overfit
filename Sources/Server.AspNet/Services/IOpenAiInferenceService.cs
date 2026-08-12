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
    /// a non-Kestrel host unchanged. The WORK is synchronous throughout — it drives the synchronous,
    /// zero-allocation decode path, which writes each token from inside the model's own callback — and every
    /// method takes the request's <see cref="CancellationToken"/>.</para>
    ///
    /// <para><b>Two of the three are awaitable, and the split is not an inconsistency.</b> Each method starts
    /// by waiting for exclusive access to a single-instance resource, and what it waits ON differs:
    /// <see cref="CompleteChat"/> rents from an <c>OverfitResourcePool</c>, whose <c>TryRent</c> has no
    /// asynchronous form, while the embedder and TTS engine are gated by a <c>SemaphoreSlim</c>, which does.
    /// So the two that can hand the request thread back while they queue do (OVERFIT040), and the one that
    /// cannot says so here rather than looking like an oversight. The waiting is the only part that is
    /// awaitable in any of them; once through the gate the work runs to completion on one thread.
    /// <b>What the synchronous one costs</b> — a request thread blocked for up to thirty seconds, and with
    /// the CLI's default of one session that is every concurrent request but one — is written above
    /// <c>OverfitInferenceService.CompleteChat</c>, together with the reasoning for keeping it (XC-26,
    /// 2026-08-12: a decision, not a default).</para>
    /// </summary>
    public interface IOpenAiInferenceService
    {
        /// <summary>The served model's id, for <c>GET /v1/models</c>.</summary>
        ModelsResponse ListModels();

        /// <summary>Current session-pool snapshot for the <c>/metrics</c> gauges (size / active / free / rejected).</summary>
        PoolStatus PoolStatus
        {
            get;
        }

        /// <summary>
        /// Runs one chat completion (streaming or not) — rents a session, replays history, generates, restores
        /// the baseline system turn — writing the whole response through <paramref name="sink"/>. Sheds with
        /// HTTP 503 through the sink when the pool is exhausted.
        /// </summary>
        void CompleteChat(ChatCompletionRequest? request, IOpenAiResponseSink sink, CancellationToken cancellationToken);

        /// <summary>
        /// Embeds the request's input in-process. Writes HTTP 501 through <paramref name="sink"/> when no
        /// embedding model was loaded. Awaits the embedder gate rather than blocking on it, so a request that
        /// is only queueing does not hold a request thread; the embedding itself then runs synchronously.
        /// </summary>
        Task EmbedAsync(EmbeddingsRequest? request, IOpenAiResponseSink sink, CancellationToken cancellationToken);

        /// <summary>
        /// Synthesizes speech (WAV / PCM). Writes HTTP 501 through <paramref name="sink"/> when no TTS model
        /// was loaded. Awaits the TTS gate rather than blocking on it, for the same reason as
        /// <see cref="EmbedAsync"/>.
        /// </summary>
        Task SynthesizeAsync(SpeechRequest? request, IOpenAiResponseSink sink, CancellationToken cancellationToken);
    }
}
