// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using DevOnBike.Overfit.Audio.Tts.Orpheus;
using DevOnBike.Overfit.Diagnostics;
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Embeddings;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Server.OpenAi;
using DevOnBike.Overfit.Serving;

namespace DevOnBike.Overfit.Server.AspNet.Services
{
    /// <summary>
    /// The default <see cref="IOpenAiInferenceService"/>: holds the session pool plus the optional embedder /
    /// TTS engine, serializes access to the single-instance embedder and TTS engine, and drives the shared
    /// <see cref="ChatCompletionExchange"/> / <see cref="EmbeddingsExchange"/> / <see cref="SpeechExchange"/>.
    /// The pool, embedder and TTS engine are owned by the caller (the CLI disposes them); this service owns
    /// only the two gates.
    /// </summary>
    public sealed class OverfitInferenceService : IOpenAiInferenceService, IDisposable
    {
        private static readonly TimeSpan RentTimeout = TimeSpan.FromSeconds(30);

        private static readonly bool Trace =
            Environment.GetEnvironmentVariable(OverfitEnvironment.ServerTrace) == "1";

        private readonly OverfitResourcePool<OverfitClient> _pool;
        private readonly string _modelName;
        private readonly string _systemMessage;
        private readonly long _created;
        private readonly SentenceEmbedder? _embedder;
        private readonly OrpheusVoiceEngine? _tts;

        // A SentenceEmbedder has one scratch arena; the TTS engine is single-instance. Serialize each.
        private readonly SemaphoreSlim _embedGate = new(1, 1);
        private readonly SemaphoreSlim _ttsGate = new(1, 1);

        private readonly ServerMetrics _metrics;
        private readonly IChatExchangeObserver _chatObserver;

        public OverfitInferenceService(
            OverfitResourcePool<OverfitClient> pool,
            string modelName,
            string systemMessage,
            SentenceEmbedder? embedder,
            OrpheusVoiceEngine? tts,
            ServerMetrics metrics,
            IClock? clock = null)
        {
            _pool = pool ?? throw new ArgumentNullException(nameof(pool));
            _modelName = modelName;
            _systemMessage = systemMessage;
            _embedder = embedder;
            _tts = tts;
            _metrics = metrics ?? throw new ArgumentNullException(nameof(metrics));
            _created = (clock ?? SystemClock.Instance).UtcNow.ToUnixTimeSeconds();

            // The chat exchange takes ONE observer; metrics always record, the phase trace joins only when
            // OVERFIT_SERVER_TRACE=1.
            _chatObserver = Trace
                ? new CompositeChatObserver(_metrics, ConsoleTraceObserver.Instance)
                : _metrics;

            // Publish the live pool gauges on the Meter (this service owns the pool).
            _metrics.BindPool(() => PoolStatus);
        }

        public ModelsResponse ListModels()
            => new()
            {
                Data = [new ModelInfo { Id = _modelName, Created = _created }]
            };

        public PoolStatus PoolStatus
        {
            get
            {
                var m = _pool.Metrics;
                return new PoolStatus(m.Size, m.Active, m.Available, m.TotalRejected, m.PeakActive);
            }
        }

        /// <summary>
        /// Runs one chat completion through <see cref="ChatCompletionExchange"/> on a session rented from the
        /// pool for the duration.
        ///
        /// <para><b>The rent blocks the calling thread for up to thirty seconds — <c>RentTimeout</c>, passed
        /// to <c>OverfitResourcePool.TryRent</c> — and that is a decision (XC-26, 2026-08-12), not an
        /// oversight.</b> Unlike the two gates below there is nothing to await: <c>TryRent</c> waits on
        /// <c>SemaphoreSlim.Wait(TimeSpan, CancellationToken)</c>, which blocks a thread, and the pool has no
        /// asynchronous rental. The token IS honoured, and the two outcomes differ: a cancelled wait throws
        /// <see cref="OperationCanceledException"/> (caught below, answered 503) and is deliberately NOT
        /// counted a rejection, while a timeout returns false, IS counted, and sheds with 503 into
        /// <c>overfit_pool_rejected_total</c>. The CLI's <c>--sessions</c> defaults to 1, so at any
        /// concurrency above one every request but one waits here, each holding a thread.</para>
        ///
        /// <para><b>Why it is not awaited:</b> an asynchronous rent would hand the thread back for the QUEUE
        /// only. Once through the gate the thread is held for the generation as well — <c>Handle</c> is
        /// synchronous and <see cref="AspNetResponseSink"/> writes each token from inside the model's decode
        /// callback, which that file records as the server's design — and the number of concurrent
        /// completions is bounded by the pool size either way. What it would cost is permanent: <c>out
        /// lease</c> cannot cross an <c>await</c>, so <c>DevOnBike.Overfit</c> would gain its first
        /// asynchronous primitive in public API, and <see cref="IOpenAiInferenceService.CompleteChat"/> would
        /// have to become task-returning.</para>
        ///
        /// <para><b>What is NOT known, and the condition that would reopen it:</b> whether those parked
        /// threads delay unrelated endpoints has never been measured on this server, and neither has any
        /// throughput effect of awaiting the rent. Both are unverified claims rather than findings, and the
        /// verdict on either belongs to a measurement. Reopen if a burst at <c>--sessions 1</c> measurably
        /// delays <c>GET /v1/models</c> or <c>GET /metrics</c> while <c>dotnet_threadpool_queue_length</c>
        /// rises.</para>
        /// </summary>
        public void CompleteChat(ChatCompletionRequest? request, IOpenAiResponseSink sink, CancellationToken cancellationToken)
        {
            OverfitResourcePool<OverfitClient>.Lease lease;
            try
            {
                if (!_pool.TryRent(RentTimeout, cancellationToken, out lease))
                {
                    WriteError(sink, 503, $"server busy — all {_pool.Size} sessions in use; retry shortly.");
                    return;
                }
            }
            catch (OperationCanceledException)
            {
                WriteError(sink, 503, "server is shutting down.");
                return;
            }

            var started = ValueStopwatch.StartNew();
            using (lease)
            {
                ChatCompletionExchange.Handle(request, lease.Value, _modelName, _systemMessage, sink, _chatObserver);
            }
            _metrics.RecordResponseTime(started.GetElapsedTime().TotalSeconds);
        }

        /// <summary>
        /// Embeds the request's input, one caller at a time.
        ///
        /// <para><b>The gate is awaited, not blocked on (OVERFIT040).</b> It used to be
        /// <c>_embedGate.Wait(cancellationToken)</c>, and under N concurrent embedding requests that parked
        /// N-1 Kestrel request threads on a semaphore doing nothing at all — the gate serializes by design,
        /// so queueing is the normal case rather than the rare one. <c>WaitAsync</c> gives those threads
        /// back for the whole time a request is only waiting its turn.</para>
        ///
        /// <para><b>What this does NOT change:</b> the embedding itself still runs to completion on one
        /// thread after the gate opens, because <c>SentenceEmbedder</c> has a single scratch arena and the
        /// path is synchronous and zero-allocation by design. The saving is the queue, not the work.</para>
        /// </summary>
        public async Task EmbedAsync(EmbeddingsRequest? request, IOpenAiResponseSink sink, CancellationToken cancellationToken)
        {
            if (_embedder == null)
            {
                WriteError(sink, 501, "embeddings are not served — start with an embedding model "
                    + "(e.g. 'overfit serve <model> --embed-model <dir>').");
                return;
            }

            await _embedGate.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                EmbeddingsExchange.Handle(request, _embedder, _modelName, sink);
                _metrics.RecordEmbeddingRequest();
            }
            finally
            {
                _embedGate.Release();
            }
        }

        /// <summary>
        /// Synthesizes speech, one caller at a time. The TTS gate is awaited rather than blocked on for the
        /// same reason as <see cref="EmbedAsync"/> — and more sharply here, because a synthesis is long
        /// enough that a second caller queues for most of it.
        /// </summary>
        public async Task SynthesizeAsync(SpeechRequest? request, IOpenAiResponseSink sink, CancellationToken cancellationToken)
        {
            if (_tts == null)
            {
                WriteError(sink, 501, "text-to-speech is not served — start with a TTS model "
                    + "(e.g. 'overfit serve <model> --tts-model <orpheus.gguf> --tts-snac <dir>').");
                return;
            }

            await _ttsGate.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                SpeechExchange.Handle(request, _tts, sink);
                _metrics.RecordSpeechRequest();
            }
            finally
            {
                _ttsGate.Release();
            }
        }

        public void Dispose()
        {
            _embedGate.Dispose();
            _ttsGate.Dispose();
        }

        private static void WriteError(IOpenAiResponseSink sink, int status, string message)
        {
            var body = new OpenAiErrorResponse { Error = new OpenAiError { Message = message } };
            sink.WriteBody(status, "application/json",
                JsonSerializer.Serialize(body, OpenAiJsonContext.Default.OpenAiErrorResponse));
        }
    }
}
