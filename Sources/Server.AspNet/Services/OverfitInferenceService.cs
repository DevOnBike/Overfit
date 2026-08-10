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

        public void Embed(EmbeddingsRequest? request, IOpenAiResponseSink sink, CancellationToken cancellationToken)
        {
            if (_embedder is null)
            {
                WriteError(sink, 501, "embeddings are not served — start with an embedding model "
                    + "(e.g. 'overfit serve <model> --embed-model <dir>').");
                return;
            }

            _embedGate.Wait(cancellationToken);
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

        public void Synthesize(SpeechRequest? request, IOpenAiResponseSink sink, CancellationToken cancellationToken)
        {
            if (_tts is null)
            {
                WriteError(sink, 501, "text-to-speech is not served — start with a TTS model "
                    + "(e.g. 'overfit serve <model> --tts-model <orpheus.gguf> --tts-snac <dir>').");
                return;
            }

            _ttsGate.Wait(cancellationToken);
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
