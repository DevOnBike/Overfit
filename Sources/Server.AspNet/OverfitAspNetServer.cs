// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Audio.Tts.Orpheus;
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Embeddings;
using DevOnBike.Overfit.Server.AspNet.Endpoints;
using DevOnBike.Overfit.Server.AspNet.Services;
using DevOnBike.Overfit.Server.OpenAi;
using DevOnBike.Overfit.Serving;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace DevOnBike.Overfit.Server.AspNet
{
    /// <summary>
    /// The AOT-ready ASP.NET (Kestrel + Minimal API) host for Overfit's OpenAI-compatible server — the server
    /// that ships in the NuGet package and the Docker image. Routing goes through the Request Delegate
    /// Generator (no reflection), JSON through the source-gen <see cref="OpenAiJsonContext"/>, and the request
    /// logic through a DI-resolved <see cref="IOpenAiInferenceService"/>, so the whole surface publishes under
    /// Native AOT while staying organized like a controller app: a <c>/v1</c> route group with one endpoint
    /// class per resource, all delegating to one service.
    /// </summary>
    public static class OverfitAspNetServer
    {
        /// <summary>
        /// Binds Kestrel on <paramref name="host"/>:<paramref name="port"/> and serves until
        /// <paramref name="cancellationToken"/> is cancelled. Blocks the calling thread. Chat rents from
        /// <paramref name="pool"/> (up to <c>pool.Size</c> decode concurrently, HTTP 503 when exhausted);
        /// embeddings and TTS are served when <paramref name="embedder"/> / <paramref name="tts"/> are supplied
        /// (501 otherwise). The pool, embedder and TTS engine are owned by the caller.
        /// </summary>
        public static void Serve(
            OverfitResourcePool<OverfitClient> pool,
            string modelName,
            string host,
            int port,
            string systemMessage,
            CancellationToken cancellationToken,
            SentenceEmbedder? embedder = null,
            OrpheusVoiceEngine? tts = null,
            Action<string>? onListening = null)
        {
            ArgumentNullException.ThrowIfNull(pool);

            using var metrics = new ServerMetrics();
            using var service = new OverfitInferenceService(pool, modelName, systemMessage, embedder, tts, metrics);

            var builder = WebApplication.CreateSlimBuilder();

            // ILogger → console via the default Microsoft.Extensions.Logging (no Serilog — AOT-clean, zero
            // extra deps), at Information so request/host lifecycle logs are visible on the console.
            builder.Logging.ClearProviders();
            builder.Logging.AddSimpleConsole(options => options.SingleLine = true);
            builder.Logging.SetMinimumLevel(LogLevel.Information);

            // Source-gen JSON + the inference service + server metrics — the same wiring the integration tests use.
            builder.Services.AddOverfitOpenAi(service, metrics);

            var app = builder.Build();

            // /health, docs, and the /v1 route group (models / chat / embeddings / speech). Docs are served from
            // the AOT-clean embedded document, not a reflection-based runtime generator.
            app.MapOverfitOpenAiApi();

            app.Lifetime.ApplicationStarted.Register(() => onListening?.Invoke($"http://{host}:{port}"));

            // The shutdown task cannot be observed here, and the constraint is the callback signature rather
            // than any diagnostic: CancellationToken.Register hands us an Action, so there is nothing to
            // await from and no return value anything could look at, and the usual escape — making the
            // lambda `async void` — is banned by OVERFIT027 because an exception out of one is uncatchable
            // and kills the host process.
            //
            // WHAT IS LOST, accepted deliberately: if StopAsync faults, nobody learns. No log, no exit code,
            // no trace. The visible symptom would be `app.Run` below failing to return after cancellation —
            // a hang at shutdown with no stated cause. The two alternatives are worse for this path: holding
            // the task in a local and inspecting it after Run returns adds cross-thread machinery to a
            // shutdown path, and an OnlyOnFaulted continuation reintroduces an unobserved task inside a
            // non-async lambda, where NOTHING would flag it — trading a visible discard for an invisible one.
#pragma warning disable OVERFIT046
            using var reg = cancellationToken.Register(() => _ = app.StopAsync());
#pragma warning restore OVERFIT046

            app.Run($"http://{host}:{port}");
        }
    }
}
