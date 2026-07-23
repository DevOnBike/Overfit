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
            SentenceEmbedder? embedder = null,
            OrpheusVoiceEngine? tts = null,
            Action<string>? onListening = null,
            CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(pool);

            using var service = new OverfitInferenceService(pool, modelName, systemMessage, embedder, tts);

            var builder = WebApplication.CreateSlimBuilder();

            // The CLI owns the console (it prints the banner via onListening); keep Kestrel's own startup
            // logging off the wire so `overfit serve` output stays clean.
            builder.Logging.ClearProviders();

            // Bind and serialize every OpenAI DTO through the source-gen context — the reflection-free path
            // Native AOT requires.
            builder.Services.ConfigureHttpJsonOptions(options =>
                options.SerializerOptions.TypeInfoResolverChain.Insert(0, OpenAiJsonContext.Default));

            builder.Services.AddSingleton<IOpenAiInferenceService>(service);

            var app = builder.Build();

            app.MapGet("/health", () => Results.Text("ok", "text/plain"));
            app.MapGet("/", () => Results.Text("ok", "text/plain"));

            // The OpenAI surface as a versioned route group, one endpoint class per resource.
            var v1 = app.MapGroup("/v1");
            v1.MapModels();
            v1.MapChat();
            v1.MapEmbeddings();
            v1.MapSpeech();

            app.Lifetime.ApplicationStarted.Register(() => onListening?.Invoke($"http://{host}:{port}"));

            // Discarding the shutdown Task is intentional (StopAsync is fire-and-forget on cancel);
            // `_ =` keeps CS4014 — promoted to error repo-wide — from tripping.
            using var reg = cancellationToken.Register(() => _ = app.StopAsync());

            app.Run($"http://{host}:{port}");
        }
    }
}
