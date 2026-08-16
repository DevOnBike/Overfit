// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Server.AspNet.Services;
using DevOnBike.Overfit.Server.OpenAi;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;

namespace DevOnBike.Overfit.Server.AspNet.Endpoints
{
    /// <summary>
    /// The one place the OpenAI-compatible surface is wired up — DI registration and endpoint mapping — so the
    /// production host (<see cref="OverfitAspNetServer"/>) and integration tests share exactly the same routes,
    /// JSON configuration and route-group structure. Tests register a fake
    /// <see cref="IOpenAiInferenceService"/> and drive the endpoints in-memory, with no model loaded.
    /// </summary>
    public static class OverfitOpenAiApi
    {
        /// <summary>
        /// Registers the OpenAI DTO source-gen JSON resolver and the inference service that backs every
        /// endpoint. Call before <see cref="WebApplicationBuilder.Build"/>.
        /// </summary>
        public static IServiceCollection AddOverfitOpenAi(
            this IServiceCollection services, IOpenAiInferenceService service, ServerMetrics metrics)
        {
            services.ConfigureHttpJsonOptions(options =>
                options.SerializerOptions.TypeInfoResolverChain.Insert(0, OpenAiJsonContext.Default));
            services.AddSingleton(service);
            services.AddSingleton(metrics);
            return services;
        }

        /// <summary>
        /// Maps the full surface: <c>/health</c>, the docs pages, and the versioned <c>/v1</c> route group with
        /// one endpoint class per resource (models, chat, embeddings, speech).
        /// </summary>
        public static WebApplication MapOverfitOpenAiApi(this WebApplication app)
        {
            // Registered before the endpoints so it wraps them: the endpoint executor is terminal middleware,
            // so everything added here surrounds it. Resolved once from the application container rather than
            // per request — the metrics object is a singleton and a per-request lookup would buy nothing.
            var metrics = app.Services.GetRequiredService<ServerMetrics>();

            app.Use(async (context, next) =>
            {
                try
                {
                    await next(context);
                    metrics.RecordResponse(context.Response.StatusCode);
                }
                catch
                {
                    // An unhandled exception reaches the client as a 500 whether or not the status has been
                    // written yet. Letting it go uncounted would make the error rate look best exactly when
                    // the server is at its worst.
                    metrics.RecordResponse(StatusCodes.Status500InternalServerError);
                    throw;
                }
            });

            app.MapGet("/health", () => Results.Text("ok", "text/plain"));
            app.MapGet("/", () => Results.Text("ok", "text/plain"));
            app.MapMetrics();
            app.MapDocs();

            var v1 = app.MapGroup("/v1");
            v1.MapModels();
            v1.MapChat();
            v1.MapEmbeddings();
            v1.MapSpeech();

            return app;
        }
    }
}
