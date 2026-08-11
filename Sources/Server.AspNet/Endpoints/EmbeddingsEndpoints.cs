// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using DevOnBike.Overfit.Server.AspNet.Services;
using DevOnBike.Overfit.Server.OpenAi;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;

namespace DevOnBike.Overfit.Server.AspNet.Endpoints
{
    /// <summary>
    /// <c>POST /v1/embeddings</c> — in-process sentence embeddings (nothing leaves the box). Thin: reads the
    /// body and delegates to the injected <see cref="IOpenAiInferenceService"/>, which returns 501 through the
    /// sink when no embedding model was loaded.
    /// </summary>
    internal static class EmbeddingsEndpoints
    {
        public static RouteGroupBuilder MapEmbeddings(this RouteGroupBuilder v1)
        {
            v1.MapPost("/embeddings", static async (HttpContext ctx, IOpenAiInferenceService service) =>
            {
                EmbeddingsRequest? req;
                try
                {
                    req = await JsonSerializer.DeserializeAsync(ctx.Request.Body, OpenAiJsonContext.Default.EmbeddingsRequest, ctx.RequestAborted);
                }
                catch (JsonException ex)
                {
                    await EndpointHelpers.WriteErrorAsync(ctx.Response, StatusCodes.Status400BadRequest, $"invalid JSON body: {ex.Message}", ctx.RequestAborted);
                    return;
                }

                EndpointHelpers.EnableSynchronousIO(ctx);
                service.Embed(req, new AspNetResponseSink(ctx.Response), ctx.RequestAborted);
            });

            return v1;
        }
    }
}
