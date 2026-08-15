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
    /// <c>POST /v1/chat/completions</c> (streaming SSE + non-streaming). Thin: reads the body and hands it to
    /// the injected <see cref="IOpenAiInferenceService"/>; all session pooling and generation live there.
    /// </summary>
    internal static class ChatEndpoints
    {
        public static RouteGroupBuilder MapChat(this RouteGroupBuilder v1)
        {
            v1.MapPost("/chat/completions", static async (HttpContext ctx, IOpenAiInferenceService service) =>
            {
                ChatCompletionRequest? req;

                try
                {
                    req = await JsonSerializer.DeserializeAsync(ctx.Request.Body, OpenAiJsonContext.Default.ChatCompletionRequest, ctx.RequestAborted);
                }
                catch (JsonException ex)
                {
                    await EndpointHelpers.WriteErrorAsync(ctx.Response, StatusCodes.Status400BadRequest, $"invalid JSON body: {ex.Message}", ctx.RequestAborted);
                    return;
                }

                EndpointHelpers.EnableSynchronousIO(ctx);
                service.CompleteChat(req, new AspNetResponseSink(ctx.Response), ctx.RequestAborted);
            });

            return v1;
        }
    }
}
