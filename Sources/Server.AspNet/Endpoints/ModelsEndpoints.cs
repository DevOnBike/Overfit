// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Server.AspNet.Services;
using DevOnBike.Overfit.Server.OpenAi;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;

namespace DevOnBike.Overfit.Server.AspNet.Endpoints
{
    /// <summary><c>GET /v1/models</c> — the served model's id, so OpenAI clients can discover it.</summary>
    internal static class ModelsEndpoints
    {
        public static RouteGroupBuilder MapModels(this RouteGroupBuilder v1)
        {
            v1.MapGet("/models", (IOpenAiInferenceService service) =>
                Results.Json(service.ListModels(), OpenAiJsonContext.Default.ModelsResponse));

            return v1;
        }
    }
}
