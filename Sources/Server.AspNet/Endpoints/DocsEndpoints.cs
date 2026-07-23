// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Server.OpenAi;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;

namespace DevOnBike.Overfit.Server.AspNet.Endpoints
{
    /// <summary>
    /// <c>GET /openapi.yaml</c> (the machine-readable contract, embedded from <c>docs/openapi.yaml</c>) and
    /// <c>GET /docs</c> (the Scalar API reference pointed at it). Served from the embedded document so it works
    /// from the single AOT binary without the reflection-based runtime OpenAPI generator.
    /// </summary>
    internal static class DocsEndpoints
    {
        public static WebApplication MapDocs(this WebApplication app)
        {
            app.MapGet("/openapi.yaml", () =>
                Results.Text(OpenApiDocument.Yaml(), "application/yaml; charset=utf-8"));

            app.MapGet("/docs", () =>
                Results.Text(OpenApiDocument.ApiReferenceHtml, "text/html; charset=utf-8"));

            return app;
        }
    }
}
