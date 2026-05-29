// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json.Nodes;
using Microsoft.OpenApi;
using Swashbuckle.AspNetCore.SwaggerGen;

namespace DevOnBike.Overfit.Demo.LocalAgent.Swagger
{
    /// <summary>
    /// Pre-fills each endpoint's request body in Swagger UI with a ready-to-run example, so a visitor
    /// can open an endpoint, hit "Try it out" → "Execute" and get a meaningful result without typing
    /// anything. Examples are keyed by route, so the two endpoints that share a request shape
    /// (<c>/chat</c> and <c>/chat/json</c>) still each show a payload tailored to what they do.
    /// </summary>
    internal sealed class RequestExamplesFilter : IOperationFilter
    {
        public void Apply(OpenApiOperation operation, OperationFilterContext context)
        {
            var route = context.ApiDescription.RelativePath?.Trim('/');
            JsonNode? example = route switch
            {
                "chat" => new JsonObject
                {
                    ["message"] = "What is the capital of France? Answer in one sentence.",
                },
                "chat/json" => new JsonObject
                {
                    ["message"] = "Return a JSON object with fields name and city for Ada, who lives in Berlin.",
                },
                "agent" => new JsonObject
                {
                    ["message"] = "Open a high priority ticket for sam@brightlabs.example about a failed SSO login.",
                },
                "rag/query" => new JsonObject
                {
                    ["question"] = "Can an EU customer get a refund after 10 days?",
                    ["topK"] = 3,
                },
                _ => null,
            };

            if (example is null || operation.RequestBody?.Content is null)
            {
                return;
            }

            // Set the example on every media type the body accepts (application/json here). Clone per
            // assignment — a JsonNode can only be attached to one parent.
            foreach (var media in operation.RequestBody.Content.Values)
            {
                media.Example = example.DeepClone();
            }
        }
    }
}
