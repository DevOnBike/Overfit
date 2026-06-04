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
    /// When <paramref name="polish"/> is set (the Bielik preset, via <c>ExamplesLanguage: "pl"</c>) the
    /// examples are in Polish, so the pre-filled requests match the loaded model's language.
    /// </summary>
    internal sealed class RequestExamplesFilter(bool polish = false) : IOperationFilter
    {
        public void Apply(OpenApiOperation operation, OperationFilterContext context)
        {
            var route = context.ApiDescription.RelativePath?.Trim('/');
            var example = polish ? PolishExample(route) : EnglishExample(route);

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

        private static JsonNode? EnglishExample(string? route) => route switch
        {
            "chat" => new JsonObject { ["message"] = "What is the capital of France? Answer in one sentence." },
            // chat/json now takes an optional JSON-Schema: the reply is constrained to CONFORM to it.
            "chat/json" => JsonNode.Parse("""
            {
              "message": "Classify the sentiment of this review as JSON: 'I absolutely love this product!'",
              "schema": "{\"type\":\"object\",\"properties\":{\"sentiment\":{\"type\":\"string\",\"enum\":[\"positive\",\"negative\",\"neutral\"]}},\"required\":[\"sentiment\"],\"additionalProperties\":false}"
            }
            """),
            "agent" => new JsonObject { ["message"] = "Open a high priority ticket for sam@brightlabs.example about a failed SSO login." },
            "rag/query" => new JsonObject { ["question"] = "Can an EU customer get a refund after 10 days?", ["topK"] = 3 },
            "decision/refund" => new JsonObject { ["message"] = "An EU customer bought the product 10 days ago, unused, and wants to withdraw from the contract." },
            // OpenAI-compatible surface — point any OpenAI client here, or Try-it-out in Swagger.
            "v1/chat/completions" => JsonNode.Parse("""
            {
              "model": "overfit",
              "messages": [{ "role": "user", "content": "Classify the sentiment of: 'I love this!'" }],
              "max_tokens": 32,
              "temperature": 0,
              "response_format": {
                "type": "json_schema",
                "json_schema": {
                  "name": "sentiment",
                  "schema": {
                    "type": "object",
                    "properties": { "sentiment": { "type": "string", "enum": ["positive", "negative", "neutral"] } },
                    "required": ["sentiment"],
                    "additionalProperties": false
                  }
                }
              }
            }
            """),
            "v1/embeddings" => JsonNode.Parse("""{ "input": ["hello world", "a second sentence"] }"""),
            _ => null,
        };

        private static JsonNode? PolishExample(string? route) => route switch
        {
            "chat" => new JsonObject { ["message"] = "Wyjaśnij krótko, czym jest rękojmia." },
            "chat/json" => JsonNode.Parse("""
            {
              "message": "Sklasyfikuj sentyment tej opinii jako JSON: 'Uwielbiam ten produkt!'",
              "schema": "{\"type\":\"object\",\"properties\":{\"sentyment\":{\"type\":\"string\",\"enum\":[\"pozytywny\",\"negatywny\",\"neutralny\"]}},\"required\":[\"sentyment\"],\"additionalProperties\":false}"
            }
            """),
            "agent" => new JsonObject { ["message"] = "Załóż zgłoszenie o wysokim priorytecie dla anna@firma.pl w sprawie nieudanego logowania SSO." },
            "rag/query" => new JsonObject { ["question"] = "Ile dni ma klient z UE na odstąpienie od umowy?", ["topK"] = 4 },
            "decision/refund" => new JsonObject { ["message"] = "Klient z UE kupił produkt 10 dni temu, nie był używany, chce odstąpić od umowy." },
            "v1/chat/completions" => JsonNode.Parse("""
            {
              "model": "bielik",
              "messages": [{ "role": "user", "content": "Sklasyfikuj sentyment: 'Uwielbiam to!'" }],
              "max_tokens": 32,
              "temperature": 0,
              "response_format": {
                "type": "json_schema",
                "json_schema": {
                  "name": "sentyment",
                  "schema": {
                    "type": "object",
                    "properties": { "sentyment": { "type": "string", "enum": ["pozytywny", "negatywny", "neutralny"] } },
                    "required": ["sentyment"],
                    "additionalProperties": false
                  }
                }
              }
            }
            """),
            "v1/embeddings" => JsonNode.Parse("""{ "input": ["pierwsze zdanie", "drugie zdanie"] }"""),
            _ => null,
        };
    }
}
