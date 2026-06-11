// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using System.Text.Json.Serialization;

namespace DevOnBike.Overfit.Server.OpenAi
{
    // OpenAI-compatible request/response shapes (snake_case via JsonPropertyName) so existing OpenAI
    // clients / SDKs / tooling (LangChain, Semantic Kernel, openai-python, UIs) point at Overfit by just
    // changing the base URL. Only the fields the in-process runtime honours are modelled; unknown fields
    // are ignored on bind. Shared by the AOT `overfit serve` self-host and the ASP.NET demo.

    public sealed class ChatCompletionRequest
    {
        [JsonPropertyName("model")] public string? Model { get; set; }
        [JsonPropertyName("messages")] public List<OpenAiMessage> Messages { get; set; } = [];
        [JsonPropertyName("stream")] public bool Stream { get; set; }
        [JsonPropertyName("temperature")] public float? Temperature { get; set; }
        [JsonPropertyName("top_p")] public float? TopP { get; set; }

        /// <summary>Min-P sampling cutoff — a llama.cpp-server extension (not in the OpenAI spec) that common
        /// local-AI clients send: keep tokens with probability ≥ <c>min_p</c> × P(top). 0 / absent = off.</summary>
        [JsonPropertyName("min_p")] public float? MinP { get; set; }

        [JsonPropertyName("max_tokens")] public int? MaxTokens { get; set; }

        /// <summary>Newer OpenAI field name for the generation cap; honoured if <c>max_tokens</c> is absent.</summary>
        [JsonPropertyName("max_completion_tokens")] public int? MaxCompletionTokens { get; set; }

        /// <summary><c>{"type":"json_object"}</c> → guaranteed well-formed JSON;
        /// <c>{"type":"json_schema","json_schema":{"schema":{...}}}</c> → output constrained to conform to the
        /// schema; absent / <c>{"type":"text"}</c> → unconstrained.</summary>
        [JsonPropertyName("response_format")] public JsonElement? ResponseFormat { get; set; }
    }

}
