// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json.Serialization;

namespace DevOnBike.Overfit.Demo.LocalAgent.OpenAi
{
    // OpenAI-compatible request/response shapes (snake_case via JsonPropertyName) so existing OpenAI
    // clients / SDKs / tooling (LangChain, Semantic Kernel, openai-python, UIs) point at Overfit by just
    // changing the base URL. Only the fields the in-process runtime honours are modelled; unknown fields
    // are ignored on bind.

    public sealed class ChatCompletionRequest
    {
        [JsonPropertyName("model")] public string? Model { get; set; }
        [JsonPropertyName("messages")] public List<OpenAiMessage> Messages { get; set; } = [];
        [JsonPropertyName("stream")] public bool Stream { get; set; }
        [JsonPropertyName("temperature")] public float? Temperature { get; set; }
        [JsonPropertyName("top_p")] public float? TopP { get; set; }
        [JsonPropertyName("max_tokens")] public int? MaxTokens { get; set; }

        /// <summary>Newer OpenAI field name for the generation cap; honoured if <c>max_tokens</c> is absent.</summary>
        [JsonPropertyName("max_completion_tokens")] public int? MaxCompletionTokens { get; set; }
    }

    public sealed class OpenAiMessage
    {
        // Nullable so streaming deltas can omit role/content (with WhenWritingNull); requests always carry role.
        [JsonPropertyName("role")] public string? Role { get; set; }
        [JsonPropertyName("content")] public string? Content { get; set; }
    }

    public sealed class ChatCompletionResponse
    {
        [JsonPropertyName("id")] public string Id { get; set; } = "";
        [JsonPropertyName("object")] public string Object { get; set; } = "chat.completion";
        [JsonPropertyName("created")] public long Created { get; set; }
        [JsonPropertyName("model")] public string Model { get; set; } = "";
        [JsonPropertyName("choices")] public List<ChatChoice> Choices { get; set; } = [];
        [JsonPropertyName("usage")] public OpenAiUsage Usage { get; set; } = new();
    }

    public sealed class ChatChoice
    {
        [JsonPropertyName("index")] public int Index { get; set; }
        [JsonPropertyName("message")] public OpenAiMessage? Message { get; set; }
        [JsonPropertyName("delta")] public OpenAiMessage? Delta { get; set; }
        [JsonPropertyName("finish_reason")] public string? FinishReason { get; set; }
    }

    public sealed class ChatCompletionChunk
    {
        [JsonPropertyName("id")] public string Id { get; set; } = "";
        [JsonPropertyName("object")] public string Object { get; set; } = "chat.completion.chunk";
        [JsonPropertyName("created")] public long Created { get; set; }
        [JsonPropertyName("model")] public string Model { get; set; } = "";
        [JsonPropertyName("choices")] public List<ChatChoice> Choices { get; set; } = [];
    }

    public sealed class OpenAiUsage
    {
        [JsonPropertyName("prompt_tokens")] public int PromptTokens { get; set; }

        // Nullable: chat completions set it; the embeddings usage object (per spec) has only
        // prompt_tokens + total_tokens, so it is left null there and omitted (WhenWritingNull).
        [JsonPropertyName("completion_tokens")] public int? CompletionTokens { get; set; }
        [JsonPropertyName("total_tokens")] public int TotalTokens { get; set; }
    }

    public sealed class EmbeddingsRequest
    {
        [JsonPropertyName("model")] public string? Model { get; set; }

        /// <summary>A single string or an array of strings (OpenAI allows both). Bound as a flexible node.</summary>
        [JsonPropertyName("input")] public System.Text.Json.JsonElement Input { get; set; }
    }

    public sealed class EmbeddingsResponse
    {
        [JsonPropertyName("object")] public string Object { get; set; } = "list";
        [JsonPropertyName("data")] public List<EmbeddingData> Data { get; set; } = [];
        [JsonPropertyName("model")] public string Model { get; set; } = "";
        [JsonPropertyName("usage")] public OpenAiUsage Usage { get; set; } = new();
    }

    public sealed class EmbeddingData
    {
        [JsonPropertyName("object")] public string Object { get; set; } = "embedding";
        [JsonPropertyName("index")] public int Index { get; set; }
        [JsonPropertyName("embedding")] public float[] Embedding { get; set; } = [];
    }

    public sealed class ModelsResponse
    {
        [JsonPropertyName("object")] public string Object { get; set; } = "list";
        [JsonPropertyName("data")] public List<ModelInfo> Data { get; set; } = [];
    }

    public sealed class ModelInfo
    {
        [JsonPropertyName("id")] public string Id { get; set; } = "";
        [JsonPropertyName("object")] public string Object { get; set; } = "model";
        [JsonPropertyName("created")] public long Created { get; set; }
        [JsonPropertyName("owned_by")] public string OwnedBy { get; set; } = "overfit";
    }
}
