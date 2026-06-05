// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json.Serialization;

namespace DevOnBike.Overfit.Server.OpenAi
{
    /// <summary>
    /// Source-generated <see cref="JsonSerializerContext"/> for the OpenAI DTOs. Reflection-free (de)serialization
    /// keeps the <c>overfit serve</c> path Native-AOT- and trim-clean — no <c>JsonSerializer</c> reflection
    /// fallback, so no IL2026 / IL3050 warnings reach the AOT publish. <c>WhenWritingNull</c> mirrors the OpenAI
    /// wire format (streaming deltas omit role/content; embeddings usage omits completion_tokens).
    /// </summary>
    [JsonSourceGenerationOptions(DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
    [JsonSerializable(typeof(ChatCompletionRequest))]
    [JsonSerializable(typeof(ChatCompletionResponse))]
    [JsonSerializable(typeof(ChatCompletionChunk))]
    [JsonSerializable(typeof(EmbeddingsRequest))]
    [JsonSerializable(typeof(EmbeddingsResponse))]
    [JsonSerializable(typeof(ModelsResponse))]
    [JsonSerializable(typeof(OpenAiErrorResponse))]
    public sealed partial class OpenAiJsonContext : JsonSerializerContext
    {
    }
}
