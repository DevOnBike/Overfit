// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json.Serialization;

namespace DevOnBike.Overfit.Server.OpenAi
{
    public sealed class EmbeddingsResponse
    {
        [JsonPropertyName("object")] public string Object { get; set; } = "list";
        [JsonPropertyName("data")] public List<EmbeddingData> Data { get; set; } = [];
        [JsonPropertyName("model")] public string Model { get; set; } = "";
        [JsonPropertyName("usage")] public OpenAiUsage Usage { get; set; } = new();
    }
}