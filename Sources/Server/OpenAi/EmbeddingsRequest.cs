// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using System.Text.Json.Serialization;

namespace DevOnBike.Overfit.Server.OpenAi
{
    public sealed class EmbeddingsRequest
    {
        [JsonPropertyName("model")] public string? Model { get; set; }

        /// <summary>A single string or an array of strings (OpenAI allows both). Bound as a flexible node.</summary>
        [JsonPropertyName("input")] public JsonElement Input { get; set; }
    }
}