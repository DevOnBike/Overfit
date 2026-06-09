// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json.Serialization;

namespace DevOnBike.Overfit.Server.OpenAi
{
    /// <summary>Minimal OpenAI-style error envelope (<c>{"error":{"message":...}}</c>).</summary>
    public sealed class OpenAiErrorResponse
    {
        [JsonPropertyName("error")] public OpenAiError Error { get; set; } = new();
    }
}