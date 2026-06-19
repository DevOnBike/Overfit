// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json.Serialization;

namespace DevOnBike.Overfit.Server.OpenAi
{
    public sealed class ChatCompletionChunk
    {
        [JsonPropertyName("id")] public string Id { get; set; } = "";
        [JsonPropertyName("object")] public string Object { get; set; } = "chat.completion.chunk";
        [JsonPropertyName("created")]
        public long Created
        {
            get; set;
        }
        [JsonPropertyName("model")] public string Model { get; set; } = "";
        [JsonPropertyName("choices")] public List<ChatChoice> Choices { get; set; } = [];
    }
}