// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json.Serialization;

namespace DevOnBike.Overfit.Server.OpenAi
{
    public sealed class ChatChoice
    {
        [JsonPropertyName("index")]
        public int Index
        {
            get; set;
        }
        [JsonPropertyName("message")]
        public OpenAiMessage? Message
        {
            get; set;
        }
        [JsonPropertyName("delta")]
        public OpenAiMessage? Delta
        {
            get; set;
        }
        [JsonPropertyName("finish_reason")]
        public string? FinishReason
        {
            get; set;
        }
    }
}