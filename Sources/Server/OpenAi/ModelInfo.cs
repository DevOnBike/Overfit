// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json.Serialization;

namespace DevOnBike.Overfit.Server.OpenAi
{
    public sealed class ModelInfo
    {
        [JsonPropertyName("id")] public string Id { get; set; } = "";
        [JsonPropertyName("object")] public string Object { get; set; } = "model";
        [JsonPropertyName("created")]
        public long Created
        {
            get; set;
        }
        [JsonPropertyName("owned_by")] public string OwnedBy { get; set; } = "overfit";
    }
}