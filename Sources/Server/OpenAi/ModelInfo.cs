using System.Text.Json.Serialization;

namespace DevOnBike.Overfit.Server.OpenAi
{
    public sealed class ModelInfo
    {
        [JsonPropertyName("id")] public string Id { get; set; } = "";
        [JsonPropertyName("object")] public string Object { get; set; } = "model";
        [JsonPropertyName("created")] public long Created { get; set; }
        [JsonPropertyName("owned_by")] public string OwnedBy { get; set; } = "overfit";
    }
}