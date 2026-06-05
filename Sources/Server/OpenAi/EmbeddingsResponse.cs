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