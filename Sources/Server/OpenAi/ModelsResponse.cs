using System.Text.Json.Serialization;

namespace DevOnBike.Overfit.Server.OpenAi
{
    public sealed class ModelsResponse
    {
        [JsonPropertyName("object")] public string Object { get; set; } = "list";
        [JsonPropertyName("data")] public List<ModelInfo> Data { get; set; } = [];
    }
}