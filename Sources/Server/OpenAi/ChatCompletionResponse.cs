using System.Text.Json.Serialization;

namespace DevOnBike.Overfit.Server.OpenAi
{
    public sealed class ChatCompletionResponse
    {
        [JsonPropertyName("id")] public string Id { get; set; } = "";
        [JsonPropertyName("object")] public string Object { get; set; } = "chat.completion";
        [JsonPropertyName("created")] public long Created { get; set; }
        [JsonPropertyName("model")] public string Model { get; set; } = "";
        [JsonPropertyName("choices")] public List<ChatChoice> Choices { get; set; } = [];
        [JsonPropertyName("usage")] public OpenAiUsage Usage { get; set; } = new();
    }
}