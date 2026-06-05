using System.Text.Json.Serialization;

namespace DevOnBike.Overfit.Server.OpenAi
{
    /// <summary>Minimal OpenAI-style error envelope (<c>{"error":{"message":...}}</c>).</summary>
    public sealed class OpenAiErrorResponse
    {
        [JsonPropertyName("error")] public OpenAiError Error { get; set; } = new();
    }
}