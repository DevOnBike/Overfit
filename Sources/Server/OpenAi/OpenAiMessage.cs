using System.Text.Json.Serialization;

namespace DevOnBike.Overfit.Server.OpenAi
{
    public sealed class OpenAiMessage
    {
        // Nullable so streaming deltas can omit role/content (with WhenWritingNull); requests always carry role.
        [JsonPropertyName("role")] public string? Role { get; set; }
        [JsonPropertyName("content")] public string? Content { get; set; }
    }
}