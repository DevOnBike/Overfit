using System.Text.Json.Serialization;

namespace DevOnBike.Overfit.Server.OpenAi
{
    /// <summary>The OpenAI <c>POST /v1/audio/speech</c> request body (text-to-speech).</summary>
    public sealed class SpeechRequest
    {
        [JsonPropertyName("model")] public string? Model { get; set; }

        /// <summary>The text to synthesize.</summary>
        [JsonPropertyName("input")] public string? Input { get; set; }

        /// <summary>A preset voice id (Orpheus: tara, leah, jess, leo, dan, mia, zac, zoe).</summary>
        [JsonPropertyName("voice")] public string? Voice { get; set; }

        /// <summary>Output container: <c>wav</c> (default) or <c>pcm</c> (raw 16-bit LE @ 24 kHz).</summary>
        [JsonPropertyName("response_format")] public string? ResponseFormat { get; set; }

        [JsonPropertyName("speed")] public float? Speed { get; set; }
    }
}
