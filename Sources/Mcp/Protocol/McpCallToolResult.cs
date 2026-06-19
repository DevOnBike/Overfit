// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json.Serialization;

namespace DevOnBike.Overfit.Mcp.Protocol
{
    /// <summary>The <c>tools/call</c> result: text content items + the MCP <c>isError</c> flag
    /// (tool-execution failures are reported here, not as JSON-RPC errors, so the host model can
    /// read them and self-correct).</summary>
    public sealed class McpCallToolResult
    {
        [JsonPropertyName("content")]
        public TextContentBody[] Content { get; set; } = [];

        [JsonPropertyName("isError")]
        public bool IsError
        {
            get; set;
        }

        public sealed class TextContentBody
        {
            [JsonPropertyName("type")]
            public string Type { get; set; } = "text";

            [JsonPropertyName("text")]
            public string Text { get; set; } = string.Empty;
        }
    }
}
