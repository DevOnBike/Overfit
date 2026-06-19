// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using System.Text.Json.Serialization;

namespace DevOnBike.Overfit.Mcp.Protocol
{
    /// <summary>One entry of the <c>tools/list</c> result. <see cref="InputSchema"/> is a
    /// <see cref="JsonElement"/> (the schema is authored as raw JSON on <see cref="McpTool"/>,
    /// parsed once at construction) — the server has no schema object model.</summary>
    public sealed class McpToolDescriptor
    {
        [JsonPropertyName("name")]
        public string Name { get; set; } = string.Empty;

        [JsonPropertyName("description")]
        public string Description { get; set; } = string.Empty;

        [JsonPropertyName("inputSchema")]
        public JsonElement InputSchema
        {
            get; set;
        }
    }
}
