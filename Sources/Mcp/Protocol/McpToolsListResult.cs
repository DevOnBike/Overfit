// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json.Serialization;

namespace DevOnBike.Overfit.Mcp.Protocol
{
    /// <summary>The <c>tools/list</c> result.</summary>
    public sealed class McpToolsListResult
    {
        [JsonPropertyName("tools")]
        public McpToolDescriptor[] Tools { get; set; } = [];
    }
}
