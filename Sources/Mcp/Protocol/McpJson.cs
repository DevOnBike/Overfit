// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;

namespace DevOnBike.Overfit.Mcp.Protocol
{
    /// <summary>Shared JSON helpers for the MCP wire layer.</summary>
    public static class McpJson
    {
        /// <summary>A reusable <c>null</c> <see cref="JsonElement"/> — responses must always carry
        /// an <c>id</c>, and <c>default(JsonElement)</c> (Undefined) is not serializable.</summary>
        public static readonly JsonElement NullElement = JsonDocument.Parse("null").RootElement.Clone();
    }
}
