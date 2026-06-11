// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json.Serialization;

namespace DevOnBike.Overfit.Mcp.Protocol
{
    /// <summary>The <c>initialize</c> result: negotiated spec revision, advertised capabilities
    /// (tools only — this server has no resources/prompts), and the server identity.</summary>
    public sealed class McpInitializeResult
    {
        [JsonPropertyName("protocolVersion")]
        public string ProtocolVersion { get; set; } = string.Empty;

        [JsonPropertyName("capabilities")]
        public CapabilitiesBody Capabilities { get; set; } = new();

        [JsonPropertyName("serverInfo")]
        public ServerInfoBody ServerInfo { get; set; } = new();

        public sealed class CapabilitiesBody
        {
            [JsonPropertyName("tools")]
            public ToolsCapabilityBody Tools { get; set; } = new();
        }

        /// <summary>Intentionally empty — <c>"tools": {}</c> advertises the tools surface.</summary>
        public sealed class ToolsCapabilityBody
        {
        }

        public sealed class ServerInfoBody
        {
            [JsonPropertyName("name")]
            public string Name { get; set; } = string.Empty;

            [JsonPropertyName("version")]
            public string Version { get; set; } = string.Empty;
        }
    }
}
