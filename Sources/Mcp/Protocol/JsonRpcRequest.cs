// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using System.Text.Json.Serialization;

namespace DevOnBike.Overfit.Mcp.Protocol
{
    /// <summary>
    /// An incoming JSON-RPC 2.0 message (request or notification), deserialized via the
    /// source-generated <see cref="McpJsonContext"/>. <see cref="Id"/> stays a
    /// <see cref="JsonElement"/> because the spec allows string OR number ids and the server
    /// must echo them verbatim; <see cref="Params"/> stays raw because its shape depends on
    /// <see cref="Method"/>. A missing id ⇒ notification (an explicit <c>"id": null</c> — which
    /// the spec says SHOULD NOT be sent — is treated the same way).
    /// </summary>
    public sealed class JsonRpcRequest
    {
        [JsonPropertyName("jsonrpc")]
        public string? JsonRpc
        {
            get; set;
        }

        [JsonPropertyName("id")]
        public JsonElement? Id
        {
            get; set;
        }

        [JsonPropertyName("method")]
        public string? Method
        {
            get; set;
        }

        [JsonPropertyName("params")]
        public JsonElement? Params
        {
            get; set;
        }
    }
}
