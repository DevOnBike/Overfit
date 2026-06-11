// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using System.Text.Json.Serialization;

namespace DevOnBike.Overfit.Mcp.Protocol
{
    /// <summary>An outgoing JSON-RPC 2.0 error response (protocol-level failures only — tool
    /// EXECUTION failures travel inside <see cref="McpCallToolResult.IsError"/> instead).</summary>
    public sealed class JsonRpcErrorResponse
    {
        [JsonPropertyName("jsonrpc")]
        public string JsonRpc { get; set; } = "2.0";

        [JsonPropertyName("id")]
        public JsonElement Id { get; set; }

        [JsonPropertyName("error")]
        public ErrorBody Error { get; set; } = new();

        public sealed class ErrorBody
        {
            [JsonPropertyName("code")]
            public int Code { get; set; }

            [JsonPropertyName("message")]
            public string Message { get; set; } = string.Empty;
        }
    }
}
