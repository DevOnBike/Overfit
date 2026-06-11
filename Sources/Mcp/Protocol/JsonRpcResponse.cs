// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using System.Text.Json.Serialization;

namespace DevOnBike.Overfit.Mcp.Protocol
{
    /// <summary>
    /// An outgoing JSON-RPC 2.0 success response. Closed constructions of this generic are
    /// registered one by one in <see cref="McpJsonContext"/> (source-gen needs concrete types —
    /// no runtime type discovery). <see cref="Id"/> must always be present on the wire, so it is
    /// a non-nullable <see cref="JsonElement"/> (use <see cref="McpJson.NullElement"/> when the
    /// request had no usable id).
    /// </summary>
    public sealed class JsonRpcResponse<TResult>
    {
        [JsonPropertyName("jsonrpc")]
        public string JsonRpc { get; set; } = "2.0";

        [JsonPropertyName("id")]
        public JsonElement Id { get; set; }

        [JsonPropertyName("result")]
        public TResult? Result { get; set; }
    }
}
