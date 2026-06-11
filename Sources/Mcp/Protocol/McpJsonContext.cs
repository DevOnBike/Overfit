// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json.Serialization;

namespace DevOnBike.Overfit.Mcp.Protocol
{
    /// <summary>
    /// Source-generated serialization metadata for the MCP wire contracts (same pattern as the
    /// OpenAI server's <c>OpenAiJsonContext</c>): compile-time (de)serializers — no reflection,
    /// Native-AOT-clean. Every closed construction of <see cref="JsonRpcResponse{TResult}"/> the
    /// server emits must be registered here explicitly.
    /// </summary>
    [JsonSourceGenerationOptions(DefaultIgnoreCondition = JsonIgnoreCondition.Never)]
    [JsonSerializable(typeof(JsonRpcRequest))]
    [JsonSerializable(typeof(JsonRpcResponse<McpInitializeResult>))]
    [JsonSerializable(typeof(JsonRpcResponse<McpToolsListResult>))]
    [JsonSerializable(typeof(JsonRpcResponse<McpCallToolResult>))]
    [JsonSerializable(typeof(JsonRpcResponse<McpEmptyResult>))]
    [JsonSerializable(typeof(JsonRpcErrorResponse))]
    public sealed partial class McpJsonContext : JsonSerializerContext
    {
    }
}
