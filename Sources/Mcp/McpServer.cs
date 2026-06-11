// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using DevOnBike.Overfit.Mcp.Protocol;

namespace DevOnBike.Overfit.Mcp
{
    /// <summary>
    /// A minimal MCP (Model Context Protocol) server over the stdio transport: JSON-RPC 2.0,
    /// one message per line on stdin/stdout (logs belong on stderr). The wire contracts are
    /// typed DTOs (de)serialized through the source-generated <see cref="McpJsonContext"/> —
    /// compile-time serializers, no reflection, no SDK dependency — so it stays
    /// Native-AOT-clean like the rest of the Overfit serving stack (same pattern as the OpenAI
    /// server's <c>OpenAiJsonContext</c>).
    ///
    /// Implements the tools surface of the protocol: <c>initialize</c> (version negotiation +
    /// capabilities), <c>notifications/initialized</c>, <c>ping</c>, <c>tools/list</c> and
    /// <c>tools/call</c>. Requests are served strictly one at a time on the caller's thread
    /// (single-tenant model session underneath — same stance as <c>OverfitOpenAiServer</c>).
    /// </summary>
    public sealed class McpServer
    {
        /// <summary>Spec revisions this server accepts; the first entry is what we answer with
        /// when the client requests a revision we don't know.</summary>
        private static readonly string[] SupportedProtocolVersions =
        [
            "2025-06-18",
            "2025-03-26",
            "2024-11-05",
        ];

        private readonly string _serverName;
        private readonly string _serverVersion;
        private readonly McpTool[] _tools;
        private readonly Dictionary<string, McpTool> _toolsByName;
        private readonly McpToolsListResult _toolsListResult;
        private readonly TextWriter? _log;

        public McpServer(string serverName, string serverVersion, IReadOnlyList<McpTool> tools, TextWriter? log = null)
        {
            ArgumentException.ThrowIfNullOrEmpty(serverName);
            ArgumentException.ThrowIfNullOrEmpty(serverVersion);
            ArgumentNullException.ThrowIfNull(tools);

            _serverName = serverName;
            _serverVersion = serverVersion;
            _log = log;
            _tools = new McpTool[tools.Count];
            _toolsByName = new Dictionary<string, McpTool>(tools.Count, StringComparer.Ordinal);
            var descriptors = new McpToolDescriptor[tools.Count];

            for (var i = 0; i < tools.Count; i++)
            {
                var tool = tools[i];
                _tools[i] = tool;

                if (!_toolsByName.TryAdd(tool.Name, tool))
                {
                    throw new ArgumentException($"Duplicate MCP tool name '{tool.Name}'.", nameof(tools));
                }

                descriptors[i] = new McpToolDescriptor
                {
                    Name = tool.Name,
                    Description = tool.Description,
                    InputSchema = tool.InputSchema,
                };
            }

            _toolsListResult = new McpToolsListResult { Tools = descriptors };
        }

        /// <summary>
        /// Serves newline-delimited JSON-RPC until <paramref name="input"/> ends (host closed our
        /// stdin — the standard MCP shutdown signal) or the token is cancelled. Blocking.
        /// </summary>
        public void Run(TextReader input, TextWriter output, CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(input);
            ArgumentNullException.ThrowIfNull(output);

            while (!cancellationToken.IsCancellationRequested)
            {
                var line = input.ReadLine();

                if (line == null)
                {
                    return;
                }

                if (string.IsNullOrWhiteSpace(line))
                {
                    continue;
                }

                HandleMessage(line, output);
            }
        }

        /// <summary>Handles one JSON-RPC message line; writes at most one response line. Exposed
        /// for in-memory testing — <see cref="Run"/> is just this in a read loop.</summary>
        public void HandleMessage(string line, TextWriter output)
        {
            JsonRpcRequest? request;

            try
            {
                request = JsonSerializer.Deserialize(line, McpJsonContext.Default.JsonRpcRequest);
            }
            catch (JsonException)
            {
                // Distinguish malformed JSON (-32700) from well-formed JSON that isn't a request
                // object (-32600) — the DTO deserializer throws the same exception for both.
                var code = -32700;
                var message = "Parse error";

                try
                {
                    JsonDocument.Parse(line).Dispose();
                    code = -32600;
                    message = "Invalid Request";
                }
                catch (JsonException)
                {
                }

                WriteError(output, id: null, code, message);
                return;
            }

            var id = request?.Id;
            var isNotification = id == null;

            if (request?.Method is not { Length: > 0 } method)
            {
                if (!isNotification)
                {
                    WriteError(output, id, code: -32600, message: "Invalid Request");
                }

                return;
            }

            try
            {
                DispatchMethod(method, id, isNotification, request.Params, output);
            }
            catch (Exception exception)
            {
                _log?.WriteLine($"[overfit-mcp] unhandled error in '{method}': {exception}");

                if (!isNotification)
                {
                    WriteError(output, id, code: -32603, message: "Internal error");
                }
            }
        }

        private void DispatchMethod(string method, JsonElement? id, bool isNotification, JsonElement? parameters, TextWriter output)
        {
            switch (method)
            {
                case "initialize":
                    WriteResult(output, id, BuildInitializeResult(parameters), McpJsonContext.Default.JsonRpcResponseMcpInitializeResult);
                    return;

                case "ping":
                    WriteResult(output, id, new McpEmptyResult(), McpJsonContext.Default.JsonRpcResponseMcpEmptyResult);
                    return;

                case "tools/list":
                    WriteResult(output, id, _toolsListResult, McpJsonContext.Default.JsonRpcResponseMcpToolsListResult);
                    return;

                case "tools/call":
                    HandleToolsCall(output, id, parameters);
                    return;

                default:
                    // notifications/initialized, notifications/cancelled, … — nothing to do, and
                    // notifications must never get a response.
                    if (!isNotification)
                    {
                        WriteError(output, id, code: -32601, message: $"Method not found: {method}");
                    }

                    return;
            }
        }

        private McpInitializeResult BuildInitializeResult(JsonElement? parameters)
        {
            var protocolVersion = SupportedProtocolVersions[0];

            if (parameters is { ValueKind: JsonValueKind.Object } p &&
                p.TryGetProperty("protocolVersion", out var requestedElement) &&
                requestedElement.ValueKind == JsonValueKind.String)
            {
                var requested = requestedElement.GetString()!;

                foreach (var supported in SupportedProtocolVersions)
                {
                    if (string.Equals(supported, requested, StringComparison.Ordinal))
                    {
                        protocolVersion = requested;
                        break;
                    }
                }
            }

            return new McpInitializeResult
            {
                ProtocolVersion = protocolVersion,
                ServerInfo = new McpInitializeResult.ServerInfoBody
                {
                    Name = _serverName,
                    Version = _serverVersion,
                },
            };
        }

        private void HandleToolsCall(TextWriter output, JsonElement? id, JsonElement? parameters)
        {
            if (parameters is not { ValueKind: JsonValueKind.Object } p ||
                !p.TryGetProperty("name", out var nameElement) ||
                nameElement.ValueKind != JsonValueKind.String)
            {
                WriteError(output, id, code: -32602, message: "Invalid params: missing tool name");
                return;
            }

            var name = nameElement.GetString()!;

            if (!_toolsByName.TryGetValue(name, out var tool))
            {
                WriteError(output, id, code: -32602, message: $"Unknown tool: {name}");
                return;
            }

            JsonElement? arguments = p.TryGetProperty("arguments", out var argumentsElement) ? argumentsElement : null;
            McpToolResult result;

            try
            {
                result = tool.Handler(arguments);
            }
            catch (Exception exception)
            {
                // Per the MCP spec, tool-execution failures flow back INSIDE the result (isError)
                // so the host model can read them and self-correct — not as protocol errors.
                _log?.WriteLine($"[overfit-mcp] tool '{name}' threw: {exception}");
                result = McpToolResult.Error($"{exception.GetType().Name}: {exception.Message}");
            }

            var callResult = new McpCallToolResult
            {
                Content =
                [
                    new McpCallToolResult.TextContentBody { Text = result.Text },
                ],
                IsError = result.IsError,
            };

            WriteResult(output, id, callResult, McpJsonContext.Default.JsonRpcResponseMcpCallToolResult);
        }

        private static void WriteResult<TResult>(
            TextWriter output,
            JsonElement? id,
            TResult result,
            System.Text.Json.Serialization.Metadata.JsonTypeInfo<JsonRpcResponse<TResult>> typeInfo)
        {
            var response = new JsonRpcResponse<TResult>
            {
                Id = id ?? McpJson.NullElement,
                Result = result,
            };

            output.WriteLine(JsonSerializer.Serialize(response, typeInfo));
            output.Flush();
        }

        private static void WriteError(TextWriter output, JsonElement? id, int code, string message)
        {
            var response = new JsonRpcErrorResponse
            {
                Id = id ?? McpJson.NullElement,
                Error = new JsonRpcErrorResponse.ErrorBody
                {
                    Code = code,
                    Message = message,
                },
            };

            output.WriteLine(JsonSerializer.Serialize(response, McpJsonContext.Default.JsonRpcErrorResponse));
            output.Flush();
        }
    }
}
