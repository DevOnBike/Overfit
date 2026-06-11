// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using DevOnBike.Overfit.Mcp;

namespace DevOnBike.Overfit.Tests.Mcp
{
    /// <summary>
    /// Protocol-level tests for <see cref="McpServer"/> — drive the JSON-RPC/stdio layer through
    /// in-memory readers/writers (no model, no process): handshake + version negotiation,
    /// tools/list shape, tools/call dispatch (success / handler throw / unknown tool), JSON-RPC
    /// error codes, and the notifications-get-no-response rule.
    /// </summary>
    public sealed class McpServerProtocolTests
    {
        private static McpTool EchoTool(string name = "echo")
        {
            return new McpTool(
                name,
                "Echoes the 'text' argument back.",
                """{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}""",
                args => McpToolResult.Success("echo: " + args!.Value.GetProperty("text").GetString()));
        }

        private static McpTool ThrowingTool()
        {
            return new McpTool(
                "boom",
                "Always throws.",
                """{"type":"object"}""",
                _ => throw new InvalidOperationException("kaboom"));
        }

        private static JsonDocument Send(McpServer server, string requestLine)
        {
            using var output = new StringWriter();
            server.HandleMessage(requestLine, output);
            var line = output.ToString().TrimEnd('\r', '\n');
            Assert.False(string.IsNullOrEmpty(line), "expected exactly one response line");
            return JsonDocument.Parse(line);
        }

        private static McpServer NewServer(params McpTool[] tools)
        {
            return new McpServer("overfit-test", "1.2.3", tools);
        }

        [Fact]
        public void Initialize_NegotiatesKnownVersion_AndReportsServerInfo()
        {
            var server = NewServer(EchoTool());

            using var response = Send(server,
                """{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}""");

            var result = response.RootElement.GetProperty("result");
            Assert.Equal("2025-03-26", result.GetProperty("protocolVersion").GetString());
            Assert.Equal("overfit-test", result.GetProperty("serverInfo").GetProperty("name").GetString());
            Assert.Equal("1.2.3", result.GetProperty("serverInfo").GetProperty("version").GetString());
            Assert.True(result.GetProperty("capabilities").TryGetProperty("tools", out _));
            Assert.Equal(1, response.RootElement.GetProperty("id").GetInt32());
        }

        [Fact]
        public void Initialize_UnknownRequestedVersion_FallsBackToLatestSupported()
        {
            var server = NewServer(EchoTool());

            using var response = Send(server,
                """{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"1999-01-01"}}""");

            Assert.Equal("2025-06-18",
                response.RootElement.GetProperty("result").GetProperty("protocolVersion").GetString());
        }

        [Fact]
        public void ToolsList_ReturnsNameDescriptionAndVerbatimSchema()
        {
            var server = NewServer(EchoTool());

            using var response = Send(server, """{"jsonrpc":"2.0","id":7,"method":"tools/list"}""");

            var tools = response.RootElement.GetProperty("result").GetProperty("tools");
            Assert.Equal(1, tools.GetArrayLength());
            var tool = tools[0];
            Assert.Equal("echo", tool.GetProperty("name").GetString());
            Assert.Equal("Echoes the 'text' argument back.", tool.GetProperty("description").GetString());
            var schema = tool.GetProperty("inputSchema");
            Assert.Equal("object", schema.GetProperty("type").GetString());
            Assert.Equal("string", schema.GetProperty("properties").GetProperty("text").GetProperty("type").GetString());
        }

        [Fact]
        public void ToolsCall_DispatchesToHandler_AndWrapsTextContent()
        {
            var server = NewServer(EchoTool());

            using var response = Send(server,
                """{"jsonrpc":"2.0","id":"abc","method":"tools/call","params":{"name":"echo","arguments":{"text":"hi"}}}""");

            var result = response.RootElement.GetProperty("result");
            Assert.False(result.GetProperty("isError").GetBoolean());
            var content = result.GetProperty("content");
            Assert.Equal(1, content.GetArrayLength());
            Assert.Equal("text", content[0].GetProperty("type").GetString());
            Assert.Equal("echo: hi", content[0].GetProperty("text").GetString());
            Assert.Equal("abc", response.RootElement.GetProperty("id").GetString());
        }

        [Fact]
        public void ToolsCall_HandlerThrows_BecomesIsErrorResult_NotProtocolError()
        {
            var server = NewServer(ThrowingTool());

            using var response = Send(server,
                """{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"boom","arguments":{}}}""");

            Assert.False(response.RootElement.TryGetProperty("error", out _));
            var result = response.RootElement.GetProperty("result");
            Assert.True(result.GetProperty("isError").GetBoolean());
            Assert.Contains("kaboom", result.GetProperty("content")[0].GetProperty("text").GetString());
        }

        [Fact]
        public void ToolsCall_UnknownTool_IsInvalidParamsError()
        {
            var server = NewServer(EchoTool());

            using var response = Send(server,
                """{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"nope","arguments":{}}}""");

            Assert.Equal(-32602, response.RootElement.GetProperty("error").GetProperty("code").GetInt32());
        }

        [Fact]
        public void UnknownMethod_IsMethodNotFound()
        {
            var server = NewServer(EchoTool());

            using var response = Send(server, """{"jsonrpc":"2.0","id":4,"method":"resources/list"}""");

            Assert.Equal(-32601, response.RootElement.GetProperty("error").GetProperty("code").GetInt32());
        }

        [Fact]
        public void MalformedJson_IsParseError_WithNullId()
        {
            var server = NewServer(EchoTool());

            using var response = Send(server, "{not json");

            Assert.Equal(-32700, response.RootElement.GetProperty("error").GetProperty("code").GetInt32());
            Assert.Equal(JsonValueKind.Null, response.RootElement.GetProperty("id").ValueKind);
        }

        [Fact]
        public void Notifications_GetNoResponse()
        {
            var server = NewServer(EchoTool());
            using var output = new StringWriter();

            server.HandleMessage("""{"jsonrpc":"2.0","method":"notifications/initialized"}""", output);
            server.HandleMessage("""{"jsonrpc":"2.0","method":"notifications/cancelled","params":{}}""", output);

            Assert.Equal(string.Empty, output.ToString());
        }

        [Fact]
        public void Ping_ReturnsEmptyObject()
        {
            var server = NewServer(EchoTool());

            using var response = Send(server, """{"jsonrpc":"2.0","id":9,"method":"ping"}""");

            var result = response.RootElement.GetProperty("result");
            Assert.Equal(JsonValueKind.Object, result.ValueKind);
            Assert.False(result.EnumerateObject().MoveNext());
        }

        [Fact]
        public void Run_ServesFullHandshakeOverStreams_AndStopsOnEof()
        {
            var server = NewServer(EchoTool());
            var input = new StringReader(
                """{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}""" + "\n" +
                """{"jsonrpc":"2.0","method":"notifications/initialized"}""" + "\n" +
                """{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"echo","arguments":{"text":"x"}}}""" + "\n");
            using var output = new StringWriter();

            server.Run(input, output);   // returns at EOF

            var lines = output.ToString().Split('\n', StringSplitOptions.RemoveEmptyEntries);
            Assert.Equal(2, lines.Length);   // initialize + tools/call; the notification is silent

            using var first = JsonDocument.Parse(lines[0]);
            Assert.Equal("2025-06-18", first.RootElement.GetProperty("result").GetProperty("protocolVersion").GetString());

            using var second = JsonDocument.Parse(lines[1]);
            Assert.Equal("echo: x",
                second.RootElement.GetProperty("result").GetProperty("content")[0].GetProperty("text").GetString());
        }

        [Fact]
        public void DuplicateToolNames_AreRejectedAtConstruction()
        {
            Assert.Throws<ArgumentException>(() => NewServer(EchoTool(), EchoTool()));
        }
    }
}
