// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Mcp
{
    /// <summary>
    /// The outcome of one MCP tool invocation: a single text content item plus the MCP
    /// <c>isError</c> flag. Per the MCP spec, tool EXECUTION failures (file not found, model
    /// refused, bad input the schema could not catch) are reported inside the result with
    /// <see cref="IsError"/> = true — not as JSON-RPC protocol errors — so the host model can
    /// see the failure text and self-correct.
    /// </summary>
    public sealed class McpToolResult
    {
        public string Text { get; }

        public bool IsError { get; }

        private McpToolResult(string text, bool isError)
        {
            Text = text;
            IsError = isError;
        }

        public static McpToolResult Success(string text)
        {
            return new McpToolResult(text, isError: false);
        }

        public static McpToolResult Error(string text)
        {
            return new McpToolResult(text, isError: true);
        }
    }
}
