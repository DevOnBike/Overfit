// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;

namespace DevOnBike.Overfit.Mcp
{
    /// <summary>
    /// One tool exposed by the <see cref="McpServer"/>: a name, a human/model-facing description,
    /// the tool's input schema as a RAW JSON Schema object string (written verbatim into
    /// <c>tools/list</c> — no schema object model, no reflection), and the handler. The handler
    /// receives the <c>arguments</c> object from <c>tools/call</c> (or <c>null</c> when absent)
    /// and returns a <see cref="McpToolResult"/>; exceptions it throws are converted into
    /// <c>isError</c> results by the server, never into protocol errors.
    /// </summary>
    public sealed class McpTool
    {
        public string Name { get; }

        public string Description { get; }

        /// <summary>A JSON Schema object as raw JSON, e.g. <c>{"type":"object","properties":{...},"required":[...]}</c>.</summary>
        public string InputSchemaJson { get; }

        /// <summary>The schema parsed once at construction (malformed schemas fail fast, not at
        /// the first <c>tools/list</c>) — written into the tool descriptor verbatim.</summary>
        public JsonElement InputSchema { get; }

        public Func<JsonElement?, McpToolResult> Handler { get; }

        public McpTool(string name, string description, string inputSchemaJson, Func<JsonElement?, McpToolResult> handler)
        {
            ArgumentException.ThrowIfNullOrEmpty(name);
            ArgumentNullException.ThrowIfNull(description);
            ArgumentException.ThrowIfNullOrEmpty(inputSchemaJson);
            ArgumentNullException.ThrowIfNull(handler);

            Name = name;
            Description = description;
            InputSchemaJson = inputSchemaJson;
            Handler = handler;

            using var schema = JsonDocument.Parse(inputSchemaJson);
            InputSchema = schema.RootElement.Clone();
        }
    }
}
