// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;

namespace DevOnBike.Overfit.LanguageModels.Tools
{
    /// <summary>
    /// A parsed tool call: the chosen tool <see cref="Name"/> and its <see cref="Arguments"/> as a
    /// raw JSON string. Produced from the constrained-generation envelope
    /// <c>{"name": "...", "arguments": {...}}</c>. Because the envelope was enforced at decode time,
    /// <see cref="TryParse"/> on a constrained reply always succeeds.
    /// </summary>
    public readonly struct ToolCall
    {
        public ToolCall(string name, string arguments)
        {
            Name = name;
            Arguments = arguments;
        }

        public string Name
        {
            get;
        }

        /// <summary>The tool arguments as a raw JSON string (an object; <c>{}</c> when absent).</summary>
        public string Arguments
        {
            get;
        }

        /// <summary>
        /// Parses a <c>{"name": ..., "arguments": ...}</c> envelope. Returns false (rather than
        /// throwing) on malformed input or a missing/!string <c>name</c>.
        /// </summary>
        public static bool TryParse(string json, out ToolCall call)
        {
            call = default;
            if (string.IsNullOrWhiteSpace(json))
            {
                return false;
            }

            try
            {
                using var doc = JsonDocument.Parse(json);
                var root = doc.RootElement;
                if (root.ValueKind != JsonValueKind.Object)
                {
                    return false;
                }
                if (!root.TryGetProperty("name", out var nameElement) ||
                    nameElement.ValueKind != JsonValueKind.String)
                {
                    return false;
                }

                var arguments = root.TryGetProperty("arguments", out var argsElement)
                    ? argsElement.GetRawText()
                    : "{}";

                call = new ToolCall(nameElement.GetString()!, arguments);
                return true;
            }
            catch (JsonException)
            {
                return false;
            }
        }
    }
}
